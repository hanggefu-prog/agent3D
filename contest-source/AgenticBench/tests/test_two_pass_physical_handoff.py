"""Exercise evaluator OPT patches against the flow's strict physical SDC gate.

The EDA executables are mocked, not the coordinator or the Tcl inventory check.
Historical placement promotion retains its pre-public-STA source dialect.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from open3dbench.backend.opt_handoff import DieArtifacts, HBTBoundary, OptHandoff
from open3dbench.backend.sdc_handoff import PORT_INVENTORY_TCL, read_def_ports


def _patched_module(tmp_path, patch_name):
    source = ROOT / "open3dbench/backend/three_d_opt.py"
    text = source.read_text()
    if patch_name == "placement_two_pass_sta_clean_clone.patch":
        # Explicit compatibility fixture for the historical 2026-08-30 API.
        # This does not claim that the legacy patch applies to the new flow.
        text = text.replace("run_global_placement_sta", "_global_snapshot")
        text = text.replace(
            '    """Run one unified placement-estimated STA pass with passive HBT pi RC."""\n\n', ""
        )
        text = text.replace(
            ", copy_function=copy_with_log_reference", ""
        )
    target = tmp_path / "open3dbench/backend/three_d_opt.py"
    target.parent.mkdir(parents=True)
    target.write_text(text)
    subprocess.run(
        ["git", "apply", "--unsafe-paths", "--directory", str(tmp_path),
         str(ROOT / "AgenticBench/local_bench/evaluator_patches" / patch_name)],
        check=True, capture_output=True, text=True, cwd=ROOT,
    )
    assert "current_handoff = next_handoff" in target.read_text()
    spec = importlib.util.spec_from_file_location(
        "open3dbench.backend._two_pass_contract_test", target
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Run the real coordinator through final selection. Checkpoint publication
    # is covered by the flow contract tests and requires physical EDA output.
    function = next(n for n in ast.parse(target.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "optimize")
    publication = next(i for i, n in enumerate(function.body)
                       if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                       and n.target.id == "die")
    function.body = function.body[:publication] + [ast.Return(value=ast.Name(id="final_snapshot", ctx=ast.Load()))]
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(tree, str(target), "exec"), module.__dict__)
    return module


def _write_die(root, die, generation):
    root.mkdir(parents=True, exist_ok=True)
    port = "HBT_B2T_1_SRC" if die == "bottom" else "HBT_B2T_1_DST"
    direction = "OUTPUT" if die == "bottom" else "INPUT"
    # Model resizer's legal renaming of a boundary-connected net after pass 1.
    net = "original_net" if generation == 0 else "{}_net_{}".format(die, generation)
    (root / "design.def").write_text(
        "PINS 1 ;\n- {} + NET {} + DIRECTION {} + USE SIGNAL ;\nEND PINS\n".format(port, net, direction)
    )
    (root / "design.odb").write_text(json.dumps({port: {"net": net, "direction": direction, "signal_type": "SIGNAL"}}))
    (root / "design.v").write_text("// {} generation {}\n".format(die, generation))
    (root / "design.sdc").write_text("# {} generation {}\n".format(die, generation))
    return DieArtifacts(root / "design.odb", root / "design.def", root / "design.v", root / "design.sdc")


def _native_inventory_check(tmp_path, expected, live):
    """Execute the unmodified Tcl gate with minimal OpenDB object doubles."""
    if shutil.which("tclsh") is None:
        pytest.skip("tclsh is needed to execute the native inventory gate")
    inventory = tmp_path / "inventory.tsv"
    inventory.write_text("".join("{}\t{}\t{}\t{}\n".format(name, row["net"], row["direction"], row["signal_type"])
                                 for name, row in expected.items()))
    script = "namespace eval ord {}\nproc ord::get_db_block {} {return block}\n"
    script += "proc block {method args} {\n switch $method {\n findBTerm {switch -- [lindex $args 0] {\n"
    for i, name in enumerate(live):
        script += "  {" + name + "} {return port" + str(i) + "}\n"
    script += "default {return NULL}}}\n getBTerms {return {" + " ".join("port" + str(i) for i in range(len(live))) + "}}\n }}\n"
    for i, (name, row) in enumerate(live.items()):
        script += "proc port" + str(i) + " {method} {switch $method {getNet {return net" + str(i) + "} getIoType {return {" + row["direction"] + "}} getSigType {return {" + row["signal_type"] + "}}}}\n"
        script += "proc net" + str(i) + " {method} {return {" + row["net"] + "}}\n"
    script += PORT_INVENTORY_TCL
    script += "\nif {[catch {open3d_validate_sdc_ports {" + str(inventory) + "}} message]} {puts stderr $message; exit 1}\n"
    return subprocess.run(["tclsh"], input=script, capture_output=True, text=True)


@pytest.mark.parametrize("patch_name", ["tp_gnn_placement_sta.patch", "placement_two_pass_sta_clean_clone.patch"])
@pytest.mark.parametrize("fail_at", [None, "01/top", "02/global"])
def test_coordinator_advances_complete_physical_pair_only(tmp_path, monkeypatch, patch_name, fail_at):
    module = _patched_module(tmp_path / "source", patch_name)
    hbt = tmp_path / "50_hbt"
    dies = {die: _write_die(hbt / die, die, 0) for die in ("bottom", "top")}
    boundary = HBTBoundary("HBT_B2T_1", "original_net", "bottom", "top", "HBT_B2T_1_SRC", "HBT_B2T_1_DST", "B2T", False, 10, 20)
    original = OptHandoff(hbt / "checkpoint.json", dies, hbt / "assembly/design.v", hbt / "assembly/design.sdc",
                          hbt / "assembly/hbt.lib", hbt / "assembly/hbt.lef", {}, (boundary,))
    monkeypatch.setattr(module, "load_opt_handoff", lambda _: original)
    monkeypatch.setenv("OPEN3D_OPT_PASSES", "2")
    calls, snapshots, base_constraints = [], [], []

    def assembly(handoff, bottom, top, output):
        output.mkdir(parents=True, exist_ok=True)
        assert handoff.boundaries is original.boundaries
        assert handoff.checkpoint == original.checkpoint

    def sta(handoff, assembly, odbs, verilogs, platform, output, openroad):
        assert all(handoff.dies[die].odb == odbs[die] and handoff.dies[die].verilog == verilogs[die]
                   for die in ("bottom", "top"))
        if fail_at == output.parent.name + "/global":
            raise module.ThreeDOptError("injected global validation failure")
        snapshots.append(handoff)
        return {"boundary_budget_model": "passive-cutpoint", "metrics": {"wns_setup": float(len(snapshots))}, "boundaries": []}

    def constraints(die, base_sdc, snapshot, handoff, output):
        assert base_sdc == original.dies[die].sdc
        assert handoff is original
        base_constraints.append((die, base_sdc))
        output.write_text(base_sdc.read_text() + "# this pass budget\n")
        return {}

    def optimize_die(handoff, die, input_odb, sdc, platform, output, *args):
        current_pass = int(output.parent.name)
        expected_root = hbt if current_pass == 1 else output.parents[1] / "01"
        assert input_odb == expected_root / die / "design.odb"
        # Both dies see the same prior completed pair, even after bottom runs.
        for owner in ("bottom", "top"):
            artifacts = handoff.dies[owner]
            assert artifacts.odb == expected_root / owner / "design.odb"
            assert artifacts.deffile == expected_root / owner / "design.def"
            assert artifacts.verilog == expected_root / owner / "design.v"
            check = _native_inventory_check(tmp_path, read_def_ports(artifacts.deffile), json.loads(artifacts.odb.read_text()))
            assert check.returncode == 0, check.stderr
        calls.append((current_pass, die, handoff))
        if fail_at == output.parent.name + "/" + die:
            raise module.ThreeDOptError("injected die validation failure")
        _write_die(output, die, current_pass)
        return {"mock": 1.0}

    monkeypatch.setattr(module, "_rebuild_assembly", assembly)
    monkeypatch.setattr(module, "run_global_placement_sta", sta)
    monkeypatch.setattr(module, "_boundary_sdc", constraints)
    monkeypatch.setattr(module, "_die_optimize", optimize_die)
    args = module.build_parser().parse_args(["--checkpoint", str(original.checkpoint), "--platform-dir", str(tmp_path), "--output-dir", str(tmp_path / "60_opt")])
    if fail_at:
        with pytest.raises(module.ThreeDOptError, match="injected"):
            module.optimize(args)
    else:
        result = module.optimize(args)
        assert result["metrics"]["wns_setup"] == 3.0
        assert [(p, d) for p, d, _ in calls] == [(1, "bottom"), (1, "top"), (2, "bottom"), (2, "top")]
        assert calls[0][2] is original and calls[1][2] is original
        assert calls[2][2] is calls[3][2] is snapshots[1]
        assert len(base_constraints) == 4
        for pass_number in (1, 2):
            record = json.loads((tmp_path / "60_opt/iterations" / ("%02d" % pass_number) / "iteration.json").read_text())
            assert record["pass"] == pass_number
    # Neither a failed die nor a failed global check is recorded as completed.
    if fail_at:
        failed_pass = fail_at.split("/")[0]
        assert not (tmp_path / "60_opt/iterations" / failed_pass / "iteration.json").exists()
    assert read_def_ports(original.dies["bottom"].deffile)["HBT_B2T_1_SRC"]["net"] == "original_net"


@pytest.mark.parametrize("field,value", [("net", "stale_net"), ("direction", "INPUT"), ("signal_type", "CLOCK")])
def test_strict_sdc_inventory_mismatch_is_still_rejected(tmp_path, field, value):
    live = {"HBT_B2T_1_SRC": {"net": "current_net", "direction": "OUTPUT", "signal_type": "SIGNAL"}}
    expected = json.loads(json.dumps(live))
    expected["HBT_B2T_1_SRC"][field] = value
    result = _native_inventory_check(tmp_path, expected, live)
    assert result.returncode == 1
    assert "SDC physical evidence net/direction/type mismatch" in result.stderr


def test_strict_sdc_inventory_missing_port_is_still_rejected(tmp_path):
    expected = {"HBT_B2T_1_SRC": {"net": "current_net", "direction": "OUTPUT", "signal_type": "SIGNAL"}}
    result = _native_inventory_check(tmp_path, expected, {})
    assert result.returncode == 1
    assert "SDC physical evidence does not match OpenDB port" in result.stderr
