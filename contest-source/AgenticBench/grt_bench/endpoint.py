"""Trusted GRT-segment RC -> continuous passive HBT STA/power -> HotSpot.

Runs in the pinned runtime, with candidate results mounted read-only. This is
global-route-estimated RC, not OpenRCX extraction or detailed-route signoff.
"""
import argparse
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import time

from . import ENDPOINT, TIMING_ENDPOINT, THERMAL_STATISTIC
from .common import confined, finite, read, records, sha, verify_records, write
from .spef import add_name_map
from .latch_audit import LATCH_PROOF_TCL, audit as audit_latches, retain_data_checks


def validate_grt(checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    cp = read(checkpoint)
    if cp.get("stage") != "global_route" or cp.get("status") != "success" or cp.get("ready_for_extract") is not False:
        raise ValueError("Expected a successful standalone GRT checkpoint")
    def verify_native(root, entries):
        verify_records(root, {k: {"bytes": v["size_bytes"], "sha256": v["sha256"]} for k, v in entries.items()})
    verify_native(checkpoint.parent, cp["artifacts"])
    if set(cp["dies"]) != {"bottom", "top"}:
        raise ValueError("GRT must contain both dies")
    for die in ("bottom", "top"):
        folder = checkpoint.parent / die
        manifest = read(folder / "grt_manifest.json")
        required = {"global_route.odb", "global_route.odb.segments", "global_route.odb.v", "global_route.odb.def", "global_route.odb.spef", "design.sdc", "grt.log"}
        if manifest.get("status") != "success" or not required <= set(manifest["artifacts"]):
            raise ValueError("GRT analysis exports missing; regenerate GRT in a new output")
        verify_native(folder, manifest["artifacts"])
    cts = (checkpoint.parent / cp["source_checkpoint"]).resolve(strict=True)
    if sha(cts) != cp["source_checkpoint_sha256"]:
        raise ValueError("CTS upstream changed")
    return cp, cts


def _thermal(output, platform, defs, hotspot):
    from open3dbench.backend import thermal
    from scripts.workspace_cts_thermal import normalize_def, validate_power, validate_tiles
    powers, global_power, difference = validate_power(output / "assembly/instance_power.rpt",
                                                      output / "assembly/power.json", thermal._read_instance_power)
    folder = output / "thermal"
    folder.mkdir()
    sizes = thermal._lef_sizes(platform)
    geometry, grids = {}, {}
    for die in ("top", "bottom"):
        normalized = folder / (die + ".def")
        proof = normalize_def(defs[die], normalized)
        if len(thermal._read_def(normalized)[2]) != proof["components"]:
            raise ValueError("Thermal DEF parsing lost instances")
        grids[die], geometry[die] = thermal._grid_power(die, powers[die], normalized, sizes, 10)
        thermal._write_floorplan(folder / (die + ".flp"), die, geometry[die], 10)
    if any(geometry["top"][k] != geometry["bottom"][k] for k in ("diearea_dbu", "dbu_per_micron")):
        raise ValueError("Die geometry differs")
    names = ["{}_{}_{}".format(d, x, y) for d in ("top", "bottom") for x in range(10) for y in range(10)]
    values = [grids[d]["{}_{}_{}".format(d, x, y)] for d in ("top", "bottom") for x in range(10) for y in range(10)]
    total = sum(sum(v.values()) for v in powers.values())
    if abs(sum(values) - total) > max(1e-12, total * 1e-9):
        raise ValueError("Tile power is not conserved")
    (folder / "power.ptrace").write_text(" ".join(names) + "\n" + " ".join("{:.12g}".format(v) for v in values) + "\n")
    thermal._write_stack(folder / "stack.lcf")
    for name, source in (("hotspot.config", "test.config"), ("materials.txt", "test.materials")):
        shutil.copyfile(hotspot / "examples/thermal" / source, folder / name)
    command = [str(hotspot / "hotspot"), "-c", "hotspot.config", "-p", "power.ptrace",
        "-grid_layer_file", "stack.lcf", "-materials_file", "materials.txt", "-model_type", "grid",
        "-detailed_3D", "on", "-ambient", "318.15", "-init_temp", "318.15",
        "-steady_file", "temperature.steady", "-grid_steady_file", "temperature.grid.steady"]
    with (folder / "hotspot.log").open("x") as log:
        subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
    if re.search(r"\b(warning|error)\b", (folder / "hotspot.log").read_text(), re.I):
        raise ValueError("HotSpot emitted a warning or error")
    validate_tiles(folder / "temperature.steady")
    temperature = thermal._temperature_metrics(folder / "temperature.steady", 318.15)
    return {"power_w": total, "temperature_peak_k": temperature["global_maximum_k"]}, {
        "temperature": temperature, "geometry": geometry, "instance_global_power_difference_w": difference,
        "power_report": global_power, "stack_model": thermal.THERMAL_STACK_MODEL_STATUS}


def run(checkpoint, platform, hotspot, output, *, sta="/usr/local/bin/sta", endpoint=ENDPOINT):
    from open3dbench.backend.opt_handoff import load_routing_handoff
    from open3dbench.backend import post_route as post
    from open3dbench.backend.passive_hbt import passive_links, stitch_routed_spefs, write_passive_assembly
    from open3dbench.backend.three_d_opt import _platform_files, _read_key_values
    from open3dbench.backend.hbt_materialize import _repair_boundary_verilog, _validate_boundary_verilog, _validate_netlist_liberty
    if endpoint not in {ENDPOINT, TIMING_ENDPOINT}:
        raise ValueError("Unsupported analysis endpoint")
    timing_only = endpoint == TIMING_ENDPOINT
    started = time.monotonic()
    checkpoint, platform, hotspot, output = map(lambda p: Path(p).resolve(), (checkpoint, platform, hotspot, output))
    if output.exists() or checkpoint.parent == output or output in checkpoint.parents:
        raise ValueError("Analysis output must be new and separate")
    cp, cts = validate_grt(checkpoint)
    handoff = load_routing_handoff(cts)
    before = {str(checkpoint): sha(checkpoint), str(cts): sha(cts), sta: sha(sta)}
    from . import spef
    from open3dbench.backend import passive_hbt, thermal as thermal_backend
    from scripts import workspace_cts_thermal
    for module in (post, passive_hbt, thermal_backend, workspace_cts_thermal, spef):
        before[str(Path(module.__file__).resolve())] = sha(module.__file__)
    # Include complete analysis collateral and exported GRT bytes in provenance.
    for root in (platform, checkpoint.parent):
        for path in root.rglob("*"):
            if path.is_file():
                before[str(path)] = sha(path)
    for path in (() if timing_only else (hotspot / "hotspot", hotspot / "examples/thermal/test.config", hotspot / "examples/thermal/test.materials")):
        before[str(path)] = sha(path)
    output.mkdir(parents=True)
    defs = {}
    for index, die in enumerate(("bottom", "top")):
        folder = output / die
        folder.mkdir()
        source = checkpoint.parent / die
        add_name_map(source / "global_route.odb.spef", folder / "raw.spef", source / "global_route.odb.v")
        post._spef_stats(folder / "raw.spef")
        shutil.copyfile(source / "global_route.odb.v", folder / "physical.v")
        rows = [{"source_die": r.source_die, "sink_die": r.sink_die, "source_port": r.source_port,
                 "sink_port": r.sink_port, "physical_net": r.physical_net} for r in handoff.boundaries]
        physical_boundaries = post._spef_hbt_nets(folder / "raw.spef")
        # Buffering may move a BTerm away from its pre-HBT net. The live GRT
        # SPEF records its actual current net; the historical handoff does not.
        physical_by_port = {port: net for net, ports in physical_boundaries.items() for port in ports}
        expected_ports = {str(row["source_port"] if row["source_die"] == die else row["sink_port"]) for row in rows}
        if set(physical_by_port) != expected_ports:
            raise ValueError("GRT boundary net mapping is incomplete")
        _repair_boundary_verilog(folder / "physical.v", die, rows, physical_by_port)
        _validate_boundary_verilog(folder / "physical.v", die, rows)
        post._normalize_flat_verilog(folder / "physical.v", folder / "design.v")
        roles = {}
        for row in handoff.boundaries:
            roles[row.source_port] = "source"
            roles[row.sink_port] = "sink"
        aliases = {}
        for net, ports in physical_boundaries.items():
            ordered = [p for p in ports if roles.get(p) == "sink"] or [p for p in ports if roles.get(p) == "source"]
            if ordered:
                aliases[net] = ordered[0]
        post._normalize_flat_spef(folder / "raw.spef", folder / "design.spef", aliases)
        defs[die] = source / "global_route.odb.def"
    assembly = output / "assembly"
    assembly.mkdir()
    links = passive_links(handoff.boundaries)
    write_passive_assembly(assembly / "design.v", output / "bottom/design.v", output / "top/design.v", links)
    plan = stitch_routed_spefs(output / "bottom/design.spef", output / "top/design.spef", assembly / "design.spef", links,
        resistance_ohm=float(handoff.timing_model["resistance_ohm"]), capacitance_pf=float(handoff.timing_model["capacitance_pf"]))
    expected = sum(post._spef_stats(output / d / "design.spef")["d_net_count"] for d in ("bottom", "top")) - len(links)
    if len(plan) != len(links) or post._spef_stats(assembly / "design.spef")["d_net_count"] != expected:
        raise ValueError("Passive HBT SPEF coverage mismatch")
    shutil.copyfile(handoff.assembly_sdc, assembly / "design.sdc")
    _, libs = _platform_files(platform)
    libs += sorted((cts.parent / "models").glob("*.lib"))
    _validate_netlist_liberty(assembly / "design.v", libs)
    env = {"REPORT_LIBERTY_SUPPRESSIONS": str(platform / "liberty_suppressions.tcl") if (platform / "liberty_suppressions.tcl").is_file() else "",
        "REPORT_LIBS": "\n".join(map(str, libs)), "REPORT_ASSEMBLY_V": str(assembly / "design.v"),
        "REPORT_ASSEMBLY_SDC": str(assembly / "design.sdc"), "REPORT_PASSIVE_SPEF": str(assembly / "design.spef"),
        "REPORT_EXPECTED_HBT_PINS": str(2 * len(links))}
    env["REPORT_LATCH_PROOF"] = str(assembly / "latch_proof.tsv")
    for key, filename in {"ANNOTATION":"parasitic_annotation.rpt", "CHECK_SETUP":"check_setup.rpt", "SETUP":"setup.rpt", "HOLD":"hold.rpt",
                          "CROSS_SETUP":"cross_die_setup.rpt", "CROSS_HOLD":"cross_die_hold.rpt", "POWER":"power.json",
                          "INSTANCE_POWER":"instance_power.rpt", "METRICS":"metrics.tsv"}.items():
        env["REPORT_" + key] = str(assembly / filename)
    # The maintained scoring view uses fixed, image-installed standalone STA.
    # Its linked hierarchy preserves boundary pins that physical OpenDB flattens.
    body = post.ASSEMBLY_STA_TCL.replace("report_power -format json >", "report_power -format json -digits 12 >").replace(
        "report_power -instances [get_cells -hier *] >", "report_power -instances [get_cells -hier *] -digits 12 >")
    if timing_only:
        body = "\n".join(line for line in body.splitlines() if not line.strip().startswith("report_power "))
    body = retain_data_checks(body) + LATCH_PROOF_TCL
    post._run_sta(body, env, assembly / "sta.log", sta,
        post._standalone_sta_suppression_rules(platform / "liberty_suppressions.tcl"))
    timing = _read_key_values(assembly / "metrics.tsv")
    if timing.get("sta_completeness_checks_passed") != 1 or timing.get("hbt_adapter_cell_count") != 0 or timing.get("passive_hbt_boundary_pin_count") != 2 * len(links):
        raise ValueError("Incomplete joint timing graph")
    latch_audit = audit_latches(assembly / "check_setup.rpt", assembly / "latch_proof.tsv")
    for analysis in ("setup", "hold"):
        if post._cross_die_path_count(assembly / ("cross_die_" + analysis + ".rpt")) == 0:
            raise ValueError("Missing cross-die timing paths")
    thermal, diagnostics = ({}, {}) if timing_only else _thermal(output, platform, defs, hotspot)
    grt = read(checkpoint.parent / cp["metrics"])
    metrics = {"wns_deficit_ns": max(0, -finite(timing.get("wns_setup"), "WNS")),
        "tns_deficit_ns": max(0, -finite(timing.get("tns_setup"), "TNS")),
        "overflow": finite(grt.get("grt_total_overflow"), "overflow", 0), **thermal}
    for name, value in metrics.items():
        finite(value, name, 0)
    if any(sha(path) != fingerprint for path, fingerprint in before.items()):
        raise ValueError("Immutable analysis input changed")
    result = {"schema_version": 1, "endpoint": endpoint, "status": "success", "metrics": metrics,
        "timing": timing, "diagnostics": diagnostics, "latch_audit": latch_audit, "wirelength_um": grt["grt_wire_length_um"],
        "parasitics": "global-route-estimated-rc", "signoff": False,
        "activity": 0.1, "ambient_k": 318.15, "grid_size": 10,
        "thermal_statistic": THERMAL_STATISTIC, "runtime_seconds": time.monotonic() - started}
    if timing_only:
        for key in ("activity", "ambient_k", "grid_size", "thermal_statistic"):
            result.pop(key)
    write(output / "metrics.json", result)
    write(output / "manifest.json", {"endpoint": endpoint, "source_checkpoint": str(checkpoint),
        "platform": str(platform),
        "source_checkpoint_sha256": sha(checkpoint), "inputs": before,
        "implementation_sha256": sha(__file__), "artifacts": records(output)})
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "platform", "hotspot", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--endpoint", choices=(ENDPOINT, TIMING_ENDPOINT), default=ENDPOINT)
    a = p.parse_args()
    run(a.checkpoint, a.platform, a.hotspot, a.output, endpoint=a.endpoint)


if __name__ == "__main__":
    main()
