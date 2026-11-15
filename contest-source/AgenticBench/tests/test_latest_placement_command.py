"""Standalone latest-family command construction, without native execution."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from local_bench import evaluation, reference_latest as latest
from local_bench.model import load_task, resolve_task_path

FRAMEWORK = Path(__file__).resolve().parents[1]
SOURCE = FRAMEWORK.parent
TASKS = ("placement_dynamic_parameters_reference_20260923",)


@pytest.fixture
def prepared(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "open3dbench").mkdir(parents=True)
    (workspace / "Place-3D").mkdir()
    pdk = SOURCE / "open3dbench/pdks.py"
    assert hashlib.sha256(pdk.read_bytes()).hexdigest() == evaluation.LATEST_REFERENCE_PDK_SHA256
    (workspace / "open3dbench/pdks.py").write_bytes(pdk.read_bytes())
    run = tmp_path / "run"
    (run / "20_partition").mkdir(parents=True)
    (run / "10_floorplan").mkdir()
    (run / "20_partition/partition_assignment.csv").write_text("inst_name,die\nu,bottom\n")
    (run / "20_partition/die_regions.json").write_text("{}")
    params = run / "10_floorplan/design_params.json"
    params.write_text(json.dumps({"num_threads": 4, "deterministic_flag": 0, "use_bb": 0}))
    config = latest.bind_recipe_paths(yaml.safe_load((latest.profile_path("bp_fe") / "config.yaml").read_text()), SOURCE)
    return SimpleNamespace(root=tmp_path, workspace=workspace, run=run, params=params, config=config)


def arguments(env, task=TASKS[0]):
    return [load_task(resolve_task_path(FRAMEWORK / "local_tasks", task)), env.config,
            env.workspace, env.run, sys.executable]


def test_literal_contract_has_exact_frozen_native_pdk_parity(prepared):
    # Native Python is loaded only by this offline parity test, by exact file
    # path whose bytes are independently bound above; production does not load it.
    path = prepared.workspace / "open3dbench/pdks.py"
    spec = importlib.util.spec_from_file_location("pinned_pdk_parity_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        pdk = module.get_pdk("ng45")
        native = max(pdk.macro_track_pitch_um, pdk.macro_origin_x_pitch_um,
                     pdk.macro_pdn_x_pitch_um) + 0.001
        assert native == evaluation._latest_reference_snap_guard(prepared.workspace, prepared.config) == 1.001
        assert pdk.macro_halo_um == prepared.config["placement"]["macro_halo_um"] == 10.0
    finally:
        del sys.modules[spec.name]


@pytest.mark.parametrize("task", TASKS)
def test_real_standalone_framework_command_without_physical_python_import(prepared, task):
    config = prepared.root / "resolved-config.yaml"
    config.write_text(yaml.safe_dump(prepared.config))
    before = prepared.params.read_bytes()
    params = evaluation._deterministic_dreamplace_params(
        prepared.run, prepared.root / "evaluation", preserve_reference=True)
    # -I excludes inherited PYTHONPATH and cwd. Only the explicit framework is
    # inserted. A hostile/missing physical Python package cannot be consulted.
    code = r'''
import sys,json,importlib.abc
from pathlib import Path
class RejectFlowImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname == "open3dbench" or fullname.startswith("open3dbench."):
            raise RuntimeError("Physical Python import forbidden in standalone control plane")
sys.meta_path.insert(0,RejectFlowImports())
sys.path.insert(0,sys.argv[1])
import yaml
from local_bench.evaluation import _dreamplace_global_placement_command
from local_bench.model import load_task,resolve_task_path
spec=load_task(resolve_task_path(Path(sys.argv[1])/"local_tasks",sys.argv[2]))
command,cwd=_dreamplace_global_placement_command(spec,yaml.safe_load(Path(sys.argv[3]).read_text()),
    Path(sys.argv[4]),Path(sys.argv[5]),sys.executable,base_params_override=Path(sys.argv[6]),
    reference_family="reference_20261004")
print(json.dumps({"command":command,"cwd":str(cwd),"native_dispatches":0}))
'''
    proc = subprocess.run([sys.executable, "-B", "-I", "-c", code, str(FRAMEWORK), task,
        str(config), str(prepared.workspace), str(prepared.run), str(params)],
        cwd=prepared.root, env={**os.environ, "PYTHONPATH": str(prepared.root / "untrusted")},
        text=True, capture_output=True, check=False, timeout=30)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    command = result["command"]
    for option, expected in (("--macro-halo-x", "10.0"), ("--macro-halo-y", "10.0"),
                             ("--macro-snap-guard-um", "1.001"), ("--base-params", str(params)),
                             ("--target-density", "0.6"), ("--gpu", "1")):
        assert command[command.index(option) + 1] == expected
    assert command[1:3] == ["-m", "src.placement.concurrent_dreamplace"]
    assert result["cwd"] == str(prepared.workspace / "Place-3D")
    assert result["native_dispatches"] == 0
    assert prepared.params.read_bytes() == before
    assert not (prepared.run / "30_placement").exists()


@pytest.mark.parametrize("fault", ["pdk-bytes", "pdk-link", "parent-link", "pdk-name", "halo", "family", "partition-family"])
def test_latest_command_contract_mutations_fail_closed(prepared, fault):
    path = prepared.workspace / "open3dbench/pdks.py"
    family = latest.BASELINE_ID
    if fault == "pdk-bytes":
        path.write_text(path.read_text() + "\n")
    elif fault == "pdk-link":
        path.unlink(); path.symlink_to(SOURCE / "open3dbench/pdks.py")
    elif fault == "parent-link":
        path.parent.rename(prepared.workspace / "relocated")
        path.parent.symlink_to(prepared.workspace / "relocated", target_is_directory=True)
    elif fault == "pdk-name":
        prepared.config["pdk"] = "asap7"
    elif fault == "halo":
        prepared.config["placement"]["macro_halo_um"] = 9.0
    elif fault == "family":
        family = "unregistered-next"
    else:
        prepared.config["partition"]["config"] = "/untrusted/partition.json"
    with pytest.raises(ValueError):
        evaluation._dreamplace_global_placement_command(*arguments(prepared), reference_family=family)


@pytest.mark.parametrize("family", [None, "reference_20260923"])
def test_legacy_family_still_uses_its_original_pdk_lookup(prepared, monkeypatch, family):
    observed = []
    def get_pdk(name):
        observed.append(name)
        return SimpleNamespace(macro_track_pitch_um=0.14, macro_origin_x_pitch_um=1.0,
                               macro_pdn_x_pitch_um=0.0)
    monkeypatch.setitem(sys.modules, "open3dbench.pdks", SimpleNamespace(get_pdk=get_pdk))
    # No new pin is imposed on the old family and its original lookup remains
    # authoritative. This does not admit latest artifacts to an old registry.
    (prepared.workspace / "open3dbench/pdks.py").write_text("legacy fixture")
    command, _ = evaluation._dreamplace_global_placement_command(*arguments(prepared), reference_family=family)
    assert observed == ["ng45"]
    assert command[command.index("--macro-snap-guard-um") + 1] == "1.001"
