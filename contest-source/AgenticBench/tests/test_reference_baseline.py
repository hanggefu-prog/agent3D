"""Reference-family contract tests; no Docker, model or native EDA calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from local_bench import reference_baseline as ref
from local_bench.benchmark_input import load_benchmark_input_profile
from scripts.codesign_runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case", ["bp_fe", "bp", "ariane133", "swerv_wrapper"])
def test_reference_profiles_pin_common_physics(case):
    profile = load_benchmark_input_profile(ref.profile_path(case) / "profile.yaml")
    config = yaml.safe_load(profile.config.read_text())
    assert config["design"] == case
    assert profile.partition_method == "tp-gnn-flat"
    assert config["threads"] == config["opt"]["threads"] == config["routing"]["threads"] == 4
    assert config["partition"]["algorithm"] == "tp-gnn-flat"
    assert config["partition"]["seed"] == 1001 and config["placement"]["seed"] == 1000
    assert config["cts"]["repair_hold"] is False
    assert config["routing"]["congestion_iterations"] == 4
    assert config["routing"]["max_iterations"] == 5
    assert config["routing"]["die_timeout_seconds"] == 10800
    assert config["routing"]["resource_reduction"] == 0.0
    assert config["thermal"]["grid_size"] == 10 and config["thermal"]["ambient_k"] == 318.15
    assert ref.validate_reference_config(config, case)["baseline_id"] == ref.BASELINE_ID
    changed = copy.deepcopy(config)
    changed["threads"] = 32
    with pytest.raises(ValueError, match="configuration differs"):
        ref.validate_reference_config(changed, case)


def test_native_frontend_configs_match_frozen_recipe():
    partition = json.loads((ROOT.parent / ref.PARTITION_CONFIG).read_text())["partition"]
    placement = json.loads((ROOT.parent / ref.PLACEMENT_CONFIG).read_text())["concurrent_dreamplace"]
    assert partition["method"] == "tp-gnn-flat"
    assert partition["tp_gnn_decoder"] == "global-2way"
    assert partition["tp_gnn_epochs"] == 20 and partition["tp_gnn_min_optimizer_steps"] == 100
    assert placement["iterations"] == 4000 and placement["learning_rate"] == .01
    assert placement["num_bins"] == 512 and placement["target_density"] == .6
    assert placement["macro_place_flag"] is True
    assert placement["density_weight"] == .00003
    assert placement["initialization_seed"] == 1000
    # Check actual native parser contracts without importing GPU dependencies.
    def literal_assignment(relative, name):
        tree = ast.parse((ROOT.parent / relative).read_text())
        return next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name
                            for target in node.targets))
    assert set(partition) <= set(literal_assignment(
        "Place-3D/src/partition/lol/cli.py", "PARTITION_DEFAULTS"))
    assert set(placement) == literal_assignment(
        "Place-3D/src/placement/concurrent_dreamplace.py", "FLOW_CONFIG_KEYS")


@pytest.mark.parametrize("case", ["bp_fe", "bp", "ariane133", "swerv_wrapper"])
def test_all_native_plans_resolve_versioned_config_paths(case):
    from open3dbench.config import load_config
    from open3dbench.runner import FlowRunner
    runner = FlowRunner(ROOT.parent, load_config(ref.profile_path(case) / "config.yaml"))
    plan = runner.plan()  # Planning only: no filesystem outputs or native tools.
    assert next(row for row in plan if row["stage"] == "timing_extract")["skip"] is None
    assert runner._resolve_place(runner.config["placement"]["config"]).is_file()


def test_reference_placement_preserves_native_kernel_flags(tmp_path):
    from local_bench.evaluation import _deterministic_dreamplace_params
    run = tmp_path / "run"
    path = run / "10_floorplan/design_params.json"
    path.parent.mkdir(parents=True)
    original = {"num_threads": 4, "def_input": "frozen.def"}
    path.write_text(json.dumps(original))
    produced = _deterministic_dreamplace_params(run, tmp_path / "new", preserve_reference=True)
    assert json.loads(produced.read_text()) == original
    legacy = _deterministic_dreamplace_params(run, tmp_path / "legacy")
    assert json.loads(legacy.read_text())["deterministic_flag"] == 1
    original["deterministic_flag"] = 1
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="runtime parameters differ"):
        _deterministic_dreamplace_params(run, tmp_path / "bad", preserve_reference=True)


def test_reference_local_command_keeps_native_macro_guard(tmp_path):
    from local_bench.evaluation import _dreamplace_global_placement_command
    from local_bench.model import load_task
    spec = load_task(ROOT / "local_tasks/placement_initialization")
    run = tmp_path / "run"
    for relative in ("20_partition/partition_assignment.csv", "20_partition/die_regions.json",
                     "10_floorplan/design_params.json"):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    config = yaml.safe_load(spec.base_config.read_text())
    command, _ = _dreamplace_global_placement_command(spec, config, ROOT.parent, run, "python")
    assert command[command.index("--macro-snap-guard-um") + 1] == "1.001"
    assert command[command.index("--macro-halo-x") + 1] == "10.0"
    assert command[command.index("--macro-halo-y") + 1] == "10.0"


def test_runtime_selects_new_profile_only_explicitly():
    runtime = object.__new__(Runtime)
    runtime.framework = ROOT
    assert runtime.profile("bp_fe").name == "bp_fe_lol_ng45_tp_gnn_sta"
    runtime.reference_baseline = ref.BASELINE_ID
    assert runtime.profile("bp_fe") == ref.profile_path("bp_fe")


def test_wrong_image_rejected_without_execution():
    for value in ref.IMAGE_IDS:
        assert ref.validate_reference_image(value) == value
    with pytest.raises(ValueError, match="pinned"):
        ref.validate_reference_image("sha256:" + "a" * 64)


@pytest.fixture
def physical(tmp_path, monkeypatch):
    run = tmp_path / "run"
    path = run / "10_floorplan/design.def"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"frozen physical geometry")
    part = run / "20_partition/partition_assignment.csv"
    part.parent.mkdir()
    part.write_bytes(b"inst_name,die\na,bottom\n")
    record = {"cases": {"bp_fe": {"physical_prefix": {
        "10_floorplan/design.def": {"bytes": path.stat().st_size, "sha256": ref.sha256(path)}},
        "partition_assignment_sha256": ref.sha256(part)}}}
    monkeypatch.setattr(ref, "identity", lambda: record)
    return run


def test_physical_identity_is_not_inferred_from_same_seed(physical):
    assert ref.validate_physical_prefix(physical, "bp_fe", fixed_partition=True)
    (physical / "10_floorplan/design.def").write_bytes(b"different floorplan")
    with pytest.raises(ValueError, match="Physical input differs"):
        ref.validate_physical_prefix(physical, "bp_fe")


def test_local_fixed_partition_cannot_silently_repartition(physical):
    (physical / "20_partition/partition_assignment.csv").write_text("inst_name,die\na,top\n")
    assert ref.validate_physical_prefix(physical, "bp_fe")
    with pytest.raises(ValueError, match="Fixed placement partition"):
        ref.validate_physical_prefix(physical, "bp_fe", fixed_partition=True)


def test_archive_index_must_match_before_adoption(tmp_path):
    (tmp_path / "reference.json").write_text("{}")
    with pytest.raises(ValueError, match="index identity"):
        ref.validate_reference_root(tmp_path, "bp_fe")


def test_symlinked_physical_input_rejected(physical):
    target = physical / "10_floorplan/design.def"
    original = target.read_bytes()
    target.unlink()
    outside = physical.parent / "outside.def"
    outside.write_bytes(original)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="regular frozen"):
        ref.validate_physical_prefix(physical, "bp_fe")
