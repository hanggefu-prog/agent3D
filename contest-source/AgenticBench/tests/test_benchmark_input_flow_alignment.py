"""Native stage-plan compatibility, without EDA or modifications to sealed runs."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from local_bench import benchmark_input as inputs


ROOT = Path(__file__).resolve().parents[2]


def native_plan(stop="partition", timing=True):
    plan = []
    for name, directory in inputs.PROFILE_STAGE_DIRS.items():
        plan.append({"stage": name, "directory": directory,
                     "skip": "not needed by partitioner" if name == "timing_extract" and not timing else None})
        if name == stop:
            break
    return plan


def stage(run, name):
    directory = run / inputs.PROFILE_STAGE_DIRS[name]
    directory.mkdir(parents=True)
    artifact = directory / "payload.txt"
    artifact.write_text(name + "\n")
    manifest = {"schema_version": 1, "stage": name, "status": "success",
                "signature": {"test": "fixed"}, "artifacts": {
                    "payload": {"path": artifact.name, "bytes": artifact.stat().st_size,
                                "sha256": inputs._sha256(artifact)}}}
    (directory / "stage_manifest.json").write_text(json.dumps(manifest))


def test_legacy_tp_gnn_profile_gains_real_step_15_from_native_plan():
    profile = inputs.load_benchmark_input_profile("bp_lol_ng45_tp_gnn_sta")
    original = (profile.root / "profile.yaml").read_bytes()
    stages = inputs._preparation_stages(profile, native_plan())
    assert list(stages) == ["input", "floorplan", "timing_extract", "partition"]
    assert (profile.root / "profile.yaml").read_bytes() == original


def test_optional_skipped_timing_extract_does_not_require_a_fake_manifest():
    profile = inputs.load_benchmark_input_profile("bp_lol_ng45_mtkahypar")
    assert inputs._preparation_stages(profile, native_plan(timing=False)) == profile.required_stages


def test_native_split_routing_maps_legacy_terminal_boundary():
    profile = inputs.load_benchmark_input_profile("bp_lol_ng45_mtkahypar_cts")
    profile = replace(profile, stop_stage="routing", required_stages={**profile.required_stages, "routing": "80_route"})
    stages = inputs._preparation_stages(profile, native_plan("detailed_routing", timing=False))
    assert list(stages)[-2:] == ["global_routing", "detailed_routing"]
    assert stages["global_routing"] == "75_global_route"


def test_plan_does_not_accept_omitting_required_producer_or_wrong_directory():
    profile = inputs.load_benchmark_input_profile("bp_lol_ng45_tp_gnn_sta")
    plan = native_plan()
    plan[-1]["skip"] = "pretend partition is optional"
    with pytest.raises(ValueError, match="required profile stage"):
        inputs._preparation_stages(profile, plan)
    plan = native_plan()
    plan[2]["directory"] = "20_partition/timing_input"
    with pytest.raises(ValueError, match="wrong directory"):
        inputs._preparation_stages(profile, plan)


def test_seed_frontend_prefix_resumes_at_real_timing_producer(tmp_path, monkeypatch):
    profile_id = "bp_lol_ng45_tp_gnn_sta"
    seed = tmp_path / "seed"
    stage(seed, "input")
    stage(seed, "floorplan")
    # A historical partition is not reusable when its separate step 15
    # prerequisite was never published. It must not be renamed or relabelled.
    stage(seed, "partition")
    original = {str(path.relative_to(seed)): path.read_bytes() for path in seed.rglob("*") if path.is_file()}
    monkeypatch.setattr(inputs, "_resolve_python", lambda value: Path(sys.executable))
    monkeypatch.setattr(inputs, "_python_identity", lambda value: {"version": "3.12.0"})
    monkeypatch.setattr(inputs, "_resolve_dreamplace", lambda *args: tmp_path / "runtime")
    monkeypatch.setattr(inputs, "_native_preparation_plan", lambda *args: native_plan())
    monkeypatch.setattr(inputs, "_git_identity", lambda value: {"root": str(value), "revision": "test", "dirty": False})
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert command[command.index("--from") + 1] == "timing_extract"
        run = Path(command[command.index("--run-dir") + 1])
        assert not (run / "20_partition").exists()
        stage(run, "timing_extract")
        stage(run, "partition")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(inputs.subprocess, "run", execute)
    output = tmp_path / "bundle"
    report = inputs.prepare_benchmark_input(profile_id, open3dbench_root=ROOT,
                                           output_dir=output, seed_run=seed)
    assert len(calls) == 1
    assert "timing_extract" in report["stages"]
    assert original == {str(path.relative_to(seed)): path.read_bytes() for path in seed.rglob("*") if path.is_file()}
    manifest = json.loads((output / "benchmark_input.json").read_text())
    assert manifest["stages"]["timing_extract"]["directory"] == "15_timing_extract"
    (output / "run/15_timing_extract/payload.txt").write_text("tampered")
    with pytest.raises(ValueError, match="artifact (size|hash) changed"):
        inputs.validate_benchmark_input(output)


def test_native_plan_is_read_from_dependency_checkout_without_eda(monkeypatch):
    profile = inputs.load_benchmark_input_profile("bp_lol_ng45_tp_gnn_sta")
    def execute(command, **kwargs):
        assert command[command.index("--repo") + 1] == str(ROOT)
        assert "plan" in command and "run" not in command
        assert kwargs["cwd"] == ROOT
        return SimpleNamespace(stdout=json.dumps({"plan": native_plan("thermal")}))
    monkeypatch.setattr(inputs.subprocess, "run", execute)
    assert inputs._native_preparation_plan(ROOT, Path(sys.executable), profile) == native_plan()
