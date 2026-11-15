"""Offline contract tests, not native six-case qualification."""
import ast
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_bench import metrics, post_placement_timing as timing
from local_bench.model import EvaluationResult, load_task, load_candidate
from local_bench.optimization_feedback import optimization_result, TASK_IDS
from local_bench.program_bridge import materialize_program
from local_bench.prompting import build_problem_description, composed_system_message
from local_bench.verifier import verify_candidate

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "AgenticBench/local_tasks"
NAMES = ("tp_gnn_post_placement_timing", "placement_initialization", "placement_preconditioner")


@pytest.mark.parametrize("name", NAMES)
def test_initial_program_and_candidate_agree_and_pass_policy(name):
    spec = load_task(TASKS / name)
    candidate = materialize_program(spec, spec.program_template.read_text(), ROOT)
    if name == "tp_gnn_post_placement_timing":
        from test_tp_gnn_native_adapter_compat import assert_historical_initial_candidate
        assert_historical_initial_candidate(spec, candidate,
            "d3d1f6a44a95c2a278c3556df12aac981aec8f94446804521e96f049d873f993")
    else:
        assert candidate == load_candidate(spec.initial_candidate)
    verify_candidate(spec, candidate)
    assert name in TASK_IDS
    text = build_problem_description(spec, candidate, ROOT)
    assert "EVOLVE-BLOCK" in text
    assert composed_system_message(spec)


@pytest.mark.parametrize("name,file,old,new", [
    ("placement_initialization", "BasicPlace.py", "init_pos = np.zeros", "init_pos = np.zeros"),
    ("placement_preconditioner", "PlaceObj.py", "self.alpha = 1.0", "self.alpha = 1.1"),
])
def test_component_patch_cannot_touch_other_component(name, file, old, new):
    spec = load_task(TASKS / name)
    program = spec.program_template.read_text()
    if name == "placement_initialization":
        program = program.replace("    return init_pos", "    init_pos = init_pos.copy()\n    return init_pos")
    else:
        program = program.replace(old, new)
    candidate = materialize_program(spec, program, ROOT)
    verify_candidate(spec, candidate)
    patch = candidate["source_patch"]
    assert file in patch
    assert ("PlaceObj.py" if file == "BasicPlace.py" else "BasicPlace.py") not in patch
    assert "NonLinearPlace.py" not in patch
    assert "_agentic3d_build_die_contexts(placedb)" in patch
    assert "macro_halo_x" not in candidate["candidate_files"]["placement.json"]["concurrent_dreamplace"]


@pytest.mark.parametrize("name", NAMES[1:])
def test_component_rejects_joint_program(name):
    spec = load_task(TASKS / name)
    joint = "\n".join((TASKS / part / "initial_program.py").read_text()
                      for part in NAMES[1:])
    with pytest.raises(ValueError, match="only"):
        materialize_program(spec, joint, ROOT)


@pytest.mark.parametrize("name", NAMES[1:])
def test_component_baseline_has_no_source_patch(name):
    candidate = load_candidate(load_task(TASKS / name).initial_candidate)
    assert candidate["source_patch"] == ""
    recipe = candidate["candidate_files"]["placement.json"]["concurrent_dreamplace"]
    assert recipe["iterations"] == 4000 and recipe["stop_overflow"] == .01


@pytest.mark.parametrize("candidate,reference,expected", [
    (0, 0, 1), (1e-9, .005, 1), (.01, .1, 10), (.02, .01, .5),
])
def test_overflow_score_saturates_at_native_tolerance(monkeypatch, candidate, reference, expected):
    monkeypatch.setattr(metrics, "_placement_native", lambda path: {
        "placement_native_hpwl": 100., "placement_overflow_max": candidate if path == Path("c") else reference})
    result = metrics.placement_component_metrics(Path("c"), Path("r"))
    assert result["overflow_score"] == expected
    assert result["combined_score"] == pytest.approx(.9 + .1 * expected)


@pytest.mark.parametrize("wns,tns,expected", [(-1, -10, 1.), (-.5, -5, 2.), (-2, -5, 1/1.4), (0, 0, 1e12)])
def test_sta_weighting_uses_no_opt_final_global_values(monkeypatch, wns, tns, expected):
    def row(path, *, expected_decoder):
        assert expected_decoder == "spatial-bin"
        return {"metrics": {"wns_setup": wns if path == "c" else -1.,
                            "tns_setup": tns if path == "c" else -10.},
                "hbt_model": {"test": 1}, "partition_config": {"decoder": "spatial-bin"},
                "bin_max_area_error": .01}
    monkeypatch.setattr(timing, "validate_snapshot", row)
    monkeypatch.setattr(metrics, "tp_gnn_learning_objective_metrics", lambda *args: {
        "partition_constraints_passed": 1., "timing_weighted_cut_score": 9999.})
    result = timing.timing_metrics("c", "r")
    assert result["combined_score"] == pytest.approx(expected)
    assert result["opt_passes"] == 0
    assert "timing_weighted_cut_score" not in result


def test_sta_command_consumes_hbt_not_opt():
    command = timing.snapshot_command(Path("/source"), Path("/run"), {"pdk": "ng45"}, "python", {})
    assert "/run/50_hbt/checkpoint.json" in command
    assert "open3dbench.backend.post_placement_sta" in command
    assert not any("60_opt" in value or "--to" == value for value in command)
    spec = load_task(TASKS / NAMES[0])
    assert (spec.start_stage, spec.stop_stage, spec.upstream_stages) == ("partition", "hbt", ["input", "floorplan"])


def test_no_opt_refuses_later_stage_before_reading_metrics(tmp_path):
    (tmp_path / "60_opt").mkdir()
    with pytest.raises(ValueError, match="later physical stage"):
        timing.validate_snapshot(tmp_path)


def test_feedback_contains_timing_not_cut_fitness_or_host_paths():
    result = EvaluationResult({"combined_score": 1.2, "final_global_wns_setup_ns": -.2,
        "final_global_tns_setup_ns": -3., "timing_weighted_cut": 42,
        "timing_weighted_cut_score": 9, "opt_passes": 0, "unknown_metric": 77},
        {"secret_path": "/host/private", "runtime": "/host/private"})
    feedback = optimization_result(result, task_id="tp_gnn_bin_timing")
    assert feedback.metrics["final_global_wns_setup_ns"] == -.2
    assert "timing_weighted_cut_score" not in feedback.metrics
    assert "unknown_metric" not in feedback.metrics
    assert "/host/private" not in json.dumps(feedback.to_dict())
    assert "without timing optimization" in feedback.artifacts["optimization_feedback"]


def snapshot_fixture(root, monkeypatch, *, decoder="spatial-bin"):
    report = {"method": "tp-gnn-flat", "hierarchy": {"mode": "flat", "consumed_as_feature": False},
              "config": {"decoder": decoder}, "training": {"loss_context_interface": "timing-context-v2"},
              "clustering": {"spatial_bins": {"postclosure": {"accepted": True, "max_area_error": .02},
                                             "handoff_constraints": {"accepted": True}}}}
    if decoder == "global-2way":
        report["clustering"] = {}
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    def record(path):
        return {"path": path.name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for stage, directory in (("partition", "20_partition"), ("placement", "30_placement"),
                             ("pdn", "40_pdn"), ("hbt", "50_hbt")):
        artifact = root / directory / "partition_manifest.json"
        write(artifact, {"method": "tp-gnn-flat", "method_report": report})
        write(root / directory / "stage_manifest.json", {"schema_version": 1, "stage": stage,
            "status": "success", "signature": {}, "method": "tp-gnn-flat", "artifacts": {"method_manifest": record(artifact)}})
    write(root / "20_partition/tp_gnn_report.json", report)
    checkpoint = root / "50_hbt/checkpoint.json"
    write(checkpoint, {"stage": "hbt"})
    monkeypatch.setattr("open3dbench.backend.opt_handoff.load_opt_handoff", lambda *args: SimpleNamespace(boundaries=[]))
    directory = root / timing.STA_DIRECTORY
    artifact = directory / "evidence.txt"
    write(artifact, {"test": "offline-only"})
    names = ("timing_snapshot", "setup_report", "hold_report", "check_setup_report",
             "parasitic_annotation_report", "passive_assembly_verilog", "passive_assembly_spef",
             "hbt_boundary_timing", "replay_tcl")
    row = {"stage": "post-placement-sta", "status": "success",
           "input": {"required_checkpoint_stage": "hbt", "physical_placement_mutated": False,
                     "checkpoint_sha256": record(checkpoint)["sha256"]},
           "analysis": {"timing_view": "unified-flat-two-die", "parasitics": "placement-estimated-die-rc-plus-passive-hbt-pi",
                        "setup": "max-delay", "signoff_ready": False},
           "artifacts": {k: artifact.name for k in names}, "artifact_records": {k: record(artifact) for k in names},
           "hbt_counts": {"total": 0}, "boundary_coverage": {"sta_unconstrained_endpoints": 0},
           "reported_paths": {"setup": {"reported_path_count": 1}}, "metrics": {"wns_setup": -.1, "tns_setup": -1.}}
    write(directory / "metrics.json", row)
    return row, directory / "metrics.json", artifact


def test_snapshot_validates_actual_artifact_hashes(tmp_path, monkeypatch):
    _, _, artifact = snapshot_fixture(tmp_path, monkeypatch)
    assert timing.validate_snapshot(tmp_path)["metrics"]["wns_setup"] == -.1
    artifact.write_text("tampered")
    with pytest.raises(ValueError, match="size changed|hash changed"):
        timing.validate_snapshot(tmp_path)


@pytest.mark.parametrize("value", [None, "0", True, float("nan"), float("inf")])
def test_missing_timing_cannot_become_zero(tmp_path, monkeypatch, value):
    row, path, _ = snapshot_fixture(tmp_path, monkeypatch)
    row["metrics"]["tns_setup"] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="finite global"):
        timing.validate_snapshot(tmp_path)


@pytest.mark.parametrize("task", NAMES)
def test_missing_reference_only_allows_exact_initial_program(task):
    from local_bench.scoped_reference import admit_reference
    spec = load_task(TASKS / task)
    candidate = materialize_program(spec, spec.program_template.read_text(), ROOT)
    assert admit_reference(spec, candidate, ROOT, Path("baseline"), None)["mode"] == "FRESH"
    candidate["notes"] += "changed"
    with pytest.raises(ValueError, match="exact native"):
        admit_reference(spec, candidate, ROOT, Path("baseline"), None)
