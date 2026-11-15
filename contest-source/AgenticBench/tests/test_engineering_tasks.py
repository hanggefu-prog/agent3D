"""Synthetic contract tests. These are NOT native qualification receipts."""
import copy
import json
from pathlib import Path

import pytest

from engineering_bench.tasks import TASK_IDS, load_task, read_json, relative
from engineering_bench.metrics import aggregate, clock_skew, target_delay, timing_guard
from engineering_bench.protocol import OnceLedger, qualify, sha, write_new
from engineering_bench.judge import validate_review, review_prompt
from engineering_bench.submission import inventory, seal, verify_seal, validate_scope


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fake_qualification(folder, task, case):
    """Test fixture only. Production never accepts this test as native evidence."""
    folder.mkdir(parents=True, exist_ok=True)
    reference = dump(folder / "reference.json", {"case": case, "legal": True,
        "endpoint": task.data["endpoint"], "primary_metric": 1.0})
    evidence = dump(folder / "evidence.json", {"fixture": True})
    value = {"kind": "engineering-native-qualification-v1", "task": task.task_id,
        "task_digest": task.digest(), "case": case, "seed": task.data["seed"],
        "input_stage": task.data["input_stage"], "endpoint": task.data["endpoint"],
        "physical_success": True, "source_sha256": "a" * 64,
        "runtime_sha256": "b" * 64, "input_sha256": "c" * 64, "evaluator_sha256": "d" * 64,
        "reference": {"path": reference.name, "sha256": sha(reference)},
        "capabilities": {name: {"passed": True, "evidence": {"path": evidence.name, "sha256": sha(evidence)}}
                         for name in task.data["required_capabilities"]}}
    return dump(folder / "qualification.json", value)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_packaged_task_budget_and_prompt(task_id):
    task = load_task(task_id)
    assert task.data["cases"] == ["bp_fe", "bp", "ariane133", "swerv_wrapper"]
    assert task.data["seed"] == 1001
    assert task.data["submissions"] == task.data["evaluations_per_case"] == task.data["repetitions"] == 1
    assert task.instruction.isascii()  # Normal English developer instructions.
    assert "1:" in task.judge_instruction and "5:" in task.judge_instruction
    assert "explanation" in review_prompt(task)
    assert len(task.digest()) == 64
    assert not task.allows("AgenticBench/engineering_bench/metrics.py")
    assert not task.allows("open3dbench/backend/passive_hbt.py")


def test_stage_only_endpoints_and_acceptance_are_explicit():
    t2, t3, t4 = (load_task(task_id) for task_id in TASK_IDS)
    assert t2.data["endpoint"] == "post_cts_placement_rc_joint_sta"
    assert t3.data["endpoint"] == "post_transform_placement_rc_fixed_sink_joint_sta"
    assert t4.data["endpoint"] == "post_cts_placement_rc_joint_skew"
    assert "placement_rc_joint_sta" in t3.data["required_capabilities"]
    assert not any("grt" in cap for task in (t2, t3, t4) for cap in task.data["required_capabilities"])
    assert "setup_hold_guard" in t2.data["required_capabilities"]
    assert "setup_hold_guard" in t3.data["required_capabilities"]
    assert "setup_hold_guard" not in t4.data["required_capabilities"]
    for task in (t2, t3, t4):
        assert "physical_equivalence" in task.data["required_capabilities"]
        assert "clock_inventory_check" in task.data["required_capabilities"]
        assert "placement_legality" in task.data["required_capabilities"]


@pytest.mark.parametrize("value", ["../x", "/tmp/x", ".", "a/../x", "a\\b", "a//b", "x\n", ""])
def test_unsafe_paths(value):
    with pytest.raises(ValueError):
        relative(value)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_json_fail_closed(tmp_path, text):
    p = tmp_path / "x.json"
    p.write_text(text)
    with pytest.raises(ValueError):
        read_json(p)


def test_clock_domain_and_edge_are_never_mixed():
    rows = {"clk/rise/max/typ": {"a": .1, "b": .3},
            "other/fall/min/typ": {"c": 10., "d": 10.05}}
    expected = {k: list(v) for k, v in rows.items()}
    assert clock_skew(rows, expected) == pytest.approx(.2)
    del rows["clk/rise/max/typ"]["b"]
    with pytest.raises(ValueError, match="sink inventory"):
        clock_skew(rows, expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, True])
def test_incomplete_clock_arrival_is_not_zero(bad):
    with pytest.raises(ValueError):
        clock_skew({"clk": {"a": bad}}, {"clk": ["a"]})


def test_target_delay_includes_unsplit_targets_and_original_sinks():
    data = {"n1": {"a": 1., "b": 2.}, "n2": {"c": 4.}}
    targets = {"n1": ["a", "b"], "n2": ["c"]}
    assert target_delay(data, targets) == 3.
    with pytest.raises(ValueError):
        target_delay({"n1": data["n1"]}, targets)


def test_timing_guard_accepts_negative_baseline_not_new_hold_failures():
    keys = ("setup_wns_ns", "setup_tns_ns", "hold_wns_ns", "hold_tns_ns")
    ref = {k: -1. for k in keys} | {"hold_violating_endpoints": ["old"]}
    tolerances = {k: .01 for k in keys}
    assert timing_guard(ref, ref, tolerances)["passed"]
    new = dict(ref, hold_violating_endpoints=["old", "new"])
    assert not timing_guard(ref, new, tolerances)["passed"]
    new = dict(ref, setup_wns_ns=-1.02)
    assert timing_guard(ref, new, tolerances)["failures"] == ["setup_wns_ns"]


def test_timing_guard_preserves_endpoint_analysis_availability():
    keys = ("setup_wns_ns", "setup_tns_ns", "hold_wns_ns", "hold_tns_ns")
    ref = {k: -1. for k in keys} | {"hold_violating_endpoints": ["old"],
        "endpoints_without_min_path": ["unconstrained"], "endpoints_without_max_path": [], "endpoint_count": 3}
    tolerances = {k: 0. for k in keys}
    assert timing_guard(ref, ref, tolerances)["passed"]
    assert timing_guard(ref, dict(ref, endpoints_without_min_path=[]), tolerances)["failures"] == ["endpoints_without_min_path"]
    assert timing_guard(ref, dict(ref, endpoint_count=2), tolerances)["failures"] == ["endpoint_count"]
    bad = dict(ref)
    del bad["endpoints_without_max_path"]
    with pytest.raises(ValueError, match="availability"):
        timing_guard(ref, bad, tolerances)


def test_exactly_once_and_no_metric_for_failed_final(tmp_path):
    task = load_task("clock_compensation")
    proof = fake_qualification(tmp_path / "proof", task, "bp")
    ledger = OnceLedger(tmp_path / "run", task, "f" * 64)
    ledger.claim("bp", proof)
    with pytest.raises(FileExistsError):
        ledger.claim("bp", proof)
    with pytest.raises(ValueError):
        ledger.finish("bp", succeeded=False, legal=False, candidate_metric=0)
    ledger.finish("bp", succeeded=False, legal=False, error="Native exit")
    with pytest.raises(FileExistsError):
        ledger.finish("bp", succeeded=True, legal=True, candidate_metric=.7)
    with pytest.raises(ValueError, match="replace"):
        OnceLedger(tmp_path / "run", task, "a" * 64)
    assert ledger.records()[0]["candidate_metric"] is None


def test_qualification_failure_does_not_burn_final(tmp_path):
    task = load_task("cross_die_cts")
    proof = fake_qualification(tmp_path / "proof", task, "bp")
    value = read_json(proof)
    value["capabilities"].pop("physical_equivalence")
    dump(proof, value)
    ledger = OnceLedger(tmp_path / "run", task, "e" * 64)
    with pytest.raises(ValueError, match="Unqualified"):
        ledger.claim("bp", proof)
    assert ledger.records() == []


def test_qualification_hash_and_case_binding(tmp_path):
    task = load_task("multi_hbt_fanout")
    proof = fake_qualification(tmp_path / "proof", task, "bp")
    assert qualify(task, "bp", proof)["reference_metric"] == 1
    with pytest.raises(ValueError):
        qualify(task, "bp_fe", proof)
    (proof.parent / "evidence.json").write_text("changed")
    with pytest.raises(ValueError, match="Changed"):
        qualify(task, "bp", proof)


def test_result_average_never_hides_failure_or_running():
    task = load_task("clock_compensation")
    rows = [{"case": case, "status": "succeeded", "legal": True,
             "reference_metric": 2., "candidate_metric": 1.} for case in task.data["cases"]]
    assert aggregate(task, rows)["all_case_mean_improvement_percent"] == 50
    rows[-1] = {"case": "swerv_wrapper", "status": "dispatched"}
    result = aggregate(task, rows)
    assert result["all_case_mean_improvement_percent"] is None
    assert result["running_cases"] == ["swerv_wrapper"]
    assert result["failed_cases"] == []
    rows[-1]["status"] = "failed"
    assert aggregate(task, rows)["failed_cases"] == ["swerv_wrapper"]
    with pytest.raises(ValueError):
        aggregate(task, rows + [rows[0]])


def test_judge_evidence_and_ungradable(tmp_path):
    (tmp_path / "source.py").write_text("pass\n")
    review = {"status": "graded", "score": 3, "summary": "Partial implementation",
        "strengths": [], "issues": [{"explanation": "No action", "evidence": [{"path": "source.py", "line": 1}]}],
        "unverified": ["Native integration"]}
    assert validate_review(json.dumps(review), [tmp_path])["score"] == 3
    review["issues"][0]["evidence"][0]["line"] = 2
    with pytest.raises(ValueError):
        validate_review(json.dumps(review), [tmp_path])
    review.update(status="ungradable", score=None, issues=[])
    assert validate_review(json.dumps(review), [tmp_path])["score"] is None


def test_seal_tracks_added_files_and_protects_evaluator(tmp_path):
    task = load_task("clock_compensation")
    source = tmp_path / "source"
    source.mkdir()
    before = inventory(source)
    path = source / "engineering/clock_compensation.py"
    path.parent.mkdir()
    path.write_text("def plan_clock_compensation(context):\n    return []\n")
    value = seal(task, source, before, tmp_path / "sealed.json")
    assert value["changed_paths"] == ["engineering/clock_compensation.py"]
    assert verify_seal(source, value) == value["submission_sha256"]
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        verify_seal(source, value)
    (source / "score.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="outside"):
        validate_scope(task, before, inventory(source))


def test_changed_symlinks_are_not_code_submissions(tmp_path):
    task = load_task("clock_compensation")
    path = tmp_path / "engineering"
    path.mkdir()
    (path / "clock_compensation.py").symlink_to("/tmp/outside")
    with pytest.raises(ValueError, match="regular file"):
        validate_scope(task, {}, inventory(tmp_path))
