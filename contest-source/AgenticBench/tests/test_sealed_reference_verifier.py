"""Exercise the real standalone verifier argument and submission boundary."""
import json
import sys

import pytest

from global_bench.verifiers import open3dbench_codesign_verifier as verifier
from local_bench.reference_store import file_sha
from test_one_pass_codesign_contract import native_one_pass
from test_sta_codesign import _write_run
from test_workspace_reference import fixture, fresh  # shared synthetic producer fixture
from test_reference_evaluated import make_evaluation, seal_stages


@pytest.mark.parametrize("tampered", [False, True])
@pytest.mark.parametrize("rounds", [5, 10])
def test_real_cli_uses_pinned_record_and_real_candidate_metric_gate(fixture, monkeypatch, tampered, rounds):
    f = fixture
    from scripts.workspace_task_selection import load_registered_task, workspace_task_id
    f.task = load_registered_task(workspace_task_id("timing", rounds), objective="timing")
    selected, _, baseline = fresh(f)
    candidate = f.tmp / "candidate"
    _write_run(candidate, -0.5, -5.)
    native_one_pass(candidate)
    (candidate / "run.json").write_text(json.dumps({"design": "bp_fe"}))
    (candidate / "60_opt/checkpoint.json").write_text("{}")
    monkeypatch.setattr(verifier, "_validate_chain", lambda *a, **k: None)
    monkeypatch.setattr(verifier, "_validate_seed_contract", lambda *a, **k: None)
    contract = f.tmp / "evaluation-contract.json"
    contract.write_text(json.dumps({"case": "bp_fe", "task_digest": f.task.digest(),
        "evaluation_endpoint": {"profile": f.task.objective.evaluator_profile, "task_id": f.task.task_id},
        "source_revision": f.runtime.source_commit, "image": f.runtime.image,
        "input_manifest_sha256": file_sha(f.bundle / "benchmark_input.json")}))
    submission = f.tmp / "submission"
    submission.mkdir()
    artifacts = []
    for name, original in (("run_manifest", candidate / "run.json"),
        ("final_checkpoint", candidate / "60_opt/checkpoint.json"),
        ("metrics", candidate / "60_opt/metrics.json")):
        target = submission / (name + ".json")
        target.write_bytes(original.read_bytes())
        artifacts.append({"name": name, "submitted_path": target.name, "sha256": file_sha(target)})
    (submission / "submission.json").write_text(json.dumps({"task_id": f.task.task_id,
        "task_digest": f.task.digest(), "base_revision": f.runtime.source_commit,
        "trial_id": "round-01", "artifacts": artifacts}))
    output = f.tmp / "verification"
    sha = "0" * 64 if tampered else file_sha(selected.record_path)
    monkeypatch.setattr(sys, "argv", ["verifier", "--mode", "timing", "--profile", f.task.objective.evaluator_profile,
        "--submission", str(submission), "--output", str(output), "--evaluated-run", str(candidate),
        "--reference-run", baseline["run"], "--reference-record", str(selected.record_path),
        "--reference-record-sha256", sha, "--evaluation-contract", str(contract)])
    assert verifier.main() == (2 if tampered else 0)
    receipt = json.loads((output / "verification.json").read_text())
    assert receipt["passed"] is (not tampered)
    if tampered:
        assert receipt["metrics"] == {}
        assert "pinned record digest" in receipt["errors"][0]
    else:
        assert receipt["metrics"]["combined_score"] == pytest.approx(2.)
        assert receipt["provenance"]["reference_record_sha256"] == sha


@pytest.mark.parametrize("threads", [2, 4, None])
def test_sealed_cts_candidate_power_threads_match_reference_contract(make_evaluation, tmp_path, threads):
    from local_bench import reference_catalog as catalog, reference_evaluated as native
    output, run, profile = make_evaluation("cts_thermal", "reference")
    producer = {"kind": "native-evaluation-v1", "case": "bp_fe", "endpoint": "cts_thermal",
                "profile": profile, "evaluation_root": str(output), "run_root": str(run)}
    task = native.load_global_task(native.FRAMEWORK / "global_tasks/lol_partition_placement_cts_thermal_one_pass")
    payload = catalog.qualify_producer(producer)
    contract = catalog.make_contract(payload, task_digest=catalog.task_digest(task.root),
                                     evaluator_digest=catalog.evaluator_digest())
    record = catalog.register_producer(tmp_path / "store", producer, origin="FRESH", contract=contract)
    candidate_output, candidate, _ = make_evaluation("cts_thermal", "candidate")
    metrics = candidate / "95_thermal_snapshot/metrics.json"
    value = json.loads(metrics.read_text())
    value["power"]["provenance"]["threads"] = threads
    metrics.write_text(json.dumps(value))
    seal_stages(candidate)
    submission = {"task_id": task.task_id, "task_digest": task.digest(), "base_revision": "1" * 40}
    def score():
        return verifier.score_with_reference_record(candidate, run, record,
            candidate_output / "evaluation-contract.json", mode="cts-thermal", profile=profile,
            submission=submission)
    if threads == 2:
        assert score()["combined_score"] == 1.
    else:
        with pytest.raises(ValueError, match="power extraction threads differ"):
            score()


@pytest.mark.parametrize("rounds", [5, 10])
def test_native_timing_register_adopt_and_score_share_exact_task_authority(make_evaluation, tmp_path, rounds):
    from local_bench import reference_catalog as catalog
    from scripts.workspace_task_selection import workspace_task_id, load_registered_task
    output, run, profile = make_evaluation("timing", "reference", rounds=rounds)
    task = load_registered_task(workspace_task_id("timing", rounds), objective="timing")
    producer = {"kind": "native-evaluation-v1", "case": "bp_fe", "endpoint": "timing",
                "profile": profile, "evaluation_root": str(output), "run_root": str(run)}
    qualified = catalog.qualify_producer(producer)
    contract = catalog.make_contract(qualified, task_digest=catalog.task_digest(task.root),
                                     evaluator_digest=catalog.evaluator_digest())
    record = catalog.register_producer(tmp_path / "store", producer, origin="FRESH", contract=contract)
    before = record.read_bytes()
    catalog.adopt(record, contract=contract, output=tmp_path / "adoption")
    candidate_output, candidate, _ = make_evaluation("timing", "candidate", rounds=rounds)
    submission = {"task_id": task.task_id, "task_digest": task.digest(), "base_revision": "1" * 40}
    result = verifier.score_with_reference_record(candidate, run, record,
        candidate_output / "evaluation-contract.json", mode="timing", profile=profile, submission=submission)
    assert result["combined_score"] == 1.0
    assert record.read_bytes() == before
    assert json.loads(record.read_text())["contract"]["task_digest"] == catalog.task_digest(task.root)
    assert qualified["source_identity"]["task_digest"] == task.digest()
    other = load_registered_task(workspace_task_id("timing", 10 if rounds == 5 else 5), objective="timing")
    with pytest.raises(ValueError):
        catalog.adopt(record, contract={**contract, "task_digest": catalog.task_digest(other.root)},
                      output=tmp_path / "wrong-budget-adoption")
    with pytest.raises(ValueError, match="trusted candidate task"):
        verifier.score_with_reference_record(candidate, run, record,
            candidate_output / "evaluation-contract.json", mode="timing", profile=profile,
            submission={**submission, "task_id": other.task_id, "task_digest": other.digest()})
