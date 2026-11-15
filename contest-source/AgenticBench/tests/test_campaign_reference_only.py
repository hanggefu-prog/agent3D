"""Reference-only control-plane fixtures; no Docker, model or solver processes."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_bench.model import EvaluationResult
from scripts import qualify_campaign_reference as module


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    batch = tmp_path / "runs"
    bundle = batch / "input/bundle"
    (bundle / "run").mkdir(parents=True)
    (bundle / "benchmark_input.json").write_text("{}")
    output = batch / "campaign/reference-timing"
    store = batch / "campaign-store"
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text("design: bp_fe\nthreads: 4\n")
    archive = tmp_path / "archive"
    archive.mkdir()
    calls = SimpleNamespace(obtain=0, publish=0, requalify=0, evaluators=[], transports=[])
    options = SimpleNamespace(success=True, requalify_failure=False, score=1.0, mutate_record=False)
    runtime = SimpleNamespace(source=tmp_path / "source", batch=batch, framework=module.ROOT,
        source_commit="a" * 40, image="sha256:" + "b" * 64, cpus=16, memory_gb=48,
        min_gpu_free_mb=2048, gpus=[2])
    def contained(path):
        value = Path(path).resolve()
        value.relative_to(batch)
        return value
    runtime.contained = contained
    runtime.profile = lambda case: profile
    monkeypatch.setattr(module, "validate_benchmark_input", lambda *a, **kw:
        {"config": str(profile / "config.yaml"), "run_dir": str(bundle / "run")})
    monkeypatch.setattr(module.baseline, "validate_reference_bundle", lambda *a, **kw:
        {"reference_root": str(archive), "original_root": str(archive)})
    monkeypatch.setattr(module, "protect_destination", lambda *a, **kw: a[0])
    selections = []

    class Selection:
        def __init__(self, runtime, *args, **kwargs):
            self.record_path = self.record_sha256 = None
            self.contract = {"endpoint": "timing", "task_digest": "c" * 64}
            self.source_run = output / "reference/result/run"
            selections.append(self)
        def to_dict(self):
            return {"policy": "FRESH", "contract": self.contract}
        def publish(self, evaluation):
            calls.publish += 1
            self.record_path = store / "records/reference.json"
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.record_path.write_text(json.dumps({"origin": "FRESH", "reference_id": "c" * 64,
                "qualification": {"baseline": {"source_run": str(self.source_run)}}}))
            self.record_sha256 = module.file_sha(self.record_path)
        def obtain(self, evaluator, evaluation):
            calls.obtain += 1
            evaluation.mkdir(parents=True)
            result = {"success": options.success, "metrics": {"combined_score": options.score},
                "execution": {"command_phases": {
                    "frontend": {"dispatch_attempted": True, "process_started": True},
                    "backend": {"dispatch_attempted": True, "process_started": True},
                    "verifier": {"dispatch_attempted": True, "process_started": True}}}}
            (evaluation / "feedback.json").write_text(json.dumps(result))
            if options.success:
                self.publish(evaluation)
            return result
        def prepare_fresh(self, evaluation, candidate):
            return evaluation.parent / "reference-purpose.json"
        def qualified_result(self, **kwargs):
            return EvaluationResult({"runs_successfully": 1, "combined_score": 1.0}, {})
    monkeypatch.setattr(module, "WorkspaceReference", Selection)
    monkeypatch.setattr(module, "LocalPlacementReference", Selection)
    from scripts import local_reference
    monkeypatch.setattr(local_reference, "LocalPartitionReference", Selection)
    monkeypatch.setattr(module, "WorkspaceEvaluator", lambda *args: calls.evaluators.append(args))

    def load_reference(path, *, contract, validate):
        assert validate is module.qualify_producer
        assert contract == selections[-1].contract
        calls.requalify += 1
        if options.requalify_failure:
            raise ValueError("Native artifact hash changed")
        value = json.loads(path.read_text())
        if options.mutate_record:
            path.write_text(path.read_text() + "\n")
        return value
    monkeypatch.setattr(module, "load_reference", load_reference)
    class Transport:
        def __init__(self, runtime):
            self.attempts = []
            calls.transports.append(self)
        def __call__(self, *args, **kwargs):
            self.attempts.append({"evaluator_dispatch_attempted": True, "native_flow_started": True})
            evaluation = args[3]
            evaluation.mkdir(parents=True)
            metrics = {"runs_successfully": int(options.success), "combined_score": options.score}
            artifacts = {"native_flow_execution": [{"process_started": True, "wall_seconds": 2.5}]}
            (evaluation / "evaluation_result.json").write_text(json.dumps({"metrics": metrics, "artifacts": artifacts}))
            return EvaluationResult(metrics, artifacts)
    monkeypatch.setattr(module, "DockerEvaluationTransport", Transport)
    monkeypatch.setattr(module, "initial_program_text", lambda *a: "trusted initial")
    monkeypatch.setattr(module, "materialize_program", lambda *a: {"identity": True})
    return SimpleNamespace(runtime=runtime, bundle=bundle, output=output, store=store, batch=batch,
        profile=profile, calls=calls, options=options, selections=selections)


def run(fixture, endpoint="timing"):
    return module.run_reference(fixture.runtime, endpoint=endpoint, case="bp_fe",
        benchmark_input=fixture.bundle, output=fixture.output, reference_store=fixture.store)


def test_timing_uses_ten_round_task_reference_only_and_independent_requalification(fixture):
    value = run(fixture)
    assert value["state"] == "qualified"
    assert fixture.calls.obtain == fixture.calls.publish == fixture.calls.requalify == 1
    runtime, case, objective, bundle, task = fixture.calls.evaluators[0]
    assert task.task_id == module.TIMING_TEN_TASK
    assert task.budget.global_rounds == task.budget.expensive_evaluations == 10
    assert runtime.workspace_rounds == 10 and runtime.workspace_task_digest == task.digest()
    assert runtime.reference_policy == "FRESH" and runtime.reference_record is None
    config = json.loads((fixture.output / "run_config.json").read_text())
    assert config["shared_lock_root"] == str(fixture.batch / "locks")
    assert config["candidate_budget_consumed"] == 0 and config["search_dispatch"] is False
    assert value["accounting"]["model_calls"] == value["accounting"]["candidate_finals"] == 0
    assert value["accounting"]["native_physical_attempts_known"] == 1
    assert value["accounting"]["evaluator_container_dispatches"] == 3
    assert value["accounting"]["registry_publications"] == 1
    assert (fixture.output / "authority.json").is_file()


@pytest.mark.parametrize("endpoint", ["placement_dynamic"])
def test_local_reuses_initial_program_and_producer_without_search(fixture, endpoint):
    value = run(fixture, endpoint)
    assert value["state"] == "qualified"
    assert fixture.calls.obtain == 0 and fixture.calls.publish == fixture.calls.requalify == 1
    assert fixture.runtime.reference_purpose_receipt is None
    assert value["accounting"]["native_physical_attempts_known"] == 1
    assert value["accounting"]["native_runtime_seconds"] == [2.5]
    assert json.loads((fixture.output / "reference/candidate/candidate.json").read_text()) == {"identity": True}


def test_failed_global_evaluator_never_publishes_or_claims_native_success(fixture):
    fixture.options.success = False
    value = run(fixture)
    assert value["state"] == "failed"
    assert fixture.calls.publish == fixture.calls.requalify == 0
    assert value["accounting"]["successful_reference_qualifications"] == 0
    assert value["accounting"]["native_physical_attempts"] is None
    assert value["accounting"]["native_physical_attempts_known"] == 0
    assert value["accounting"]["native_physical_attempts_unknown"] == 1
    assert not (fixture.output / "authority.json").exists()


def test_process_success_is_insufficient_if_native_requalification_fails(fixture):
    fixture.options.requalify_failure = True
    value = run(fixture)
    assert value["state"] == "failed" and "hash changed" in value["error"]
    assert value["accounting"]["registry_publications"] == 1
    assert not (fixture.output / "authority.json").exists()


def test_changed_reference_record_fails_postpublication_audit(fixture):
    fixture.options.mutate_record = True
    value = run(fixture)
    assert value["state"] == "failed" and "changed" in value["error"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, None])
def test_missing_or_invalid_score_cannot_qualify(fixture, score):
    fixture.options.score = score
    value = run(fixture)
    assert value["state"] == "failed"
    assert fixture.calls.requalify == 0


def test_existing_failed_output_is_not_retried(fixture):
    fixture.options.success = False
    run(fixture)
    before = (fixture.output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(fixture)
    assert fixture.calls.obtain == 1
    assert (fixture.output / "summary.json").read_bytes() == before


def test_output_symlink_refused_before_dispatch(fixture):
    fixture.output.parent.mkdir()
    target = fixture.batch / "existing"
    target.mkdir()
    fixture.output.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        run(fixture)
    assert fixture.calls.obtain == 0


def test_private_campaign_lock_root_refused(fixture):
    fixture.runtime.batch = fixture.batch / "campaign"
    with pytest.raises(ValueError, match="shared runs"):
        run(fixture)
    assert fixture.calls.obtain == 0


def test_case_mismatch_is_fail_closed(fixture):
    (fixture.profile / "config.yaml").write_text("design: bp\n")
    with pytest.raises(ValueError, match="selected case"):
        run(fixture)
    assert not fixture.output.exists()


@pytest.mark.parametrize("field,value", [("cpus", 32), ("memory_gb", 64), ("min_gpu_free_mb", 1024)])
def test_resources_cannot_silently_change(fixture, field, value):
    setattr(fixture.runtime, field, value)
    with pytest.raises(ValueError, match="authorized"):
        run(fixture)
    assert fixture.calls.obtain == 0


def test_store_cannot_be_inside_attempt(fixture):
    fixture.store = fixture.output / "reference-store"
    with pytest.raises(ValueError, match="disjoint"):
        run(fixture)


def test_store_symlink_refused_before_dispatch(fixture):
    target = fixture.batch / "actual-store"
    target.mkdir()
    fixture.store.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        run(fixture)
    assert fixture.calls.obtain == 0


def test_cli_uses_common_runtime_and_returns_failure_for_unqualified_output(fixture, monkeypatch):
    called = []
    def runtime(*args, **kwargs):
        called.append((args, kwargs))
        return fixture.runtime
    monkeypatch.setattr(module, "QualificationRuntime", runtime)
    fixture.options.success = False
    assert module.main(["--source", str(fixture.runtime.source), "--batch-root", str(fixture.batch),
        "--benchmark-input", str(fixture.bundle), "--output", str(fixture.output),
        "--reference-store", str(fixture.store), "--endpoint", "timing", "--case", "bp_fe",
        "--image", fixture.runtime.image, "--gpu", "2"]) == 1
    assert called[0][1] == {"cpus": 16, "memory_gb": 48, "share_gpu": True, "min_gpu_free_mb": 2048}
