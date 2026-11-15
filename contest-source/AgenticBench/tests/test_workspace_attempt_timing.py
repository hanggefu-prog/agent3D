"""Workspace timing regressions with mocked Docker commands, no EDA/model calls."""
from contextlib import contextmanager
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts import workspace_codesign_evaluator as bridge
from test_workspace_codesign_evaluator import make_runtime, prefix, repository, task


@pytest.fixture
def environment(repository, tmp_path, monkeypatch):
    runtime = make_runtime(repository, tmp_path)
    bundle, baseline, _ = prefix(runtime)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp", "timing", bundle, task())
    clock = SimpleNamespace(value=0.0, offset=1000.0)
    monkeypatch.setattr(bridge.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(bridge.time, "time", lambda: clock.offset + clock.value)
    calls = []

    def validate_input(value):
        clock.value += 2
        return {"run_dir": str(baseline)}

    @contextmanager
    def slot():
        clock.value += 4
        yield 1

    original_snapshot = evaluator._snapshot
    def snapshot(*args):
        clock.value += 3
        return original_snapshot(*args)

    def submission(output, *args):
        target = output / "submission"
        target.mkdir()
        return target

    def run(argv, **kwargs):
        assert argv[:2] == ["docker", "run"], "Only fake evaluation Docker calls are expected"
        phase = argv[argv.index("--name") + 1].rsplit("-", 1)[-1]
        calls.append((phase, argv, kwargs))
        clock.value += {"frontend": 7, "backend": 11, "verify": 3}[phase]
        output = runtime.batch / "evaluated"
        if phase == "frontend":
            bridge._json(output / "run/run.json", {"stages": {
                name: {"status": "success"} for name in bridge.FRONTEND}})
        elif phase == "verify":
            bridge._json(output / "verification/verification.json", {
                "passed": True, "metrics": {"combined_score": 1.0}})
        return SimpleNamespace(returncode=0)

    runtime.slot = slot
    monkeypatch.setattr(bridge, "source_changes", lambda *args: [])
    monkeypatch.setattr(bridge, "isolated_clone", lambda source, target: shutil.copytree(source, target))
    monkeypatch.setattr(bridge, "apply_candidate_patch", lambda *args: None)
    monkeypatch.setattr(bridge, "validate_benchmark_input", validate_input)
    monkeypatch.setattr(bridge, "validate_trusted_stage", lambda *args: None)
    monkeypatch.setattr(bridge, "validate_frontend_input_boundary", lambda *args: {})
    monkeypatch.setattr(evaluator, "_snapshot", snapshot)
    monkeypatch.setattr(evaluator, "_submission", submission)
    monkeypatch.setattr(bridge.subprocess, "run", run)
    return SimpleNamespace(runtime=runtime, evaluator=evaluator, clock=clock, calls=calls,
                           subject=repository, baseline=baseline, output=runtime.batch / "evaluated", run=run)


def evaluate(env):
    return env.evaluator.evaluate(env.subject, env.output)


def ledger(env):
    return bridge._read(env.output.parent / "evaluated.workspace-attempt.json")


def test_full_evaluation_separates_admission_preparation_flow_and_verifier(environment):
    env = environment
    original_manifest = (env.baseline / "run.json").read_bytes()
    result = evaluate(env)
    assert result["success"] and result["metrics"] == {"combined_score": 1.0}
    assert result["timing"] == {
        "preparation_wall_seconds": 5, "admission_wait_seconds": 4,
        "native_flow_command_wall_seconds": 18, "runtime_overlay_command_wall_seconds": None,
        "verifier_command_wall_seconds": 3, "timeout_cleanup_wall_seconds": None,
        "total_wall_seconds": 32}
    assert result["runtime_seconds"] == 32
    assert "admission" in result["runtime_seconds_scope"]
    execution = result["execution"]
    assert execution["command_dispatches"] == execution["command_processes_started"] == 3
    assert execution["native_flow_command_dispatches"] == 2
    assert execution["native_physical_attempts"] is None
    assert execution["uncertain_command_process_starts"] == 0
    record = ledger(env)
    assert record["started_at"] == 1000 and record["queued_at"] == 1005
    assert record["admitted_at"] == 1009 and record["finished_at"] == 1032
    assert record["subject"] == str(env.subject) and record["gpu"] == 1
    assert record["reference_run"] is None
    assert record["execution"] == execution
    receipt = bridge._read(env.output / "frontend-command-receipt.json")
    assert receipt["dispatch_attempted"] is True and receipt["process_started"] is True
    assert receipt["native_execution_confirmed"] is None
    assert receipt["process_wall_seconds"] == 7
    events = [json.loads(row) for row in (env.output.parent / "evaluated.workspace-attempt.events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events][:4] == ["started", "preparation-completed", "queued", "admitted"]
    assert events[-1]["event"] == "completed"
    assert all(row["at_utc"].endswith("+00:00") for row in events)
    assert (env.baseline / "run.json").read_bytes() == original_manifest
    # Native stage budgets and resources are unchanged. The one shared deadline
    # decreases across commands rather than restarting for each phase.
    timeouts = [row[2]["timeout"] for row in env.calls]
    assert timeouts == [env.evaluator.task.environment.walltime_seconds,
                        env.evaluator.task.environment.walltime_seconds - 7,
                        env.evaluator.task.environment.walltime_seconds - 18]


def test_prevalidation_failure_has_no_command_or_fabricated_native_time(environment, monkeypatch):
    env = environment
    def invalid(*args):
        env.clock.value += 2
        raise ValueError("Bad frozen input")
    monkeypatch.setattr(bridge, "validate_benchmark_input", invalid)
    result = evaluate(env)
    assert result["success"] is False and result["metrics"] == {}
    assert result["timing"]["preparation_wall_seconds"] == 2
    assert result["timing"]["admission_wait_seconds"] is None
    assert result["timing"]["native_flow_command_wall_seconds"] is None
    assert result["execution"]["command_dispatches"] == 0
    assert ledger(env)["failed_phase"] == "preparation"
    assert ledger(env)["started_at"] == 1000 and ledger(env)["subject"] == str(env.subject)
    assert env.calls == []


def test_admission_failure_keeps_preparation_and_queue_evidence(environment):
    env = environment
    @contextmanager
    def slot():
        env.clock.value += 4
        raise RuntimeError("Admission failed")
        yield  # pragma: no cover
    env.runtime.slot = slot
    result = evaluate(env)
    assert result["timing"]["preparation_wall_seconds"] == 5
    assert result["timing"]["admission_wait_seconds"] == 4
    assert result["execution"]["command_dispatches"] == 0
    record = ledger(env)
    assert record["failed_phase"] == "admission"
    assert record["queued_at"] == 1005
    assert "admitted_at" not in record


def test_command_construction_failure_does_not_count_as_dispatch(environment, monkeypatch):
    env = environment
    def fail(*args, **kwargs):
        raise ValueError("Invalid container mount")
    monkeypatch.setattr(env.evaluator, "_docker", fail)
    result = evaluate(env)
    assert result["success"] is False
    assert result["execution"]["command_dispatches"] == 0
    assert result["timing"]["native_flow_command_wall_seconds"] is None
    assert ledger(env)["gpu"] == 1
    assert env.calls == []


def test_uncertain_process_failure_keeps_completed_phase_and_null_start(environment, monkeypatch):
    env = environment
    def run(argv, **kwargs):
        if argv[argv.index("--name") + 1].endswith("-backend"):
            env.clock.value += 6
            raise OSError("Client disconnected")
        return env.run(argv, **kwargs)
    monkeypatch.setattr(bridge.subprocess, "run", run)
    result = evaluate(env)
    assert result["success"] is False and result["metrics"] == {}
    execution = result["execution"]
    assert execution["native_flow_command_dispatches"] == 2
    assert execution["command_processes_started"] == 1
    assert execution["uncertain_command_process_starts"] == 1
    assert execution["command_phases"]["backend"]["process_started"] is None
    assert result["timing"]["native_flow_command_wall_seconds"] == 13
    assert result["timing"]["verifier_command_wall_seconds"] is None
    assert ledger(env)["started_at"] == 1000 and ledger(env)["gpu"] == 1


def test_failed_attempt_is_not_retried_or_overwritten(environment, monkeypatch):
    env = environment
    monkeypatch.setattr(bridge, "validate_benchmark_input", lambda *args: (_ for _ in ()).throw(ValueError("Invalid")))
    evaluate(env)
    paths = [env.output.parent / "evaluated.workspace-attempt.json",
             env.output.parent / "evaluated.workspace-attempt.events.jsonl", env.output / "feedback.json"]
    contents = [path.read_bytes() for path in paths]
    with pytest.raises(FileExistsError):
        evaluate(env)
    assert [path.read_bytes() for path in paths] == contents


@pytest.mark.parametrize("exit_code", [0, 1])
def test_direct_command_receipt_distinguishes_exit_from_process_start(tmp_path, monkeypatch, exit_code):
    evaluator = object.__new__(bridge.WorkspaceEvaluator)
    monkeypatch.setattr(bridge.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=exit_code))
    argv = ["docker", "run", "--name", "test-owned", "image"]
    if exit_code:
        with pytest.raises(RuntimeError, match="frontend failed"):
            evaluator._call("frontend", argv, tmp_path, bridge.time.monotonic() + 100)
    else:
        evaluator._call("frontend", argv, tmp_path, bridge.time.monotonic() + 100)
    path = tmp_path / "frontend-command-receipt.json"
    receipt = bridge._read(path)
    assert receipt["process_started"] is True
    assert receipt["native_execution_confirmed"] is None
    assert receipt["exit_code"] == exit_code
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        evaluator._call("frontend", argv, tmp_path, bridge.time.monotonic() + 100)
    assert path.read_bytes() == saved


def test_timeout_preserves_process_wall_time_and_separate_owned_cleanup(tmp_path, monkeypatch):
    evaluator = object.__new__(bridge.WorkspaceEvaluator)
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(bridge.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(bridge.time, "time", lambda: 1000 + clock.value)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1] == "run":
            clock.value += 7
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        clock.value += 2
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(bridge.subprocess, "run", run)
    argv = ["docker", "run", "--name", "test-owned", "image"]
    with pytest.raises(RuntimeError, match="exceeded the evaluation walltime"):
        evaluator._call("frontend", argv, tmp_path, 100)
    receipt = bridge._read(tmp_path / "frontend-command-receipt.json")
    assert receipt["process_started"] is None
    assert receipt["process_wall_seconds"] == 7
    assert receipt["cleanup_wall_seconds"] == 2
    assert receipt["cleanup_exit_code"] == 0
    assert calls[1][0] == ["docker", "stop", "--time", "5", "test-owned"]
    assert calls[1][1]["timeout"] == 20


def test_total_and_command_times_are_monotonic_when_wall_clock_changes(environment, monkeypatch):
    env = environment
    def run(*args, **kwargs):
        result = env.run(*args, **kwargs)
        env.clock.offset -= 100
        return result
    monkeypatch.setattr(bridge.subprocess, "run", run)
    result = evaluate(env)
    assert result["timing"]["native_flow_command_wall_seconds"] == 18
    assert result["timing"]["total_wall_seconds"] == 32
    assert ledger(env)["finished_at"] < ledger(env)["started_at"]
