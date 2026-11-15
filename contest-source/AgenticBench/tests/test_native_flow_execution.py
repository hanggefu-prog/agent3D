"""Native-flow accounting boundaries; subprocesses are mocked, never executed."""

import json
import subprocess
from datetime import datetime
from types import SimpleNamespace

import pytest

from local_bench import evaluation


@pytest.mark.parametrize("returncode", [0, 7])
def test_receipt_preserves_dispatch_and_does_not_count_inner_tools(monkeypatch, returncode):
    command = ["python3", "-m", "open3dbench", "run", "fixed.yaml"]
    environment = {"PRESERVED": "value", "PRIVATE_TOKEN": "not-in-receipt"}
    kwargs = dict(cwd="/isolated/source", timeout=17.5, env=environment,
                  text=True, check=False, stdout=object(), stderr=subprocess.STDOUT)
    captured = []
    proc = subprocess.CompletedProcess(command, returncode)

    def run(actual, **actual_kwargs):
        captured.append((actual, actual_kwargs))
        return proc

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    ticks = iter((100.0, 103.25))
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: next(ticks))
    artifacts = {"native_flow_execution": []}
    assert evaluation._run_native_flow(
        command, invocation="stage", artifacts=artifacts, **kwargs
    ) is proc
    assert captured == [(command, kwargs)]
    receipt, = artifacts["native_flow_execution"]
    assert receipt["command"] == command and receipt["command"] is not command
    assert receipt["timeout_seconds"] == 17.5
    assert receipt["process_started"] is True
    assert receipt["returncode"] == returncode
    assert receipt["status"] == ("completed" if returncode == 0 else "failed")
    assert receipt["wall_seconds"] == 3.25
    assert datetime.fromisoformat(receipt["started_at_utc"]).utcoffset().total_seconds() == 0
    assert datetime.fromisoformat(receipt["completed_at_utc"]).utcoffset().total_seconds() == 0
    assert "env" not in receipt and "not-in-receipt" not in json.dumps(receipt)
    assert not any("openroad" in key.lower() for key in receipt)


@pytest.mark.parametrize("error,started,status", [
    (subprocess.TimeoutExpired(["flow"], 3), True, "timeout"),
    (FileNotFoundError("flow missing"), False, "launch_failed"),
    (PermissionError("not executable"), False, "launch_failed"),
    (RuntimeError("uncertain subprocess failure"), None, "uncertain_exception"),
    (KeyboardInterrupt(), None, "uncertain_exception"),
])
def test_exception_receipts_are_truthful_and_original_exception_is_preserved(
    monkeypatch, error, started, status
):
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise error

    monkeypatch.setattr(evaluation.subprocess, "run", fail)
    artifacts = {"native_flow_execution": []}
    with pytest.raises(type(error)) as caught:
        evaluation._run_native_flow(["flow"], invocation="stage", artifacts=artifacts,
                                    cwd="/isolated", timeout=3)
    assert caught.value is error
    assert len(calls) == 1  # Never retry a failed or uncertain dispatch.
    receipt, = artifacts["native_flow_execution"]
    assert receipt["process_started"] is started
    assert receipt["status"] == status
    assert receipt["returncode"] is None
    assert receipt["wall_seconds"] >= 0
    assert receipt["completed_at_utc"] is not None
    assert receipt["error_type"] == type(error).__name__


@pytest.fixture
def evaluator_fixture(tmp_path, monkeypatch):
    spec = SimpleNamespace(task_id="fixture", kind="fixture", availability="available", frozen_inputs=(),
                           upstream_stages=(), evaluation_boundary="fixture",
                           base_config=tmp_path / "base.yaml", stop_stage="routing")
    monkeypatch.setattr(evaluation, "load_task", lambda path: spec)
    monkeypatch.setattr(evaluation, "load_candidate", lambda path: {})
    monkeypatch.setattr(evaluation, "verify_candidate", lambda *args: {})
    monkeypatch.setattr(evaluation, "validate_patch_scope", lambda *args: {"paths": []})
    monkeypatch.setattr(evaluation, "_write_resolved_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "_replay_start_stage", lambda *args: "partition")
    monkeypatch.setattr(evaluation, "isolated_clone", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation, "select_evaluator_source_patch", lambda *args: None)
    monkeypatch.setattr(evaluation, "materialize_dreamplace_source_tree", lambda *args: None)
    monkeypatch.setattr(evaluation, "apply_candidate_patch", lambda *args: {})
    monkeypatch.setattr(evaluation, "validate_candidate_workspace", lambda *args: {})
    monkeypatch.setattr(evaluation, "prepare_dreamplace_overlay", lambda *args: None)
    monkeypatch.setattr(evaluation, "prepare_openroad_runtime", lambda *args: None)
    monkeypatch.setattr(evaluation, "_validate_sta_codesign_reference", lambda *args: {})
    monkeypatch.setattr(evaluation, "extract_metrics", lambda *args, **kwargs: {"combined_score": 1.0})
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="{}")

    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    return SimpleNamespace(spec=spec, calls=calls, run=fake_run, output=tmp_path / "evaluation",
                           args=(tmp_path / "task", tmp_path / "candidate", tmp_path / "source"))


@pytest.mark.parametrize("failure", ["policy", "plan", "build", "dry_run"])
def test_pre_native_outcomes_persist_empty_evidence(evaluator_fixture, monkeypatch, failure):
    fixture = evaluator_fixture

    def fail(*args, **kwargs):
        raise ValueError("fixture pre-native failure")

    if failure == "policy":
        monkeypatch.setattr(evaluation, "verify_candidate", fail)
    elif failure == "plan":
        def plan_fail(command, **kwargs):
            fixture.calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 1, stdout="fixture plan failure")
        monkeypatch.setattr(evaluation.subprocess, "run", plan_fail)
    elif failure == "build":
        monkeypatch.setattr(evaluation, "prepare_openroad_runtime", fail)
    result = evaluation.evaluate_task(*fixture.args, fixture.output, dry_run=failure == "dry_run")
    assert result.artifacts["native_flow_execution"] == []
    assert result.metrics["runs_successfully"] == 0
    assert all("plan" in command for command, _ in fixture.calls)
    saved = json.loads((fixture.output / "evaluation_result.json").read_text())
    assert saved["artifacts"]["native_flow_execution"] == []


@pytest.mark.parametrize("outcome", ["success", "nonzero", "timeout", "launch_failed", "uncertain"])
def test_stage_dispatch_evidence_survives_failed_results(evaluator_fixture, monkeypatch, outcome):
    fixture = evaluator_fixture

    def run(command, **kwargs):
        if "plan" in command:
            return fixture.run(command, **kwargs)
        fixture.calls.append((command, kwargs))
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "launch_failed":
            raise FileNotFoundError("fixture launch failure")
        if outcome == "uncertain":
            raise RuntimeError("fixture uncertain failure")
        return subprocess.CompletedProcess(command, 5 if outcome == "nonzero" else 0)

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    result = evaluation.evaluate_task(*fixture.args, fixture.output, timeout_seconds=91)
    receipt, = result.artifacts["native_flow_execution"]
    assert receipt["invocation"] == "stage"
    assert receipt["timeout_seconds"] == 91
    assert receipt["process_started"] is {"launch_failed": False, "uncertain": None}.get(outcome, True)
    assert len(fixture.calls) == 2  # One plan; exactly one native-flow invocation.
    assert result.metrics["runs_successfully"] == (1 if outcome == "success" else 0)
    saved = json.loads((fixture.output / "evaluation_result.json").read_text())
    assert saved["artifacts"]["native_flow_execution"] == [receipt]


def test_cts_snapshot_is_separate_invocation_with_existing_remaining_budget(evaluator_fixture, monkeypatch):
    from local_bench import cts_thermal

    fixture = evaluator_fixture
    fixture.spec.kind = "tp_gnn_placement_cts_thermal"
    post_command = ["python3", "-m", "fixture_thermal_snapshot"]
    monkeypatch.setattr(cts_thermal, "snapshot_command", lambda *args: post_command)
    result = evaluation.evaluate_task(*fixture.args, fixture.output, timeout_seconds=100)
    stage, thermal = result.artifacts["native_flow_execution"]
    assert [stage["invocation"], thermal["invocation"]] == ["stage", "cts_power_thermal"]
    assert stage["timeout_seconds"] == 100
    assert 0 < thermal["timeout_seconds"] <= 100
    assert thermal["command"] == post_command
    assert all(row["process_started"] is True for row in (stage, thermal))
    assert len(fixture.calls) == 3
    assert fixture.calls[1][1]["env"] == fixture.calls[2][1]["env"]
    assert fixture.calls[2][1]["timeout"] == thermal["timeout_seconds"]
    assert result.metrics["runs_successfully"] == 1


def test_cts_snapshot_timeout_preserves_both_invocation_receipts(evaluator_fixture, monkeypatch):
    from local_bench import cts_thermal

    fixture = evaluator_fixture
    fixture.spec.kind = "tp_gnn_placement_cts_thermal"
    post_command = ["python3", "-m", "fixture_thermal_snapshot"]
    monkeypatch.setattr(cts_thermal, "snapshot_command", lambda *args: post_command)

    def run(command, **kwargs):
        if command == post_command:
            fixture.calls.append((command, kwargs))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return fixture.run(command, **kwargs)

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    result = evaluation.evaluate_task(*fixture.args, fixture.output, timeout_seconds=100)
    stage, thermal = result.artifacts["native_flow_execution"]
    assert stage["status"] == "completed" and thermal["status"] == "timeout"
    assert stage["process_started"] is thermal["process_started"] is True
    assert thermal["returncode"] is None and thermal["wall_seconds"] >= 0
    assert result.metrics["runs_successfully"] == 0
    saved = json.loads((fixture.output / "evaluation_result.json").read_text())
    assert saved["artifacts"]["native_flow_execution"] == [stage, thermal]
    assert len(fixture.calls) == 3


def test_exhausted_shared_budget_does_not_create_thermal_attempt(evaluator_fixture, monkeypatch):
    from local_bench import cts_thermal

    fixture = evaluator_fixture
    fixture.spec.kind = "tp_gnn_placement_cts_thermal"
    monkeypatch.setattr(cts_thermal, "snapshot_command", lambda *args: ["fixture_thermal"])
    ticks = iter((0.0, 1.0, 2.0, 3.0, 101.0))
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: next(ticks))
    result = evaluation.evaluate_task(*fixture.args, fixture.output, timeout_seconds=100)
    stage, = result.artifacts["native_flow_execution"]
    assert stage["invocation"] == "stage" and stage["process_started"] is True
    assert result.artifacts["error_type"] == "TimeoutError"
    assert result.metrics["runs_successfully"] == 0
    assert len(fixture.calls) == 2
