"""Attempt accounting only: all admission, preparation and subprocess work is mocked."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from scripts import codesign_runtime as module


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    clock = SimpleNamespace(value=0.0, wall_offset=1000.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(module.time, "time", lambda: clock.wall_offset + clock.value)
    runtime = object.__new__(module.Runtime)
    runtime.batch = tmp_path
    runtime.gpus = [2]
    runtime.share_gpu = True
    runtime.min_gpu_free_mb = 2048
    seen = []

    @contextmanager
    def slot():
        clock.value += 3
        yield 2

    def docker(command, gpu, scratch):
        seen.append("prepare")
        clock.value += 5
        return ["mock-evaluator", *command]

    def run(argv, **kwargs):
        seen.append("dispatch")
        clock.value += 7
        return SimpleNamespace(returncode=0)

    runtime.slot, runtime.docker = slot, docker
    monkeypatch.setattr(module.subprocess, "run", run)
    return SimpleNamespace(runtime=runtime, clock=clock, seen=seen,
                           output=tmp_path / "round-01/result", log=tmp_path / "round-01/evaluation.log")


def execute(fixture):
    return fixture.runtime.execute(["evaluate"], fixture.output, fixture.log, {"kind": "test"})


def artifacts(fixture):
    parent = fixture.output.parent
    ledger = json.loads((parent / "result.attempt.json").read_text())
    events = [json.loads(line) for line in (parent / "result.attempt.events.jsonl").read_text().splitlines()]
    return ledger, events


def test_preserves_phase_timestamps_and_separates_admission_preparation_and_dispatch(fixture):
    assert execute(fixture) == 0
    ledger, events = artifacts(fixture)
    assert ledger["state"] == "exited"
    assert ledger["queued_at"] == 1000
    assert ledger["started_at"] == ledger["admitted_at"] == 1003
    assert ledger["evaluator_started_at"] == 1008
    assert ledger["finished_at"] == 1015
    assert ledger["admission_wait_seconds"] == 3
    assert ledger["preparation_wall_seconds"] == 5
    assert ledger["evaluator_wall_seconds"] == 7
    assert ledger["total_wall_seconds"] == 15
    assert ledger["gpu"] == 2
    assert ledger["scratch"].endswith("/round-01/result.scratch")
    assert ledger["evaluator_dispatch_attempted"] is True
    assert ledger["evaluator_process_started"] is True
    assert "native_physical_started" not in ledger
    assert [row["event"] for row in events] == ["queued", "admitted", "preparation-started",
        "preparation-completed", "evaluator-dispatch", "evaluator-exited", "exited"]
    assert all(row["at_utc"].endswith("+00:00") for row in events)
    assert [row["elapsed_seconds"] for row in events] == [0, 3, 3, 8, 8, 15, 15]
    assert fixture.runtime.last_execution["total_wall_seconds"] == 15


def test_build_preparation_failure_is_not_evaluator_dispatch(fixture):
    def failed_build(*args):
        fixture.clock.value += 5
        raise RuntimeError("Compiler failed")
    fixture.runtime.docker = failed_build
    with pytest.raises(RuntimeError, match="Compiler failed"):
        execute(fixture)
    ledger, events = artifacts(fixture)
    assert ledger["state"] == "infrastructure_failed"
    assert ledger["failed_phase"] == "preparation"
    assert ledger["queued_at"] == 1000 and ledger["started_at"] == 1003
    assert ledger["gpu"] == 2 and "scratch" in ledger
    assert ledger["preparation_wall_seconds"] == 5
    assert ledger["evaluator_wall_seconds"] is None
    assert ledger["evaluator_dispatch_attempted"] is False
    assert ledger["evaluator_process_started"] is False
    assert "evaluator_started_at" not in ledger
    assert not any(row["event"] == "evaluator-dispatch" for row in events)
    assert fixture.seen == []


def test_admission_failure_keeps_queued_timestamp_and_no_dispatch(fixture):
    @contextmanager
    def failed_slot():
        fixture.clock.value += 4
        raise RuntimeError("Admission check failed")
        yield  # pragma: no cover
    fixture.runtime.slot = failed_slot
    with pytest.raises(RuntimeError, match="Admission"):
        execute(fixture)
    ledger, _ = artifacts(fixture)
    assert ledger["failed_phase"] == "admission"
    assert ledger["queued_at"] == 1000
    assert ledger["admission_wait_seconds"] == ledger["total_wall_seconds"] == 4
    assert ledger["preparation_wall_seconds"] is None
    assert ledger["evaluator_dispatch_attempted"] is False
    assert "started_at" not in ledger


def test_uncertain_dispatch_failure_is_not_a_claim_of_native_execution(fixture, monkeypatch):
    def failed_run(*args, **kwargs):
        fixture.clock.value += 2
        raise OSError("Evaluator client failure")
    monkeypatch.setattr(module.subprocess, "run", failed_run)
    with pytest.raises(OSError, match="Evaluator client"):
        execute(fixture)
    ledger, _ = artifacts(fixture)
    assert ledger["failed_phase"] == "evaluator"
    assert ledger["evaluator_dispatch_attempted"] is True
    assert ledger["evaluator_process_started"] is None
    assert ledger["evaluator_wall_seconds"] == 2
    assert ledger["preparation_wall_seconds"] == 5


def test_nonzero_evaluator_exit_is_retained_without_infrastructure_reclassification(fixture, monkeypatch):
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    assert execute(fixture) == 1
    ledger, _ = artifacts(fixture)
    assert ledger["state"] == "exited" and ledger["exit_code"] == 1
    assert ledger["evaluator_process_started"] is True


def test_elapsed_durations_use_monotonic_clock_not_wall_clock(fixture, monkeypatch):
    def run(*args, **kwargs):
        fixture.clock.value += 7
        fixture.clock.wall_offset -= 100
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", run)
    execute(fixture)
    ledger, _ = artifacts(fixture)
    assert ledger["finished_at"] < ledger["evaluator_started_at"]
    assert ledger["evaluator_wall_seconds"] == 7
    assert ledger["total_wall_seconds"] == 15


@pytest.mark.parametrize("existing", ["ledger", "events", "output"])
def test_existing_attempt_is_never_retried_or_overwritten(fixture, existing):
    parent = fixture.output.parent
    parent.mkdir(parents=True)
    path = {"ledger": parent / "result.attempt.json", "events": parent / "result.attempt.events.jsonl",
            "output": fixture.output}[existing]
    if existing == "output":
        path.mkdir()
    else:
        path.write_text("preserved evidence\n")
    with pytest.raises(RuntimeError, match="will not silently retry"):
        execute(fixture)
    assert fixture.seen == []
    if existing != "output":
        assert path.read_text() == "preserved evidence\n"


def test_failed_attempt_and_event_history_cannot_be_retried(fixture):
    fixture.runtime.docker = lambda *args: (_ for _ in ()).throw(RuntimeError("Build failed"))
    with pytest.raises(RuntimeError, match="Build failed"):
        execute(fixture)
    parent = fixture.output.parent
    paths = [parent / "result.attempt.json", parent / "result.attempt.events.jsonl"]
    preserved = [path.read_bytes() for path in paths]
    with pytest.raises(RuntimeError, match="will not silently retry"):
        execute(fixture)
    assert [path.read_bytes() for path in paths] == preserved
    assert fixture.runtime.last_execution is None
