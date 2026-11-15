"""Execution evidence tests; no subprocess, model, native EDA or Docker calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_local_task_smoke as smoke
from test_local_task_smoke import runtime, TASK, SPEC, FakeProvider, FakeTransport, invoke
from local_bench.model import load_candidate


def test_legacy_counter_does_not_prove_native_launch(tmp_path):
    counts = smoke.execution_accounting(SimpleNamespace(physical_calls=6), tmp_path / "ref")
    assert counts["candidate_physical_attempts"] is None
    assert counts["reference_physical_attempts"] is None
    assert counts["execution_accounting_available"] is False


def test_separate_reference_dispatch_physical_and_uncertainty(tmp_path):
    ref = tmp_path / "ref"
    transport = SimpleNamespace(attempts=[
        {"output_dir": str(ref), "evaluator_dispatch_attempted": True, "native_flow_started": True},
        {"output_dir": str(tmp_path / "build-failure"), "evaluator_dispatch_attempted": False,
         "native_flow_started": False},
        {"output_dir": str(tmp_path / "native-failure"), "evaluator_dispatch_attempted": True,
         "native_flow_started": True},
        {"output_dir": str(tmp_path / "no-result"), "evaluator_dispatch_attempted": True,
         "native_flow_started": None},
    ])
    counts = smoke.execution_accounting(transport, ref)
    assert counts["reference_physical_attempts"] == 1
    assert counts["candidate_evaluator_dispatches"] == 2
    assert counts["candidate_physical_attempts"] is None
    assert counts["candidate_physical_attempts_known"] == 1
    assert counts["candidate_physical_attempts_unknown"] == 1


@pytest.mark.parametrize("states,expected", [([], False), ([True], True),
    ([False], False), ([None], None), ([True, None], True)])
def test_result_native_evidence_controls_candidate_count(runtime, states, expected):
    candidate = runtime.batch / "one/candidate"
    candidate.mkdir(parents=True)
    smoke.write_json(candidate / "candidate.json", load_candidate(SPEC.initial_candidate))
    def execute(command, output, log, contract):
        runtime.last_execution = {"evaluator_dispatch_attempted": True, "evaluator_process_started": True}
        output.mkdir()
        smoke.write_json(output / "evaluation_result.json", {
            "metrics": {"runs_successfully": 0, "combined_score": 0},
            "artifacts": {"native_flow_execution": [{"process_started": value} for value in states]}})
        return 1
    runtime.execute = execute
    transport = smoke.DockerEvaluationTransport(runtime)
    transport(TASK, candidate, runtime.source, candidate.parent / "result")
    assert transport.attempts[0]["native_flow_started"] is expected
    assert transport.physical_calls == (1 if expected is True else 0)


def test_build_failure_has_zero_dispatch_but_is_an_attempt(runtime):
    candidate = runtime.batch / "one/candidate"
    candidate.mkdir(parents=True)
    smoke.write_json(candidate / "candidate.json", load_candidate(SPEC.initial_candidate))
    def execute(*args):
        runtime.last_execution = {"evaluator_dispatch_attempted": False,
            "evaluator_process_started": False, "preparation_wall_seconds": 3.5}
        raise RuntimeError("compiler failed")
    runtime.execute = execute
    transport = smoke.DockerEvaluationTransport(runtime)
    with pytest.raises(smoke.EvaluationInfrastructureError, match="compiler failed"):
        transport(TASK, candidate, runtime.source, candidate.parent / "result")
    assert len(transport.attempts) == 1
    assert transport.physical_calls == 0
    assert transport.attempts[0]["native_flow_started"] is False
    assert transport.attempts[0]["preparation_wall_seconds"] == 3.5


def test_failed_old_attempt_is_not_rewritten(runtime):
    local = object.__new__(smoke.LocalSmokeRuntime)
    local.__dict__.update(runtime.__dict__)
    candidate = runtime.batch / "one/candidate"
    candidate.mkdir(parents=True)
    smoke.write_json(candidate / "candidate.json", load_candidate(SPEC.initial_candidate))
    ledger = candidate.parent / "result.attempt.json"
    before = '{"state":"infrastructure_failed","original_evidence":true}\n'
    ledger.write_text(before)
    local.last_execution = {"evaluator_dispatch_attempted": True}  # old evidence must be cleared
    transport = smoke.DockerEvaluationTransport(local)
    with pytest.raises(smoke.EvaluationInfrastructureError, match="silently retry"):
        transport(TASK, candidate, runtime.source, candidate.parent / "result")
    assert ledger.read_text() == before
    assert transport.attempts[0]["evaluator_dispatch_attempted"] is False
    assert transport.physical_calls == 0


def test_shared_cache_is_opt_in_and_recorded(runtime):
    summary = invoke(runtime)
    config = json.loads((runtime.batch / "smoke/run_config.json").read_text())
    assert config["compiler_cache"]["enabled"] is False
    assert summary["api_latency_seconds"] is None


@pytest.mark.parametrize("size", [0, 65, True, 1.5])
def test_compiler_cache_size_is_checked_before_output(runtime, size):
    with pytest.raises(ValueError, match="compiler_cache_gb"):
        invoke(runtime, compiler_cache_gb=size)
    assert not (runtime.batch / "smoke").exists()


def test_non_native_task_rejects_compiler_cache_flag(runtime):
    with pytest.raises(ValueError, match="routing-source"):
        invoke(runtime, compiler_cache_dir=runtime.batch / "compiler-cache")
    assert not (runtime.batch / "smoke").exists()
