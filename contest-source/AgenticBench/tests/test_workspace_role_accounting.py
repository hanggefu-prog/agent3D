"""Independent role/evaluator accounting with mocked agents, Docker and EDA."""
import pytest

from scripts import run_workspace_agentic3d as driver
from test_run_workspace_agentic3d import harness, read, run


def test_role_counts_are_invocations_not_api_requests_and_missing_evidence_is_unknown(harness):
    summary = run(harness)
    counts = summary["role_turn_accounting"]
    assert counts["attempted"] == counts["completed"] == 25
    assert counts["failed"] == 0
    assert counts["generated_candidates"] == summary["generated_candidates"] == 5
    assert counts["api_request_count"] is None and counts["api_latency_seconds"] is None
    assert "tool wait" in counts["wall_time_scope"]
    assert all(row["attempted"] == row["completed"] == 5 for row in counts["by_phase"].values())
    assert all(row["wall_seconds"] >= 0 for row in counts["by_phase"].values())
    assert all(row["invocation_attempted"] is True for row in summary["role_turns"])
    assert read(harness, "role_turns.json") == summary["role_turns"]
    # Existing mocks intentionally have no evaluator execution receipt. These
    # are unknown native counts, not successful-looking zero-cost evaluations.
    for category, expected in (("reference_execution", 1), ("candidate_execution", 5)):
        result = summary[category]
        assert result["evaluator_invocations"] == result["feedback_records"] == expected
        assert result["native_flow_command_dispatches"] is None
        assert result["native_physical_attempts"] is None
        assert result["receipt_coverage"]["native_flow_command_dispatches"] == {"known": 0, "expected": expected}
    assert summary["trusted_measured_improvements"] == summary["role_accepted_improvements"] == 2


@pytest.mark.parametrize("option", ["start_error", "isolation_ok"])
def test_container_or_probe_failure_is_not_a_role_invocation(harness, option):
    setattr(harness.options, option, option == "start_error")
    with pytest.raises(Exception):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["role_turn_accounting"]["attempted"] == 0
    assert summary["generated_candidates"] == summary["final_evaluations"] == 0


@pytest.mark.parametrize("option, value", [("plan", False), ("fail_phase", "task")])
def test_task_validation_or_model_failure_does_not_generate_a_candidate(harness, option, value):
    setattr(harness.options, option, value)
    with pytest.raises(RuntimeError):
        run(harness)
    summary = read(harness, "summary.json")
    counts = summary["role_turn_accounting"]
    assert counts["attempted"] == 3 and counts["completed"] == 2 and counts["failed"] == 1
    assert counts["by_phase"]["task"]["attempted"] == counts["by_phase"]["task"]["failed"] == 1
    assert summary["generated_candidates"] == summary["final_evaluations"] == 0
    assert summary["candidate_execution"]["evaluator_invocations"] == 0


def test_client_exception_without_result_is_one_failed_role_invocation(harness, monkeypatch):
    def failed_turn(**kwargs):
        raise OSError("Client failed before result receipt")
    monkeypatch.setattr(driver, "run_workspace_turn", failed_turn)
    with pytest.raises(OSError):
        run(harness)
    summary = read(harness, "summary.json")
    counts = summary["role_turn_accounting"]
    assert counts["attempted"] == counts["failed"] == 1 and counts["completed"] == 0
    assert counts["by_phase"]["global-plan"]["wall_seconds"] >= 0
    assert summary["role_turns"][0]["api_request_count"] is None


def test_evaluator_exception_preserves_generated_candidate_and_consumed_final(harness):
    harness.options.evaluator_error = OSError("No space left on device")
    with pytest.raises(OSError):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["generated_candidates"] == summary["final_evaluations"] == 1
    assert summary["role_turn_accounting"]["completed"] == 3
    assert summary["candidate_execution"]["evaluator_invocations"] == 1
    assert summary["candidate_execution"]["native_flow_command_dispatches"] is None
    assert summary["failed_final_evaluations"] == 1


def test_reference_failure_does_not_imply_role_calls_or_candidate_evaluations(harness):
    harness.options.reference_ok = False
    with pytest.raises(RuntimeError):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["reference_execution"]["evaluator_invocations"] == 1
    assert summary["reference_execution"]["native_physical_attempts"] is None
    assert summary["candidate_execution"]["evaluator_invocations"] == 0
    assert summary["role_turn_accounting"]["attempted"] == summary["generated_candidates"] == 0


def test_verified_improvement_and_role_accepted_incumbent_are_counted_separately(harness):
    harness.options.post_decision = "reject"
    summary = run(harness)
    assert summary["trusted_measured_improvements"] == 2
    assert summary["role_accepted_improvements"] == 0
    assert summary["best"]["round"] == 4 and summary["incumbent"]["round"] == 0


def test_known_command_evidence_is_aggregated_separately_for_reference_and_finals(harness, monkeypatch):
    previous = driver.WorkspaceEvaluator
    class Evaluator(previous):
        def evaluate(self, subject, output, reference):
            result = super().evaluate(subject, output, reference)
            short = output.name == "round-02"
            result["execution"] = {
                "command_dispatches": 1 if short else 3,
                "native_flow_command_dispatches": 1 if short else 2,
                "command_processes_started": 1 if short else 3,
                "uncertain_command_process_starts": 0,
                "native_physical_attempts": None}
            return result
    monkeypatch.setattr(driver, "WorkspaceEvaluator", Evaluator)
    summary = run(harness)
    assert summary["reference_execution"]["command_dispatches"] == 3
    assert summary["reference_execution"]["native_flow_command_dispatches"] == 2
    assert summary["candidate_execution"]["command_dispatches"] == 13
    assert summary["candidate_execution"]["native_flow_command_dispatches"] == 9
    assert summary["candidate_execution"]["receipt_coverage"]["command_dispatches"] == {"known": 5, "expected": 5}
    assert summary["candidate_execution"]["native_physical_attempts"] is None


def test_partial_or_invalid_receipts_do_not_become_complete_totals():
    rows = [{"execution": {"command_dispatches": 3, "native_flow_command_dispatches": 2}},
            {"execution": {"command_dispatches": True, "native_flow_command_dispatches": -1}}]
    result = driver._execution_accounting(rows, 2)
    assert result["command_dispatches"] is None
    assert result["native_flow_command_dispatches"] is None
    assert result["known_subtotals"]["command_dispatches"] == 3
    assert result["receipt_coverage"]["command_dispatches"] == {"known": 1, "expected": 2}


def test_failed_receipt_write_before_invocation_is_not_counted(harness, monkeypatch):
    original = driver.write_json
    failed = False
    def write(path, value):
        nonlocal failed
        if path.name == "role_turns.json" and not failed:
            failed = True
            raise OSError("Cannot retain invocation intent")
        return original(path, value)
    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(OSError):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["role_turn_accounting"]["attempted"] == 0
    assert not harness.calls.turns
