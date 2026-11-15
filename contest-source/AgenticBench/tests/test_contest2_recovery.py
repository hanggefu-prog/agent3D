"""Host orchestration regressions: no real EDA or paid model requests."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from contest2.budget import BudgetLedger, BudgetExceeded
from contest2 import reconciliation
from grt_bench.common import digest, read, write
from grt_bench.queue import Queue


def ledger(root, limit=3):
    return BudgetLedger(root / "ledger.sqlite", task_id="binary_loss",
                        evaluation_limit=limit, model_cost_limit_rmb="15")


def test_rejection_is_transactional_idempotent_and_never_dispatches(tmp_path):
    budget = ledger(tmp_path, 1)
    def reject(_):
        return budget.reject_submission("bad", "a" * 64, "Function signature differs", baseline="binary_loss")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(reject, range(8)))
    assert all(r == rows[0] for r in rows)
    assert budget.snapshot()["evaluations_accepted"] == 1
    assert budget.snapshot()["evaluations_remaining"] == 0
    assert budget.pending_jobs("bad") == []
    with pytest.raises(ValueError):
        budget.reject_submission("bad", "b" * 64, "different", baseline="binary_loss")
    with pytest.raises(BudgetExceeded):
        budget.reject_submission("next", "a" * 64, "same", baseline="binary_loss")


def execution(root):
    queue = Queue(root / "control/queue.sqlite")
    queue.register_host("local", dict(cpu=2, memory_gb=2, storage_gb=2, gpus=[], build_slots=0))
    job = queue.submit({"candidate": "a" * 64, "task_digest": "task"},
                      dict(cpu=1, memory_gb=1, storage_gb=1, gpu_count=0, build_slots=0))
    claim = queue.claim("local")
    queue.cancel(job)
    owner = root / "owner.json"
    reconciliation.record_owner(owner)
    return queue, job, owner


def test_uncertainty_retains_allocation_until_exact_container_is_gone(tmp_path, monkeypatch):
    queue, job, owner = execution(tmp_path)
    state = {"Status": "running", "Running": True, "Restarting": False}
    probes = []
    def docker(argv, **options):
        probes.append(argv)
        if "ls" in argv:
            return SimpleNamespace(returncode=0, stdout='{"Names":"owned","ID":"id1"}\n{"Names":"other","ID":"id2"}\n')
        assert argv[3] == "id1"
        return SimpleNamespace(returncode=0, stdout=json.dumps(state))
    monkeypatch.setattr(reconciliation.subprocess, "run", docker)
    def inspect():
        return reconciliation.inspect_execution(queue, queue.rows(), {"owned"}, owner, tmp_path / "audit.json")
    assert not inspect()["terminated"]
    assert queue.rows()[0]["state"] == "cancelling"
    state.update(Status="exited", Running=False)
    assert inspect()["terminated"]
    assert queue.rows()[0]["state"] == "interrupted"
    assert read(tmp_path / "audit.json")["containers"][0]["id"] == "id1"


def test_docker_unavailable_or_live_original_owner_never_replays(tmp_path, monkeypatch):
    queue, job, owner = execution(tmp_path)
    monkeypatch.setattr(reconciliation.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    assert not reconciliation.inspect_execution(queue, queue.rows(), {"owned"}, owner, tmp_path / "audit.json")["terminated"]
    assert queue.rows()[0]["state"] == "cancelling"
    write(owner, {"pid": os.getpid() + 1, "identity": "alive"}, replace=True)
    monkeypatch.setattr(reconciliation, "process_identity", lambda _: "alive")
    result = reconciliation.inspect_execution(queue, queue.rows(), {"owned"}, owner, tmp_path / "audit.json")
    assert "still alive" in result["reason"] and not result["terminated"]


def test_smoke_allows_failed_model_attempts_but_only_two_successful_edits():
    from contest2.smoke import successful_turns
    state = {"turns": {"first-edit": {"status": "finished", "result": {"success": False}},
            "first-edit-retry-1": {"status": "finished", "result": {"success": True}},
            "feedback-edit": {"status": "finished", "result": {"success": True}}}}
    assert successful_turns(state) == ["first-edit-retry-1", "feedback-edit"]


def test_native_runtime_has_no_deadline_and_keeps_heartbeat(tmp_path, monkeypatch):
    from contest2.execution_policy import install_runtime
    from grt_bench import runtime
    import subprocess
    seen = []
    class Process:
        def wait(self, timeout=None):
            seen.append(timeout)
            if len(seen) == 1:
                raise subprocess.TimeoutExpired("docker", timeout)
            return 0
    fn = install_runtime()
    monkeypatch.setitem(fn.__globals__, "stop_and_confirm", lambda *a: None)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: Process())
    beat = []
    fn(["docker", "run"], tmp_path / "native", "owned", timeout=1, heartbeat=lambda: beat.append(True))
    assert seen == [15, 15] and beat

from contest2.controller import Task2Controller, TASK_IDS
from local_tasks.tp_gnn_local_task_definitions import materialize_binary_loss


def controller(root):
    c = Task2Controller.__new__(Task2Controller)
    c.root = root.resolve()
    c.baseline, c.task_id = "binary_loss", TASK_IDS["binary_loss"]
    c.ledger, c._lock, c._futures = ledger(root), threading.RLock(), {}
    c.evaluations, c.candidates = root / "evaluations", root / "candidates"
    c.evaluation_mode = "formal"
    c.matrix = SimpleNamespace(identity="matrix", cells=[{"case": "bp_fe-ng45", "seed": "1003"}])
    return c


def test_real_function_interface_rejection_is_counted_and_has_reason(tmp_path):
    c = controller(tmp_path)
    result = c.submit_evaluation("def compute_tp_gnn_binary_loss(wrong): return 0", "bad")
    assert result["status"] == "rejected" and result["reason"]
    assert c.submit_evaluation("def compute_tp_gnn_binary_loss(wrong): return 0", "bad") == result
    assert c.ledger.snapshot()["evaluations_accepted"] == 1
    assert c.ledger.pending_jobs("bad") == []


def test_failed_matrix_feedback_contains_case_seed_cause_and_log(tmp_path):
    c = controller(tmp_path)
    folder = c.evaluations / "one/cells/bp_fe-ng45/1003"
    write(folder / "cell.json", {"case": "bp_fe-ng45", "seed": "1003", "status": "failed",
        "error_type": "ValueError", "error_message": "PDN connectivity failed"})
    (folder / "result").mkdir()
    (folder / "result/stage.log").write_text("setup\nERROR: PDN disconnected on top die\n")
    result = c._failure_feedback("one")
    assert result["expected_cells"] == 1 and result["succeeded_cells"] == 0
    row = result["cells"][0]
    assert row["case"] == "bp_fe-ng45" and row["seed"] == "1003"
    assert "PDN connectivity" in row["error_message"]
    assert "PDN disconnected" in row["diagnostics"][0]["excerpt"]
    assert row["diagnostics"][0]["source"] == "stage.log"


def test_controller_reconciles_unknown_execution_then_returns_failure_details(tmp_path, monkeypatch):
    c = controller(tmp_path)
    source = (Path(__file__).resolve().parents[1] / "agent/binary_loss/program.py").read_text()
    from local_tasks.tp_gnn_local_task_definitions import validate_program
    source = validate_program(source, c.task_id)
    candidate = c._candidate_id(source)
    write(c.candidates / candidate / "candidate.json", materialize_binary_loss(source))
    c.ledger.accept_matrix("one", candidate, "b" * 64, baseline=c.baseline)
    c.transport = SimpleNamespace(queue=Queue(tmp_path / "queue.sqlite"))
    state = {"terminated": False, "reason": "Container still running"}
    monkeypatch.setattr(reconciliation, "inspect_execution", lambda *args: dict(state))
    assert c.evaluation_status("one")["status"] == "reconciling"
    state["terminated"] = True
    result = c.evaluation_status("one")
    assert result["status"] == "failed" and result["feedback"]["cells"]
    assert c.ledger.snapshot()["evaluations_accepted"] == 1
    assert c.ledger.snapshot()["dispatch_allowed"]


def test_native_policy_is_idempotent_and_keeps_reference_source_bytes():
    from contest2.execution_policy import install
    from local_bench import evaluation
    before = Path(evaluation.__file__).read_bytes()
    install()
    proxy = evaluation.subprocess
    install()
    assert evaluation.subprocess is proxy
    assert Path(evaluation.__file__).read_bytes() == before
    assert evaluation._execute_prepared._contest_unlimited


def test_failure_feedback_survives_truncated_matrix_and_cell_summaries(tmp_path):
    c = controller(tmp_path)
    folder = c.evaluations / "one/cells/bp_fe-ng45/1003"
    folder.mkdir(parents=True)
    (c.evaluations / "one/evaluation_result.json").write_text("{")
    (folder / "cell.json").write_text("{")
    feedback = c._failure_feedback("one")
    assert feedback["complete"] is False
    row = feedback["cells"][0]
    assert row["case"] == "bp_fe-ng45" and row["seed"] == "1003"
    assert "Unreadable cell summary" in row["error_message"]
