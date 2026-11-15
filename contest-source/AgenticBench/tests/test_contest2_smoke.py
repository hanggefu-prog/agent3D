"""Automatic single-cell smoke mode cannot become a formal score."""
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from contest2.budget import BudgetLedger, BudgetConflict, BudgetExceeded
from contest2 import provider_preflight, runner
from contest2.sampling import smoke_matrix, validate_public_matrix, SMOKE_CELL
from grt_bench.common import read
from local_bench.matrix import LocalEvaluationMatrix
from test_contest2_sampling import matrix_fixture


def ledger(path, mode="formal"):
    return BudgetLedger(path, task_id="binary_loss", evaluation_limit=10,
                        model_cost_limit_rmb="15", allowed_model="deepseek-flash",
                        evaluation_mode=mode)


def test_smoke_accepts_one_cell_then_allows_feedback_model_turn(tmp_path):
    value = ledger(tmp_path / "budget.sqlite", "smoke")
    value.accept_matrix("one", "first", "a" * 64, baseline="binary_loss")
    value.finish_evaluation("one", "succeeded")
    jobs = value.pending_jobs("one")
    assert [(r["case_id"], str(r["seed"])) for r in jobs] == [SMOKE_CELL]
    assert value.snapshot()["evaluations_remaining"] == 0
    value.reserve_model_call("feedback", "deepseek-flash", "0.10")
    assert value.snapshot()["dispatch_allowed"]
    with pytest.raises(BudgetExceeded, match="exactly one"):
        value.accept_matrix("two", "next", "b" * 64, baseline="binary_loss")
    with pytest.raises(ValueError, match="formal validation"):
        value.freeze("first", "c" * 64)
    with pytest.raises(ValueError, match="fixed single-cell"):
        value.accept_evaluation("generic", "binary_loss", "first")


@pytest.mark.parametrize("original,next_mode", [("formal", "smoke"), ("smoke", "formal")])
def test_work_ledger_mode_is_immutable(tmp_path, original, next_mode):
    path = tmp_path / "budget.sqlite"
    ledger(path, original)
    with pytest.raises(BudgetConflict, match="evaluation_mode"):
        ledger(path, next_mode)


def test_existing_unmarked_formal_ledger_cannot_be_adopted_as_smoke(tmp_path):
    path = tmp_path / "budget.sqlite"
    original = ledger(path)
    original.accept_matrix("old", "candidate", "a" * 64, baseline="binary_loss")
    with original._transaction() as db:
        db.execute("DELETE FROM settings WHERE key='evaluation_mode'")
    with pytest.raises(BudgetConflict, match="evaluation_mode"):
        ledger(path, "smoke")
    assert ledger(path).snapshot()["evaluations_accepted"] == 1


def test_smoke_selection_requires_complete_official_training_input(tmp_path):
    path, spec = matrix_fixture(tmp_path)
    original = LocalEvaluationMatrix(path, spec)
    selected = smoke_matrix(original, "binary_loss", tmp_path / "control/smoke.json")
    assert [(r["case"], r["seed"]) for r in selected.cells] == [SMOKE_CELL]
    assert len(original.cells) == 7
    with pytest.raises(ValueError, match="policy differs"):
        validate_public_matrix(selected, "binary_loss", "train")
    with pytest.raises(ValueError, match="policy differs"):
        smoke_matrix(selected, "binary_loss", tmp_path / "other.json")


def test_smoke_cli_uses_normal_data_defaults_and_new_work_directory(tmp_path, monkeypatch):
    source = tmp_path / "contest-source"
    source.mkdir()
    captured = []
    monkeypatch.setattr(runner, "run", lambda args: captured.append(args) or 0)
    args = ["smoke", "--baseline", "binary_loss", "--source", str(source)]
    assert runner.main(args) == runner.main(args) == 0
    first, second = captured
    assert first.matrix == tmp_path / "contest-data/task2/binary_loss/train/qualified_matrix.json"
    assert first.work_root.parent == tmp_path / "contest-work/smoke"
    assert first.work_root != second.work_root
    assert not first.check


def test_formal_cli_keeps_baseline_work_and_budget_directories_separate(tmp_path, monkeypatch):
    source = tmp_path / "contest-source"
    source.mkdir()
    captured = []
    monkeypatch.setattr(runner, "run", lambda args: captured.append(args) or 0)
    for baseline in ("binary_loss", "loss_decoder"):
        runner.main(["run", "--baseline", baseline, "--source", str(source)])
    assert [args.work_root for args in captured] == [
        tmp_path / "contest-work/task2/binary_loss", tmp_path / "contest-work/task2/loss_decoder"]


def test_key_file_is_not_accepted_inside_readable_agent_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    key = source / "private.key"
    key.write_text("test-provider-secret")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    args = SimpleNamespace(api_key_file=key, data_root=tmp_path / "data")
    with pytest.raises(ValueError, match="outside source"):
        runner._provider_key(args, source, source / "AgenticBench", tmp_path / "work")


@pytest.mark.parametrize("error", [TimeoutError(), urllib.error.HTTPError(
    "https://api.deepseek.com/models", 401, "denied", {}, None)])
def test_provider_failure_is_bounded_and_does_not_leak_the_key(tmp_path, monkeypatch, error):
    def fail(request, **kwargs):
        assert kwargs["timeout"] == 15
        raise error
    monkeypatch.setattr(provider_preflight.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="no model request") as caught:
        provider_preflight.check_provider("private-test-secret", tmp_path)
    evidence = read(tmp_path / "control/provider-preflight.json")
    assert evidence["passed"] is False and evidence["model_requests"] == 0
    assert "private-test-secret" not in str(caught.value) + json.dumps(evidence)


def test_provider_checks_model_catalog_without_a_generation_request(tmp_path, monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, count): return b'{"data":[{"id":"deepseek-flash"}]}'
    def query(request, **kwargs):
        assert request.get_method() == "GET"
        assert request.full_url == "https://api.deepseek.com/models"
        return Response()
    monkeypatch.setattr(provider_preflight.urllib.request, "urlopen", query)
    assert provider_preflight.check_provider("private-test-secret", tmp_path)["passed"]


@pytest.mark.parametrize("failure", [None, "provider", "agent"])
def test_normal_runner_cleans_services_and_records_success_or_failure(tmp_path, monkeypatch, failure):
    source = tmp_path / "source"
    source.mkdir()
    root = tmp_path / "work"
    events = []
    controllers = []
    class Controller:
        def __init__(self, **kwargs):
            self.reading_directory = root / "agent-evaluations"
            self.ledger = kwargs["ledger"]
            self.matrix = SimpleNamespace(cases=["bp_fe-ng45"],
                cells=[{"case": "bp_fe-ng45", "baseline_run": str(tmp_path / "input")}])
            controllers.append(self)
        def close(self): events.append("controller-closed")
    class Bridge:
        def __init__(self, controller): self.directory = tmp_path / "bridge"
        def __enter__(self): events.append("bridge-started"); return self
        def __exit__(self, *args): events.append("bridge-closed")
    class Gateway:
        def __init__(self, *args, **kwargs): events.append("gateway-started")
        def terminate(self): events.append("gateway-stopped")
        def wait(self, **kwargs): return 0
    monkeypatch.setattr(runner, "Task2Controller", Controller)
    monkeypatch.setattr(runner, "AgentBridge", Bridge)
    monkeypatch.setattr(runner.subprocess, "Popen", Gateway)
    monkeypatch.setattr(runner, "_gateway_ready", lambda *args: None)
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/provided/codex")
    monkeypatch.setattr(runner, "_runtime", lambda *a, **k: (None, {"image": "frozen", "capacity": {"gpus": ["0"]}}))
    monkeypatch.setattr(runner, "gpu_preflight", lambda *args: None)
    monkeypatch.setattr(runner, "_provider_key", lambda *args: "test-secret")
    def preflight(*args):
        if failure == "provider": raise RuntimeError("provider unavailable")
    monkeypatch.setattr(runner, "check_provider", preflight)
    monkeypatch.setattr(runner, "command", lambda **kwargs: ["docker", "run"])
    def agent(**kwargs):
        if failure == "agent": raise RuntimeError("agent failed")
        budget = controllers[0].ledger
        budget.accept_matrix("one", "candidate", "a" * 64, baseline="binary_loss")
        budget.finish_evaluation("one", "succeeded")
        return "candidate"
    monkeypatch.setattr(runner, "run_agent", agent)
    args = SimpleNamespace(source=source, work_root=root, command="run", runtime=tmp_path / "runtime.json",
        baseline="binary_loss", matrix=tmp_path / "matrix.json", check=False,
        agent_cpu=4, agent_memory_gb=32, agent_gpu=None)
    if failure:
        with pytest.raises(RuntimeError): runner.run(args)
    else:
        assert runner.run(args) == 0
    result = read(root / "session-result.json")
    assert result["status"] == ("failed" if failure else "succeeded")
    assert "controller-closed" in events
    if failure == "provider":
        assert "gateway-started" not in events
        assert controllers[0].ledger.snapshot()["model_calls"] == []
    else:
        assert "gateway-stopped" in events and "bridge-closed" in events
    if not failure:
        assert result["services_stopped"]
