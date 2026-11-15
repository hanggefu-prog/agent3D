"""Exercise both shipped agents; mock expensive model/native operations only."""
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from contest2.agent_api import AgentAPIError
from contest2.budget import BudgetLedger
from contest2.controller import Task2Controller, TASK_IDS
from contest2.smoke import stage_agent
from grt_bench.common import write
from local_tasks.tp_gnn_local_task_definitions import (
    materialize_binary_loss, materialize_loss_decoder, validate_program,
)

ROOT = Path(__file__).resolve().parents[1]


def load_agent(folder, baseline):
    file = folder / "agent.py"
    shutil.copyfile(ROOT / "agent" / baseline / "agent.py", file)
    if not (folder / "program.py").exists():
        (folder / "program.py").write_text("def objective(): return 0\n")
    spec = importlib.util.spec_from_file_location("tested_agent", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent()


class Context:
    def __init__(self, folder, baseline, outcomes, mode="formal", rejection=None):
        self.folder, self.task, self.outcomes = folder, baseline, outcomes
        self.evaluation_mode = mode
        self.evaluation_cells = [{"case": "bp_fe-ng45", "seed": "1003"}]
        self.history, self.prompts, self.events, self.inputs = [], [], [], []
        self.sources = {}
        self.model_active = False
        self.rejection = rejection
        self.submit_calls = 0
        self.cost_open = True
        self.model_success = True
        self.fail_first_model = False
        self.fail_on = set()

    def status(self):
        return {"dispatch_allowed": self.cost_open,
                "evaluations_remaining": len(self.outcomes) - len(self.history),
                "history": [dict(row) for row in self.history]}

    def codex_turn(self, prompt, request_id=None):
        assert not self.model_active
        self.model_active = True
        self.prompts.append(prompt)
        self.events.append("model-start")
        program = self.folder / "program.py"
        self.inputs.append(program.read_text())
        program.write_text("def objective(): return " + str(len(self.prompts)) + "\n")
        self.events.append("model-return")
        self.model_active = False
        return {"success": self.model_success and len(self.prompts) not in self.fail_on
                and not (self.fail_first_model and len(self.prompts) == 1)}

    def submit_evaluation(self, source, request_id=None):
        assert not self.model_active, "native evaluation is inside a model call"
        self.submit_calls += 1
        if self.rejection == "before" and self.submit_calls == 1:
            raise AgentAPIError("bridge connection failed before delivery")
        self.events.append("submit")
        candidate = str(len(self.history) + 1) * 64
        self.history.append({"request_id": request_id, "candidate_id": candidate,
                             "status": "accepted", "reason": None, "baseline": self.task})
        self.sources[candidate] = source
        if self.rejection == "after" and self.submit_calls == 1:
            raise AgentAPIError("connection interrupted after acceptance")
        return {"request_id": request_id}

    def wait_evaluation(self, request_id):
        assert not self.model_active, "native wait consumes model-call timeout"
        self.events.append("wait")
        i = next(i for i, row in enumerate(self.history) if row["request_id"] == request_id)
        status, score = self.outcomes[i]
        row = self.history[i]
        row["status"] = status
        return {**row, "source": self.sources[row["candidate_id"]],
                "feedback": {"score": score}, "reason": "native failure" if status == "failed" else None}


@pytest.mark.parametrize("baseline", TASK_IDS)
def test_multiple_rounds_use_feedback_and_return_best_verified_source(tmp_path, baseline):
    context = Context(tmp_path, baseline, [("succeeded", 8), ("failed", None), ("succeeded", 3)])
    assert load_agent(tmp_path, baseline).run(context) == "1" * 64
    assert context.events == ["model-start", "model-return", "submit", "wait"] * 3
    assert '"score": 8' in context.prompts[1]
    assert "native failure" in context.prompts[2]
    assert context.inputs[1] == context.inputs[2] == context.sources["1" * 64]
    assert (tmp_path / "program.py").read_text() == context.sources["1" * 64]


@pytest.mark.parametrize("baseline", TASK_IDS)
@pytest.mark.parametrize("rejection,turns", [("before", 2), ("after", 1)])
def test_bridge_failure_before_or_after_intake_does_not_duplicate_acceptance(tmp_path, baseline, rejection, turns):
    context = Context(tmp_path, baseline, [("succeeded", 4)], rejection=rejection)
    assert load_agent(tmp_path, baseline).run(context) == "1" * 64
    assert len(context.history) == 1 and len(context.prompts) == turns
    if rejection == "before":
        assert "bridge connection failed" in context.prompts[1]
    else:
        assert context.submit_calls == 1


def test_uncertain_api_keeps_polling_the_same_request(monkeypatch):
    from contest2.agent_api import AgentContext
    import time
    api = AgentContext.__new__(AgentContext)
    states = iter(["uncertain", "reconciling", "running", "succeeded"])
    calls = []
    def status(request):
        calls.append(request)
        return {"status": next(states)}
    api.evaluation_status = status
    monkeypatch.setattr(time, "sleep", lambda _: None)
    assert api.wait_evaluation("same")["status"] == "succeeded"
    assert calls == ["same"] * 4


@pytest.mark.parametrize("baseline", TASK_IDS)
@pytest.mark.parametrize("mode", ["formal", "smoke"])
def test_model_failure_retries_without_evaluation_charge(tmp_path, monkeypatch, baseline, mode):
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    context = Context(tmp_path, baseline, [("succeeded", 4)], mode=mode)
    context.fail_first_model = True
    assert load_agent(tmp_path, baseline).run(context) == "1" * 64
    assert context.submit_calls == len(context.history) == 1
    assert len(context.prompts) == (3 if mode == "smoke" else 2)
    assert "PREVIOUS MODEL FAILURE" in context.prompts[1]


@pytest.mark.parametrize("baseline", TASK_IDS)
def test_smoke_uses_default_agent_and_leaves_second_draft_unevaluated(tmp_path, baseline):
    context = Context(tmp_path, baseline, [("succeeded", 4)], mode="smoke")
    assert load_agent(tmp_path, baseline).run(context) == "1" * 64
    assert context.events == ["model-start", "model-return", "submit", "wait", "model-start", "model-return"]
    assert (tmp_path / "first_program.py").read_text() != (tmp_path / "second_program.py").read_text()
    staged = stage_agent(SimpleNamespace(root=tmp_path / "campaign", baseline=baseline,
        task_dir=ROOT / "local_tasks" / TASK_IDS[baseline]))
    assert (staged / "agent.py").read_bytes() == (ROOT / "agent" / baseline / "agent.py").read_bytes()


@pytest.mark.parametrize("baseline", TASK_IDS)
def test_resume_restores_best_and_last_feedback(tmp_path, baseline):
    context = Context(tmp_path, baseline, [("succeeded", 8), ("succeeded", 3)])
    agent = load_agent(tmp_path, baseline)
    context.submit_evaluation("def objective(): return 42\n", request_id="existing")
    context.wait_evaluation("existing")
    context.events.clear()
    assert agent.run(context) == "1" * 64
    assert len(context.prompts) == 1
    assert context.inputs[0] == context.sources["1" * 64]
    assert '"score": 8' in context.prompts[0]


@pytest.mark.parametrize("baseline", TASK_IDS)
@pytest.mark.parametrize("reason", ["evaluations", "cost"])
def test_budget_and_model_failure_do_not_submit_extra_work(tmp_path, baseline, reason):
    context = Context(tmp_path, baseline, [] if reason == "evaluations" else [("succeeded", 1)])
    context.cost_open = reason != "cost"
    context.model_success = reason != "model-failure"
    with pytest.raises(RuntimeError):
        load_agent(tmp_path, baseline).run(context)
    assert context.submit_calls == 0
    if reason != "model-failure":
        assert context.prompts == []


def stored_controller(root, baseline, ledger):
    # No EDA execution: use the real ledger, capsule validation, and controller API.
    controller = Task2Controller.__new__(Task2Controller)
    controller.baseline, controller.task_id = baseline, TASK_IDS[baseline]
    controller.evaluation_mode = "formal"
    controller.codex_binary = None
    controller.root = root.resolve()
    controller.matrix = SimpleNamespace(identity=baseline + "-matrix", cells=[])
    controller.candidates = root / "candidates"
    controller.evaluations = root / "evaluations"
    controller.ledger, controller._futures = ledger, {}
    source = validate_program((ROOT / "agent" / baseline / "program.py").read_text(), controller.task_id)
    candidate = controller._candidate_id(source)
    materialize = materialize_binary_loss if baseline == "binary_loss" else materialize_loss_decoder
    write(controller.candidates / candidate / "candidate.json", materialize(source))
    ledger.accept_matrix(baseline, candidate, "a" * 64, baseline=baseline)
    ledger.finish_evaluation(baseline, "succeeded")
    write(controller.evaluations / baseline / "evaluation_result.json", {"metrics": {
        "runs_successfully": 1, "combined_score": 9, "matrix": {"cases": [], "cells": []}}})
    return controller, candidate, source


def test_real_controller_isolates_histories_and_returns_verified_source(tmp_path):
    books = {b: BudgetLedger(tmp_path / b / "budget.sqlite", task_id=b,
                            model_cost_limit_rmb="15", allowed_model="deepseek-flash")
             for b in TASK_IDS}
    binary, candidate, source = stored_controller(tmp_path, "binary_loss", books["binary_loss"])
    decoder, _, _ = stored_controller(tmp_path, "loss_decoder", books["loss_decoder"])
    for current in (binary, decoder):
        state = current.status()
        assert state["evaluations_remaining"] == 9
        assert [row["baseline"] for row in state["history"]] == [current.baseline]
    assert binary.evaluation_status("binary_loss")["source"] == source
    assert decoder.evaluation_status("loss_decoder")["source"]
    views = tmp_path / "agent-evaluations"
    assert (views / "binary_loss/binary_loss/feedback.json").is_file()
    assert (views / "loss_decoder/loss_decoder/feedback.json").is_file()
    assert not (views / "binary_loss/loss_decoder").exists()
    with pytest.raises((KeyError, ValueError)):
        binary.evaluation_status("loss_decoder")
    changed = source.replace("return ", "return 1.01 * ", 1)
    assert changed != source
    write(binary.candidates / candidate / "candidate.json",
          materialize_binary_loss(changed), replace=True)
    with pytest.raises(ValueError, match="frozen identity"):
        binary.evaluation_status("binary_loss")


@pytest.mark.parametrize("baseline", TASK_IDS)
def test_rejected_candidate_consumes_one_attempt_then_feedback_drives_next_edit(tmp_path, baseline):
    context = Context(tmp_path, baseline, [("rejected", None), ("succeeded", 5)])
    assert load_agent(tmp_path, baseline).run(context) == "2" * 64
    assert len(context.history) == context.submit_calls == 2
    assert '"status": "rejected"' in context.prompts[1]


@pytest.mark.parametrize("baseline", TASK_IDS)
@pytest.mark.parametrize("mode", ["formal", "smoke"])
def test_model_failure_after_a_score_keeps_developing_and_preserves_feedback(tmp_path, monkeypatch, baseline, mode):
    import json
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    context = Context(tmp_path, baseline, [("succeeded", 5)] if mode == "smoke" else [("succeeded", 5), ("succeeded", 8)], mode=mode)
    context.fail_on = {2}
    winner = load_agent(tmp_path, baseline).run(context)
    assert winner == ("1" if mode == "smoke" else "2") * 64
    assert len(context.prompts) == 3
    assert '"score": 5' in context.prompts[2]
    assert "PREVIOUS MODEL FAILURE" in context.prompts[2]
    assert json.loads((tmp_path / "feedback.json").read_text())["status"] == "succeeded"


@pytest.mark.parametrize("baseline", TASK_IDS)
def test_unresolved_model_budget_does_not_report_normal_completion(tmp_path, baseline):
    context = Context(tmp_path, baseline, [("succeeded", 8), ("succeeded", 3)])
    context.submit_evaluation("def objective(): return 1\n", "old-request")
    context.wait_evaluation("old-request")
    original = context.status
    context.status = lambda: {**original(), "dispatch_allowed": False, "unresolved_model_calls": 1,
                              "model_stop_reasons": ["insufficient_model_reservation"]}
    with pytest.raises(RuntimeError, match="Model usage is unresolved"):
        load_agent(tmp_path, baseline).run(context)
    assert context.submit_calls == 1 and not context.prompts
