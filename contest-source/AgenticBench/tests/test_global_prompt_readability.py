"""Prompt presentation changes must preserve the search and review contracts."""
import copy
import json

import pytest

from global_bench.model import load_global_task
from scripts import run_workspace_codesign as codex
from scripts import run_workspace_agentic3d as agentic
from scripts.workspace_task_selection import LATEST_TASKS, LATEST_THERMAL_TASKS


@pytest.mark.parametrize("task_id", [*LATEST_TASKS.values(), *LATEST_THERMAL_TASKS.values()])
def test_current_task_explains_problem_before_measurement(task_id):
    task = load_global_task(codex.ROOT / "global_tasks" / task_id)
    text = task.instruction
    assert text.index("## Background") < text.index("## Your task") < text.index("## How results")
    for phrase in ("logic-on-logic (LoL)", "three-dimensional integrated circuit (3D-IC)",
                   "global two-way", "hybrid bonding terminal (HBT)", "combined_score"):
        assert phrase in text
    for plumbing in ("4d882f", "reference_20261004", "manifest", "SHA", "21600",
                     "four physical threads", "15_timing_extract", "qualification records"):
        assert plumbing not in text
    assert "unknown" in text and "failed evaluations" in " ".join(text.lower().split())
    assert len(text.split()) < 420
    if task.objective.kind == "sta_wns_tns":
        assert "exactly one timing" in text
        assert "worst negative slack (WNS)" in text
        assert "total negative slack (TNS)" in text
        assert "0.6" in text and "0.4" in text and "effectively zero" in text
        assert task.objective.parameters["opt_passes"] == 1
    else:
        assert "one timing optimization" in text
        assert "200 equal-area silicon tiles" in text
        assert "No routing" in text and "318.15" in text and "0.1" in text
        assert task.objective.parameters["opt_passes"] == 1


def test_codex_wrapper_is_short_and_does_not_hide_starting_program():
    task = load_global_task(codex.ROOT / "global_tasks" / LATEST_TASKS[10])
    text = codex.round_prompt(task, "bp_fe", 1, 10, {"history": []})
    assert len(text.replace(task.instruction, "").split()) < 290
    assert "source checkout in /workspace is your starting program" in text
    assert "algorithm" in text and "partition.config" in text and "placement.config" in text
    assert "No fixed code block" not in text  # Use plain instructions, not legacy boilerplate.
    assert text.count(task.instruction) == 1


@pytest.mark.parametrize("phase", list(agentic.PHASE_ROLES))
def test_agentic_context_is_included_once_without_mutating_evidence(phase):
    task = load_global_task(codex.ROOT / "global_tasks" / LATEST_TASKS[10])
    context = {"user_task": task.instruction, "candidate_round": 2,
               "phase": phase, "role": agentic.PHASE_ROLES[phase],
               "history": [{"sentinel": "ONE_HISTORY_COPY", "success": False, "metrics": {}}],
               "prior_best_verified": {"score": 1.01},
               "best_verified": {"score": 1.02}, "accepted_incumbent": {"score": 1.0}}
    before = copy.deepcopy(context)
    text = agentic._prompt(task, "bp_fe", phase, 2, context)
    payload = text.split("---BEGIN TRUSTED CONTEXT---\n", 1)[1].split("\n---END TRUSTED CONTEXT---", 1)[0]
    assert json.loads(payload) == {key: value for key, value in context.items() if key != "user_task"}
    assert context == before
    assert text.count("ONE_HISTORY_COPY") == 1
    assert text.count("# Improve timing through partitioning and placement") == 1
    assert len(text.split("---BEGIN TRUSTED CONTEXT---", 1)[0].replace(task.instruction, "").split()) < 430
    if phase == "task":
        assert "Agentic3D Task agent" in text and "Codex agent" not in text
        assert "/scratch/plan-round-02.md" in text
    else:
        assert "read-only" in text and "Do not edit the candidate" in text
        assert "1 to 12" in text
        if phase in {"local-post", "global-adjudication"}:
            assert "context.prior_best_verified" in text
            assert "/results/round-02/feedback.json" in text
            assert '"decision": "reject"' in text
