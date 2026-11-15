"""Static task budget variants; no model, Docker, or native flow execution."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from global_bench.cli import DEFAULT_TASK_ROOT, _catalog
from global_bench.verifier import load_trusted_verifier
from local_bench.reference_catalog import task_digest
from scripts.workspace_codesign_evaluator import endpoint_for_task
from scripts.workspace_task_selection import (FRAMEWORK, TASKS, TIMING_TEN_TASK,
    workspace_task_id, load_registered_task, select_driver_task)


def test_ten_round_variant_changes_only_budget_and_task_identity():
    old = load_registered_task(TASKS["timing"], objective="timing")
    new = load_registered_task(TIMING_TEN_TASK, objective="timing")
    left, right = old.to_dict(), new.to_dict()
    for data in (left, right):
        for key in ("task_id", "title", "root", "instruction_path", "instruction", "budget"):
            data.pop(key)
    assert left == right
    assert new.instruction == old.instruction.replace("five-evaluation budget", "ten-evaluation budget")
    assert old.budget.global_rounds == old.budget.expensive_evaluations == 5
    assert new.budget.global_rounds == new.budget.expensive_evaluations == 10
    assert new.budget.concurrent_trials == old.budget.concurrent_trials
    assert new.digest() != old.digest()
    assert task_digest(new.root) != task_digest(old.root)
    assert load_trusted_verifier(new.verifier.verifier_id).timeout_seconds == 21600
    before, after = endpoint_for_task("timing", old), endpoint_for_task("timing", new)
    assert after == {**before, "task_id": TIMING_TEN_TASK}


@pytest.mark.parametrize("objective,rounds", [("timing", 4), ("timing", 50), ("timing", 10.0),
    ("timing", True), ("thermal", 10), ("routed_thermal", 10), ("timing_10", 10)])
def test_only_explicit_registered_variants_are_allowed(objective, rounds):
    with pytest.raises(ValueError):
        workspace_task_id(objective, rounds)


def test_driver_selection_uses_same_frozen_digest_and_rejects_pin_mismatch():
    task = load_registered_task(TIMING_TEN_TASK, objective="timing")
    runtime = SimpleNamespace(workspace_rounds=10, workspace_task_digest=task.digest())
    assert select_driver_task(runtime, "timing", framework=FRAMEWORK) == task
    runtime.workspace_task_digest = "0" * 64
    with pytest.raises(ValueError, match="pinned wrapper"):
        select_driver_task(runtime, "timing", framework=FRAMEWORK)
    with pytest.raises(ValueError, match="Unregistered"):
        load_registered_task(TIMING_TEN_TASK, objective="thermal")
    with pytest.raises(ValueError, match="exactly 10"):
        endpoint_for_task("timing", replace(task, budget=replace(task.budget, expensive_evaluations=5)))


def test_direct_driver_defaults_to_unchanged_five_round_task():
    task = select_driver_task(SimpleNamespace(), "timing", framework=FRAMEWORK)
    assert task.task_id == TASKS["timing"]
    assert task.budget.global_rounds == task.budget.expensive_evaluations == 5


def test_global_catalog_lists_both_static_timing_budgets():
    catalog = _catalog(DEFAULT_TASK_ROOT)
    assert TASKS["timing"] in catalog
    assert TIMING_TEN_TASK in catalog
    assert catalog[TASKS["timing"]].budget.global_rounds == 5
    assert catalog[TIMING_TEN_TASK].budget.global_rounds == 10
    assert catalog[TIMING_TEN_TASK].digest() == load_registered_task(
        TIMING_TEN_TASK, objective="timing").digest()


def test_latest_thermal_five_round_variant_changes_only_identity_and_budget():
    from scripts.workspace_task_selection import LATEST_THERMAL_TASKS
    five, ten = [load_registered_task(LATEST_THERMAL_TASKS[n], objective="thermal") for n in (5, 10)]
    left, right = five.to_dict(), ten.to_dict()
    for data in (left, right):
        for key in ("task_id", "title", "root", "instruction_path", "budget"):
            data.pop(key)
    assert left == right
    assert five.budget.global_rounds == five.budget.expensive_evaluations == 5
    assert five.budget.concurrent_trials == ten.budget.concurrent_trials
    assert five.digest() != ten.digest()
    assert endpoint_for_task("thermal", five) == {**endpoint_for_task("thermal", ten), "task_id": five.task_id}
