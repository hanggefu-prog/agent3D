"""Versioned workspace-task selection; budgets never rewrite task documents."""
from pathlib import Path

from global_bench.model import load_global_task


FRAMEWORK = Path(__file__).resolve().parents[1]
TASKS = {"timing": "lol_partition_placement_timing_one_pass",
         "thermal": "lol_partition_placement_cts_thermal_one_pass",
         "routed_thermal": "lol_partition_placement_thermal"}
TIMING_TEN_TASK = "lol_partition_placement_timing_one_pass_10"
LATEST_BASELINE = "reference_20261004"
LATEST_PROFILE = "tp_gnn_placement_one_pass_20261004"
LATEST_TASKS = {5: "lol_partition_placement_timing_20261004", 10: "lol_partition_placement_timing_20261004_10"}
LATEST_THERMAL_PROFILE = "tp_gnn_placement_cts_thermal_20261008"
LATEST_THERMAL_TASKS = {5: "lol_partition_placement_cts_thermal_20261008",
                        10: "lol_partition_placement_cts_thermal_20261008_10"}
LATEST_PROFILES = {LATEST_PROFILE, LATEST_THERMAL_PROFILE}
VARIANTS = {(objective, 5): task_id for objective, task_id in TASKS.items()}
VARIANTS[("timing", 10)] = TIMING_TEN_TASK


def workspace_task_id(objective, rounds=5, *, baseline=None):
    if baseline == LATEST_BASELINE:
        choices = {"timing": LATEST_TASKS, "thermal": LATEST_THERMAL_TASKS}.get(objective, {})
        if type(rounds) is not int or rounds not in choices:
            raise ValueError("Latest family requires a registered objective and round budget")
        return choices[rounds]
    if type(rounds) is not int or (objective, rounds) not in VARIANTS:
        raise ValueError("Only registered five-round tasks or the explicit ten-round timing task are supported")
    return VARIANTS[(objective, rounds)]


def validate_selected_task(task, objective, rounds, *, expected_digest=None):
    baseline = LATEST_BASELINE if task.task_id in (*LATEST_TASKS.values(), *LATEST_THERMAL_TASKS.values()) else None
    if (task.task_id != workspace_task_id(objective, rounds, baseline=baseline)
            or type(task.budget.global_rounds) is not int
            or type(task.budget.expensive_evaluations) is not int
            or task.budget.global_rounds != rounds
            or task.budget.expensive_evaluations != rounds):
        raise ValueError("The selected task must already define exactly {} rounds and {} candidate finals".format(rounds, rounds))
    if expected_digest is not None and task.digest() != expected_digest:
        raise ValueError("Workspace task digest differs from the pinned wrapper selection")
    return task


def load_registered_task(task_id, *, objective, framework=FRAMEWORK):
    """Rebuild only a checked-in authority, never an arbitrary submitted path."""
    choices = [(rounds, name) for (kind, rounds), name in VARIANTS.items()
               if kind == objective and name == task_id]
    if objective == "timing":
        choices += [(rounds, name) for rounds, name in LATEST_TASKS.items() if name == task_id]
    if objective == "thermal":
        choices += [(rounds, name) for rounds, name in LATEST_THERMAL_TASKS.items() if name == task_id]
    if len(choices) != 1:
        raise ValueError("Unregistered workspace task/objective identity")
    rounds, name = choices[0]
    task = load_global_task(Path(framework) / "global_tasks" / name)
    return validate_selected_task(task, objective, rounds)


def select_driver_task(runtime, objective, *, framework, loader=load_global_task):
    """Keep direct legacy lifecycles while honoring an explicit wrapper pin."""
    requested = getattr(runtime, "workspace_rounds", None)
    task = loader(Path(framework) / "global_tasks" /
                  workspace_task_id(objective, 5 if requested is None else requested,
                                    baseline=getattr(runtime, "reference_baseline", None)))
    if requested is not None:
        validate_selected_task(task, objective, requested,
                               expected_digest=getattr(runtime, "workspace_task_digest", None))
    return task
