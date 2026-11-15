"""A small explicit campaign plan. Preflight is read-only and never calls a model.

The once-only final ledger is separate from preparation, source development and
native qualification. A missing adapter is an infrastructure blocker, not a bad
candidate score and not permission to run an unrelated legacy evaluator.
"""
from pathlib import Path
import re

from .tasks import TASK_IDS, load_task, read_json
from .protocol import qualify, digest
from .scoring import CURRENT_SCORING_VERSION, scoring_version, selected_contract


def load_campaign(path):
    path = Path(path).resolve(strict=True)
    plan = read_json(path)
    keys = {"schema_version", "name", "developer_model", "judge_model", "seed",
            "submissions_per_task", "evaluations_per_case", "cases", "tasks",
            "framework_revision", "flow_revision", "hosts"}
    if not isinstance(plan, dict) or set(plan) - {'scoring_version'} != keys or type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Invalid engineering campaign")
    scoring_version(plan)
    for key in ("submissions_per_task", "evaluations_per_case"):
        if type(plan[key]) is not int or plan[key] != 1:
            raise ValueError("This campaign allows one submission/final per case")
    if plan["tasks"] != list(TASK_IDS):
        raise ValueError("Campaign must cover the three independent engineering tasks")
    for task_id in TASK_IDS:
        task = load_task(task_id)
        if task.data["cases"] != plan["cases"] or type(plan["seed"]) is not int or task.data["seed"] != plan["seed"]:
            raise ValueError("Case/seed mismatch")
    for key in ("developer_model", "judge_model"):
        if not isinstance(plan[key], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}", plan[key]):
            raise ValueError("Model must be explicit")
    assigned = []
    for host, config in plan["hosts"].items():
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]+", host):
            raise ValueError("Unsafe SSH alias")
        if set(config) != {"root", "cases", "reference_root", "max_parallel", "cpus_per_job", "memory_gb_per_job"}:
            raise ValueError("Invalid host plan")
        for key in ("root", "reference_root"):
            value = config[key]
            if not isinstance(value, str) or not Path(value).is_absolute() or Path(value) == Path("/") or ".." in Path(value).parts:
                raise ValueError("Use an explicit isolated absolute host directory")
        for key in ("max_parallel", "cpus_per_job", "memory_gb_per_job"):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError("Positive resource limits required")
        if config["cpus_per_job"] > 32 or config["memory_gb_per_job"] > 128:
            raise ValueError("Resource request exceeds the bounded worker profile")
        assigned.extend(config["cases"])
    if sorted(assigned) != sorted(plan["cases"]):
        raise ValueError("Assign each case to exactly one host, not one copy per host")
    return plan


def preflight(plan, qualification_root):
    """All cases must qualify before developing that task against the interface."""
    root = Path(qualification_root)
    version = plan.get('scoring_version', CURRENT_SCORING_VERSION)
    tasks = []
    for task_id in plan["tasks"]:
        task = load_task(task_id)
        cases = []
        for case in plan["cases"]:
            path = root / task_id / case / "qualification.json"
            try:
                proof = qualify(task, case, path)
                if proof['scoring_version'] != version:
                    raise ValueError('Qualification has another scoring version')
                cases.append({"case": case, "status": "qualified", "qualification_sha256": proof["sha256"]})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                cases.append({"case": case, "status": "blocked", "reason": str(exc)})
        tasks.append({"task": task_id, "task_digest": task.digest(),
                      "scoring_contract": selected_contract(task, version),
                      "ready": all(c["status"] == "qualified" for c in cases), "cases": cases})
    return {"schema_version": 1, "campaign_sha256": digest(plan), "tasks": tasks,
            "ready": all(t["ready"] for t in tasks), "candidate_finals_dispatched": 0,
            "model_calls_dispatched": 0}
