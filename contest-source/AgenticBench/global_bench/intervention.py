"""Global Analyzer handoffs for open-ended workspace exploration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .model import GlobalTaskSpec


INTERVENTION_POLICIES = frozenset({"quick", "suffix", "final", "full"})
INTERVENTION_BACKENDS = frozenset({"workspace_agent", "evolve_task"})


def _json_object(value: Mapping[str, Any], label: str) -> Dict[str, Any]:
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must contain finite JSON values: {}".format(label, error)) from error
    loaded = json.loads(encoded)
    if not isinstance(loaded, dict):
        raise ValueError("{} must be a JSON object".format(label))
    return loaded


@dataclass(frozen=True)
class GlobalSearchContext:
    """Trusted state observed by Global Analyzer before it proposes a trial."""

    task_id: str
    user_command: str
    global_round: int
    accepted_revision: str
    objectives: Mapping[str, Any]
    hard_constraints: Tuple[str, ...]
    stage_artifacts: Mapping[str, Any]
    observations: Tuple[Mapping[str, Any], ...]
    remaining_budget: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.user_command.strip() or not self.accepted_revision.strip():
            raise ValueError("task_id, user_command, and accepted_revision must not be empty")
        if self.global_round < 0:
            raise ValueError("global_round cannot be negative")
        _json_object(self.objectives, "objectives")
        _json_object(self.stage_artifacts, "stage_artifacts")
        _json_object(self.remaining_budget, "remaining_budget")
        for observation in self.observations:
            _json_object(observation, "observation")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_command": self.user_command,
            "global_round": self.global_round,
            "accepted_revision": self.accepted_revision,
            "objectives": _json_object(self.objectives, "objectives"),
            "hard_constraints": list(self.hard_constraints),
            "stage_artifacts": _json_object(self.stage_artifacts, "stage_artifacts"),
            "observations": [
                _json_object(item, "observation") for item in self.observations
            ],
            "remaining_budget": _json_object(self.remaining_budget, "remaining_budget"),
        }


@dataclass(frozen=True)
class Intervention:
    """One hypothesis-driven trial request with no source/configuration scope."""

    intervention_id: str
    global_round: int
    decision: str
    hypothesis: str
    target_stages: Tuple[str, ...]
    objectives: Mapping[str, Any]
    hard_constraints: Tuple[str, ...]
    context_artifacts: Tuple[str, ...]
    evaluation_policy: str
    execution_backend: str
    upstream_checkpoint: Optional[str]
    budget: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.intervention_id.strip():
            raise ValueError("intervention_id must not be empty")
        if self.global_round < 0:
            raise ValueError("global_round cannot be negative")
        if self.decision not in {"explore", "complete", "blocked"}:
            raise ValueError("decision must be explore, complete, or blocked")
        if not self.hypothesis.strip():
            raise ValueError("hypothesis must not be empty")
        if self.decision == "explore" and not self.target_stages:
            raise ValueError("an exploratory intervention needs at least one target stage")
        if self.decision != "explore" and self.target_stages:
            raise ValueError("complete/blocked interventions cannot dispatch target stages")
        if self.evaluation_policy not in INTERVENTION_POLICIES:
            raise ValueError("unsupported evaluation policy")
        if self.execution_backend not in INTERVENTION_BACKENDS:
            raise ValueError("unsupported execution backend")
        _json_object(self.objectives, "objectives")
        _json_object(self.budget, "budget")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], task: GlobalTaskSpec) -> "Intervention":
        allowed = {
            "schema_version",
            "intervention_id",
            "global_round",
            "decision",
            "hypothesis",
            "target_stages",
            "objectives",
            "hard_constraints",
            "context_artifacts",
            "evaluation_policy",
            "execution_backend",
            "upstream_checkpoint",
            "budget",
        }
        missing = allowed - set(value)
        extra = set(value) - allowed
        if missing or extra:
            raise ValueError(
                "intervention has missing={} extra={}".format(
                    sorted(missing), sorted(extra)
                )
            )
        if int(value["schema_version"]) != 1:
            raise ValueError("unsupported intervention schema")
        targets = tuple(str(item) for item in value["target_stages"])
        unknown = set(targets) - set(task.evaluation.stage_ids())
        if unknown:
            raise ValueError("intervention targets unknown stages: {}".format(sorted(unknown)))
        checkpoint = value["upstream_checkpoint"]
        return cls(
            intervention_id=str(value["intervention_id"]),
            global_round=int(value["global_round"]),
            decision=str(value["decision"]),
            hypothesis=str(value["hypothesis"]),
            target_stages=targets,
            objectives=dict(value["objectives"]),
            hard_constraints=tuple(str(item) for item in value["hard_constraints"]),
            context_artifacts=tuple(str(item) for item in value["context_artifacts"]),
            evaluation_policy=str(value["evaluation_policy"]),
            execution_backend=str(value["execution_backend"]),
            upstream_checkpoint=None if checkpoint is None else str(checkpoint),
            budget=dict(value["budget"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = 1
        value["target_stages"] = list(self.target_stages)
        value["hard_constraints"] = list(self.hard_constraints)
        value["context_artifacts"] = list(self.context_artifacts)
        return value


@dataclass(frozen=True)
class InterventionResult:
    intervention_id: str
    trial_id: str
    base_revision: str
    candidate_revision: Optional[str]
    changed_files: Tuple[str, ...]
    claimed_affected_stages: Tuple[str, ...]
    submission: str
    observations: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "intervention_id": self.intervention_id,
            "trial_id": self.trial_id,
            "base_revision": self.base_revision,
            "candidate_revision": self.candidate_revision,
            "changed_files": list(self.changed_files),
            "claimed_affected_stages": list(self.claimed_affected_stages),
            "submission": self.submission,
            "observations": [
                _json_object(item, "observation") for item in self.observations
            ],
        }


GLOBAL_WORKSPACE_ANALYZER_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "intervention_id",
        "global_round",
        "decision",
        "hypothesis",
        "target_stages",
        "objectives",
        "hard_constraints",
        "context_artifacts",
        "evaluation_policy",
        "execution_backend",
        "upstream_checkpoint",
        "budget",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "intervention_id": {"type": "string", "minLength": 1},
        "global_round": {"type": "integer", "minimum": 0},
        "decision": {"enum": ["explore", "complete", "blocked"]},
        "hypothesis": {"type": "string", "minLength": 1},
        "target_stages": {"type": "array", "items": {"type": "string"}},
        "objectives": {"type": "object"},
        "hard_constraints": {"type": "array", "items": {"type": "string"}},
        "context_artifacts": {"type": "array", "items": {"type": "string"}},
        "evaluation_policy": {"enum": sorted(INTERVENTION_POLICIES)},
        "execution_backend": {"enum": sorted(INTERVENTION_BACKENDS)},
        "upstream_checkpoint": {"type": ["string", "null"]},
        "budget": {"type": "object"},
    },
}


def build_global_analyzer_prompt(
    task: GlobalTaskSpec,
    context: GlobalSearchContext,
) -> str:
    """Build the global-mode prompt without prescribing source paths or code."""

    stages = ", ".join(task.evaluation.stage_ids())
    return """You are the read-only Global Analyzer for a workspace benchmark.

The acting agent receives an isolated, writable copy of the complete subject
repository. Do not define source_scope, config_scope, allowed files, a code
patch, or concrete parameter values. Instead, diagnose the evidence and return
one Intervention: a falsifiable hypothesis, one or more target stages, hard
constraints, context artifacts, execution backend, evaluation policy, and a
bounded experiment budget. target_stages are diagnostic and evaluation hints;
they are not edit restrictions. Trusted legality checks and evaluators are
outside the agent workspace and cannot be changed.

Random seeds are evaluator-owned and are not optimization parameters. The
immutable common-random-number contract is:
{seed_policy}

Task instruction:
---BEGIN TASK---
{instruction}
---END TASK---

Round-0 user command:
{user_command}

Known stages: {stages}

Trusted search context:
{context}
""".format(
        instruction=task.instruction,
        user_command=context.user_command,
        stages=stages,
        seed_policy=json.dumps(asdict(task.seed_policy), sort_keys=True),
        context=json.dumps(context.to_dict(), ensure_ascii=False, indent=2),
    )
