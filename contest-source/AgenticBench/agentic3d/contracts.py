"""Machine-readable contracts shared by the Agentic3D agents.

The contracts intentionally contain paths and evidence references rather than
Python objects from Open3DBench.  This keeps the control plane independent from
the repository revision being optimized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


from local_bench.task_inventory import EXECUTABLE_TASK_STAGES


STAGES = ("partition", "layout", "place_opt", "routing")
LOCAL_TASK_TEMPLATES = tuple(EXECUTABLE_TASK_STAGES)


def require_stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return stage


@dataclass(frozen=True)
class StageHandoff:
    workflow_id: str
    cycle: int
    stage: str
    user_task: str
    repository_context: str
    global_plan: str
    local_feedback: str
    scope_contract: Optional[str] = None
    accepted_inputs: List[str] = field(default_factory=list)
    immutable_constraints: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_stage(self.stage)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateUpdate:
    workflow_id: str
    cycle: int
    stage: str
    task_status: str
    task_state: str
    round_results: List[str]
    discovered_artifacts: List[Mapping[str, Any]]
    scope_mode: str = "enforced"
    worktree: Optional[str] = None
    baseline_commit: Optional[str] = None
    candidate_commit: Optional[str] = None
    scope_contract: Optional[str] = None
    scope_report: Optional[str] = None
    integrated_commit: Optional[str] = None
    local_feedback: Optional[str] = None

    def __post_init__(self) -> None:
        require_stage(self.stage)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


GLOBAL_ANALYZER_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["round", "diagnosis", "global_plan", "decision"],
    "properties": {
        "round": {"type": "integer", "minimum": 0},
        "diagnosis": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "ppa_objective", "bottlenecks", "evidence"],
            "properties": {
                "summary": {"type": "string"},
                "ppa_objective": {"type": "string"},
                "bottlenecks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["stage", "severity", "reason"],
                        "properties": {
                            "stage": {"type": "string", "enum": list(STAGES)},
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "high", "medium", "low"],
                            },
                            "reason": {"type": "string"},
                        },
                    },
                },
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
        },
        "global_plan": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "target_stage",
                "rationale",
                "action_hint",
                "success_criteria",
                "required_artifacts",
                "frozen_constraints",
                "local_tasks",
            ],
            "properties": {
                "target_stage": {"type": "string", "enum": list(STAGES)},
                "rationale": {"type": "string"},
                "action_hint": {"type": "string"},
                "success_criteria": {"type": "array", "items": {"type": "string"}},
                "required_artifacts": {"type": "array", "items": {"type": "string"}},
                "frozen_constraints": {"type": "array", "items": {"type": "string"}},
                "local_tasks": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "task_id",
                            "template",
                            "stage",
                            "hypothesis",
                            "objective",
                            "acceptance_gates",
                            "correctness_gates",
                            "evaluation_budget",
                            "source_scope",
                            "config_scope",
                            "context_artifacts",
                        ],
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                            },
                            "template": {
                                "type": "string",
                                "enum": list(LOCAL_TASK_TEMPLATES),
                            },
                            "stage": {"type": "string", "enum": list(STAGES)},
                            "hypothesis": {"type": "string", "minLength": 1},
                            "objective": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["goal", "metric_weights"],
                                "properties": {
                                    "goal": {"type": "string", "minLength": 1},
                                    "metric_weights": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "number",
                                            "minimum": 0,
                                        },
                                    },
                                },
                            },
                            "acceptance_gates": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "correctness_gates": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "evaluation_budget": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 1000,
                            },
                            "source_scope": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "config_scope": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "context_artifacts": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "decision": {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "candidate_status", "reason"],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["dispatch", "complete", "blocked"],
                },
                "candidate_status": {
                    "type": "string",
                    "enum": ["not_applicable", "accept", "reject", "pending"],
                },
                "reason": {"type": "string"},
            },
        },
    },
}


LOCAL_ANALYZER_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "stage",
        "analysis_phase",
        "conclusion",
        "evidence",
        "single_hypothesis",
        "scope_proposal",
        "allowed_changes",
        "forbidden_changes",
        "acceptance_gates",
        "correctness_gates",
        "replay_rule",
        "rollback_rule",
        "decision",
    ],
    "properties": {
        "stage": {"type": "string", "enum": list(STAGES)},
        "analysis_phase": {
            "type": "string",
            "enum": ["pre_task", "post_task"],
        },
        "conclusion": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "single_hypothesis": {"type": "string"},
        "scope_proposal": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "source_paths",
                "config_paths",
                "allow_create",
                "allow_delete",
                "allow_rename",
                "allow_mode_change",
                "allow_symlink",
                "allowed_extensions",
                "max_changed_files",
                "max_added_lines",
                "max_deleted_lines",
            ],
            "properties": {
                "source_paths": {"type": "array", "items": {"type": "string"}},
                "config_paths": {"type": "array", "items": {"type": "string"}},
                "allow_create": {"type": "boolean"},
                "allow_delete": {"type": "boolean"},
                "allow_rename": {"type": "boolean"},
                "allow_mode_change": {"type": "boolean"},
                "allow_symlink": {"type": "boolean"},
                "allowed_extensions": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^\\.[A-Za-z0-9_+-]+$"},
                },
                "max_changed_files": {"type": "integer", "minimum": 0, "maximum": 50},
                "max_added_lines": {"type": "integer", "minimum": 0, "maximum": 5000},
                "max_deleted_lines": {"type": "integer", "minimum": 0, "maximum": 5000},
            },
        },
        "allowed_changes": {"type": "array", "items": {"type": "string"}},
        "forbidden_changes": {"type": "array", "items": {"type": "string"}},
        "acceptance_gates": {"type": "array", "items": {"type": "string"}},
        "correctness_gates": {"type": "array", "items": {"type": "string"}},
        "replay_rule": {"type": "string"},
        "rollback_rule": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["recommend", "accept", "need_evidence", "reject"],
        },
    },
}
