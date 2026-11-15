"""Stateful read-only Global Analyzer for workspace benchmark mode."""

from __future__ import annotations

from typing import Any, Dict

from agentic3d.analyzers import AnalyzerConfig, StatefulAnalyzer

from .intervention import (
    GLOBAL_WORKSPACE_ANALYZER_SCHEMA,
    GlobalSearchContext,
    Intervention,
    build_global_analyzer_prompt,
)
from .model import GlobalTaskSpec


def context_from_payload(payload: Dict[str, Any]) -> GlobalSearchContext:
    return GlobalSearchContext(
        task_id=str(payload["task_id"]),
        user_command=str(payload["user_command"]),
        global_round=int(payload["global_round"]),
        accepted_revision=str(payload["accepted_revision"]),
        objectives=dict(payload["objectives"]),
        hard_constraints=tuple(str(item) for item in payload["hard_constraints"]),
        stage_artifacts=dict(payload["stage_artifacts"]),
        observations=tuple(dict(item) for item in payload["observations"]),
        remaining_budget=dict(payload["remaining_budget"]),
    )


class WorkspaceGlobalAnalyzer(StatefulAnalyzer):
    """Reuse Codex's native read tools while enforcing the Intervention schema."""

    role = "Global Analyzer (workspace benchmark)"
    schema = GLOBAL_WORKSPACE_ANALYZER_SCHEMA

    def __init__(self, config: AnalyzerConfig, task: GlobalTaskSpec) -> None:
        self.task = task
        super().__init__(config)

    def _prompt(self, payload: Dict[str, Any], round_number: int) -> str:
        context = context_from_payload(payload)
        if context.global_round != round_number:
            raise ValueError("GlobalSearchContext round does not match analyzer state")
        skill = self.skill_path.read_text(encoding="utf-8")
        return "{}\n\nGlobal Analyzer Skill:\n---BEGIN SKILL---\n{}\n---END SKILL---\n".format(
            build_global_analyzer_prompt(self.task, context), skill
        )

    def _validate(
        self,
        output: Dict[str, Any],
        round_number: int,
        payload: Dict[str, Any],
    ) -> None:
        intervention = Intervention.from_mapping(output, self.task)
        if intervention.global_round != round_number:
            raise RuntimeError("Global Analyzer returned the wrong round")
        if intervention.decision != "explore" and intervention.target_stages:
            raise RuntimeError("complete/blocked decisions must not dispatch target stages")

    def analyze_intervention(self, context: GlobalSearchContext) -> Intervention:
        output = self.analyze(context.to_dict())
        return Intervention.from_mapping(output, self.task)
