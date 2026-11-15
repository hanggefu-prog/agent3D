"""Code-fixed Global Analyzer -> workspace agent -> trusted evaluation order."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Optional

from .impact import EvaluationPlan, ImpactResolver
from .intervention import GlobalSearchContext, Intervention, InterventionResult
from .model import GlobalTaskSpec


@dataclass(frozen=True)
class GlobalCandidateFeedback:
    trial_id: str
    passed: bool
    metrics: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    reason: str

    def to_dict(self):
        value = asdict(self)
        value["schema_version"] = 1
        return value


@dataclass(frozen=True)
class GlobalRoundResult:
    context: GlobalSearchContext
    intervention: Intervention
    intervention_result: Optional[InterventionResult]
    evaluation_plan: Optional[EvaluationPlan]
    feedback: Optional[GlobalCandidateFeedback]

    def to_dict(self):
        return {
            "schema_version": 1,
            "context": self.context.to_dict(),
            "intervention": self.intervention.to_dict(),
            "intervention_result": (
                None if self.intervention_result is None else self.intervention_result.to_dict()
            ),
            "evaluation_plan": (
                None if self.evaluation_plan is None else self.evaluation_plan.to_dict()
            ),
            "feedback": None if self.feedback is None else self.feedback.to_dict(),
        }


class GlobalRoundCoordinator:
    """Fix one global round without constraining either agent implementation."""

    def __init__(
        self,
        task: GlobalTaskSpec,
        modeler: Callable[[GlobalSearchContext], Intervention],
        executor: Callable[[Intervention, GlobalSearchContext], InterventionResult],
        evaluator: Callable[
            [InterventionResult, EvaluationPlan], GlobalCandidateFeedback
        ],
    ):
        self.task = task
        self.modeler = modeler
        self.executor = executor
        self.evaluator = evaluator

    def run_round(self, context: GlobalSearchContext) -> GlobalRoundResult:
        if context.task_id != self.task.task_id:
            raise ValueError("global context belongs to a different task")
        intervention = self.modeler(context)
        if intervention.global_round != context.global_round:
            raise ValueError("Global Analyzer returned the wrong round")
        if intervention.decision != "explore":
            return GlobalRoundResult(context, intervention, None, None, None)

        result = self.executor(intervention, context)
        if result.intervention_id != intervention.intervention_id:
            raise ValueError("acting agent returned the wrong Intervention")
        if result.base_revision != context.accepted_revision:
            raise ValueError("acting agent used a stale accepted revision")
        plan = ImpactResolver(self.task.evaluation).resolve(
            result.changed_files,
            claimed_stages=tuple(intervention.target_stages)
            + tuple(result.claimed_affected_stages),
            policy=intervention.evaluation_policy,
        )
        feedback = self.evaluator(result, plan)
        if feedback.trial_id != result.trial_id:
            raise ValueError("trusted evaluator returned feedback for a different trial")
        return GlobalRoundResult(context, intervention, result, plan, feedback)
