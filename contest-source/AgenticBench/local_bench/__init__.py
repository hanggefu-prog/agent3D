"""OpenEvolve-style local optimization environments for Open3DBench."""

from .evaluation import evaluate_task
from .matrix import LocalEvaluationMatrix, qualify_matrix
from .model import EvaluationResult, TaskSpec, load_task
from .openevolve_search import OpenEvolveAskTell
from .search_session import prepare_search_problem, run_task_search
from .search_api import (
    AskTellRunner,
    CallbackSearchBackend,
    CandidateFeedback,
    CandidateProposal,
    FirstImprovementRunner,
    ProgramEvaluationHarness,
    SearchBackend,
    SearchContext,
    SearchProblem,
    SearchRunResult,
    TaskEvaluationOptions,
)

__all__ = [
    "AskTellRunner",
    "CallbackSearchBackend",
    "CandidateFeedback",
    "CandidateProposal",
    "EvaluationResult",
    "OpenEvolveAskTell",
    "prepare_search_problem",
    "run_task_search",
    "FirstImprovementRunner",
    "ProgramEvaluationHarness",
    "SearchBackend",
    "SearchContext",
    "SearchProblem",
    "SearchRunResult",
    "TaskEvaluationOptions",
    "TaskSpec",
    "evaluate_task",
    "load_task",
    "LocalEvaluationMatrix",
    "qualify_matrix",
]
