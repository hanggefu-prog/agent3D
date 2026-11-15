"""Agentic3D orchestration primitives."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .analyzers import AnalyzerConfig, GlobalAnalyzer, LocalAnalyzer
    from .codex_exec import CodexExecConfig, CodexRunResult
    from .contracts import CandidateUpdate, StageHandoff
    from .hierarchical import (
        CorrectionAggregationBackend,
        GlobalModelingContext,
        GlobalRoundResult,
        HierarchicalRoundCoordinator,
        IntegratedPPAEvaluator,
        IntegratedProposal,
        LocalCorrectionResult,
        StageTask,
        StageTaskMaterializer,
        local_task_template_catalog,
    )
    from .open3dbench import AdapterNotFoundError, Open3DBenchAdapter
    from .orchestrator import Agentic3DOrchestrator, WorkflowConfig, WorkflowState
    from .ppa import GlobalPPACollector
    from .task_agent import TaskAgent, TaskAgentConfig, TaskAgentState

__all__ = [
    "CodexExecConfig",
    "CodexRunResult",
    "run_codex",
    "TaskAgent",
    "TaskAgentConfig",
    "TaskAgentState",
    "AnalyzerConfig",
    "GlobalAnalyzer",
    "LocalAnalyzer",
    "CandidateUpdate",
    "StageHandoff",
    "AdapterNotFoundError",
    "Open3DBenchAdapter",
    "Agentic3DOrchestrator",
    "WorkflowConfig",
    "WorkflowState",
    "GlobalPPACollector",
    "GlobalModelingContext",
    "StageTask",
    "StageTaskMaterializer",
    "LocalCorrectionResult",
    "IntegratedProposal",
    "IntegratedPPAEvaluator",
    "CorrectionAggregationBackend",
    "GlobalRoundResult",
    "HierarchicalRoundCoordinator",
    "local_task_template_catalog",
]


def __getattr__(name: str) -> Any:
    """Load public execution primitives lazily.

    Lazy loading keeps ``python -m agentic3d.codex_exec`` free of runpy's
    double-import warning while preserving the compact package-level API.
    """

    if name in {"CodexExecConfig", "CodexRunResult", "run_codex"}:
        from . import codex_exec

        return getattr(codex_exec, name)
    if name in {"TaskAgent", "TaskAgentConfig", "TaskAgentState"}:
        from . import task_agent

        return getattr(task_agent, name)
    if name in {"AnalyzerConfig", "GlobalAnalyzer", "LocalAnalyzer"}:
        from . import analyzers

        return getattr(analyzers, name)
    if name in {"CandidateUpdate", "StageHandoff"}:
        from . import contracts

        return getattr(contracts, name)
    if name in {"AdapterNotFoundError", "Open3DBenchAdapter"}:
        from . import open3dbench

        return getattr(open3dbench, name)
    if name in {"Agentic3DOrchestrator", "WorkflowConfig", "WorkflowState"}:
        from . import orchestrator

        return getattr(orchestrator, name)
    if name == "GlobalPPACollector":
        from . import ppa

        return ppa.GlobalPPACollector
    if name in {
        "GlobalModelingContext",
        "StageTask",
        "StageTaskMaterializer",
        "LocalCorrectionResult",
        "IntegratedProposal",
        "IntegratedPPAEvaluator",
        "CorrectionAggregationBackend",
        "GlobalRoundResult",
        "HierarchicalRoundCoordinator",
        "local_task_template_catalog",
    }:
        from . import hierarchical

        return getattr(hierarchical, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
