"""Code-defined collaboration loop for the Agentic3D agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from local_bench.skill_catalog import (
    load_skill_catalog,
    required_skills,
    skill_by_directory,
)

from .analyzers import AnalyzerConfig, GlobalAnalyzer, LocalAnalyzer
from .contracts import CandidateUpdate, StageHandoff, require_stage
from .hierarchical import (
    local_task_template_catalog,
    modeled_tasks_from_global_analysis,
)
from .open3dbench import Open3DBenchAdapter
from .ppa import GlobalPPACollector
from .scope_guard import (
    SCOPED_STAGES,
    IntegrationGate,
    ScopeCompiler,
    ScopeGuard,
    StageScope,
    WorktreeManager,
)
from .task_agent import TERMINAL_STATUSES, TaskAgent, TaskAgentConfig


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _repository_state_root(repository: Path) -> Path:
    """Return an external, stable state directory for one target checkout."""
    configured = os.environ.get("AGENTIC3D_RUNS_ROOT")
    if configured:
        base = Path(configured).expanduser().resolve()
    else:
        xdg_state = os.environ.get("XDG_STATE_HOME")
        base = (
            Path(xdg_state).expanduser().resolve() / "agentic3d"
            if xdg_state
            else Path.home() / ".local" / "state" / "agentic3d"
        )
    resolved = repository.expanduser().resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved.name).strip("-")
    slug = slug or "repository"
    identity = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return base / "repositories" / f"{slug}-{identity}"


@dataclass(frozen=True)
class WorkflowConfig:
    repository: Path
    framework_root: Path
    user_task: str
    workflow_dir: Path
    codex_binary: str = "codex"
    model: Optional[str] = None
    timeout_seconds: Optional[float] = None
    max_cycles: int = 8
    task_max_rounds: int = 3
    task_stall_limit: int = 2
    profile: Optional[str] = None
    config_overrides: Sequence[str] = ()
    strict_config: bool = False
    worktrees_root: Optional[Path] = None
    scope_mode: str = "enforced"

    def __post_init__(self) -> None:
        if not self.user_task.strip():
            raise ValueError("user_task must not be empty")
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        if self.scope_mode not in {"enforced", "direct"}:
            raise ValueError("scope_mode must be 'enforced' or 'direct'")


@dataclass
class WorkflowState:
    workflow_id: str
    user_task: str
    repository: str
    adapter: str
    status: str
    scope_mode: str = "enforced"
    next_cycle: int = 0
    current_stage: Optional[str] = None
    latest_global_plan: Optional[str] = None
    active_tasks: Dict[str, str] = field(default_factory=dict)
    accepted_candidates: Dict[str, str] = field(default_factory=dict)
    cycles: List[Dict[str, Any]] = field(default_factory=list)
    terminal_reason: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


class Agentic3DOrchestrator:
    """Fix the Global → Local → Task → Local → Global protocol in code."""

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config
        self.repository = config.repository.expanduser().resolve()
        self.framework_root = config.framework_root.expanduser().resolve()
        self.workflow_dir = config.workflow_dir.expanduser().resolve()
        if not self.repository.is_dir():
            raise ValueError(f"repository does not exist: {self.repository}")
        if not (self.framework_root / "skills").is_dir():
            raise ValueError(f"framework skills are missing: {self.framework_root}")
        self.adapter = Open3DBenchAdapter.detect(
            self.repository, self.framework_root / "adapters"
        )
        self.state_path = self.workflow_dir / "state.json"
        self.context_path = self.workflow_dir / "repository-context.json"
        self.ppa_path = self.workflow_dir / "global-ppa-report.json"
        configured_worktrees = config.worktrees_root or (
            Path(os.environ["AGENTIC3D_WORKTREES_ROOT"])
            if os.environ.get("AGENTIC3D_WORKTREES_ROOT")
            else self.workflow_dir.parent.parent / "worktrees"
        )
        self.worktrees_root = (
            configured_worktrees.expanduser().resolve() / self.workflow_dir.name
        )

    def _save_state(self, state: WorkflowState) -> None:
        state.updated_at = _now()
        _write_json(self.state_path, asdict(state))

    def _load_or_create(self, resume: bool) -> WorkflowState:
        if self.state_path.exists():
            if not resume:
                raise ValueError(
                    f"workflow already exists: {self.workflow_dir}; use --resume"
                )
            state = WorkflowState(
                **json.loads(self.state_path.read_text(encoding="utf-8"))
            )
            if state.user_task != self.config.user_task:
                raise ValueError("resume user_task does not match workflow state")
            if Path(state.repository).resolve() != self.repository:
                raise ValueError("resume repository does not match workflow state")
            if state.scope_mode != self.config.scope_mode:
                raise ValueError("resume scope_mode does not match workflow state")
            return state
        if resume:
            raise ValueError(f"cannot resume missing workflow: {self.workflow_dir}")
        self.workflow_dir.mkdir(parents=True, exist_ok=False)
        state = WorkflowState(
            workflow_id=self.workflow_dir.name,
            user_task=self.config.user_task,
            repository=str(self.repository),
            adapter=self.adapter.manifest.name,
            status="running",
            scope_mode=self.config.scope_mode,
        )
        self._save_state(state)
        return state

    def _analyzer_config(self, directory: Path, skill_name: str) -> AnalyzerConfig:
        return AnalyzerConfig(
            workspace=self.repository,
            analyzer_dir=directory,
            skill_path=self.framework_root / "skills" / skill_name / "SKILL.md",
            codex_binary=self.config.codex_binary,
            model=self.config.model,
            timeout_seconds=self.config.timeout_seconds,
            profile=self.config.profile,
            config_overrides=self.config.config_overrides,
            strict_config=self.config.strict_config,
        )

    def _global_analyzer(self) -> GlobalAnalyzer:
        return GlobalAnalyzer(
            self._analyzer_config(
                self.workflow_dir / "global-analyzer", "global-analyzer"
            )
        )

    def _local_analyzer(self, stage: str) -> LocalAnalyzer:
        return LocalAnalyzer(
            self._analyzer_config(
                self.workflow_dir / "local-analyzers" / stage, "local-analyzer"
            ),
            stage,
        )

    def _snapshot_stage_skill(self, stage: str, task_dir: Path) -> Path:
        names = {
            "partition": "optimize-mol-partition",
            "layout": "optimize-mol-layout",
            "place_opt": "optimize-openroad-place",
            "routing": "optimize-openroad-routing",
        }
        skill_root = self.framework_root / "skills"
        source_dir = skill_root / names[stage]
        target_dir = task_dir / "skill"
        task_dir.mkdir(parents=True, exist_ok=True)
        if not target_dir.exists():
            # Freeze helpers, tests, and prompts together with SKILL.md. This
            # lets Agentic3D live outside the target repository and makes a
            # resumed task independent from later framework updates.
            shutil.copytree(source_dir, target_dir)
            catalog = load_skill_catalog(skill_root)
            primary = skill_by_directory(catalog, names[stage])
            support_root = task_dir / "supporting-skills"
            for dependency in required_skills(catalog, primary.name):
                support = support_root / dependency.directory
                support.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(dependency.root, support)
        return target_dir / "SKILL.md"

    def _select_task_dir(
        self, state: WorkflowState, stage: str, worktree: Path, cycle: int
    ) -> Path:
        path = worktree / ".agentic3d" / "task-agent"
        state.active_tasks[stage] = str(path)
        return path

    def _select_direct_task_dir(
        self, state: WorkflowState, stage: str, cycle: int
    ) -> Path:
        path = (
            self.repository
            / ".agentic3d"
            / "direct"
            / self.workflow_dir.name
            / "task-agents"
            / stage
            / f"cycle-{cycle:03d}"
        )
        state.active_tasks[stage] = str(path)
        return path

    def _prepare_scoped_workspace(
        self,
        *,
        cycle: int,
        stage: str,
        cycle_dir: Path,
        local_feedback: Dict[str, Any],
    ) -> tuple[Path, Path, StageScope, Path]:
        if stage not in SCOPED_STAGES:
            raise ValueError(
                f"deterministic source mediation is currently enabled only for "
                f"{', '.join(SCOPED_STAGES)}; got {stage}"
            )
        metadata_path = cycle_dir / "worktree.json"
        scope_path = cycle_dir / "scope.json"
        context_path = cycle_dir / "candidate-repository-context.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            worktree = Path(metadata["path"]).expanduser().resolve()
            scope = StageScope.read(scope_path)
            if not worktree.is_dir():
                raise ValueError(f"persisted candidate worktree is missing: {worktree}")
            Open3DBenchAdapter.detect(
                worktree, self.framework_root / "adapters"
            ).write_context(context_path)
            return worktree, scope_path, scope, context_path

        manager = WorktreeManager(self.repository)
        baseline = manager.head()
        scope = ScopeCompiler.compile(
            stage=stage,
            baseline_commit=baseline,
            adapter_profile=self.adapter.stage(stage),
            proposal=local_feedback["scope_proposal"],
        )
        scope.write(scope_path)
        worktree = self.worktrees_root / f"cycle-{cycle:03d}-{stage}"
        manager.create(worktree, baseline=baseline)
        _write_json(
            metadata_path,
            {
                "path": str(worktree),
                "stage": stage,
                "cycle": cycle,
                "baseline_commit": baseline,
                "scope_contract": str(scope_path),
            },
        )
        Open3DBenchAdapter.detect(
            worktree, self.framework_root / "adapters"
        ).write_context(context_path)
        return worktree, scope_path, scope, context_path

    def _task_goal(self, stage: str) -> str:
        return (
            f"用户的全局任务：{self.config.user_task}\n"
            f"你的责任阶段：{stage}。完成 Global Analyzer 派发给该阶段的子目标，"
            "并为全局任务提交一个可审计的候选修改。你无权声明端到端的全局任务已经完成。"
            "以最新 handoff 和 `repository_context` 为本轮的权威约束，同时读取阶段 "
            "Skill、当前代码、持久状态、真实日志和结果文件。在当前 Open3DBench "
            "工作区中修改算法或配置，并完成编译、测评和验证。"
        )

    @staticmethod
    def _handoff_markdown(
        handoff: StageHandoff,
        global_plan: Dict[str, Any],
        local_feedback: Dict[str, Any],
    ) -> str:
        return (
            "# 结构化阶段交接\n\n"
            "本文件由 Agentic3DOrchestrator 自动生成。Task Agent 必须将其视为当前工作循环的执行契约。"
            "输入优先级依次为：用户目标与不可变约束、Global Plan 中的阶段与目标、Local Analyzer "
            "给出的假设与门槛、阶段 Skill 的修改边界、`repository_context` 中的版本路径、当前实测证据、"
            "历史状态。最新 handoff 中的路径优先于历史记录中的旧路径。\n\n"
            "## 执行契约\n\n```json\n"
            + json.dumps(handoff.to_dict(), ensure_ascii=False, indent=2)
            + "\n```\n\n## 全局计划\n\n```json\n"
            + json.dumps(global_plan, ensure_ascii=False, indent=2)
            + "\n```\n\n## 局部分析反馈\n\n```json\n"
            + json.dumps(local_feedback, ensure_ascii=False, indent=2)
            + "\n```\n"
        )

    def _run_task(
        self,
        state: WorkflowState,
        stage: str,
        handoff: StageHandoff,
        global_plan: Dict[str, Any],
        local_feedback: Dict[str, Any],
        worktree: Path,
        scope_path: Optional[Path],
    ):
        task_dir = (
            self._select_task_dir(state, stage, worktree, handoff.cycle)
            if scope_path is not None
            else self._select_direct_task_dir(state, stage, handoff.cycle)
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        skill = self._snapshot_stage_skill(stage, task_dir)
        (task_dir / "handoff.md").write_text(
            self._handoff_markdown(handoff, global_plan, local_feedback),
            encoding="utf-8",
        )
        resume = (task_dir / "state.json").exists()
        return TaskAgent(
            TaskAgentConfig(
                name=f"{stage} Task Agent",
                goal=self._task_goal(stage),
                workspace=worktree,
                task_dir=task_dir,
                skill_path=skill,
                codex_binary=self.config.codex_binary,
                model=self.config.model,
                timeout_seconds=self.config.timeout_seconds,
                max_rounds=self.config.task_max_rounds,
                stall_limit=self.config.task_stall_limit,
                profile=self.config.profile,
                config_overrides=self.config.config_overrides,
                strict_config=self.config.strict_config,
                scope_path=scope_path,
            )
        ).run(resume=resume)

    def _candidate_update(
        self,
        state: WorkflowState,
        cycle: int,
        stage: str,
        task_state: Any,
        worktree: Path,
        scope_path: Optional[Path],
        scope_report: Path,
    ) -> CandidateUpdate:
        round_results = [
            str(record["round_result_path"])
            for record in task_state.rounds
            if record.get("round_result_path")
        ]
        return CandidateUpdate(
            workflow_id=state.workflow_id,
            cycle=cycle,
            stage=stage,
            task_status=task_state.status,
            task_state=str(Path(state.active_tasks[stage]) / "state.json"),
            round_results=round_results,
            discovered_artifacts=Open3DBenchAdapter.detect(
                worktree, self.framework_root / "adapters"
            ).collect_artifacts(stage),
            scope_mode=self.config.scope_mode,
            worktree=str(worktree) if self.config.scope_mode == "enforced" else None,
            baseline_commit=task_state.baseline_commit,
            candidate_commit=task_state.candidate_commit,
            scope_contract=str(scope_path) if scope_path is not None else None,
            scope_report=str(scope_report),
        )

    def run(self, *, resume: bool = False) -> WorkflowState:
        state = self._load_or_create(resume)
        if state.status in {"completed", "blocked", "failed", "max_cycles"}:
            return state
        self.adapter.write_context(self.context_path)
        GlobalPPACollector(self.adapter).write(self.ppa_path)
        global_analyzer = self._global_analyzer()
        task_catalog = local_task_template_catalog(self.framework_root / "local_tasks")
        task_catalog_path = self.workflow_dir / "local-task-catalog.json"
        _write_json(task_catalog_path, task_catalog)

        if state.latest_global_plan is None:
            global_output = global_analyzer.analyze(
                {
                    "kind": "user_task_round_0",
                    "user_task": self.config.user_task,
                    "repository_context": str(self.context_path),
                    "local_task_catalog": task_catalog,
                    "candidate_update": None,
                    "global_ppa_report": str(self.ppa_path),
                }
            )
            modeled_tasks_from_global_analysis(
                global_output, self.framework_root / "local_tasks"
            )
            round0_path = self.workflow_dir / "global-plan-round-000.json"
            _write_json(round0_path, global_output)
            state.latest_global_plan = str(round0_path)
            state.current_stage = require_stage(
                global_output["global_plan"]["target_stage"]
            )
            if global_output["decision"]["status"] == "blocked":
                state.status = "blocked"
                state.terminal_reason = global_output["decision"]["reason"]
                self._save_state(state)
                return state
            self._save_state(state)

        while state.next_cycle < self.config.max_cycles:
            cycle = state.next_cycle
            stage = require_stage(str(state.current_stage))
            cycle_dir = self.workflow_dir / "cycles" / f"{cycle:03d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            self.adapter.write_context(self.context_path)
            GlobalPPACollector(self.adapter).write(self.ppa_path)
            global_output = json.loads(
                Path(str(state.latest_global_plan)).read_text(encoding="utf-8")
            )
            local = self._local_analyzer(stage)
            pre_feedback = local.analyze(
                {
                    "kind": "pre_task_local_analysis",
                    "user_task": self.config.user_task,
                    "stage": stage,
                    "repository_context": str(self.context_path),
                    "global_plan": state.latest_global_plan,
                    "candidate_update": None,
                    "global_ppa_report": str(self.ppa_path),
                }
            )
            pre_path = cycle_dir / "local-feedback-pre.json"
            _write_json(pre_path, pre_feedback)
            scope: Optional[StageScope]
            scope_path: Optional[Path]
            if self.config.scope_mode == "enforced":
                worktree, scope_path, scope, candidate_context_path = (
                    self._prepare_scoped_workspace(
                        cycle=cycle,
                        stage=stage,
                        cycle_dir=cycle_dir,
                        local_feedback=pre_feedback,
                    )
                )
            else:
                worktree = self.repository
                scope_path = None
                scope = None
                candidate_context_path = self.context_path
            handoff = StageHandoff(
                workflow_id=state.workflow_id,
                cycle=cycle,
                stage=stage,
                user_task=self.config.user_task,
                repository_context=str(candidate_context_path),
                global_plan=str(state.latest_global_plan),
                local_feedback=str(pre_path),
                scope_contract=str(scope_path) if scope_path is not None else None,
                accepted_inputs=list(
                    state.accepted_candidates.values()
                )
                + list(global_output["global_plan"].get("required_artifacts", [])),
                immutable_constraints=list(
                    global_output["global_plan"].get("frozen_constraints", [])
                ),
            )
            handoff_path = cycle_dir / "handoff.json"
            _write_json(handoff_path, handoff.to_dict())
            task_state = self._run_task(
                state,
                stage,
                handoff,
                global_output,
                pre_feedback,
                worktree,
                scope_path,
            )
            final_scope_report_path = cycle_dir / "candidate-scope-report.json"
            candidate_commit: Optional[str]
            if scope is not None:
                candidate_commit = task_state.candidate_commit or ScopeGuard(
                    worktree, scope
                ).head()
                task_state.candidate_commit = candidate_commit
                final_scope_report = ScopeGuard(worktree, scope).audit_candidate(
                    candidate_head=candidate_commit
                )
                final_scope_report.write(final_scope_report_path)
            else:
                candidate_commit = None
                _write_json(
                    final_scope_report_path,
                    {
                        "passed": True,
                        "mode": "direct",
                        "bypassed": True,
                        "stage": stage,
                        "reason": (
                            "scope_mode=direct: Task Agent modified the canonical "
                            "workspace without source-scope enforcement"
                        ),
                    },
                )
            candidate = self._candidate_update(
                state,
                cycle,
                stage,
                task_state,
                worktree,
                scope_path,
                final_scope_report_path,
            )
            candidate_path = cycle_dir / "candidate-update.json"
            _write_json(candidate_path, candidate.to_dict())
            candidate_adapter = Open3DBenchAdapter.detect(
                worktree, self.framework_root / "adapters"
            )
            candidate_adapter.write_context(candidate_context_path)
            candidate_ppa_path = cycle_dir / "candidate-global-ppa-report.json"
            GlobalPPACollector(candidate_adapter).write(candidate_ppa_path)
            post_feedback = local.analyze(
                {
                    "kind": "post_task_local_analysis",
                    "user_task": self.config.user_task,
                    "stage": stage,
                    "repository_context": str(candidate_context_path),
                    "global_plan": state.latest_global_plan,
                    "candidate_update": str(candidate_path),
                    "global_ppa_report": str(candidate_ppa_path),
                }
            )
            if scope is not None:
                post_scope = ScopeCompiler.compile(
                    stage=stage,
                    baseline_commit=scope.baseline_commit,
                    adapter_profile=self.adapter.stage(stage),
                    proposal=post_feedback["scope_proposal"],
                )
                if post_scope.computed_digest() != scope.computed_digest():
                    raise RuntimeError(
                        "post-task Local Analyzer attempted to change the executed scope"
                    )
            post_path = cycle_dir / "local-feedback-post.json"
            _write_json(post_path, post_feedback)
            global_next = global_analyzer.analyze(
                {
                    "kind": "candidate_update",
                    "user_task": self.config.user_task,
                    "repository_context": str(candidate_context_path),
                    "local_task_catalog": task_catalog,
                    "previous_global_plan": state.latest_global_plan,
                    "candidate_update": str(candidate_path),
                    "local_feedback": str(post_path),
                    "global_ppa_report": str(candidate_ppa_path),
                }
            )
            modeled_tasks_from_global_analysis(
                global_next, self.framework_root / "local_tasks"
            )
            global_path = cycle_dir / "global-plan-next.json"
            _write_json(global_path, global_next)
            state.cycles.append(
                {
                    "cycle": cycle,
                    "stage": stage,
                    "handoff": str(handoff_path),
                    "task_state": str(
                        Path(state.active_tasks[stage]) / "state.json"
                    ),
                    "candidate_update": str(candidate_path),
                    "local_feedback": str(post_path),
                    "global_plan": str(global_path),
                }
            )
            state.next_cycle += 1
            state.latest_global_plan = str(global_path)
            decision = global_next["decision"]
            if decision["candidate_status"] == "accept":
                if scope is not None and candidate_commit is not None:
                    integrated_commit = IntegrationGate(self.repository).integrate(
                        scope=scope,
                        worktree=worktree,
                        candidate_head=candidate_commit,
                        message=(
                            f"agentic3d({stage}): accept {state.workflow_id} "
                            f"cycle {cycle}"
                        ),
                    )
                    candidate_value = candidate.to_dict()
                    candidate_value["integrated_commit"] = integrated_commit
                    _write_json(candidate_path, candidate_value)
                state.accepted_candidates[stage] = str(candidate_path)
            if decision["status"] == "complete":
                state.status = "completed"
                state.terminal_reason = decision["reason"]
                self._save_state(state)
                return state
            if decision["status"] == "blocked":
                state.status = "blocked"
                state.terminal_reason = decision["reason"]
                self._save_state(state)
                return state
            state.current_stage = require_stage(
                global_next["global_plan"]["target_stage"]
            )
            self._save_state(state)

        state.status = "max_cycles"
        state.terminal_reason = f"reached max_cycles={self.config.max_cycles}"
        self._save_state(state)
        return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic3d-run",
        description=(
            "Run the code-defined Global Analyzer → Local Analyzer → Task Agent loop."
        ),
    )
    parser.add_argument("--repo", type=Path, required=True)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Round-0 user task command")
    task_group.add_argument(
        "--task-file",
        type=Path,
        help="UTF-8 file containing the round-0 user task command",
    )
    parser.add_argument("--workflow-dir", type=Path)
    parser.add_argument("--worktrees-root", type=Path)
    parser.add_argument(
        "--scope-mode",
        choices=("enforced", "direct"),
        default=os.environ.get("AGENTIC3D_SCOPE_MODE", "enforced"),
        help="enforced uses ScopeGuard/worktrees; direct lets Task Agents edit the repository",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--profile")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--task-max-rounds", type=int, default=3)
    parser.add_argument("--task-stall-limit", type=int, default=2)
    parser.add_argument("--strict-config", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repo.expanduser().resolve()
    framework_root = Path(__file__).resolve().parents[1]
    user_task = (
        args.task_file.expanduser().read_text(encoding="utf-8")
        if args.task_file is not None
        else args.task
    )
    workflow_dir = args.workflow_dir
    if workflow_dir is None:
        workflow_root = _repository_state_root(repository)
        workflow_dir = workflow_root / "workflows" / f"workflow-{uuid.uuid4().hex[:8]}"
    try:
        state = Agentic3DOrchestrator(
            WorkflowConfig(
                repository=repository,
                framework_root=framework_root,
                user_task=user_task,
                workflow_dir=workflow_dir,
                codex_binary=args.codex_bin,
                model=args.model,
                timeout_seconds=args.timeout,
                max_cycles=args.max_cycles,
                task_max_rounds=args.task_max_rounds,
                task_stall_limit=args.task_stall_limit,
                profile=args.profile,
                config_overrides=tuple(args.config),
                strict_config=args.strict_config,
                worktrees_root=args.worktrees_root,
                scope_mode=args.scope_mode,
            )
        ).run(resume=args.resume)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"kind": "workflow_error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"kind": "workflow_result", **asdict(state)}, ensure_ascii=False))
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
