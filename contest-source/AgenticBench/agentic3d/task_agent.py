"""Stateful, multi-round Task Agent built on top of ``codex exec``.

Planning and coding deliberately happen inside one Codex thread.  Each round
must persist its plan before editing, then compile/evaluate/validate the change
and return a structured decision that drives the next round.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .codex_exec import CodexExecConfig, CodexExecError, CodexRunResult, run_codex
from .scope_guard import ScopeGuard, StageScope


TERMINAL_STATUSES = {
    "completed",
    "failed",
    "max_rounds",
    "stalled",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class TaskAgentConfig:
    """Configuration shared by every round of one Task Agent."""

    name: str
    goal: str
    workspace: Path
    task_dir: Path
    skill_path: Path
    codex_binary: str = "codex"
    model: Optional[str] = None
    sandbox: str = "workspace-write"
    profile: Optional[str] = None
    config_overrides: Sequence[str] = field(default_factory=tuple)
    timeout_seconds: Optional[float] = None
    max_rounds: int = 3
    stall_limit: int = 2
    skip_git_repo_check: bool = False
    strict_config: bool = False
    scope_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.goal.strip():
            raise ValueError("goal must not be empty")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.stall_limit <= 0:
            raise ValueError("stall_limit must be positive")
        if self.sandbox != "workspace-write":
            raise ValueError("Task Agent requires the workspace-write sandbox")


@dataclass
class TaskAgentState:
    """Durable task state; this is the recovery source outside model context."""

    task_id: str
    name: str
    goal: str
    skill_path: str
    status: str
    created_at: str
    updated_at: str
    next_round: int = 1
    thread_id: Optional[str] = None
    stalled_rounds: int = 0
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    baseline_commit: Optional[str] = None
    candidate_commit: Optional[str] = None
    scope_contract: Optional[str] = None
    latest_scope_report: Optional[str] = None
    terminal_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TaskAgentState":
        return cls(**value)


ROUND_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["round", "plan", "execution", "decision"],
    "properties": {
        "round": {"type": "integer", "minimum": 1},
        "plan": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "analysis",
                "hypothesis",
                "components",
                "files_to_modify",
                "changes",
                "validation",
                "rollback",
            ],
            "properties": {
                "analysis": {"type": "string"},
                "hypothesis": {"type": "string"},
                "components": {"type": "array", "items": {"type": "string"}},
                "files_to_modify": {"type": "array", "items": {"type": "string"}},
                "changes": {"type": "array", "items": {"type": "string"}},
                "validation": {"type": "array", "items": {"type": "string"}},
                "rollback": {"type": "string"},
            },
        },
        "execution": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "summary",
                "files_changed",
                "commands_run",
                "validation_results",
            ],
            "properties": {
                "summary": {"type": "string"},
                "files_changed": {"type": "array", "items": {"type": "string"}},
                "commands_run": {"type": "array", "items": {"type": "string"}},
                "validation_results": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "passed", "evidence"],
                        "properties": {
                            "name": {"type": "string"},
                            "passed": {"type": "boolean"},
                            "evidence": {"type": "string"},
                        },
                    },
                },
            },
        },
        "decision": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "status",
                "reason",
                "progress",
                "goal_evidence",
                "next_focus",
            ],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["continue", "goal_achieved", "blocked"],
                },
                "reason": {"type": "string"},
                "progress": {
                    "type": "string",
                    "enum": ["improved", "neutral", "regressed", "unknown"],
                },
                "goal_evidence": {"type": "array", "items": {"type": "string"}},
                "next_focus": {"type": "string"},
            },
        },
    },
}


class TaskAgent:
    """Run plan/edit/evaluate rounds in a single persistent Codex thread."""

    def __init__(self, config: TaskAgentConfig) -> None:
        self.config = config
        self.workspace = config.workspace.expanduser().resolve()
        self.task_dir = config.task_dir.expanduser().resolve()
        self.skill_path = config.skill_path.expanduser().resolve()
        self.scope_path = (
            config.scope_path.expanduser().resolve()
            if config.scope_path is not None
            else None
        )
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {self.workspace}")
        if not self.skill_path.is_file():
            raise ValueError(f"skill file does not exist: {self.skill_path}")
        if not _is_within(self.task_dir, self.workspace):
            raise ValueError("task_dir must be inside workspace so Codex can persist plans")
        if not _is_within(self.skill_path, self.workspace):
            raise ValueError("skill_path must be inside workspace")
        self.scope: Optional[StageScope] = None
        self.scope_guard: Optional[ScopeGuard] = None
        if self.scope_path is not None:
            if not self.scope_path.is_file():
                raise ValueError(f"scope contract does not exist: {self.scope_path}")
            self.scope = StageScope.read(self.scope_path)
            self.scope_guard = ScopeGuard(self.workspace, self.scope)
        self.state_path = self.task_dir / "state.json"
        self.schema_path = self.task_dir / "round_output.schema.json"

    def _new_state(self) -> TaskAgentState:
        now = _utc_now()
        return TaskAgentState(
            task_id=self.task_dir.name,
            name=self.config.name,
            goal=self.config.goal,
            skill_path=str(self.skill_path),
            status="running",
            created_at=now,
            updated_at=now,
            baseline_commit=self.scope.baseline_commit if self.scope else None,
            candidate_commit=self.scope_guard.head() if self.scope_guard else None,
            scope_contract=str(self.scope_path) if self.scope_path else None,
        )

    def _load_or_create_state(self, resume: bool) -> TaskAgentState:
        if self.state_path.exists():
            if not resume:
                raise ValueError(
                    f"task already exists: {self.task_dir}; pass resume=True to continue"
                )
            state = TaskAgentState.from_dict(
                json.loads(self.state_path.read_text(encoding="utf-8"))
            )
            if state.name != self.config.name or state.goal != self.config.goal:
                raise ValueError("resume config name/goal does not match persisted task")
            if Path(state.skill_path).resolve() != self.skill_path:
                raise ValueError("resume skill does not match persisted task")
            if not self.schema_path.is_file():
                raise ValueError("persisted task is missing round_output.schema.json")
            last_codex = state.rounds[-1].get("codex", {}) if state.rounds else {}
            resumable_timeout = (
                state.status == "failed"
                and bool(last_codex.get("timed_out"))
                and state.thread_id is not None
            )
            if state.status in {"blocked", "interrupted"} or resumable_timeout:
                # Blocked/interrupted rounds are safe pauses, not irreversible
                # terminal states. A timed-out legacy state with a persisted
                # thread is migrated to the same resumable behavior.
                state.status = "running"
                state.terminal_reason = None
                self._save_state(state)
            if state.status in TERMINAL_STATUSES:
                return state
            return state

        if resume:
            raise ValueError(f"cannot resume missing task: {self.task_dir}")
        if self.task_dir.exists():
            # The orchestrator materializes the current handoff and an immutable
            # stage-skill snapshot before the first Codex call.  Permit only
            # those bootstrap files; any other pre-existing content still
            # protects the user from accidentally reusing a task directory.
            allowed_bootstrap = {
                "handoff.md",
                "skill.snapshot.md",
                "skill",
                "supporting-skills",
            }
            unexpected = {
                item.name for item in self.task_dir.iterdir()
            } - allowed_bootstrap
            if unexpected:
                raise ValueError(
                    "new task_dir contains unexpected files: "
                    + ", ".join(sorted(unexpected))
                )
        else:
            self.task_dir.mkdir(parents=True, exist_ok=False)
        self.schema_path.write_text(
            json.dumps(ROUND_OUTPUT_SCHEMA, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        state = self._new_state()
        _atomic_json(self.state_path, state.to_dict())
        return state

    def _round_prompt(
        self, state: TaskAgentState, round_number: int, plan_path: Path
    ) -> str:
        prior = state.rounds[-1] if state.rounds else None
        prior_text = (
            json.dumps(prior, ensure_ascii=False, indent=2)
            if prior
            else "这是第一轮，没有上一轮结果。"
        )
        skill_relative = self.skill_path.relative_to(self.workspace)
        supporting_root = self.task_dir / "supporting-skills"
        supporting_skills = sorted(supporting_root.glob("*/SKILL.md"))
        supporting_text = (
            "\n".join(
                "- {}".format(path.relative_to(self.workspace))
                for path in supporting_skills
            )
            if supporting_skills
            else "- 本次直接调用未附带额外基础设施 Skill。"
        )
        plan_relative = plan_path.relative_to(self.workspace)
        state_relative = self.state_path.relative_to(self.workspace)
        handoff_path = self.task_dir / "handoff.md"
        handoff_text = (
            handoff_path.read_text(encoding="utf-8")
            if handoff_path.is_file()
            else "本轮没有新的外部执行 handoff。"
        )
        scope_text = (
            json.dumps(self.scope.to_dict(), ensure_ascii=False, indent=2)
            if self.scope is not None
            else "本次直接调用没有启用确定性源码 scope。"
        )
        return f"""你是 {self.config.name}，一个在同一上下文中连续完成分析、规划、编码和验证的 Task Agent。
当前是同一阶段任务的第 {round_number} 轮。不要把 Planner 与 Coding 拆成不同角色、子任务、Agent 或会话。

任务目标：
{self.config.goal}

必须遵守的工程 Skill：{skill_relative}
由编排器冻结的基础设施 Skills（按本轮需要读取和调用）：
{supporting_text}
持久状态：{state_relative}

上一轮记录：
{prior_text}

当前阶段 handoff（其中的新路径、实验标识和指标覆盖历史记录中的过时值，但不能改变用户目标）：
{handoff_text}

机器强制执行的源码 scope（你的声明不能扩大它）：
{scope_text}

输入与决策优先级（前者约束后者）：
1. 用户的全局目标和不可违反的约束。
2. Global Plan 选择的阶段、PPA 目标（`ppa_objective`）、成功标准（`success_criteria`）和冻结输入。
3. Local Analyzer 在 handoff 中给出的单一假设（`single_hypothesis`）、允许修改范围（`allowed_changes`）和验收门槛。
4. 阶段 Skill 定义的修改边界（action space）和正确性门槛，以及基础设施 Skills 固化的环境检查、构建、输入、评估和结果读取流程。
5. `repository_context` 提供的当前版本真实路径、命令和产物位置。
6. 当前代码、日志、结果和验证报告；这些内容构成本轮的事实依据。
7. 持久状态和上一轮记录；与最新 handoff 冲突时，以最新 handoff 为准。

本轮必须严格按以下顺序工作：
1. 完整读取 handoff、阶段 Skill、`repository_context`、当前代码、已有日志和结果、持久状态，以及上一轮证据。优先执行 Local Analyzer 给出的 `single_hypothesis`；可以细化实现机制，但不得静默替换该假设。如果证据表明假设不可执行，须明确记录原因，并返回重新分析所需的证据。
2. 在修改任何源码或配置之前，创建 `{plan_relative}`。计划必须写明：现状分析、假设、本轮要修改的部件/文件、预期指标变化、编译步骤、测评步骤、正确性验证和回滚条件。最终 JSON 的 `plan.files_to_modify` 必须使用 repository-relative 精确路径，并完整声明本轮实际改动的所有源码和配置文件。
3. 计划落盘后才能开始修改。只在阶段 Skill 和机器 scope 的交集内完成一组连贯的小改动，不得夹带其他阶段的修改。不得自行执行 git commit、reset、clean、checkout、merge、rebase 或修改 Git 历史；这些操作只由 ScopeGuard 和 IntegrationGate 执行。
4. 按照阶段 Skill 和 adapter 提供的命令完成编译、测评及正确性检查；保存实际命令、代码差异、日志、指标和证据路径。验证失败、产物缺失或基线与候选不可比时，不得宣称阶段候选已经完成。
5. 最终严格按照输出 schema 返回本轮 JSON。这里的 `goal_achieved` 仅表示“本阶段候选达到 handoff 中的 `success_criteria`，且全部 `validation_results` 均通过”，不表示全局用户任务已经完成；未达到该条件时使用 `continue`。
6. 先完成阶段 Skill 和 `repository_context` 中所有可由本 Agent 安全执行的步骤。只有确实缺少权限、必要输入或不可替代的外部条件时，才能使用 `blocked`。此时须在 `reason` 和 `next_focus` 中写明缺失项、责任方、已经尝试的步骤及明确的恢复条件；不得以笼统的“需要远程评测”作为阻塞理由。

即使你在上下文中已经有方案，也必须先把本轮 plan 写入指定路径，再执行修改。
"""

    @staticmethod
    def _parse_round_output(result: CodexRunResult, round_number: int) -> Dict[str, Any]:
        try:
            output = json.loads(result.final_message)
        except json.JSONDecodeError as exc:
            raise ValueError(f"round {round_number} returned invalid JSON: {exc}") from exc
        if not isinstance(output, dict):
            raise ValueError(f"round {round_number} output must be an object")
        if output.get("round") != round_number:
            raise ValueError(f"round output says {output.get('round')!r}, expected {round_number}")
        decision = output.get("decision")
        execution = output.get("execution")
        if not isinstance(decision, dict) or decision.get("status") not in {
            "continue",
            "goal_achieved",
            "blocked",
        }:
            raise ValueError(f"round {round_number} has an invalid decision")
        validations = execution.get("validation_results") if isinstance(execution, dict) else None
        if not isinstance(validations, list) or not validations:
            raise ValueError(f"round {round_number} has no validation results")
        if any(not isinstance(item, dict) or "passed" not in item for item in validations):
            raise ValueError(f"round {round_number} has malformed validation results")
        return output

    def _save_state(self, state: TaskAgentState) -> None:
        state.updated_at = _utc_now()
        _atomic_json(self.state_path, state.to_dict())

    def run(self, *, resume: bool = False) -> TaskAgentState:
        state = self._load_or_create_state(resume)
        if state.status in TERMINAL_STATUSES:
            return state

        while state.next_round <= self.config.max_rounds:
            round_number = state.next_round
            round_root = self.task_dir / "rounds" / f"{round_number:03d}"
            if round_root.exists():
                # Preserve an interrupted attempt for audit and retry the same
                # logical round without deleting user/agent artifacts.
                round_root = round_root.with_name(
                    f"{round_number:03d}-retry-{uuid.uuid4().hex[:6]}"
                )
            round_root.mkdir(parents=True, exist_ok=False)
            plan_path = round_root / "plan.md"
            codex_run_dir = round_root / "codex"
            prompt = self._round_prompt(state, round_number, plan_path)
            round_head = self.scope_guard.head() if self.scope_guard else None

            def capture_thread(event: Dict[str, Any]) -> None:
                emitted = event.get("thread_id") if event.get("type") == "thread.started" else None
                if emitted and state.thread_id is None:
                    # Save immediately so an interrupted Codex process can be
                    # resumed instead of silently losing its reasoning context.
                    state.thread_id = str(emitted)
                    self._save_state(state)

            try:
                result = run_codex(
                    prompt,
                    CodexExecConfig(
                        workspace=self.workspace,
                        run_dir=codex_run_dir,
                        codex_binary=self.config.codex_binary,
                        model=self.config.model,
                        sandbox=self.config.sandbox,
                        profile=self.config.profile,
                        config_overrides=self.config.config_overrides,
                        output_schema=self.schema_path,
                        timeout_seconds=self.config.timeout_seconds,
                        skip_git_repo_check=self.config.skip_git_repo_check,
                        strict_config=self.config.strict_config,
                        resume_thread_id=state.thread_id,
                    ),
                    on_event=capture_thread,
                )
            except (CodexExecError, OSError, ValueError) as exc:
                state.status = "failed"
                state.terminal_reason = f"round {round_number} launch failed: {exc}"
                self._save_state(state)
                return state

            if result.thread_id:
                if state.thread_id and result.thread_id != state.thread_id:
                    state.status = "failed"
                    state.terminal_reason = "Codex resumed into a different thread"
                    self._save_state(state)
                    return state
                state.thread_id = result.thread_id

            record: Dict[str, Any] = {
                "round": round_number,
                "plan_path": str(plan_path),
                "codex": result.to_dict(),
            }
            state.rounds.append(record)
            state.next_round += 1

            if not result.success:
                if result.timed_out and state.thread_id:
                    state.status = "interrupted"
                    state.terminal_reason = (
                        f"round {round_number} timed out; resume the persisted thread"
                    )
                else:
                    state.status = "failed"
                    state.terminal_reason = (
                        f"round {round_number} Codex failed with exit {result.exit_code}"
                    )
                self._save_state(state)
                return state
            if not plan_path.is_file() or not plan_path.read_text(encoding="utf-8").strip():
                state.status = "failed"
                state.terminal_reason = (
                    f"round {round_number} modified/executed without a persisted plan"
                )
                self._save_state(state)
                return state

            try:
                output = self._parse_round_output(result, round_number)
            except ValueError as exc:
                state.status = "failed"
                state.terminal_reason = str(exc)
                self._save_state(state)
                return state

            output_path = round_root / "round_result.json"
            _atomic_json(output_path, output)
            record["round_result_path"] = str(output_path)
            record["decision"] = output["decision"]

            if self.scope_guard is not None and round_head is not None:
                report = self.scope_guard.audit(
                    expected_head=round_head,
                    declared_files=output["plan"].get("files_to_modify", []),
                )
                report_path = round_root / "scope-report.json"
                report.write(report_path)
                record["scope_report"] = str(report_path)
                state.latest_scope_report = str(report_path)
                if not report.passed:
                    self.scope_guard.rollback(expected_head=round_head)
                    record["decision"] = {
                        "status": "scope_rejected",
                        "reason": "deterministic source scope audit failed",
                        "violations": report.violations,
                    }
                    state.stalled_rounds += 1
                    if state.stalled_rounds >= self.config.stall_limit:
                        state.status = "stalled"
                        state.terminal_reason = (
                            "deterministic scope audit rejected "
                            f"{state.stalled_rounds} consecutive rounds"
                        )
                        self._save_state(state)
                        return state
                    self._save_state(state)
                    continue
                checkpoint = self.scope_guard.checkpoint(
                    message=(
                        f"agentic3d({self.scope.stage}): "
                        f"{state.task_id} round {round_number}"
                    )
                )
                state.candidate_commit = checkpoint or self.scope_guard.head()
                record["candidate_commit"] = state.candidate_commit

            decision = output["decision"]
            validations_passed = all(
                item["passed"] for item in output["execution"]["validation_results"]
            )
            if decision["progress"] == "improved":
                state.stalled_rounds = 0
            else:
                state.stalled_rounds += 1

            if (
                decision["status"] == "goal_achieved"
                and validations_passed
                and bool(decision["goal_evidence"])
            ):
                state.status = "completed"
                state.terminal_reason = decision["reason"]
                self._save_state(state)
                return state
            if decision["status"] == "blocked":
                state.status = "blocked"
                state.terminal_reason = decision["reason"]
                self._save_state(state)
                return state
            if state.thread_id is None:
                state.status = "failed"
                state.terminal_reason = (
                    "Codex did not emit a thread id; refusing to continue in a new context"
                )
                self._save_state(state)
                return state
            if state.stalled_rounds >= self.config.stall_limit:
                state.status = "stalled"
                state.terminal_reason = (
                    f"no reported improvement for {state.stalled_rounds} consecutive rounds"
                )
                self._save_state(state)
                return state

            self._save_state(state)

        state.status = "max_rounds"
        state.terminal_reason = f"reached max_rounds={self.config.max_rounds}"
        self._save_state(state)
        return state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic3d-task-agent",
        description="Run one stateful Task Agent through explicit plan/code/eval rounds.",
    )
    parser.add_argument("--name", required=True, help="Task Agent/stage name")
    parser.add_argument("--goal", required=True, help="Measurable task goal")
    parser.add_argument("--skill", type=Path, required=True, help="Stage SKILL.md")
    parser.add_argument("-C", "--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--scope", type=Path, help="Signed StageScope JSON")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--profile")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--stall-limit", type=int, default=2)
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument("--strict-config", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    task_dir = args.task_dir
    if task_dir is None:
        task_id = f"{args.name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
        artifact_root = os.environ.get("AGENTIC3D_RUNS_ROOT")
        task_root = (
            Path(artifact_root).expanduser().resolve()
            if artifact_root
            else workspace / ".agentic3d"
        )
        task_dir = task_root / "tasks" / task_id
    try:
        state = TaskAgent(
            TaskAgentConfig(
                name=args.name,
                goal=args.goal,
                workspace=workspace,
                task_dir=task_dir,
                skill_path=args.skill,
                codex_binary=args.codex_bin,
                model=args.model,
                profile=args.profile,
                config_overrides=tuple(args.config),
                timeout_seconds=args.timeout,
                max_rounds=args.max_rounds,
                stall_limit=args.stall_limit,
                skip_git_repo_check=args.skip_git_repo_check,
                strict_config=args.strict_config,
                scope_path=args.scope,
            )
        ).run(resume=args.resume)
    except (ValueError, OSError) as exc:
        print(json.dumps({"kind": "task_error", "error": str(exc)}), file=sys.stderr)
        return 2

    print(json.dumps({"kind": "task_result", **state.to_dict()}, ensure_ascii=False))
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
