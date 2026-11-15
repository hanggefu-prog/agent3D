"""Stateful Global and Local Analyzer implementations."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .codex_exec import CodexExecConfig, CodexExecError, run_codex
from .contracts import (
    GLOBAL_ANALYZER_SCHEMA,
    LOCAL_ANALYZER_SCHEMA,
    LOCAL_TASK_TEMPLATES,
    require_stage,
)


from local_bench.task_inventory import EXECUTABLE_TASK_STAGES

_TEMPLATE_STAGES = EXECUTABLE_TASK_STAGES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_object_reference(value: Any, label: str) -> Dict[str, Any]:
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return loaded


def _require_candidate_scope_pass(candidate: Dict[str, Any]) -> None:
    report = _read_object_reference(
        candidate.get("scope_report"), "candidate scope_report"
    )
    mode = candidate.get("scope_mode", "enforced")
    if mode == "direct":
        if report.get("mode") != "direct" or report.get("bypassed") is not True:
            raise RuntimeError("direct candidate is missing an explicit scope bypass report")
        return
    if mode != "enforced" or report.get("passed") is not True:
        raise RuntimeError("candidate did not pass deterministic source scope audit")


@dataclass(frozen=True)
class AnalyzerConfig:
    workspace: Path
    analyzer_dir: Path
    skill_path: Path
    codex_binary: str = "codex"
    model: Optional[str] = None
    timeout_seconds: Optional[float] = None
    profile: Optional[str] = None
    config_overrides: Sequence[str] = ()
    strict_config: bool = False


class StatefulAnalyzer:
    role = "Analyzer"
    schema: Dict[str, Any]

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.workspace = config.workspace.expanduser().resolve()
        self.analyzer_dir = config.analyzer_dir.expanduser().resolve()
        self.skill_path = config.skill_path.expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {self.workspace}")
        if not self.skill_path.is_file():
            raise ValueError(f"analyzer skill is missing: {self.skill_path}")
        # Analyzer state may live on the dedicated results disk.  The analyzer
        # itself remains read-only against ``workspace``; its trace directory
        # is framework-owned and need not pollute the source checkout.
        self.analyzer_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.analyzer_dir / "state.json"
        self.schema_path = self.analyzer_dir / "output.schema.json"
        # The schema is generated framework state, not user evidence. Refresh it
        # on construction so a resumed workflow picks up contract migrations
        # such as phase-specific Local Analyzer decisions.
        _write_json(self.schema_path, self.schema)

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        state = {
            "role": self.role,
            "thread_id": None,
            "next_round": 0,
            "runs": [],
            "created_at": _now(),
            "updated_at": _now(),
        }
        _write_json(self.state_path, state)
        return state

    def _prompt(self, payload: Dict[str, Any], round_number: int) -> str:
        raise NotImplementedError

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._load_state()
        round_number = int(state["next_round"])
        run_name = f"{round_number:03d}"
        run_dir = self.analyzer_dir / "runs" / run_name
        if run_dir.exists():
            run_dir = run_dir.with_name(f"{run_name}-retry-{uuid.uuid4().hex[:6]}")
        input_path = run_dir.parent / f"{run_dir.name}-input.json"
        _write_json(input_path, payload)
        prompt = self._prompt(payload, round_number)
        def capture_thread(event: Dict[str, Any]) -> None:
            emitted = (
                event.get("thread_id")
                if event.get("type") == "thread.started"
                else None
            )
            if emitted and state.get("thread_id") is None:
                state["thread_id"] = str(emitted)
                state["updated_at"] = _now()
                _write_json(self.state_path, state)

        try:
            result = run_codex(
                prompt,
                CodexExecConfig(
                    workspace=self.workspace,
                    run_dir=run_dir,
                    codex_binary=self.config.codex_binary,
                    model=self.config.model,
                    sandbox="read-only",
                    profile=self.config.profile,
                    config_overrides=self.config.config_overrides,
                    output_schema=self.schema_path,
                    timeout_seconds=self.config.timeout_seconds,
                    strict_config=self.config.strict_config,
                    resume_thread_id=state.get("thread_id"),
                ),
                on_event=capture_thread,
            )
        except (CodexExecError, OSError, ValueError) as exc:
            raise RuntimeError(f"{self.role} launch failed: {exc}") from exc
        if not result.success:
            raise RuntimeError(
                f"{self.role} round {round_number} failed with exit {result.exit_code}"
            )
        if state.get("thread_id") and result.thread_id != state["thread_id"]:
            raise RuntimeError(f"{self.role} resumed into a different Codex thread")
        output = json.loads(result.final_message)
        if not isinstance(output, dict):
            raise RuntimeError(f"{self.role} returned a non-object result")
        self._validate(output, round_number, payload)
        output_path = run_dir / "analysis.json"
        _write_json(output_path, output)
        state["thread_id"] = result.thread_id or state.get("thread_id")
        state["next_round"] = round_number + 1
        state["runs"].append(
            {
                "round": round_number,
                "input": str(input_path),
                "output": str(output_path),
                "codex": result.to_dict(),
            }
        )
        state["updated_at"] = _now()
        _write_json(self.state_path, state)
        return output

    def _validate(
        self, output: Dict[str, Any], round_number: int, payload: Dict[str, Any]
    ) -> None:
        raise NotImplementedError


class GlobalAnalyzer(StatefulAnalyzer):
    """Diagnose global PPA, model local tasks, and select one dispatch stage."""

    role = "Global Analyzer"
    schema = GLOBAL_ANALYZER_SCHEMA

    def _prompt(self, payload: Dict[str, Any], round_number: int) -> str:
        skill = self.skill_path.read_text(encoding="utf-8")
        first_round = round_number == 0
        round_context = (
            "这是第 0 轮：解析 orchestrator 传入的 `user_task`，并建立可测量的全局任务定义。"
            if first_round
            else "这是候选结果回灌后的全局复盘：审计候选，再决定下一步。"
        )
        return f"""你是 Agentic3D 的 Global Analyzer，负责“全局诊断 → 阶段选择”（Diagnose → Select）。你始终只读，不修改代码、配置、实验产物或工作流状态。
当前是全局分析第 {round_number} 轮。{round_context}

Global Analyzer Skill：
---BEGIN SKILL---
{skill}
---END SKILL---

结构化输入：
{json.dumps(payload, ensure_ascii=False, indent=2)}

工作要求：
1. 完整读取 `repository_context` 指向的版本适配信息，以及输入中实际存在的全局 PPA、阶段日志、候选更新（`candidate_update`）、Local Analyzer 反馈和正确性证据。文件路径本身不是证据；必须读取文件内容后再作判断。
2. 第 0 轮以 `user_task` 为最高层目标，在结构化输出中明确重述 PPA 目标（`ppa_objective`）、不可违反的约束、可测量的验收标准，以及尚缺少的基线（baseline）。不得虚构尚未测得的结果。
3. 诊断 `partition`、`layout`、`place_opt`、`routing` 四个阶段的瓶颈。当 `decision.status=dispatch` 时，只选择一个 `target_stage`，并用真实证据说明理由；证据不足时，只派发建立可信基线所需的最小工作。
4. Global Analyzer 同时是局部任务建模器。以结构化输入中的 `local_task_catalog` 为唯一模板清单；派发时必须在 `global_plan.local_tasks` 中把本轮工作建模成一到多个局部 benchmark 任务，明确唯一可证伪假设、局部目标及权重、源码/配置 scope、局部和正确性门槛、预算以及需要读取的 artifact。所有任务必须属于当前 `target_stage`。`place_opt` 当前支持 HBT assignment policy；只能派发清单中对应阶段的模板，不得推断已开放其他 OPT 修改范围。
5. 只给出行动提示和局部任务定义，不提供代码补丁。在一次下游候选评估期间，冻结已经接受的上游输入。如果证据表明必须返回上游，须显式派发新的工作循环（cycle），并说明哪些下游候选会因此失效。`source_scope` 和 `config_scope` 必须是所选模板原始 scope 的子集，不能借任务建模扩大修改权限。
6. 只有端到端 PPA 指标和全部正确性证据共同证明用户目标已经达成时，`decision.status` 才能设为 `complete`。只有在外部条件无法由任何阶段安全解决时，才能设为 `blocked`；其余情况均设为 `dispatch`。
   适配器中的 `preparation_commands` 属于可派发的正常准备工作。生成目录尚不存在、尚未执行源码转换脚本，或 `optional_tools` 缺失，都不能单独构成 `blocked`；尤其不要把名为 `binaries/converted_input` 的生成数据目录误判为必须下载二进制工具包。
   如果新版统一流程可以复用原 Place-LoL benchmark，优先派发一次性链接、挂载或复制准备；不得仅因新版目录下没有重新解压 benchmark archive 而判定外部阻塞。
7. 第 0 轮必须设置 `decision.candidate_status=not_applicable`。后续轮次必须设置为 `accept`、`reject` 或 `pending`。只有成功标准（`success_criteria`）、正确性门槛、基线与候选的可比性，以及预先声明的复现实验检查全部通过，才可设置为 `accept`。
8. 当 `decision.status` 不是 `dispatch` 时，`global_plan.target_stage` 只表示证据所属或最相关的阶段，不构成新的阶段派发，并且 `global_plan.local_tasks` 必须为空。
9. 严格返回符合输出 schema 的 JSON；不要输出 Markdown，也不要添加 JSON 之外的解释。
"""

    def _validate(
        self, output: Dict[str, Any], round_number: int, payload: Dict[str, Any]
    ) -> None:
        if output.get("round") != round_number:
            raise RuntimeError(
                f"Global Analyzer returned round {output.get('round')}, expected {round_number}"
            )
        decision_object = output.get("decision", {})
        decision = decision_object.get("status")
        candidate_status = decision_object.get("candidate_status")
        target = output.get("global_plan", {}).get("target_stage")
        local_tasks = output.get("global_plan", {}).get("local_tasks")
        if decision not in {"dispatch", "complete", "blocked"}:
            raise RuntimeError("Global Analyzer returned an invalid decision")
        if candidate_status not in {
            "not_applicable",
            "accept",
            "reject",
            "pending",
        }:
            raise RuntimeError("Global Analyzer returned an invalid candidate status")
        if round_number == 0 and candidate_status != "not_applicable":
            raise RuntimeError("round-0 Global Analyzer cannot judge a candidate")
        if candidate_status == "accept":
            candidate = _read_object_reference(
                payload.get("candidate_update"), "accepted candidate_update"
            )
            if candidate.get("task_status") != "completed":
                raise RuntimeError(
                    "Global Analyzer cannot accept a candidate whose Task Agent "
                    "did not complete its stage contract"
                )
            _require_candidate_scope_pass(candidate)
            local_feedback = _read_object_reference(
                payload.get("local_feedback"), "accepted local_feedback"
            )
            if (
                local_feedback.get("analysis_phase") != "post_task"
                or local_feedback.get("decision") != "accept"
            ):
                raise RuntimeError(
                    "Global Analyzer cannot accept a candidate without a "
                    "post-task Local Analyzer acceptance"
                )
        require_stage(str(target))
        if not isinstance(local_tasks, list):
            raise RuntimeError("Global Analyzer omitted structured local_tasks")
        if decision == "dispatch" and not local_tasks:
            raise RuntimeError("a dispatch decision must model at least one local task")
        if decision != "dispatch" and local_tasks:
            raise RuntimeError("a non-dispatch decision cannot model new local tasks")
        task_ids = [str(item.get("task_id", "")) for item in local_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise RuntimeError("Global Analyzer returned duplicate local task IDs")
        for task in local_tasks:
            template = str(task.get("template", ""))
            if template not in LOCAL_TASK_TEMPLATES:
                raise RuntimeError("Global Analyzer selected an unknown local task template")
            modeled_stage = str(task.get("stage", ""))
            if modeled_stage != target or _TEMPLATE_STAGES[template] != modeled_stage:
                raise RuntimeError(
                    "modeled local task template/stage does not match target_stage"
                )
            for scope_name in ("source_scope", "config_scope"):
                for path in task.get(scope_name, []):
                    value = str(path)
                    if not value or value.startswith("/") or ".." in value.split("/"):
                        raise RuntimeError(
                            f"Global Analyzer returned an unsafe {scope_name} path"
                        )
        if round_number == 0 and decision == "complete":
            raise RuntimeError("Global Analyzer cannot complete before dispatching round 0")


class LocalAnalyzer(StatefulAnalyzer):
    """Define a stage hypothesis before execution and audit it afterward."""

    role = "Local Analyzer"
    schema = LOCAL_ANALYZER_SCHEMA

    def __init__(self, config: AnalyzerConfig, stage: str) -> None:
        self.stage = require_stage(stage)
        super().__init__(config)

    def _prompt(self, payload: Dict[str, Any], round_number: int) -> str:
        skill = self.skill_path.read_text(encoding="utf-8")
        kind = str(payload.get("kind", ""))
        if kind == "pre_task_local_analysis":
            phase = "pre_task"
            phase_requirements = """这是执行前分析。诊断可信基线，提出本轮唯一且可证伪的假设，并预先声明允许修改的范围、指标门槛、正确性门槛、复现规则和回滚条件。必须在 `scope_proposal` 中使用 repository-relative 精确路径；它只是提案，随后还会与 adapter 硬边界取交集。`decision` 只能是 `recommend` 或 `need_evidence`。"""
        elif kind == "post_task_local_analysis":
            phase = "post_task"
            phase_requirements = """这是执行后审计。核对 Task Agent 实际执行的假设、代码差异、日志、结果和验证证据，并读取候选的确定性 `scope_report`。`scope_proposal` 必须复述本次执行前范围，不能在执行后扩大。`single_hypothesis` 必须复述本次被审计的原假设，不得另行提出下一轮假设。`decision` 只能是 `accept`、`reject` 或 `need_evidence`。"""
        else:
            raise ValueError(f"unsupported Local Analyzer payload kind: {kind!r}")
        return f"""你是 Agentic3D 的 {self.stage} Local Analyzer，负责“局部监测 → 证据指导”（Monitor → Guide）。你始终只读，不得修改代码、配置、日志或实验产物，也不得启动另一个 Agent。
当前是该阶段第 {round_number} 次局部分析，analysis_phase={phase}。
{phase_requirements}

Local Analyzer Skill：
---BEGIN SKILL---
{skill}
---END SKILL---

结构化输入：
{json.dumps(payload, ensure_ascii=False, indent=2)}

工作要求：
1. 首先读取 `repository_context` 中 `{self.stage}` 阶段的源码与配置边界、日志根目录、结果根目录、指标、正确性门槛和版本适配命令；再读取 `global_plan`、已有阶段产物，以及输入中存在的 Task Agent 结果。
2. 明确区分实测数据、合成测试、重建数据、用户目标和缺失证据。引用路径不是验证结论，必须检查文件内容。
3. 始终只处理 Global Analyzer 选择的 `{self.stage}` 阶段，禁止夹带其他阶段的修改。`allowed_changes` 必须是一组共同服务于同一假设、彼此关联的小改动。`scope_proposal.source_paths` 和 `config_paths` 必须来自 repository context 中该阶段 adapter 路径或其子路径；不得用自然语言、绝对路径、`..` 或其他阶段路径。默认禁止新增、删除、重命名、mode change 和 symlink，只有假设确实需要且证据充分时才可单独授权。
4. `acceptance_gates` 必须能够通过真实日志或结果文件验证，并与 `global_plan.success_criteria` 一致；同时列出 `correctness_gates`、`replay_rule` 和 `rollback_rule`。
5. 缺少可信基线、关键产物或可比性证据时，必须设置 `decision=need_evidence`，不得宣称已有改进；同时明确缺少哪些文件、建议 Task Agent 执行哪条 adapter 命令，以及恢复判断所需的条件。
6. 输出中的 `analysis_phase` 必须是 `{phase}`。严格返回符合输出 schema 的 JSON；不要输出 Markdown，也不要添加 JSON 之外的解释。
"""

    def _validate(
        self, output: Dict[str, Any], round_number: int, payload: Dict[str, Any]
    ) -> None:
        if output.get("stage") != self.stage:
            raise RuntimeError(
                f"Local Analyzer returned stage {output.get('stage')!r}, "
                f"expected {self.stage!r}"
            )
        kind = payload.get("kind")
        expected_phase = {
            "pre_task_local_analysis": "pre_task",
            "post_task_local_analysis": "post_task",
        }.get(kind)
        if output.get("analysis_phase") != expected_phase:
            raise RuntimeError(
                "Local Analyzer returned an analysis phase inconsistent with its input"
            )
        allowed_decisions = (
            {"recommend", "need_evidence"}
            if expected_phase == "pre_task"
            else {"accept", "reject", "need_evidence"}
        )
        if output.get("decision") not in allowed_decisions:
            raise RuntimeError(
                f"Local Analyzer returned an invalid {expected_phase} decision"
            )
        proposal = output.get("scope_proposal")
        if not isinstance(proposal, dict):
            raise RuntimeError("Local Analyzer did not return a structured scope proposal")
        proposed_paths = list(proposal.get("source_paths", [])) + list(
            proposal.get("config_paths", [])
        )
        if expected_phase == "pre_task" and not proposed_paths:
            raise RuntimeError("pre-task Local Analyzer returned an empty scope proposal")
        if expected_phase == "post_task" and output.get("decision") == "accept":
            candidate = _read_object_reference(
                payload.get("candidate_update"), "accepted candidate_update"
            )
            if candidate.get("task_status") != "completed":
                raise RuntimeError(
                    "Local Analyzer cannot accept an incomplete Task Agent candidate"
                )
            _require_candidate_scope_pass(candidate)
        if not str(output.get("single_hypothesis", "")).strip():
            raise RuntimeError("Local Analyzer omitted its single hypothesis")
