"""Hierarchical task modeling and correction aggregation for Agentic3D.

The Global Analyzer models bounded local benchmark tasks.  Each local task may
use any ask/tell search backend.  Locally accepted corrections are then merged
by a deterministic outer ask step and evaluated once by a whole-flow PPA
evaluator, whose normalized feedback is returned through tell.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from local_bench.model import TaskSpec, load_task, validate_candidate_payload
from local_bench.policy import flatten_paths, patch_paths
from local_bench.search_api import (
    AskTellRunner,
    CandidateFeedback,
    CandidateProposal,
    ProposalEvaluator,
    SearchBackend,
    SearchContext,
    SearchProblem,
    SearchRunResult,
)

from .contracts import LOCAL_TASK_TEMPLATES, require_stage


from local_bench.task_inventory import EXECUTABLE_TASK_STAGES

_TEMPLATE_STAGE = EXECUTABLE_TASK_STAGES


def local_task_template_catalog(template_root: Path) -> Dict[str, Any]:
    """Publish the exact maintained task scopes available to Global Analyzer."""

    root = template_root.expanduser().resolve()
    catalog: Dict[str, Any] = {}
    for template in LOCAL_TASK_TEMPLATES:
        spec = load_task(root / template)
        catalog[template] = {
            "template": template,
            "stage": _TEMPLATE_STAGE[template],
            "task_id": spec.task_id,
            "kind": spec.kind,
            "benchmark_input": spec.benchmark_input_id,
            "allowed_source_paths": list(spec.allowed_source_paths),
            "allowed_config_paths": list(spec.allowed_config_paths),
            "candidate_files": {
                item.path: list(item.allowed_paths) for item in spec.candidate_files
            },
            "program_mode": spec.program_mode,
            "program_entrypoint": spec.program_entrypoint,
        }
    return catalog


def _safe_relative(value: str, label: str) -> str:
    path = PurePosixPath(str(value))
    if not str(path) or path.is_absolute() or ".." in path.parts:
        raise ValueError("{} must be a safe repository-relative path: {}".format(label, value))
    return path.as_posix()


def _json_mapping(value: Mapping[str, Any], label: str) -> Dict[str, Any]:
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must contain finite JSON values: {}".format(label, error)) from error
    loaded = json.loads(encoded)
    if not isinstance(loaded, dict):
        raise ValueError("{} must be a JSON object".format(label))
    return loaded


@dataclass(frozen=True)
class GlobalModelingContext:
    """Read-only evidence supplied to the Global Analyzer task modeler."""

    user_command: str
    global_round: int
    global_ppa_report: Mapping[str, Any]
    accepted_revision: str
    remaining_budget: int
    prior_rounds: Tuple[Mapping[str, Any], ...] = ()
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_command.strip():
            raise ValueError("user_command must not be empty")
        if self.global_round < 0:
            raise ValueError("global_round cannot be negative")
        if not self.accepted_revision.strip():
            raise ValueError("accepted_revision must not be empty")
        if self.remaining_budget < 0:
            raise ValueError("remaining_budget cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_command": self.user_command,
            "global_round": self.global_round,
            "global_ppa_report": _json_mapping(
                self.global_ppa_report, "global_ppa_report"
            ),
            "accepted_revision": self.accepted_revision,
            "remaining_budget": self.remaining_budget,
            "prior_rounds": [dict(item) for item in self.prior_rounds],
            "artifacts": _json_mapping(self.artifacts, "artifacts"),
        }


@dataclass(frozen=True)
class StageTask:
    """One Global-Analyzer-modeled instance of a maintained local task."""

    task_id: str
    template: str
    template_dir: Path
    stage: str
    global_round: int
    hypothesis: str
    objective: Mapping[str, Any]
    acceptance_gates: Tuple[str, ...]
    correctness_gates: Tuple[str, ...]
    evaluation_budget: int
    source_scope: Tuple[str, ...]
    config_scope: Tuple[str, ...]
    context_artifacts: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in self.task_id
        ):
            raise ValueError("task_id contains unsupported characters")
        if self.template not in LOCAL_TASK_TEMPLATES:
            raise ValueError("unknown local task template: {}".format(self.template))
        require_stage(self.stage)
        if _TEMPLATE_STAGE[self.template] != self.stage:
            raise ValueError("template {} does not belong to stage {}".format(self.template, self.stage))
        if self.global_round < 0:
            raise ValueError("global_round cannot be negative")
        if not self.hypothesis.strip():
            raise ValueError("StageTask hypothesis must not be empty")
        if self.evaluation_budget < 1:
            raise ValueError("evaluation_budget must be positive")
        _json_mapping(self.objective, "objective")
        for path in self.source_scope:
            _safe_relative(path, "source_scope")
        for path in self.config_scope:
            parts = path.split(".")
            if not parts or any(not part or part in {".", ".."} for part in parts):
                raise ValueError("config_scope must be a non-empty dotted path")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "template": self.template,
            "template_dir": str(self.template_dir.expanduser().resolve()),
            "stage": self.stage,
            "global_round": self.global_round,
            "hypothesis": self.hypothesis,
            "objective": _json_mapping(self.objective, "objective"),
            "acceptance_gates": list(self.acceptance_gates),
            "correctness_gates": list(self.correctness_gates),
            "evaluation_budget": self.evaluation_budget,
            "source_scope": list(self.source_scope),
            "config_scope": list(self.config_scope),
            "context_artifacts": list(self.context_artifacts),
        }


def modeled_tasks_from_global_analysis(
    analysis: Mapping[str, Any],
    template_root: Path,
) -> Tuple[StageTask, ...]:
    """Decode the Global Analyzer's structured ``global_plan.local_tasks``."""

    plan = analysis.get("global_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("global analysis is missing global_plan")
    rows = plan.get("local_tasks")
    if not isinstance(rows, list):
        raise ValueError("global_plan.local_tasks must be an array")
    round_number = int(analysis.get("round", -1))
    root = template_root.expanduser().resolve()
    tasks = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every modeled local task must be an object")
        template = str(row.get("template", ""))
        template_dir = (root / template).resolve()
        if root not in template_dir.parents or not template_dir.is_dir():
            raise ValueError("modeled task template is unavailable: {}".format(template))
        task = StageTask(
            task_id=str(row.get("task_id", "")),
            template=template,
            template_dir=template_dir,
            stage=str(row.get("stage", "")),
            global_round=round_number,
            hypothesis=str(row.get("hypothesis", "")),
            objective=dict(row.get("objective", {})),
            acceptance_gates=tuple(
                str(item) for item in row.get("acceptance_gates", [])
            ),
            correctness_gates=tuple(
                str(item) for item in row.get("correctness_gates", [])
            ),
            evaluation_budget=int(row.get("evaluation_budget", 0)),
            source_scope=tuple(str(item) for item in row.get("source_scope", [])),
            config_scope=tuple(str(item) for item in row.get("config_scope", [])),
            context_artifacts=tuple(
                str(item) for item in row.get("context_artifacts", [])
            ),
        )
        _validate_stage_task_scope(task)
        tasks.append(task)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("modeled local task IDs must be unique")
    return tuple(tasks)


def _validate_stage_task_scope(task: StageTask) -> TaskSpec:
    template = load_task(task.template_dir)
    if task.template_dir.name != task.template:
        raise ValueError("template name and directory do not match")
    extra_sources = sorted(set(task.source_scope) - set(template.allowed_source_paths))
    extra_config = sorted(set(task.config_scope) - set(template.allowed_config_paths))
    if extra_sources or extra_config:
        raise ValueError(
            "modeled task expands template scope: source={} config={}".format(
                extra_sources, extra_config
            )
        )
    return template


class StageTaskMaterializer:
    """Turn a modeled task overlay into a normal loadable local task directory."""

    def materialize(self, task: StageTask, output_dir: Path) -> TaskSpec:
        template = _validate_stage_task_scope(task)
        source_scope = set(task.source_scope)
        config_scope = set(task.config_scope)

        destination = output_dir.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=False)
        raw = yaml.safe_load((task.template_dir / "task.yaml").read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("template task.yaml must contain an object")
        data = copy.deepcopy(raw)
        data["id"] = task.task_id
        data["title"] = "{} [global round {}]".format(template.title, task.global_round)
        data["description"] = task.hypothesis
        data["base_config"] = "base_config.yaml"
        data["initial_candidate"] = "initial_candidate"
        data["prompt"] = "prompt.md"

        shutil.copy2(template.base_config, destination / "base_config.yaml")
        shutil.copytree(template.initial_candidate, destination / "initial_candidate")
        policy = data.setdefault("candidate_policy", {})
        policy["allowed_source_paths"] = list(task.source_scope)
        policy["allowed_config_paths"] = list(task.config_scope)
        prompt_context = data.setdefault("prompt_context", {})
        prompt_context["source_ranges"] = [
            row
            for row in prompt_context.get("source_ranges", [])
            if str(row.get("path", "")) in source_scope
        ]
        program = data.get("program")
        if isinstance(program, dict) and program.get("mode") == "task_adapter":
            assert template.program_template is not None
            assert template.program_adapter is not None
            shutil.copy2(template.program_template, destination / "initial_program.py")
            shutil.copy2(template.program_adapter, destination / "program_adapter.py")
            program["template"] = "initial_program.py"
            program["adapter"] = "program_adapter.py"

        modeled = json.dumps(task.to_dict(), ensure_ascii=False, indent=2)
        template_prompt = template.prompt.read_text(encoding="utf-8").strip()
        (destination / "prompt.md").write_text(
            "# Global Analyzer modeled task\n\n"
            "This run-local task narrows a maintained benchmark template. The "
            "structured model below is authoritative for the hypothesis, objective, "
            "budget, and acceptance gates. It cannot expand the template policy.\n\n"
            "```json\n{}\n```\n\n# Maintained template instructions\n\n{}\n".format(
                modeled, template_prompt
            ),
            encoding="utf-8",
        )
        (destination / "stage_task.json").write_text(modeled + "\n", encoding="utf-8")
        (destination / "task.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        generated = load_task(destination)
        if set(generated.allowed_source_paths) != source_scope:
            raise RuntimeError("materialized source scope changed unexpectedly")
        if set(generated.allowed_config_paths) != config_scope:
            raise RuntimeError("materialized config scope changed unexpectedly")
        return generated


@dataclass(frozen=True)
class LocalCorrectionResult:
    """Locally accepted result returned by one inner ask/tell loop."""

    correction_id: str
    task: StageTask
    base_revision: str
    candidate: Mapping[str, Any]
    feedback: CandidateFeedback
    verification: Mapping[str, Any]
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    accepted: bool = True

    def __post_init__(self) -> None:
        if not self.correction_id.strip():
            raise ValueError("correction_id must not be empty")
        if not self.base_revision.strip():
            raise ValueError("base_revision must not be empty")
        validate_candidate_payload(self.candidate)
        if self.accepted:
            if not self.feedback.valid:
                raise ValueError("an accepted local correction requires valid feedback")
            if self.feedback.proposal_id != self.correction_id:
                raise ValueError("local feedback proposal_id must match correction_id")
            if self.verification.get("passed") is not True:
                raise ValueError("an accepted local correction requires verifier pass evidence")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "correction_id": self.correction_id,
            "task": self.task.to_dict(),
            "base_revision": self.base_revision,
            "candidate": validate_candidate_payload(self.candidate),
            "feedback": self.feedback.to_dict(),
            "verification": _json_mapping(self.verification, "verification"),
            "artifacts": _json_mapping(self.artifacts, "artifacts"),
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class IntegratedProposal:
    """Conflict-checked aggregate submitted to one whole-flow PPA evaluation."""

    proposal_id: str
    base_revision: str
    selected_corrections: Tuple[str, ...]
    candidate: Mapping[str, Any]
    source_paths: Tuple[str, ...]
    config_paths: Tuple[str, ...]
    provenance: Tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.proposal_id.strip() or not self.base_revision.strip():
            raise ValueError("integrated proposal ID and base revision must be non-empty")
        if len(self.selected_corrections) != len(set(self.selected_corrections)):
            raise ValueError("integrated proposal correction IDs must be unique")
        candidate = validate_candidate_payload(self.candidate)
        actual_sources = tuple(sorted(patch_paths(candidate.get("source_patch", ""))))
        actual_config = tuple(
            sorted(flatten_paths(candidate.get("config_overrides", {})))
        )
        if actual_sources != tuple(sorted(self.source_paths)):
            raise ValueError("integrated proposal source manifest does not match candidate")
        if actual_config != tuple(sorted(self.config_paths)):
            raise ValueError("integrated proposal config manifest does not match candidate")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "proposal_id": self.proposal_id,
            "base_revision": self.base_revision,
            "selected_corrections": list(self.selected_corrections),
            "candidate": validate_candidate_payload(self.candidate),
            "source_paths": list(self.source_paths),
            "config_paths": list(self.config_paths),
            "provenance": [dict(item) for item in self.provenance],
        }

    def to_program(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_program(cls, program: str) -> "IntegratedProposal":
        value = json.loads(program)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("integrated proposal must be a schema_version 1 object")
        return cls(
            proposal_id=str(value["proposal_id"]),
            base_revision=str(value["base_revision"]),
            selected_corrections=tuple(str(item) for item in value["selected_corrections"]),
            candidate=validate_candidate_payload(value["candidate"]),
            source_paths=tuple(str(item) for item in value["source_paths"]),
            config_paths=tuple(str(item) for item in value["config_paths"]),
            provenance=tuple(dict(item) for item in value["provenance"]),
        )


def _merge_mapping(target: Dict[str, Any], incoming: Mapping[str, Any], prefix: str) -> None:
    for key, value in incoming.items():
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, Mapping):
            _merge_mapping(target[key], value, path)
        elif target[key] != value:
            raise ValueError("local corrections conflict at {}".format(path))


def aggregate_corrections(
    corrections: Sequence[LocalCorrectionResult], proposal_id: str
) -> IntegratedProposal:
    selected = [item for item in corrections if item.accepted]
    if not selected:
        raise ValueError("no locally accepted corrections are available to aggregate")
    revisions = {item.base_revision for item in selected}
    if len(revisions) != 1:
        raise ValueError("local corrections do not share one base revision")
    correction_ids = [item.correction_id for item in selected]
    if len(correction_ids) != len(set(correction_ids)):
        raise ValueError("local correction IDs must be unique")

    config: Dict[str, Any] = {}
    candidate_files: Dict[str, Any] = {}
    patches = []
    owners: Dict[str, str] = {}
    provenance = []
    for correction in selected:
        candidate = validate_candidate_payload(correction.candidate)
        paths = patch_paths(candidate.get("source_patch", ""))
        outside = sorted(set(paths) - set(correction.task.source_scope))
        if outside:
            raise ValueError(
                "correction {} changes source outside its modeled scope: {}".format(
                    correction.correction_id, ", ".join(outside)
                )
            )
        for path in paths:
            if path in owners:
                raise ValueError(
                    "source patch conflict on {} between {} and {}".format(
                        path, owners[path], correction.correction_id
                    )
                )
            owners[path] = correction.correction_id
        changed_config = flatten_paths(candidate.get("config_overrides", {}))
        rejected_config = sorted(
            path
            for path in changed_config
            if not any(
                path == allowed or path.startswith(allowed + ".")
                for allowed in correction.task.config_scope
            )
        )
        if rejected_config:
            raise ValueError(
                "correction {} changes config outside its modeled scope: {}".format(
                    correction.correction_id, ", ".join(rejected_config)
                )
            )
        _merge_mapping(config, candidate.get("config_overrides", {}), "config_overrides")
        _merge_mapping(candidate_files, candidate.get("candidate_files", {}), "candidate_files")
        if candidate.get("source_patch", "").strip():
            patches.append(candidate["source_patch"].rstrip())
        provenance.append(
            {
                "correction_id": correction.correction_id,
                "task_id": correction.task.task_id,
                "template": correction.task.template,
                "local_score": correction.feedback.score,
                "program_sha256": correction.feedback.program_sha256,
            }
        )

    candidate = validate_candidate_payload(
        {
            "schema_version": 1,
            "config_overrides": config,
            "candidate_files": candidate_files,
            "source_patch": "\n".join(patches) + ("\n" if patches else ""),
            "notes": "Aggregated locally accepted corrections: {}".format(
                ", ".join(correction_ids)
            ),
        }
    )
    return IntegratedProposal(
        proposal_id=proposal_id,
        base_revision=next(iter(revisions)),
        selected_corrections=tuple(correction_ids),
        candidate=candidate,
        source_paths=tuple(sorted(owners)),
        config_paths=tuple(sorted(flatten_paths(config))),
        provenance=tuple(provenance),
    )


class CorrectionAggregationBackend:
    """One-shot outer SearchBackend whose ask aggregates local corrections."""

    name = "local-correction-aggregator"

    def __init__(
        self,
        corrections: Sequence[LocalCorrectionResult],
        *,
        proposal_id: str,
    ) -> None:
        self.corrections = tuple(corrections)
        self.proposal_id = proposal_id
        self.integrated: Optional[IntegratedProposal] = None
        self.global_feedback: Optional[CandidateFeedback] = None
        self._asked = False

    def ask(self, context: SearchContext) -> Sequence[CandidateProposal]:
        if self._asked:
            raise RuntimeError("correction aggregation may ask only once")
        self._asked = True
        self.integrated = aggregate_corrections(self.corrections, self.proposal_id)
        return [
            CandidateProposal(
                proposal_id=self.proposal_id,
                program=self.integrated.to_program(),
                metadata={
                    "proposal_kind": "integrated_local_corrections",
                    "base_revision": self.integrated.base_revision,
                    "selected_corrections": list(self.integrated.selected_corrections),
                },
            )
        ]

    def tell(self, feedback: Sequence[CandidateFeedback]) -> None:
        if len(feedback) != 1 or feedback[0].proposal_id != self.proposal_id:
            raise ValueError("outer tell must return exactly the aggregated proposal feedback")
        self.global_feedback = feedback[0]

    def done(self) -> bool:
        return self.global_feedback is not None


class IntegratedPPAEvaluator:
    """Adapt one whole-flow runner to the standard ProposalEvaluator boundary.

    ``run_full_flow`` receives a decoded, conflict-checked IntegratedProposal
    and returns a mapping with ``metrics`` and optional ``artifacts``, ``score``,
    and ``status``.  Exceptions become failed feedback instead of bypassing the
    outer tell step.
    """

    def __init__(
        self,
        run_full_flow: Callable[[IntegratedProposal], Mapping[str, Any]],
    ) -> None:
        self.run_full_flow = run_full_flow

    def evaluate_proposal(self, proposal: CandidateProposal) -> CandidateFeedback:
        started = time.monotonic()
        digest = hashlib.sha256(proposal.program.encode("utf-8")).hexdigest()
        try:
            integrated = IntegratedProposal.from_program(proposal.program)
            if integrated.proposal_id != proposal.proposal_id:
                raise ValueError("integrated proposal ID does not match CandidateProposal")
            result = self.run_full_flow(integrated)
            if not isinstance(result, Mapping):
                raise TypeError("whole-flow evaluator must return a mapping")
            metrics = dict(result.get("metrics", {}))
            artifacts = dict(result.get("artifacts", {}))
            raw_score = result.get("score", metrics.get("combined_score"))
            successful = bool(metrics.get("runs_successfully", True))
            score = float(raw_score) if raw_score is not None and successful else None
            status = str(
                result.get("status", "accepted" if score is not None else "failed")
            )
            if status not in {"accepted", "failed", "rejected", "cached"}:
                raise ValueError("whole-flow evaluator returned an invalid status")
            if status in {"failed", "rejected"}:
                score = None
            elif score is None:
                status = "failed"
            return CandidateFeedback(
                proposal_id=proposal.proposal_id,
                program_sha256=digest,
                status=status,
                score=score,
                metrics=metrics,
                artifacts=artifacts,
                metadata=dict(proposal.metadata),
                error_type=result.get("error_type"),
                error_message=result.get("error_message"),
                runtime_seconds=float(
                    result.get("runtime_seconds", time.monotonic() - started)
                ),
            )
        except Exception as error:
            return CandidateFeedback(
                proposal_id=proposal.proposal_id,
                program_sha256=digest,
                status="failed",
                score=None,
                metadata=dict(proposal.metadata),
                error_type=type(error).__name__,
                error_message=str(error),
                runtime_seconds=time.monotonic() - started,
            )


@dataclass(frozen=True)
class GlobalRoundResult:
    context: GlobalModelingContext
    tasks: Tuple[StageTask, ...]
    corrections: Tuple[LocalCorrectionResult, ...]
    outer_search: SearchRunResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "context": self.context.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
            "corrections": [item.to_dict() for item in self.corrections],
            "outer_search": self.outer_search.to_dict(),
        }


class HierarchicalRoundCoordinator:
    """Fix one Global-model → local loops → aggregate ask → PPA tell round."""

    def run_round(
        self,
        context: GlobalModelingContext,
        task_modeler: Callable[[GlobalModelingContext], Sequence[StageTask]],
        local_optimizer: Callable[[StageTask], LocalCorrectionResult],
        global_evaluator: ProposalEvaluator,
    ) -> GlobalRoundResult:
        tasks = tuple(task_modeler(context))
        if not tasks:
            raise ValueError("Global Analyzer task modeling returned no local tasks")
        if any(task.global_round != context.global_round for task in tasks):
            raise ValueError("modeled task global_round does not match its context")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("modeled task IDs must be unique within a global round")
        # ``remaining_budget`` counts authoritative outer whole-flow PPA/STA
        # evaluations.  Each StageTask owns an independent, cheaper inner
        # evaluation budget; summing those two currencies would prevent
        # first-improvement local search from doing its intended work.
        if context.remaining_budget < 1:
            raise ValueError("a hierarchical round requires one remaining global evaluation")

        corrections = tuple(local_optimizer(task) for task in tasks)
        if any(item.task.task_id != task.task_id for item, task in zip(corrections, tasks)):
            raise ValueError("local optimizer returned a correction for the wrong StageTask")
        proposal_id = "global-round-{:03d}".format(context.global_round)
        backend: SearchBackend = CorrectionAggregationBackend(
            corrections, proposal_id=proposal_id
        )
        problem = SearchProblem(
            task_id=proposal_id,
            description=(
                "Aggregate locally accepted corrections and evaluate whole-flow PPA "
                "for: {}".format(context.user_command)
            ),
            initial_program="{}",
            metadata={"global_round": context.global_round},
        )
        outer = AskTellRunner().run(
            backend,
            global_evaluator,
            problem,
            max_evaluations=1,
        )
        return GlobalRoundResult(
            context=context,
            tasks=tasks,
            corrections=corrections,
            outer_search=outer,
        )
