"""Framework-neutral ask/tell search API for OpenAgentic3DBench.

Search algorithms own candidate generation and population state.  The benchmark
owns program materialization, verification, isolated Open3DBench execution,
metric extraction, caching, and durable history.  Keeping those responsibilities
separate lets EvoStage, OpenEvolve, and user-defined agents share one objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from .evaluation import evaluate_task
from .model import ROUTING_SOURCE_TASK_KINDS, TaskSpec, load_task
from .program_bridge import materialize_program
from .verifier import verify_candidate


@dataclass(frozen=True)
class SearchProblem:
    """Immutable problem information supplied to every search backend."""

    task_id: str
    description: str
    initial_program: str
    maximize: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    system_message: str = ""
    language: str = "python"
    response_format: str = "complete_program"
    max_program_chars: Optional[int] = None
    prompt_templates: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateProposal:
    """One complete candidate program returned by ``SearchBackend.ask``."""

    proposal_id: str
    program: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.proposal_id.strip():
            raise ValueError("proposal_id must be non-empty")
        if not isinstance(self.program, str) or not self.program.strip():
            raise ValueError("candidate program must be non-empty text")


@dataclass(frozen=True)
class CandidateFeedback:
    """One evaluation returned to ``SearchBackend.tell``."""

    proposal_id: str
    program_sha256: str
    status: str
    score: Optional[float]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    runtime_seconds: float = 0.0

    @property
    def valid(self) -> bool:
        return (isinstance(self.score, (int, float)) and not isinstance(self.score, bool)
                and math.isfinite(self.score) and self.status in {"accepted", "cached"})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "program_sha256": self.program_sha256,
            "status": self.status,
            "score": self.score,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(frozen=True)
class SearchContext:
    """Read-only state passed to one ``ask`` call."""

    problem: SearchProblem
    evaluations: int
    remaining_budget: int
    history: Tuple[CandidateFeedback, ...]
    incumbent: Optional[CandidateFeedback]


@runtime_checkable
class SearchBackend(Protocol):
    """Minimal interface implemented by a search framework or agent system."""

    def ask(self, context: SearchContext) -> Sequence[CandidateProposal]:
        ...

    def tell(self, feedback: Sequence[CandidateFeedback]) -> None:
        ...

@runtime_checkable
class ProposalEvaluator(Protocol):
    """Evaluation side of the ask/tell boundary."""

    def evaluate_proposal(self, proposal: CandidateProposal) -> CandidateFeedback:
        ...


@dataclass(frozen=True)
class SearchRunResult:
    backend: str
    stop_reason: str
    budget: int
    history: Tuple[CandidateFeedback, ...]
    incumbent: Optional[CandidateFeedback]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "backend": self.backend,
            "stop_reason": self.stop_reason,
            "budget": self.budget,
            "evaluations": len(self.history),
            "incumbent": self.incumbent.to_dict() if self.incumbent else None,
            "history": [item.to_dict() for item in self.history],
        }


def _incumbent(
    history: Sequence[CandidateFeedback], maximize: bool
) -> Optional[CandidateFeedback]:
    valid = [item for item in history if item.valid]
    if not valid:
        return None
    key = lambda item: float(item.score)  # type: ignore[arg-type,return-value]
    return (max if maximize else min)(valid, key=key)


def _backend_name(backend) -> str:
    return getattr(backend, "name", type(backend).__name__)


def _backend_done(backend) -> bool:
    # Legacy convenience, never required of a two-method backend.
    done = getattr(backend, "done", None)
    return bool(done()) if callable(done) else False


class AskTellRunner:
    """Drive any ``SearchBackend`` against any ``ProposalEvaluator``."""

    def run(
        self,
        backend: SearchBackend,
        evaluator: ProposalEvaluator,
        problem: SearchProblem,
        *,
        max_evaluations: int,
        feedback_transform: Optional[Callable[[CandidateFeedback], CandidateFeedback]] = None,
    ) -> SearchRunResult:
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        history: list[CandidateFeedback] = []
        proposal_ids: set[str] = set()
        stop_reason = "budget_exhausted"

        while len(history) < max_evaluations:
            if _backend_done(backend):
                stop_reason = "backend_done"
                break
            context = SearchContext(
                problem=problem,
                evaluations=len(history),
                remaining_budget=max_evaluations - len(history),
                history=tuple(history),
                incumbent=_incumbent(history, problem.maximize),
            )
            proposals = tuple(backend.ask(context))
            if not proposals:
                stop_reason = "backend_done"
                break
            if len(proposals) > context.remaining_budget:
                raise ValueError(
                    "SearchBackend.ask returned {} proposals with only {} evaluations "
                    "remaining".format(len(proposals), context.remaining_budget)
                )
            batch_ids = [proposal.proposal_id for proposal in proposals]
            duplicates = sorted(
                proposal_id
                for proposal_id in set(batch_ids)
                if batch_ids.count(proposal_id) > 1 or proposal_id in proposal_ids
            )
            if duplicates:
                raise ValueError(
                    "proposal_id values must be unique for the run: {}".format(
                        ", ".join(duplicates)
                    )
                )
            proposal_ids.update(batch_ids)

            feedback = tuple(
                evaluator.evaluate_proposal(proposal) for proposal in proposals
            )
            for proposal, result in zip(proposals, feedback):
                if result.proposal_id != proposal.proposal_id:
                    raise ValueError(
                        "evaluator returned feedback for {!r}, expected {!r}".format(
                            result.proposal_id, proposal.proposal_id
                        )
                    )
                expected_sha = hashlib.sha256(proposal.program.encode("utf-8")).hexdigest()
                if result.program_sha256 != expected_sha:
                    raise ValueError("evaluator returned feedback for a different program")
                if result.status in {"accepted", "cached"} and not result.valid:
                    raise ValueError("accepted feedback requires a finite score")
                if result.status not in {"accepted", "cached", "rejected", "failed"}:
                    raise ValueError("unknown evaluation status")
                if result.status in {"rejected", "failed"} and result.score is not None:
                    raise ValueError("failed/rejected feedback must have score=None")
            if feedback_transform is not None:
                feedback = tuple(feedback_transform(item) for item in feedback)
            history.extend(feedback)
            backend.tell(feedback)

        return SearchRunResult(
            backend=_backend_name(backend),
            stop_reason=stop_reason,
            budget=max_evaluations,
            history=tuple(history),
            incumbent=_incumbent(history, problem.maximize),
        )


class FirstImprovementRunner:
    """Run an inner search until the first strict improvement is observed.

    The supplied incumbent normally comes from the currently accepted local
    stage implementation.  Evaluations performed here are local/proxy
    evaluations and are deliberately accounted separately from an outer
    whole-flow STA/PPA budget.
    """

    def run(
        self,
        backend: SearchBackend,
        evaluator: ProposalEvaluator,
        problem: SearchProblem,
        *,
        incumbent: CandidateFeedback,
        max_evaluations: int,
        min_improvement: float = 0.0,
    ) -> SearchRunResult:
        if not incumbent.valid:
            raise ValueError("first-improvement incumbent must be valid")
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        if min_improvement < 0.0:
            raise ValueError("min_improvement cannot be negative")

        history: list[CandidateFeedback] = []
        proposal_ids: set[str] = set()
        stop_reason = "inner_budget_exhausted"
        accepted = incumbent

        def improved(candidate: CandidateFeedback) -> bool:
            if not candidate.valid:
                return False
            assert candidate.score is not None and accepted.score is not None
            if problem.maximize:
                return candidate.score > accepted.score + min_improvement
            return candidate.score < accepted.score - min_improvement

        while len(history) < max_evaluations:
            if _backend_done(backend):
                stop_reason = "backend_done"
                break
            context = SearchContext(
                problem=problem,
                evaluations=len(history),
                remaining_budget=max_evaluations - len(history),
                history=tuple(history),
                incumbent=accepted,
            )
            proposals = tuple(backend.ask(context))
            if not proposals:
                stop_reason = "backend_done"
                break
            if len(proposals) > context.remaining_budget:
                raise ValueError(
                    "SearchBackend.ask returned {} proposals with only {} inner "
                    "evaluations remaining".format(
                        len(proposals), context.remaining_budget
                    )
                )
            for proposal in proposals:
                if proposal.proposal_id in proposal_ids:
                    raise ValueError(
                        "proposal_id values must be unique for the run: {}".format(
                            proposal.proposal_id
                        )
                    )
                proposal_ids.add(proposal.proposal_id)
                feedback = evaluator.evaluate_proposal(proposal)
                if feedback.proposal_id != proposal.proposal_id:
                    raise ValueError(
                        "evaluator returned feedback for {!r}, expected {!r}".format(
                            feedback.proposal_id, proposal.proposal_id
                        )
                    )
                history.append(feedback)
                backend.tell((feedback,))
                if improved(feedback):
                    accepted = feedback
                    stop_reason = "first_improvement"
                    return SearchRunResult(
                        backend=_backend_name(backend),
                        stop_reason=stop_reason,
                        budget=max_evaluations,
                        history=tuple(history),
                        incumbent=accepted,
                    )

        return SearchRunResult(
            backend=_backend_name(backend),
            stop_reason=stop_reason,
            budget=max_evaluations,
            history=tuple(history),
            incumbent=accepted,
        )


class CallbackSearchBackend:
    """Small adapter for frameworks that already expose equivalent callbacks."""

    def __init__(
        self,
        name: str,
        ask: Callable[[SearchContext], Sequence[CandidateProposal]],
        tell: Callable[[Sequence[CandidateFeedback]], None],
        done: Optional[Callable[[], bool]] = None,
    ):
        if not name.strip():
            raise ValueError("backend name must be non-empty")
        self._name = name
        self._ask = ask
        self._tell = tell
        self._done = done or (lambda: False)

    @property
    def name(self) -> str:
        return self._name

    def ask(self, context: SearchContext) -> Sequence[CandidateProposal]:
        return self._ask(context)

    def tell(self, feedback: Sequence[CandidateFeedback]) -> None:
        self._tell(feedback)

    def done(self) -> bool:
        return bool(self._done())


@dataclass(frozen=True)
class TaskEvaluationOptions:
    """Framework-independent configuration for program evaluation."""

    task_dir: Path
    output_dir: Path
    evaluation_mode: str = "full"
    open3dbench_root: Optional[Path] = None
    baseline_run: Optional[Path] = None
    reference_run: Optional[Path] = None
    base_config: Optional[Path] = None
    compact_evaluations: bool = False
    timeout_seconds: int = 21600
    evaluation_matrix: Optional[Path] = None
    matrix_workers: int = 1


class ProgramEvaluationHarness:
    """Materialize, evaluate, cache, and record complete candidate programs."""

    def __init__(self, options: Any, *, evaluation_transport: Optional[Callable[..., Any]] = None,
                 cache_results: bool = True):
        if options.evaluation_mode not in {"verify", "interface", "full"}:
            raise ValueError("evaluation_mode must be verify, interface, or full")
        self.options = options
        # A trusted launcher may move physical execution into a pinned Docker
        # image. Materialization, policy checks and durable history stay here.
        self._evaluation_transport = evaluation_transport
        self._cache_results = cache_results
        self.spec: TaskSpec = load_task(options.task_dir)
        self.matrix = None
        if getattr(options, "evaluation_matrix", None) is not None:
            from .matrix import LocalEvaluationMatrix
            if options.evaluation_mode != "full":
                raise ValueError("Matrix evaluation requires full mode")
            if any(getattr(options, k, None) is not None for k in ("baseline_run", "reference_run", "base_config")):
                raise ValueError("Use evaluation_matrix or single-cell paths, not both")
            self.matrix = LocalEvaluationMatrix(options.evaluation_matrix, self.spec)
        if options.evaluation_mode != "verify":
            from .model import require_executable_task
            require_executable_task(self.spec)
        config_path = self.spec.root / "config.yaml"
        self._max_program_chars = None
        if config_path.is_file():
            import yaml
            self._max_program_chars = (yaml.safe_load(config_path.read_text()) or {}).get("max_code_length")
        if (
            options.evaluation_mode == "full"
            and self.spec.kind in ROUTING_SOURCE_TASK_KINDS
            and options.reference_run is None
            and self.matrix is None
        ):
            raise ValueError(
                "full routing source evaluation requires a frozen reference run"
            )
        if (
            options.evaluation_mode != "verify" or self.spec.program_mode == "task_adapter"
        ) and options.open3dbench_root is None:
            raise ValueError(
                "Open3DBench root is required for interface/full evaluation and task-adapter programs"
            )
        self.output_dir = options.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: list[Dict[str, Any]] = []
        self._cache: Dict[str, Dict[str, Any]] = {}

    @property
    def records(self) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def _append(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)
            with (self.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    def _prepare_evaluation_result_dir(
        self, evaluation_dir: Path
    ) -> tuple[Path, Optional[Path]]:
        """Optionally place clone/build trees on local scratch storage.

        Frozen stage data and durable logs remain under ``evaluation_dir`` so
        large design files can still use same-filesystem hard links on NAS.
        Only the disposable ``workspace`` and ``runtime`` directories are
        redirected.  The mode is deliberately limited to compact evaluations,
        whose public contract already removes both directories after scoring.
        """

        result_dir = evaluation_dir / "result"
        scratch_value = os.environ.get("OPEN3D_EVALUATION_SCRATCH_ROOT")
        if not scratch_value:
            return result_dir, None
        if not self.options.compact_evaluations:
            raise ValueError(
                "OPEN3D_EVALUATION_SCRATCH_ROOT requires compact evaluations"
            )
        scratch_root = Path(scratch_value).expanduser().resolve()
        run_key = hashlib.sha256(str(self.output_dir).encode("utf-8")).hexdigest()[:16]
        scratch_dir = (
            scratch_root
            / self.spec.task_id
            / run_key
            / evaluation_dir.name
        )
        if scratch_dir.exists() or scratch_dir.is_symlink():
            raise FileExistsError(
                "evaluation scratch directory already exists: {}".format(scratch_dir)
            )
        scratch_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True, exist_ok=False)
        for name in ("workspace", "runtime"):
            target = scratch_dir / name
            target.mkdir()
            (result_dir / name).symlink_to(target, target_is_directory=True)
        return result_dir, scratch_dir

    @staticmethod
    def _discard_evaluation_scratch(result_dir: Path, scratch_dir: Optional[Path]) -> None:
        if scratch_dir is None:
            return
        for name in ("workspace", "runtime"):
            link = result_dir / name
            if link.is_symlink():
                link.unlink()
        shutil.rmtree(scratch_dir, ignore_errors=True)

    def record_invalid(self, program: str, error: Exception) -> None:
        self._append(
            {
                "order": len(self.records) + 1,
                "candidate_sha256": hashlib.sha256(
                    program.encode("utf-8", errors="replace")
                ).hexdigest(),
                "status": "rejected",
                "score": None,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )

    def evaluate(self, candidate: Dict[str, Any]) -> Optional[float]:
        if self.matrix is not None:
            self.matrix.assert_unchanged()
        verification = verify_candidate(self.spec, candidate, getattr(self.options, "open3dbench_root", None))
        canonical = json.dumps(
            verification["candidate"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            cached = (dict(self._cache[digest])
                      if self._cache_results and digest in self._cache else None)
            order = len(self._records) + 1
        if cached is not None:
            record = {
                "order": order,
                "candidate_sha256": digest,
                "candidate": cached.get("candidate"),
                "status": "cached" if cached.get("score") is not None else "failed",
                "score": cached.get("score"),
                "metrics": cached.get("metrics", {}),
                "artifacts": {**cached.get("artifacts", {}), "cached_from_order": cached.get("order")},
                "runtime_seconds": 0.0,
            }
            self._append(record)
            return record["score"]

        evaluation_dir = self.output_dir / "evaluations" / "{:04d}-{}".format(
            order, digest[:12]
        )
        candidate_dir = evaluation_dir / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=False)
        (candidate_dir / "candidate.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        started = time.monotonic()
        result_dir: Optional[Path] = None
        scratch_dir: Optional[Path] = None
        if self.options.evaluation_mode == "verify":
            score: Optional[float] = 1.0 - (order - 1) * 1.0e-9
            metrics = {
                "combined_score": score,
                "runs_successfully": 1.0,
                "verifier_only": 1.0,
            }
            artifacts = {"verification": verification["patch"]}
        elif self.matrix is not None:
            result = self.matrix.evaluate(candidate_dir, self.options.open3dbench_root,
                evaluation_dir / "matrix", transport=self._evaluation_transport,
                workers=self.options.matrix_workers, timeout_seconds=self.options.timeout_seconds,
                compact=self.options.compact_evaluations)
            metrics, artifacts = result.metrics, result.artifacts
            score = metrics.get("combined_score") if metrics.get("runs_successfully") == 1 else None
        else:
            result_dir, scratch_dir = self._prepare_evaluation_result_dir(evaluation_dir)
            try:
                result = (self._evaluation_transport or evaluate_task)(
                    self.spec.root,
                    candidate_dir,
                    self.options.open3dbench_root or Path("."),
                    result_dir,
                    baseline_run=self.options.baseline_run,
                    reference_run=self.options.reference_run,
                    base_config=self.options.base_config,
                    dry_run=self.options.evaluation_mode == "interface",
                    keep_workspace=not self.options.compact_evaluations,
                    compact_artifacts=self.options.compact_evaluations,
                    timeout_seconds=self.options.timeout_seconds,
                )
            finally:
                self._discard_evaluation_scratch(result_dir, scratch_dir)
            metrics = result.metrics
            artifacts = result.artifacts
            if self.options.evaluation_mode == "full" and not metrics.get(
                "runs_successfully"
            ):
                score = None
            else:
                measured_score = metrics.get("combined_score")
                if (isinstance(measured_score, (int, float)) and not isinstance(measured_score, bool)
                        and math.isfinite(measured_score)):
                    score = float(measured_score)
                else:
                    score = None
                    artifacts = {**artifacts, "error_type": "InvalidScore",
                                 "error_message": "Evaluator did not return a finite combined_score"}
        record = {
            "order": order,
            "candidate_sha256": digest,
            "candidate": str(candidate_dir / "candidate.json"),
            "status": "accepted" if score is not None else "failed",
            "score": score,
            "metrics": metrics,
            "artifacts": artifacts,
            "runtime_seconds": time.monotonic() - started,
        }
        self._append(record)
        with self._lock:
            self._cache[digest] = dict(record)
        return score

    def evaluate_program(self, program: str) -> Optional[float]:
        if self._max_program_chars is not None and len(program) > self._max_program_chars:
            raise ValueError("Candidate exceeds the task's maximum complete program length")
        candidate = materialize_program(
            self.spec,
            program,
            self.options.open3dbench_root,
        )
        return self.evaluate(candidate)

    def evaluate_proposal(self, proposal: CandidateProposal) -> CandidateFeedback:
        # Preserve rejected code too; materialization may fail before any
        # candidate directory exists. This is not a model-facing log.
        with self._lock:
            with (self.output_dir / "search_proposals.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"proposal_id": proposal.proposal_id,
                    "program": proposal.program, "metadata": dict(proposal.metadata)},
                    ensure_ascii=False) + "\n")
        before = len(self.records)
        try:
            self.evaluate_program(proposal.program)
        except Exception as error:
            self.record_invalid(proposal.program, error)
        rows = self.records
        if len(rows) != before + 1:
            raise RuntimeError("one proposal must produce exactly one history record")
        record = rows[-1]
        feedback = CandidateFeedback(
            proposal_id=proposal.proposal_id,
            program_sha256=hashlib.sha256(proposal.program.encode("utf-8")).hexdigest(),
            status=str(record.get("status", "failed")),
            score=float(record["score"]) if record.get("score") is not None else None,
            metrics=dict(record.get("metrics", {})),
            artifacts=dict(record.get("artifacts", {})),
            metadata=dict(proposal.metadata),
            error_type=record.get("error_type") or record.get("artifacts", {}).get("error_type"),
            error_message=record.get("error_message") or record.get("artifacts", {}).get("error_message"),
            runtime_seconds=float(record.get("runtime_seconds", 0.0)),
        )
        with self._lock:
            with (self.output_dir / "search_history.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(
                    json.dumps(feedback.to_dict(), sort_keys=True, ensure_ascii=False)
                    + "\n"
                )
        return feedback

    def summary(self) -> Dict[str, Any]:
        rows = self.records
        accepted = [row for row in rows if row.get("score") is not None]
        best = max(accepted, key=lambda row: row["score"]) if accepted else None
        cache_hits = sum(row.get("status") == "cached" for row in rows)
        return {
            "schema_version": 1,
            "task": self.spec.task_id,
            "evaluation_mode": self.options.evaluation_mode,
            "score_semantics": (
                "verifier transport score; not QoR"
                if self.options.evaluation_mode == "verify"
                else "equal-case mean of paired-seed task combined_scores" if self.matrix is not None
                else "task combined_score"
            ),
            "evaluations": len(rows),
            "unique_evaluations": len(rows) - cache_hits,
            "cache_hits": cache_hits,
            "accepted": len(accepted),
            "rejected_or_failed": len(rows) - len(accepted),
            "best": best,
        }
