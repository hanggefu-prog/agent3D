"""Trusted task preparation: search engines receive data, never evaluator paths."""
from dataclasses import replace
from pathlib import Path

import yaml

from .model import EvaluationResult, load_candidate, load_task
from .program_bridge import initial_program_text
from .prompting import build_problem_description, composed_system_message
from .search_api import AskTellRunner, ProgramEvaluationHarness, SearchProblem
from .optimization_feedback import TASK_IDS, TEMPLATES, optimization_result, template_source
from .tp_gnn_openevolve import _reason


def prepare_search_problem(options):
    """Called by the benchmark host, not by the search backend."""
    spec = load_task(options.task_dir)
    if options.base_config is not None:
        spec = replace(spec, base_config=Path(options.base_config).resolve())
    candidate = load_candidate(spec.initial_candidate)
    config = yaml.safe_load((spec.root / "config.yaml").read_text())
    templates = {}
    if spec.task_id in TASK_IDS:
        templates = {Path(name).stem: (template_source(spec) / name).read_text()
                     for name in TEMPLATES}
    matrix = None
    if getattr(options, "evaluation_matrix", None) is not None:
        from .matrix import LocalEvaluationMatrix
        matrix = LocalEvaluationMatrix(options.evaluation_matrix, spec)
    return SearchProblem(
        task_id=spec.task_id,
        description=build_problem_description(spec, candidate, options.open3dbench_root,
                                              include_initial_program=False) + (matrix.prompt() if matrix else ""),
        initial_program=initial_program_text(spec, candidate),
        system_message=composed_system_message(spec),
        language="python", response_format="complete_program",
        max_program_chars=config.get("max_code_length"),
        prompt_templates=templates,
        metadata={"availability": spec.availability, **({"matrix_sha256": matrix.identity,
                  "cells_per_evaluation": len(matrix.cells)} if matrix else {})},
    )


def search_feedback(problem, feedback):
    """Apply task feedback policy consistently for every ask/tell backend."""
    if problem.task_id not in TASK_IDS:
        return feedback
    artifacts = dict(feedback.artifacts)
    if feedback.error_message:
        artifacts["error_message"] = feedback.error_message
    projected = optimization_result(
        EvaluationResult(metrics=dict(feedback.metrics), artifacts=artifacts),
        task_id=problem.task_id,
        require_score=feedback.valid,
    )
    return replace(feedback, metrics=projected.metrics, artifacts=projected.artifacts,
                   error_message=_reason(artifacts.get("error_message")))


def run_task_search(backend, options, *, max_evaluations, evaluation_transport=None):
    """Only ask(context) and tell(feedback) are called on the search backend.

    The host owns options, source/reference paths, evaluation and durable receipts.
    No implicit initial/reference evaluation is added to the candidate budget.
    """
    output = Path(options.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("A new ask/tell session requires an empty output directory; resume is not implemented")
    problem = prepare_search_problem(options)
    harness = ProgramEvaluationHarness(options, evaluation_transport=evaluation_transport)
    return AskTellRunner().run(
        backend, harness, problem, max_evaluations=max_evaluations,
        feedback_transform=lambda item: search_feedback(problem, item),
    )
