"""Current six-task feedback and real OpenEvolve templates; no physical runs."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from local_bench.model import EvaluationResult, load_candidate, load_task
from local_bench.metrics import _routing_policy_diagnostics, _net_order_diagnostics, _optional_runtime
from local_bench.optimization_feedback import TASK_IDS, TEMPLATES, optimization_result
from local_bench.prompting import export_openevolve_bundle, feedback_context
from local_bench.search_api import CandidateFeedback, TaskEvaluationOptions
from local_bench.search_session import prepare_search_problem, search_feedback

ROOT = Path(__file__).resolve().parents[2]
DIRS = ("placement_parameters_reference_20260923", "placement_initialization",
        "placement_preconditioner", "global_routing_ripup_policy", "global_routing_maze_cost",
        "global_routing_iteration_controller")


@pytest.mark.parametrize("directory", DIRS)
def test_export_and_ask_tell_share_task_projection(tmp_path, directory):
    task = ROOT / "AgenticBench/local_tasks" / directory
    spec = load_task(task)
    problem = prepare_search_problem(TaskEvaluationOptions(task_dir=task, open3dbench_root=ROOT,
                                                          output_dir=tmp_path / "session"))
    assert problem.task_id in TASK_IDS
    assert "three-dimensional integrated circuit (3D-IC)" in " ".join(problem.description.split())
    for text in (problem.description, problem.system_message):
        assert "sha256" not in text and str(ROOT) not in text
        assert "ADOPT" not in text and "FRESH" not in text
        assert "causal" not in text.lower()
    headings = (("## Background", "## Your task", "## How results are compared", "## Input reference")
                if directory in {"placement_initialization", "placement_preconditioner"}
                else ("## Background", "## Your task", "## How solutions are evaluated", "## Using previous results"))
    positions = [problem.description.index(h) for h in headings]
    assert positions == sorted(positions)
    metrics = {"combined_score": 1.0256789123, "runs_successfully": 1.0,
               "placement_native_hpwl": 1234, "grt_total_overflow": 0,
               "runtime_seconds": None, "bottom_policy_signal_max": 100,
               "legacy_diagnostic": 99, "nesterov_identity_controls_only": True,
               "nesterov_control_first_step": 0}
    raw = EvaluationResult(metrics=metrics, artifacts={"log": "PRIVATE_RAW_LOG /private/internal"})
    feedback = CandidateFeedback("candidate", "abc", "accepted", metrics["combined_score"],
                                 metrics=metrics, artifacts=raw.artifacts)
    projected = search_feedback(problem, feedback)
    expected = optimization_result(raw, task_id=spec.task_id)
    assert projected.metrics == expected.metrics and projected.artifacts == expected.artifacts
    assert "legacy_diagnostic" not in projected.metrics
    assert "bottom_policy_signal_max" not in projected.metrics
    assert "runtime_seconds" not in projected.metrics
    assert "PRIVATE_RAW_LOG" in raw.artifacts["log"]  # Original receipt stays intact.
    history_text = feedback_context([feedback.to_dict()], spec)
    assert "PRIVATE_RAW_LOG" not in history_text and "legacy_diagnostic" not in history_text
    assert "Missing metrics remain unknown" in history_text

    pytest.importorskip("openevolve")
    from openevolve.config import Config
    from openevolve.prompt.sampler import PromptSampler
    from local_bench.openevolve_adapter import _to_openevolve_result
    bundle = export_openevolve_bundle(spec, load_candidate(spec.initial_candidate), ROOT, tmp_path / "bundle")
    config = Config.from_yaml(bundle["config"])
    assert config.diff_based_evolution is False
    assert {p.name for p in Path(config.prompt.template_dir).iterdir()} == set(TEMPLATES)
    assert {p.stem: p.read_text() for p in Path(config.prompt.template_dir).iterdir()} == problem.prompt_templates
    native = _to_openevolve_result(raw, spec.task_id)
    assert native.metrics == expected.metrics
    example = {"code": problem.initial_program, "metrics": native.metrics,
               "metadata": {"parent_metrics": {"combined_score": 1}}}
    rendered = PromptSampler(config.prompt).build_prompt(
        current_program=problem.initial_program, parent_program=problem.initial_program,
        program_metrics=native.metrics, previous_programs=[example], top_programs=[example],
        inspirations=[example], diff_based_evolution=False, program_artifacts=native.artifacts)
    user = rendered["user"]
    for required in ("Selected evaluated example", "Alternative program", "Selected program",
                     "1.0256789123", "Try one main change at a time", "not a chronological"):
        assert required in user
    for forbidden in ("PRIVATE_RAW_LOG", "Excellent", "All metrics improved"):
        assert forbidden not in user


@pytest.mark.parametrize("directory", DIRS)
@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), "0", True])
def test_optional_invalid_diagnostics_omitted_and_zero_preserved(directory, missing):
    spec = load_task(ROOT / "AgenticBench/local_tasks" / directory)
    result = optimization_result(EvaluationResult(metrics={
        "combined_score": 0.0, "runtime_seconds": missing, "placement_max_density": missing,
        "grt_total_overflow": 0.0, "placement_native_hpwl": 0.0}, artifacts={}), task_id=spec.task_id)
    assert result.metrics["combined_score"] == 0
    assert "runtime_seconds" not in result.metrics and "placement_max_density" not in result.metrics
    assert any(result.metrics.get(k) == 0 for k in ("grt_total_overflow", "placement_native_hpwl"))
    json.dumps(result.to_dict(), allow_nan=False)


def write_log(tmp_path, text):
    path = tmp_path / "75_global_route/bottom/route.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_iteration_index_resets_are_not_lost_invocations(tmp_path):
    write_log(tmp_path, "\n".join(
        f"OPEN3D_ITERATION_CONTROL die=bottom calls={i} changed={changed} "
        f"cost_scale={scale} search_scale=1 ripup_offset=0 cost_height=20 search_window=8 ripup_threshold=0"
        for i, changed, scale in [(3, 1, 2), (1, 0, 1), (2, 1, 0.5)]))
    d = _routing_policy_diagnostics(tmp_path, "bottom", "iteration")
    assert d["calls"] == d["rows"] == 3
    assert d["changed"] == 2 and d["iteration_index_last"] == 2
    assert d["cost_scale_last"] == d["cost_scale_min"] == 0.5
    assert d["cost_scale_max"] == 2
    assert not any("signal" in k for k in d)


def test_sparse_ripup_counts_and_observed_output_samples(tmp_path):
    write_log(tmp_path, "\n".join(
        f"OPEN3D_RIPUP_POLICY die=bottom calls={calls} changed={changed} changed_ratio={changed/calls} "
        f"selected={selected} selected_ratio={selected/calls} score={score}"
        for calls, changed, selected, score in [(1, 0, 0, 0), (50000, 800, 1000, 0.75)]))
    d = _routing_policy_diagnostics(tmp_path, "bottom", "ripup")
    assert d["calls"] == 50000 and d["changed"] == 800 and d["selected"] == 1000
    assert d["selected_ratio_last"] == 0.02 and d["score_min"] == 0
    assert d["score_last"] == d["score_max"] == 0.75


def test_current_maze_field_only_no_legacy_alias_or_fake_zero(tmp_path):
    write_log(tmp_path, "OPEN3D_MAZE_COST die=bottom calls=1 changed=0 changed_ratio=0 "
                        "multiplier_last=nan multiplier_mean=7 aggressiveness=8\n")
    d = _routing_policy_diagnostics(tmp_path, "bottom", "maze_cost")
    assert d["rows"] == 0  # Malformed output evidence does not qualify a hook.
    assert not any("multiplier" in k or "signal" in k for k in d)
    write_log(tmp_path, "OPEN3D_MAZE_COST die=bottom calls=1 changed=0 multiplier_last=1\n")
    d = _routing_policy_diagnostics(tmp_path, "bottom", "maze_cost")
    assert d["multiplier_last"] == 1 and "changed_ratio_max" not in d


def test_missing_hook_is_no_evidence_not_perfect_overlap(tmp_path):
    write_log(tmp_path, "")
    assert _net_order_diagnostics(tmp_path, "bottom") == {"calls": 0, "changed_calls": 0}
    assert _routing_policy_diagnostics(tmp_path, "bottom", "ripup") == {"rows": 0, "calls": 0, "changed": 0}


@pytest.mark.parametrize("value", [None, -1, True, "12", float("nan")])
def test_missing_runtime_is_not_zero(value):
    result = {"runtime_seconds": 0}
    _optional_runtime(result, {"runtime_seconds": value})
    assert "runtime_seconds" not in result
    _optional_runtime(result, {"runtime_seconds": 0})
    assert result["runtime_seconds"] == 0


def test_failure_is_sanitized_without_inventing_score(tmp_path):
    task = ROOT / "AgenticBench/local_tasks" / DIRS[-1]
    problem = prepare_search_problem(TaskEvaluationOptions(task_dir=task, open3dbench_root=ROOT,
                                                          output_dir=tmp_path / "session"))
    raw = CandidateFeedback("bad", "abc", "rejected", None, metrics={},
                            error_message="Invalid return shape in /private/internal at https://example.com/log")
    result = search_feedback(problem, raw)
    assert result.score is None and result.metrics == {} and not result.valid
    assert "/private/internal" not in json.dumps(asdict(result))
    assert "Invalid return shape" in result.error_message


def test_shared_templates_are_in_package_data():
    import tomllib
    package = ROOT / "AgenticBench"
    config = tomllib.loads((package / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["local_bench"]
    files = {p.name for pattern in patterns for p in (package / "local_bench").glob(pattern)}
    assert set(TEMPLATES) <= files


@pytest.mark.parametrize("directory", DIRS)
def test_real_native_two_round_ask_tell_uses_only_projected_feedback(tmp_path, directory):
    pytest.importorskip("openevolve")
    from openevolve.config import Config
    from local_bench.openevolve_search import OpenEvolveAskTell
    from local_bench.search_session import run_task_search
    task = ROOT / "AgenticBench/local_tasks" / directory
    options = TaskEvaluationOptions(task_dir=task, open3dbench_root=ROOT,
                                   output_dir=tmp_path / "session", evaluation_mode="interface")
    problem = prepare_search_problem(options)
    messages, evaluations = [], []
    def generate(system, user):
        messages.append((system, user))
        return "```python\n" + problem.initial_program + "\n```"
    def evaluate(*args, **kwargs):
        evaluations.append(args)
        return EvaluationResult(metrics={"combined_score": 1.021, "runs_successfully": 1,
            "grt_total_overflow": 0, "placement_native_hpwl": 2345,
            "legacy_diagnostic": 987}, artifacts={"log": "PRIVATE_RAW_LOG"})
    backend = OpenEvolveAskTell(Config(diff_based_evolution=False), generate=generate)
    run = run_task_search(backend, options, max_evaluations=2, evaluation_transport=evaluate)
    assert [item.status for item in run.history] == ["accepted", "cached"]
    assert len(evaluations) == 1 and len(messages) == 2
    assert "no measured score" in messages[0][1]
    assert "1.021" in messages[1][1]
    assert messages[1][1].count('"combined_score":') == 1
    assert "Selection fitness; dimensionless, higher is better" in messages[1][1]
    assert "optimization_feedback" not in messages[1][1]  # No duplicate JSON blob.
    assert "PRIVATE_RAW_LOG" not in messages[1][1] and "legacy_diagnostic" not in messages[1][1]
    assert "PRIVATE_RAW_LOG" in (options.output_dir / "history.jsonl").read_text()
    assert "PRIVATE_RAW_LOG" in (options.output_dir / "search_history.jsonl").read_text()
    assert all("PRIVATE_RAW_LOG" not in json.dumps(item.artifacts) for item in run.history)
