"""Offline prompt/feedback regressions: no API or physical execution."""
import json
from pathlib import Path

import pytest

from local_bench.feedback_semantics import metric_definition


def test_placement_feedback_on_thin_host_string_api():
    """The server's Python 3.8 host must render native placement feedback."""
    class HostString(str):
        def removeprefix(self, prefix):
            raise AttributeError("Python 3.8 has no str.removeprefix")

    assert "for bottom;" in metric_definition(HostString("placement_overflow_bottom"), "placement_initialization")
    assert "(top);" in metric_definition(HostString("placement_max_density_top"), "placement_preconditioner")
from local_bench.model import EvaluationResult, load_candidate, load_task
from local_bench.observed_diagnostics import assignment_comparison, placement_log_diagnostics
from local_bench.optimization_feedback import (
    NESTEROV_METRICS, PLACEMENT_METRICS, POLICY_DETAILS, POLICY_METRICS,
    ROUTING_METRICS, TP_TIMING_GROUPS, optimization_result,
)
from local_bench.prompting import build_problem_description
from local_bench.program_bridge import initial_program_text

ROOT = Path(__file__).resolve().parents[1]
CURRENT = (
    "tp_gnn_post_placement_timing", "placement_initialization", "placement_preconditioner",
    "placement_parameters_reference_20260923", "global_routing_ripup_policy",
    "global_routing_maze_cost", "global_routing_iteration_controller", "hbt_assignment_policy",
)


@pytest.mark.parametrize("directory", CURRENT)
def test_current_prompt_keeps_initial_program_and_readable_interface(directory):
    spec = load_task(ROOT / "local_tasks" / directory)
    prompt = build_problem_description(spec, load_candidate(spec.initial_candidate), ROOT.parent)
    assert initial_program_text(spec, load_candidate(spec.initial_candidate)) in prompt
    assert prompt.index("## Background") < prompt.index("## Your task")
    assert "metric_definitions" in prompt
    assert "| ---" in prompt
    assert "read-only" in prompt.lower()


@pytest.mark.parametrize("directory", CURRENT)
def test_every_current_emitted_metric_has_a_definition(directory):
    spec = load_task(ROOT / "local_tasks" / directory)
    keys = set(PLACEMENT_METRICS | NESTEROV_METRICS | ROUTING_METRICS | POLICY_METRICS)
    keys.update(k for names in TP_TIMING_GROUPS.values() for k in names)
    keys.update(prefix + "policy_" + name + "_" + stat
                for prefix in ("bottom_", "top_")
                for names in POLICY_DETAILS.values() for name in names
                for stat in ("min", "max", "last"))
    keys.update({"bottom_policy_selected", "top_policy_selected",
                 "bottom_policy_iteration_index_last", "top_policy_iteration_index_last"})
    from local_bench.feedback_semantics import DEFINITIONS
    keys.update(DEFINITIONS)
    raw = EvaluationResult(metrics={key: 1.0 for key in keys}, artifacts={"log": "private raw log"})
    projected = optimization_result(raw, task_id=spec.task_id)
    feedback = json.loads(projected.artifacts["optimization_feedback"])
    assert set(feedback["metric_definitions"]) == set(projected.metrics)
    assert all(len(text) > 15 for text in feedback["metric_definitions"].values())
    assert projected.metrics["combined_score"] == 1.0
    assert "private raw log" not in projected.artifacts["optimization_feedback"]
    assert len(projected.artifacts["optimization_feedback"].encode("utf-8")) < 32768


def test_definitions_explain_nonobvious_formulas():
    assert "+1000" in metric_definition("overflow_score", "global_routing_maze_cost_source")
    assert "minus unique_x_tracks" in metric_definition("shared_x_track_count", "hbt_assignment_policy")
    assert "not area-weighted" in metric_definition("partition_assignment_changed_fraction", "tp_gnn_post_placement_timing")


def test_component_prompts_expose_inactive_native_control_paths():
    initializer = load_task(ROOT / "local_tasks/placement_initialization")
    preconditioner = load_task(ROOT / "local_tasks/placement_preconditioner")
    init_prompt = build_problem_description(initializer, load_candidate(initializer.initial_candidate), ROOT.parent)
    precond_prompt = build_problem_description(preconditioner, load_candidate(preconditioner.initial_candidate), ROOT.parent)
    assert "random_center_init_flag=0" in init_prompt
    assert "does not call `set_overflow`" in precond_prompt


def assignment(root, rows):
    path = root / "20_partition/partition_assignment.csv"
    path.parent.mkdir(parents=True)
    path.write_text("inst_name,die\n" + rows)


def test_assignment_comparison_aligns_names_and_keeps_physical_tiers(tmp_path):
    candidate, reference = tmp_path / "c", tmp_path / "r"
    assert assignment_comparison(candidate, reference) == {}
    assignment(reference, "a,bottom\nb,top\n")
    assignment(candidate, "b,top\na,bottom\n")
    assert assignment_comparison(candidate, reference)["partition_assignment_identical"] == 1
    (candidate / "20_partition/partition_assignment.csv").write_text("inst_name,die\na,top\nb,bottom\n")
    result = assignment_comparison(candidate, reference)
    assert result == {"partition_assignment_identical": 0, "partition_assignment_changed_count": 2,
                      "partition_assignment_changed_fraction": 1.0}
    assert assignment_comparison(candidate, None) == {}


@pytest.mark.parametrize("rows", ["a,bottom\na,top\n", "a,other\nb,top\n", "a,bottom\nc,top\n", ""])
def test_invalid_assignment_is_not_silently_called_unchanged(tmp_path, rows):
    assignment(tmp_path / "r", "a,bottom\nb,top\n")
    assignment(tmp_path / "c", rows)
    with pytest.raises(ValueError):
        assignment_comparison(tmp_path / "c", tmp_path / "r")


def gp_log(root, text):
    path = root / "30_placement/stages/02_global_place/run.log"
    path.parent.mkdir(parents=True)
    path.write_text(text)


def test_placement_progress_uses_complete_rows_not_legalization(tmp_path):
    gp_log(tmp_path, "wHPWL 100 Overflow [0.5, 0.4] MaxDensity [1.2, 1.3]\n"
           "wHPWL 70 Overflow [0.1, 0.2] MaxDensity [1.0, 0.9]\n"
           "legalization wHPWL 999\n")
    result = placement_log_diagnostics(tmp_path)
    assert result["placement_logged_rows"] == 2
    assert result["placement_first_logged_hpwl"] == 100
    assert result["placement_logged_hpwl_relative_change"] == pytest.approx(-0.3)
    assert result["placement_logged_overflow_change"] == pytest.approx(-0.3)


def test_missing_and_invalid_placement_progress_remain_unknown(tmp_path):
    assert placement_log_diagnostics(tmp_path) == {}
    gp_log(tmp_path, "wHPWL 100 Overflow [nan, 0.1] MaxDensity [1, 1]\n")
    assert placement_log_diagnostics(tmp_path) == {}
    path = tmp_path / "30_placement/stages/02_global_place/run.log"
    path.write_text("wHPWL 0 Overflow 0 MaxDensity 0\n")
    result = placement_log_diagnostics(tmp_path)
    assert result["placement_first_logged_hpwl"] == 0
    assert "placement_logged_hpwl_relative_change" not in result


def test_missing_measurements_do_not_get_definitions_or_fake_values():
    result = optimization_result(EvaluationResult(metrics={
        "combined_score": 1, "placement_first_logged_hpwl": float("nan"),
        "placement_logged_rows": None}, artifacts={}), task_id="placement_initialization")
    assert result.metrics == {"combined_score": 1}
    feedback = json.loads(result.artifacts["optimization_feedback"])
    assert set(feedback["metric_definitions"]) == {"combined_score"}


def test_placement_diagnostics_reach_metrics_without_changing_score(tmp_path):
    from local_bench.metrics import placement_component_metrics
    for directory, scale in ((tmp_path / "c", 1), (tmp_path / "r", 2)):
        gp_log(directory, f"wHPWL {100*scale} Overflow [0.5, 0.4] MaxDensity [1.2, 1.3]\n"
               f"wHPWL {70*scale} Overflow [0.1, 0.2] MaxDensity [1.0, 0.9]\n")
    result = placement_component_metrics(tmp_path / "c", tmp_path / "r")
    assert result["combined_score"] == pytest.approx(1.9)
    assert result["placement_logged_rows"] == 2
    projected = optimization_result(EvaluationResult(metrics=result, artifacts={}), task_id="placement_preconditioner")
    assert projected.metrics["placement_logged_hpwl_relative_change"] == pytest.approx(-0.3)


def test_timing_assignment_comparison_reaches_current_feedback(tmp_path, monkeypatch):
    from local_bench import post_placement_timing as timing, metrics
    candidate, reference = tmp_path / "c", tmp_path / "r"
    assignment(candidate, "a,bottom\nb,top\n")
    assignment(reference, "a,bottom\nb,top\n")
    monkeypatch.setattr(timing, "validate_snapshot", lambda *args, **kwargs: {
        "metrics": {"wns_setup": -1., "tns_setup": -10.}, "hbt_model": {}, "partition_config": {}})
    monkeypatch.setattr(metrics, "tp_gnn_learning_objective_metrics", lambda *args: {"partition_constraints_passed": 1.})
    result = timing.timing_metrics(candidate, reference, expected_decoder="global-2way")
    assert result["combined_score"] == 1.
    projected = optimization_result(EvaluationResult(metrics=result, artifacts={}), task_id="tp_gnn_post_placement_timing")
    assert json.loads(projected.artifacts["optimization_feedback"])["partition_diagnostics"]["partition_assignment_identical"] == 1
