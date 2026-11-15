"""Opt-in routing score contract v2: overflow hard gate, then W_ref / W."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from local_bench.feedback_semantics import metric_definition, metric_definitions
from local_bench.metrics import extract_metrics
from local_bench.model import EvaluationResult, load_task
from local_bench.optimization_feedback import optimization_result
from local_bench.prompt_context import feedback_text
from local_bench.prompting import build_problem_description, task_contract
from local_bench.routing_scoring import parse_scoring

from tests.test_routing_wirelength_priority import write_run


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "routing_ripup_policy_source": "global_routing_ripup_policy",
    "routing_maze_cost_source": "global_routing_maze_cost",
    "routing_iteration_controller_source": "global_routing_iteration_controller",
}
V2 = {"version": 2, "overflow_gate": {"max_ratio": 1.0, "abs_slack": 10.0}}
V1_KEYS = {"combined_score", "overflow_cost", "wirelength_cost", "overflow_weight",
           "wirelength_weight", "overflow_score", "wirelength_score"}


def task_copy(tmp_path, directory, scoring=None):
    source = ROOT / "local_tasks" / directory
    target = tmp_path / directory
    shutil.copytree(str(source), str(target), ignore=shutil.ignore_patterns("__pycache__"))
    data = yaml.safe_load((target / "task.yaml").read_text(encoding="utf-8"))
    data["base_config"] = str((source / data["base_config"]).resolve())
    if scoring is not None:
        data["scoring"] = scoring
    (target / "task.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


@pytest.mark.parametrize("kind", TASKS)
def test_default_v1_outputs_are_unchanged(tmp_path, kind):
    reference = write_run(tmp_path / "reference", kind, overflow=1000.0, wirelength=100.0)
    candidate = write_run(tmp_path / "candidate", kind, overflow=1010.0, wirelength=95.0)
    default = extract_metrics(kind, candidate, reference)
    explicit_v1 = extract_metrics(kind, candidate, reference, scoring={})
    assert default == explicit_v1
    assert V1_KEYS <= set(default)
    assert "routing_scoring_version" not in default
    expected = 1.0 / (0.1 * (2020.0 + 1000.0) / (2000.0 + 1000.0) + 0.9 * 190.0 / 200.0)
    assert default["combined_score"] == pytest.approx(expected)


@pytest.mark.parametrize("kind", TASKS)
@pytest.mark.parametrize("reference_overflow", [0.0, 1000.0])
def test_v2_reference_scores_one(tmp_path, kind, reference_overflow):
    reference = write_run(tmp_path / "reference", kind, overflow=reference_overflow)
    result = extract_metrics(kind, reference, reference, scoring=V2)
    assert result["routing_scoring_version"] == 2
    assert result["overflow_gate_passed"] == 1.0
    assert result["combined_score"] == pytest.approx(1.0)
    assert not {"overflow_cost", "overflow_score", "overflow_weight"} & set(result)


@pytest.mark.parametrize("kind", TASKS)
def test_v2_score_is_wirelength_ratio_when_gate_passes(tmp_path, kind):
    reference = write_run(tmp_path / "reference", kind, overflow=1000.0, wirelength=100.0)
    candidate = write_run(tmp_path / "candidate", kind, overflow=1004.0, wirelength=80.0)
    result = extract_metrics(kind, candidate, reference, scoring=V2)
    # Total overflow 2008 <= 1.0 * 2000 + 10.
    assert result["overflow_gate_limit"] == pytest.approx(2010.0)
    assert result["overflow_gate_passed"] == 1.0
    assert result["combined_score"] == pytest.approx(200.0 / 160.0)
    assert result["overflow_gate_reason"].startswith("overflow gate passed")


@pytest.mark.parametrize("kind", TASKS)
def test_v2_gate_failure_is_zero_with_explicit_reason(tmp_path, kind):
    reference = write_run(tmp_path / "reference", kind, overflow=1000.0, wirelength=100.0)
    candidate = write_run(tmp_path / "candidate", kind, overflow=1006.0, wirelength=50.0)
    result = extract_metrics(kind, candidate, reference, scoring=V2)
    assert result["overflow_gate_passed"] == 0.0
    assert result["combined_score"] == 0.0
    # The wire-length measurement is still reported, but does not score.
    assert result["wirelength_score"] == pytest.approx(2.0)
    reason = result["overflow_gate_reason"]
    assert reason.startswith("overflow gate failed")
    assert "2012 > limit 2010" in reason and "combined_score is 0" in reason


@pytest.mark.parametrize("kind", TASKS)
def test_v2_zero_overflow_reference_uses_absolute_slack(tmp_path, kind):
    reference = write_run(tmp_path / "reference", kind, overflow=0.0, wirelength=100.0)
    strict = {"version": 2, "overflow_gate": {"abs_slack": 0}}
    tolerant = {"version": 2, "overflow_gate": {"max_ratio": 1.0, "abs_slack": 4}}
    zero = write_run(tmp_path / "zero", kind, overflow=0.0, wirelength=90.0)
    small = write_run(tmp_path / "small", kind, overflow=2.0, wirelength=90.0)
    assert extract_metrics(kind, zero, reference, scoring=strict)["combined_score"] == pytest.approx(200.0 / 180.0)
    assert extract_metrics(kind, small, reference, scoring=strict)["combined_score"] == 0.0
    # Total overflow 4 <= 1.0 * 0 + 4.
    assert extract_metrics(kind, small, reference, scoring=tolerant)["combined_score"] == pytest.approx(200.0 / 180.0)
    # In v1 the same zero-overflow reference gives a smooth, nonzero trade-off.
    assert extract_metrics(kind, small, reference)["combined_score"] > 0.0


@pytest.mark.parametrize("kind", TASKS)
def test_v2_hook_gate_still_applies(tmp_path, kind):
    reference = write_run(tmp_path / "reference", kind)
    candidate = write_run(tmp_path / "candidate", kind, wirelength=1.0, hook=False)
    result = extract_metrics(kind, candidate, reference, scoring=V2)
    assert result["overflow_gate_passed"] == 1.0
    assert result["combined_score"] == result["grt_complete"] == 0.0


@pytest.mark.parametrize("kind", TASKS)
def test_v2_bootstrap_without_reference_is_explicitly_ungated(tmp_path, kind):
    candidate = write_run(tmp_path / "candidate", kind, wirelength=500_000.0)
    result = extract_metrics(kind, candidate, None, scoring=V2)
    assert result["overflow_gate_evaluated"] == 0.0
    assert result["combined_score"] == pytest.approx(1.0 / 2.0)
    assert "not evaluated" in result["overflow_gate_reason"]


@pytest.mark.parametrize("value, message", [
    ("v2", "must be an object"),
    ({"version": 3}, "unsupported scoring.version"),
    ({"version": True}, "must be an integer"),
    ({"version": 2}, "requires an overflow_gate"),
    ({"version": 2, "overflow_gate": {"max_ratio": 1.0}}, "abs_slack is required"),
    ({"version": 2, "overflow_gate": {"abs_slack": -1}}, "non-negative"),
    ({"version": 2, "overflow_gate": {"abs_slack": float("nan")}}, "non-negative"),
    ({"version": 2, "overflow_gate": {"abs_slack": 0, "weight": 1}}, "unknown keys"),
    ({"version": 2, "overflow_gate": {"abs_slack": 0}, "extra": 1}, "unknown keys"),
    ({"version": 1, "overflow_gate": {"abs_slack": 0}}, "requires scoring.version 2"),
])
def test_scoring_block_validation(value, message):
    with pytest.raises(ValueError, match=message):
        parse_scoring(value, "routing_maze_cost_source")


def test_scoring_is_rejected_for_non_routing_kinds():
    with pytest.raises(ValueError, match="only for routing source kinds"):
        parse_scoring({"version": 2, "overflow_gate": {"abs_slack": 0}}, "placement_initialization_source")
    with pytest.raises(ValueError, match="only for routing source kinds"):
        extract_metrics("placement_initialization_source", Path("."), None, scoring=V2)


def test_explicit_v1_and_absent_normalize_identically():
    assert parse_scoring(None, "routing_maze_cost_source") == {}
    assert parse_scoring({"version": 1}, "routing_maze_cost_source") == {}
    assert parse_scoring({"version": 2, "overflow_gate": {"abs_slack": 3}},
                         "routing_maze_cost_source") == {
        "version": 2, "overflow_gate": {"max_ratio": 1.0, "abs_slack": 3.0}}


@pytest.mark.parametrize("kind, directory", sorted(TASKS.items()))
def test_maintained_routing_tasks_default_to_v1_contract(kind, directory):
    spec = load_task(ROOT / "local_tasks" / directory)
    assert spec.scoring == {}
    assert "scoring" not in task_contract(spec)
    prompt = build_problem_description(spec, {}, include_initial_program=False)
    assert "routing scoring version 2" not in prompt


@pytest.mark.parametrize("kind, directory", sorted(TASKS.items()))
def test_v2_task_contract_and_prompt_state_the_active_formula(tmp_path, kind, directory):
    default = load_task(ROOT / "local_tasks" / directory)
    spec = load_task(task_copy(tmp_path, directory, scoring=V2))
    assert spec.scoring == V2
    contract = task_contract(spec)
    assert contract.pop("scoring") == V2
    base = task_contract(default)
    for item in (contract, base):
        item.pop("base_config")
        item["program"] = {key: Path(value).name if value else value
                           for key, value in item["program"].items()}
    assert contract == base
    prompt = build_problem_description(spec, {}, include_initial_program=False)
    default_prompt = build_problem_description(default, {}, include_initial_program=False)
    assert prompt.startswith(default_prompt)
    override = prompt[len(default_prompt):]
    assert "routing scoring version 2" in override
    assert "O <= 1 * O_ref + 10" in override
    assert "combined_score = 0" in override


def test_v2_feedback_text_is_version_accurate(tmp_path):
    kind = "routing_maze_cost_source"
    task_id = "global_routing_maze_cost_source"
    reference = write_run(tmp_path / "reference", kind, overflow=1000.0)
    candidate = write_run(tmp_path / "candidate", kind, overflow=1006.0, wirelength=50.0)
    metrics = extract_metrics(kind, candidate, reference, scoring=V2)
    metrics["runs_successfully"] = 1.0
    feedback = json.loads(optimization_result(
        EvaluationResult(metrics=metrics), task_id=task_id).artifacts["optimization_feedback"])
    assert feedback["metrics"]["overflow_gate_passed"] == 0.0
    assert feedback["metrics"]["routing_scoring_version"] == 2
    assert "overflow_cost" not in feedback["metrics"]
    assert "Routing scoring version 2" in feedback["interpretation"]
    assert "overflow gate failed" in feedback["interpretation"]
    assert "0.1" not in feedback["metric_definitions"]["combined_score"]
    assert "W_ref/W" in feedback["metric_definitions"]["combined_score"]
    text = feedback_text(metrics, task_id)
    assert '"overflow_gate_passed": 0.0' in text
    assert "overflow gate failed" in text
    v1 = extract_metrics(kind, candidate, reference)
    v1_definitions = metric_definitions(v1, task_id)
    assert "+1000" in v1_definitions["overflow_cost"]
    assert metric_definition("overflow_cost", task_id) == v1_definitions["overflow_cost"]
    assert "Not used by routing scoring v2" in metric_definition("overflow_cost", task_id, 2)
