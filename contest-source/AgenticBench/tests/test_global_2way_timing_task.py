"""Global 2-way baseline switch, with unchanged no-OPT timing fitness."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from local_bench import metrics, post_placement_timing as timing
from local_bench.cli import main
from local_bench.evaluation import evaluate_task
from local_bench.model import EvaluationResult, load_candidate, load_task
from local_bench.optimization_feedback import optimization_result
from local_bench.program_bridge import materialize_program
from local_bench.prompting import _diversity_themes, feedback_context
from local_bench.scoped_reference import admit_reference
from local_bench.search_api import CandidateFeedback, SearchContext, TaskEvaluationOptions
from local_bench.search_session import prepare_search_problem, search_feedback
from local_bench.task_inventory import CORE_TASK_DIRECTORIES
from local_bench.verifier import verify_candidate
from test_bin_timing_and_placement_components import snapshot_fixture

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "AgenticBench/local_tasks"
NAME = "tp_gnn_post_placement_timing"
TASK = TASKS / NAME


def test_global_task_initial_candidate_and_immutable_scope():
    spec = load_task(TASK)
    candidate = materialize_program(spec, spec.program_template.read_text(), ROOT)
    from test_tp_gnn_native_adapter_compat import assert_historical_initial_candidate
    assert_historical_initial_candidate(spec, candidate,
        "d3d1f6a44a95c2a278c3556df12aac981aec8f94446804521e96f049d873f993")
    verify_candidate(spec, candidate)
    assert spec.kind == NAME
    assert spec.start_stage == "partition" and spec.stop_stage == "hbt"
    assert spec.upstream_stages == ["input", "floorplan"]
    assert spec.allowed_config_paths == ["partition.algorithm"]
    assert spec.allowed_source_paths == ["Place-3D/src/partition/lol/methods/tp_gnn.py"]
    config = yaml.safe_load(spec.base_config.read_text())
    assert config["partition"]["config"] == "config/partition/tp_gnn_fast.json"
    timing.validate_recipe(config, ROOT, expected_decoder="global-2way")
    fast = json.loads((ROOT / "Place-3D" / config["partition"]["config"]).read_text())["partition"]
    bins = json.loads((ROOT / "Place-3D/config/partition/tp_gnn_spatial_bin.json").read_text())["partition"]
    assert {k: v for k, v in bins.items() if k not in {
        "tp_gnn_decoder", "tp_gnn_bins_x", "tp_gnn_bins_y", "tp_gnn_bin_balance_tolerance"}} == fast


def test_core_uses_timing_task_not_old_cut_task(capsys):
    assert len(CORE_TASK_DIRECTORIES) == 10 and CORE_TASK_DIRECTORIES[0] == NAME
    assert "tp_gnn_bin_timing" not in CORE_TASK_DIRECTORIES
    assert "tp_gnn_learning_objective_reference_20260923" not in CORE_TASK_DIRECTORIES
    main(["list", "--core"])
    assert json.loads(capsys.readouterr().out)[0]["id"] == NAME
    for retired in ("tp_gnn_bin_timing", "tp_gnn_learning_objective_reference_20260923"):
        with pytest.raises(FileNotFoundError):
            load_task(TASKS / retired)


def test_prompt_system_and_ask_tell_remain_timing_driven(tmp_path):
    spec = load_task(TASK)
    problem = prepare_search_problem(TaskEvaluationOptions(TASK, tmp_path / "out", open3dbench_root=ROOT))
    for text in (problem.description, problem.system_message):
        assert "global 2-way" in text
        assert "WNS" in text and "TNS" in text
        assert "without timing optimization" in text or "no timing" in text
        assert "bin-based partition" not in text and "each nonempty bin" not in text
        assert str(ROOT) not in text and "sha256" not in text
    assert "0.6 * WNS_cost + 0.4 * TNS_cost" in problem.description
    assert "Macro assignments are decided first" not in problem.description
    assert "global 2-way decoder" in " ".join(_diversity_themes(spec))
    raw = {"combined_score": 1.1, "final_global_wns_setup_ns": -1,
           "final_global_tns_setup_ns": -10, "timing_weighted_cut": 42,
           "timing_weighted_cut_score": 999, "spatial_bin_max_area_error": 0,
           "normalized_sta_cost": 1 / 1.1, "opt_passes": 0}
    result = optimization_result(EvaluationResult(raw, {"log": "/private/log"}), task_id=NAME)
    assert result.metrics["timing_weighted_cut"] == 42
    assert "timing_weighted_cut_score" not in result.metrics
    assert "spatial_bin_max_area_error" not in result.metrics
    assert "fixed global-2way" in result.artifacts["optimization_feedback"]
    fb = CandidateFeedback("c", "sha", "accepted", 1.1, metrics=raw)
    assert search_feedback(problem, fb).metrics == result.metrics
    assert "spatial_bin_max_area_error" not in feedback_context([fb.to_dict()], spec)


@pytest.mark.parametrize("actual,expected,accepted", [
    ("global-2way", "global-2way", True),
    ("spatial-bin", "spatial-bin", True),
    ("spatial-bin", "global-2way", False),
    ("global-2way", "spatial-bin", False),
    ("ta3d-hierarchical", "global-2way", False),
])
def test_native_decoder_identity_is_required(tmp_path, monkeypatch, actual, expected, accepted):
    snapshot_fixture(tmp_path, monkeypatch, decoder=actual)
    if not accepted:
        with pytest.raises(ValueError, match="partition artifacts"):
            timing.validate_snapshot(tmp_path, expected_decoder=expected)
    else:
        row = timing.validate_snapshot(tmp_path, expected_decoder=expected)
        assert row["partition_config"]["decoder"] == expected
        assert ("bin_max_area_error" in row) == (actual == "spatial-bin")


@pytest.mark.parametrize("wns,tns,expected", [(-1, -10, 1), (-.5, -5, 2), (-2, -5, 1/1.4),
                                            (-.8, -12, 1/.96), (0, 0, 1e12)])
def test_global_timing_score_does_not_restore_cut_fitness(monkeypatch, wns, tns, expected):
    seen = []
    def row(path, *, expected_decoder):
        seen.append((path, expected_decoder))
        return {"metrics": {"wns_setup": wns if path == "c" else -1,
                            "tns_setup": tns if path == "c" else -10},
                "partition_config": {"decoder": expected_decoder}, "hbt_model": {"test": 1}}
    monkeypatch.setattr(timing, "validate_snapshot", row)
    monkeypatch.setattr(metrics, "tp_gnn_learning_objective_metrics", lambda *a: {
        "partition_constraints_passed": 1., "timing_weighted_cut_score": 9999., "timing_weighted_cut": 42})
    result = metrics.extract_metrics(NAME, "c", "r", task_id=NAME)
    assert seen == [("c", "global-2way"), ("r", "global-2way")]
    assert result["combined_score"] == pytest.approx(expected)
    assert result["opt_passes"] == 0 and result["sta_wns_weight"] == .6 and result["sta_tns_weight"] == .4
    assert result["partition_decoder"] == "global-2way"
    assert result["timing_weighted_cut"] == 42
    assert "timing_weighted_cut_score" not in result and "spatial_bin_max_area_error" not in result


def test_wrong_task_kind_and_reference_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        metrics.extract_metrics(NAME, tmp_path, None, task_id="tp_gnn_bin_timing")
    spec = load_task(TASK)
    candidate = materialize_program(spec, spec.program_template.read_text(), ROOT)
    assert admit_reference(spec, candidate, ROOT, tmp_path, None)["mode"] == "FRESH"
    modified = {**candidate, "notes": "not the initial program"}
    with pytest.raises(ValueError, match="exact native"):
        admit_reference(spec, modified, ROOT, tmp_path, None)
    run = tmp_path / "run"
    run.mkdir()
    for old_id in ("tp_gnn_bin_timing", "tp_gnn_learning_objective_reference_20260923"):
        (tmp_path / "evaluation_result.json").write_text(json.dumps({"artifacts": {"task": old_id}}))
        with pytest.raises(ValueError, match="different task"):
            admit_reference(spec, candidate, ROOT, tmp_path, run)


def test_wrong_bin_config_stops_before_clone_or_native(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Wrong decoder reached native/clone")
    monkeypatch.setattr("local_bench.evaluation.isolated_clone", forbidden)
    monkeypatch.setattr("local_bench.evaluation._run_native_flow", forbidden)
    config = yaml.safe_load(load_task(TASK).base_config.read_text())
    config["partition"]["config"] = "config/partition/tp_gnn_spatial_bin.json"
    wrong_config = tmp_path / "wrong-bin.yaml"
    wrong_config.write_text(yaml.safe_dump(config))
    result = evaluate_task(TASK, load_task(TASK).initial_candidate, ROOT, tmp_path / "eval",
                          base_config=wrong_config, dry_run=True)
    assert result.metrics["runs_successfully"] == 0
    assert "global-2way" in result.artifacts["error_message"]
    assert result.artifacts["native_flow_execution"] == []


def test_global_snapshot_rejects_bad_hash_or_opt(tmp_path, monkeypatch):
    _, _, artifact = snapshot_fixture(tmp_path, monkeypatch, decoder="global-2way")
    artifact.write_text("tampered")
    with pytest.raises(ValueError, match="size changed|hash changed"):
        timing.validate_snapshot(tmp_path, expected_decoder="global-2way")
    (tmp_path / "60_opt").mkdir()
    with pytest.raises(ValueError, match="later physical stage"):
        timing.validate_snapshot(tmp_path, expected_decoder="global-2way")


def test_revised_prompt_matches_current_decoder_inputs_and_feedback(tmp_path):
    spec = load_task(TASK)
    problem = prepare_search_problem(TaskEvaluationOptions(TASK, tmp_path / "out", open3dbench_root=ROOT))
    text = " ".join(problem.description.split())
    headings = ("## Background", "## Your task", "## How embeddings become a partition",
                "## How solutions are evaluated", "## Function inputs",
                "## Implementation requirements", "## What to return")
    assert [text.index(h) for h in headings] == sorted(text.index(h) for h in headings)
    for statement in (
        "same seeded pair of embeddings, not from the raw labels",
        "mean centers are not renormalized", "Label 1 is top",
        "standard-cell supernodes are never split",
        "Neither area rounding nor final repair uses a cut or timing objective",
        "partition-independent two-dimensional (2D) prototype",
        "Macro counts, macro-only areas and tier-eligibility attributes are not loss inputs",
        "Sampled negatives are not opposite-tier labels",
        "tp_gnn_repair_moved_area_fraction",
        "Raw reference WNS/TNS are not separate fields",
        "current selected parent", "score of about 1.0417", "score reaches 1e12",
        "Missing metrics remain unknown, never zero",
    ):
        assert statement in text
    for name in ("combined_score", "final_global_wns_setup_ns", "final_global_tns_setup_ns",
                 "normalized_wns_cost", "normalized_tns_cost"):
        projected = optimization_result(EvaluationResult({"combined_score": 1, name: 1}, {}), task_id=NAME)
        assert name in projected.metrics and "`" + name + "`" in text
    assert "causal" not in text.lower() and "versus frozen" not in text.lower()
    system = " ".join(problem.system_message.split())
    assert "does not require both to improve" in system
    assert "without timing optimization" in system
    for marker in ("# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"):
        assert marker in text and marker in system and marker in problem.initial_program
    config = yaml.safe_load((TASK / "config.yaml").read_text())
    assert config["diff_based_evolution"] is False
    assert config["max_iterations"] == 50 and config["max_code_length"] == 18000
    assert config["evaluator"]["timeout"] == 21600
    assert config["database"] == {"population_size": 16, "archive_size": 10, "num_islands": 3}
    policy = yaml.safe_load((TASK / "task.yaml").read_text())["candidate_policy"]
    assert policy["max_changed_files"] == 1
    assert policy["max_added_lines"] == 360 and policy["max_deleted_lines"] == 12
    for literal in ("18,000", "3,000", "360 lines", "12 lines", "21,600"):
        assert literal in text


def test_documented_helpers_and_methods_match_loss_validator():
    from local_tasks.tp_gnn_loss_patch import (
        SAFE_TORCH_APIS, SAFE_FUNCTIONAL_APIS, FORBIDDEN_TENSOR_ATTRIBUTES,
    )
    from local_tasks.tp_gnn_timing_context import (
        _validate_objective, OBJECTIVE_ARGS,
    )
    text = (TASK / "prompt.md").read_text()
    for name in SAFE_TORCH_APIS | SAFE_FUNCTIONAL_APIS | FORBIDDEN_TENSOR_ATTRIBUTES:
        assert "`" + name + "`" in text
    program = (TASK / "initial_program.py").read_text()
    program = program.replace("    return -(", """    count = embeddings.numel()
    flat = embeddings.reshape(-1)
    peak = flat.amax()
    scale = min(max(1, source_batch_size), len(positive_scores))
    return -(""")
    _validate_objective(ast.parse(program).body[0], OBJECTIVE_ARGS)


@pytest.mark.parametrize("expression", ["embeddings.item()", "embeddings.backward()", "torch.load(embeddings)"])
def test_prompt_method_examples_do_not_expand_torch_or_autograd_permissions(expression):
    spec = load_task(TASK)
    program = spec.program_template.read_text().replace("    return -(",
        "    forbidden = " + expression + "\n    return -(")
    with pytest.raises(ValueError, match="forbidden tensor attribute|unsupported torch API"):
        materialize_program(spec, program, ROOT)


def test_current_timing_prompt_in_real_openevolve_ask_tell(tmp_path):
    """Exercise the native sampler, parser and population with synthetic receipts only."""
    pytest.importorskip("openevolve")
    from openevolve.config import Config
    from local_bench.openevolve_search import OpenEvolveAskTell

    spec = load_task(TASK)
    problem = prepare_search_problem(TaskEvaluationOptions(TASK, tmp_path / "out", open3dbench_root=ROOT))
    program = problem.initial_program.replace("    return -(", "    return 0.99 * -(")
    seen = []
    def generate(system, user):
        seen.append((system, user))
        return "```python\n" + program + "\n```"
    config = Config(diff_based_evolution=False)
    config.database.num_islands = 1
    backend = OpenEvolveAskTell(config, generate=generate)
    try:
        first = backend.ask(SearchContext(problem, 0, 3, (), None))[0]
        verify_candidate(spec, materialize_program(spec, first.program, ROOT))
        raw_metrics = {"combined_score": 1/.96, "runs_successfully": 1,
                       "partition_constraints_passed": 1,
                       "final_global_wns_setup_ns": -.08, "final_global_tns_setup_ns": -12,
                       "normalized_wns_cost": .8, "normalized_tns_cost": 1.2,
                       "tp_gnn_feasibility_weighted_cut_delta": 7,
                       "tp_gnn_repair_moved_area_fraction": .125,
                       "tp_gnn_positive_pair_cosine_median": .8,
                       "tp_gnn_negative_pair_cosine_median": .2,
                       "tp_gnn_positive_pair_sample_count": 64,
                       "tp_gnn_negative_pair_sample_count": 128,
                       "instance_count": 1000, "tp_gnn_optimizer_steps_completed": 50,
                       "tp_gnn_final_loss": None, "timing_weighted_cut_score": 999}
        good = search_feedback(problem, CandidateFeedback(first.proposal_id,
            hashlib.sha256(first.program.encode()).hexdigest(), "accepted", 1/.96,
            metrics=raw_metrics, artifacts={"manifest": "PRIVATE_RECEIPT"}))
        backend.tell([good])
        second = backend.ask(SearchContext(problem, 1, 2, (good,), good))[0]
        system, user = seen[-1]
        assert system == problem.system_message + "\n\n" + problem.description
        assert first.program in user  # Current selected parent, not just the initial interface.
        for name in ("final_global_wns_setup_ns", "final_global_tns_setup_ns",
                     "normalized_wns_cost", "normalized_tns_cost", "tp_gnn_feasibility_weighted_cut_delta",
                     "tp_gnn_repair_moved_area_fraction", "tp_gnn_positive_pair_cosine_median",
                     "tp_gnn_negative_pair_cosine_median"):
            assert name in user
        assert "PRIVATE_RECEIPT" not in user and "timing_weighted_cut_score" not in user
        assert "instance_count" not in user and "tp_gnn_optimizer_steps_completed" not in user
        assert "tp_gnn_final_loss" not in user  # Missing is omitted, not zero.
        assert "# Selected parent" in user
        # With only one measured source, there is no distinct reference example.
        assert "# One different evaluated example" not in user
        assert "combined_score" in user and "## How embeddings become a partition" in system
        bad = search_feedback(problem, CandidateFeedback(second.proposal_id,
            hashlib.sha256(second.program.encode()).hexdigest(), "rejected", None,
            error_message="TP-GNN objective uses forbidden tensor attribute detach"))
        backend.tell([bad])
        third = backend.ask(SearchContext(problem, 2, 1, (good, bad), good))[0]
        assert "forbidden tensor attribute detach" in seen[-1][1]
        assert "not a scored parent" in seen[-1][1] and '"score": null' in seen[-1][1]
        assert len(backend.database.programs) == 1
        assert third.program
    finally:
        backend.close()
