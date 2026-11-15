"""Prompt/definition regressions only: no model, Docker, GPU or native EDA."""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from local_bench.benchmark_input import load_benchmark_input_profile
from local_bench.evostage_adapter import EvolutionOptions, run_evolution
from local_bench.metrics import _ratio_score
from local_bench.model import load_candidate, load_task
from local_bench.prompting import (
    PARTITION_DIVERSITY_THEMES, TIMING_PARTITION_DIVERSITY_THEMES, DIVERSITY_THEMES,
    ROUTING_NET_ORDER_DIVERSITY_THEMES, ROUTING_SOURCE_DIVERSITY_THEMES,
    _diversity_themes, build_problem_description, export_openevolve_bundle, task_contract,
)
from local_tasks import routing_scalar_common as ordering

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TASKS = (
    "tp_gnn_post_placement_timing",
    "placement_parameters_reference_20260923",
    "placement_initialization",
    "placement_preconditioner",
)
ROUTING_TASKS = tuple("global_routing_" + name for name in
                      ("ripup_policy", "maze_cost", "iteration_controller"))


@pytest.mark.parametrize("task", REFERENCE_TASKS + ROUTING_TASKS)
def test_exploration_themes_match_task_domain(task):
    spec = load_task(ROOT / "local_tasks" / task)
    themes = _diversity_themes(spec)
    if task == REFERENCE_TASKS[0]:
        assert "post-placement WNS/TNS" in " ".join(themes)
        assert "density overflow" not in " ".join(themes)
        assert "wirelength pressure" not in " ".join(themes)
    elif task in REFERENCE_TASKS:
        assert themes == DIVERSITY_THEMES
    elif task.endswith("net_ordering"):
        assert themes == ROUTING_NET_ORDER_DIVERSITY_THEMES
    else:
        assert themes == ROUTING_SOURCE_DIVERSITY_THEMES


@pytest.mark.parametrize("task", REFERENCE_TASKS + ROUTING_TASKS)
def test_all_seven_prompts_render_with_current_read_only_source(task, tmp_path):
    spec = load_task(ROOT / "local_tasks" / task)
    description = build_problem_description(spec, load_candidate(spec.initial_candidate), ROOT.parent)
    assert "unknown" in description and "zero" in description
    assert "sha256" not in description
    assert "# Benchmark contract" not in description
    assert task_contract(spec)["execution"]["boundary"] == spec.evaluation_boundary
    assert "EVOLVE-BLOCK-START" in description
    config = yaml.safe_load((spec.root / "config.yaml").read_text())
    assert config["diff_based_evolution"] is False
    assert config["max_iterations"] == 50
    bundle = export_openevolve_bundle(spec, load_candidate(spec.initial_candidate), ROOT.parent, tmp_path)
    exported = yaml.safe_load(Path(bundle["config"]).read_text())
    assert exported["diff_based_evolution"] is False
    assert exported["max_iterations"] == config["max_iterations"]
    assert exported["database"] == config["database"]
    assert exported["evaluator"]["timeout"] == config["evaluator"]["timeout"]


def test_export_protects_complete_program_protocol_for_legacy_template(tmp_path):
    spec = load_task(ROOT / "local_tasks/tp_gnn_post_placement_timing")
    bundle = export_openevolve_bundle(spec, load_candidate(spec.initial_candidate), ROOT.parent, tmp_path)
    assert yaml.safe_load(Path(bundle["config"]).read_text())["diff_based_evolution"] is False


@pytest.mark.parametrize("family", ["reference_20260923", "reference_20261004"])
@pytest.mark.parametrize("case", ["bp_fe", "bp", "ariane133", "swerv_wrapper"])
@pytest.mark.parametrize("task", ["placement_parameters_reference_20260923"])
def test_rendered_prompt_binds_actual_case_profile_and_boundary(task, case, family):
    spec = load_task(ROOT / "local_tasks" / task)
    profile = load_benchmark_input_profile(case + "_lol_ng45_" + family)
    assert spec.task_id in profile.compatible_tasks
    selected = replace(spec, base_config=profile.config, benchmark_input_id=profile.profile_id)
    description = build_problem_description(selected, load_candidate(spec.initial_candidate))
    assert "Design: " + case + ";" in description
    assert profile.profile_id not in description
    assert str(profile.config) not in description
    contract = task_contract(selected)
    assert contract["benchmark_input"] == profile.profile_id
    assert contract["base_config"] == str(profile.config)
    assert contract["execution"]["boundary"] == spec.evaluation_boundary
    assert hashlib.sha256(profile.config.read_bytes()).hexdigest() not in description
    # Prompt selection must not mutate the task's historical default or gates.
    assert spec.benchmark_input_id == "bp_fe_lol_ng45_reference_20260923"
    assert selected.value_constraints == spec.value_constraints


@pytest.mark.parametrize("benchmark_id", [None, "swerv_wrapper_lol_ng45_reference_20261004"])
def test_evostage_override_reaches_generated_contract_before_any_model_call(tmp_path, benchmark_id):
    spec = load_task(ROOT / "local_tasks" / REFERENCE_TASKS[0])
    config = load_benchmark_input_profile("swerv_wrapper_lol_ng45_reference_20261004").config
    options = EvolutionOptions(spec.root, tmp_path, tmp_path / "out", "verify",
                               base_config=config, benchmark_id=benchmark_id)
    class StopAfterPrompt(Exception):
        pass
    with patch("local_bench.evostage_adapter._load_external", return_value={}), \
         patch("local_bench.evostage_adapter.EvolutionHarness", return_value=SimpleNamespace(spec=spec)), \
         patch("local_bench.evostage_adapter.evolution_prompt", side_effect=StopAfterPrompt) as render:
        with pytest.raises(StopAfterPrompt):
            run_evolution("hillclimb", None, options)
    actual = render.call_args.args[0]
    assert actual.base_config == config
    assert actual.benchmark_input_id == benchmark_id  # Never falsely retain bp_fe.
    assert spec.base_config != config


def test_tp_tensor_contract_and_source_excerpt_include_real_loss_call():
    spec = load_task(ROOT / "local_tasks" / REFERENCE_TASKS[0])
    description = build_problem_description(spec, load_candidate(spec.initial_candidate), ROOT.parent)
    for text in ("[P]", "[Q]", "[N, D]", "torch.long", "Python integers",
                 "candidate-created mask", "graph_weights", "node_features",
                 "F.logsigmoid(positive_scores).sum()", "fixed-order gradient accumulation"):
        assert text in description
    assert "producer/controller is not connected yet" not in description


def test_ripup_scalar_language_rejects_dynamic_table_indexing():
    from local_tasks.global_routing_ripup_policy import program_adapter as adapter
    spec = load_task(ROOT / "local_tasks/global_routing_ripup_policy")
    from local_bench.program_bridge import materialize_program
    with pytest.raises(ValueError):
        materialize_program(spec, 'def score_ripup_edge(ctx):\n    t = (0.1, 0.2)\n    return t[ctx.die_id]\n', ROOT.parent)


@pytest.mark.parametrize("reference,candidate,expected", [(0, 0, 1), (0, 1, 0), (2, 1, 2), (2, 0, 2e12)])
def test_ratio_reader_preserves_existing_score(reference, candidate, expected):
    # Historical metric math remains readable; current task prompts document
    # their own objective rather than this former cut-score objective.
    assert _ratio_score(candidate, reference, 1) == expected


def test_iteration_context_does_not_promise_nonexistent_clamps():
    prompt = (ROOT / "local_tasks" / ROUTING_TASKS[-1] / "prompt.md").read_text()
    assert "without an explicit upper clamp" in prompt
    assert "search_window = max(0, enlargement) / max(1, max(x_grid, y_grid))" in prompt
    assert "reroute_phase = max(0, iteration) / max(1, congestion_iterations)" in prompt
    assert "All other\nfields are normalized to `[0, 1]`" not in prompt


def test_latest_profile_declares_ordering_interface_without_changing_boundary():
    # Catalog compatibility is not native scoring qualification. The current
    # profiles already include ordering; prompt changes must not edit admission.
    for case in ("bp_fe", "bp", "ariane133", "swerv_wrapper"):
        profile = load_benchmark_input_profile(case + "_lol_ng45_reference_20261004")
        assert "global_routing_net_ordering_source" in profile.compatible_tasks
    assert task_contract(load_task(ROOT / "local_tasks" / REFERENCE_TASKS[1]))[
        "execution"]["boundary"] == "dreamplace-global-placement"
