"""Native OpenEvolve boundary tests; no provider, GPU or EDA execution."""
import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

from local_bench.model import EvaluationResult, load_candidate, load_task
from local_bench import openevolve_adapter as bridge
from local_bench.prompting import export_openevolve_bundle
from local_bench.reference_catalog import task_digest
from local_bench.tp_gnn_openevolve import TASK_ID, TEMPLATES, optimization_result

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "AgenticBench/local_tasks/tp_gnn_post_placement_timing"


def receipt():
    return EvaluationResult(metrics={
        "combined_score": 1.0256789123, "timing_weighted_cut": 412.3,
        "tp_gnn_final_loss": 9.1, "area_balance_error": None,
        "runtime_container_count": 2,
    }, artifacts={
        "error_message": "Rejected /private/internal/run.json " + "a" * 64,
        "manifest": {"path": "/private/internal"},
        "container_output_tail": "PRIVATE_RAW_LOG", "patch": "PRIVATE_PATCH",
    })


def test_projection_preserves_raw_receipt_and_unknowns():
    raw = receipt()
    before = copy.deepcopy(raw.to_dict())
    projected = optimization_result(raw)
    assert raw.to_dict() == before
    assert projected.metrics["area_balance_error"] is None
    assert "runtime_container_count" not in projected.metrics
    assert set(projected.artifacts) == {"optimization_feedback"}
    text = projected.artifacts["optimization_feedback"]
    assert "PRIVATE" not in text and "/private/" not in text and "a" * 64 not in text
    assert json.loads(text)["metrics"]["combined_score"] == 1.0256789123


@pytest.mark.parametrize("value", [None, True, "0", float("nan"), float("inf")])
def test_unavailable_diagnostics_are_omitted_from_tell(value):
    raw = receipt()
    raw.metrics["tp_gnn_final_loss"] = value
    raw.metrics["legacy_embedding_diagnostic"] = 123
    raw.metrics["tp_gnn_feasibility_cut_delta"] = 0
    projected = optimization_result(raw)
    assert "tp_gnn_final_loss" not in projected.metrics
    assert "legacy_embedding_diagnostic" not in projected.metrics
    assert projected.metrics["tp_gnn_feasibility_cut_delta"] == 0
    assert "tp_gnn_final_loss" not in json.loads(
        projected.artifacts["optimization_feedback"]
    )["metrics"]
    assert "tp_gnn_final_loss" in raw.metrics


@pytest.mark.parametrize("score", [None, True, float("nan"), float("inf"), "1.0"])
def test_missing_or_invalid_reward_is_not_invented(score):
    raw = receipt()
    raw.metrics["combined_score"] = score
    with pytest.raises(ValueError, match="finite combined_score"):
        optimization_result(raw)


@pytest.mark.parametrize("docker", [False, True])
def test_actual_evaluator_cache_boundary_projects_feedback(tmp_path, monkeypatch, docker):
    raw = receipt()
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# Cached boundary test only\n")
    monkeypatch.setenv("OPEN3DBENCH_ROOT", str(ROOT))
    monkeypatch.setenv("OPEN3D_LOCAL_EVAL_ROOT", str(tmp_path / "evaluations"))
    monkeypatch.delenv("OPENAGENTIC3D_BASE_CONFIG", raising=False)
    if docker:
        monkeypatch.setenv("OPENAGENTIC3D_EVAL_DOCKER_IMAGE", "unused-cached-image")
    else:
        monkeypatch.delenv("OPENAGENTIC3D_EVAL_DOCKER_IMAGE", raising=False)
    monkeypatch.setattr(bridge, "_context_digest", lambda *args: "fixture")
    monkeypatch.setattr(bridge, "_cached_result", lambda *args: raw)
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached boundary must not execute physical work")
    monkeypatch.setattr(bridge, "_materialize_candidate", forbidden)
    monkeypatch.setattr(bridge, "evaluate_task", forbidden)
    result = bridge.evaluate_from_task_dir(str(TASK), str(candidate))
    artifacts = result.artifacts if hasattr(result, "artifacts") else result["artifacts"]
    assert set(artifacts) == {"optimization_feedback"}
    assert "manifest" in raw.artifacts


def test_other_tasks_keep_their_feedback():
    result = bridge._to_openevolve_result(receipt(), "global_routing_parameters")
    artifacts = result.artifacts if hasattr(result, "artifacts") else result["artifacts"]
    assert "manifest" in artifacts


def test_templates_participate_in_task_identity(tmp_path):
    target = tmp_path / "task"
    shutil.copytree(TASK, target)
    before = task_digest(target)
    path = target / "openevolve_templates/full_rewrite_user.txt"
    path.parent.mkdir()
    path.write_text("Changed task-specific model instructions.\n")
    assert task_digest(target) != before


def test_evolution_prompt_prioritizes_search_without_weakening_contract():
    from local_tasks.tp_gnn_loss_patch import (
        SAFE_TORCH_APIS, SAFE_FUNCTIONAL_APIS,
    )
    prompt = (TASK / "prompt.md").read_text()
    headings = ("## Background", "## Your task", "## How solutions are evaluated",
                "## Improving the loss", "## Using previous results", "## Function inputs",
                "## Implementation requirements", "### Numerical safety", "## What to return")
    assert [prompt.index(heading) for heading in headings] == sorted(
        prompt.index(heading) for heading in headings
    )
    for name in SAFE_TORCH_APIS | SAFE_FUNCTIONAL_APIS:
        assert "`" + name + "`" in prompt
    for rule in ("18,000", "3,000", "0.5 +/- 0.03", "0.5 +/- 0.05", "1e-12",
                 "No scatter/segment helper", "zero coverage", "not a driver",
                 "not a cut probability", "population variance", "graph_net_ids",
                 "Missing metrics remain unknown, never zero"):
        assert rule in prompt
    assert "causal" not in prompt.lower()
    assert "versus frozen" not in prompt.lower()
    for full, short in (("three-dimensional", "3D"), ("logic-on-logic", "LoL"),
                        ("hybrid-bonding terminals", "HBTs"),
                        ("tier partitioning with graph neural networks", "TP-GNN"),
                        ("graph neural network", "GNN"),
                        ("static timing analysis", "STA"),
                        ("abstract syntax tree", "AST"), ("root-mean-square", "RMS")):
        assert full + " (" + short + ")" in " ".join(prompt.split())
        # For GNN, ignore the already-explained method name TP-GNN.
        checked = prompt.replace("TP-GNN", "method") if short == "GNN" else prompt
        assert checked.lower().index(full.split()[0].lower()) < checked.index(short)
    config = yaml.safe_load((TASK / "config.yaml").read_text())
    assert config["diff_based_evolution"] is False
    assert config["max_code_length"] == 18000


def test_native_sampler_export_and_full_rewrite_roundtrip(tmp_path):
    pytest.importorskip("openevolve", reason="Set PYTHONPATH to native OpenEvolve checkout")
    from openevolve.config import Config
    from openevolve.prompt.sampler import PromptSampler
    from openevolve.utils.code_utils import parse_full_rewrite
    from local_bench.program_bridge import materialize_program

    spec = load_task(TASK)
    bundle = export_openevolve_bundle(spec, load_candidate(spec.initial_candidate),
                                     ROOT, tmp_path / "bundle")
    config = Config.from_yaml(bundle["config"])
    assert config.diff_based_evolution is False
    assert set(p.name for p in Path(config.prompt.template_dir).iterdir()) == set(TEMPLATES)
    sampler = PromptSampler(config.prompt)
    program = Path(bundle["initial_program"]).read_text()
    feedback = optimization_result(receipt())
    example = {"code": program, "metrics": feedback.metrics,
               "metadata": {"parent_metrics": {"combined_score": 1, "tp_gnn_final_loss": 8}}}
    rendered = sampler.build_prompt(current_program=program, parent_program=program,
        program_metrics=feedback.metrics, previous_programs=[example],
        top_programs=[example], inspirations=[example],
        diff_based_evolution=config.diff_based_evolution, program_artifacts=feedback.artifacts)
    assert rendered["system"] == config.prompt.system_message
    user = rendered["user"]
    assert "1.0256789123" in user and "area_balance_error" in user
    assert "graph_net_ids" in user and "Selected evaluated example" in user
    assert "Alternative program" in user and "Selected program" in user
    assert "Excellent" not in user and "All metrics improved" not in user
    assert "PRIVATE_RAW_LOG" not in user and "/private/internal" not in user
    assert "not a chronological" in user and "lower physical costs" in user
    for text in (rendered["system"], user):
        plain = " ".join(text.split())
        assert "Try one main change at a time" in plain
        assert "causal" not in plain.lower()
    assert "You may combine ideas that have already improved comparable legal partitions" in " ".join(rendered["system"].split())
    assert "Keep useful parts of the parent" in user
    assert "## Improving the loss" in rendered["system"]
    assert "not a cut probability or exact" in rendered["system"]
    parsed = parse_full_rewrite("```python\n" + program + "\n```")
    materialize_program(spec, parsed, ROOT)


def test_current_routing_export_gets_shared_templates(tmp_path):
    spec = load_task(TASK.parent / "global_routing_ripup_policy")
    bundle = export_openevolve_bundle(spec, load_candidate(spec.initial_candidate),
                                     ROOT, tmp_path / "bundle")
    config = yaml.safe_load(Path(bundle["config"]).read_text())
    assert set(p.name for p in Path(config["prompt"]["template_dir"]).iterdir()) == set(TEMPLATES)
