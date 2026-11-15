"""Real CPU/autograd boundary tests; no model API, EDA or benchmark launch."""
import ast
import math
import sys
import types
from pathlib import Path

import pytest

from local_bench.model import load_task, load_candidate
from local_bench.program_bridge import materialize_program, initial_program_text
from local_bench.program_safety import single_definition
from local_bench.prompting import build_problem_description, composed_system_message, feedback_context
from local_bench.verifier import verify_candidate
from local_tasks import tp_gnn_timing_context as adapter

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "AgenticBench/local_tasks/tp_gnn_post_placement_timing"


def test_star_preserves_all_nets_weights_and_contracted_identity():
    groups = [["a", "b"], ["c"], ["d"]]
    context = adapter._agentic_loss_graph(groups,
        [["a", "b"], ["a", "c", "d", "c"], ["d", "c"], ["c", "d"], []],
        [1, 4, 2, 3, 1], [5, 2, 1], "partition-last")
    assert context == ([0, 0, 1, 1], [1, 2, 2, 2], [1, 1, 2, 3],
                       [2, 2, 2, 3], [1, 4, 2, 3, 1], [5, 2, 1])


def test_high_degree_net_is_linear_and_not_truncated():
    groups = [[str(i)] for i in range(200)]
    context = adapter._agentic_loss_graph(groups, [[str(i) for i in range(200)]],
                                         [7], [1] * 200, "partition-last")
    assert len(context[0]) == 199
    assert sum(context[3]) == pytest.approx(7)


@pytest.mark.parametrize("weights", [None, [], [float("nan")], [0.5], [float("inf")]])
def test_invalid_weights_fail_closed(weights):
    with pytest.raises(ValueError):
        adapter._agentic_loss_graph([["a"], ["b"]], [["a", "b"]], weights, [1, 1], "partition-last")


@pytest.mark.parametrize("groups,nets,areas,regime", [
    ([["a"], ["a"]], [["a"]], [1, 1], "partition-last"),
    ([["a"]], [["missing"]], [1], "partition-last"),
    ([["a"]], [["a"]], [0], "partition-last"),
    ([["a"]], [["a"]], [1], "partition-first"),
])
def test_invalid_context_fails_closed(groups, nets, areas, regime):
    with pytest.raises(ValueError):
        adapter._agentic_loss_graph(groups, nets, [1], areas, regime)


def program():
    task = load_task(TASK)
    return initial_program_text(task, load_candidate(task.initial_candidate))


@pytest.mark.parametrize("body", [
    "return model(x, hierarchy_tensor).sum()", "return config.epochs * positive_scores.sum()",
    "node_features += 1\n    return positive_scores.sum()",
    "node_features[0] = 1\n    return positive_scores.sum()",
    "return graph_weights.fill_(1).sum()", "return graph_scores.data.sum()",
    "torch = embeddings\n    return torch.sum()",
])
def test_context_cannot_mutate_or_reach_enclosing_training(body):
    candidate = program().split('    """')[0] + "    " + body + "\n"
    with pytest.raises(ValueError):
        adapter.materialize(candidate, ROOT)


def test_patch_scope_and_old_signature_rejected():
    task = load_task(TASK)
    candidate = materialize_program(task, program(), ROOT)
    proof = verify_candidate(task, candidate)
    assert proof["patch"]["paths"] == [adapter.TP_GNN]
    assert proof["patch"]["added_lines"] <= task.max_added_lines
    legacy = "def compute_tp_gnn_loss(positive_scores, negative_scores, embeddings):\n    return positive_scores.sum()\n"
    with pytest.raises(ValueError, match="signature"):
        materialize_program(task, legacy, ROOT)


def native_modules(monkeypatch, candidate=None):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "Place-3D"))
    source = (ROOT / adapter.TP_GNN).read_text()
    objective, _ = single_definition(candidate or program(), ast.FunctionDef, "compute_tp_gnn_loss")
    result = []
    for name, text in (("tp_original", source), ("tp_context", adapter._patch_tp_gnn(source, objective))):
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        exec(compile(text, str(ROOT / adapter.TP_GNN), "exec"), mod.__dict__)
        result.append(mod)
    return result


def train(mod, extended, *, empty=False):
    n = 12
    features = [[float((i * (j + 1)) % 7) for j in range(10)] for i in range(n)]
    neighbors = [{(i - 1) % n, (i + 1) % n} for i in range(n)]
    config = mod.TPGNNConfig(device="cpu", embedding_dim=8, epochs=2, min_epochs=2,
        batch_size=6, random_walks=1, walk_length=1, negative_samples=2,
        feature_regime="partition-last")
    args = [features, [0]*n, 0, neighbors, config]
    if extended:
        nets = [] if empty else [[str(i), str((i + 1) % n)] for i in range(n)]
        context = adapter._agentic_loss_graph([[str(i)] for i in range(n)], nets,
            [] if empty else [1 + i / n for i in range(n)], [1]*n, "partition-last")
        args.append(context)
    return mod.train_random_walk_embeddings(*args)


@pytest.mark.parametrize("empty", [False, True])
def test_real_training_initial_loss_preserves_embeddings_and_budget(monkeypatch, empty):
    old, new = native_modules(monkeypatch)
    original, original_report = train(old, False)
    actual, report = train(new, True, empty=empty)
    assert actual == original
    assert report["final_loss"] == original_report["final_loss"]
    assert report["optimizer_steps_completed"] == original_report["optimizer_steps_completed"]
    assert report["loss_context_interface"] == "timing-context-v2"
    assert report["loss_context_star_edges"] == (0 if empty else 12)


@pytest.mark.parametrize("empty", [False, True])
def test_timing_structural_loss_real_autograd_and_repeatability(monkeypatch, empty):
    candidate = program().replace("    return -(", "    baseline = -(")
    candidate = candidate.replace("# EVOLVE-BLOCK-END", """    penalty = (graph_weights * (1 - graph_scores)).sum() / graph_weights.sum().clamp(min=1e-12)
    return baseline + 0.2 * penalty
# EVOLVE-BLOCK-END""")
    adapter.materialize(candidate, ROOT)
    old, new = native_modules(monkeypatch, candidate)
    actual, report = train(new, True, empty=empty)
    repeated, repeated_report = train(new, True, empty=empty)
    assert actual == repeated
    assert report["final_loss"] == repeated_report["final_loss"]
    assert math.isfinite(report["final_loss"])
    original, _ = train(old, False)
    assert (actual == original) == empty


def test_model_text_semantics_not_plumbing():
    task = load_task(TASK)
    prompt = build_problem_description(task, load_candidate(task.initial_candidate), ROOT)
    system = composed_system_message(task)
    for text in (prompt, system):
        for absent in ("sha256", "20_partition/", "qualification", "reference_20260923", "threads", str(ROOT)):
            assert absent not in text
    for signal in adapter.OBJECTIVE_ARGS:
        assert signal in prompt
    assert "WNS" in system and "TNS" in system
    assert "graph_scores" in prompt


def test_feedback_retains_signals_not_provenance():
    task = load_task(TASK)
    feedback = feedback_context([{
        "order": 1, "status": "failed", "score": None,
        "metrics": {"timing_weighted_cut": 42, "tp_gnn_final_loss": None,
                    "manifest_path": "/private/experiment/run.json"},
        "artifacts": {"patch": "hidden patch", "error_message": "Bad shape at /private/experiment/tensor"},
    }], task)
    assert '"timing_weighted_cut": 42' in feedback
    assert '"tp_gnn_final_loss": null' not in feedback  # Missing optional diagnostics are omitted.
    assert "Bad shape" in feedback
    assert "/private/" not in feedback and "hidden patch" not in feedback
