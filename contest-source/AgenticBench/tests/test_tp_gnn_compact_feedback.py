"""Compact model feedback and audit-only diagnostics; no EDA or model API."""
import json

import numpy as np
import pytest

from local_bench.model import EvaluationResult
from local_bench.optimization_feedback import optimization_result, TP_TIMING_GROUPS
from test_tp_gnn_timing_context import native_modules, train

TASK = "tp_gnn_post_placement_timing"


def test_grouped_projection_preserves_raw_and_primary_score():
    metrics = {key: .5 for group in TP_TIMING_GROUPS.values() for key in group}
    metrics.update(combined_score=1.25, instance_count=100, total_net_weight=12,
                   opt_passes=0, sta_graph_complete=1, normalized_sta_cost=.8,
                   tp_gnn_optimizer_steps_completed=50, tp_gnn_training_runtime_seconds=10,
                   tp_gnn_embedding_pair_cosine_median=.9, cut_pin_count=7,
                   tp_gnn_feasibility_cut_delta=9, tp_gnn_positive_pair_cosine_median=None)
    raw = EvaluationResult(metrics, {"raw_log": "PRIVATE_RECEIPT"})
    before = raw.to_dict()
    projected = optimization_result(raw, task_id=TASK)
    assert raw.to_dict() == before
    assert projected.metrics["combined_score"] == 1.25
    expected = {key for group in TP_TIMING_GROUPS.values() for key in group}
    assert set(projected.metrics) == expected - {"tp_gnn_positive_pair_cosine_median"}
    payload = json.loads(projected.artifacts["optimization_feedback"])
    assert "metrics" not in payload and "PRIVATE_RECEIPT" not in str(payload)
    for group, names in TP_TIMING_GROUPS.items():
        assert payload[group] == {key: projected.metrics[key] for key in names if key in projected.metrics}
    assert "combined_score only" in payload["selection_target"]
    pytest.importorskip("openevolve")
    from local_bench.openevolve_adapter import _to_openevolve_result
    direct = _to_openevolve_result(raw, TASK)
    assert direct.metrics == projected.metrics and direct.artifacts == projected.artifacts


def test_failure_and_missing_values_are_not_success():
    raw = EvaluationResult({"runs_successfully": 0, "tp_gnn_repair_moved_area_fraction": None,
                            "tp_gnn_negative_pair_cosine_median": float("nan")},
                           {"error_message": "Timeout at /private/run/training.log"})
    projected = optimization_result(raw, task_id=TASK, require_score=False)
    payload = json.loads(projected.artifacts["optimization_feedback"])
    assert projected.metrics == {"runs_successfully": 0}
    assert payload["fitness"] == {} and payload["decoder_repair"] == {}
    assert "Timeout" in payload["status"]["rejection_reason"]
    assert "/private/" not in str(payload)


def test_pair_audit_uses_actual_pairs_and_is_bounded(monkeypatch):
    mod, _ = native_modules(monkeypatch)
    matrix = np.array([[1., 0.], [1., 0.], [-1., 0.]])
    pairs = ([0]*5000, [1]*5000, [0]*5000, [2]*5000)
    state = np.random.get_state()
    audit = mod.training_pair_cosine_diagnostics(matrix, pairs)
    assert np.array_equal(state[1], np.random.get_state()[1])
    assert audit["positive_cosine_median"] == 1
    assert audit["negative_cosine_median"] == -1
    assert audit["positive_sample_count"] == audit["negative_sample_count"] == 4096
    assert audit["positive_population_count"] == 5000
    empty = mod.training_pair_cosine_diagnostics(matrix, ([], [], [], []))
    assert empty["positive_cosine_median"] is None
    invalid = mod.training_pair_cosine_diagnostics(matrix * 0, ([0], [1], [0], [2]))
    assert invalid["negative_cosine_median"] is None
    with pytest.raises(ValueError):
        mod.training_pair_cosine_diagnostics(matrix, ([0], [], [], []))


def test_audit_does_not_change_training_and_is_repeatable(monkeypatch):
    mod, _ = native_modules(monkeypatch)
    embeddings, report = train(mod, False)
    repeated, second = train(mod, False)
    assert embeddings == repeated
    assert report["final_embedding_diagnostics"] == second["final_embedding_diagnostics"]
    pair = report["final_embedding_diagnostics"]["training_pair_cosines"]
    assert pair["positive_sample_count"] > 0 and pair["negative_sample_count"] > 0
    for label in ("positive", "negative"):
        assert -1.000001 <= pair[label + "_cosine_median"] <= 1.000001
    # Disable observation only, leaving training and its random state untouched.
    monkeypatch.setattr(mod, "training_pair_cosine_diagnostics", lambda *args: {})
    no_audit, plain = train(mod, False)
    assert no_audit == embeddings
    for key in ("final_loss", "optimizer_steps_completed", "epochs_completed"):
        assert plain[key] == report[key]
