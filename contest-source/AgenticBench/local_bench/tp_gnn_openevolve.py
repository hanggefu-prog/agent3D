"""Model-facing projection for native OpenEvolve; raw receipts stay untouched."""
from __future__ import annotations

import json
import math
import re

from .model import EvaluationResult


TASK_ID = "tp_gnn_learning_objective_reference_20260923"
TEMPLATES = ("full_rewrite_user.txt", "previous_attempt.txt", "top_program.txt",
             "inspiration_program.txt")
METRICS = frozenset({
    "combined_score", "runs_successfully", "timing_weighted_cut_score",
    "timing_weighted_cut", "timing_weighted_cut_ratio", "timing_total_net_weight",
    "cut_net_count", "cut_net_ratio", "cut_pin_count", "area_balance_error",
    "macro_count_error", "macro_area_error", "partition_complete",
    "partition_constraints_passed", "instance_count", "top_count", "bottom_count",
    "tp_gnn_final_loss", "tp_gnn_epochs_completed", "tp_gnn_optimizer_steps_completed",
    "tp_gnn_embedding_total_variance", "tp_gnn_embedding_pair_cosine_median",
    "tp_gnn_area_balance_cut_delta", "tp_gnn_feasibility_cut_delta",
    "tp_gnn_training_runtime_seconds",
} | {"tp_gnn_" + stage + "_" + metric
     for stage in ("raw", "balanced", "feasible")
     for metric in ("cut_net_count", "timing_weighted_cut")})


def _reason(value):
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"(?<!\w)(?:/|\./|\.\./|[A-Za-z]:\\)\S+", "<path>", text)
    text = re.sub(r"\b[0-9a-f]{40,64}\b", "<digest>", text)
    return text[:1200]


def optimization_result(result: EvaluationResult, *, require_score: bool = True) -> EvaluationResult:
    """Whitelist actual metrics, never invent missing measurements or rewards."""
    score = result.metrics.get("combined_score")
    if require_score and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
        raise ValueError("TP-GNN OpenEvolve result requires a finite combined_score")
    metrics = {
        key: value if isinstance(value, (int, float)) and math.isfinite(value) else None
        for key, value in result.metrics.items() if key in METRICS
        # Optional diagnostics must be actual finite measurements, not placeholders.
        if not key.startswith("tp_gnn_") or (
            not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    }
    feedback = {
        "metrics": metrics,
        "rejection_reason": _reason(result.artifacts.get("error_message")),
        "interpretation": (
            "Maximize combined_score only after feasibility gates pass. "
            "Lower final timing_weighted_cut is better; training loss and embedding "
            "statistics are diagnostics, not rewards. Missing metrics are unknown, "
            "not zero. A zero score alone does not identify a failure cause."
        ),
    }
    return EvaluationResult(metrics=metrics, artifacts={
        "optimization_feedback": json.dumps(feedback, ensure_ascii=False, allow_nan=False)
    })
