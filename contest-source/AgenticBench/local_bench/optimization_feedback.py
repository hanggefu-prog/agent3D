"""Task-scoped model inputs. Full evaluator receipts remain host-owned."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .model import EvaluationResult
from . import tp_gnn_openevolve as tp
from .feedback_semantics import metric_definitions

PLACEMENT_TASKS = frozenset({
    "placement_initialization", "placement_preconditioner",
    "placement_dynamic_parameters_reference_20260923",
    "placement_preconditioner_source_reference_20260923",
})
ROUTING_TASKS = frozenset("global_routing_" + name + "_source" for name in (
    "net_ordering", "ripup_policy", "maze_cost", "iteration_controller"))
TASK_IDS = PLACEMENT_TASKS | ROUTING_TASKS | {
    tp.TASK_ID, "tp_gnn_bin_timing", "tp_gnn_post_placement_timing", "hbt_assignment_policy",
    "tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}
TEMPLATES = tp.TEMPLATES

# Only the current global-2way timing task uses this compact feedback contract.
# Historical tasks and the complete evaluator receipt retain their own fields.
TP_TIMING_GROUPS = {
    "fitness": ("combined_score", "final_global_wns_setup_ns", "final_global_tns_setup_ns",
                "normalized_wns_cost", "normalized_tns_cost"),
    "validity": ("runs_successfully", "partition_complete", "partition_constraints_passed",
                 "area_balance_error", "macro_count_error", "macro_area_error"),
    "partition_diagnostics": ("cut_net_count", "timing_weighted_cut", "timing_weighted_cut_ratio",
                              "partition_assignment_changed_count", "partition_assignment_changed_fraction",
                              "partition_assignment_identical"),
    "decoder_repair": (
        "tp_gnn_raw_timing_weighted_cut", "tp_gnn_balanced_timing_weighted_cut",
        "tp_gnn_feasible_timing_weighted_cut", "tp_gnn_area_balance_weighted_cut_delta",
        "tp_gnn_feasibility_weighted_cut_delta", "tp_gnn_area_balance_weighted_cut_relative_change",
        "tp_gnn_feasibility_weighted_cut_relative_change", "tp_gnn_repair_moved_supernodes",
        "tp_gnn_repair_moved_area_fraction"),
    "training_diagnostics": (
        "tp_gnn_final_loss", "tp_gnn_epochs_completed", "tp_gnn_embedding_total_variance",
        "tp_gnn_positive_pair_cosine_median", "tp_gnn_negative_pair_cosine_median",
        "tp_gnn_positive_pair_sample_count", "tp_gnn_negative_pair_sample_count"),
}

PLACEMENT_METRICS = frozenset({
    "combined_score", "runs_successfully", "hpwl_score", "overflow_score",
    "placement_complete", "placement_native_hpwl", "placement_overflow_max",
    "placement_overflow_mean", "placement_max_density", "target_density",
    "placement_overflow_bottom", "placement_overflow_top",
    "placement_max_density_bottom", "placement_max_density_top",
    "overflow_reward_tolerance",
    "placement_logged_rows", "placement_first_logged_hpwl", "placement_first_logged_overflow_max",
    "placement_logged_hpwl_relative_change", "placement_logged_overflow_change",
})
NESTEROV_METRICS = frozenset({
    "nesterov_control_updates", "nesterov_control_first_step", "nesterov_control_last_step",
    "nesterov_control_segments", "nesterov_identity_controls_only",
} | {"nesterov_" + name + "_" + stat
     for name in ("native_step_hint", "shared_multiplier", "bottom_multiplier", "top_multiplier")
     for stat in ("initial", "final", "min", "max")})
ROUTING_METRICS = frozenset({
    "combined_score", "runs_successfully", "overflow_score", "wirelength_score",
    "overflow_cost", "wirelength_cost", "overflow_weight", "wirelength_weight",
    "runtime_seconds", "grt_complete", "routing_complete",
} | {prefix + name for prefix in ("", "bottom_", "top_")
     for name in ("grt_total_overflow", "grt_wirelength_um", "grt_congestion_bins")})
ORDER_METRICS = frozenset({"net_order_diagnostics_present"} | {
    prefix + "net_order_" + name for prefix in ("", "bottom_", "top_")
    for name in ("calls", "changed_calls", "changed_ratio_max", "overflow_changed_ratio_max",
                 "top10_overlap_min", "score_span_max", "overflow_net_fraction_max", "reroute_phase_last")})
POLICY_METRICS = frozenset({"policy_diagnostics_present", "policy_diagnostic_rows"} | {
    prefix + "policy_" + name for prefix in ("", "bottom_", "top_")
    for name in ("rows", "calls", "changed", "changed_ratio_max")})
POLICY_DETAILS = {
    "global_routing_ripup_policy_source": ("score", "selected_ratio"),
    "global_routing_maze_cost_source": ("multiplier",),
    "global_routing_iteration_controller_source": (
        "cost_scale", "search_scale", "ripup_offset", "cost_height", "search_window", "ripup_threshold"),
}


def template_source(spec):
    if spec.task_id == tp.TASK_ID:
        return spec.root / "openevolve_templates"
    return Path(__file__).parent / "templates" / "optimization"


def optimization_result(result, *, task_id, require_score=True):
    if "matrix" in result.metrics:
        return matrix_feedback(result, task_id=task_id, require_score=require_score)
    if task_id == tp.TASK_ID:
        return tp.optimization_result(result, require_score=require_score)
    if task_id not in TASK_IDS:
        return result
    score = result.metrics.get("combined_score")
    if require_score and (isinstance(score, bool) or not isinstance(score, (int, float))
                          or not math.isfinite(score)):
        raise ValueError("Optimization feedback requires a finite combined_score")
    if task_id == "hbt_assignment_policy":
        # Only independently collected assignment and STA measurements.
        keys = {
            "combined_score", "runs_successfully", "verifier_only", "final_global_wns_setup_ns",
            "final_global_tns_setup_ns", "normalized_sta_cost", "normalized_wns_cost",
            "normalized_tns_cost", "assigned_count", "unassigned_count",
            "ideal_combined_hpwl_dbu", "legal_combined_hpwl_dbu", "combined_hpwl_increase_dbu",
            "total_displacement_dbu", "mean_displacement_dbu", "max_displacement_dbu",
            "unique_x_tracks", "unique_y_tracks", "shared_x_track_count", "shared_y_track_count",
            "dbu_per_micron", "assignment_runtime_seconds", "hbt_runtime_seconds", "sta_runtime_seconds",
        }
        interpretation = (
            "Maximize the reciprocal of 0.6 normalized WNS deficit plus 0.4 normalized TNS deficit. "
            "Timing is measured after legal HBT materialization with joint two-die placement RC, "
            "ideal clocks and no timing optimization. Legal combined HPWL includes both halves "
            "of crossing nets at legal HBT positions, not whole-chip or vertical wirelength. "
            "Geometry and runtime are diagnostics, not rewards. Track reuse counts are not "
            "routing overflow or design-rule violations. This is not signoff timing. "
            "Compare only evaluations with the same qualified initial policy and placed design."
        )
        if result.metrics.get("verifier_only") == 1:
            interpretation = (
                "Offline static verification only. combined_score and runs_successfully are "
                "synthetic protocol-test values, not measured HBT timing or physical success. "
                "No HBT assignment, materialization or STA was executed."
            )
    elif task_id in {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}:
        keys = {key for group in TP_TIMING_GROUPS.values() for key in group}
        keys -= {"normalized_wns_cost", "normalized_tns_cost"}
        keys |= {"wns_deficit_improvement", "tns_deficit_improvement", "opt_passes",
                 "timing_evaluation_boundary"}
        keys |= {"verifier_only", "binary_probability_uncertain_fraction",
                 "binary_probability_uncertain_area_fraction", "binary_raw_top_area_ratio",
                 "binary_final_top_area_ratio", "binary_closure_moved_area_fraction"}
        interpretation = (
            "Source-connected partition task: each candidate uses this baseline's freshly "
            "qualified initial-program reference for the same case and seed. The score is "
            "100 times 0.4 WNS deficit improvement plus 0.6 TNS deficit improvement after "
            "one fixed OPT pass. Cut, HPWL, loss, confidence and repair are diagnostics."
        )
        if result.metrics.get("verifier_only") == 1:
            interpretation += (
                " Offline static verification only: combined_score and runs_successfully are "
                "synthetic protocol values. No partition training, placement, HBT or STA was run."
            )
    elif task_id in {"tp_gnn_bin_timing", "tp_gnn_post_placement_timing"}:
        keys = tp.METRICS | {
            "final_global_wns_setup_ns", "final_global_tns_setup_ns", "normalized_sta_cost",
            "normalized_wns_cost", "normalized_tns_cost", "sta_wns_weight", "sta_tns_weight",
            "opt_passes", "sta_graph_complete", "spatial_bin_max_area_error"}
        keys = keys - {"timing_weighted_cut_score"}
        if task_id == "tp_gnn_post_placement_timing":
            keys = {key for group in TP_TIMING_GROUPS.values() for key in group}
        interpretation = (
            "Maximize combined_score from 0.6 normalized WNS deficit and 0.4 normalized TNS deficit. "
            "Both are unified two-die setup timing after placement and HBT, without timing optimization. "
            "Cut, loss and embedding metrics are diagnostics, not rewards. Positive slack is not a "
            "negative cost. This is placement-estimated pre-clock-tree timing, not signoff."
        )
        interpretation += (" The partition decoder is fixed global-2way, with no per-bin gates."
                           if task_id == "tp_gnn_post_placement_timing"
                           else " The partition decoder is fixed spatial-bin.")
    elif task_id in PLACEMENT_TASKS:
        keys = PLACEMENT_METRICS
        if task_id.startswith("placement_dynamic_"):
            keys |= NESTEROV_METRICS
        interpretation = (
            "Maximize combined_score after validation. Lower final weighted half-perimeter "
            "wirelength and worst-die density overflow are better; the score permits trade-offs. "
            "Maximum density and controller statistics are diagnostics, not rewards. "
            "Native step hints are not accepted step lengths. This is global placement, "
            "not legalized placement or routed timing."
        )
        if task_id in {"placement_initialization", "placement_preconditioner"}:
            interpretation += " Overflow reward saturates at 0.01, the existing stopping tolerance."
    else:
        keys = ROUTING_METRICS | (ORDER_METRICS if "net_ordering" in task_id else POLICY_METRICS)
        keys |= {prefix + "policy_" + name + "_" + stat
                 for prefix in ("bottom_", "top_")
                 for name in POLICY_DETAILS.get(task_id, ())
                 for stat in ("last", "min", "max")}
        if "ripup_policy" in task_id:
            keys |= {"bottom_policy_selected", "top_policy_selected"}
        if "iteration_controller" in task_id:
            keys |= {"bottom_policy_iteration_index_last", "top_policy_iteration_index_last"}
        interpretation = (
            "Maximize combined_score only after both dies finish and hook evidence is present. "
            "Lower total global-route wirelength and overflow are better; lower cost terms "
            "and higher ratio scores are better. Inspect each die separately before attributing "
            "a change. Hook activity is not an extra reward. This is not detailed routing, "
            "design-rule checking or routed timing. "
        )
        if result.metrics.get("routing_scoring_version") == 2:
            from .routing_scoring import ROUTING_V2_FEEDBACK_KEYS
            keys = (keys - {"overflow_score", "overflow_cost", "overflow_weight",
                            "wirelength_weight"}) | ROUTING_V2_FEEDBACK_KEYS
            interpretation = (
                "Routing scoring version 2: combined_score is 0 when candidate total GRT overflow "
                "exceeds overflow_gate_limit (max_ratio times reference total overflow plus "
                "abs_slack); otherwise it is grt_complete times reference/candidate total "
                "global-route wirelength. Overflow below the gate earns no extra credit. "
                "Inspect each die separately before attributing a change. Hook activity is not "
                "an extra reward. This is not detailed routing, design-rule checking or routed timing. "
            )
            reason = result.metrics.get("overflow_gate_reason")
            if isinstance(reason, str):
                interpretation += "Gate verdict: " + reason + ". "
        if "iteration_controller" in task_id:
            interpretation += ("policy_calls counts logged invocations; policy_changed counts "
                               "invocations with any nonidentity correction. Iteration indices may reset. ")
        elif "net_ordering" not in task_id:
            interpretation += ("Calls, changed and selected counts are observed cumulative "
                               "lower bounds from sparse logs, not exact final totals. Min/max/last "
                               "values describe logged samples, not every invocation. ")
    metrics = {}
    for key, value in result.metrics.items():
        if key not in keys:
            continue
        if key == "nesterov_identity_controls_only" and isinstance(value, bool):
            metrics[key] = int(value)
        elif not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value):
            metrics[key] = value
    feedback = {
        "metrics": metrics,
        "metric_definitions": metric_definitions(metrics, task_id),
        "feedback_fields": {
            "tell.proposal_id": "Identifier linking this feedback to its proposed program; not a metric.",
            "tell.program_sha256": "Content fingerprint of the evaluated program; identity, not fitness.",
            "tell.status": "accepted/cached are scored; rejected/failed are not valid population members. The TP status group below contains the rejection reason only.",
            "tell.score": "Selection value; null for rejected/failed candidates, not a measured zero.",
            "tell.runtime_seconds": "Evaluation-wrapper wall seconds, potentially including admission/build; not API or pure native time. Zero on pre-evaluation rejection is not a measured solver duration.",
            "tell.error_type/error_message": "Failure category and concise explanation; not a physical measurement.",
            "rejection_reason": "Concise implementation/evaluation failure; absence does not establish physical improvement.",
        },
        "rejection_reason": tp._reason(result.artifacts.get("error_message")),
        "interpretation": interpretation + " Missing metrics remain unknown, never zero. "
                          "A zero score alone does not identify a failure cause.",
    }
    if task_id == "tp_gnn_post_placement_timing":
        feedback.pop("metrics")
        feedback.update({group: {key: metrics[key] for key in names if key in metrics}
                         for group, names in TP_TIMING_GROUPS.items()})
        feedback["status"] = {"rejection_reason": feedback.pop("rejection_reason")}
        feedback["selection_target"] = "combined_score only; cut/loss/embedding metrics are diagnostic only."
        feedback["diagnostic_notes"] = (
            "Weighted-cut deltas are after minus before; relative changes divide by the before value "
            "and are omitted when it is zero. Repair movement compares balanced to feasible assignment; "
            "area fraction uses original instance areas, including macros, divided by total instance area. "
            "Pair cosines use final embeddings on up to 4096 evenly spaced actual pairs per class from "
            "the last completed training batch, not all graph pairs or tier labels. "
            "Loss magnitudes are not comparable across differently scaled objectives."
        )
    return EvaluationResult(metrics=metrics, artifacts={
        "optimization_feedback": json.dumps(feedback, ensure_ascii=False, allow_nan=False)
    })


def matrix_feedback(result, *, task_id, require_score=True):
    """Keep aggregate and cell feedback distinct; never average incompatible raw units."""
    from copy import deepcopy
    matrix = deepcopy(result.metrics["matrix"])
    score = result.metrics.get("combined_score")
    if require_score and (not matrix["complete"] or isinstance(score, bool)
                          or not isinstance(score, (int, float)) or not math.isfinite(score)):
        raise ValueError("Optimization matrix feedback requires a complete finite score")
    for row in matrix["cells"]:
        if "metrics" in row:
            projected = optimization_result(EvaluationResult(row["metrics"], row.get("artifacts", {})),
                                             task_id=task_id, require_score=row["status"] == "succeeded")
            row["metrics"] = projected.metrics
        row.pop("artifacts", None)
    metrics = {"runs_successfully": result.metrics["runs_successfully"], "matrix": matrix}
    if score is not None:
        metrics["combined_score"] = score
    baseline_score = 0 if task_id in {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"} else 1
    feedback = {"combined_score": score, "matrix": matrix,
                "interpretation": "Maximize the equal-case mean of mean paired-seed task scores. "
                f"Each cell uses its own initial-program reference (score {baseline_score}). Raw physical metrics "
                "belong to individual cells; they are not cross-case averages. Failed or missing "
                "cells invalidate the candidate; never choose a favorable seed.",
                "matrix_result": result.artifacts.get("matrix_result")}
    return EvaluationResult(metrics, {"optimization_feedback": json.dumps(feedback, allow_nan=False),
                                     "matrix_result": result.artifacts.get("matrix_result")})
