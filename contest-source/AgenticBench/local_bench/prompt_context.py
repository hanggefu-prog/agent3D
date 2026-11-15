"""Small display-only views; population metrics and durable receipts stay intact."""
import math

from .feedback_semantics import metric_definition


def prompt_metrics(metrics, task_id):
    """Select measured objectives and a few actionable diagnostics, never fill gaps."""
    keys = ["combined_score", "runs_successfully", "verifier_only"]
    if task_id.startswith("tp_gnn_"):
        keys += ["final_global_wns_setup_ns", "final_global_tns_setup_ns",
                 "normalized_wns_cost", "normalized_tns_cost",
                 "partition_constraints_passed", "area_balance_error",
                 "macro_count_error", "macro_area_error", "timing_weighted_cut",
                 "partition_assignment_changed_fraction", "tp_gnn_repair_moved_area_fraction",
                 "tp_gnn_feasibility_weighted_cut_delta",
                 "tp_gnn_final_loss", "tp_gnn_embedding_total_variance",
                 "tp_gnn_positive_pair_cosine_median", "tp_gnn_negative_pair_cosine_median"]
    elif task_id.startswith("placement_"):
        keys += ["placement_complete", "placement_native_hpwl", "hpwl_score", "overflow_score",
                 "placement_overflow_bottom", "placement_overflow_top", "placement_overflow_max",
                 "placement_max_density", "placement_logged_hpwl_relative_change",
                 "placement_logged_overflow_change", "nesterov_identity_controls_only"]
    elif task_id.startswith("global_routing_"):
        keys += ["grt_complete", "grt_total_overflow", "grt_wirelength_um",
                 "bottom_grt_total_overflow", "top_grt_total_overflow",
                 "bottom_grt_wirelength_um", "top_grt_wirelength_um",
                 "policy_diagnostics_present", "net_order_diagnostics_present"]
        stem = "net_order" if "net_ordering" in task_id else "policy"
        keys += [die + "_" + stem + "_changed_ratio_max" for die in ("bottom", "top")]
        if metrics.get("routing_scoring_version") == 2:
            keys += ["routing_scoring_version", "overflow_gate_passed", "overflow_gate_limit",
                     "wirelength_score"]
    elif task_id == "hbt_assignment_policy":
        keys += ["final_global_wns_setup_ns", "final_global_tns_setup_ns",
                 "assigned_count", "unassigned_count", "legal_combined_hpwl_dbu",
                 "combined_hpwl_increase_dbu", "mean_displacement_dbu", "dbu_per_micron"]
    else:
        keys += [key for key in metrics if key not in keys][:8]
    flags = {"nesterov_identity_controls_only", "policy_diagnostics_present",
             "net_order_diagnostics_present"}
    return {key: metrics[key] for key in keys if key in metrics and (
            (key in flags and isinstance(metrics[key], bool)) or
            (not isinstance(metrics[key], bool) and isinstance(metrics[key], (int, float))
             and math.isfinite(metrics[key])))}


def feedback_text(metrics, task_id):
    if "matrix" in metrics:
        import json
        matrix = metrics["matrix"]
        return json.dumps({"combined_score": metrics.get("combined_score"),
            "aggregation": "equal-case mean of paired-seed task scores; reference=1",
            "complete": matrix["complete"], "cases": matrix["cases"],
            "cells": [{**{k: r[k] for k in ("case", "seed", "status", "score", "result_dir") if k in r},
                       "metrics": prompt_metrics(r.get("metrics", {}), task_id),
                       **({"native_failure_diagnostics": r["native_failure_diagnostics"]}
                          if "native_failure_diagnostics" in r else {}),
                       **({"error": r["error_message"]} if "error_message" in r else {})}
                      for r in matrix["cells"]]}, ensure_ascii=False, allow_nan=False)
    selected = prompt_metrics(metrics, task_id)
    if not selected:
        return "No measured metrics are available."
    # Exact numeric values appear once, with their meaning on the same line.
    version = metrics.get("routing_scoring_version", 1)
    lines = [
        '- "{}": {} — {}'.format(key, value, metric_definition(key, task_id, version)
                                 or "Task-provided measurement; follow the task's definition.")
        for key, value in selected.items()]
    reason = metrics.get("overflow_gate_reason")
    if version == 2 and isinstance(reason, str):
        lines.append('- "overflow_gate_reason": ' + reason)
    return "\n".join(lines)


def distinct_reference(parent, programs):
    """One best-first example with different source, even across duplicate IDs."""
    def source(code):
        return "\n".join(line.rstrip() for line in code.strip().splitlines())
    parent_source = source(parent.code)
    return next((program for program in programs if source(program.code) != parent_source), None)
