"""Short model/human-readable definitions; never infer missing measurements."""

DEFINITIONS = {
    "combined_score": "Selection fitness; dimensionless, higher is better. Compare only valid runs under the same task contract.",
    "runs_successfully": "1 when the evaluator's required execution checks pass, 0 on failure; not signoff.",
    "placement_complete": "1 when the required global-placement outputs pass validation; not legalization.",
    "partition_complete": "1 when every required instance has a valid tier assignment.",
    "partition_constraints_passed": "1 when all required partition feasibility gates pass.",
    "area_balance_error": "Absolute difference between top-tier area share and 0.5; fraction, lower is more balanced.",
    "macro_count_error": "Absolute difference between top-tier macro-count share and 0.5; fraction.",
    "macro_area_error": "Absolute difference between top-tier macro-area share and 0.5; fraction.",
    "final_global_wns_setup_ns": "Worst setup slack of the joint two-die graph, ns; less negative is better. Placement RC, ideal clocks, no OPT.",
    "final_global_tns_setup_ns": "Sum of negative setup endpoint slacks of the joint graph, ns; closer to zero is better.",
    "normalized_wns_cost": "Candidate WNS deficit divided by reference deficit, with the task's zero-deficit rule; lower is better.",
    "normalized_tns_cost": "Candidate TNS deficit divided by reference deficit, with the task's zero-deficit rule; lower is better.",
    "normalized_sta_cost": "0.6 times normalized WNS cost plus 0.4 times normalized TNS cost; lower is better.",
    "cut_net_count": "Number of nets with instances on both tiers; count, diagnostic only.",
    "timing_weighted_cut": "Sum of frozen weights of cut nets; weighted count, diagnostic only for timing fitness.",
    "timing_weighted_cut_ratio": "Cut-net weight divided by all-net weight; fraction, diagnostic only.",
    "partition_assignment_changed_count": "Instances whose bottom/top assignment differs from the reference; count, names aligned, no tier-label swapping.",
    "partition_assignment_changed_fraction": "Changed instance count divided by total instances; fraction, not area-weighted.",
    "partition_assignment_identical": "1 if all named instances keep the reference tier, otherwise 0; identical assignment does not imply identical downstream placement.",
    "tp_gnn_final_loss": "Last measured training loss; scale depends on the candidate, not a comparable reward across different losses.",
    "tp_gnn_epochs_completed": "Completed training epochs; count, may exceed the requested minimum due to the minimum-step policy.",
    "tp_gnn_embedding_total_variance": "Sum of per-dimension population variances of final embeddings; diagnostic, not a reward.",
    "tp_gnn_repair_moved_supernodes": "Supernodes moved between balanced and feasible assignments; count, not geometric displacement.",
    "tp_gnn_repair_moved_area_fraction": "Original instance area moved by balanced-to-feasible repair divided by total area, including macros.",
    "hpwl_score": "Reference/candidate final weighted HPWL ratio with the task's zero handling; higher is better.",
    "overflow_score": "Task-specific reference/candidate overflow ratio; higher is better. Split placement tasks saturate at 0.01.",
    "wirelength_score": "Reference/candidate global-route wirelength ratio; higher is better.",
    "overflow_cost": "(Candidate total GRT overflow + 1000)/(reference total GRT overflow + 1000); lower is better.",
    "wirelength_cost": "Candidate/reference total GRT wirelength; lower is better.",
    "overflow_weight": "Fixed weight of overflow cost in routing fitness; not a measured physical quantity.",
    "wirelength_weight": "Fixed weight of wirelength cost in routing fitness; not a measured physical quantity.",
    "placement_native_hpwl": "Final complete GP row's native weighted HPWL, in native placement coordinates; not routed or legal-HBT wirelength.",
    "target_density": "Configured target placement density; dimensionless, not a measured maximum.",
    "overflow_reward_tolerance": "Overflow reward floor for split placement tasks; 0.01, not an extra legality gate.",
    "placement_logged_rows": "Complete valid GP metric rows observed in the current log; count, not optimizer steps.",
    "placement_first_logged_hpwl": "Weighted HPWL in the first complete GP log row; native coordinates, not a direct pre-optimizer initializer measurement.",
    "placement_first_logged_overflow_max": "Worst-die overflow in the first complete GP log row; dimensionless.",
    "placement_logged_hpwl_relative_change": "(Final minus first logged weighted HPWL)/first logged HPWL; lower means greater reduction, not fitness.",
    "placement_logged_overflow_change": "Final minus first logged worst-die overflow; dimensionless, negative means reduction.",
    "runtime_seconds": "Runtime reported by the metric producer, seconds; not API inference time or guaranteed pure solver time.",
    "grt_complete": "1 when both dies finish GRT and the task's hook-evidence gate passes; not DRT or signoff.",
    "routing_complete": "Routing completion flag supplied by the evaluator; does not certify detailed-route quality.",
    "policy_diagnostics_present": "1 when required policy-hook evidence is present; activity is not improvement.",
    "policy_diagnostic_rows": "Number of parsed policy diagnostic log rows across dies.",
    "nesterov_control_updates": "Observed Nesterov control invocations across logged segments; count.",
    "nesterov_control_first_step": "First logged native optimizer step index.",
    "nesterov_control_last_step": "Last logged native optimizer step index; indices may restart between segments.",
    "nesterov_control_segments": "Number of logged control segments, separated by step/call resets.",
    "nesterov_identity_controls_only": "1 if every observed control is identity, otherwise 0; not a quality measure.",
    "verifier_only": "1 for static protocol verification only; synthetic scores are not physical results.",
    "assigned_count": "Number of required HBT connections assigned to legal sites.",
    "unassigned_count": "Number of required HBT connections without a legal assigned site.",
    "ideal_combined_hpwl_dbu": "Sum of both tier halves of crossing-net HPWL at ideal HBT sites, DBU; excludes vertical length.",
    "legal_combined_hpwl_dbu": "Same crossing-net HPWL at legal HBT sites, DBU; not whole-chip wirelength.",
    "combined_hpwl_increase_dbu": "Legal minus ideal crossing-net combined HPWL, DBU; diagnostic only.",
    "total_displacement_dbu": "Sum of HBT Manhattan displacements from ideal sites, DBU.",
    "mean_displacement_dbu": "Mean HBT Manhattan displacement from ideal sites, DBU.",
    "max_displacement_dbu": "Maximum HBT Manhattan displacement from ideal sites, DBU.",
    "unique_x_tracks": "Distinct x tracks used by assigned HBTs; count, not congestion.",
    "unique_y_tracks": "Distinct y tracks used by assigned HBTs; count, not congestion.",
    "shared_x_track_count": "assigned_count minus unique_x_tracks: extra HBT uses beyond one per x track; not the number of shared tracks or DRCs.",
    "shared_y_track_count": "assigned_count minus unique_y_tracks: extra HBT uses beyond one per y track; not the number of shared tracks or DRCs.",
    "dbu_per_micron": "Integer database units per micrometer; divide DBU lengths by this value.",
    "assignment_runtime_seconds": "Measured HBT assignment time, seconds; excludes other stages.",
    "hbt_runtime_seconds": "Measured HBT-stage time, seconds; not necessarily disjoint from assignment time.",
    "sta_runtime_seconds": "Measured STA time, seconds; no API inference included.",
}


ROUTING_V2_DEFINITIONS = {
    "combined_score": "Routing scoring v2: 0 when the overflow gate fails; otherwise grt_complete*W_ref/W (reference/candidate total GRT wirelength). Higher is better.",
    "routing_scoring_version": "Routing score contract version; 2 means overflow hard gate followed by wirelength ratio.",
    "wirelength_cost": "With a reference: candidate/reference total GRT wirelength; bootstrap without a reference: 1+candidate wirelength_um/1e6. Lower is better; it is the only scored quantity when the gate passes.",
    "wirelength_score": "1/wirelength_cost: reference/candidate total GRT wirelength when a reference is supplied. Equals combined_score only when the overflow gate passes.",
    "overflow_gate_passed": "1 when candidate total GRT overflow <= max_ratio*reference overflow + abs_slack, else 0 (and combined_score is 0); 0 also when no reference was supplied.",
    "overflow_gate_evaluated": "1 when a reference run was available so the overflow gate was checked; 0 for unqualified bootstrap runs.",
    "overflow_gate_limit": "max_ratio*reference total GRT overflow + abs_slack; largest admissible candidate total overflow, routing-resource units.",
    "overflow_gate_max_ratio": "Fixed multiplier on reference total overflow in the v2 gate; contract constant, not a measurement.",
    "overflow_gate_abs_slack": "Fixed absolute overflow slack added in the v2 gate; contract constant, not a measurement.",
    "overflow_gate_reason": "Human-readable gate verdict with the compared numbers.",
    "reference_grt_total_overflow": "Reference total (bottom+top) GRT overflow used by the v2 gate.",
    "reference_grt_wirelength_um": "Reference total (bottom+top) GRT wirelength used by the v2 score, micrometers.",
}


def metric_definition(key, task_id, routing_scoring_version=1):
    if routing_scoring_version == 2 and task_id.startswith("global_routing_"):
        if key in ROUTING_V2_DEFINITIONS:
            return ROUTING_V2_DEFINITIONS[key]
        if key in {"overflow_cost", "overflow_score", "overflow_weight", "wirelength_weight"}:
            return "Not used by routing scoring v2; overflow only enters through overflow_gate_passed."
    if key == "overflow_score" and task_id.startswith("global_routing_"):
        return "1/max(overflow_cost,1e-12); with a reference, (reference overflow+1000)/(candidate overflow+1000). Higher is better, not a raw overflow ratio."
    if key == "overflow_cost" and task_id.startswith("global_routing_"):
        return "With a reference: (candidate total overflow+1000)/(reference total overflow+1000); bootstrap without a reference: 1+candidate overflow/1000. Lower is better."
    if key == "wirelength_cost" and task_id.startswith("global_routing_"):
        return "With a reference: candidate/reference total GRT wirelength; bootstrap without a reference: 1+candidate wirelength_um/1e6. Lower is better."
    if key == "wirelength_score" and task_id.startswith("global_routing_"):
        return "1/max(wirelength_cost,1e-12); reference/candidate GRT wirelength ratio when a reference is supplied. Higher is better."
    if key in DEFINITIONS:
        return DEFINITIONS[key]
    if key.startswith("placement_overflow_"):
        return "Final GP density overflow for " + key[len("placement_overflow_"):] + "; dimensionless, lower is better; not routing overflow."
    if key.startswith("placement_max_density"):
        return "Final GP maximum bin density (" + key[len("placement_max_density"):].strip("_") + "); dimensionless and may exceed 1; diagnostic, not overflow."
    for stage in ("raw", "balanced", "feasible"):
        if key == "tp_gnn_" + stage + "_timing_weighted_cut":
            return stage.capitalize() + " assignment's sum of frozen cut-net weights; diagnostic only."
    if key.startswith("tp_gnn_") and "weighted_cut_" in key:
        phase = "raw-to-balanced" if "area_balance" in key else "balanced-to-feasible"
        return phase + (" weighted-cut change divided by the before value; omitted if before is zero." if key.endswith("relative_change") else " weighted cut after minus before; negative means reduction.")
    if key.startswith("tp_gnn_") and "_pair_" in key:
        label = "positive" if "positive" in key else "negative"
        return ("Number of " + label + " pairs sampled, at most 4096." if key.endswith("sample_count") else "Median final-embedding cosine for sampled " + label + " pairs, [-1,1].") + " Last completed training batch, not all graph pairs or tier labels."
    if key.startswith("nesterov_"):
        stat = key.rsplit("_", 1)[-1]
        if "native_step_hint" in key:
            return stat + " native alpha hint before backtracking; optimizer-coordinate scale, not the accepted step."
        return stat + " observed " + key[9:].rsplit("_", 1)[0].replace("_", " ") + "; dimensionless applied control, not extra fitness."
    scope, name = "both dies", key
    for prefix in ("bottom_", "top_"):
        if key.startswith(prefix):
            scope, name = prefix[:-1] + " die", key[len(prefix):]
            break
    if name.startswith("grt_"):
        meanings = {"grt_total_overflow": "sum of routing demand exceeding capacity; routing-resource units, lower is better",
                    "grt_wirelength_um": "global-route wirelength, micrometers, lower is better",
                    "grt_congestion_bins": "reported congested-bin count, diagnostic only"}
        if name in meanings:
            return scope + ": " + meanings[name] + "; not detailed routing."
    if name.startswith("policy_"):
        field = name[7:]
        iteration = "iteration_controller" in task_id
        common = {"rows": "parsed diagnostic rows", "calls": "logged invocations" if iteration else "observed cumulative call-count lower bound",
                  "changed": "logged nonidentity invocations" if iteration else "observed cumulative changed-decision count lower bound",
                  "changed_ratio_max": "maximum logged changed ratio (0/1 per invocation for iteration control)",
                  "selected": "observed cumulative reroute-selected count lower bound",
                  "iteration_index_last": "last native iteration index; can reset, not a call count"}
        if field in common:
            return scope + ": " + common[field] + "; diagnostic, not reward."
        base, stat = field.rsplit("_", 1)
        details = {"score": "rip-up output score; >=0.5 selects rerouting", "selected_ratio": "reroute-selected fraction",
                   "multiplier": "applied maze edge-cost multiplier", "cost_scale": "applied congestion-cost multiplier",
                   "search_scale": "applied search-window multiplier", "ripup_offset": "applied integer threshold offset",
                   "cost_height": "resulting native congestion cost height", "search_window": "resulting native window in grid steps",
                   "ripup_threshold": "resulting native integer removal threshold"}
        if base in details and stat in {"min", "max", "last"}:
            return scope + ": " + stat + " logged " + details[base] + "; sampled log evidence, not a complete trace."
    return None


def metric_definitions(metrics, task_id):
    version = metrics.get("routing_scoring_version", 1) if hasattr(metrics, "get") else 1
    result = {}
    for key in metrics:
        meaning = metric_definition(key, task_id, version)
        if meaning is not None:
            result[key] = meaning
    return result
