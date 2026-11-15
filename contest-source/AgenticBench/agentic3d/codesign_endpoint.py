"""Explicit endpoint/budget contracts for NEW timing/thermal experiment batches.

Legacy five-STA runs keep their original prompts and resume fingerprints.
"""
from __future__ import annotations

import math

from . import sta_protocol as protocol

PROFILES = {"timing": "tp_gnn_placement_two_pass", "thermal": "tp_gnn_placement_cts_thermal"}
ONE_PASS_PROFILES = {"timing": "tp_gnn_placement_one_pass",
                     "thermal": "tp_gnn_placement_cts_thermal_one_pass"}
TARGETS = {"timing": "tp_gnn_placement_sta", "thermal": "tp_gnn_placement_cts_thermal"}
DEFAULT_ROUNDS = 10


def validate_rounds(rounds):
    """A final-attempt budget is an explicit positive integer, never a coercion."""
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer final evaluation budget")
    return rounds


def _passes(objective, profile=None):
    if objective not in PROFILES:
        raise ValueError("invalid objective")
    if profile is None or profile == PROFILES[objective]:
        return 2
    if profile == ONE_PASS_PROFILES[objective]:
        return 1
    raise ValueError("objective/profile contract mismatch")


def _pass_prompt(text, passes):
    if passes == 2:
        return text
    return (text.replace("OPT pass 1 -> OPT pass 2 -> STA", "one native OPT pass -> STA")
            .replace("exactly two OPT passes", "exactly one native OPT pass")
            .replace("exactly TWO OPT passes", "exactly ONE native OPT pass")
            .replace("two-pass\ntiming endpoint", "one-pass\ntiming endpoint")
            .replace("two-opt-cts-cell-power-v1", "one-opt-cts-cell-power-v1")
            .replace("Compare pass1/pass2,", "Compare initial/final single-pass STA,"))


def common(objective, rounds, *, profile=None):
    validate_rounds(rounds)
    passes = _passes(objective, profile)
    text = protocol.COMMON.replace("Exactly FIVE final evaluations", f"Exactly {rounds} final evaluations")
    if objective == "timing":
        return _pass_prompt(text, passes)
    start, rest = text.split("Use ONLY this experiment's source snapshot", 1)
    del start
    text = f"""# Thermal co-design: immutable experiment contract
Optimize the mean temperature in kelvin over ALL physical HotSpot tiles after
frozen input/floorplan -> TP-GNN -> concurrent two-die placement -> the maintained
fixed PDN/HBT -> exactly TWO OPT passes -> CTS -> cell power -> HotSpot backend. Score = reference mean K
/ candidate mean K (higher is better). Peak temperature, timing, power and per-die
temperatures are diagnostics, not alternative final objectives. Trade-offs between
dies are allowed. Require complete successful physical stages, legal frontend,
power and temperature qualification. Power uses vectorless input activity 0.1 and
placement-estimated RC, including CTS-added cells. Do not run GRT/DRT/RCX/report;
do not claim routed, DRC-clean or signoff results on the basis of this score.
Reference uses TP-GNN global-2way and Concurrent DREAMPlace. The fixed stack,
10x10 thermal grid, 318.15 K ambient, clocks, seeds, inputs, footprint, backend and
judge are not searchable. The entire pre-CTS flow is identical to the two-pass
timing endpoint; append CTS and power/thermal without changing the OPT contract.
The objective is temperature, not STA. Historical full-route thermal references
and scores are incompatible with this two-opt-cts-cell-power-v1 boundary.
Exactly {rounds} final evaluations including failed final attempts. Each outer
round may use at most THREE cheap inner evaluations. Final-only local tasks use
one candidate and the SAME outer evaluation, not an extra trial. Accept only
strict improvement over the best valid outer score. Proxy improvements are
hypotheses: timing-weighted cut and HPWL/overflow may correlate poorly with heat.
Syntax repair/reflection do not grant free physical evaluations. The driver owns
every physical evaluation. Skills/tasks cannot change these boundaries.

Use ONLY this experiment's source snapshot""" + rest.replace("immutable STA objective", "immutable thermal objective")
    return _pass_prompt(text, passes)


def role_prompt(role, objective, *, profile=None):
    passes = _passes(objective, profile)
    text = getattr(protocol, role)
    if objective == "thermal":
        text = text.replace("Compare pass1/pass2, WNS/TNS trade-offs,",
                            "Compare per-die mean/peak temperature, cell power, density, CTS outcomes,")
        text = text.replace("proxy-to-STA", "proxy-to-temperature").replace("source/STA", "source/backend")
        text = text.replace("extra STA", "extra final evaluation").replace("STA gain", "thermal gain")
    return _pass_prompt(text, passes)


def final_score(result, objective, *, profile=None):
    passes = _passes(objective, profile)
    metrics = result.get("metrics", {})
    if metrics.get("runs_successfully") != 1:
        return None
    keys = ["combined_score"]
    if objective == "timing":
        if any(metrics.get(k) != v for k, v in {
            "opt_passes": passes, "sta_graph_complete": 1, "sta_unconstrained_endpoints": 0}.items()):
            return None
        keys += ["final_global_wns_setup_ns", "final_global_tns_setup_ns"]
    elif objective == "thermal":
        keys += ["global_average_temperature_k", "normalized_average_temperature_cost"]
        if any(metrics.get(k) != v for k, v in {
                "opt_passes": passes, "cts_completed": 1,
                "thermal_evaluation_boundary": ("one-opt-cts-cell-power-v1" if passes == 1
                                                 else "two-opt-cts-cell-power-v1"),
                "power_model": "die-local-openroad-placement-estimated"}.items()):
            return None
        if metrics.get("thermal_grid_size") != 10 or metrics.get("thermal_ambient_k") != 318.15:
            return None
    else:
        raise ValueError("unknown objective")
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            return None
    if objective == "thermal" and (metrics["global_average_temperature_k"] <= 0 or
            metrics["normalized_average_temperature_cost"] <= 0 or not math.isclose(
                metrics["combined_score"] * metrics["normalized_average_temperature_cost"], 1., rel_tol=1e-8)):
        return None
    return metrics["combined_score"] if metrics["combined_score"] > 0 else None


def validate_reference(case, bundle, result, objective, *, profile=None):
    suffix = "_lol_ng45_reference_20260923" if _passes(objective, profile) == 1 else "_lol_ng45_tp_gnn_sta"
    if bundle.get("benchmark_id") != case + suffix:
        raise ValueError("frozen input belongs to another case/profile")
    score = final_score(result, objective, profile=profile)
    if score is None or not math.isclose(score, 1., rel_tol=1e-9):
        raise ValueError(f"unqualified {objective} bootstrap reference")
