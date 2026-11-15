# EVOLVE-BLOCK-START
def score_ripup_edge(ctx):
    """Preserve native cleanup and recover clean detours only on low-pressure dies."""
    if ctx.baseline_decision >= 0.5:
        return 1.0

    # Feedback favored detour recovery in low-overflow cells, while broader
    # intervention regressed the two higher-overflow bp_fe seeds.
    if ctx.die_overflow > 0.012:
        return 0.0
    if ctx.is_clock >= 0.5 or ctx.has_ndr >= 0.5 or ctx.is_hbt >= 0.5:
        return 0.0
    if ctx.edge_overflow > 0.0 or ctx.max_utilization > 0.5:
        return 0.0
    if ctx.history_pressure > 0.45 or ctx.criticality > 0.20:
        return 0.0

    # Require both substantial excess path length and enough absolute route
    # length for a successful shortcut to matter to the wirelength objective.
    if ctx.detour_ratio >= 0.40 and ctx.route_length >= 0.08:
        return 1.0
    return 0.0
# EVOLVE-BLOCK-END