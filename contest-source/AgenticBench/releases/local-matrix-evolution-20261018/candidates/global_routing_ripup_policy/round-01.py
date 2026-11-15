# EVOLVE-BLOCK-START
def score_ripup_edge(ctx):
    """Select clean, substantial detours in addition to native rip-up choices."""
    # Preserve native congestion handling; only add conservative wirelength recovery.
    if ctx.baseline_decision >= 0.5:
        return 1.0

    if ctx.edge_overflow > 0.0:
        return 0.0
    if ctx.max_utilization >= 0.5 or ctx.history_pressure >= 0.45:
        return 0.0
    if ctx.is_clock >= 0.5 or ctx.has_ndr >= 0.5 or ctx.is_hbt >= 0.5:
        return 0.0
    if ctx.criticality >= 0.65:
        return 0.0

    if ctx.detour_ratio >= 0.30 and ctx.route_length >= 0.025:
        return 1.0
    return 0.0
# EVOLVE-BLOCK-END