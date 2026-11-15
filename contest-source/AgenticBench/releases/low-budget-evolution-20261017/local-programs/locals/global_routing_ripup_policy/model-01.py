# EVOLVE-BLOCK-START
def score_ripup_edge(ctx):
    """Return at least 0.5 when the current routed edge should be rerouted."""
    # Preserve native removals; additionally retry only clean, substantial detours.
    if ctx.baseline_decision >= 0.5:
        return 1.0
    if ctx.is_clock >= 0.5 or ctx.has_ndr >= 0.5 or ctx.is_hbt >= 0.5:
        return 0.0
    if ctx.criticality >= 0.5:
        return 0.0
    if ctx.detour_ratio >= 0.30 and ctx.route_length >= 0.03:
        if ctx.edge_overflow <= 0.02 and ctx.max_utilization <= 0.55:
            if ctx.history_pressure <= 0.38:
                return 1.0
    return 0.0
# EVOLVE-BLOCK-END