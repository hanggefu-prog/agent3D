# EVOLVE-BLOCK-START
def control_reroute_iteration(ctx):
    """Adapt only congestion pressure; preserve native search and rip-up controls."""
    # Shift gently toward shorter routes as overflow eases, but defend stalled regressions.
    phase = min(1.0, max(0.0, ctx.reroute_phase))
    trend = ctx.congestion_trend
    stagnation = ctx.stagnation
    urgency = 4.0 * ctx.max_overflow + 12.0 * ctx.die_overflow
    urgency += 0.25 * ctx.ndr_pressure
    urgency = min(1.0, max(0.0, urgency))

    progress = min(1.0, max(0.0, 2.0 * (trend - 0.5)))
    regression = min(1.0, max(0.0, 2.0 * (0.5 - trend)))

    congestion_scale = 1.0
    if regression > 0.0 and stagnation > 0.12 and urgency > 0.12:
        congestion_scale += (0.04 + 0.10 * stagnation) * urgency * regression
    else:
        relief = (0.02 + 0.09 * phase) * (1.0 - 0.65 * urgency)
        relief += 0.025 * progress * (1.0 - urgency)
        relief += 0.015 * phase * ctx.wirelength_pressure * (1.0 - urgency)
        if stagnation > 0.30:
            relief *= max(0.25, 1.0 - 1.5 * stagnation)
        congestion_scale -= relief

    return (
        congestion_scale,
        ctx.baseline_search_scale,
        ctx.baseline_ripup_offset,
    )
# EVOLVE-BLOCK-END