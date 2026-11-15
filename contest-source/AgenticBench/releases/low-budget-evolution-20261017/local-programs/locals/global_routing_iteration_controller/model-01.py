# EVOLVE-BLOCK-START
def control_reroute_iteration(ctx):
    """Bias progressing reroutes toward shorter paths with a bounded window."""
    phase = min(1.0, max(0.0, ctx.reroute_phase))
    severity = min(1.0, max(ctx.die_overflow, ctx.max_overflow))
    progress = min(1.0, max(0.0, 2.0 * (ctx.congestion_trend - 0.5)))
    regression = min(1.0, max(0.0, 2.0 * (0.5 - ctx.congestion_trend)))

    if regression >= 0.2 or ctx.stagnation >= 0.28 or severity >= 0.35:
        search_scale = ctx.baseline_search_scale
    else:
        confidence = 0.35 + 0.65 * progress
        short_bias = (0.08 + 0.12 * phase) * confidence
        short_bias *= 1.0 - 0.6 * severity
        search_scale = ctx.baseline_search_scale - short_bias

    return (
        ctx.baseline_congestion_scale,
        search_scale,
        ctx.baseline_ripup_offset,
    )
# EVOLVE-BLOCK-END