# EVOLVE-BLOCK-START
def control_reroute_iteration(ctx):
    """Return congestion scale, search scale, and rip-up threshold offset."""
    return (
        ctx.baseline_congestion_scale,
        ctx.baseline_search_scale,
        ctx.baseline_ripup_offset,
    )
# EVOLVE-BLOCK-END
