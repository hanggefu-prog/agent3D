# EVOLVE-BLOCK-START
def score_maze_step(ctx):
    """Scale costs to modestly favor wirelength while retaining congestion response."""
    # Compress expensive/congested edges; keep a small uplift on ordinary edges.
    multiplier = 1.08
    multiplier -= 0.10 * ctx.baseline_cost
    multiplier -= 0.04 * ctx.edge_overflow
    multiplier -= 0.02 * ctx.history_pressure
    return multiplier
# EVOLVE-BLOCK-END