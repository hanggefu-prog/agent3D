# EVOLVE-BLOCK-START
def score_maze_step(ctx):
    """Scale edge cost for long nets to favor shorter maze paths."""
    # Long nets receive a modest edge-versus-turn rebalance toward wirelength.
    multiplier = 1.0 + 0.45 * ctx.wirelength
    return multiplier
# EVOLVE-BLOCK-END