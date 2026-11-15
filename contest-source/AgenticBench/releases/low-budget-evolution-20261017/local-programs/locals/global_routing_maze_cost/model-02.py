# EVOLVE-BLOCK-START
def score_maze_step(ctx):
    """Apply the measured long-net rebalance only on the overflowing die."""
    # Bottom had zero overflow and only added wirelength; retain native costs there.
    if ctx.die_id < 0.5:
        return 1.0
    return 1.0 + 0.45 * ctx.wirelength
# EVOLVE-BLOCK-END