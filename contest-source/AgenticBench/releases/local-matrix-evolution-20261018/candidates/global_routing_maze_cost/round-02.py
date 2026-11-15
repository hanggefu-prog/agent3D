# EVOLVE-BLOCK-START
def score_maze_step(ctx):
    # Favor compact long-net routes, but become congestion-conservative under pressure.
    die_pressure = ctx.die_overflow
    length_bias = 0.025 * ctx.wirelength * (1.0 - 0.75 * die_pressure)
    congestion_guard = die_pressure * (
        0.020 * ctx.edge_overflow + 0.006 * ctx.history_pressure
    )
    return 1.0 + length_bias + congestion_guard
# EVOLVE-BLOCK-END