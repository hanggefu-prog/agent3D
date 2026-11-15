# EVOLVE-BLOCK-START
def score_ripup_edge(ctx):
    """Return at least 0.5 when the current routed edge should be rerouted."""
    return ctx.baseline_decision
# EVOLVE-BLOCK-END
