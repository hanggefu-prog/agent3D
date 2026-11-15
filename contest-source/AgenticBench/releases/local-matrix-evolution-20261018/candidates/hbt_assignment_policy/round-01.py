# EVOLVE-BLOCK-START
def net_priority(context):
    """Prioritize high-pin nets, with a bounded boost for long ideal geometry."""
    span = max(context["die_span_dbu"], 1)
    normalized_length = context["ideal_combined_hpwl_dbu"] / span
    return context["pin_count"] * (1.0 + normalized_length)


def site_cost(net_context, site_context):
    """Minimize combined two-tier wire length to isolate the ordering idea."""
    return site_context["combined_hpwl_dbu"]
# EVOLVE-BLOCK-END