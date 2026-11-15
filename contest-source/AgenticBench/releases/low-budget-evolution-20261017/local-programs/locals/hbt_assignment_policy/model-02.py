# EVOLVE-BLOCK-START
def net_priority(context):
    """Prioritize pin count, then broader nets that are more exposed to added delay."""
    die_span = max(context["die_span_dbu"], 1.0)
    ideal_hpwl = max(context["ideal_combined_hpwl_dbu"], 0.0)
    breadth = ideal_hpwl / (die_span + ideal_hpwl)
    return context["pin_count"] + 0.49 * breadth


def site_cost(net_context, site_context):
    """Keep combined legal wirelength as the candidate-site objective."""
    return site_context["combined_hpwl_dbu"]
# EVOLVE-BLOCK-END