# EVOLVE-BLOCK-START
def net_priority(context):
    """Assign nets with more pins first."""
    return context["pin_count"]


def site_cost(net_context, site_context):
    """Minimize the combined wire length at a legal terminal position."""
    return site_context["combined_hpwl_dbu"]
# EVOLVE-BLOCK-END
