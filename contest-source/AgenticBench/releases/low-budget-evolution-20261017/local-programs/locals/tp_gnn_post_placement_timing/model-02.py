# EVOLVE-BLOCK-START
def compute_tp_gnn_loss(
    positive_scores,
    negative_scores,
    embeddings,
    positive_sources,
    positive_targets,
    negative_sources,
    negative_targets,
    source_batch_size,
    epoch,
    max_epochs,
    graph_scores,
    graph_sources,
    graph_targets,
    graph_net_ids,
    graph_weights,
    net_weights,
    node_features,
    node_areas,
):
    """Retain useful graph cohesion while focusing it on higher-weight timing nets."""
    pair_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    if graph_scores.numel() == 0:
        return pair_loss
    pair_scale = float(len(positive_scores) + len(negative_scores)) / float(
        max(1, source_batch_size)
    )
    progress = float(epoch) / float(max(1, max_epochs - 1))
    graph_strength = 0.25 + 0.5 * (1.0 - progress)
    net_mean = net_weights.mean()
    net_spread = torch.sqrt(torch.square(net_weights - net_mean).mean() + 1e-12)
    edge_priority = 1.0 + 0.75 * torch.tanh(
        (net_weights[graph_net_ids] - net_mean) / net_spread
    )
    focused_weights = graph_weights * edge_priority
    graph_loss = (
        focused_weights * F.softplus(-graph_scores)
    ).sum() / torch.clamp(focused_weights.sum(), min=1e-12)
    return pair_loss + graph_strength * pair_scale * graph_loss
# EVOLVE-BLOCK-END