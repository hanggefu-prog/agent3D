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
    """Add annealed timing-weighted edge cohesion to the native sampled loss."""
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
    graph_loss = (
        graph_weights * F.softplus(-graph_scores)
    ).sum() / torch.clamp(graph_weights.sum(), min=1e-12)
    return pair_loss + graph_strength * pair_scale * graph_loss
# EVOLVE-BLOCK-END