# EVOLVE-BLOCK-START
class PreconditionOp:
    """Use an L2 combination of pin and density conditioning terms.

    Round 01's regional overflow balancing matched the native score exactly, so
    it is removed here.  The sole conditioning change replaces the additive
    denominator with hypot(pin_term, density_term).  This remains at least as
    large as either term, but avoids double-counting their curvature and permits
    bounded larger steps where both terms are comparable.  The native alpha
    schedule, tier ownership, filler mapping, and all update/fixed masks remain
    unchanged.
    """

    def __init__(self, placedb, data_collections, op_collections, die_contexts=None):
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.iteration = 0
        self.alpha = 1.0
        self.best_overflow = None
        self.latest_overflow = None

        if len(placedb.regions) > 0:
            self.movablenode2fence_region_map_clamp = (
                data_collections.node2fence_region_map[: placedb.num_movable_nodes]
                .clamp(max=len(placedb.regions))
                .long()
            )
            self.filler2fence_region_map = torch.zeros(
                placedb.num_filler_nodes,
                device=data_collections.pos[0].device,
                dtype=torch.long,
            )
            if placedb.filler_start_map is not None:
                for i in range(max(0, len(placedb.filler_start_map) - 1)):
                    filler_beg, filler_end = placedb.filler_start_map[i : i + 2]
                    self.filler2fence_region_map[filler_beg:filler_end] = i

    def set_overflow(self, overflow):
        observed = overflow.detach().clone()
        self.latest_overflow = observed
        if self.best_overflow is None:
            self.best_overflow = observed.clone()
        elif torch.isfinite(observed).all() and (
            not torch.isfinite(self.best_overflow).all()
            or observed.mean() < self.best_overflow.mean()
        ):
            self.best_overflow = observed.clone()

    def __call__(self, grad, density_weight, update_mask=None, fix_nodes_mask=None):
        with torch.no_grad():
            pin_term = self.op_collections.pws_op(
                self.data_collections.net_weights
            )

            if density_weight.size(0) == 1:
                density_term = (
                    self.alpha
                    * density_weight
                    * self.data_collections.node_areas
                )
            else:
                density_term = self.data_collections.node_areas.clone()
                movable_region = self.movablenode2fence_region_map_clamp.clamp(
                    max=density_weight.numel() - 1
                )
                density_term[: self.placedb.num_movable_nodes] *= (
                    density_weight[movable_region]
                )

                if self.placedb.num_filler_nodes > 0:
                    filler_region = self.filler2fence_region_map.clamp(
                        max=density_weight.numel() - 1
                    )
                    filler_start = (
                        self.placedb.num_nodes - self.placedb.num_filler_nodes
                    )
                    density_term[filler_start:] *= density_weight[filler_region]

                density_term.mul_(self.alpha)

            precond = torch.hypot(pin_term, density_term)
            precond.clamp_(min=1.0)
            grad[: self.placedb.num_nodes].div_(precond)
            grad[self.placedb.num_nodes : self.placedb.num_nodes * 2].div_(precond)

            if update_mask is not None:
                grad = grad.view(2, -1)
                disabled_regions = ~update_mask
                movable_mask = disabled_regions[
                    self.movablenode2fence_region_map_clamp
                ]
                grad[0, : self.placedb.num_movable_nodes].masked_fill_(
                    movable_mask, 0
                )
                grad[1, : self.placedb.num_movable_nodes].masked_fill_(
                    movable_mask, 0
                )

                if self.placedb.num_filler_nodes > 0:
                    filler_mask = disabled_regions[self.filler2fence_region_map]
                    filler_start = (
                        self.placedb.num_nodes - self.placedb.num_filler_nodes
                    )
                    grad[0, filler_start:].masked_fill_(filler_mask, 0)
                    grad[1, filler_start:].masked_fill_(filler_mask, 0)
                grad = grad.view(-1)

            if fix_nodes_mask is not None:
                grad = grad.view(2, -1)
                fixed_movable = fix_nodes_mask[: self.placedb.num_movable_nodes]
                grad[0, : self.placedb.num_movable_nodes].masked_fill_(
                    fixed_movable, 0
                )
                grad[1, : self.placedb.num_movable_nodes].masked_fill_(
                    fixed_movable, 0
                )
                grad = grad.view(-1)

            self.iteration += 1
            if (
                len(self.placedb.regions) > 0
                and self.latest_overflow is not None
                and torch.isfinite(self.latest_overflow).all()
                and self.latest_overflow.max() < 0.3
                and self.alpha < 1024
                and (self.iteration % 20) == 0
            ):
                self.alpha *= 2
                logging.info(
                    "preconditioning alpha = %g, best_overflow %g, overflow %g"
                    % (
                        self.alpha,
                        self.best_overflow.mean().item(),
                        self.latest_overflow.mean().item(),
                    )
                )

        return grad
# EVOLVE-BLOCK-END