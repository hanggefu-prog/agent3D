# EVOLVE-BLOCK-START
class PreconditionOp:
    """Precondition with a concave pin-weight term.

    Linear incident-pin scaling can over-damp highly connected nodes.  Applying
    a bounded 3/4 power compresses only that scaling spread, while retaining the
    native density-weighted area term, tier ownership, and all update masks.
    The failed prior matrix supplied no measurements, so no gain is inferred
    from it; this is a distinct single-mechanism proposal.
    """
    def __init__(self, placedb, data_collections, op_collections, die_contexts=None):
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.iteration = 0
        self.alpha = 1.0
        self.best_overflow = None
        self.overflows = []
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
            if placedb.num_filler_nodes and placedb.filler_start_map is not None:
                for i in range(max(0, len(placedb.filler_start_map) - 1)):
                    filler_beg, filler_end = placedb.filler_start_map[i:i + 2]
                    self.filler2fence_region_map[filler_beg:filler_end] = i

    def set_overflow(self, overflow):
        self.overflows.append(overflow)
        if self.best_overflow is None:
            self.best_overflow = overflow
        elif self.best_overflow.mean() > overflow.mean():
            self.best_overflow = overflow

    def __call__(self, grad, density_weight, update_mask=None, fix_nodes_mask=None):
        with torch.no_grad():
            sum_pin_weights_in_nodes = self.op_collections.pws_op(
                self.data_collections.net_weights
            )
            pin_precond = sum_pin_weights_in_nodes.clamp_min(0).pow(0.75)

            if density_weight.numel() == 1:
                precond = (
                    pin_precond
                    + self.alpha
                    * density_weight
                    * self.data_collections.node_areas
                )
            else:
                node_areas = self.data_collections.node_areas.clone()
                movable_region = self.movablenode2fence_region_map_clamp.clamp(
                    max=density_weight.numel() - 1
                )
                node_areas[:self.placedb.num_movable_nodes] *= density_weight[
                    movable_region
                ]
                if self.placedb.num_filler_nodes:
                    filler_region = self.filler2fence_region_map.clamp(
                        max=density_weight.numel() - 1
                    )
                    filler_start = (
                        self.placedb.num_nodes - self.placedb.num_filler_nodes
                    )
                    node_areas[filler_start:] *= density_weight[filler_region]
                precond = pin_precond + self.alpha * node_areas

            precond.clamp_(min=1.0)
            num_nodes = self.placedb.num_nodes
            grad[:num_nodes].div_(precond)
            grad[num_nodes:num_nodes * 2].div_(precond)

            if update_mask is not None:
                grad = grad.view(2, -1)
                inactive_regions = ~update_mask
                movable_mask = inactive_regions[
                    self.movablenode2fence_region_map_clamp
                ]
                grad[0, :self.placedb.num_movable_nodes].masked_fill_(
                    movable_mask, 0
                )
                grad[1, :self.placedb.num_movable_nodes].masked_fill_(
                    movable_mask, 0
                )
                if self.placedb.num_filler_nodes:
                    filler_mask = inactive_regions[self.filler2fence_region_map]
                    filler_start = num_nodes - self.placedb.num_filler_nodes
                    grad[0, filler_start:].masked_fill_(filler_mask, 0)
                    grad[1, filler_start:].masked_fill_(filler_mask, 0)
                grad = grad.view(-1)

            if fix_nodes_mask is not None:
                grad = grad.view(2, -1)
                movable_fix_mask = fix_nodes_mask[
                    :self.placedb.num_movable_nodes
                ]
                grad[0, :self.placedb.num_movable_nodes].masked_fill_(
                    movable_fix_mask, 0
                )
                grad[1, :self.placedb.num_movable_nodes].masked_fill_(
                    movable_fix_mask, 0
                )
                grad = grad.view(-1)

            self.iteration += 1
            if (
                len(self.placedb.regions) > 0
                and self.overflows
                and self.overflows[-1].max() < 0.3
                and self.alpha < 1024
                and self.iteration % 20 == 0
            ):
                self.alpha *= 2
                logging.info(
                    "preconditioning alpha = %g, best_overflow %g, overflow %g"
                    % (self.alpha, self.best_overflow, self.overflows[-1])
                )

        return grad
# EVOLVE-BLOCK-END