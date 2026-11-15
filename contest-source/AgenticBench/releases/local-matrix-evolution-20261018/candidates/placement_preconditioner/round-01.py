# EVOLVE-BLOCK-START
class PreconditionOp:
    """Density-continuation preconditioner for concurrent two-tier placement.

    Qualification still has nontrivial final overflow, while the maintained
    two-region path never calls ``set_overflow`` and therefore leaves the
    native alpha continuation inactive.  This version uses growth of each
    live density weight as the available optimization-progress signal: the
    density-area term receives a bounded square-root continuation multiplier.
    Early calls exactly match native scaling; later damping grows gradually
    and independently by tier, avoiding a call-count schedule and preserving
    pin weights, geometry, ownership, fillers, and all supplied masks.
    """
    def __init__(self, placedb, data_collections, op_collections, die_contexts=None):
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.iteration = 0
        self.alpha = 1.0
        self.best_overflow = None
        self.overflows = []
        self.initial_density_weight = None

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
            if placedb.filler_start_map is not None and placedb.num_filler_nodes:
                for i in range(max(0, len(placedb.filler_start_map) - 1)):
                    filler_beg, filler_end = placedb.filler_start_map[i : i + 2]
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

            density_weight_flat = density_weight.reshape(-1)
            density_epsilon = torch.finfo(density_weight_flat.dtype).eps
            if self.initial_density_weight is None:
                self.initial_density_weight = density_weight_flat.detach().clone()
                self.initial_density_weight.clamp_(min=density_epsilon)

            weight_ratio = (
                density_weight_flat.detach().clamp(min=density_epsilon)
                / self.initial_density_weight
            )
            weight_ratio = torch.nan_to_num(
                weight_ratio, nan=1.0, posinf=4096.0, neginf=1.0
            ).clamp_(min=1.0, max=4096.0)
            adaptive_alpha = weight_ratio.sqrt_().clamp_(max=64.0)
            scaled_density_weight = density_weight_flat * adaptive_alpha

            if density_weight_flat.numel() == 1:
                precond = (
                    sum_pin_weights_in_nodes
                    + scaled_density_weight[0] * self.data_collections.node_areas
                )
            else:
                node_areas = self.data_collections.node_areas.clone()
                movable_region = self.movablenode2fence_region_map_clamp.clamp(
                    max=density_weight_flat.numel() - 1
                )
                node_areas[: self.placedb.num_movable_nodes] *= scaled_density_weight[
                    movable_region
                ]

                if self.placedb.num_filler_nodes:
                    filler_region = self.filler2fence_region_map.clamp(
                        max=density_weight_flat.numel() - 1
                    )
                    filler_start = (
                        self.placedb.num_nodes - self.placedb.num_filler_nodes
                    )
                    node_areas[filler_start:] *= scaled_density_weight[filler_region]

                precond = sum_pin_weights_in_nodes + node_areas

            precond.clamp_(min=1.0)
            grad[: self.placedb.num_nodes].div_(precond)
            grad[self.placedb.num_nodes : self.placedb.num_nodes * 2].div_(precond)

            if update_mask is not None:
                grad = grad.view(2, -1)
                stopped_regions = ~update_mask
                movable_mask = stopped_regions[
                    self.movablenode2fence_region_map_clamp
                ]
                grad[0, : self.placedb.num_movable_nodes].masked_fill_(
                    movable_mask, 0
                )
                grad[1, : self.placedb.num_movable_nodes].masked_fill_(
                    movable_mask, 0
                )

                if self.placedb.num_filler_nodes:
                    filler_mask = stopped_regions[self.filler2fence_region_map]
                    filler_start = (
                        self.placedb.num_nodes - self.placedb.num_filler_nodes
                    )
                    grad[0, filler_start:].masked_fill_(filler_mask, 0)
                    grad[1, filler_start:].masked_fill_(filler_mask, 0)
                grad = grad.view(-1)

            if fix_nodes_mask is not None:
                grad = grad.view(2, -1)
                movable_fix_mask = fix_nodes_mask[: self.placedb.num_movable_nodes]
                grad[0, : self.placedb.num_movable_nodes].masked_fill_(
                    movable_fix_mask, 0
                )
                grad[1, : self.placedb.num_movable_nodes].masked_fill_(
                    movable_fix_mask, 0
                )
                grad = grad.view(-1)

            self.iteration += 1

        return grad
# EVOLVE-BLOCK-END