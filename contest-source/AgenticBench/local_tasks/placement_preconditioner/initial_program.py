# EVOLVE-BLOCK-START
class PreconditionOp:
    """Preconditioning engine is critical for convergence.
    Need to be carefully designed.
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
                placedb.num_filler_nodes, device=data_collections.pos[0].device, dtype=torch.long
            )
            if placedb.filler_start_map is not None:
                # Concurrent LoL can skip the empty non-fence density region.  The
                # filler map therefore follows filler_start_map directly:
                # region 0/1 are bottom/top, and an optional final interval is
                # the non-fence canvas region in legacy DREAMPlace cases.
                for i in range(max(0, len(self.placedb.filler_start_map) - 1)):
                    filler_beg, filler_end = self.placedb.filler_start_map[i : i + 2]
                    self.filler2fence_region_map[filler_beg:filler_end] = i

    def set_overflow(self, overflow):
        self.overflows.append(overflow)
        if self.best_overflow is None:
            self.best_overflow = overflow
        elif self.best_overflow.mean() > overflow.mean():
            self.best_overflow = overflow

    def __call__(self, grad, density_weight, update_mask=None, fix_nodes_mask=None):
        """Introduce alpha parameter to avoid divergence.
        It is tricky for this parameter to increase.
        """
        with torch.no_grad():
            # The preconditioning step in python is time-consuming, as in each gradient
            # pass, the total net weight should be re-calculated.
            sum_pin_weights_in_nodes = self.op_collections.pws_op(self.data_collections.net_weights)
            if density_weight.size(0) == 1:
                precond = (sum_pin_weights_in_nodes
                    + self.alpha * density_weight * self.data_collections.node_areas
                )
            else:
                node_areas = self.data_collections.node_areas.clone()
                movable_region = self.movablenode2fence_region_map_clamp.clamp(
                    max=density_weight.numel() - 1
                )
                node_areas[: self.placedb.num_movable_nodes] *= density_weight[movable_region]
                if self.placedb.num_filler_nodes:
                    filler_region = self.filler2fence_region_map.clamp(max=density_weight.numel() - 1)
                    filler_start = self.placedb.num_nodes - self.placedb.num_filler_nodes
                    node_areas[filler_start:] *= density_weight[filler_region]
                precond = sum_pin_weights_in_nodes + self.alpha * node_areas

            precond.clamp_(min=1.0)
            grad[0 : self.placedb.num_nodes].div_(precond)
            grad[self.placedb.num_nodes : self.placedb.num_nodes * 2].div_(precond)

            ### stop gradients for terminated electric field
            if update_mask is not None:
                grad = grad.view(2, -1)
                update_mask = ~update_mask
                movable_mask = update_mask[self.movablenode2fence_region_map_clamp]
                filler_mask = update_mask[self.filler2fence_region_map]
                grad[0, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                grad[1, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                grad[0, self.placedb.num_nodes - self.placedb.num_filler_nodes :].masked_fill_(filler_mask, 0)
                grad[1, self.placedb.num_nodes - self.placedb.num_filler_nodes :].masked_fill_(filler_mask, 0)
                grad = grad.view(-1)
            if fix_nodes_mask is not None:
                grad = grad.view(2, -1)
                grad[0, :self.placedb.num_movable_nodes].masked_fill_(fix_nodes_mask[:self.placedb.num_movable_nodes], 0)
                grad[1, :self.placedb.num_movable_nodes].masked_fill_(fix_nodes_mask[:self.placedb.num_movable_nodes], 0)
                grad = grad.view(-1)
            self.iteration += 1

            # only work in benchmarks without fence region, assume overflow has been updated
            if len(self.placedb.regions) > 0 and self.overflows and self.overflows[-1].max() < 0.3 and self.alpha < 1024:
                if (self.iteration % 20) == 0:
                    self.alpha *= 2
                    logging.info(
                        "preconditioning alpha = %g, best_overflow %g, overflow %g"
                        % (self.alpha, self.best_overflow, self.overflows[-1])
                    )

        return grad
# EVOLVE-BLOCK-END
