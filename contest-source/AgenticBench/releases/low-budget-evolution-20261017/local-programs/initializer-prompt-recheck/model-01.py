# EVOLVE-BLOCK-START
def initialize_position(params, placedb, die_contexts=None):
    """Gently equalize movable-cell density per tier while preserving coordinate ranks."""
    init_pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)

    # Preserve all supplied physical coordinates, including fixed nodes.
    init_pos[0:placedb.num_physical_nodes] = placedb.node_x
    init_pos[
        placedb.num_nodes:placedb.num_nodes + placedb.num_physical_nodes
    ] = placedb.node_y

    if params.global_place_flag and params.random_center_init_flag:
        logging.info("move cells to the center of layout with random noise")
        init_pos[0:placedb.num_movable_nodes] = np.random.normal(
            loc=(placedb.xl * 1.0 + placedb.xh * 1.0) / 2,
            scale=(placedb.xh - placedb.xl) * 0.001,
            size=placedb.num_movable_nodes,
        )
        init_pos[
            placedb.num_nodes:placedb.num_nodes + placedb.num_movable_nodes
        ] = np.random.normal(
            loc=(placedb.yl * 1.0 + placedb.yh * 1.0) / 2,
            scale=(placedb.yh - placedb.yl) * 0.001,
            size=placedb.num_movable_nodes,
        )

    # This branch executes in the maintained flow, where random-center init is off.
    if (
        params.global_place_flag
        and not params.random_center_init_flag
        and placedb.num_movable_nodes > 0
        and len(placedb.regions) > 0
    ):
        movable_regions = placedb.node2fence_region_map[
            0:placedb.num_movable_nodes
        ]
        blend = 0.25

        for i, region in enumerate(placedb.regions):
            region_nodes = np.nonzero(movable_regions == i)[0]
            if region_nodes.size == 0 or region.size == 0:
                continue

            target_density = 0.6
            context_utilization = None
            if die_contexts is not None:
                for context in die_contexts:
                    if context[1] == i:
                        context_utilization = context[2]
                        if context[3] is not None:
                            target_density = context[3]
            target_density = np.clip(target_density, 1.0e-3, 1.0)

            node_x = init_pos[region_nodes]
            node_y = init_pos[placedb.num_nodes + region_nodes]
            node_w = placedb.node_size_x[region_nodes]
            node_h = placedb.node_size_y[region_nodes]
            node_cx = node_x + 0.5 * node_w
            node_cy = node_y + 0.5 * node_h

            # Keep nodes in their current legal component when a region has
            # multiple disjoint boxes. Any anomalous input is assigned to the
            # closest legal component of its owned region.
            box_choice = np.full(region_nodes.size, -1, dtype=np.int32)
            for j, box in enumerate(region):
                inside = (
                    (box_choice < 0)
                    & (node_x >= box[0])
                    & (node_y >= box[1])
                    & (node_x + node_w <= box[2])
                    & (node_y + node_h <= box[3])
                )
                box_choice[inside] = j

            missing = box_choice < 0
            if np.any(missing):
                best_distance = np.full(region_nodes.size, np.inf)
                for j, box in enumerate(region):
                    legal_xh = np.maximum(box[0], box[2] - node_w)
                    legal_yh = np.maximum(box[1], box[3] - node_h)
                    projected_x = np.clip(node_x, box[0], legal_xh)
                    projected_y = np.clip(node_y, box[1], legal_yh)
                    distance = (
                        (node_x - projected_x) * (node_x - projected_x)
                        + (node_y - projected_y) * (node_y - projected_y)
                    )
                    better = missing & (distance < best_distance)
                    box_choice[better] = j
                    best_distance[better] = distance[better]

            for j, box in enumerate(region):
                local_mask = box_choice == j
                local_nodes = region_nodes[local_mask]
                if local_nodes.size == 0:
                    continue

                x = init_pos[local_nodes]
                y = init_pos[placedb.num_nodes + local_nodes]
                w = placedb.node_size_x[local_nodes]
                h = placedb.node_size_y[local_nodes]
                legal_xh = np.maximum(box[0], box[2] - w)
                legal_yh = np.maximum(box[1], box[3] - h)
                x = np.clip(x, box[0], legal_xh)
                y = np.clip(y, box[1], legal_yh)
                cx = x + 0.5 * w
                cy = y + 0.5 * h

                area = np.maximum(w * h, 1.0e-12)
                total_area = np.sum(area)
                box_width = np.maximum(box[2] - box[0], 1.0e-12)
                box_height = np.maximum(box[3] - box[1], 1.0e-12)
                box_area = box_width * box_height

                utilization = total_area / box_area
                if region.shape[0] == 1 and context_utilization is not None:
                    utilization = context_utilization
                footprint_scale = np.sqrt(
                    np.clip(utilization / target_density, 0.05, 1.0)
                )

                order_x = np.argsort(cx, kind="mergesort")
                ordered_area_x = area[order_x]
                quantile_x_ordered = (
                    np.cumsum(ordered_area_x) - 0.5 * ordered_area_x
                ) / total_area
                quantile_x = np.empty(local_nodes.size, dtype=placedb.dtype)
                quantile_x[order_x] = quantile_x_ordered

                order_y = np.argsort(cy, kind="mergesort")
                ordered_area_y = area[order_y]
                quantile_y_ordered = (
                    np.cumsum(ordered_area_y) - 0.5 * ordered_area_y
                ) / total_area
                quantile_y = np.empty(local_nodes.size, dtype=placedb.dtype)
                quantile_y[order_y] = quantile_y_ordered

                x05 = cx[order_x[np.argmin(np.abs(quantile_x_ordered - 0.05))]]
                x95 = cx[order_x[np.argmin(np.abs(quantile_x_ordered - 0.95))]]
                y05 = cy[order_y[np.argmin(np.abs(quantile_y_ordered - 0.05))]]
                y95 = cy[order_y[np.argmin(np.abs(quantile_y_ordered - 0.95))]]

                desired_span_x = box_width * footprint_scale
                desired_span_y = box_height * footprint_scale
                current_span_x = np.maximum(x95 - x05, 0.0)
                current_span_y = np.maximum(y95 - y05, 0.0)
                span_x = np.minimum(
                    box_width, np.maximum(desired_span_x, current_span_x)
                )
                span_y = np.minimum(
                    box_height, np.maximum(desired_span_y, current_span_y)
                )

                mean_cx = np.sum(area * cx) / total_area
                mean_cy = np.sum(area * cy) / total_area
                window_xl = np.clip(
                    mean_cx - 0.5 * span_x, box[0], box[2] - span_x
                )
                window_yl = np.clip(
                    mean_cy - 0.5 * span_y, box[1], box[3] - span_y
                )

                target_cx = window_xl + quantile_x * span_x
                target_cy = window_yl + quantile_y * span_y
                target_cx = np.clip(
                    target_cx, box[0] + 0.5 * w, box[2] - 0.5 * w
                )
                target_cy = np.clip(
                    target_cy, box[1] + 0.5 * h, box[3] - 0.5 * h
                )
                target_x = target_cx - 0.5 * w
                target_y = target_cy - 0.5 * h

                init_pos[local_nodes] = np.clip(
                    (1.0 - blend) * x + blend * target_x,
                    box[0],
                    legal_xh,
                )
                init_pos[placedb.num_nodes + local_nodes] = np.clip(
                    (1.0 - blend) * y + blend * target_y,
                    box[1],
                    legal_yh,
                )

    if placedb.num_filler_nodes:
        if len(placedb.regions) > 0:
            for i, region in enumerate(placedb.regions):
                filler_beg, filler_end = placedb.filler_start_map[i:i + 2]
                subregion_areas = (
                    (region[:, 2] - region[:, 0])
                    * (region[:, 3] - region[:, 1])
                )
                total_area = np.sum(subregion_areas)
                subregion_area_ratio = subregion_areas / total_area
                subregion_num_filler = np.round(
                    (filler_end - filler_beg) * subregion_area_ratio
                )
                subregion_num_filler[-1] = (
                    (filler_end - filler_beg)
                    - np.sum(subregion_num_filler[:-1])
                )
                subregion_num_filler_start_map = np.concatenate(
                    [np.zeros([1]), np.cumsum(subregion_num_filler)], 0
                ).astype(np.int32)

                for j, subregion in enumerate(region):
                    sub_filler_beg, sub_filler_end = (
                        subregion_num_filler_start_map[j:j + 2]
                    )
                    init_pos[
                        placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_beg:
                        placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_end
                    ] = np.random.uniform(
                        low=subregion[0],
                        high=(
                            subregion[2]
                            - placedb.filler_size_x_fence_region[i]
                        ),
                        size=sub_filler_end - sub_filler_beg,
                    )
                    init_pos[
                        placedb.num_nodes
                        + placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_beg:
                        placedb.num_nodes
                        + placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_end
                    ] = np.random.uniform(
                        low=subregion[1],
                        high=(
                            subregion[3]
                            - placedb.filler_size_y_fence_region[i]
                        ),
                        size=sub_filler_end - sub_filler_beg,
                    )

            if len(placedb.filler_start_map) > len(placedb.regions) + 1:
                filler_beg, filler_end = placedb.filler_start_map[-2:]
                init_pos[
                    placedb.num_physical_nodes + filler_beg:
                    placedb.num_physical_nodes + filler_end
                ] = np.random.uniform(
                    low=placedb.xl,
                    high=(
                        placedb.xh
                        - placedb.filler_size_x_fence_region[-1]
                    ),
                    size=filler_end - filler_beg,
                )
                init_pos[
                    placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_beg:
                    placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_end
                ] = np.random.uniform(
                    low=placedb.yl,
                    high=(
                        placedb.yh
                        - placedb.filler_size_y_fence_region[-1]
                    ),
                    size=filler_end - filler_beg,
                )
        else:
            init_pos[
                placedb.num_physical_nodes:placedb.num_nodes
            ] = np.random.uniform(
                low=placedb.xl,
                high=(
                    placedb.xh
                    - placedb.node_size_x[-placedb.num_filler_nodes]
                ),
                size=placedb.num_filler_nodes,
            )
            init_pos[
                placedb.num_nodes + placedb.num_physical_nodes:
                placedb.num_nodes * 2
            ] = np.random.uniform(
                low=placedb.yl,
                high=(
                    placedb.yh
                    - placedb.node_size_y[-placedb.num_filler_nodes]
                ),
                size=placedb.num_filler_nodes,
            )

    return init_pos
# EVOLVE-BLOCK-END