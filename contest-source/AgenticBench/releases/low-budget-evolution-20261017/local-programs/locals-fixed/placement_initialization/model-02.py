# EVOLVE-BLOCK-START
def initialize_position(params, placedb, die_contexts=None):
    """Retain a density-scaled fraction of each tier's original placement geometry."""
    init_pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)

    # Preserve every physical coordinate before selectively moving movable nodes.
    init_pos[0:placedb.num_physical_nodes] = placedb.node_x
    init_pos[
        placedb.num_nodes:placedb.num_nodes + placedb.num_physical_nodes
    ] = placedb.node_y

    if params.global_place_flag and params.random_center_init_flag:
        logging.info(
            "contract original tier-local geometry toward legal region centers"
        )
        num_movable = placedb.num_movable_nodes
        movable_regions = placedb.node2fence_region_map[0:num_movable]

        trusted_regions = np.ones(len(placedb.regions), dtype=np.bool_)
        region_spread = np.full(
            len(placedb.regions), 0.45, dtype=placedb.dtype
        )

        if die_contexts is not None:
            trusted_regions[:] = False
            for context in die_contexts:
                region_id = context[1]
                if region_id >= 0 and region_id < len(placedb.regions):
                    trusted_regions[region_id] = True
                    utilization = context[2]
                    target_density = context[3]
                    if target_density is None:
                        target_density = 0.6
                    if utilization is not None and target_density > 0:
                        occupancy_ratio = np.minimum(
                            np.maximum(utilization / target_density, 0.0),
                            1.0,
                        )
                        # Use part of the ideal density footprint so the start
                        # remains wirelength-biased rather than fully dispersed.
                        region_spread[region_id] = np.minimum(
                            np.maximum(
                                0.55 * np.sqrt(occupancy_ratio), 0.20
                            ),
                            0.60,
                        )

        for region_id, region in enumerate(placedb.regions):
            if not trusted_regions[region_id] or region is None or region.size == 0:
                continue

            node_ids = np.nonzero(movable_regions == region_id)[0]
            if node_ids.size == 0:
                continue

            widths = placedb.node_size_x[node_ids]
            heights = placedb.node_size_y[node_ids]
            original_cx = placedb.node_x[node_ids] + widths * 0.5
            original_cy = placedb.node_y[node_ids] + heights * 0.5

            # Retain the parent's legal nearest-subregion ownership.
            chosen_box = np.zeros(node_ids.size, dtype=np.int32) - 1
            best_distance = np.full(
                node_ids.size, 1.0e30, dtype=placedb.dtype
            )
            for box_id, box in enumerate(region):
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                fits = (widths <= box_width) & (heights <= box_height)
                box_cx = (box[0] + box[2]) * 0.5
                box_cy = (box[1] + box[3]) * 0.5
                dx = (original_cx - box_cx) / np.maximum(
                    box_width, 1.0e-12
                )
                dy = (original_cy - box_cy) / np.maximum(
                    box_height, 1.0e-12
                )
                distance = dx * dx + dy * dy
                update = fits & (distance < best_distance)
                chosen_box[update] = box_id
                best_distance[update] = distance[update]

            for box_id, box in enumerate(region):
                local_mask = chosen_box == box_id
                local_ids = node_ids[local_mask]
                if local_ids.size == 0:
                    continue

                local_widths = placedb.node_size_x[local_ids]
                local_heights = placedb.node_size_y[local_ids]
                local_original_cx = (
                    placedb.node_x[local_ids] + local_widths * 0.5
                )
                local_original_cy = (
                    placedb.node_y[local_ids] + local_heights * 0.5
                )
                local_areas = local_widths * local_heights
                area_sum = np.maximum(np.sum(local_areas), 1.0e-12)
                source_cx = np.sum(
                    local_original_cx * local_areas
                ) / area_sum
                source_cy = np.sum(
                    local_original_cy * local_areas
                ) / area_sum

                legal_xh = box[2] - local_widths
                legal_yh = box[3] - local_heights
                legal_center_x = (box[0] + legal_xh) * 0.5
                legal_center_y = (box[1] + legal_yh) * 0.5
                spread = region_spread[region_id]

                candidate_x = (
                    legal_center_x
                    + spread * (local_original_cx - source_cx)
                    + np.random.normal(
                        loc=0.0,
                        scale=(box[2] - box[0]) * 0.001,
                        size=local_ids.size,
                    )
                )
                candidate_y = (
                    legal_center_y
                    + spread * (local_original_cy - source_cy)
                    + np.random.normal(
                        loc=0.0,
                        scale=(box[3] - box[1]) * 0.001,
                        size=local_ids.size,
                    )
                )

                init_pos[local_ids] = np.minimum(
                    np.maximum(candidate_x, box[0]), legal_xh
                )
                init_pos[placedb.num_nodes + local_ids] = np.minimum(
                    np.maximum(candidate_y, box[1]), legal_yh
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
                    begin = (
                        placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_beg
                    )
                    end = (
                        placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_end
                    )
                    init_pos[begin:end] = np.random.uniform(
                        low=subregion[0],
                        high=(
                            subregion[2]
                            - placedb.filler_size_x_fence_region[i]
                        ),
                        size=sub_filler_end - sub_filler_beg,
                    )
                    init_pos[
                        placedb.num_nodes + begin:placedb.num_nodes + end
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
                begin = placedb.num_physical_nodes + filler_beg
                end = placedb.num_physical_nodes + filler_end
                init_pos[begin:end] = np.random.uniform(
                    low=placedb.xl,
                    high=(
                        placedb.xh
                        - placedb.filler_size_x_fence_region[-1]
                    ),
                    size=filler_end - filler_beg,
                )
                init_pos[
                    placedb.num_nodes + begin:placedb.num_nodes + end
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