# EVOLVE-BLOCK-START
def initialize_position(params, placedb, die_contexts=None):
    init_pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)
    # x position
    init_pos[0:placedb.num_physical_nodes] = placedb.node_x
    if params.global_place_flag and params.random_center_init_flag:  # move to center of layout
        logging.info(
            "move cells to the center of layout with random noise")
        init_pos[0:placedb.num_movable_nodes] = np.random.normal(
            loc=(placedb.xl * 1.0 + placedb.xh * 1.0) / 2,
            scale=(placedb.xh - placedb.xl) * 0.001,
            size=placedb.num_movable_nodes)

    # y position
    init_pos[placedb.num_nodes:placedb.num_nodes +
                  placedb.num_physical_nodes] = placedb.node_y
    if params.global_place_flag and params.random_center_init_flag:  # move to center of layout
        init_pos[placedb.num_nodes:placedb.num_nodes +
                      placedb.num_movable_nodes] = np.random.normal(
                          loc=(placedb.yl * 1.0 + placedb.yh * 1.0) / 2,
                          scale=(placedb.yh - placedb.yl) * 0.001,
                          size=placedb.num_movable_nodes)

    if placedb.num_filler_nodes:  # uniformly distribute filler cells in the layout
        if len(placedb.regions) > 0:
            ### uniformly spread fillers in fence region
            ### for cells in the fence region
            for i, region in enumerate(placedb.regions):
                filler_beg, filler_end = placedb.filler_start_map[i : i + 2]
                subregion_areas = (region[:, 2] - region[:, 0]) * (region[:, 3] - region[:, 1])
                total_area = np.sum(subregion_areas)
                subregion_area_ratio = subregion_areas / total_area
                subregion_num_filler = np.round((filler_end - filler_beg) * subregion_area_ratio)
                subregion_num_filler[-1] = (filler_end - filler_beg) - np.sum(subregion_num_filler[:-1])
                subregion_num_filler_start_map = np.concatenate(
                    [np.zeros([1]), np.cumsum(subregion_num_filler)], 0
                ).astype(np.int32)
                for j, subregion in enumerate(region):
                    sub_filler_beg, sub_filler_end = subregion_num_filler_start_map[j : j + 2]
                    init_pos[
                        placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_beg : placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_end
                    ] = np.random.uniform(
                        low=subregion[0],
                        high=subregion[2] - placedb.filler_size_x_fence_region[i],
                        size=sub_filler_end - sub_filler_beg,
                    )
                    init_pos[
                        placedb.num_nodes
                        + placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_beg : placedb.num_nodes
                        + placedb.num_physical_nodes
                        + filler_beg
                        + sub_filler_end
                    ] = np.random.uniform(
                        low=subregion[1],
                        high=subregion[3] - placedb.filler_size_y_fence_region[i],
                        size=sub_filler_end - sub_filler_beg,
                    )

            ### for cells outside fence region
            if len(placedb.filler_start_map) > len(placedb.regions) + 1:
                filler_beg, filler_end = placedb.filler_start_map[-2:]
                init_pos[
                    placedb.num_physical_nodes + filler_beg : placedb.num_physical_nodes + filler_end
                ] = np.random.uniform(
                    low=placedb.xl,
                    high=placedb.xh - placedb.filler_size_x_fence_region[-1],
                    size=filler_end - filler_beg,
                )
                init_pos[
                    placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_beg : placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_end
                ] = np.random.uniform(
                    low=placedb.yl,
                    high=placedb.yh - placedb.filler_size_y_fence_region[-1],
                    size=filler_end - filler_beg,
                )

        else:
            init_pos[placedb.num_physical_nodes : placedb.num_nodes] = np.random.uniform(
                low=placedb.xl,
                high=placedb.xh - placedb.node_size_x[-placedb.num_filler_nodes],
                size=placedb.num_filler_nodes,
            )
            init_pos[
                placedb.num_nodes + placedb.num_physical_nodes : placedb.num_nodes * 2
            ] = np.random.uniform(
                low=placedb.yl,
                high=placedb.yh - placedb.node_size_y[-placedb.num_filler_nodes],
                size=placedb.num_filler_nodes,
            )
    return init_pos
# EVOLVE-BLOCK-END
