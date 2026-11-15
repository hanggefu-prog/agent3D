# Improve placement of a two-tier chip

## Background

A logic-on-logic (LoL) three-dimensional integrated circuit (3D-IC) divides logic
between two stacked tiers. This task places an already partitioned design.
Concurrent DREAMPlace optimizes both tiers together using smooth wirelength and
density penalties, followed by a fixed Nesterov update. Tier assignments stay fixed.

## Your task

Improve the starting positions by changing only `initialize_position`. The
gradient preconditioner stays native; this task does not inherit another search's
best program. A useful starting pattern should help the later placer reach
shorter wires without leaving excessive density overflow.

## How results are compared

Evaluation stops at global placement, before legalization or static timing
analysis (STA). Minimize final weighted half-perimeter wirelength (HPWL) while
controlling the worst of the two dies' density overflows. For a valid run:

`score = 0.9 * H_ref / max(H, 1e-12) + 0.1 * max(O_ref, 0.01) / max(O, 0.01)`.

Higher is better and the native reference scores 1. Overflow below the existing
0.01 stopping tolerance earns no further reward. Inspect per-die overflow and
maximum density as diagnostics; they are not separate rewards. Runtime is
reported but is not part of fitness. Missing values are unknown, never zero.
A failed run or rejected program is not evidence of a physical improvement.

## Shared constraints

Keep the partition, fixed objects, net weights, objective, density update,
Nesterov optimizer and stopping rules unchanged. Use at most 4000 global
placement iterations, a 512-bin grid, target density 0.6, initial density weight
0.00003 and learning rate 0.01. Physical macro halos follow the selected process
design kit (PDK) and are not search variables. No imports, external input/output,
file access, subprocesses, network access, global runtime changes or new random
seeds. Preserve array/tensor shapes, dtype and device. Do not change inputs or
database metadata. Use supplied numerical libraries and bounded finite operations.
Avoid dense all-pairs structures and work that scales quadratically in node count.

Try one main change at a time. Retain mechanisms supported by comparable
evaluations and explain the change briefly in the returned definition's docstring.

## Function inputs and output

Change only `initialize_position(params, placedb, die_contexts=None)`.
The native gradient preconditioner is used for every candidate, including the
reference. Improve the starting coordinates, not the placement optimizer.

`placedb` supplies original positions, node sizes, movable/fixed counts, pin/net
connectivity, die fence boxes and node-to-region assignments. `die_contexts` is
either None or two tuples `(name, region_id, utilization, target_density, aspect_ratio)`
for bottom and top. Utilization is total movable area divided by that die's fence
area; it is not a local density map. If needed, estimate a coarse local map from
the supplied geometry without changing the database.

Return a NumPy vector of length `2 * placedb.num_nodes`: all x coordinates
followed by all y coordinates. Preserve fixed nodes exactly and keep movable
objects in their assigned regions. The evaluator uses the native initializer for
the baseline. The existing seeded NumPy random stream may be used, but never reset
or replace it. Initialization is called once, so consider whether a geometry-
aware starting pattern can help the later optimizer spread cells and shorten nets.

## Input reference

The initializer runs once before global placement. The maintained concurrent
driver sets `random_center_init_flag=0`, so edits inside the native random-center
branch will not affect these evaluations. You may transform the supplied movable
coordinates in this function while preserving fixed nodes and region ownership;
do not change the read-only flag. Read the supplied objects;
write only the new position vector. Geometry uses DREAMPlace's native coordinate
system, including the virtual offset between tiers, not necessarily micrometers.
Keep that coordinate system; do not project the top tier into the bottom tier.

| Input or field | Type, layout and meaning |
| --- | --- |
| `params.global_place_flag`, `params.random_center_init_flag` | Boolean-like flags controlling the native starting-position policy; read-only. |
| `placedb.num_nodes`, `num_physical_nodes`, `num_movable_nodes`, `num_filler_nodes` | Integer counts. Node order is movable physical nodes, fixed physical nodes, then artificial filler nodes. Fillers model unused placement area; they are not circuit cells. |
| `placedb.node_x/node_y` | NumPy arrays for physical nodes, in node-index order; original lower-left coordinates. |
| `placedb.node_size_x/node_size_y` | NumPy arrays of node widths/heights in the same coordinate units; never change them. |
| `placedb.xl/yl/xh/yh` | Numeric bounds of the placement canvas, not separate per-tier core bounds. |
| `placedb.regions` | List of arrays shaped `[boxes,4]`, each row `[xl,yl,xh,yh]`. Region 0 is bottom, 1 is top in the maintained two-tier flow. Keep the whole node rectangle within its region. |
| `placedb.node2fence_region_map` | Integer node-to-region IDs; use the movable-node slice. Do not treat IDs outside the region range as valid die indices. |
| `placedb.filler_start_map` | Integer offsets into the filler-only suffix. Region i owns `[map[i],map[i+1])`; add `num_physical_nodes` to obtain full node indices. |
| `placedb.filler_size_x_fence_region/y_fence_region` | Per-region filler dimensions; use these when sampling legal filler coordinates. |
| `die_contexts` | None when the trusted two-region mapping is unavailable; otherwise bottom/top tuples `(name, region_id, utilization, target_density, aspect_ratio)`. Ratios are dimensionless; aspect ratio is bounding-box width/height. A tuple's target density may itself be None. |
| Return value | New NumPy array `[2*num_nodes]`, using `placedb.dtype`: all x, then all y. Fixed physical coordinates must be unchanged. |

An empty filler suffix needs no filler work. Do not invent optional database
attributes: use the fields demonstrated here and in the supplied program.
NumPy and logging are supplied. You may define local numerical helpers inside
`initialize_position`, using plain arguments and constant defaults. Call helpers
directly; no recursion, decorators, aliases or further nested definitions.
Imports remain unavailable. Use explicit None/size checks on documented fields
instead of reflective access such as `hasattr`.

The first logged GP metric is measured after the optimizer has started. It is
not a direct measurement of the vector returned by this initializer.

### Feedback field guide

The supplied `metric_definitions` gives a short meaning and unit for each returned
metric. Missing metrics are unknown, not zero. Only `combined_score` is the
selection objective; activity and progress measurements explain behavior but
are not additional rewards.

## What to return

Return only the complete function inside EVOLVE-BLOCK markers.
