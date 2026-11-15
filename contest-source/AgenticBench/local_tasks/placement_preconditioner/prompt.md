# Improve placement of a two-tier chip

## Background

A logic-on-logic (LoL) three-dimensional integrated circuit (3D-IC) divides logic
between two stacked tiers. This task places an already partitioned design.
Concurrent DREAMPlace optimizes both tiers together using smooth wirelength and
density penalties, followed by a fixed Nesterov update. Tier assignments stay fixed.

## Your task

Improve gradient scaling by changing only `PreconditionOp`. Position
initialization stays native; this task does not inherit another search's best
program. Better conditioning should help the same optimizer reach shorter wires
while retaining effective density forces.

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

## Class inputs and output

Change only `PreconditionOp`. Every run uses native position initialization,
not an initializer discovered by another search. Keep the supplied methods and
signatures: `__init__(self, placedb, data_collections, op_collections, die_contexts=None)`,
`set_overflow(self, overflow)`, and
`__call__(self, grad, density_weight, update_mask=None, fix_nodes_mask=None)`.

The native preconditioner divides each coordinate gradient by a positive scale
based on connected pin weights and density-weighted node area. In simplified
form, `g'_i = g_i / max(P_i, epsilon)`. This changes optimization step geometry,
not the underlying wirelength or density objectives. Improve conditioning while
keeping density forces effective and avoiding unstable steps.

`data_collections` supplies tensor node areas, pin weights, region maps and
filler ranges. `op_collections` supplies existing reductions. `die_contexts`
contains `(name, region_id, utilization, target_density, aspect_ratio)` tuples
when available; utilization is die-wide, not a supplied per-bin field.
`grad` contains x then y gradients; preserve the native in-place update contract,
fixed-node zeros, region update masks and filler handling. Persistent state may
only belong to this preconditioner instance. Do not mutate physical database
fields, objective inputs or density weights, or bypass the supplied masks.

## Input reference

Construction sets up one preconditioner instance. In the maintained two-region
path, the outer loop updates gamma and does not call `set_overflow`; do not rely
on that callback to activate a change. Keep the method for interface compatibility.
Use the state supplied to construction and `__call__`. `__call__` scales a gradient whenever
the native objective requests preconditioning, which need not be once per
optimizer step. Preserve its in-place gradient update and return that gradient.

| Input or field | Type, layout and meaning |
| --- | --- |
| `placedb` | Read-only geometry/count metadata. Nodes are ordered movable physical, fixed physical, then fillers; region 0/1 is bottom/top. Coordinates remain in the native virtual two-tier canvas. |
| `data_collections.node_areas` | Floating tensor `[num_nodes]`, including fillers, in squared native coordinate units. |
| `data_collections.net_weights` | Floating tensor in native net-index order; fixed objective weights, not candidate parameters. |
| `data_collections.node2fence_region_map` | Integer node-to-region map; use its movable slice and preserve native handling of out-of-region IDs. |
| `data_collections.pos[0]` | Position tensor used to identify the placement dtype/device; do not mutate it. |
| `op_collections.pws_op(net_weights)` | Existing reduction returning a per-node sum of incident pin weights, shaped `[num_nodes]`. Do not replace connectivity or net weights. |
| `die_contexts` | Optional bottom/top tuples `(name, region_id, utilization, target_density, aspect_ratio)`; dimensionless ratios. None means unavailable, and target density may also be None. Utilization is die-wide, not a bin map. |
| `overflow` | Floating tensor of current region overflow; dimensionless. `set_overflow` observes it without mutating it. |
| `grad` | Floating tensor `[2*num_nodes]`, x gradients then y gradients, on the placement device. These are objective derivatives, not coordinates; update this tensor in place. |
| `density_weight` | Floating tensor: one shared weight or a vector of region weights. The native code selects by region; preserve filler and non-fence handling. Never change this input. |
| `update_mask` | Optional Boolean region tensor: True allows that region to update; False means zero its movable/filler gradients. None skips this region mask. |
| `fix_nodes_mask` | Optional Boolean node mask: True freezes that movable node's x/y gradient. Preserve the native movable slice; None adds no such mask. |

Use `placedb.filler_start_map` as offsets within the filler suffix; it can
contain an additional non-fence interval. With no fillers, do not index an
imaginary filler region. Preserve dtype/device and fixed-node behavior.
Torch and logging are supplied. Additional numerical instance methods are
allowed; keep the three entrypoint signatures. Helpers take `self`, use plain
arguments and constant defaults, and are called directly without recursion,
decorators or aliases. Instance-local state is allowed, global state is not.

The maintained runtime uses Python 3.8 and Torch 1.7.1. The adapter supplies
`torch.nan_to_num` for real tensors while preserving dtype/device and finite
values; it does not modify global Torch state. Use this module function instead
of newer tensor methods. Other numerical APIs must exist in the installed
runtime. Imports and physical-input mutation remain forbidden.

GP progress summaries describe the whole optimization, not the preconditioner
alone. Gradient-scale distributions and accepted-step statistics are not
measured by the current feedback; do not infer them from the final score.

### Feedback field guide

The supplied `metric_definitions` gives a short meaning and unit for each returned
metric. Missing metrics are unknown, not zero. Only `combined_score` is the
selection objective; activity and progress measurements explain behavior but
are not additional rewards.

## What to return

Return only the complete class inside EVOLVE-BLOCK markers.
