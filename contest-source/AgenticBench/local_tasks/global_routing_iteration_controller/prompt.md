# Task: improve global-routing iteration controller

## Background

In a logic-on-logic (LoL) three-dimensional integrated circuit (3D-IC), routes
connect cells within each die and reach hybrid bonding terminals (HBTs) for
cross-die connections. After clock tree synthesis (CTS), global routing (GRT)
assigns coarse paths through a grid with limited wire capacity. FastRoute
repeatedly removes and reroutes connections to reduce overflow while avoiding
unnecessary wire length. Some nets use non-default routing rules (NDRs), which
can change their routing demand.

## Your task

Evolve `control_reroute_iteration(ctx)`, which adjusts congestion pressure,
search range and removal threshold once per main overflow iteration. It runs
after the native heuristics update their state and before maze rerouting.
Return `(congestion_scale, search_scale, ripup_offset)`.

The congestion scale is clamped to [0.25, 4.0] and multiplies cost height, which
remains at least 1. The search scale is clamped to [0.5, 2.5]; the scaled window
is rounded and limited to [1, max(1, x_grid/2)]. The offset is clamped to
[-4, 4] and rounded before being added to the removal threshold, whose final
range is [-8, 8]. Each non-finite component independently uses its identity
value. Returning (1.0, 1.0, 0.0) preserves the native schedule.

Explore, for example, stronger intervention during stagnation or a gradual shift
toward shorter routes as overflow improves. The three controls need not move
together. Increasing effort or congestion pressure is not always beneficial;
the native phase rules, termination criteria and iteration budget remain intact.

A shared policy and a die-specific policy are both allowed. Use the supplied
physical observations to justify asymmetry; do not assume that the bottom or
top die is always more congested.

## How solutions are evaluated

Both dies must finish global routing with valid measurements, and at least one
die must provide valid hook evidence. A die that converges early can legitimately
skip the hook. Let `O = O_bottom + O_top` be total overflow and
`W = W_bottom + W_top` be total global-route wire length in micrometers.

`overflow_cost = (O + 1000) / (O_ref + 1000)`

`wirelength_cost = W / W_ref`

`combined_score = grt_complete / max(0.10 * overflow_cost + 0.90 * wirelength_cost, 1e-12)`

The reference wire length must be positive. The overflow offset preserves a
useful signal when reference overflow is zero.

Higher `combined_score` is better; a valid reference compared with itself
scores 1.0. Wire length has weight 0.9 and overflow 0.1. The objective allows
trade-offs between metrics and dies, not an independent no-regression test for
each. `grt_complete` includes the hook-evidence gate, not just process exit.

The input placement and clock tree, routing layers, capacities, native loop
budget and all other algorithm components stay unchanged. Only the declared
function can evolve. This evaluation does not measure detailed routing, antenna
repair, design rule checking (DRC), routed timing or thermal behavior.

## Using previous results

Check validity and hook evidence before comparing scores. Feedback includes
`grt_total_overflow`, `grt_wirelength_um`, `grt_congestion_bins`, and
their `bottom_` and `top_` values. Wire length and overflow are costs:
lower is better. Ratio scores increase with improvement; normalized cost
terms decrease with improvement. Runtime is a diagnostic only when measured,
not a reward or pure algorithm time. Missing metrics remain unknown, never zero.

Activity diagnostics explain whether the policy changed routing decisions, not
whether it improved the physical result. A changed policy with unchanged final
metrics can reflect routing insensitivity or convergence to similar results;
it does not establish that the actual paths are identical. Repair reported
implementation failures before judging the strategy. Try one main change at
a time and combine ideas when comparable measured results support them.

Iteration feedback counts logged invocations and invocations with any
nonidentity correction. The native log field named `calls` is an iteration
index, not a cumulative count; feedback labels its last value
`iteration_index_last`. That index can reset during NDR handling.
Per-die min/max/last diagnostics report `cost_scale`, `search_scale`,
`ripup_offset` and the resulting `cost_height`, `search_window` and
`ripup_threshold`. These reveal the applied control without claiming to
measure a full congestion trajectory.

## Immutable context

Every field is finite. `baseline_congestion_scale` and
`baseline_search_scale` are exactly `1.0`, and `baseline_ripup_offset` is
exactly `0.0`; returning that tuple is exact baseline behavior. Most other
fields are bounded to `[0, 1]`; `search_window` and `reroute_phase` are
nonnegative ratios without an explicit upper clamp. Do not assume they are
always at most one, or that reroute phase cannot reset after an NDR restart:

- `die_overflow`: current total overflow relative to routed-net count;
- `max_overflow`: current maximum single-edge overflow pressure;
- `congestion_trend`: `0.5` means flat/unknown, above `0.5` means the previous
  iteration improved overflow, and below `0.5` means regression;
- `stagnation`: bounded consecutive hook calls since best overflow improved;
- `wirelength_pressure`: total 2D route usage relative to grid capacity;
- `reroute_phase`: current iteration divided by the fixed iteration budget;
- `cost_height`, `search_window`, `ripup_threshold`: normalized maintained
  control state before the learned correction;
- `history_weight`: whether historical usage currently contributes to cost;
- `hbt_pressure`, `ndr_pressure`, `critical_pressure`: fractions of current
  routed nets in these classes. HBT membership comes from the frozen boundary
  plan;
- `die_id`: `0.0` for bottom and `1.0` for top.

The exact normalization is part of the frozen interface. Define
`N = max(1, routed_net_count)`, `K = max(1, h_capacity + v_capacity)`,
`C = max(1, x_grid * y_grid) * K`, `O = max(0, total_overflow)`,
`M = max(0, maximum_edge_overflow)`, `H = max(0, costheight)`,
`U = max(0, total_2d_route_usage)`, and `S` as the number of hook calls since
the best overflow improved (zero on improvement). Then:

- `die_overflow = O / (O + N)` and `max_overflow = M / (M + K)`;
- `cost_height = H / (H + 50)` and `wirelength_pressure = U / (U + C)`;
- `stagnation = S / max(1, S + 5)`;
- `congestion_trend = 0.5 + 0.5 * clamp((O_previous - O) / O_previous, -1, 1)`
  when previous overflow is positive, otherwise `0.5`;
- `reroute_phase = max(0, iteration) / max(1, congestion_iterations)`;
- `search_window = max(0, enlargement) / max(1, max(x_grid, y_grid))`;
- `ripup_threshold = 0.5 + atan(native_ripup_threshold) / pi`;
- `history_weight = 1` when native history mode `L != 0`, otherwise `0`;
- each class pressure is the corresponding current routed-net count divided
  by `N`.

These are input transforms, not candidate-editable settings. If a candidate
needs an upper-bounded phase/window internally, explicitly bound its own
expression without modifying the native context or loop budget.

Every invocation reports die, iteration, whether any output changed maintained
behavior, both applied scales, the rip-up offset, and final control values.

## Allowed language

Use numeric/boolean constants, mutable local scalar assignments, augmented
assignments, nested `if`/`elif`/`else`, conditional expressions, early returns,
and statically bounded `for` loops over constant `range(...)` bounds. One loop
may execute at most 64 times and nested loops at most 256 total iterations;
`break`, `continue`, and `pass` are supported. Scalars assigned in every branch
remain available afterward.

Supported arithmetic is `+`, `-`, `*`, `/`, `%`, `//`, and `**`, comparisons
and Boolean operators. Built-ins are `abs`, `min`, `max`, `float`, and `round`.
Safe `math` functions include `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`, `tanh`, `sin`,
`cos`, `asin`, `acos`, `atan`, `atan2`, `hypot`, `pow`, `floor`, `ceil`,
`fabs`, `erf`, `copysign`, and `fmod`.

Both `ctx.field` and `ctx["field"]` read declared immutable context fields.
Local scalars may reuse field names. Nested branches preserve Python
function-local assignments; every local must be assigned on every path reaching
its use.

Read-only flat numeric tuple/list tables and string-keyed numeric dictionaries
are supported. Declare each table once at function-body level, outside branches
and loops (1–64 entries per table, at most 256 entries total). Entries capture
context or already-defined scalar expressions when declared. Use integer literal
indices, including negative indices, literal dictionary keys, or a constant-range
loop variable whose entire range fits the table. `len(table)` is supported,
including in `range(len(table))`; the 64/256 loop budgets still apply.
Table mutation, aliasing, reassignment, nested containers, slices, runtime-context
indices, imports, data-dependent loops, I/O, randomness and external calls are
forbidden. Return the fixed three-scalar control tuple as before.
The patch-added-line ceiling is 1,800 including trusted hooks, not a target size.


## Reading the function interface

One call follows native heuristic updates and precedes main maze rerouting in an overflow iteration. Native iteration indices can reset.

| Argument | Type, ordering and missing cases |
| --- | --- |
| `ctx` in `control_reroute_iteration` | Read-only scalar context; `ctx.field` and `ctx["field"]` are equivalent. No arrays, mutable route objects or implicit node-index order are exposed. |
| Field values | Finite numeric scalars. Use the definitions and exact transforms in **Immutable context** above; normalized pressure is dimensionless, not a raw percentage. |
| `die_id` | Exactly 0 for bottom and 1 for top. This is physical tier identity, not a congestion ranking. |
| Missing information | Documented fields are required by the hook. Unexposed timing paths, grids or per-site data cannot be recovered from a missing value or treated as zero. |

Return three scalars in the exact order (congestion_scale, search_scale, ripup_offset); the trusted clamps are described above. Do not mutate context or retain state between calls. Feedback definitions
state whether a value is a count, index, normalized pressure or logged sample
statistic; those quantities are not interchangeable.

### Feedback field guide

The supplied `metric_definitions` gives a short meaning and unit for each returned
metric. Missing metrics are unknown, not zero. Only `combined_score` is the
selection objective; activity and progress measurements explain behavior but
are not additional rewards.

## What to return

Return the complete required function with its exact signature inside
EVOLVE-BLOCK markers. Include a short comment explaining the change.
No imports, extra definitions, JSON, diffs, tests or surrounding prose.
The program limit is 64,000 characters. The source-edit budget includes
the generated hook: 1,800 added and 24 deleted lines across at most three files.
Only the documented scalar inputs and operations are available; do not
assume access to full grids, route databases or unreported timing measurements.
