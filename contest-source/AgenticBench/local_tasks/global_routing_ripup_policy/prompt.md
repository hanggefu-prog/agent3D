# Task: improve global-routing ripup policy

## Background

In a logic-on-logic (LoL) three-dimensional integrated circuit (3D-IC), routes
connect cells within each die and reach hybrid bonding terminals (HBTs) for
cross-die connections. After clock tree synthesis (CTS), global routing (GRT)
assigns coarse paths through a grid with limited wire capacity. FastRoute
repeatedly removes and reroutes connections to reduce overflow while avoiding
unnecessary wire length. Some nets use non-default routing rules (NDRs), which
can change their routing demand.

## Your task

Evolve `score_ripup_edge(ctx)`, which decides whether an eligible routed
Steiner-tree edge should be removed and rerouted. A score at least 0.5 selects
it; a smaller score keeps it. Returning `ctx.baseline_decision` reproduces
the native decision. A non-finite result falls back to that decision.

The native rule selects a route if usage reaches capacity minus the current
rip-up threshold. It can also select a timing-critical route that is at least
twice its previous length. Your function runs after this decision is computed
but before routing resources are released. It changes eligibility, not net
ordering, path costs or the iteration schedule.

Explore whether local congestion, historical pressure and detour justify
rerouting, or whether an already useful connection should be left alone.
Changing a score without crossing 0.5 cannot change the decision.

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

Rip-up feedback reports observed cumulative calls, changed decisions and
selected decisions. `selected_ratio_*` summarizes how often removal was
selected in logged cumulative samples; `score_*` summarizes logged output
scores. Logs are sparse: counts are lower bounds, and min/max/last are sampled
values, not exact final totals or extrema over every call. These diagnostics
can distinguish an inactive rule from broad removal or selective changes.

## Immutable context

All fields are finite. Except for `baseline_decision`, which is exactly `0.0`
or `1.0`, fields are normalized to `[0, 1]`:

- `baseline_decision`: maintained FastRoute's decision; returning it is exact
  baseline behavior;
- `edge_overflow`: aggregate positive overflow along this routed edge;
- `max_utilization`: largest bounded usage pressure along the route;
- `history_pressure`: aggregate historical congestion on the route;
- `detour_ratio`: excess current route length above endpoint Manhattan length;
- `route_length`: current route length relative to die grid dimensions;
- `ripup_threshold`: bounded representation of FastRoute's current threshold;
- `criticality`: timing criticality; sentinel/unconstrained slack maps to zero;
- `is_clock`, `has_ndr`, `is_hbt`: net-class indicators. HBT membership is
  matched against the frozen HBT boundary-port plan;
- `die_id`: `0.0` for bottom and `1.0` for top;
- `die_overflow`: current total die-overflow pressure.

The features are recomputed before every edge decision. A non-finite return
falls back to `baseline_decision`. Diagnostics are interpreted as described above.

For threshold design, the input transforms are explicit. Along the current
route let `u = max(0, reduced_usage)`, `h = max(0, historical_usage)`,
`c = max(1, directional_capacity)`, `E = sum(max(0, u-c))`,
`H = sum(h)`, `C = sum(c)`, and `L = max(0, route_grid_steps)`.
Then `edge_overflow = E / (E + max(1, L))`,
`history_pressure = H / (H + C)`, and `max_utilization = max(u / (u+c))`
over that routed edge. `detour_ratio = max(0, L-Manhattan) / L` for positive
L, otherwise zero; `route_length = L / (L + max(1, x_grid+y_grid))`.
`die_overflow = O / (O + max(1, routed_net_count))` with nonnegative total
overflow O. `ripup_threshold = 0.5 + atan(native_threshold)/pi`.
Criticality is zero for sentinel/non-finite slack, otherwise
`1 / (1 + max(0, native_slack))`; it is not a calibrated timing probability.
These observations do not permit changing capacity, timing, or routing budgets.

## Allowed language

Use numeric/boolean constants, mutable local scalar assignments, augmented
assignments, nested `if`/`elif`/`else`, conditional expressions, early returns,
and statically bounded `for` loops over constant `range(...)` bounds. One loop
may execute at most 64 times and nested loops at most 256 total iterations;
`break`, `continue`, and `pass` are supported. Scalars assigned in all arms of
an `if` remain available afterward.
The isolated patch permits up to 1,800 added lines, including trusted hook code;
this is a code-size ceiling, not a requirement to generate long policies.

Supported arithmetic is `+`, `-`, `*`, `/`, `%`, `//`, and `**`, comparisons
and Boolean operators. Built-ins are `abs`, `min`, `max`, `float`, and `round`.
Safe `math` functions include `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`, `tanh`, `sin`,
`cos`, `asin`, `acos`, `atan`, `atan2`, `hypot`, `pow`, `floor`, `ceil`,
`fabs`, `erf`, `copysign`, and `fmod`.

Both `ctx.field` and `ctx["field"]` read the same declared immutable field.
Local variables may reuse field names, e.g. `die_id = ctx.die_id`.
Nested branches and `elif` preserve Python function-local assignment semantics;
every local must be assigned on all paths that reach its use.

Read-only flat numeric tuple/list tables and string-keyed numeric dictionaries
are supported. Declare each table once at function-body level, outside branches
and loops (1–64 entries per table, at most 256 entries total). Entries may use
context or already-defined scalar expressions and capture their values when
declared. Use integer literal indices (including negative indices), literal
dictionary keys, or a constant-range loop variable whose entire range fits the
table. `len(table)` is supported, including in `range(len(table))`; the rip-up
64/256 loop limits still apply. Table mutation, aliasing, reassignment, nested
containers, slices, and indices computed from runtime context are unsupported.
Imports, data-dependent loops, I/O, randomness and external calls are forbidden.

For example, a bounded classifier can use shared feature weights and optionally
select a threshold per die:

```python
def score_ripup_edge(ctx):
    weights = (0.6, 0.3, 0.1)
    features = (ctx["edge_overflow"], ctx.history_pressure, ctx.detour_ratio)
    thresholds = {"bottom": 0.30, "top": 0.35}
    score = 0.0
    for i in range(len(weights)):
        score += weights[i] * features[i]
    if ctx.die_id < 0.5:
        threshold = thresholds["bottom"]
    else:
        threshold = thresholds["top"]
    return 1.0 if score >= threshold else 0.0
```

This is an interface example, not a preferred starting policy. Shared thresholds
are equally valid; do not assume one die is always more congested.
`max_utilization` is the maximum of `usage / (usage + capacity)`: 0.5 means
usage equals capacity. `edge_overflow = sum(excess) / (sum(excess) + route_steps)`.
`detour_ratio = max(0, route_steps - Manhattan_distance) / route_steps`.
`die_overflow = total_overflow / (total_overflow + net_count)`.
These are distinct bounded signals, not interchangeable raw percentages.
Keep arithmetic finite and guard division/log/sqrt domains.

Wirelength recovery can admit clean but long-detoured edges that baseline would
keep. Suppressing needless rip-up can also reduce churn and detours. Test these
as separate hypotheses, then combine only when feedback supports both. Merely
changing the returned score without crossing 0.5 cannot affect routing.


## Reading the function interface

One call considers an eligible routed connection under the native rip-up traversal; it does not score the whole design.

| Argument | Type, ordering and missing cases |
| --- | --- |
| `ctx` in `score_ripup_edge` | Read-only scalar context; `ctx.field` and `ctx["field"]` are equivalent. No arrays, mutable route objects or implicit node-index order are exposed. |
| Field values | Finite numeric scalars. Use the definitions and exact transforms in **Immutable context** above; normalized pressure is dimensionless, not a raw percentage. |
| `die_id` | Exactly 0 for bottom and 1 for top. This is physical tier identity, not a congestion ranking. |
| Missing information | Documented fields are required by the hook. Unexposed timing paths, grids or per-site data cannot be recovered from a missing value or treated as zero. |

Return a finite scalar; the trusted comparison uses 0.5 as the reroute threshold. Do not mutate context or retain state between calls. Feedback definitions
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
