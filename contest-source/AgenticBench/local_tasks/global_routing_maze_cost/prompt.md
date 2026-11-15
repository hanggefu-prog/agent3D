# Task: improve global-routing maze cost

## Background

In a logic-on-logic (LoL) three-dimensional integrated circuit (3D-IC), routes
connect cells within each die and reach hybrid bonding terminals (HBTs) for
cross-die connections. After clock tree synthesis (CTS), global routing (GRT)
assigns coarse paths through a grid with limited wire capacity. FastRoute
repeatedly removes and reroutes connections to reduce overflow while avoiding
unnecessary wire length. Some nets use non-default routing rules (NDRs), which
can change their routing demand.

## Your task

Evolve `score_maze_step(ctx)`, a multiplier on the native grid-edge cost
used by Dijkstra maze search. It runs for each horizontal or vertical edge
considered during expansion, including the secondary edge used by the turn
shortcut. Returning 1.0 preserves the native cost. A non-finite value becomes
1.0; finite multipliers are clamped to [0.05, 20.0], keeping costs positive.

The native edge cost combines a logistic congestion penalty and a linear
overflow penalty. Search also adds a separate via/turn cost. Your multiplier
can change the selected paths without changing removal eligibility or the
outer-loop budget. A uniform multiplier can change the edge-versus-turn
balance, but does not itself respond to local congestion.

Explore a compact response to current overflow, congestion history, net class,
or reroute progress. Guard numerical domains and keep work per call small:
this function can run millions of times.

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

Maze feedback reports observed cumulative calls, counts of multipliers
different from 1.0, maximum logged changed ratio, and `multiplier_min`,
`multiplier_max`, `multiplier_last` for each die. Logging is sparse:
counts are lower bounds and multiplier extrema describe logged samples only.
A changed multiplier proves activity, not a different path or improved routing.

## Immutable context

Every field is finite and normalized to `[0, 1]`:

- `baseline_cost`: bounded representation of FastRoute's maintained edge cost;
- `congestion_pressure`: current reduced usage relative to capacity;
- `edge_overflow`: positive overflow pressure on this grid edge;
- `history_pressure`: accumulated historical congestion on this edge;
- `is_horizontal`: horizontal-versus-vertical direction indicator;
- `is_turn`: the expansion also pays a via/turn cost;
- `criticality`: timing criticality; sentinel/unconstrained slack maps to zero;
- `is_clock`, `is_hbt`: net-class indicators. HBT membership is matched to the
  frozen boundary-port plan;
- `wirelength`: bounded Steiner-tree length estimate, sampled at maze-iteration entry;
- `die_id`: `0.0` for bottom and `1.0` for top;
- `die_overflow`: total current die-overflow pressure;
- `reroute_phase`: bounded iteration indicator (it can reset).

Sparse diagnostics are interpreted as described above. Because the hook is in the search inner loop, a highly complex
policy can increase runtime even if it is verifier-valid.

For threshold design, define `u = max(0, reduced_usage)`,
`c = max(1, directional_capacity)`, `h = max(0, historical_usage)`,
`e = max(0, u-c)`, `O = max(0, total_die_overflow)`,
`N = max(1, routed_net_count)`, and `L = max(0, current_net_length)`.
The input transforms are `congestion_pressure = u/(u+c)`,
`edge_overflow = e/(e+c)`, `history_pressure = h/(h+c)`,
`die_overflow = O/(O+N)`, and `wirelength = L/(L+max(1, x_grid+y_grid))`.
`baseline_cost = native_edge_cost/(native_edge_cost+max(1, cost_height))`;
`reroute_phase = max(0, iteration)/max(1, iteration+8)` (the native iteration
can reset after an NDR restart, so this is not globally monotone across resets).
Criticality is zero for sentinel/non-finite slack, otherwise
`1/(1+max(0, native_slack))`; it is not a calibrated timing probability.
Class/direction/turn indicators are exactly zero or one. These transforms are
read-only observations, not tunable physical settings.

## Allowed language and runtime budget

Use numeric/boolean constants, mutable local scalar assignments, augmented
assignments, nested `if`/`elif`/`else`, conditional expressions, early returns,
and statically bounded loops over constant `range(...)` bounds. One loop may
execute at most 8 times and nested loops at most 16 total iterations; `break`,
`continue`, and `pass` are supported. These tighter loop limits apply only
because the function can be called millions of times; branches and multi-stage
scalar logic remain unrestricted within the overall abstract syntax tree (AST) and code limit.
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
table. `len(table)` is supported, including in `range(len(table))`; the maze
8/16 loop limits still apply. Table mutation, aliasing, reassignment, nested
containers, slices, and indices computed from runtime context are unsupported.
Imports, data-dependent loops, I/O, randomness and external calls are forbidden.

For example, a shared policy can combine a small feature table:

```python
def score_maze_step(ctx):
    weights = (0.04, 0.02)
    features = (ctx["edge_overflow"], ctx.history_pressure)
    multiplier = 1.0
    for i in range(len(weights)):
        multiplier += weights[i] * features[i]
    return multiplier
```

To explore die-specific behavior, branch on `ctx.die_id` and choose different
coefficients. Shared policies are equally valid; do not assume one die is always
more congested. Context and feedback should justify any asymmetry.

`congestion_pressure = usage / (usage + capacity)`: 0.5 means usage equals
capacity, not 50% utilization. `edge_overflow = excess / (excess + capacity)`.
`reroute_phase = iteration / (iteration + 8)` is not the fraction of the fixed
iteration budget used. Thresholds should respect these normalizations.
Keep costs finite and guard division/log/sqrt domains. A uniform multiplier
changes edge-versus-turn cost balance; it does not by itself express spatial or
die-specific congestion awareness.


## Reading the function interface

One call observes one grid edge during maze expansion, including supported turn-shortcut checks; a net can produce many calls.

| Argument | Type, ordering and missing cases |
| --- | --- |
| `ctx` in `score_maze_step` | Read-only scalar context; `ctx.field` and `ctx["field"]` are equivalent. No arrays, mutable route objects or implicit node-index order are exposed. |
| Field values | Finite numeric scalars. Use the definitions and exact transforms in **Immutable context** above; normalized pressure is dimensionless, not a raw percentage. |
| `die_id` | Exactly 0 for bottom and 1 for top. This is physical tier identity, not a congestion ranking. |
| Missing information | Documented fields are required by the hook. Unexposed timing paths, grids or per-site data cannot be recovered from a missing value or treated as zero. |

Return a finite positive cost multiplier; the trusted clamp and non-finite handling are described above. Do not mutate context or retain state between calls. Feedback definitions
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
