# Improve connections between two stacked chip tiers

## Background

A logic-on-logic (LoL) three-dimensional integrated circuit (3D-IC) places logic
on two stacked tiers. A hybrid bonding terminal (HBT) connects the two parts of
a network that crosses between them. Its position affects wire length on both
tiers, the load seen by the driver, and the paths that later routing must use.
A short geometric connection is useful, but it does not always give the best
timing when many terminals compete for nearby sites.

The design has already been partitioned and legally placed. The assignment
algorithm chooses one legal HBT site for each required crossing. It first orders
the networks, then selects a site for each network from the available candidates.
Occupied sites and nearby escape-track conflicts are removed as assignment proceeds.

## Your task

Improve this assignment by changing two functions in one policy:

- `net_priority(context)` decides which networks are assigned first. Higher
  values go first; equal values are ordered by the original network name.
- `site_cost(net_context, site_context)` ranks the legal candidates for the
  current network. Lower values are better. Ties use distance from the ideal
  position, then x and y coordinates.

The reference gives networks with more pins priority and minimizes combined
wire length when choosing a site. You may change either or both decisions.
Keep one main idea per candidate, such as prioritizing networks by pin count
and geometric span or balancing wire length against displacement. Explain
that idea briefly in a function docstring. Do not
hard-code a particular design or return a table of precomputed assignments.

## How results are compared

The evaluator materializes the legal HBTs, then runs joint two-tier static timing
analysis (STA). It uses placement-estimated wire resistance and capacitance,
the same passive HBT model for every candidate, and ideal clocks before clock
tree synthesis (CTS). No timing optimization (OPT) is performed.

The objective uses setup worst negative slack (WNS) and total negative slack
(TNS), measured in nanoseconds. For each metric let `d = max(0, -slack)`.
Its normalized cost is `d_candidate / d_reference` when the reference deficit
is greater than `1e-12`. If both deficits are at most `1e-12`, the cost is 1;
otherwise a near-zero reference gives cost `1 + d_candidate / 1e-12`.

`score = 1 / max(0.6 * WNS_cost + 0.4 * TNS_cost, 1e-12)`.

Higher is better and the reference scores 1. WNS and TNS may trade off under
these weights. Positive slack earns no additional reward. A candidate must
complete assignment, materialization and timing validation before it can score.
This is early timing analysis, not routed or signoff timing.

Legal-site half-perimeter wirelength (HPWL), displacement and
runtime are diagnostics, not extra rewards. The HBT wirelength diagnostic sums
the bottom pin bounding box extended to the selected terminal and the top pin
bounding box extended to that same terminal. It includes both horizontal
portions of crossing nets at their legal HBT positions; it is not the whole-chip
HPWL or a measure of vertical wire length. Improvement before site legalization
does not establish improvement in the evaluated result.

## Inputs and output

Both functions receive read-only scalar mappings. Coordinates and geometric
lengths use integer database units (DBU); `dbu_per_micron` gives their scale.
`die_span_dbu` is the larger die width or height and can be used to normalize
geometric lengths. The interface does not expose names or mutable design objects.

The network mapping provides:

- `pin_count`, `bottom_pin_count`, `top_pin_count`: counts on the crossing net.
- `bottom_bbox_hpwl_dbu`, `top_bbox_hpwl_dbu`: each tier's pin-only HPWL.
- `ideal_combined_hpwl_dbu`: combined HPWL at the unchanged ideal position.
- `die_span_dbu`, `dbu_per_micron`: geometry scales.

The site mapping provides `combined_hpwl_dbu`, `bottom_hpwl_dbu`,
`top_hpwl_dbu`, `ideal_displacement_dbu`, `x_fraction` and `y_fraction`.
The fractions locate the site's origin within the die outline. Costs are
evaluated only for sites that currently pass the trusted legality rules.
This version exposes geometry, not per-network timing criticality, a routed
congestion map, per-site delay prediction, driver resistance, or a mutable
occupancy map. Use the final timing feedback to compare geometry-based policies;
do not assume that shorter connections always improve timing.

Return one finite number from each function. Network priorities are evaluated
once before assignment; site costs are evaluated for the current network's
candidate sites. Identical inputs must give identical outputs. Changing earlier
assignments can change the legal candidates seen by later networks.

## Constraints

Keep tier assignment, cell and macro positions, external pins, the power grid,
network connectivity, HBT count and geometry, clocks, timing constraints, and
the resistance/capacitance model unchanged. Do not change ideal positions,
site generation, escape-track guards, candidate generation or tie-breaking.
The candidate search starts with eight nearest sites per native geometric seed
and expands only when no legal candidate is found. It does not expand just to
look for a lower policy cost. Clock connectivity is handled by the trusted flow;
this policy neither invents clocks nor builds a clock tree.

Use scalar arithmetic, comparisons, conditional branches, local assignments,
and `abs`, `min`, `max`. The interpreter also supports `math.sqrt`, `math.log`,
`math.log1p`, `math.exp`, `math.expm1`, `math.tanh`, `math.fabs`, `math.floor`,
`math.ceil`, `math.hypot` and `math.pow`, without imports. Unary functions take
one argument; `hypot` and `pow` take two. Guard domains and small denominators. Do not use imports, loops, recursion, helper functions, power
operators, array allocation, persistent state, random numbers, input mutation,
files, processes, network access, or additional STA calls. The complete program
is limited to 16,000 UTF-8 bytes and 500 abstract syntax tree nodes. A candidate
error, non-finite value or incomplete assignment is a failure, not a request to
fall back to the reference or try another solver.

Use comparable feedback to decide which changes help. Missing measurements
remain unknown; policy activity alone is not a timing improvement. Return only
the two complete function definitions inside EVOLVE-BLOCK markers, not a diff.

## Calling the policy

| Argument | Type, meaning and use |
| --- | --- |
| `context` in `net_priority` | Read-only mapping of the network fields listed above. Called once per crossing net before assignment; higher priority is handled earlier. |
| `net_context` in `site_cost` | Same network mapping; independent of candidate site. |
| `site_context` in `site_cost` | Read-only mapping for one currently legal candidate site. Called during site comparison; lower cost is preferred. |
| Geometric fields | Numeric scalars in DBU, except dimensionless x/y fractions. Counts are integers. Preserve the supplied coordinate scale; divide by `dbu_per_micron` only when a micrometer value is needed. |
| Return values | Finite scalar priority/cost. No input may be modified and no state may persist across calls. |

There are no arrays or node-index ordering in this interface. Documented fields
must be present for a valid invocation; missing timing/congestion information is
not supplied as zero. The task remains definition-only until native execution
and reference qualification are connected.

### Feedback field guide

The supplied `metric_definitions` gives a short meaning and unit for each returned
metric. Missing metrics are unknown, not zero. Only `combined_score` is the
selection objective; activity and progress measurements explain behavior but
are not additional rewards.
