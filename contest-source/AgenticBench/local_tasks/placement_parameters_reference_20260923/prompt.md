# Task: improve Nesterov step control

## Background

In a logic-on-logic (LoL) three-dimensional integrated circuit (3D-IC), logic
is distributed across two stacked dies. Placement determines cell locations
within each die and the geometry of connections between them. Short connections
help, but cells also need enough space to avoid concentrated density pressure.
This task uses analytical global placement with a fixed die assignment: a
smooth wirelength objective and density penalties guide continuous cell
coordinates through Nesterov optimization.

## Your task

Implement `adjust_learning_rate` to choose the size of the gradient move at
each optimizer step, using wirelength progress, density overflow and gradient
information. It returns a shared multiplier and relative bottom/top multipliers;
it does not choose absolute learning rates. Explore, for example, a bounded
response to stalled progress or a gradual balance between the dies. A larger
step may speed progress but can also create instability or extra backtracking.

## How solutions are evaluated

The score uses the final complete global-placement metric row, not an early
iteration or a later legalization result. Let H be weighted half-perimeter
wirelength (HPWL) and O the larger of the bottom and top density overflows.

`combined_score = 0.9 * R(H) + 0.1 * R(O)`

For a positive reference, `R(x) = x_ref / max(x_candidate, 1e-12)`; for a
zero reference, R is 1 if the candidate is also zero and 0 otherwise. Ratios
are not upper-clipped. Higher is better; the weighted sum allows trade-offs,
so an improved score does not imply both terms improved.

Parsing, scope checks, successful global placement, complete outputs and finite
native measurements are required. The partition, objective, density settings,
optimizer acceptance rules, stopping criteria, random seed and iteration budget
are not editable. Global placement does not certify legalized placement, routed
timing, routing quality or thermal behavior.

The placement budget is 4,000 iterations with 512 density bins per axis,
target density 0.6, initial density weight 0.00003 and stop overflow 0.01.
Cross-die net weight is 1, macro halos are 10 micrometers per side, and the
initial learning-rate parameter is 0.01. These settings are not search variables.

## Using previous results

Check validity first, then compare `combined_score`, `hpwl_score` and
`overflow_score`. Use `placement_native_hpwl`, `placement_overflow_max`,
`placement_overflow_bottom` and `placement_overflow_top` to see what changed
physically and whether one die accounts for the trade-off.
`placement_max_density` and its per-die values are additional observations,
not score terms. `target_density` is shown only when measured metadata supplies
it. Missing metrics remain unknown, never zero.

Try one main change at a time. Keep useful parts of a candidate and combine
ideas when comparable results support them. A rejected program provides
implementation feedback, not evidence that its placement strategy is poor.
Do not infer timing or routability improvements from placement metrics alone.

The current controller diagnostics report invocation count, first/last step,
stage segments, whether all controls were identity, and initial/final/min/max
values of the shared and per-die multipliers. `nesterov_native_step_hint_*`
records the native proposal before backtracking, not the accepted step length
or an Adam learning rate. Changing multipliers demonstrates activity, not a
better final placement.

## Function interface and observations

```python
def adjust_learning_rate(native_step_size, step_num, log_hpwl, log_hpwl_prev,
                         overflow_by_die, log_lambda_by_die, step_multiplier_prev,
                         log_gradient_norm_by_die, die_multipliers_prev):
    return 1.0, (1.0, 1.0)
```

Despite the legacy entrypoint name, the return value is **not an absolute
learning rate**. Return `(shared_step_multiplier, (bottom_relative, top_relative))`.
The first argument is the native positive `alpha_k` hint before the native
backtracking loop; it is not the configuration's 0.01 initialization parameter.
The controller is called once per optimizer step, not once per trial. Its
returned factors remain fixed during all native backtracking trials for that step.

The trusted hook applies the shared factor and region factor to the **gradient
term before acceleration**, i.e. `u_next = v - native_step * scale * gradient`;
the native acceleration remains `v_next = u_next + coef * (u_next - u_previous)`.
It does not scale momentum or move positions after the step. The native
constraint projection and all Nesterov history updates remain in their original
order. Each trial uses the current native alpha; the native alpha estimate,
acceptance rule and maximum ten backtracking trials are unchanged. No trial is
skipped or added by the controller. Native fixed
and stopped-region handling is unchanged. Fixed physical-node coordinates are
not included in the die multiplier map.

The shared factor must be finite and in `[0.0001, 100]`, otherwise the candidate
fails. Die factors must both be finite and positive. Their ratio is projected
to `[0.25, 4]`, their geometric mean normalized to one, and their logarithms
smoothed with weight 0.2 against the preceding effective die factors. These are
trusted interface rules, not evolvable code. `(1, (1, 1))` from the initial state
returns the exact native arithmetic path. Controller invocation evidence is
required. Barzilai–Borwein (BB) step estimation is not enabled; keep the supplied
no-BB Nesterov method.

Observations describe the current native reference point `v_k`: log HPWL,
previous log HPWL, bottom/top overflow, log density weight, and log norm of the
preconditioned gradient actually used by native no-BB Nesterov. First-step previous values are
current HPWL and unit multipliers. Control history resets with each native
global-placement stage. Use only scalar arithmetic, `math` operations, the
provided two-element tuples and bounded iteration. No file/network/process
effects, reseeding, imports, alternate solvers or objective changes are allowed.


## Implementation requirements

Return only the complete `adjust_learning_rate` function inside EVOLVE-BLOCK
markers, with a short comment explaining the change. Keep the exact signature.
Local scalar helpers may be defined inside the function with plain arguments
and constant defaults. Call them directly; no recursion, aliases, decorators or
further nested definitions. No imports, extra top-level definitions, JSON,
diffs, tests or surrounding prose.
Direct calls include `abs`, `bool`, `enumerate`, `float`, `int`, `len`, `max`,
`min`, `range`, `round`, `sum`, `zip`, your helpers and `math` operations. Loops and comprehensions may
iterate over the supplied two-die tuples, literal tuples/lists of at most eight
items, or `range` with literal integer arguments between zero and eight;
`enumerate` and `zip` may wrap these bounded iterators. No while loops.
The program limit is 32,000 characters; the source-edit allowance is at most
800 added and 400 deleted lines across the two permitted files. Keep arithmetic
finite and guard divisions and logarithms. Do not replace the optimizer,
change the fixed partition or modify the placement budget.

## Input reference

The function is called once at the native reference point before backtracking.
All inputs are read-only Python numbers or two-element tuples, ordered
`(bottom, top)`. No argument is optional in the maintained controller.

| Argument | Meaning, units and edge cases |
| --- | --- |
| `native_step_size` | Positive finite float: native alpha before backtracking, in optimizer scale, not an absolute physical displacement. |
| `step_num` | Nonnegative integer native iteration index, starting at zero; may reset at a new placement stage. |
| `log_hpwl`, `log_hpwl_prev` | Natural logarithms of current and previous native HPWL, floored at `1e-30` before taking the log. On the first call, previous equals current. |
| `overflow_by_die` | Two finite dimensionless density overflows; these are not routing overflow counts. |
| `log_lambda_by_die` | Natural logs of the two native density weights, each floored at `1e-30`. |
| `step_multiplier_prev` | Previous effective shared multiplier, dimensionless; initially 1. |
| `log_gradient_norm_by_die` | Natural log of each die's L2 norm of the preconditioned gradient used by the optimizer, floored at `1e-30`. Includes movable and filler coordinates, not fixed physical nodes. |
| `die_multipliers_prev` | Previous effective die multipliers after projection and smoothing, dimensionless; initially `(1,1)`. |

Return the shared multiplier and the two relative die multipliers, as described
above. Inputs describe current state; returned factors affect the next gradient
move, not previously computed momentum. Log values can be negative and are
not normalized to [0,1].

### Feedback field guide

The supplied `metric_definitions` gives a short meaning and unit for each returned
metric. Missing metrics are unknown, not zero. Only `combined_score` is the
selection objective; activity and progress measurements explain behavior but
are not additional rewards.
# Reference selection

The task ID is retained for compatibility. The host selects the reference
profile explicitly. A current-source profile uses a newly qualified initial
program on the same frozen partition, source and runtime as its candidates;
dated profiles keep their original reference contracts. Scores from different
profiles are not comparable.
