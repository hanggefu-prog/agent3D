# HBT assignment policy

Updated 2026-10-16. Task ID: `hbt_assignment_policy`.

The task has a connected native evaluator. It accepts one complete program with
two scalar functions and evaluates it on an immutable physical input through
PDN. Each input needs its own initial-program reference; another task's HBT
checkpoint or a post-GRT score is not a scoring reference for this task.

## Scope and physical boundary

In a two-die logic-on-logic design, hybrid bonding terminals connect cross-die
nets. The task changes network priority and legal-site cost together:

| Function | Decision | Initial program |
| --- | --- | --- |
| `net_priority(context)` | Assign higher-priority nets first | `pin_count` |
| `site_cost(net_context, site_context)` | Prefer lower-cost legal sites | `combined_hpwl_dbu` |

The trusted planner owns the optimal-region ideal, lattice, candidate search,
site occupancy and escape guards. Search begins with eight nearest sites per
native seed point and expands only when no legal candidate exists. SciPy is
required; no exhaustive fallback is admitted. Ties use canonical net names
and then ideal displacement, x and y. A failed policy is never replaced by the
reference policy or repaired by another solver.

Only geometric inputs are exposed: per-die pin counts and bounding boxes,
combined HPWL, ideal displacement, normalized site coordinates and DBU scale.
Combined HPWL is the sum of the two die-local pin-plus-HBT bounding boxes. It
is neither whole-chip wirelength nor vertical wirelength. The interface does
not claim to expose timing criticality, driver RC or congestion predictions.

The program permits scalar arithmetic, comparisons, assignments, conditionals
and `abs/min/max`. Imports, loops, recursion, external I/O, mutable inputs and
global state are forbidden. Limits are 16,000 source bytes and 500 AST nodes.
The native interpreter also enforces finite arithmetic and an operation budget.

The evaluator copies `00_input` through `40_pdn` without rebasing or resealing
them. It checks artifact hashes, the complete upstream manifest chain,
global-2way partition provenance and the design/platform identity. It executes
only HBT assignment/materialization and independent joint STA:

```text
fixed input -> 50_hbt -> 55_post_placement_sta
```

No OPT, CTS, GRT, DRT or thermal stage is executed. Timing uses the unified
two-die circuit, placement-estimated RC, passive HBT pi models and ideal pre-CTS
clocks. It is not routed or signoff timing. Physical execution uses four CPU
threads with a maximum six-hour candidate budget; it requires no GPU.

## Score and independent verification

For setup WNS and TNS, use deficit `max(0, -slack)` in ns. Each cost is the
candidate deficit divided by the matching reference deficit. If the reference
deficit is at most `1e-12 ns`, the cost is 1 when both deficits are at most that
threshold, otherwise `1 + candidate_deficit / 1e-12`.

```text
combined_score = 1 / max(0.6 * WNS_cost + 0.4 * TNS_cost, 1e-12)
```

Higher is better. The native reference scores 1. Geometry and runtime are
diagnostics, not additional rewards. This local score is separate from the
three-objective post-GRT task and its 0.7/0.3 timing weights.

The independent reader checks complete one-to-one net coverage, integer DBU,
grid/body/blockage legality, unique sites and escape spacing. It recomputes
ideals, HPWL and displacement from the problem and assignment, checks the
materialized HBT coordinates, verifies the exact executed policy and real hook
activity, and audits the physical handoff. Hash-bound native STA `metrics.tsv`
must agree with both timing summaries. Missing clocks, unconstrained endpoints,
missing or nonfinite values, later flow stages and changed upstream inputs fail.

Only the exact initial program can create a fresh scoring reference. Candidate
adoption requires the same task, physical implementation, platform collateral,
installed OpenROAD binary, pinned image and unchanged five-stage input. Reference
metrics are re-extracted from physical evidence. Updating source or runtime
does not silently promote old reference qualifications.

## Running and reading results

Use the standard `local_bench evaluate` or ask/tell interface in the pinned EDA
container. Supply the committed source, matching case configuration and frozen
PDN input. Evaluate the initial program without `--reference-run` first, then
pass that result's `run` directory as `--reference-run` for candidates. All
evaluations need `--baseline-run`; arbitrary candidates cannot bootstrap their
own reference. Temporary work and outputs stay outside the source checkout.

Each result contains `candidate_snapshot.json`, the resolved configuration,
`evaluation_result.json`, HBT logs and assignment files, and the joint STA
reports. The reference receipt binds implementation hashes. The model receives
filtered WNS/TNS, normalized costs, score and measured geometric diagnostics;
raw receipts remain available to the host. Verify-only synthetic scores never
qualify as physical results.

The native qualification record is in
`releases/local-compat-20261016/`; it distinguishes the tested physical inputs
from unqualified inputs and records the real changed-policy probe. No model
calls are needed for this qualification.

Relevant implementation:

- [Task and scope](../local_tasks/hbt_assignment_policy/task.yaml)
- [Prompt](../local_tasks/hbt_assignment_policy/prompt.md)
- [Initial program](../local_tasks/hbt_assignment_policy/initial_program.py)
- [Static grammar](../local_bench/hbt_policy.py)
- [Physical verifier](../local_bench/hbt_native.py)
