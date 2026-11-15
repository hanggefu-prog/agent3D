# AgenticBench task settings report

Checked against the code on 2026-10-30. This report describes the ten
maintained local tasks as the code defines them: identity, edit scope,
evaluation boundary and the exact score each one optimizes. It starts no
experiment and qualifies no reference. A task listed here has a connected
native evaluator, which is not the same as a qualified reference for a given
case.

**Language.** The maintained report is written in English, as are the task
prompts, `task.yaml` files and the other guides. The earlier report was mostly
in Chinese and is archived byte-identical at
[archive/TASK_SETTINGS_REPORT_zh_20261004.md](archive/TASK_SETTINGS_REPORT_zh_20261004.md).
See [archive/README.md](archive/README.md) for why and for what it still
records.

[`scripts/check_doc_consistency.py`](../scripts/check_doc_consistency.py)
checks the task IDs, weights and formulas below against the code. Run it after
editing this file.

## Where a task is defined

A local task is the directory `local_tasks/<directory>/`. It holds the
following files.

- **`task.yaml`.** The contract: ID, kind, base config, program entrypoint,
  execution stages, editable scope, patch limits, value constraints and prompt
  context budget. `local_bench.model.load_task` parses it into a `TaskSpec`.
- **`prompt.md`.** The optimization problem in natural English.
- **`config.yaml`.** The search configuration, including
  `prompt.system_message`.
- **`initial_program.py` and `program_adapter.py`.** The model returns one
  complete function or class. The trusted adapter turns it into a source patch
  or a policy file, so the model never writes a patch directly.

The model does not see `prompt.md` alone. The system prompt is the shared
constraints in [`local_bench/prompting.py`](../local_bench/prompting.py) plus
the task's `prompt.system_message`. The user context adds the task contract,
read-only source excerpts limited to `prompt_context.source_ranges`, the
initial program and filtered feedback from earlier candidates. Being allowed
to read a source path does not mean being allowed to edit it.

The task inventory is `local_bench.task_inventory.CORE_TASK_DIRECTORIES`.
`RETIRED_TASK_DIRECTORIES` in the same module lists eleven removed directories.
Those names exist only to stop the directories from being reintroduced; they
are not aliases.

## Inventory

| Directory | Task ID | Kind | Entrypoint | Stages run (upstream reused) |
| --- | --- | --- | --- | --- |
| `tp_gnn_post_placement_timing` | `tp_gnn_post_placement_timing` | `tp_gnn_post_placement_timing` | `compute_tp_gnn_loss` | partition through hbt (input, floorplan) |
| `tp_gnn_binary_loss_timing` | `tp_gnn_binary_loss_timing` | `tp_gnn_binary_loss_timing` | `compute_binary_partition_loss` | partition through opt (input, floorplan) |
| `tp_gnn_loss_decoder_timing` | `tp_gnn_loss_decoder_timing` | `tp_gnn_loss_decoder_timing` | `compute_tp_gnn_loss` | partition through opt (input, floorplan) |
| `placement_initialization` | `placement_initialization` | `placement_initialization_source` | `initialize_position` | placement (input, floorplan, partition) |
| `placement_preconditioner` | `placement_preconditioner` | `placement_preconditioner_source` | `PreconditionOp` | placement (input, floorplan, partition) |
| `placement_parameters_reference_20260923` | `placement_dynamic_parameters_reference_20260923` | `placement_dynamic_parameters` | `adjust_learning_rate` | placement (input, floorplan, partition) |
| `hbt_assignment_policy` | `hbt_assignment_policy` | `hbt_assignment_policy` | `net_priority` | hbt (input, floorplan, partition, placement, pdn) |
| `global_routing_ripup_policy` | `global_routing_ripup_policy_source` | `routing_ripup_policy_source` | `score_ripup_edge` | global_routing (input through cts) |
| `global_routing_maze_cost` | `global_routing_maze_cost_source` | `routing_maze_cost_source` | `score_maze_step` | global_routing (input through cts) |
| `global_routing_iteration_controller` | `global_routing_iteration_controller_source` | `routing_iteration_controller_source` | `control_reroute_iteration` | global_routing (input through cts) |

"Input through cts" means the stages input, floorplan, partition, placement,
pdn, hbt, opt and cts.

The directory name and the task ID differ for the Nesterov task and the three
routing tasks. Matrices, `list --core` and result records use the task ID. The
CLI also accepts the directory name. The date in
`placement_parameters_reference_20260923` names an immutable producer
contract. The task is still current. See
[current_nesterov_profile.md](current_nesterov_profile.md) for its
current-source input profile.

## Edit scope

Every limit below comes from `candidate_policy` in the task's `task.yaml`.

| Task ID | Editable source paths | Config / candidate files | Max files / added / deleted lines |
| --- | --- | --- | --- |
| `tp_gnn_post_placement_timing` | `Place-3D/src/partition/lol/methods/tp_gnn.py` | `partition.algorithm` | 1 / 360 / 12 |
| `tp_gnn_binary_loss_timing` | none | `binary_loss_program.json` as `partition.evolution_program` | 0 / 0 / 0 |
| `tp_gnn_loss_decoder_timing` | none | `loss_decoder_program.json` as `partition.evolution_program` | 0 / 0 / 0 |
| `placement_initialization` | `Place-3D/thirdparty/DREAMPlace/dreamplace/BasicPlace.py` | `placement.target_density`; `placement.json` as `placement.config` | 1 / 800 / 400 |
| `placement_preconditioner` | `Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py` | same as above | 1 / 800 / 400 |
| `placement_dynamic_parameters_reference_20260923` | `.../dreamplace/NesterovAcceleratedGradientOptimizer.py`, `.../dreamplace/NonLinearPlace.py` | same as above | 2 / 800 / 400 |
| `hbt_assignment_policy` | none | `hbt_policy.json` as `hbt.policy` | 0 / 0 / 0 |
| `global_routing_ripup_policy_source` | `.../fastroute/src/RipUp.cpp` plus the two evaluator wrappers | none | 3 / 1800 / 24 |
| `global_routing_maze_cost_source` | `.../fastroute/src/maze.cpp` plus the two evaluator wrappers | none | 3 / 1800 / 24 |
| `global_routing_iteration_controller_source` | `.../fastroute/src/FastRoute.cpp` plus the two evaluator wrappers | none | 3 / 1800 / 24 |

FastRoute sources live under `OpenROAD-3D/openroad/src/grt/src/fastroute/src/`.
The two evaluator wrappers are `open3dbench/backend/three_d_route.py` and
`open3dbench/runner.py`.

The placement tasks also pin every value they expose through
`value_constraints`. Examples are `iterations` 4000, `num_bins` 512,
`target_density` 0.6, `stop_overflow` 0.01, `density_weight` 0.00003 and
`learning_rate` 0.01. The JSON file can restate those values but cannot change
them.

The routing patch limit of 1800 added lines covers the trusted C++ translation
and the wrapper code. It is not an invitation to write long candidate
functions. The adapter parses the candidate's restricted Python, translates it
to C++ and inserts it at a fixed native call site. Three examples of the
translated contract:

- **Rip-up.** The edge is ripped up when `score_ripup_edge >= 0.5`.
- **Maze cost.** The step multiplier is clamped to `[0.05, 20]`.
- **Iteration controller.** It returns bounded congestion/search scales and a
  rip-up offset.

## Scores

`combined_score` is the only ranking value. Larger is better. Every other
metric is a diagnostic. In the formulas below `R_ref` is the reference run
named by the matrix cell or `--reference-run`. `1e-12` is
`metrics.RATIO_DENOMINATOR_FLOOR`, a finite-value guard and not a score cap.

### TP-GNN post-placement timing

Extractor: `post_placement_timing.timing_metrics`, boundary
`post-placement-hbt-no-opt`. The run executes the global 2-way partition, then
placement, PDN and HBT, then joint two-die setup STA. **No OPT** runs. With
`dW = max(0, -WNS)` and `dT = max(0, -TNS)`:

```text
cost_W = rel(dW, dW_ref)      cost_T = rel(dT, dT_ref)
combined_score = 1 / max(0.6 * cost_W + 0.4 * cost_T, 1e-12)
```

`rel(v, r)` is `metrics._relative_cost`:

- `v / r` when `r > 1e-12`;
- otherwise `1` when `v <= 1e-12`;
- otherwise `1 + v / 1e-12`.

Without a reference both costs are 1. The candidate must also pass the
partition balance gates of `tp_gnn_learning_objective_metrics`. When a
reference is given, it must match the candidate's HBT model and partition
recipe. Results report `sta_wns_weight = 0.6` and `sta_tns_weight = 0.4`.

### Placement initialization and placement preconditioner

Extractor: `metrics.placement_component_metrics`, boundary
`dreamplace-global-placement`. The extractor reads the last complete native
DREAMPlace global-placement row. HPWL is DREAMPlace weighted HPWL, not routed
wirelength. `ovf` is the larger of the two per-die density overflows.

```text
hpwl_score     = HPWL_ref / max(HPWL, 1e-12)
overflow_score = max(ovf_ref, 0.01) / max(ovf, 0.01)
combined_score = 0.9 * hpwl_score + 0.1 * overflow_score
```

The overflow reward saturates at the stop tolerance 0.01, so an overflow near
zero cannot dominate the score. Without a reference both component scores are
1. In each task the other component stays native, and neither task inherits
the other's best candidate.

### Dynamic Nesterov control

Extractor: `metrics.placement_nesterov_metrics`. It is `placement_metrics`
plus Nesterov step-control diagnostics. Results must show that the
`NesterovStepControl` hook ran.

```text
hpwl_score     = HPWL_ref / max(HPWL, 1e-12)
overflow_score = ovf_ref  / max(ovf, 1e-12)
combined_score = 0.9 * hpwl_score + 0.1 * overflow_score
```

The weights are `PLACEMENT_HPWL_WEIGHT` and `PLACEMENT_OVERFLOW_WEIGHT`.
Ratios are not clipped. A zero reference metric scores 1 only when the
candidate value is also zero, and 0 otherwise. Without a reference the
unqualified proxies are `1 / (1 + HPWL/1e6)` and `1 / (1 + ovf)`.

### HBT assignment policy

Extractor: `hbt_native.timing_metrics`, which calls
`hbt_policy.score_timing`. The run reuses a verified input-through-PDN prefix,
applies the HBT policy and runs joint setup STA. No OPT runs.

```text
combined_score = 1 / max(0.6 * rel(dW, dW_ref) + 0.4 * rel(dT, dT_ref), 1e-12)
```

`rel` is the same function as for TP-GNN. Setup TNS must be non-positive. See
[HBT_ASSIGNMENT_TASK.md](HBT_ASSIGNMENT_TASK.md) for the policy file contract.

### Global-routing source tasks (score version 1, default)

Extractor: `metrics._routing_source_policy_metrics`. `W` is the sum of the
FastRoute `GRT-0018` wirelength over both dies. `O` is the sum of final GRT
overflow over both dies. Both dies must finish. `grt_complete` is 1 only when
at least one die logged hook diagnostics.

```text
overflow_cost   = (O + 1000) / (O_ref + 1000)
wirelength_cost = W / W_ref
combined_score  = grt_complete / max(0.1 * overflow_cost + 0.9 * wirelength_cost, 1e-12)
```

The weights are `ROUTING_ORDER_OVERFLOW_WEIGHT` (0.1) and
`ROUTING_ORDER_WIRELENGTH_WEIGHT` (0.9). Without a reference the bootstrap
costs are `1 + O/1000` and `1 + W/1e6`. If the reroute loop never reaches the
hook, a successful GRT run does not count as hook evidence.

## Two timing-weight conventions

The repository uses two different WNS/TNS weightings on purpose. They score
different endpoints and must not be mixed.

| Track | Weights | Code | Endpoint |
| --- | --- | --- | --- |
| Local TP-GNN and local HBT | 0.6 WNS / 0.4 TNS | `metrics.STA_WNS_WEIGHT`, `metrics.STA_TNS_WEIGHT`, `post_placement_timing.timing_metrics`, `hbt_policy.score_timing` | Post-placement/HBT setup STA without OPT, as one scalar `combined_score` |
| Global post-GRT evolution | 0.7 WNS / 0.3 TNS | `grt_bench.contract.DEFAULT_SCORING["timing_weights"]` | Post-GRT timing objective in a multi-objective Pareto search. Routing overflow and peak temperature are separate objectives. Mean power must stay within 110% of the paired baseline (`power_max_ratio` 1.10) |

Why the local weights were kept:

- They are part of the existing local task IDs, prompts
  (`local_tasks/tp_gnn_post_placement_timing/prompt.md` and
  `local_tasks/hbt_assignment_policy/prompt.md`) and qualified references.
- Changing them would change every recorded score.
- A 0.7/0.3 local variant would need a new task ID or score version, with its
  own references.

The post-GRT weights belong to the global track, which another work package
maintains. A score from one convention is never comparable with a score from
the other.

## Routing score version 2 (opt-in)

[`local_bench/routing_scoring.py`](../local_bench/routing_scoring.py) adds an
optional second score contract for the three routing source tasks. Version 1
stays the default. No checked-in `task.yaml` declares a `scoring` block, so
the existing task contracts, prompts, scores and references are unchanged.

A new task version opts in with:

```yaml
scoring:
  version: 2
  overflow_gate:
    max_ratio: 1.0   # optional, default 1.0
    abs_slack: 0     # required, in total-overflow units
```

Version 2 replaces the smooth overflow blend with a hard feasibility gate:

```text
gate passes iff O <= max_ratio * O_ref + abs_slack
combined_score = grt_complete * W_ref / W   when the gate passes
combined_score = 0                          when the gate fails
```

Without a reference the gate cannot be evaluated. In that case
`overflow_gate_evaluated = 0` and the score is the unqualified bootstrap proxy
`grt_complete / (1 + W/1e6)`. Version-2 feedback exposes the gate result,
limit and reference values (`ROUTING_V2_FEEDBACK_KEYS`). Version-2 results
need their own references. Version-1 references do not carry over.

## Routing evaluator-wrapper byte identity

The routing tasks list `open3dbench/backend/three_d_route.py` and
`open3dbench/runner.py` as editable for one reason only: the trusted adapter
must pass die/HBT context to the FastRoute hook. Those files are not search
space. A raw `candidate.json` can bypass the adapter, for example through the
CLI, the search API, an OpenEvolve program or a matrix cell. To close that
path, [`local_bench/routing_wrapper_integrity.py`](../local_bench/routing_wrapper_integrity.py)
works as follows.

1. It re-derives the adapter's wrapper diff from the same Open3DBench checkout.
2. It requires every wrapper section in the candidate patch to be
   byte-identical to that diff.

A candidate may omit a wrapper section, in which case that wrapper stays
native. Any other wrapper content is rejected.

Limitation: the verifier runs this check only when it knows the Open3DBench
root. These callers pass the root:

- the evaluation pipeline (`evaluation.py`);
- the CLI matrix pre-check in `cli.py`;
- the search API;
- the OpenEvolve adapter.

These callers do not, so they skip this check while the other scope checks
still apply:

- `scripts/build_routing_candidate_image.py`;
- `scripts/qualify_archive_grt.py`;
- `scripts/check_task_sources.py`;
- the matrix initial-candidate check.

## Matrix profile

[`local_bench/reference_configs/default_matrix_profile.yaml`](../local_bench/reference_configs/default_matrix_profile.yaml)
declares the default local evaluation matrix in one place:

- cases `bp_fe` and `ariane133` on ng45;
- seeds `s1` to `s3`, with I/O 1, partition and placement `100n` for
  n = 1001..1003, and routing -1;
- worker counts for each task, with 16 routing threads and a per-case routing
  `resource_reduction` override.

The profile is derived from the
`releases/local-matrix-evolution-20261018` scripts. It says
`qualified: false`: it is a template, not a reference. `render-matrix` turns
it into a reference-free matrix. Qualified references still come from
`qualify-matrix`. See [LOCAL_EVALUATION_MATRICES.md](LOCAL_EVALUATION_MATRICES.md).

## CPU smoke input

`local_bench/benchmark_inputs/ibex_lol_ng45_cpu_smoke` is a small,
`qualified: false` input profile for first-time users. It runs `ibex` on ng45
with TP-GNN on CPU (`partition_tp_gnn_cpu.json`) and GPU placement turned
off. It is compatible only with `placement_initialization`. Apart from the
design name and the CPU switches, its config is the same as the task base
config. A test enforces this. See the
[5-minute path](tasks.md#5-minute-path-for-external-users) for commands. This
input has not been run beyond a dry-run plan, and its results are not
comparable with any reference.

## Global tasks

Global workspace tasks live in `global_tasks/`, and the current post-GRT
evolution lives in `grt_bench/`. Another work package maintains both. They
are described in [global_tasks.md](global_tasks.md),
[POST_GRT_OPTIMIZATION.md](POST_GRT_OPTIMIZATION.md) and
[GLOBAL_TIMING_THERMAL_20261008.md](GLOBAL_TIMING_THERMAL_20261008.md). This
report covers them only for the timing-weight convention above.

## History

- Dated revision notes from 2026-10-04 to 2026-10-16 are in the
  [archived report](archive/TASK_SETTINGS_REPORT_zh_20261004.md). They cover
  the move to the global 2-way decoder, the replacement of net ordering by
  HBT, the split of the combined placement task, and the retired-task
  inventory.
- Their links and qualification statements are historical. They are not
  current admission rules.
- Further detail is in [TASK_REVISION_20261006.md](TASK_REVISION_20261006.md)
  and [PARTITION_GLOBAL_2WAY_TIMING.md](PARTITION_GLOBAL_2WAY_TIMING.md).


## Additional GNN loss and decoder tasks

On the contest-2 branch, these two partition tasks use the one-pass OPT timing
endpoint and the 0.4 WNS / 0.6 TNS deficit-improvement score. The binary task evolves a scalar loss over
probabilities, trusted differentiable expected and per-net cuts, frozen timing
weights, physical areas and features. Its encoder/head, 200 updates,
temperature/noise continuation, checkpoint rule, threshold and closure remain
fixed. The legacy task evolves both the timing-context-v2 loss and a global
two-way decoder; the host validates group labels and recomputes protected
area-weighted centers/closure affinities. Their exact interfaces, language
limits and source-connected status are in [the task guide](tasks.md#two-additional-partition-research-definitions).

Each task needs its own exact-initial-program FRESH reference for the selected
case, source and installed runtime. Existing partition/HPWL or other-task
receipts cannot qualify it. The optional
[`run_gnn_local_evolution.py`](../scripts/run_gnn_local_evolution.py) launcher
prepares shared frontend/timing inputs, verifies a fresh task reference, then
permits exactly five single-request generated slots with no implicit retry.
Credentials remain outside all Docker-mounted trees and durable audit records.
The script is a portable, explicitly invoked experiment launcher; shipping it
does not authorize a run or a model request.
