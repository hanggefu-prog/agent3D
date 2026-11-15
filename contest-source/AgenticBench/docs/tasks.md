# Current local tasks

Run `python3 -m local_bench list --core` to inspect the ten maintained task
definitions. All ten have connected native evaluators.
A connected evaluator does not by itself qualify a task/case reference.

| Task directory | What the agent changes | Evaluation boundary |
| --- | --- | --- |
| `tp_gnn_post_placement_timing` | Differentiable TP-GNN training loss | Global 2-way partition, fixed placement and HBT, then setup WNS/TNS without OPT |
| `tp_gnn_binary_loss_timing` | Binary probability-head loss, including regularization | Same no-OPT timing objective; source-connected, fresh case reference required |
| `tp_gnn_loss_decoder_timing` | Legacy embedding loss and global clustering decoder together | Same no-OPT timing objective; source-connected, fresh case reference required |
| `placement_initialization` | Initial placement coordinates | Fixed-partition global placement: HPWL and density overflow |
| `placement_preconditioner` | Gradient preconditioning | Same placement objective and frozen partition |
| `placement_parameters_reference_20260923` | Dynamic Nesterov control | Same placement objective; the dated directory name is still current |
| `hbt_assignment_policy` | Network priority and legal-site cost | Joint no-OPT post-HBT setup timing on a verified input-through-PDN prefix |
| `global_routing_ripup_policy` | Which eligible edges are ripped up | Two-die global-routing wirelength and overflow |
| `global_routing_maze_cost` | Maze-routing edge cost | Same global-routing objective |
| `global_routing_iteration_controller` | Bounded reroute iteration policy | Same global-routing objective |

Each task owns `task.yaml`, `prompt.md`, its initial program, and its program
adapter under `local_tasks/`. The YAML defines the editable scope and evaluation
boundary. The prompt explains the optimization problem, argument meanings and
feedback. The harness injects the initial program at runtime, so it need not be
duplicated in the task prose. The [ask/tell guide](ask_tell.md) describes integration
with OpenEvolve or another search framework.

Only measured, valid `combined_score` values rank candidates. Other metrics are
diagnostics with explicit meanings; missing measurements remain unknown. Failed
or rejected generated candidates consume an evaluation slot. Global-placement
HPWL is not routed wirelength, and GRT is not DRT or signoff.

The [task settings report](TASK_SETTINGS_REPORT.md) gives every task's ID, scope
and exact score formula. The two local timing tasks weight setup deficits
0.6 WNS / 0.4 TNS. The global post-GRT track uses 0.7 WNS / 0.3 TNS. The two
scores are never comparable. The [harness design report](HARNESS_DESIGN_REPORT.md)
describes the evaluation pipeline.

## 5-minute path for external users

This path uses the CPU-only smoke input `ibex_lol_ng45_cpu_smoke` and the
`placement_initialization` task. The input is marked `qualified: false`. It is
for checking an installation, and its scores are not benchmark results. Run
every command from `AgenticBench/` in a source checkout with the submodules
initialized. Use a Python 3.8+ environment that has the repository's
requirements, including PyYAML and PyTorch, and make sure its `python3` comes
first on `PATH`. The dry run calls `python3 -m open3dbench plan`.

Steps 3 to 5 also need a DREAMPlace install. The default is the in-tree build
`Place-3D/build/dreamplace/install`. If DREAMPlace lives elsewhere, for
example `/opt/dreamplace` in the release image, export
`OPEN3D_LOCAL_DREAMPLACE_INSTALL`. Both `prepare-input` and `evaluate` read
this variable, and `prepare-input --dreamplace-install` overrides it for step
3 only. `evaluate` makes an isolated `git clone` of the source tree, so steps
4 and 5 need a git checkout. An exported tree with no `.git` fails with
`isolated clone failed`.

```sh
cd AgenticBench
export PYTHONPATH=.:..:../Place-3D
export OPEN3D_LOCAL_DREAMPLACE_INSTALL=/opt/dreamplace   # only if not in-tree

# 1. List the ten maintained tasks and inspect one contract (no tools needed).
python3 -m local_bench list --core
python3 -m local_bench inspect placement_initialization

# 2. Dry run: verify the initial program and let Open3DBench plan the stages.
#    No physical tool runs. combined_score stays 0 and dry_run is 1.
python3 -m local_bench evaluate placement_initialization \
  --open3dbench-root .. --output-dir /tmp/ibex-smoke-dry \
  --base-config local_bench/benchmark_inputs/ibex_lol_ng45_cpu_smoke/config.yaml \
  --dry-run

# 3. Prepare the frozen input/floorplan/partition prefix on CPU.
python3 -m local_bench prepare-input ibex_lol_ng45_cpu_smoke \
  --open3dbench-root .. --output-dir /tmp/ibex-smoke-input

# 4. Evaluate the native initial program on that prefix (DREAMPlace on CPU).
#    Without --reference-run only the exact initial program is admitted;
#    the result is a new local reference with combined_score 1 by definition.
python3 -m local_bench evaluate placement_initialization \
  --open3dbench-root .. --benchmark-input /tmp/ibex-smoke-input \
  --base-config local_bench/benchmark_inputs/ibex_lol_ng45_cpu_smoke/config.yaml \
  --output-dir /tmp/ibex-smoke-ref

# 5. Score a modified candidate against that reference.
python3 -m local_bench evaluate placement_initialization \
  --candidate my_candidate.py \
  --open3dbench-root .. --benchmark-input /tmp/ibex-smoke-input \
  --base-config local_bench/benchmark_inputs/ibex_lol_ng45_cpu_smoke/config.yaml \
  --reference-run /tmp/ibex-smoke-ref/run --output-dir /tmp/ibex-smoke-cand
```

All five steps were run on 2026-10-19 in the release image
(`sha256:a339ce29…`, 4 CPUs, no GPU, no network), with
`OPEN3D_LOCAL_DREAMPLACE_INSTALL=/opt/dreamplace`. Steps 1 and 2 take
seconds. Step 3 took 335 s, step 4 took 47 s (`combined_score` 1.0), and step
5 took 43 s. Placement is deterministic on this input: a second run whose
candidate edit had no effect (see below) gave a byte-identical
`placement.gp.def`. The candidate in step 5 added a 5 % Gaussian
jitter to the initial x positions and scored `combined_score` 0.995 (HPWL
+0.6 %). An edit to the `random_center_init_flag` branch alone changes nothing,
because this config sets that flag to 0. The whole path takes about 7 minutes,
most of it in step 3. A reference made this way is a local smoke reference only. It
is not a qualified benchmark reference. Multi-case scoring uses a qualified
matrix ([LOCAL_EVALUATION_MATRICES.md](LOCAL_EVALUATION_MATRICES.md)).

## Shared code and separate evaluation profiles

`local_bench/` contains orchestration and validation, not physical algorithms.
See its [module map](../local_bench/README.md) for dependencies shared with global
tasks. Old dates in reference modules identify immutable producer contracts;
they are not obsolete task definitions to delete automatically.

Standalone STA/thermal replays live in `evaluation_profiles/`. They are distinct
from the ten optimization tasks and may retain historical input recipes for
explicit comparison. Global workspace tasks live in `global_tasks/`; see the
[global task guide](global_tasks.md).

The old Mt-KaHyPar configuration task, cut-score/TA3D/pin-cut evolution variants,
combined initialization/preconditioner task, Adam controller and net-ordering
task are retired. Eleven retired-task-only input recipes and the TA3D/pin-cut
scorers have also been removed. Old experiment reports remain historical evidence,
not executable instructions or new reference qualification.


## Two additional partition research definitions

The [binary loss task](../local_tasks/tp_gnn_binary_loss_timing/task.yaml) fixes
the current `tp_gnn_binary_regularized.json` recipe and exposes only
`compute_binary_partition_loss`. Its inputs are probabilities, the trusted
differentiable expected cut, per-net cut probabilities and timing weights,
physical areas, frozen node features and schedule
scalars. Loss expressions, weights expressed in the function, probability
regularizers and internal loss schedules may evolve. The encoder, prediction
head, 200 updates, temperature/noise continuation, checkpoint-selection rule,
0.5 threshold and final area/macro closure are protected. The initial program
reproduces cut + 50 area balance + the scheduled area-weighted binary penalty.
Uncertainty, raw hard-area balance and repair movement are separate diagnostics.

The [legacy loss-and-decoder task](../local_tasks/tp_gnn_loss_decoder_timing/task.yaml)
keeps the random-walk GNN and its timing-context-v2 loss inputs. One atomic
program contains both `compute_tp_gnn_loss` and `decode_tp_gnn_embeddings`.
The second function may change whole-design two-way clustering, returning one
binary label per indivisible group within a fixed iteration budget. The host
recomputes physical-area-weighted centers and native distance affinities for
the unchanged final macro/area closure. Candidate clustering centers cannot
replace that protected repair objective. Raw weighted k-means remains diagnostic.

Both definitions inherit the existing partition endpoint: partition, placement,
PDN and HBT, followed by joint two-die setup STA with placement RC, a passive
HBT model and ideal clocks. No OPT, CTS or routing runs. The only ranking metric
is `1 / max(0.6 * normalized_WNS_deficit + 0.4 * normalized_TNS_deficit, 1e-12)`.
Cut, HPWL, training loss, confidence and runtime are diagnostics. Final top
area share is 47–53%; top macro count/area shares are 45–55% where applicable.
Missing timing fails rather than becoming zero.

The default is BP_FE/ng45/LoL, I/O seed 1, partition seed 1001 and placement
seed 1000. No multi-case or multi-seed matrix is implied. The two baselines use
different training recipes; their score-1 references are distinct. Comparing
scores normalized to those separate references does not rank the architectures;
cross-task comparison needs matched raw measurements and experiment settings.

Task YAML, English prompts, initial programs, adapters and literal candidate
verification are present. The protected `partition.evolution_program` input
transports a source capsule to the native hook; the runner freezes and hashes
it as partition evidence. Native hooks, trusted context producers and physical
collectors are connected in source, and both tasks are executable templates.
Each case still requires a fresh reference from its exact initial program and
fixed recipe before candidate scoring. Without a qualified reference only the
exact initial program may create that reference. Prior loss-only receipts,
another task's references and the binary HPWL diagnostic cannot qualify either
task. Source connectivity and offline tests are not native EDA qualification;
external qualification receipts must establish that separately. Shipped search
templates do not authorize model calls or physical experiments.
