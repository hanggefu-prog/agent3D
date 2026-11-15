# Current-source Nesterov reference

The local task `placement_dynamic_parameters_reference_20260923` retains its
existing ID, nine-argument controller, no-BB Nesterov algorithm, legal source
scope and HPWL/overflow score. Its September and October reference profiles
remain available under their original source/runtime/input contracts.

The current profile uses the maintained deterministic GPU accumulation kernels
(`deterministic_flag=1`). Dated profiles retain their audited value of zero and
cannot be executed by silently switching to the current kernels.

For a new deployment, explicitly pass
`AgenticBench/local_bench/reference_configs/nesterov_current.yaml` as the
`base_config` argument of `local_bench.evaluation.evaluate_task`. The host may
change the registered design name; the remaining recipe is fixed. The input is
a verified input/floorplan/TP-GNN partition prefix. This selection is not an
alias for either dated reference family and does not reuse their scores.

First materialize the task's exact initial program and evaluate it with that
base config and `baseline_run`, omitting `reference_run`. Only the exact initial
program can create this fresh reference. Subsequent candidates pass the same
base config and input plus the successful initial evaluation's `run` directory
as `reference_run`. Run both inside the same pinned image with
`OPEN3DBENCH_IMAGE_ID` set to its actual image ID.

Admission binds the task adapter, initial program, optimizer/controller source,
evaluator, full recipe, runtime image and upstream manifest hashes. Adoption
rechecks native placement parameters and active controller traces and extracts
the reference metrics again. A changed source/runtime/recipe requires a new
reference; a different candidate cannot bootstrap its own score.

The endpoint is global placement only, before legalization and backend stages.
This local profile is separate from the paired-seed post-GRT campaign.
