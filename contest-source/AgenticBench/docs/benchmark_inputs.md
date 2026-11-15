# Unified benchmark inputs

Placement evolution uses one checked-in input profile:
`bp_fe_lol_ng45_tp_gnn_placement`. It fixes the benchmark definition and the upstream
physical state before any candidate is evaluated.

## Current flat-producer qualification

The current Open3DBench flow publishes `tp-gnn-flat`, including explicit native
evidence that synthesized hierarchy is not consumed as a feature. This is not
interchangeable with the historical `tp-gnn` identity. New BP_FE placement runs
must explicitly select `bp_fe_lol_ng45_tp_gnn_flat_placement`; the historical
profile and its hashes remain unchanged. The new profile fixes the same physical
settings, seeds and global-2way decoder, and additionally requires the real
`15_timing_extract` producer. Its `partition_contract` binds the expected method
and decoder. Qualification checks the stage identity, method manifest and native
report together; the report must match the copy embedded in the hash-published
method manifest. Renaming a result or accepting an arbitrary alias is forbidden.

The task-smoke interfaces already support this explicit selection:

```bash
python3 AgenticBench/scripts/prepare_task_smoke_input.py \
  --source "$SOURCE" --batch-root "$RUNS" --image "$IMAGE" --gpu "$GPU" \
  --profile bp_fe_lol_ng45_tp_gnn_flat_placement \
  --output "$RUNS/task-smoke-v2/inputs/bp_fe-flat-placement/bundle"
python3 AgenticBench/scripts/run_local_task_smoke.py \
  --source "$SOURCE" --batch-root "$RUNS" --image "$IMAGE" --gpu "$GPU" \
  --task placement_parameters --rounds 5 --model gpt-5.6-sol \
  --benchmark-input "$RUNS/task-smoke-v2/inputs/bp_fe-flat-placement/bundle" \
  --output "$RUNS/task-smoke-v2/local/placement_parameters"
```

Use `placement_init_preconditioner` and a separate output directory for the
source task. Both consume the selected bundle's configuration, not the task's
historical default. The run receipt records the profile, configuration hash and
actual producer method. Preparation creates a new bundle; it never relabels or
overwrites an earlier failed attempt. Unit-level interface qualification is not
evidence of completed physical evaluations.

## Fixed contract

The profile fixes:

- design `bp_fe`, integration `lol`, and the maintained 10-layer `ng45` PDK;
- RTL/default input preparation and deterministic floorplanning policy;
- TP-GNN global-2way partitioning;
- the exact normalized Open3DBench configuration used by both placement tasks;
- successful `00_input`, `10_floorplan`, and `20_partition` stage manifests;
- the size and SHA-256 digest of every artifact published by those stages;
- the Python/PyYAML environment, DREAMPlace install, Open3DBench and benchmark
  harness revision/dirty state, generation command, and runtime used to create
  the bundle.

The generated bundle is data, not source code, and lives outside both source
repositories. Its portable layout is:

```text
bp_fe_lol_ng45_tp_gnn_placement/
  benchmark_input.json
  prepare.log
  run/
    config.yaml
    run.json
    00_input/
    10_floorplan/
    20_partition/
```

Paths to the frozen stages are relative to the bundle. A bundle can therefore
be copied to another server and revalidated there. Paths in the environment
provenance are descriptive and are not used to locate the copied artifacts.

## Prepare once on a server

Use the same Open3DBench Python environment that will run evaluations. The
command validates Python and DREAMPlace before launching Open3DBench:

```bash
openagentic3dbench-task prepare-input bp_fe_lol_ng45_tp_gnn_placement \
  --open3dbench-root /data/gcr/Open3DBench-V2.0 \
  --python /data/gcr/Open3DBench-V2.0/.venv/bin/python \
  --dreamplace-install /data/gcr/Open3DBench-V2.0/Place-3D/build/dreamplace/install \
  --output-dir /data/gcr/openagentic3dbench-inputs/bp_fe_lol_ng45_tp_gnn_placement
```

If generation was interrupted, rerun with `--resume`. If a completed manifest
already exists, `--resume` performs a full integrity check and does not rerun
the stages.

When a validated Open3DBench run already provides the same input and floorplan,
reuse those two expensive stages while regenerating the required Mt-KaHyPar
partition:

```bash
openagentic3dbench-task prepare-input bp_fe_lol_ng45_tp_gnn_placement \
  --open3dbench-root /data/gcr/Open3DBench-V2.0 \
  --seed-run /data/gcr/baselines/bp_fe \
  --output-dir /data/gcr/openagentic3dbench-inputs/bp_fe_lol_ng45_tp_gnn_placement
```

The seed manifests and all their artifacts are verified before copying. The
new bundle runs and validates `20_partition` itself; a seed partition is never
adopted implicitly.

Validate before every experiment or after copying the bundle:

```bash
openagentic3dbench-task validate-input \
  /data/gcr/openagentic3dbench-inputs/bp_fe_lol_ng45_tp_gnn_placement \
  --task placement_dynamic_parameters
```

TP-GNN objective evolution has a separate
`bp_fe_lol_ng45_tp_gnn` profile. Its frozen `20_partition` is the unchanged-loss
reference, while candidate evaluation copies only `00_input` and `10_floorplan`
and recomputes partitioning. Keeping this profile separate avoids invalidating
the shared placement bundle when the TP-GNN reference contract changes.

Some tasks consume an immutable subset of a later frozen stage while still
recomputing that stage. Such inputs are declared under
`task.yaml:execution.frozen_inputs`; the common evaluator resolves them from the
validated benchmark run, checks required files and JSON assertions, copies the
complete directory into `harness_inputs/frozen/<name>`, verifies byte-for-byte
hash equality, and injects only the copied path into the resolved configuration.
For TP-GNN this contract adopts full-coverage partition-last timing features and
net weights from the reference partition while the candidate retrains and
repartitions. Task-specific artifact adoption is not hard-coded in
`evaluation.py`.

## Consume from every search baseline

EvoStage baselines use the manifest directly:

```bash
openagentic3dbench-evolve run placement_parameters \
  --method evostage \
  --evostage-root /data/gcr/EvoStage-main \
  --provider deepseek \
  --evaluation-mode full \
  --open3dbench-root /data/gcr/Open3DBench-V2.0 \
  --benchmark-input /data/gcr/openagentic3dbench-inputs/bp_fe_lol_ng45_tp_gnn_placement \
  --reference-run /data/gcr/references/bp_fe_lol_ng45 \
  --output-dir /data/gcr/evolution/placement_dynamic_parameters/evostage
```

Native OpenEvolve exports use the same bundle through the environment:

```bash
export OPENAGENTIC3D_BENCHMARK_INPUT=/data/gcr/openagentic3dbench-inputs/bp_fe_lol_ng45_tp_gnn_placement
```

The older `--baseline-run` and `OPEN3DBENCH_BASELINE_RUN` interfaces remain
available for external or historical runs, but they do not provide the unified
bundle's compatibility and integrity checks.
