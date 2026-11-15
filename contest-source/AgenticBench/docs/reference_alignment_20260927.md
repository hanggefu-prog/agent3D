# Versioned reference alignment (2026-09-27)

Status: implemented and tested offline. No new native scoring reference, model
call, candidate final, GPU placement or remote deployment is implied by this
change. Existing experiments, failures, task definitions and archived scores
remain unchanged. Do not resume a historical attempt with the new contract.

## Default workspace endpoints

| Objective | New task | Physical endpoint |
| --- | --- | --- |
| Timing | `lol_partition_placement_timing_one_pass` | Exactly one native OPT pass, pre-CTS passive-HBT setup STA |
| CTS thermal | `lol_partition_placement_cts_thermal_one_pass` | Exactly one native OPT pass, CTS, cell power, HotSpot snapshot |
| Routed thermal | `lol_partition_placement_thermal` | Native one-pass OPT, CTS, GRT/DRT, report and routed HotSpot |

One OPT pass means the native coordinated bottom/top optimization pass. It is
not two full `60_opt` executions. The new workspace endpoints do not apply the
legacy two-pass patch. The verifier selects an explicit profile and rejects
one/two-pass cross-comparisons. Both die records and the native pass record are
required. Timing is an early pre-CTS endpoint, not signoff.

Both `run_workspace_codesign.py` and `run_workspace_agentic3d.py` default to the
new reference family. `run_global_task_smoke.py` and the scheduled workspace
entry point use the same family. The legacy focused-program/knowledge-v2
drivers and historical two-pass tasks are not silently migrated. A workspace
profile is not a newly registered focused CLI evaluation profile.

Each new search still qualifies one fresh endpoint-specific reference before
any model call. That reference is separate from the five candidate finals.
The frozen release qualified original CTS plus GRT, not a complete four-case
routed-thermal endpoint; routed thermal needs its own new qualification.

## Shared recipe

The four versioned profiles are
`local_bench/benchmark_inputs/{bp_fe,bp,ariane133,swerv_wrapper}_lol_ng45_reference_20260923/`.
They bind the `open3dbench-four-case-cts-grt-20260923-v1` archive identity and
the two recorded Docker digest representations of its pinned image.

| Setting | Reference-aligned value |
| --- | --- |
| Integration / PDK | LoL / ng45 |
| Physical threads | 4; independent from the container CPU quota |
| Pins / partition / placement seeds | 1 / 1001 / 1000 |
| Partition | `tp-gnn-flat`, `global-2way`, timing driven, CUDA, fast 20-epoch recipe, minimum 100 optimizer steps |
| Placement | Concurrent DREAMPlace, native Nesterov, 4000 iterations, initial LR 0.01, 512 bins, density 0.6 |
| Native Nesterov mode | Original non-BB backtracking path; do not enable BB to accommodate a controller |
| Macros | Placement enabled, 10 um per-side halo, native ng45 snap guard 1.001 um |
| Deterministic kernel flag | Preserve native reference default (absent/0), not the old local task's forced 1 |
| OPT | One pass, buffer limit 10%, utilization limit 75% |
| CTS | Native OpenROAD/balanced, target skew 0.1 ns, no hold repair |
| GRT / DRT | 4 congestion iterations / max iterations 5, 10800 s die timeout, resource reduction 0, seed -1 |
| Thermal | 10-cell grid, ambient 318.15 K; endpoint-specific power provenance remains mandatory |

The TP-GNN learning rate remains its native implementation default 0.001; it
is not invented as an unsupported YAML/JSON option. Macro, PDK, power and
backend contracts remain owned by their native stages. A DRT `max_iterations`
setting is not a GRT tuning dimension.

## Physical inputs, not just matching YAML

`scripts/adopt_reference_20260923.py` imports the original `00_input`,
`10_floorplan` and `20_partition` into a new bundle without EDA. It checks the
archive index, frozen physical fingerprints, assignment hash, native flat
partition report and all copied stage artifacts. The historical timing inputs
remain under `20_partition/timing_input`; no artificial stage 15 is created.

The importer requires a clean isolated source and a new output directory. It
copies files independently, rebases only copied run-owned path metadata and
records both original producer provenance and the current importer identity.
It never labels the current source as the producer of historical artifacts.
The source archive is mounted read-only at both its actual and original paths.

The scheduled preparation phase takes an explicit `--reference-root`. Direct
offline adoption has the following interface (inside the pinned environment):

```text
python AgenticBench/scripts/adopt_reference_20260923.py
  --reference-root /path/to/agentic3d-reference-20260923-v1
  --case bp_fe --source /path/to/clean-isolated-source
  --output /path/to/new-batch/input
```

The local placement variants reuse the exact frozen partition. Global
co-design reuses frozen input/floorplan, then recomputes timing features and
partition with current source because partition is part of its search space.
Thus configuration and input alignment do not prove byte-identical new
assignments or numerical metrics. New scoring references must demonstrate
their measured result; archived scores are never installed as qualified scores.

## Two versioned placement tasks

- `placement_preconditioner_source_reference_20260923`: native initialization
  and preconditioning baseline; the initial adapter emits an empty source
  patch. Candidates retain the existing source-evolution interface.
- `placement_dynamic_parameters_reference_20260923`: a new, active Nesterov
  gradient-step control interface instead of the old Adam LR parameter hook.
  A trusted callback scales the gradient term before native acceleration and
  state updates; it does not move coordinates after the optimizer step.

The identity dynamic controller is called and logged but takes the exact native
arithmetic branch. It must qualify independently of the source task. Missing
callback evidence cannot be treated as a successful active interface. See
`local_tasks/placement_parameters_reference_20260923/QUALIFICATION.md` for the
offline evidence and the required native identity/nontrivial-control checks.
Offline tensor-double tests do not prove Torch or GPU numerical equivalence.

The old two placement tasks and their Adam/2000-iteration history remain
available under their original IDs. Their old producer-identity failures are
not erased. The evaluator now obtains the expected partition producer from
trusted selected configuration before applying candidate overrides, so
`tp-gnn-flat` cannot be rejected merely because the historical default was
`tp-gnn`; mismatched or unbound native evidence still fails.

## Scope and qualification limits

This change aligns the workspace global recipe and the two new local placement
variants. It does not assert that every historical local task now shares one
algorithm: TA3D, pin-cut-fitness and Mt-KaHyPar tasks define different research
targets and are not silently converted to global-2way TP-GNN. Their migration
requires an explicit task-family decision and new qualification.

No model/EDA retry, budget extension, cache cleanup, driver change, credential
copy, global configuration change or task-tracking reactivation is part of this
work. All future physical validation must use new attempt directories and
separate accounting for references, model calls, generated candidates, finals,
rejections, native attempts, successful evaluations and improvements.
