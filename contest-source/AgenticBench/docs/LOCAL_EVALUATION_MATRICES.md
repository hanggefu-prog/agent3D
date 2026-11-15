# Multi-case, paired-seed local evolution

The original eight local tasks accept an optional evaluation matrix. The
single-case API and its historical seed defaults remain unchanged. The two
additional GNN loss/decoder tasks currently bind partition seed 1001 and have
no defined seed-matrix contract. They are rejected by matrix rendering and
evaluation; the existing eight-task profile and its reference identities remain
unchanged. A matrix
preserves each task's existing algorithm scope, verifier, physical endpoint and
scalar objective. It does not apply the global post-GRT Pareto objective to
placement-only or HBT-only tasks.

Each candidate uses the same complete Cartesian product of cases and seed
vectors. Each cell is normalized by its own qualified initial-program reference.
The selection score is the arithmetic mean across seeds within a case, then the
equal-weight mean across cases. The unchanged reference scores 1. All cells must
succeed and return finite scores; a failed cell invalidates the aggregate. Raw
WNS, HPWL, overflow and other measurements remain cell diagnostics rather than
being averaged across designs with different scales.

## Matrix file

Use JSON or YAML. Paths are relative to the matrix file or absolute. Case IDs
identify a design/integration/PDK combination; every seed of a case must have that
same physical identity. No case weights are accepted.

```yaml
schema_version: 1
task: placement_initialization
cases: [bp_fe-ng45, bp-ng45]
seeds:
  - {id: s1, io: 1, partition: 1001, placement: 1001, routing: -1}
  - {id: s2, io: 1, partition: 1002, placement: 1002, routing: -1}
cells:
  - {case: bp_fe-ng45, seed: s1, base_config: bp_fe/s1.yaml, baseline_run: inputs/bp_fe/s1/run, reference_run: null}
  - {case: bp_fe-ng45, seed: s2, base_config: bp_fe/s2.yaml, baseline_run: inputs/bp_fe/s2/run, reference_run: null}
  - {case: bp-ng45, seed: s1, base_config: bp/s1.yaml, baseline_run: inputs/bp/s1/run, reference_run: null}
  - {case: bp-ng45, seed: s2, base_config: bp/s2.yaml, baseline_run: inputs/bp/s2/run, reference_run: null}
```

The explicit base configs must contain matching `pins.seed`, `partition.seed`,
`placement.seed` and `routing.seed`. The host injects those seeds into both the
flow configuration and embedded partition/placement parameters after verifying
the candidate. Placement-only invocation receives the seed through its native flow-config input.
Native partition/placement output is checked for the executed seeds.

Use existing Open3DBench preparation/replay to produce the required immutable
upstream stages for each cell. A placement task needs the matching partition;
HBT and routing need the matching placement as well. The evaluator checks actual
hash-bound upstream seed evidence. Renaming one checkpoint as several different
seeds is rejected. Keep I/O at 1 and routing at -1; no fictitious OPT or HBT RNG is
introduced. Different seed vectors can represent different upstream physical
inputs to a deterministic local algorithm.

For dynamic Nesterov use the explicit current-source recipe in
`local_bench/reference_configs/nesterov_current.yaml`, with the selected design
and seed values. Dated placement reference profiles remain fixed and cannot be
silently widened to new seeds.

## Default matrix profile

The repository default for cases, seeds, workers, timeout and routing overrides
is the declarative profile
`local_bench/reference_configs/default_matrix_profile.yaml`
(`local_default_bp_fe_ariane133_3seed`). It was taken from the release scripts
of `releases/local-matrix-evolution-20261018`, which are unchanged. It contains:

- cases `bp_fe-ng45` (`bp_fe`) and `ariane133-ng45` (`ariane133`);
- seeds `s1` to `s3` = `{io: 1, partition: 100n, placement: 100n, routing: -1}`
  for n = 1001, 1002, 1003;
- six workers for each frontend/HBT task, with the Nesterov task using
  `nesterov_current.yaml` as its base config;
- two workers for each routing task, with `threads` and `routing.threads` 16
  and `routing.resource_reduction` 0.5 for `bp_fe` and 0.0 for `ariane133`;
- `timeout_seconds` 3600 for each cell.

The loader (`local_bench/matrix_profile.py`) rejects unknown keys, duplicate
cases or seeds, a task set that differs from the eight maintained tasks,
non-scalar overrides, overrides of `design`/`type`/`pdk` or of any `*.seed`,
and any profile that claims `qualified: true`. A profile is a template. It
never carries reference runs.

`render-matrix` turns the profile plus one frozen upstream run per cell into a
reference-free matrix:

```sh
cat > baseline_runs.json <<'EOF'
{"bp_fe-ng45":     {"s1": "/runs/bp_fe/s1", "s2": "/runs/bp_fe/s2", "s3": "/runs/bp_fe/s3"},
 "ariane133-ng45": {"s1": "/runs/ariane133/s1", "s2": "/runs/ariane133/s2", "s3": "/runs/ariane133/s3"}}
EOF
python3 -m local_bench render-matrix placement_initialization \
  --baseline-runs baseline_runs.json --output-dir /path/to/matrix
```

It writes `configs/<case>/<seed>.yaml` and `matrix.json` with
`reference_run: null` in every cell, and prints the task's worker and timeout
settings. A missing cell is an error. The rendered matrix still has to go
through `qualify-matrix` below before it can score candidates. Use
`--profile` to render from a different profile file. That profile must pass
the same validation.

## Routing score versions

The three routing tasks score with version 1 (`0.1` overflow and `0.9`
wirelength blend) unless their `task.yaml` declares `scoring.version: 2`. No
maintained task declares it. Version 2 (overflow gate, then `W_ref / W`) is
meant for new task versions. A matrix qualified under version 1 is not a
reference for version 2. See [the task report](TASK_SETTINGS_REPORT.md#routing-score-version-2-opt-in).

## Qualify and evaluate

Commands below run from `AgenticBench`. Native tools must run in a pinned Docker
image. The optional Docker transport reuses the global resource queue, pinned
container runner, and installed routing-candidate image builder.

Create a host-owned runtime JSON with `queue`, `host`, `image`, and optionally
`compiler_cache` and `admission_timeout`. The queue must already contain the
host's capacity. For rafa1 the existing capacity is two GPU devices with three
slots each; do not register a second independent six-slot pool.

```sh
python3 -m local_bench qualify-matrix placement_initialization \
  --matrix inputs.yaml --open3dbench-root /path/to/source \
  --output-dir /path/to/references --matrix-workers 6 \
  --matrix-runtime runtime.json

python3 -m local_bench evaluate placement_initialization \
  --candidate candidate.py --open3dbench-root /path/to/source \
  --matrix /path/to/references/qualification/qualified_matrix.json \
  --output-dir /path/to/candidate-result --matrix-workers 6 \
  --matrix-runtime runtime.json
```

Qualification executes the exact initial program on every input. It publishes
`qualified_matrix.json` only after every cell succeeds. Bootstrap proxy scores
from routing qualification are not optimization scores. Qualification is an
explicit separate operation and never spends a search candidate budget.
Routing references must also pass the task's hook-evidence gate on every
case/seed cell. A clean route that never enters the policy's loop is not a
qualified reference for that policy. Choose exercised inputs or freeze an
appropriate routing-capacity configuration before qualification; keep it fixed
for all candidates. An old receipt without this evidence is rejected on load.
References retain their physical evidence; compact candidate evaluations may
remove disposable workspaces and run trees using the existing local retention
policy. Full cell receipts and logs live under `cells/<case>/<seed>/`.

Routing requires a clean source checkout with its pinned OpenROAD/OpenSTA
submodules initialized. Routing builds happen once per candidate matrix, using the existing verified
installed-image builder and optional shared compiler cache. Builds reserve no
GPU. HBT and GRT cells also reserve no GPU; frontend cells receive a slot from
the shared queue. Leases end immediately after each container terminates. A
TP-GNN timing cell runs through placement in a GPU container, then confirms that
container has terminated before shrinking its live lease to CPU-only resources.
A host-sealed continuation binds the request, candidate, configuration and
placement checkpoint. A second container runs PDN, HBT and STA without a GPU;
it does not repeat partition or placement. Both phases share one time budget.
Unknown termination retains the affected allocation. Admission and cell
execution have explicit time limits.

Without a Docker transport, native evaluation must already be inside the
maintained runtime and is sequential. Parallel direct execution is rejected to
avoid unbounded resource contention. Custom trusted transports can use the same
single-cell callable interface; an optional `prepare_candidate` hook prepares
one runtime before cell dispatch. They must honor the seed and path arguments,
isolate execution, and enforce resources themselves.

## Evolution API and feedback

```python
from pathlib import Path
from local_bench import TaskEvaluationOptions, run_task_search
from local_bench.matrix_docker import DockerMatrixTransport

options = TaskEvaluationOptions(
    task_dir=Path("local_tasks/placement_initialization"),
    output_dir=Path("/path/to/new-search"),
    open3dbench_root=Path("/path/to/source"),
    evaluation_matrix=Path("/path/to/qualified_matrix.json"),
    matrix_workers=6,
)
transport = DockerMatrixTransport(**runtime_settings)
result = run_task_search(backend, options, max_evaluations=2,
                         evaluation_transport=transport)
```

One search budget unit is one candidate matrix. The prompt names cases and seed
vectors and states the aggregation rule. Feedback includes the aggregate score,
per-case mean/min/max/population standard deviation, each seed's score, selected
physical diagnostics, and result directories. Both ask/tell OpenEvolve and the
EvoStage bridge consume this feedback. The EvoStage CLI accepts the same
`--matrix`, `--matrix-workers` and `--matrix-runtime` switches.
Failed cells also include bounded native exception excerpts with log paths, when
available, so the next proposal can distinguish an unsupported API from a
physical failure. These excerpts are diagnostic text and never produce scores.
When a wrapper names a failed native log, feedback follows that log, including
its compact copy. A missing causal log does not justify substituting an older
stage's warning. Compact retention includes PDN logs; algorithm activity flags
remain visible even when false.

Matrices are frozen for a search session. Changing a config, upstream manifest
or reference receipt invalidates even an otherwise cached candidate. Duplicate
cells, missing cells, duplicate seed vectors, wrong-task references and mixed
seed/design inputs are rejected. Existing task/native reference validation still
runs for every cell. There is no implicit resume, retry, favorable-seed selection,
or held-out selection in this feature. Use a separate frozen matrix for tests;
only the optimization matrix should be supplied during evolution.
