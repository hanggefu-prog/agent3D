# Local multi-case, paired-seed matrices: 2026-10-18

The eight maintained local evolution tasks now accept explicit case/seed
matrices. The existing single-case API, task objectives and source verifiers
remain compatible. See [the configuration and API guide](../../docs/LOCAL_EVALUATION_MATRICES.md).
This extends the completed [low-budget evolution audit](../low-budget-evolution-20261017/README.md);
it does not spend new model calls or claim that every local algorithm has been
physically rerun with the new matrix interface.

## Contract

- One candidate uses one source on every case/seed cell, with its own paired
  initial-program reference. Reference qualification is explicit and publishes
  a usable matrix only after every cell passes.
- The existing cell score is averaged across seeds, then equally across cases.
  A failed, missing or nonfinite cell prevents an aggregate score. Raw physical
  measurements remain per-cell diagnostics. Reference replay scores 1.
- One search evaluation budget unit is one candidate matrix. The prompt and
  feedback expose cases, seed vectors, case means and variability, seed scores,
  physical diagnostics and result paths. Frozen-reference checks also precede
  cached-candidate reuse.
- The Docker transport reuses the shared global resource queue and installed
  candidate-image builder. Each C++ candidate is built once for its full matrix;
  no build, HBT or GRT cell reserves a GPU.
- OpenEvolve's scalar-only population database is supplemented from trusted
  ask/tell history when rendering a selected parent's matrix feedback, including
  after island migration changes its database ID. EvoStage accepts the same
  matrix/runtime options.

## Native validation on rafa1-gcr

Both tests used BP_FE and BP with ng45, two paired frontend seeds (1001 and 1002),
fixed I/O seed 1 and routing seed -1. The resource pool retained its existing
two-device, three-slots-per-device capacity. No formal campaign was changed.

| Path | Reference cells | Candidate cells | Result | Total wall time |
| --- | ---: | ---: | --- | ---: |
| Placement initialization | 4 | 4 identity + 4 modified | All 12 pass | 506.13 s |
| GRT rip-up policy | 4 | 4 modified | All 8 pass | 804.89 s |

The placement identity replay scored exactly 1 in all four cells. The modified
initializer applied a small, generic sinusoidal perturbation to movable starting
coordinates. Its scores were:

| Case | Seed 1001 | Seed 1002 | Case mean | Population standard deviation |
| --- | ---: | ---: | ---: | ---: |
| BP_FE | 0.997435 | 0.999200 | 0.998317 | 0.000882 |
| BP | 1.006214 | 0.923167 | 0.964691 | 0.041523 |

The aggregate is **0.9815040470**, so the unchanged reference remains better.
Selecting BP's favorable seed alone would have misrepresented this change.
This is a matrix/feedback validation, not a newly optimized placement method.

The GRT candidate reused the earlier model-produced detour retry policy. All
four cells scored **1.0** and the logged policy-change counters stayed zero:
the source compiled and hooks executed, but these inputs did not exercise its
extra selection condition. Do not interpret this as an algorithm improvement.
Reference and candidate cached compilation took 202.65 s and 214.22 s;
packaging took 7.12 s and 7.26 s. There were exactly two image builds for eight
GRT cells, rather than one build per cell.

Placement reached four simultaneous GPU slots (the matrix has four cells).
GRT reached two simultaneous CPU cells and zero GPU slots. All jobs belonging
to these completed tests have terminal states and no remaining leases. Four
additional placement-preparation jobs exported original DEFs from ODB with hash
checks; they are not extra candidate evaluations. Free SSD space at collection
was 87.92 GiB.

## Evidence and fixes

Feature implementation: `b7676d7ce79c8538f6c83b8c3639b09135d7aea4`.
The final integration branch also includes prompt-history recovery, an existing
TP-GNN diagnostic display repair, and CLI output-overwrite protection.

The placement test used the staged framework; the six evaluator/seed/physical
validation modules match the committed release byte for byte. The GRT test used
the clean committed release with initialized OpenROAD/OpenSTA submodules.
Hashes, exact programs, cell measurements, result paths, build phases and resource
records are in [native-results.json](native-results.json).

Native results remain under
`/data-ssd/gcr/agenticbench-integration-20261016/`:

- `local-matrix-native-20261018-v3/references/qualification/qualified_matrix.json`
- `local-matrix-native-20261018-v3/search/`
- `local-matrix-grt-20261018-v3/references/qualification/qualified_matrix.json`
- `local-matrix-grt-20261018-v3/search/`

Early setup attempts are preserved separately and are not counted as successful
native evaluations. They exposed an unsupported placement CLI seed argument,
missing on-demand DEF exports, an uninitialized OpenROAD checkout, and the
builder's expected rejection of dirty source. The maintained flow-config seed
interface, hash-checked ODB export and clean source deployment resolved them;
no incomplete reference matrix was published.

## Coverage limits

The final maintained regression gate passed **1,239 tests and 180 subtests**,
with **8 skips**, in 66.72 s. It includes real OpenEvolve 0.4.0 ask/tell tests;
the selected environment does not provide every optional backend. Exact skip
reasons and the test command are in [regression-summary.json](regression-summary.json).
The historical full suite is not claimed to pass.

Protocol regressions cover all eight maintained tasks, incomplete matrices,
seed/input/reference mismatches, equal weighting, frozen caches, native seed
handoff, resource leases and prompt feedback. Real physical matrix coverage in
this change is placement initialization and GRT rip-up; the preceding audit
covers all eight local tasks, three engineering tasks and current global
post-GRT evolution at their documented budgets.

Local TP-GNN currently runs partition and downstream placement/HBT/STA in one
native container. Its GPU lease therefore lasts for that whole cell. Other
cells release their leases as their containers terminate. The global flow's
existing stage-level GPU release is unchanged. This feature does not introduce
mid-container release for the local partition path.
