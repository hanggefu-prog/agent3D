# Open3DBench Development Guide

## Mission

Open3DBench should become one reproducible, extensible 3D-IC flow rather than
two neighboring MoL and LoL projects connected by file-copy conventions.

The unified flow must support both:

- **MoL (memory-on-logic):** heterogeneous tiers, with memory macros primarily
  on the top die and logic primarily on the bottom die.
- **LoL (logic-on-logic):** two logic tiers, including partition-then-place and
  joint/True-3D placement methods.

Unification does **not** mean hiding the physical differences between MoL and
LoL. It means that both use the same experiment lifecycle, stage interfaces,
artifact schema, execution environment, and evaluation contract. Method- and
integration-specific behavior should be implemented behind explicit adapters.

## Accepted Thin Architecture (2026-08-04)

The first implementation deliberately uses a smaller architecture than the
long-term generalized IR described later in this guide. This decision reflects
the current scope: two maintained 3D PDKs, known benchmarks, and existing
DEF/Verilog/SDC interfaces. Do not add a large prepare database or require users
to understand internal capability objects for this milestone.

The accepted user and artifact model is:

- one YAML file with `design`, `type`, `pdk`, `input`, `partition`,
  `placement`, `hbt`, `opt`, `cts`, `routing`, and `thermal`;
- `type` is simply `mol` or `lol`; integration-specific defaults and validity
  checks live in a small registry;
- users select a maintained peer PDK profile (`ng45` M10 or `asap7`);
  layer, cell, row, PDN, RC, HBT-grid, and routing contracts live in the small
  platform registry rather than leaking PDK-name branches into stages;
- native DEF, Verilog, SDC, and ODB checkpoints remain the physical source of
  truth; `partition.csv`, `hbt.csv`, and a local name map are the only new
  sidecars required by the first implementation;
- source instance and net names remain canonical wherever possible. Do not
  invent repository-wide numeric IDs merely to normalize current benchmarks;
- each run has fixed stage directories and one root `run.json`; per-stage
  metadata stays small and local;
- prepare is a thin input action: validate paths, run an existing contest
  conversion when required, or copy a checkpoint into `00_input`. It must not
  mutate released benchmark files;
- synthesis is optional. Benchmark and checkpoint inputs already provide the
  synthesized/floorplanned state. RTL input invokes Yosys plus initial
  OpenROAD floorplanning inside `00_input`; synthesis is an input frontend,
  not a stage of the main 3D flow;
- the main flow contains only 3D infrastructure. Existing 2D ORFS designs and
  `do-2D_flow` remain unchanged as input-generation tools and comparison
  baselines; do not register them as main-flow stages or 3D algorithms.

The initial run layout is:

```text
runs/<design>/<run-id>/
  config.yaml
  run.json
  00_input/
  10_floorplan/
  20_partition/
  30_placement/
  40_pdn/
  50_hbt/
  60_opt/
  70_cts/
  80_route/
  90_report/
  95_thermal/
```

`00_input/` owns synthesis/checkpoint import only. `10_floorplan/` is the
partition-independent physical pre-plan: it publishes `floorplan.json`,
`io_plan.csv`, `preplan_manifest.json`, the finalized bottom-die BPin geometry,
and a planned PDN contract. All external I/O is fixed on the bottom die. The
maintained policy is deterministic random legal slots, a partition-free
DREAMPlace projection, then final OpenROAD PPL optimization. This is the sole
supported I/O entry for RTL and checkpoint inputs; do not reintroduce fixed
imported pins or a projection-free PPL branch. Partitioning does not own or move
external I/O.

The root `open3dbench` package is a thin orchestrator around maintained
Place-3D and OpenROAD stages. Prefer explicit, small compatibility checks over
a generic plugin/capability framework until current MoL and LoL vertical slices
are stable. The explicit validated reuse of the two-stage frontend prefix is
implemented; a generalized cross-stage cache and the broader Design IR ideas
below remain future options rather than requirements for the first unified flow.

The authoritative benchmark matrix contains 12 chip designs, all with
registered RTL synthesis frontends and LoL support: `ariane133`, `ariane136`,
`bp`, `bp_be`, `bp_fe`, `bp_multi`, `bp_quad`, `swerv_wrapper`, `mempool`,
`gemmini`, `ibex`, and `jpeg`. MoL supports the ten macro-bearing designs.
Gemmini has peer ng45 and ASAP7 frontends with the same generated RTL and exact
SRAM interfaces but PDK-specific physical views and registered footprints. Ibex and JPEG
are explicitly no-macro and therefore LoL-only. Small checkpoint-only designs
are intentionally excluded from the maintained 3D benchmark matrix.
Keep benchmark names, RTL config paths, macro classification, and backend
directory aliases in one catalog rather than scattered conditionals.

## Design Principles

### HBT and CTS clock ownership

HBT materialization must load the real SDC before buffering data I/O, preserve
STA-recognized clock connectivity, and leave clock-HBT leaf separation to CTS.
Temporary clock-net protection covers only `buffer_ports`; restore the original
state before creating boundary BTerms. Do not synthesize fake clocks or filter
clock names heuristically. TritonCTS uses persistent `OPENROAD_CTS_BUFFER`
ownership (ODB and DEF), not `SOURCE TIMING` plus a master name: Resizer uses
the same source type. Keep the no-clock-tree guard and strict warning audit.
Restart old runs from a pre-CTS checkpoint; do not relabel legacy post-CTS
results. See `docs/CTS3D.md` and the `resizer_clock_buffer` native regression.

### General principles

1. **One flow, explicit variants.** A user should select an integration type,
   design, placement method, and configuration through one top-level entry
   point. They should not need to know which legacy directory or shell script
   owns the next stage.
2. **Stable stage contracts.** Partitioning, placement, HBT planning,
   legalization, backend implementation, and evaluation communicate through
   documented artifacts and manifests, not implicit paths or copied files.
3. **Shared orchestration, specialized algorithms.** Common control flow and
   bookkeeping belong in the framework layer. MoL/LoL physics and individual
   research methods remain separate implementations selected by configuration.
4. **Reproducibility first.** Every run records configuration, source revision,
   image identity, seed, method, input checksums, stage status, runtime, and
   output locations. Packaged paper artifacts must remain distinguishable from
   newly generated results.
5. **Comparable evaluation.** All methods hand a canonical placed design to the
   same OpenROAD-3D and HotSpot evaluation layer. Metrics must identify their
   provenance and use consistent names and units.
6. **No silent fallback.** Missing binaries, data, GPU support, or optional
   tools must be reported clearly. A fallback is allowed only when it is
   explicitly configured and recorded in the run manifest.
7. **Incremental migration.** Keep working legacy entry points during the
   transition. New code should target the unified contracts; compatibility
   wrappers may call the new flow until old entry points can be deprecated.
8. **Research extensibility.** A new partitioner, placer, HBT planner, or metric
   implementation should be addable without modifying unrelated stages.

## Primary Algorithm Evaluation Tracks

The user-facing inventory of currently reachable methods, legal LoL/MoL
combinations, control baselines and validation coverage is maintained in
`docs/ALGORITHM_AND_FLOW_MATRIX.md`. Keep that document synchronized with
`open3dbench/registry.py` and the runner whenever a method is added, connected
or promoted to a stronger validation status.

The framework exists to provide full-flow evaluation and strong, reproducible
baselines for four algorithm tracks. These tracks are first-class workflows,
not incidental steps hidden inside an end-to-end script.

### 1. 3D Partition

Contract: a versioned logical netlist plus die capacities and eligibility
constraints in, a complete die assignment out.

The public assignment is deliberately small and identical for LoL and MoL:
`partition_assignment.csv` contains only `inst_name,die`. `inst_name` is the
original DEF component name; contest-only IDs such as `C123` stay inside the
LoL adapter. Cell type, geometry, and macro status come from the design input
instead of being copied into this CSV.

- LoL normally allows logic cells and macros to move between logic dies.
- MoL uses the same contract with heterogeneous eligibility constraints: logic
  cells may be fixed to the logic die while memory macros are assignable to
  compatible dies. Fixed load counts toward capacity balance.
- Required direct metrics include area/capacity balance, macro count and area
  balance where applicable, cut nets, cut pins, weighted/timing-critical cuts,
  predicted HBT count, runtime, memory, and determinism across seeds.
- Full-flow metrics must also run the partition through a fixed reference
  placer, HBT policy, and backend so a partition method is not ranked only by
  a proxy objective.
- Initial baselines include random balanced assignment, a simple greedy/FM
  reference, hMETIS, Mt-KaHyPar, TP-GNN, PyG min-cut, and the existing MoL
  GNN/min-cut/max-cut methods when their constraints are compatible.

### 2. 3D Placement with a Fixed Partition

Contract: canonical design plus an immutable partition assignment in, legal
per-die component coordinates and cross-die net geometry out. The placer must
not silently change the supplied tier assignment.

- Required direct metrics include per-die legality, overlap, density/overflow,
  2D and 3D HPWL, cross-die wirelength contribution, congestion proxies,
  displacement where relevant, runtime, memory, and determinism.
- Full-flow evaluation adds HBT legalization, 3D optimization, CTS, routing,
  timing, power, area, DRC, routability, and thermal results.
- Initial LoL baseline: the existing concurrent two-die DREAMPlace flow.
- Initial MoL baselines: the existing analytical and tiling flows, adapted so
  partition input is external and immutable rather than recomputed internally.

### 3. HBT Assignment and Legalization

Contract: cross-die nets, ideal HBT positions, legal HBT sites/rules, die
geometry, and optional blockages in; exactly one legal assigned HBT site per
required cross-die connection out.

Ideal-position generation and legalization are separate interfaces. This lets
one experiment compare legalizers on identical ideals and another compare HBT
objectives while holding legalization fixed.

- Required direct metrics include assigned/unassigned count, legality,
  displacement from ideal, HPWL increase, congestion/routing cost, runtime,
  memory, and deterministic tie breaking.
- Full-flow metrics include GRT/DRT completion, DRC, HBT-via correctness,
  timing, and power impact.
- Initial baselines include nearest-free-slot greedy (the existing LoL slot
  legalizer), a global min-cost matching/min-cost-flow legalizer, and the
  existing routing-aware greedy HBT placer. Centroid and optimal-region-center
  ideal-position policies remain explicit baselines.
- Sparse global min-cost assignment starts from the requested Manhattan-nearest
  candidate count and doubles it only when a full one-to-one matching does not
  exist. Reports preserve both requested and effective candidate counts; exact
  mode remains available by omitting the limit.

### 4. True-3D Placement

Contract: canonical design and platform constraints in; partition, placement,
and HBT ideals/legal positions out as one atomic result. A True-3D method is
not represented by dummy outputs from the three staged interfaces.

- Direct evaluation combines the partition, placement, and HBT metric groups.
- Full-flow evaluation uses the exact same canonical backend handoff as the
  staged flow.
- Initial strong baselines are the authorized ICCAD 2023 placers and
  TCAD25/CUPID already supported by `Place-3D`. Source-available methods should
  implement the same contract directly; external binaries use adapters.

Every track supports two evaluation levels: **stage-only**, for fast algorithm
iteration, and **full-flow**, for scientifically meaningful PPA/routability/
thermal comparison. The evaluator configuration and downstream reference
methods are part of the experiment identity.

## Target Flow

The intended logical pipeline is:

```text
experiment config
  -> input preparation and validation
  -> tier assignment / partition (optional for a joint placer)
  -> macro and/or standard-cell placement
  -> HBT planning and legalization where required
  -> canonical placed-design handoff
  -> per-die backend implementation and die-aware routing
  -> timing, power, area, wirelength, routability, and thermal evaluation
  -> normalized metrics and archived run manifest
```

After placement/HBT handoff, the physical-design pipeline is explicitly:

```text
canonical placed design
  -> 3D optimization (buffering, sizing, repair, die-aware timing model)
  -> 3D CTS (clock topology, per-die buffers, cross-die clock HBTs)
  -> 3D global routing
  -> 3D detailed routing and HBT terminal-to-via finalization
  -> signoff-style reports and thermal evaluation
```

`3D OPT`, `3D CTS`, and `3D routing` are modular backend stages with their own
inputs, outputs, metrics, and baselines. A convenient combined backend flow may
compose them, but must not erase their experiment boundaries.

The orchestrator should expose a command equivalent to:

```bash
open3dbench run --integration <mol|lol> --design <name> \
  --method <method> --config <file>
```

The exact CLI name may change while it is being implemented, but the user model
should remain this simple. Batch execution should be a thin layer over the same
single-run API.

## Target Software Architecture

The root-level `open3dbench` Python package will provide framework code while
existing component directories continue to own algorithms during migration:

```text
open3dbench/
  cli.py                    # run/evaluate/validate/list commands
  core/
    design.py               # canonical 3D Design IR
    artifacts.py            # versioned artifact readers/writers
    manifest.py             # provenance and stage status
    units.py                # explicit DBU/micron conversion
  schemas/                  # JSON schemas and schema versions
  stages/                   # abstract stage protocols and validators
  adapters/
    lol/                    # Place-3D LoL implementations/binaries
    mol/                    # Place-3D MoL implementations
    openroad/               # backend stage adapters
  evaluators/               # stage and full-flow metric collectors
  workflows/                # staged, true3d, backend, and full-flow DAGs
platforms/
  nangate45_3d/
    platform.yaml           # stack, layers, libraries, sites, tracks, HBT rules
experiments/                # checked-in resolved experiment templates
tests/
```

The orchestrator depends on stage protocols; algorithms never import the
orchestrator. Adapters translate legacy input/output formats at the boundary.
`Place-3D` is now the shared frontend implementation module, but it is not the
eventual root orchestrator or canonical Design IR. Promote generalized concepts
into `open3dbench`; keep LoL contest rules and MoL physical assumptions behind
their respective adapters rather than weakening either model for symmetry.

### Canonical 3D Design IR

The IR is a versioned artifact bundle, not one giant in-memory class that every
external binary must import. Its first implementation should use inspectable
JSON/CSV/JSONL plus DEF/ODB where physical fidelity is needed:

- design/platform metadata and coordinate units;
- dies, capacities, placement regions, orientations, and layer windows;
- library masters and die compatibility;
- instances with type, dimensions, fixed state, and allowed dies;
- nets and pins with optional timing weights;
- current partition assignment;
- current per-die placement;
- cross-die connection/HBT records;
- references to canonical DEF/ODB checkpoints.

Stable instance and net IDs must survive all adapters. Suffixes such as
`_top`/`_bottom`, contest IDs such as `C1`/`N1`, and side-by-side visualization
offsets are serialization details, never canonical identity.

### Stage Protocols

Each stage receives an immutable input artifact, a resolved config, a platform
object, and an isolated output directory. It returns a typed artifact plus a
stage manifest. The initial protocols are:

- `Partitioner.run(DesignIR, PartitionConfig) -> PartitionResult`
- `Placer.run(DesignIR, PartitionResult, PlacementConfig) -> PlacementResult`
- `HBTIdealizer.run(DesignIR, PlacementResult, HBTConfig) -> HBTIdeals`
- `HBTLegalizer.run(DesignIR, HBTIdeals, Platform) -> HBTAssignment`
- `True3DPlacer.run(DesignIR, True3DConfig) -> True3DResult`
- `Optimizer3D.run(PhysicalState, BackendConfig) -> PhysicalState`
- `CTS3D.run(PhysicalState, BackendConfig) -> PhysicalState`
- `Router3D.run(PhysicalState, BackendConfig) -> RoutedState`
- `Evaluator.run(stage input, stage output, protocol) -> Metrics`

Validators enforce completeness, identity preservation, eligibility, geometry,
units, and required artifacts before the next stage runs.

### Workflow DAGs

The framework exposes composable workflows rather than one rigid pipeline:

- `partition`: partition plus stage evaluator;
- `placement`: imported/fixed partition plus placement evaluator;
- `hbt`: imported ideal HBTs plus legalizer evaluator;
- `true3d`: joint method plus combined evaluator;
- `backend`: imported canonical placement through OPT/CTS/routing;
- `full`: staged or True-3D frontend followed by the selected backend;
- `matrix`: controlled baseline/seed/design sweeps over any workflow.

Artifacts are content-addressed for resume/caching. A resumed stage must match
the input, config, platform, code, and image fingerprints recorded previously.

## Canonical Stage and Artifact Contract

Each maintained run lives in one self-contained directory:

```text
runs/<design>/<run-id>/
  config.yaml             # normalized configuration actually executed
  run.json
  00_input/
  10_floorplan/
  20_partition/
  30_placement/
  40_pdn/
  50_hbt/
  60_opt/
  70_cts/
```

A stage may be skipped when it is genuinely inapplicable, but the manifest must
record the skip and its reason. A joint placer, for example, can own partition,
placement, and HBT generation while still publishing the same handoff artifacts.

Every completed directory owns a `stage_manifest.json` containing:

- schema version, stage, integration, method, design and PDK;
- normalized stage and upstream signatures;
- implementation and runtime fingerprints;
- repository commit/dirty state, tool versions and container identity;
- relative artifact paths, sizes and SHA-256 hashes.

Placement through report also publish `checkpoint.json`. From PDN onward the
checkpoint must reference the immediately preceding physical checkpoint in the
same run. CTS publishes `ready_for_route`, routing publishes
`ready_for_extract`, and report publishes `ready_for_thermal`. Thermal closes
the release flow with a small checkpoint that references report rather than
duplicating its physical databases.

## Repository Responsibilities During Migration

- `Place-3D/` owns the unified LoL/MoL frontend implementations, benchmark
  adapters, contest conversion, and the current file-oriented `Partitioner`,
  `Placer`, and `HBTPlanner` contracts. LoL and MoL share one DREAMPlace source,
  build, result root, and validation suite while retaining regime-specific
  algorithms and physical semantics.
- `Place-3D/src/partition/{lol,mol}` and `Place-3D/src/placement` are the source
  of current frontend algorithms. New work must not recreate top-level
  `Place-LoL/` or `Place-MoL/` source/build trees.
- `OpenROAD-3D/` owns backend implementation, die-aware routing, HBT routing
  support, PPA extraction, and HotSpot thermal evaluation. Its `openroad/`
  directory is the pinned `OpenROAD-3D` source submodule, whose `src/sta`
  directory is the pinned `OpenSTA-3D` submodule. It should consume a canonical
  handoff instead of reaching into placement-specific result trees.
- `docker/` defines the shared runtime. The final image contains installed
  tools and checksum-allowlisted runtime binaries; generated data, historical
  outputs, restricted team submissions, and licenses remain external.
- The repository root should own the unified CLI, schemas, orchestration,
  top-level build/verification targets, and framework documentation.
- the PDK registry and generated OpenROAD platform directories are the source
  of truth for the maintained ng45 M10 and full-metal ASAP7-3D profiles.
  Their common stages consume the same fields; PDK-specific derivation belongs
  in platform generators and data, not duplicated orchestration.

## Implemented Today

- Scalar post-GRT campaigns can opt into host-authenticated best-so-far storage
  retention. Every admitted candidate retains source, diffs and exact metrics;
  record setters retain full text logs, while inactive redundant libraries and
  unneeded physical suffixes can retire through dependency-bound plans. Historical
  reading does not replace fresh native verification. See
  `AgenticBench/docs/STORAGE_BEST_TRACE_RETENTION.md`.

- Maintained local evolution tasks accept an explicit case-by-seed matrix while
  retaining the single-cell API. Cells use qualified paired initial-program
  references and unchanged task objectives; only complete matrices receive the
  equal-case mean of seed scores. The local Docker transport reuses the global
  resource queue and verified routing image builder (one build per candidate).
  Prompt feedback includes case/seed diagnostics and variation. Native seeds
  are host-owned and checked against upstream and executed physical evidence.
  See `AgenticBench/docs/LOCAL_EVALUATION_MATRICES.md`. Local TP-GNN now releases
  its GPU after the placement container is confirmed stopped, then continues
  PDN/HBT/STA in a CPU-only container from a host-sealed same-run checkpoint.
  The CPU lease stays active and both phases share one evaluation budget.
  Matrix-level resume and automatic retries remain unsupported.

The following capabilities exist and should be preserved during unification.

### Shared infrastructure

- Run and stage reading views (`RUN_SUMMARY.md`, `SUMMARY.md`) explain existing
  measurements and point to their structured sources; factual `events.jsonl`
  records distinguish tool process completion from stage validation. These are
  regenerable diagnostics, not evaluator inputs or physical release artifacts.
  Offline summaries require a new output directory outside the source run and
  never rewrite sealed reference results. The reading contract is documented
  in `docs/AGENT_LOG_READING.md`; the original manifests/verifiers remain
  authoritative. Missing or partial metrics must never be promoted to zero or
  final signoff values by a summary.
  New command transcripts add independent `[open3dbench-log]` input/execution/
  output boundaries without changing native lines. Copied upstream log bodies
  are explicitly bracketed, not treated as another algorithm execution; source
  line references include those markers. Sealed logs are never annotated in place.
  All generated summaries, diagnostic explanations, and log annotations use
  English. Preserve original tool output and user-supplied values verbatim.

- The common Docker definition builds and installs DREAMPlace, OpenROAD-3D,
  OpenSTA-3D, Yosys 0.58, Mt-KaHyPar, hMETIS and HotSpot. The final layer adds
  only checksum-allowlisted contest runtimes staged by
  `docker/stage_release_assets.sh`; large wheel inputs are bind-mounted during
  the build and do not survive as duplicate image layers.
- `OpenROAD-3D/openroad` and its nested `src/sta` are independently maintained
  Git repositories pinned by submodule commits. `make openroad` incrementally
  rebuilds the release image through Docker/BuildKit. Maintained runs select
  only the image-installed OpenROAD/OpenSTA/Yosys binaries; host build outputs
  and `USE_LOCAL_OPENROAD` are not supported execution paths.
- TCAD25 has explicit `direct` and `external` execution contracts. Direct mode
  invokes the restricted AppImage in `open3dbench:release` when the supplied
  Gurobi license permits containers. External mode runs the same CPU binary on
  a node-locked licensed host, records hashes/runtime, and lets the release
  image validate and adopt `solution.txt`. The `bp_fe` CPU path was validated
  through HotSpot on 2026-08-12; the runner never silently falls back from GPU
  to CPU.
- The root `python3 -m open3dbench` thin orchestrator accepts one compact YAML
  configuration, resolves MoL/LoL algorithm choices, and writes fixed per-stage
  directories plus `run.json`. It supports RTL or checkpoint input. RTL
  synthesis and initial floorplanning use the existing ORFS design
  configuration, then produce one portable `design_params.json` shared by LoL
  and MoL. The file uses paths relative to `Place-3D`; per-design contest JSON
  is not a frontend dependency.
- Every registered RTL case now has one catalogued 3-D synthesis contract.
  The runner, rather than a case-specific wrapper makefile, forces flat
  standard-cell synthesis, disables RTLMP, passes ABC an explicit clock period
  in picoseconds, and preserves only audited hard-macro wrapper modules. The
  resolved Make values and OpenDB hard-macro inventory are written to
  `synthesis_metrics.json`; known inventories fail fast on mismatch. The
  contract participates in the input digest, so a policy or unit change cannot
  reuse an older ORFS work tree.
- `00_input` through `95_thermal` publish content-addressed stage manifests with
  relative artifact paths, configuration/upstream/implementation signatures,
  source/tool/container provenance, UTC start/completion timestamps, measured
  wall-clock runtime, and SHA-256 validation. The root `run.json` repeats each
  stage runtime. `--resume` reuses only a signature- and artifact-identical stage. Placement through report
  publish self-contained, validated two-die `checkpoint.json` files with an
  explicit upstream chain; thermal publishes a small final checkpoint that
  references report. OpenROAD Tcl and ng45
  LEF/Liberty/timing collateral participate in the implementation signature.
- `00_input` plus `10_floorplan` are an explicit immutable frontend prefix.
  `run --to floorplan` prepares it once and `run --frontend-prefix` verifies
  stage signatures, source/tool/container identity and every artifact hash
  before atomically copying it into a new run and starting at partition. The
  group-ablation driver accepts `--base-config`, so registered designs such as
  Ariane can reuse that prefix without inheriting the BP_FE baseline.
- `10_floorplan` runs the maintained partition-independent I/O policy. It places
  every external pin on the bottom-die perimeter by randomizing legal PPL
  slots, running a temporary full-design DREAMPlace placement on the source 2-D
  outline, affine-projecting those sinks into the registered smaller 3-D core,
  rebuilding rows/tracks on that target outline, and rerunning PPL. The target
  proxy may overlap because it guides pins only; it is never published as a
  legal placement. The final BPin geometry is preserved when the placement is
  split; top has no external BTerms. Each catalog case owns one explicit,
  round-number 3-D footprint in `open3dbench.benchmarks`; the eight RTL cases
  use 0.44--0.54 of their source 2-D area, including `bp_fe` 800x600 to 600x400.
- DREAMPlace DEFs cross an explicit macro-anchor adapter. The raw result is
  retained as `30_placement/dreamplace.raw.def`; the canonical `design.def`
  preserves DREAMPlace's lower-left coordinates and writes block macros as
  `N/R0`. Die initialization rejects macro overlap explicitly because standard
  OpenROAD placement checks do not reliably make macro-to-macro overlap fatal.
- The PDK macro contract uses one per-side halo through Concurrent DREAMPlace
  global placement, per-die macro/cell legalization, and OpenROAD backend
  acceptance/keepout. ng45 defaults to 10um per side (20um macro-body spacing)
  and ASAP7 to 2um per side (4um spacing). `placement.macro_halo_um` is the
  explicit user override; effective values and `top-level-pdk` provenance are
  recorded in placement parameters and manifests.
- LoL partition outputs use the physical bottom/top core boxes as placement
  fence regions, including the exact top virtual-die offset. The partition
  optimizer therefore cannot place macros legally in the die outline but
  outside the OpenROAD core. TP-GNN consumes the current run's
  target-projected `10_floorplan/design.def` and matching parameters; it does not depend on
  an ignored, pre-generated result directory.
- `30_placement/die_init/{bottom,top}` contains independently readable
  ODB/DEF/Verilog checkpoints, die-specific masters, warning-free OpenDP/check
  results, the selected PDK macro halo converted to a soft PDN constraint,
  and the exact bottom BPin geometry. Coordinate projection is detected from
  the placed instances: LoL's side-by-side coordinates are projected into each
  die, while MoL's already-local coordinates are retained unchanged; mixed
  coordinates are rejected. LoL block macros are moved to the nearest
  non-overlapping core location when projection is required. If nearest-first
  greedily fragments a reduced core, die init rolls back the complete macro set
  and applies deterministic largest-first shelf packing instead of weakening
  the maintained two-halo macro-body separation. The manifest records whether that fallback
  was used, displacement, track alignment and successful macro-spacing audit.
  After ownership is known, die init inserts
  tap/endcap cells before OpenDP and removes row fragments narrower than one
  die-local mesh period (bottom 10um, top 4um). OpenDP is allowed to search
  the complete core; its smaller implicit displacement window can otherwise
  strand standard cells beside a legal external-placer macro halo.
- Endcap insertion owns uniqueness at the source: OpenROAD TAP must insert at
  most one vertical endcap per `(row, side)` in one `place_endcaps` pass, even
  when partially overlapping polygon outline/hole edges revisit that endpoint.
  Do not unlock fixed physical cells or delete duplicates after legalization.
  The `tap.endcap_overlapping_edges` regression checks uniqueness, both row
  endpoints, placement legality, and ripup/reinsertion on cut-row geometry.
- `40_pdn` consumes those validated physical cells and independently validates
  VDD/VSS on both ODBs without inserting them again. The retained `ng45`
  baseline uses local M1/M4/M7 on both dies. The PDN-stage copy removes the
  soft placement halo, fills only same-net trunk gaps clipped at macro
  boundaries, and leaves the immutable die-init checkpoint unchanged. Both platforms use independent
  ideal VDD/VSS sources per die and require nonempty special wires plus fatal
  VDD/VSS `check_power_grid` success. Vias removed by PDNGEN at obstructions are
  recorded separately from true electrical disconnects.
- `pdk: ng45` resolves to the OpenROAD `nangate45` platform and its frontend
  defaults through `open3dbench/pdks.py`; it is no longer a label-only field.
- `pdk: asap7` is a peer platform generated from pinned ORFS collateral, not a
  branch of the ng45 scripts. It retains native M1-M9 on both independent die
  ODBs, uses the upstream M1/M2/M5/M6 PDN and OpenRCX rules, and supplies its
  own cells, rows, I/O layers, RC layers, HBT M9 grid, Liberty units and
  FakeRAMs through the same small PDK contract. FakeRAM Y origins are snapped
  to the 0.048um M4 track lattice; narrow macros also receive the separate M5
  PDN phase snap. Core partition, placement, HBT, OPT, CTS, route, report and
  thermal algorithms contain no ASAP7-name branch.
- The default ASAP7 BP_FE LoL path is validated end to end through HotSpot.
  Its full M1-M9 route has zero overflow, unrouted nets and antenna violations,
  with one bounded bottom-die DRC and zero top-die DRC. Final timing/power use
  the same continuous-passive assembly contract as ng45, with no HBT adapter.
- LoL hMETIS and Mt-KaHyPar run as direct processes inside the release
  container. The unified flow must not depend on a nested Docker daemon.
- The learned LoL partition baselines now have explicit device contracts:
  TP-GNN, PyG min-cut and PyG max-cut request CUDA, while hMETIS and
  Mt-KaHyPar remain CPU methods. TP-GNN uses differentiable `index_add`
  message aggregation because the image's PyTorch 1.7 cuSPARSE path is not
  supported on Ada GPUs. Fresh `bp_fe` RTL-to-partition runs for all three GPU
  methods passed artifact validation and same-seed assignment hashes matched.
- The maintained TP-GNN configuration is timing-driven. Before partitioning,
  the unified runner performs placement-estimated OpenROAD STA on the
  partition-independent DREAMPlace I/O prototype, using the exact Liberty set
  from `design_params.json` and ideal pre-CTS clocks. It extracts constrained
  path slack, slew, delay and source-distance features, records the report and
  CSV in `20_partition/timing_input`, and fails instead of silently reverting
  to physical-only TP-GNN if the report has no constrained paths, no WNS, no
  observed instances, or only zero-valued features. This early STA is a
  partition feature provider, not a post-OPT QoR report; it uses placement RC
  and no SPEF or global routing.
- TP-GNN's paper-aligned and fast profiles additionally require at least 90%
  instance hierarchy coverage. Projected-position coverage and hierarchy
  coverage are separate evidence: anonymous gates from a flat netlist are not
  silently assigned an `unknown` hierarchy and reported as a faithful TP-GNN
  run. The report records the measured ratio and rejects deficient inputs.
- MoL partition and placement settings live under
  `config/{partition,placement}/mol`; the old mixed benchmark configuration is
  not a dependency of the unified entry point.
- OpenROAD-3D and HotSpot are available in the common runtime.
- `OpenROAD-3D/openroad` is based on official OpenROAD commit
  `790a4a35df35f7f95a65335ca571cf8bf2981dd2`, with the retained Open3D changes
  documented in `OpenROAD-3D/openroad/OPEN3D_SOURCE.md` and
  `OpenROAD-3D/OPENROAD_COMPATIBILITY_AUDIT.md`. The vendored ABC Makefile,
  PPL core files, and FLUTE tables are build assets required by this repository's
  offline source-copy build.
- The flow handles official OpenROAD's current IO placer interface (the removed
  `-random` option is no longer passed) and falls back to Bash's built-in
  `time` when GNU `/usr/bin/time` is unavailable.
- `Place-3D/src/api.py` defines immutable file handoffs for partition,
  placement, and HBT planning; `Place-3D/scripts/verify.sh` checks shell entry
  points, runtime imports, and the unified LoL/MoL unit suite.
- Every partition method retains its native CSV and the root runner publishes
  one unit-explicit `partition_metrics.json` with common count, balance, and
  cut fields. Method-native areas remain labeled as such until a physical-area
  evaluator derives square-micron values from the canonical LEF/DEF.

### LoL

- Canonical input preparation and conversion for ICCAD 2022/2023 formats.
- A maintained partition-then-place flow with swappable partition methods,
  concurrent two-die DREAMPlace, HBT generation, official evaluation, overlap
  checks, staged outputs, and `run_manifest.json`.
- Concurrent DREAMPlace uses one backend-aware density policy for every design,
  not a per-case lookup table. The default base is 0.60, derived from a 0.75
  post-backend ceiling with 25% HBT/OPT/CTS area-growth reserve. Each die uses
  the base unless its measured utilization is higher, in which case the fence
  model selects `min(1.0, utilization + 0.01)`. The resolved values must be
  recorded in the placement manifest; explicit experiments may override only
  the base through `placement.target_density`.
- A maintained joint/True-3D contest flow for released contest packages,
  including TCAD25/CUPID, with conversion and evaluation.
- Small extension interfaces for partitioners, fixed-partition placers, and HBT
  planners. Joint/True-3D methods intentionally use `Placer` without a fake
  partition result.
- An experimental movable-HBT Concurrent DREAMPlace baseline splits each
  cross-die net around one shared HBT, optimizes bottom cells, top cells, and
  HBTs with independent density fields, then passes ideal HBT coordinates to
  the existing grid legalizer. Its padded HBT density geometry comes directly
  from the selected PDK HBT-grid JSON; it no longer reads a packaged contest `.input` or inherits that
  format's unrelated terminal pitch.
- `default` and `inflated` benchmark variants.

### MoL

- GNN, min-cut, and max-cut macro partition choices.
- A durable MoL partition assignment/manifest that can be reused unchanged by
  multiple placement baselines.
- `mol-analytical` placement with prototype placement, macro refinement,
  legalization, and bottom-die cell placement.
- `mol-tiling` placement with skyline-style macro packing and bottom-die cell
  placement.
- A shared Concurrent DREAMPlace adapter that consumes the same fixed MoL
  partition as the two released placers.
- Final DEF generation for downstream backend evaluation.

### Backend and evaluation

- OpenROAD-based backend PPA evaluation for LoL and MoL placements.
- Optional die-by-die GRT/DRT with per-net layer constraints.
- Explicit HBT terminal insertion, routing, merge-back, fixed-via creation, and
  final DRC checking.
- HotSpot-based thermal evaluation.
- Support for packaged reproducibility DEFs and custom MoL DEF export.

### Important implementation boundaries found in the current code

- `Place-3D` now gives LoL and MoL the same stage-level Python surface, but the
  handoff is not yet the canonical root Design IR. The partition assignment is
  now shared, while contest input remains a LoL adapter detail.
- MoL partitioning is extracted from placement and the analytical, tiling, and
  Concurrent DREAMPlace baselines consume an explicit fixed partition. Its
  semantics still differ intentionally: standard cells stay on the logic die
  while eligible memory macros are partitioned.
- `Place-3D/src/hbt/assignment.py` provides one DBU-based post-placement HBT
  algorithm layer for LoL and MoL. It implements optimal-region and centroid
  ideals, exact placement-produced and fixed ideals, deterministic nearest-free
  legalization, a routing-aware combined-HPWL greedy method, and global
  min-cost matching. All methods consume the same PDK grid/site problem and
  emit common displacement and combined-HPWL metrics. The two-DEF adapter has
  been exercised on 563-crossing LoL and 952-crossing MoL `bp_fe` placements.
- Track-aware greedy keeps the exact deterministic objective and candidate
  order while caching escape-site conflicts incrementally and evaluating only
  newly expanded KD-tree candidates. The 25,452-HBT Ariane133 regression is
  bit-for-bit identical to the previous assignment and reduces assignment
  runtime from 198.97 s to 70.03 s; it is an algorithmic complexity fix, not a
  candidate cap or quality fallback.
- The movable-HBT placer is a research baseline, not yet a full independent HBT
  legalization benchmark: DREAMPlace optimizes continuous HBT ideals, then the
  existing planner performs contest-grid legalization.
- The two-ODB materializer models HBT masters as `CLASS COVER` in the ng45 LEF.
  This is intentional: OpenROAD detailed placement must legalize ordinary cells
  without consuming a row site or moving the shared cross-die HBT anchor. The
  post-PDN materializer now preserves both existing ODBs, discovers the unique
  driver of every crossing from those ODBs, inserts the same fixed COVER anchor
  into each die, and adds a source/sink boundary BTerm/BPin on the original
  local net. Physical HBT ITerms remain disconnected so the marker cannot add a
  second local driver.
- `50_hbt` is one shared LoL/MoL runner stage. The user selects one
  `hbt.algorithm`: `optimal-nearest`, `centroid-nearest`, `from-placement`,
  `greedy-hpwl`, `track-aware-greedy`, `global-mincost`, or `fixed`. The stage emits `hbt.csv`, the
  canonical name map, both post-PDN ODBs, and a timed logical assembly. The
  assembly inserts `HBT_B2T_TIMING`/`HBT_T2B_TIMING` Liberty cells and must link
  and pass setup/hold STA before the checkpoint is marked `ready_for_opt`.
- Before assignment, `50_hbt` reads the actual platform `make_tracks.tcl` and
  rejects an HBT lattice unless its landing-center X/Y phases match the local
  landing layer on both dies (M10 for `ng45`, M9 for `asap7`) and both HBT
  pitches are integer multiples of those track pitches. The machine-readable
  result is `hbt_grid_alignment.json` and is a required content-addressed
  artifact.
- `docs/HBT_OPT_HANDOFF.md` defines the OPT boundary. Global timing consumes the
  assembly view; die-local physical optimization consumes the two ODBs and
  translates values through the stable boundary ports in `hbt.csv`.
  `open3dbench.backend.opt_handoff.load_opt_handoff` is the typed entry point;
  OPT implementations must not independently reinterpret stage files.
- HBT materialization preserves the original OpenDB net name and repairs only
  Verilog aliases that OpenROAD did not export. The repair distinguishes a
  vector select such as `bus[3]` from the different flat escaped identifier
  `\bus[3] `, and it runs again after every OPT export. Declared-but-disconnected
  HBT ports are rejected before assembly construction.
- `open3dbench.backend.three_d_opt` implements the shared LoL/MoL `60_opt`
  baseline and exposes only `opt.algorithm: openroad`. Its only timing view is
  the continuous passive assembly formed by joining both die RC halves with a
  `3 ohm / 0.6 fF` HBT pi section. Required time comes from the passive near
  node; arrival and slew come from the passive far node; source load includes
  the remote die through that same network. Each die then runs native
  `repair_design`, setup repair, and detailed legalization exactly once, in
  bottom-then-top order. The post-pass passive checkpoint is published as
  `ready_for_cts: true`; OPT has no iteration-count or convergence knobs.
- OPT still creates a directional-placeholder assembly internally so OpenROAD
  can project both ODB placements and serialize the two separate net halves as
  placement-estimated SPEF. It is RC-construction machinery only: it does not
  report timing, create boundary budgets, control convergence, select the best
  iteration, or provide the checkpoint validation view. Both validation Tcl
  paths replay the passive assembly.
- OpenSTA now exports arbitrary detailed-RC node ground/coupling/pin
  capacitance, downstream capacitance, path resistance and Elmore delay. A
  single-master-clock cutpoint query additionally reports rise/fall
  arrival/slew/required/slack without inserting a graph vertex. Every passive
  snapshot records near/far results in `passive_boundary_timing.tsv`. The audited
  4162-HBT `bp_fe` snapshot produced all 16648 rows; 24 rows correspond
  to six already-`-INF` drivers and two finite clock-fall rows have no required
  time. Large adapter deltas are real remote-load feedback (for example 397
  far loads behind `HBT_T2B_7732`), not HBT-RC delay.
- `open3dbench.backend.passive_reconstruction` independently rebuilds both die
  STA views from the final passive cutpoint snapshot and compares one worst
  rise/fall setup path per endpoint against the continuous passive assembly.
  The audited `bp_fe` result matched all 11225 cross-die endpoint/
  transition states, but failed the 50 ps arrival/slack and 5 ps required hard
  limits: maximum errors were 66.51/66.70/6.82 ps. The local view is about
  10.0 ps early and 10.5 ps optimistic in slack on average. Therefore
  the scalar cutpoint transfer is accepted as an optimization-guidance
  approximation, while the continuous passive assembly remains authoritative
  for every published global metric and is not a signoff view.
- Every physical crossing omitted by a partitioner is formalized before HBT
  assignment. The runner derives endpoint ownership from both die DEFs, marks
  crossings that are SDC clocks, and writes an augmented stage-local crossing
  CSV; this covers bottom-only external I/O whose source or sink cells landed
  entirely on top while keeping the original partition result immutable.
  Pre-CTS OPT keeps each unbuffered die clock ideal and gives the receiving
  die-local view an HBT boundary clock. Post-CTS passive STA propagates one
  master clock through both physical trees and the continuous HBT RC network.
- Pre-route die repair and global assembly STA both use placement-estimated RC;
  OPT temporarily serializes the flat estimate as SPEF so each pair of
  HBT-separated D_NETs can be merged losslessly. This is not OpenRCX extraction
  or signoff RC. OPT projects both ODB placements into one timing-only assembly
  OpenDB with overlapping tier coordinates. Both dies use local metal3/metal5
  for signal/clock RC estimation. Pre-CTS hold is report-only by design.
- Global STA publication is guarded by a Liberty coverage check plus OpenSTA
  `check_setup` for unclocked registers, invalid generated clocks,
  unconstrained endpoints and combinational loops. The ng45 top platform's
  nominal Liberty is missing six inverter cells, so the platform loader selects
  the strict-superset Liberty variant and rejects any remaining unmodelled leaf
  instance. Nets with multiple BTerms are protected from resizer buffering,
  which otherwise creates output/HBT feedback loops during Verilog export.
- The authoritative global timing graph includes placement-estimated intra-die
  wire RC, bottom I/O geometry, and passive HBT RC. A 4162-HBT `bp_fe`
  single-pass OPT reduced WNS/TNS from -149.374/-150227.152 ns to
  -1.265/-67.970 ns while keeping hold WNS at +0.091 ns. This remains pre-route
  rather than signoff because
  routing/coupling are absent, clocks are ideal, and ng45 HBT data is
  provisional. The path reconstruction audit is in
  `docs/PASSIVE_RECONSTRUCTION_20260817.md`.
- Per-die setup repair uses OpenROAD's default move sequence (unbuffer, size,
  pin swap, buffer, clone, and split load), with only multi-BTerm HBT boundary
  nets protected. Every OpenROAD warning is written to a reviewed JSON sidecar;
  unknown warnings, ambiguous clock periods, and `STA-1010` fail the stage.
- Physical OPT/CTS/routing and local STA consumers share a port-aware SDC
  projector backed by both DEF inventories and the HBT handoff. Remote reset
  exception endpoints require a verified direct-net HBT mapping or a unique
  non-inverting buffer chain proved by native OpenDB/Liberty data; numeric I/O
  budgets stay on the owning die. Native OpenDB port/net/direction checks and
  per-command/hash audit sidecars replace blanket missing-port/STA-0471 warning
  allowances. The authoritative assembly SDC remains unchanged. See
  `docs/DIE_SDC_PROJECTION.md` for the intentionally bounded Tcl subset.
- `open3dbench.backend.three_d_cts` implements the shared LoL/MoL `70_cts`
  transition. The maintained baseline runs top and bottom TritonCTS exactly
  once each. The completed top tree is embedded in the continuous passive
  assembly so its visible capacitance can be exposed as a temporary load at
  the bottom HBT coordinate. The built bottom leaf is transferred to the
  formal boundary BTerm/COVER anchor and the temporary cell is deleted before
  export. Global setup/hold/skew are measured only with propagated clocks on
  the same continuous passive HBT network used by OPT. Skew includes only the
  primary clock and each register's active edge; unclocked functional-latch
  gates remain explicit completeness exceptions, not zero-arrival sinks. CTS does not perform
  setup repair, gate sizing, pin swapping, cloning or iterative root-buffer
  calibration; the configured skew target is reported rather than forced.
  After tree construction, the default only reports propagated-clock hold and
  does not mutate data paths. Explicit `repair_hold` mode gives bottom and top
  at most one hold-only repair. Every clock buffer/net, HBT anchor and
  multi-BTerm boundary net is dont-touch; each repair is bounded to one
  internal pass, 1% additional buffers and 75% utilization. Residual hold is
  retained as a metric instead of triggering an unbounded closure loop. The
  passive all-HBT boundary snapshot is rebuilt after bottom hold repair before
  top repair. With hold repair disabled, the completed two-tree passive
  assembly is published directly; full data-boundary export and a duplicate
  final assembly/STA pass are skipped. This remains
  placement-estimated-RC CTS, not route-extracted signoff timing.
  Post-CTS data-boundary transfer is symmetric in analysis mode: every HBT
  exports near/far max/min rise/fall state, and the local SDCs carry matching
  `set_input/output_delay -max/-min` plus max/min input transitions. A
  25,452-HBT Ariane133 replay produced and parsed all 203,616 cutpoint rows;
  both die ODBs accepted the generated SDCs with zero OpenROAD warnings. This
  closes the former bug where local hold repair could not see a remote-die
  earliest arrival/required constraint.
  Coordinated OpenROAD CTS is mandatory in the maintained full backend. The
  former `ideal` pass-through control was removed because it produced an
  unbuffered multi-thousand-sink clock and did not represent a usable physical
  implementation path.
- `open3dbench.backend.three_d_route` implements the shared `80_route`
  transition. It consumes only the CTS checkpoint and runs native FastRoute
  plus TritonRoute independently in local M2-M10 on both dies. The fixed
  HBT COVER is pinless; its BTerm is the sole die-local endpoint. Publication
  requires every signal/HBT endpoint connected, zero antenna violations,
  preserved PDN/NDR state, die-owned layers and a strict warning audit. DRC is
  published as a bounded-run metric with an explicit `drc_clean` flag. The
  route checkpoint carries both routed ODBs and is marked
  `ready_for_extract`; assembly is copied unchanged and no SPEF is generated.
  The maintained top GRT baseline uses two congestion iterations, which
  removes guide overflow on the audited `bp_fe` LoL case.
  Routing schedules both die GRTs before either DRT. A standalone `global-route`
  command validates the historical CTS manifest and publishes a GRT-only
  sub-checkpoint/metrics without changing root routing status. Per-die GRT
  database, guides, audit state and logs are hash-validated for DRT continuation;
  GRT success never implies `ready_for_extract`. The original per-die wall-time
  budget covers GRT plus DRT, not an independent full allowance for each.
- `open3dbench.backend.post_route` implements `90_report`. It runs OpenRCX on
  each routed ODB and preserves a normalized die-local SPEF for local reports.
  Final STA does not load the directional HBT Liberty adapter. Instead, a
  second normalization retains both halves of every boundary net (including
  constant-driven sources), flattens both routed RC networks, merges each pair
  into one `OPEN3D_HBT_LINK_*` D_NET, and inserts the same `0.3 fF - 3 ohm -
  0.3 fF` pi section used by OPT/CTS. The final Verilog contains no
  `HBT_TIMING_*` cell, and vectorless power is reported on this continuous
  passive view, so source switching power includes the remote routed load.
  Publication requires exact `bottom D_NET + top D_NET - HBT link` coverage,
  all boundary pins, a complete timing graph, real bottom/top paths in both
  cross-die reports, zero adapter cells, empty global `check_setup`, and zero
  unexpected standalone OpenSTA warnings. The audited 4,162-HBT regression
  produces 31,603 final D_NETs,
  setup WNS/TNS -1.973538/-3047.011746 ns, hold WNS/TNS
  -0.107110/-179.583457 ns, and 0.152 W vectorless total power. The larger
  violation relative to the former adapter view is expected remote routed-load
  feedback, not a fixed HBT delay change.
- `open3dbench.backend.thermal` implements `95_thermal`. It consumes the
  report's per-instance vectorless power (0.1 primary-input activity), decodes
  timing-only names back to the physical DEF identities, distributes macro
  power by LEF-area overlap on aligned 10x10 grids, and requires exact power
  conservation. The warning-free HotSpot steady-state ng45 baseline reports
  321.42 K (48.27 C) for the fresh LoL `bp_fe` run. Stack materials and HBT
  power remain explicitly provisional.
- Resizer and CTS cells are not allowed to leave a stage with unconnected
  power pins. OPT and CTS reconnect every POWER/GROUND ITerm to the die-local
  VDD/VSS net after insertion and require `floating_pg_iterms = 0`; routing
  checks the same invariant at entry without repairing its input.
- Bottom and top standard-cell LEFs use the same native local M1/M2 pin
  geometry and differ only in master suffixes. Die flipping or mirroring is an
  assembly-coordinate transform; it must never renumber a die-local routing
  stack. This keeps pin access, track rules and RC extraction identical across
  the two independently implemented ODBs.
- The former combined-stack `open3dbench.release_smoke` gate performed fresh,
  unsymlinked RTL
  runs through all eight stages for both integrations. The first contract-clean
  final regression produced 16 valid manifests and ten physical checkpoints:
  LoL reached 0.078268 ns global skew with setup/hold WNS
  -0.569017/-0.015155 ns; MoL reached 0.083043 ns with
  -0.236974/+0.013380 ns. Both met the 0.10 ns
  target with zero unexpected warnings, an empty assembly `check_setup`, and an
  exact same-run placement-to-CTS checkpoint chain. A second `--resume` pass
  left all 16 manifest hashes unchanged. Those metrics predate the native
  per-die layer correction and are historical evidence only.
- Native-stack qualification covers ng45 M10 and ASAP7 through
  OpenRCX/OpenSTA and HotSpot. ng45 uses M2-M10 routing with M1/M4/M7 PDN;
  ASAP7 uses M2-M9 routing with M1/M2/M5/M6 PDN. Both preserve CTS NDRs,
  reach zero unrouted nets and zero antenna violations on qualified paths, and
  publish bounded DRC rather than hiding it. Authoritative post-route views use
  one passive stitched SPEF and contain zero HBT adapter cells.
- Die-by-die GRT/DRT, guide validation, HBT net splitting, and terminal-to-via
  finalization are the most mature 3D backend extensions.
- The maintained ng45 router preserves CTS NDR bindings on both independent
  local stacks and records input/stripped/preserved counts. Both dies require complete routing,
  zero antenna violations, and strict layer ownership; bounded-run DRC is a
  reportable quality metric rather than a stage-success condition.
- Detailed routing is deliberately bounded by `routing.max_iterations`, whose
  release default is 5, and `routing.die_timeout_seconds`, whose release
  default is 10,800 seconds (three hours) for each independently routed die.
  Hitting the iteration limit preserves the routed checkpoint, records the
  actual DRC count and `drc_clean` flag, and continues through extraction and
  STA. Hitting the wall-time limit terminates that die, writes a structured
  routing failure, and marks the run failed; the runner must not continue
  indefinitely or claim that a bounded result is DRC-clean when it is not.
- OpenRCX `RCX-0111` loop warnings are classified as expected only when the
  corresponding bounded die route has nonzero DRC. They remain visible as
  `bounded-route-rc-loop` records. Matching OpenSTA `STA-1650` warnings for
  unresolved SPEF loop nodes are retained as `bounded-route-spef-loop-node`;
  every other unknown RCX or STA warning is still fatal.
- A source-mounted `open3dbench:release` timing-driven TP-GNN `bp_fe` run has
  validated this bounded policy through `90_report`: exactly five DRT passes
  on each die, 335 bottom plus 1,328 top residual DRC, zero overflow,
  antenna and unrouted nets, 100% routed-net extraction coverage, empty global
  `check_setup`, and post-route assembly setup/hold WNS -0.769882/+0.055111 ns.
  All ten stage manifests were reusable; the run is explicitly not DRC-clean
  signoff.
- The common backend does not select LoL/MoL scripts. OPT, coordinated CTS,
  bounded two-die routing, per-die OpenRCX, assembly STA and HotSpot consume the
  same checkpoint contracts for both integrations. A heterogeneous MoL top
  die with zero movable cells records routability refinement as an explicit
  no-op; its SRAMs, HBT anchors, PDN, CTS, routing and extraction remain active.
- The maintained MoL `bp_fe` reference (`mincut`, `analytical`,
  `track-aware-greedy`, OpenROAD OPT/CTS/routing, OpenRCX/STA and HotSpot) has
  completed all eleven stages in one release image. It produced 952 HBTs, zero
  unrouted/antenna/overflow counts, 1,039 bounded residual DRC, 100% extraction
  coverage, setup/hold WNS -0.32614/+0.06720 ns and 47.95 C peak temperature,
  with no unexpected warnings.
- `open3dbench.release_ablation` fixes one representative LoL and MoL path,
  replaces one partition/placement/HBT/CTS component at a time, copies only the immutable prefix before
  that component, and executes the unchanged suffix. Its incremental summary
  preserves earlier results across interrupted batches. Atomic stages remove
  only their own incomplete generated directory on `--resume`; routing retains
  its separately validated per-die completion markers.

## Known Gaps

- Best-so-far retention requires explicit root/task/evaluator authorization and
  currently supports scalar `score` tasks only. Multi-objective campaigns need an
  explicit reporting trace; unknown failures and unresolved dependencies stay pinned.

- Die-SDC projection supports the maintained literal Tcl subset, not arbitrary
  Tcl programs. It rejects ambiguous selectors and remote exceptions requiring
  unproven traversal through buffer/logic cones; direct BTerm-to-HBT net
  identity and uniquely traced, Liberty-proved non-inverting buffer chains
  are supported transfer proofs. Conditional port-presence guards
  retain local-query semantics and do not establish whole-design equivalence.
- Legacy partial routing directories without the GRT sub-checkpoint hashes
  cannot be imported into the new two-phase resume path. Preserve them and
  select a new output directory; do not fabricate provenance for old artifacts.
- The RTL-to-bottom-only two-pass I/O-to-partition-to-placement-to-two-die-DB-
  to-independent-per-die-PDN slice is validated on `bp_fe` for all 19 non-TCAD
  pairs: five LoL partitioners with two LoL placers and three MoL partitioners
  with three MoL placers. The gate starts from registered RTL and validates all
  five content-addressed early stages plus placement/PDN checkpoints. The new
  HBT checkpoint has additionally been validated through assembly STA on
  551/563-crossing LoL runs, a 962-crossing MoL run, and a 20,035-crossing LoL
  max-cut stress run; the entire 19-pair matrix has not yet been extended
  through HBT.
- The current PDN baseline does not model power HBTs or cross-die IR drop. Its
  independent ideal sources validate physical grid construction and local
  connectivity only. Signal HBT assignment now preserves that saved PDN state;
  power HBT construction remains separate work.
- TCAD25 now has a unified joint-result adapter that publishes the canonical
  partition, unsuffixed placement DEF and placement-produced HBT ideals. Its
  real licensed end-to-end release regression remains a release gate until the
  dated run is recorded below.
- The exhaustive 19-pair release gate currently uses `bp_fe`; the other 12
  catalogued chip designs have frontend/contract coverage but not the same full
  algorithm matrix regression.
- KLayout is not yet packaged in the final Docker image, so final GDS-oriented
  validation still depends on the server environment.
- The standard `do-2D_flow` convenience target assumes synthesis artifacts
  already exist; use the normal `finish` dependency DAG or run `synth` first.
- Build steps can modify tracked OpenTimer generated binaries/configuration;
  generated build products need a clear source-versus-artifact policy.
- The checksum-allowlisted TCAD25/CUPID runtime is included in the release
  image, but its customer Gurobi license is never embedded. The launcher mounts
  a supplied container-enabled license read-only. A
  node-locked Academic license is explicitly supported through the hashed host
  handoff because Gurobi rejects that license inside containers.
- TODO: qualify a container-native license path for the existing TCAD25
  AppImage, which contains Gurobi 9.5.2.  First add a short release-image
  license preflight for a user-supplied Academic WLS credential and record the
  result without storing its secrets.  If the currently issued WLS credential
  does not accept this end-of-support client, evaluate an Educational
  Institution floating Token Server with the server outside Docker and the
  TCAD25 container as a client.  Until one path is qualified, TCAD25 remains an
  optional licensed track with the external `solution.txt` handoff rather than
  a mandatory one-container release gate.
- The current HBT site generator intentionally has no standard-cell/PDN
  blockage projection because pinless COVER anchors and the local M10/M10 or
  M9/M9 landing boundary are allowed to overlap lower physical implementation.
  A future PDK requiring
  a silicon keepout must publish that keepout through the platform grid instead
  of adding algorithm-specific exclusions.
- The main layer, row, cell, HBT-grid, timing-unit, and routing-window
  assumptions are now platform data for ng45/ASAP7. Remaining
  Nangate45-specific via construction and any implicit 2000 DBU/um assumptions
  still require audit before another technology family is added.
- ASAP7 BP_FE LoL now has a complete one-component-at-a-time full-flow
  ablation: five partitioners, two fixed-partition placers and seven HBT
  policies all pass the eleven stages through HotSpot in the release image.
  All 14 cases have zero unrouted signal nets and zero antenna violations;
  bounded residual DRC is retained as a result.  ASAP7 BP_FE MoL now also has
  a qualified mincut/concurrent-DREAMPlace/global-mincost reference plus 12
  one-component-at-a-time full-flow cases: three partitioners, three placers
  and six HBT policies.  Every MoL case passes all eleven stages with zero
  unrouted signal nets and zero antenna violations.  These are not Cartesian
  products.  The shared 3ohm/0.6fF HBT model and PDK-neutral thermal stack
  remain provisional rather than characterized signoff data.  The compact
  ledgers are `release/asap7_bp_fe_ablation_results.json` and
  `release/asap7_bp_fe_mol_ablation_results.json`.
- The HBT-to-OPT interface, iterative die-by-die OPT coordinator, and
  coordinated two-die CTS baseline are now versioned and validated. The ng45
  HBT timing numbers remain provisional rather than characterized signoff data.
- Assembly STA intentionally uses the same placement-estimated RC as die-local
  OPT instead of pre-route SPEF. A clean `check_setup` plus close assembly/die
  WNS proves the pre-CTS post-place model is internally consistent; it must not
  be labeled routed or signoff timing because actual routes, coupling,
  routed-clock parasitics and characterized HBT data are still absent.
- The pre-CTS passive crossing-clock handoff currently requires exactly one
  unambiguous top-level clock. Multi-clock benchmarks need an explicit
  HBT-boundary-to-master-clock field in the handoff contract.
- Flat synthesis removes most standard-cell RTL hierarchy from ordinary DEF
  names. The maintained algorithm is therefore explicitly named
  `tp-gnn-flat`: it neither fabricates nor consumes RTL hierarchy and performs
  distance-ordered contraction on the flat gate graph.
- Existing metric collectors contain hard-coded method/design lists and do not
  emit one versioned metric schema.
- Concurrent DREAMPlace is the first placer to consume the explicit PDK macro
  halo before backend handoff. Other external placers still rely on the audited
  `die_init` projection/repair path when they cannot express that constraint.

## Roadmap

### P0: Freeze facts and define contracts

- Add versioned schemas for experiment config, run manifest, canonical placement
  handoff, HBT data, and normalized metrics.
- Define common Python protocols for `Prepare`, `Partition`, `Place`,
  `HBTPlan`, `Legalize`, `Backend`, and `Evaluate` stages.
- Decide canonical instance/tier naming, coordinate units, orientation rules,
  HBT representation, and metric units.
- Add schema validation and small fixture-based contract tests.
- Create one tiny Nangate45_3D smoke design and golden adapter artifacts before
  refactoring existing flows.
- Add a capability matrix that marks every method/stage as implemented,
  adapter-only, experimental, or planned.

### P1: Build the IR, platform model, and evaluators

- Implement the canonical Design IR and importers for LoL contest input,
  `Place-3D` MoL LEF/DEF, and OpenROAD DEF/ODB checkpoints.
- Create `platforms/nangate45_3d/platform.yaml`; move DBU, layer windows,
  tracks, sites, libraries, HBT geometry, pitch, and via names behind the
  platform API.
- Implement validators and versioned metric schemas for partition, placement,
  HBT, True-3D, OPT, CTS, and routing.
- Implement stage-only and full-flow evaluation protocol descriptors.

### P2: Modularize the four frontend algorithm tracks

- Preserve and harden the completed `Place-3D` foundation: LoL partitioners use
  the common stage contract; MoL partitioning emits a durable assignment; and
  the analytical, tiling, and Concurrent DREAMPlace baselines consume that
  assignment without repartitioning.
- Extend the explicit PDK macro halo contract from Concurrent DREAMPlace to
  each external placer that can represent the constraint;
  retain backend repair only as a measured compatibility fallback.
- Generalize the current `Place-3D` file handoffs into the versioned root Design
  IR without erasing LoL/MoL eligibility and capacity differences.
- Preserve and expand the implemented unified `50_hbt` stage: all DBU-based
  idealizers/legalizers feed the same post-PDN COVER-anchor, boundary-BTerm and
  assembly-timing materializer. Add a platform keepout source only when the PDK
  model requires physical exclusion around a COVER anchor.
- Wrap existing contest/TCAD25 flows as `True3DPlacer` adapters.

### P3: Add unified orchestration

- Extend the implemented thin root CLI beyond its current plan/run/list,
  resume/dry-run, fixed run layout, and root-manifest slice.
- Validate and harden the implemented adapters for the existing LoL
  partition/placement flows and MoL analytical/tiling flows.
- Replace coarse legacy backend invocations with atomic stage adapters and
  complete per-stage manifests and canonical metrics.
- Keep existing scripts as compatibility wrappers around the orchestrator where
  practical.

The first CLI slice should support `validate`, `list`, and stage-only
`partition`/`hbt` runs before attempting the entire backend. This provides
useful, testable infrastructure early.

### P4: Normalize backend handoff and routing

- Extend die-SDC projection only with explicit semantics and regression tests;
  retain strict unknown-object failures and unchanged assembly constraints.
- Qualify GRT-to-DRT checkpoint equivalence on both maintained PDKs beyond the
  scheduling/hash unit tests and the current real-design GRT replay checks.
- Make OpenROAD-3D consume the canonical placed-design artifact directly.
- Remove the manual MoL evaluation-pack export step for generated runs while
  preserving packaged paper inputs as an explicit source type.
- Consolidate integration-specific backend launchers behind one backend adapter;
  retain specialized Tcl/Python implementation where physics requires it.
- Emit one versioned metrics schema with consistent provenance and units.
- Convert Nangate45-specific routing constants to platform queries/environment
  generated from `platform.yaml`.
- Preserve the existing die-by-die GRT/DRT and HBT finalization as the initial
  strong routing baseline, with standard monolithic routing as a comparison.

### P5: Implement reportable 3D OPT and CTS stages

- Replace the current explicit provisional HBT R/C baseline with characterized
  ng45 data while retaining the versioned assembly/boundary interface.
- Extend the implemented global-STA/per-die-OpenROAD baseline with additional
  cross-die-aware optimization algorithms and post-route extracted parasitics.
- Preserve and expand the implemented formal-clock-HBT, independent per-die
  TritonCTS, and propagated assembly STA baseline.
  Add multi-clock mapping and additional coordinated/useful-skew algorithms.
- Record clock HBT count, skew, latency, insertion delay, WNS/TNS, buffer count,
  area, power, legality, runtime, and downstream routed timing.

### P6: Verification and reproducibility

- Extend the opt-in compact historical reading and guarded retirement policy to
  other task families only after defining their trace and native checkpoint pins;
  preserve independent fresh-evaluation evidence and legacy task compatibility.

- Add CPU-only contract/unit tests and GPU-enabled placement smoke tests.
- Preserve the implemented fresh RTL-to-CTS LoL/MoL release smoke and extend it
  when another maintained clock or integration family is added.
- Extend the implemented resume/checksum/provenance checks with explicit
  failure-manifest coverage.
- Ensure build outputs do not dirty tracked source files, or document and enforce
  the chosen generated-artifact policy.
- Add CI tiers that reflect data and binary availability without pretending that
  restricted contest packages are public dependencies.

### P7: Deprecate legacy coupling and qualify peer PDKs

- Update user documentation to lead with the unified CLI.
- Mark direct integration-specific orchestration scripts as internal or legacy.
- Remove path-copy conventions only after compatibility tests show equivalent
  outputs and metrics.
- Complete the Nangate45-name/constant audit and keep ASAP7-3D as a peer
  platform package without branching core algorithms by PDK name. Add future
  PDKs only through the same explicit platform contract and generation path.
- Extend ASAP7 validation from the frozen default BP_FE path to component
  substitutions only after the peer platform remains stable; calibrate its HBT
  and thermal stack separately from algorithm regression.

## Development Rules for Agents and Contributors

1. Read this file, the relevant component README, and nearby tests before editing.
2. Treat the current working tree as user-owned. Do not discard unrelated or
   generated changes without explicit approval.
3. Do not claim a roadmap item is implemented merely because a wrapper exists.
   Update **Implemented Today**, **Known Gaps**, and **Roadmap** together when the
   architecture materially changes.
4. Prefer a shared schema or adapter over new cross-directory hard-coded paths.
5. Keep algorithms independent from orchestration and filesystem layout.
6. Preserve reproducibility inputs. Never overwrite packaged evaluation data by
   default; generated runs must use a separate, identifiable location.
7. Do not commit downloaded benchmarks, restricted contest binaries, credentials,
   large run outputs, or machine-specific absolute paths.
8. Commands that can overwrite a run must require an explicit flag. Resume must
   validate the prior manifest and input checksums.
9. Stage success means required outputs exist and pass validation, not merely that
   a subprocess returned zero.
10. Tests should be proportional to the change: unit tests for algorithms and
    schemas, contract tests for adapters, and smoke/end-to-end tests for flow
    wiring. GPU-dependent tests must be clearly marked.
11. Keep MoL and LoL comparisons scientifically honest: record seeds, variants,
    tool versions, evaluator source, and any fallback or manual intervention.
12. When adding a new method, document its inputs, outputs, dependencies,
    determinism, supported integrations, and metric provenance.
13. Build and run maintained tools only inside Docker. Source may be mounted
    for orchestration and editing, but OpenROAD, OpenSTA, Yosys, DREAMPlace and
    other compiled executables must come from the selected image. Incremental
    tool rebuilds update that image rather than publishing host build paths.

## Definition of Done for the Unified Flow

The first unification milestone is complete when a clean checkout can use one
documented root command to run and compare: (1) LoL and MoL partition methods,
(2) LoL and MoL fixed-partition placers, (3) at least three HBT legalization
baselines on identical ideals, and (4) a LoL True-3D placer. Each produces the
same versioned run layout, canonical handoff, normalized stage metrics, optional
OPT/CTS/routing full-flow metrics, and auditable manifest without manual file
copying or editing integration-specific scripts. Nangate45_3D is the only PDK
required for this milestone.
