# Frozen post-GRT optimization tasks

This task family implements multi-objective evolution on top of the post-GRT
matrix scheduler. It does not use the legacy global task's scalar timing score.
`grt_bench.optimization` defines the task and source verifier;
`grt_bench.archive` owns transactional candidate history and parent selection.
The shipped prompt is `global_tasks/lol_post_grt_pareto/instruction.md`.

## Objectives and acceptance

The physical endpoint, case inputs, three paired seeds, case weights, floors,
timing weights, power limit, evaluator digest and source baseline are frozen into
one task SHA-256 identity.
Changing the prompt or permitted components creates a different task as well.

| Objective | Calculation | Direction |
| --- | --- | --- |
| `timing` | 0.7 x normalized WNS deficit + 0.3 x normalized TNS deficit | Minimize |
| `overflow` | Normalized total overflow from both dies' GRT reports | Minimize |
| `temperature_peak_k` | Normalized peak temperature across 200 active silicon tiles | Minimize |

WNS and TNS deficits remain separate raw measurements, both in ns:
`max(0, -setup WNS)` and `max(0, -setup TNS)`. `power_w` remains the total cell
power across both dies at activity 0.1, in W. It is a constraint, not a fourth
objective. For **every case**, require:

`mean(candidate power across paired seeds) <= 1.10 * mean(baseline power across paired seeds)`.

This uses raw watts without a floor. It compares means, not the mean of seed
ratios or the worst seed. Equality passes. A zero-power baseline permits only
zero candidate power. A gain in another case cannot offset a violation. Every
generation uses the original frozen baseline, so the allowance does not compound
through parents. Complete verified candidates that exceed the limit remain in
history as `feasible: false`; they cannot enter the frontier or become parents.

Task schema v2 requires three `objectives` and a `scoring` object. Its default
`timing_weights` are `wns_deficit_ns: 0.7` and `tns_deficit_ns: 0.3`, with
`power_max_ratio: 1.10`. Both weights must be positive and sum to one. These are
host-frozen settings, not agent tuning knobs. Changing them creates a new task.
The optimization policy is `paired-ratio-weighted-timing-power-cap-v2`.

Temperature is the maximum reported tile temperature across both dies (10 x 10
tiles per die), measured separately for each case/seed. It is not a maximum over
seeds or cases, nor an estimate of unresolved sub-tile hotspots. Other stack
layers are excluded. Per-die average temperatures remain diagnostic fields.
Timing and power use the same post-GRT passive-HBT assembly and GRT parasitics. This is
a GRT estimate, not detailed-route signoff. Positive timing slack is clipped to
zero deficit; it does not add a separate reward for extra timing margin.

The endpoint is `post-grt-passive-hbt-peak-v2`; the producer records
`thermal_statistic: maximum-silicon-tile-temperature`. The verifier independently
reduces the validated raw silicon tiles and rejects the old `temperature_k`
metric and mean-temperature receipts. This changes the task/evaluator identity.
Existing v1 tasks and archives remain immutable and require their original
evaluator. The ordinary evaluator-fix replay cannot change objectives; it does
not migrate a v1 reference into a qualified v2 reference.

For raw metric `m`, case `c` and paired seed `s`, compute
`r[c,s,m] = (candidate[c,s,m] + floor[m]) / (baseline[c,s,m] + floor[m])`.
Average the seed ratios within each case, then take the frozen case-weighted
mean. Combine only the normalized WNS and TNS results using the frozen timing
weights. Keep overflow and peak temperature separate. The five raw-metric ratios
remain diagnostics; the power ratio does not determine constraint eligibility.
Report per-case samples, mean, sample standard deviation and worst sample, plus
each case's baseline mean power, candidate mean, limit and excess in W.
The three-objective baseline vector is exactly `(1,1,1)`.

Candidate A dominates B only if A is no worse on all three aggregate objectives
and strictly better on at least one. Only power-feasible candidates participate.
Every feasible non-dominated source candidate stays on the frontier, including
distinct sources with equal objective vectors.
An optimization task requires `epsilon=0`; the standalone `frontier`
utility still supports epsilon boxes. A candidate can improve one objective
while worsening another and still enter the frontier. There is no further
per-case veto or noise significance threshold. Use the seed diagnostics when
choosing parents. Failed or incomplete matrices produce no Pareto point.

All evaluated candidates and their immutable source snapshots remain available
after domination. Only the training frontier is eligible for **new** parent
selections. Selecting a parent records its ID, reason, timestamp and frontier
digest. Parallel children can finish after their selected parent becomes
dominated. Held-out/auxiliary results cannot update this archive; the existing
host-controlled final-candidate test release remains required.

## Fixed input boundary

For campaigns starting at partition, `frontend_inputs` binds each case's
original `00_input` and `10_floorplan` stage-manifest hashes and producer image
into the task identity. These physical inputs are immutable benchmark fixtures.
`grt_bench.frozen_frontend` validates all artifacts and imports the two stages
byte-for-byte; it preserves their original provenance and starts candidate
execution at partition. Candidate edits must never regenerate these fixtures
or silently change the I/O geometry. Partition, placement and later stages still
use normal candidate source and runtime signatures. A missing or changed fixture
fails verification instead of triggering regeneration.

Adding this boundary to an existing task requires a new task identity. The
original baseline may be qualified through physical-result replay only after
every pinned input and physical artifact is independently verified. Candidate
sources are re-frozen under the new task's policy; identical source inventories
can have different candidate IDs because the task policy is part of that ID.
Old task receipts and failed attempts remain unchanged.

## Prompt-defined modification scope

The prompt begins with explicit YAML front matter, followed by ordinary prose:

```yaml
---
components: [partition, placement, hbt, timing_opt, cts, grt]
---
Improve post-GRT timing, overflow and peak temperature within the power limit.
```

Select any nonempty subset. Optional `files: [...]` narrows that selection to
specific existing files. Natural-language prose describes the research intent;
it cannot expand filesystem permissions. The host enumerates the baseline's
regular files and freezes the exact allowed file list in the task:

| Component | Current editable boundary |
| --- | --- |
| `partition` | Python methods plus the named algorithm functions in `lol/core.py` and `lol/constraints.py` |
| `placement` | Concurrent placement policy, DREAMPlace objective/optimizer/initialization modules and selected Python operator wrappers |
| `hbt` | `Place-3D/src/hbt/assignment.py`: ideal positions and legal-site assignment |
| `timing_opt` | C/C++ sources and headers under `OpenROAD-3D/openroad/src/rsz/src/` and `include/` |
| `cts` | C/C++ sources and headers under `OpenROAD-3D/openroad/src/cts/src/` and `include/` |
| `grt` | C/C++ sources and headers under `OpenROAD-3D/openroad/src/grt/src/` and `include/` |

The exact boundaries live in `grt_bench/algorithm_scope.py`. Placement includes
`BasicPlace`, `NonLinearPlace`, `PlaceObj`, the Nesterov optimizer,
`HBTConcurrentPlace`, selected functions in `concurrent_lol`, and Python wrappers
for wirelength, density, congestion, legalization and detailed placement.
Only named function bodies may change in mixed algorithm/contract modules:
`build_unique_nets`, `move_delta_cut`, `macro_balanced_fixed_assignment`,
`project_wirelength_pin_pos`, `backend_aware_target_density`,
`_apply_cross_die_net_weights` and `_apply_die_aware_initialization`.
Their signatures, decorators, imports and surrounding code remain protected.

The worker mounts the frozen candidate's Python algorithm files over their exact
installed DREAMPlace locations. Producer and independent verifier both check
the complete module list and file hashes. A separate Python cache directory
prevents stale image bytecode from bypassing source edits. Merely mounting the
repository would not change the installed launcher's imports.
Native DREAMPlace C++/CUDA kernels remain fixed; extending them requires a
verified incremental image builder. `PlaceDB`, input/output adapters, `Params`,
`EvalMetrics` and the launcher are also protected.
The rafa1 qualification image does not install the optional independent-set
matching (ISM) detailed-placement operator. Its source wrapper exists, but the
fixed flow does not invoke it; a wrapper edit does not make its missing native
runtime available. This qualification covers the active concurrent placer.
HBT edits require an exact file grant; its adapters, serialization and physical
materialization remain protected. CTS/GRT source edits trigger the same verified
installed-image build as Resizer edits, including when they are the only selected
native component. Tcl/SWIG interfaces, tests and build files are outside these
C++ boundaries. Task configuration files, case inputs, PDKs, seeds, evaluator,
stage control, OpenSTA, build recipes and verifier code stay immutable.
The task still permits exactly one OPT pass. Resizer code is shared with other
OpenROAD calls; its effects are evaluated across the whole fixed downstream
flow rather than claimed to be isolated to one invocation.

The default prompt enables all six components. It permits early parameter tuning
but asks the agent to explore algorithm changes and other stages when progress
stalls. Narrower prompts can still select a subset. Updating the component scope
or prompt creates a new task identity; earlier qualification receipts do not
qualify the expanded task or a new HBT/CTS/GRT candidate.

## Verifier and trust boundary

Before submission **and again before worker execution**, the host verifies:

1. Candidate bytes match the content-addressed snapshot; every ancestor manifest
   hashes to its identity and the chain reaches the frozen root baseline.
2. Every lineage edge uses the task-derived policy; both edge and cumulative
   diffs stay inside the exact allowed files. Creates/deletes, altered symlinks
   and executable bits, missing files and invalid Python syntax fail admission.
   For mixed modules, AST comparison also rejects changes outside the named
   function bodies on every ancestry edge.
3. Stored scope/build flags match recomputation, including inherited native
   changes. The candidate comes from a recorded frontier-parent selection.
4. Runtime verification binds native source to an installed candidate image and
   Python algorithms to their actual installed module paths. Independent
   physical/result verification still checks every case and
   seed, stage/checkpoint hashes, OPT count, timing completeness and power/thermal
   consistency before a worker can return a verified result.

Archive admission accepts only the host queue's successful training jobs. It
compares the queue result, worker receipt and independent verifier receipt,
checks task/plan/analysis/legality hashes and rechecks analysis artifact hashes.
It recomputes aggregation and per-case power eligibility from these verified cell
metrics. Per-cell verification proves measurements; the host checks the power
constraint only after the full paired matrix is available. User-supplied
aggregate vectors or JSON claiming `status: verified` alone cannot be ingested.
The first complete baseline matrix becomes the immutable paired reference.
SQLite transactions serialize concurrent admissions and parent selections.

Host control files, queue tokens, archive databases and snapshot stores must not
be writable by candidate containers. This concise verifier establishes edit and
evaluation contracts; it does not prove arbitrary Python/C++ semantic equivalence
or detect every form of benchmark-specific overfitting. Such claims require
separate equivalence checks or review. Train/test division is chosen and frozen
by the host before task definition. The current five-case qualification inventory
is still not a committed scientific train/test split.

## Host workflow

Use a new control root/version for the new evaluator; do not rewrite the task or
receipts of an already running evaluation. Here `ROOT` denotes that root,
`BASELINE` an imported root source snapshot, and `BASE` a bound matrix JSON from
`prepare`. Input/config bindings and runtime provisioning follow
[POST_GRT_MATRIX.md](POST_GRT_MATRIX.md). Commands run from `AgenticBench`.

```bash
# 1. Freeze objectives, current evaluator, source baseline and prompt scope.
python -m grt_bench task-define "$ROOT" "$BASE" "$BASELINE" \
  global_tasks/lol_post_grt_pareto/instruction.md
# Set TASK to ROOT/control/tasks/<returned digest>.json.

# 2. Build/register the baseline runtime using the existing build workflow,
# then run its full train matrix through the existing scheduler/workers.
python -m grt_bench submit "$ROOT/control/queue.sqlite" "$TASK" \
  "$BASELINE" "$RESOURCES" --root "$ROOT"
python -m grt_bench archive-ingest "$ROOT" "$TASK" "$BASELINE"

# 3. Inspect the frontier and choose a promising parent explicitly.
python -m grt_bench archive-show "$ROOT" "$TASK"
python -m grt_bench archive-select "$ROOT" "$TASK" "$PARENT" \
  --reason 'Explore lower power while retaining this candidate’s timing gain'
python -m grt_bench candidate-checkout "$ROOT" "$TASK" "$WORKSPACE" \
  --selection "$SELECTION"
# The checkout receipt gives the prompt and exact allowed_files. Edit those files.

# 4. Freeze using the returned selection ID; register/build its runtime.
python -m grt_bench candidate-freeze "$ROOT" "$TASK" "$WORKSPACE" \
  --selection "$SELECTION"
python -m grt_bench candidate-verify "$ROOT" "$TASK" "$CANDIDATE"
python -m grt_bench submit "$ROOT/control/queue.sqlite" "$TASK" \
  "$CANDIDATE" "$RESOURCES" --root "$ROOT"

# 5. After all required train cells succeed, atomically update the archive.
python -m grt_bench archive-ingest "$ROOT" "$TASK" "$CANDIDATE"
```

`archive-ingest` is idempotent for identical evidence and rejects conflicting
evidence. Missing/failed/extra attempts are not silently dropped or retried.
`archive-show` returns evaluated rows, the current frontier, selections and child
lineage. The agent can use it to select a parent without a hard-coded preference
over the three objectives. The legacy scalar agent loop is not automatically
repurposed into a Pareto agent; these commands are the new iteration interface.
The [explicit round prompt contract](POST_GRT_AGENT_ROUNDS.md) combines parent
selection and editing in one invocation, with a bound host checkout tool and
structured final response. The prompt renderer does not spend the budget. The
server-side campaign controller launches each invocation and enforces the durable
model-call and evaluation limits, including the prompt review gate. Review is
required every round by default; `prompt_review: first_round` waives it for later
unrevised prompts only. Missing/failed cells are never retried unless a campaign
opts into the strictly classified `infrastructure_retry` policy
([agent adapters and campaign options](AGENT_ADAPTERS.md)).

Offline tests cover source/prompt policy violations, ancestry/build flag forgery,
worker and CLI admission, partial/failed/test matrices, altered receipts and raw
evidence, tradeoffs/equal vectors, concurrent insertion and in-flight selections.
They exercise the host lifecycle using synthetic worker evidence and do not claim
new native EDA qualification. Because evaluator identity covers these modules,
old qualified receipts cannot be relabeled as this task's reference.

## Historical v1 native qualification on rafa1

The frozen mean-temperature v1 task on 2026-10-13 completed a fresh five-case baseline and one actual
placement-policy edit, each with three paired seeds. All 30 cells passed the
independent physical/result verifier. The candidate changes
`BACKEND_AREA_GROWTH_FACTOR` from 1.25 to 1.30; every native parameter file records
the expected value. It reused the verified installed image without a C++ build.

The baseline took 54.71 minutes and the candidate took 64.08 minutes from batch
submission to the last verified result. The candidate rebuilt its signature-bound
input/floorplan prefixes; that setup is included in its batch time and each prefix
is shared across the case's three seeds. Both batches reached six GPU slots
(three per device) and ten live evaluations. GPU release was checked against
broker events and immutable placement checkpoints; backend containers used no GPU.

The candidate improved the aggregate TNS-deficit and overflow objectives while
worsening WNS deficit and slightly increasing power. It entered the exact Pareto
frontier alongside the baseline. Re-ingestion independently rechecked the raw
evidence and left the archived evaluations unchanged. A protected source edit
and an incomplete matrix were both rejected before archive admission.

See the [qualification record](../qualification/grt-20261013-pareto/README.md)
and [detailed report](../qualification/grt-20261013-pareto/native/final-audit/REPORT.md)
for the frozen identities, per-stage timings and receipts. This qualifies the
evaluation/selection/archive workflow; the scientific train/test split remains
unfrozen, and a new Resizer C++ candidate was not exercised by this experiment.
These receipts do not qualify the new peak-temperature v2 task. The v2 regression
suite covers hotspots on either die, exclusion of non-silicon layers, forged
mean values, legacy metric/statistic rejection and malformed tile coverage.
All 66 host regression tests passed. A read-only diagnostic check also validated
the peak reduction on all 30 preserved native maps, following their recorded
artifact hashes; see the [v2 statistic regression](../qualification/grt-20261013-peak/README.md).
The active rafa1 v1 task and archive were not changed by that check.
Those historical tasks used separate WNS and TNS objectives. The current task
uses weighted timing and a power constraint, and has not been deployed or
qualified by those runs. Its 94 offline regressions cover the scoring order,
power boundary, per-case checks, fixed-baseline ancestry, infeasible history and
parent exclusion. These tests do not claim a new native qualification.


## Paired stage suffixes (2026-10-15)

Tasks with an `incremental` contract compare the candidate's frozen file
inventory with its recorded parent's inventory. The earliest affected stage
sets the execution boundary. HBT assignment reuses input through PDN; placement
reuses input through partition. RepairSetup/RepairHold start at OPT, CTS sources
at CTS, and FastRoute kernels at GRT. Shared Resizer/GRT code and unknown
dependencies conservatively start at partition, because native APIs are also
consumed by partition timing extraction and HBT materialization. A combined edit
starts at the earliest consumer of any changed file. Reverted files count as
changes from the selected parent. If an ancestor modified an earlier component,
the baseline dependency graph cannot prove that component independent of later
code: replay conservatively starts at that earlier component. Repeated changes
within the same component keep its normal boundary.

The parent must have one complete, independently verified matrix for the same
task. Each child imports only the matching case and full seed vector. The proof
binds both source inventories, dependency drivers, task/configuration, installed
runtime lineage, original stage manifests and predecessor chain. Artifact hashes
and physical checkpoints are validated before copying, after copying, and by the
independent result consumer. Original source/runtime provenance is preserved;
no manifest hash is rewritten to make an incompatible checkpoint reusable. A
missing or inconsistent checkpoint fails closed.

The baseline and every candidate reuse the task-frozen input/floorplan prefix.
Evolution starts at partition; I/O preplacement is outside the optimization
scope. Analysis-only requalification retains explicit links to the original
physical runs. Suffix imports and the agent's reading library follow those
validated links instead of treating an analysis directory as a physical run.

CPU suffixes request zero GPUs from the start. Partition/placement suffixes use
the existing six GPU slots and release them before the CPU backend. The campaign
records each cell's actual allocation, source job, boundary and compatibility
proof. This feature is integrated into the campaign controller; a manually
submitted legacy CLI job does not infer a parent checkpoint on its own.

DREAMPlace now seeds Python, NumPy and PyTorch before reading the database on
both the ordinary I/O proxy and concurrent LoL entry paths. The maintained
adapters enable deterministic density accumulation. Native repeated runs on
both GPUs produced byte-identical DEFs for all five optimization cases, for both
I/O proxies and concurrent placement. The new image/source identity requires a
fresh baseline; old nondeterministic placements remain historical evidence.
See [the implementation and native qualification record](../qualification/grt-20261015-incremental/README.md).
