# Post-GRT matrix evaluator: three objectives with a power constraint

This is a new task family. Old timing/CTS-thermal authorities are not scoring
authorities for this endpoint. Native qualification is separate from offline
tests. Do not describe the six-case template as qualified or held-out training.

## Physical endpoint

Partition -> placement -> PDN -> HBT -> one OPT -> CTS -> both-die GRT ->
segment-estimated RC -> continuous-passive-HBT joint STA and cell power ->
HotSpot. No DRT or OpenRCX signoff is implied. GRT now retains actual route
segments, SPEF, DEF and Verilog in its hashed manifest. Old GRT outputs lack these
exports and must be regenerated in a new directory, not relabeled.

The GRT process runs `estimate_parasitics -global_routing` while its exact
router state is live and exports a hashed SPEF. `grt_bench.endpoint` consumes
these SPEFs, stitches them with physical HBT RC,
and uses one propagated-clock timing graph for WNS/TNS and power. Power activity
is 0.1; HotSpot has 200 equal-area silicon tiles, 318.15 K ambient and the
existing provisional shared stack. The `temperature_peak_k` objective is the
maximum of those 200 active silicon tiles, separately per case/seed. Stack
interface/package layers are excluded; this finite tile resolution does not
resolve sub-tile hotspots. Average temperatures remain diagnostics. The endpoint
identity is `post-grt-passive-hbt-peak-v2`; old mean-temperature v1 receipts retain
their original identity and evaluator. `grt_bench.verify` independently recomputes
raw metrics from reports, checks hashes, stage completion, OPT count, native
seed evidence, power conservation and temperature completeness. Neither CTS
metrics nor candidate-reported scores substitute for these measurements.

Boundary aliases come from each GRT SPEF's current BTerm-to-net mapping.
Post-placement buffering can change that mapping; reconnecting a boundary to
its historical pre-buffer net would corrupt the assembly timing graph.

The scoring view retains explicit max/min-delay checks on data-enabled latches.
Disabling their enable pins also removes their constrained data endpoints in
OpenSTA. The new view therefore keeps those checks, records live constrained
min/max path slacks for every acknowledged latch data pin, and rejects any
other missing-clock or unconstrained-endpoint warning. This does not create
artificial clocks or suppress a whole warning category.

A host may rescore an immutable completed GRT run after an evaluator-only fix.
The host may name and select a strict subset of unchanged cases while changing
evaluator identity. Every retained case, its weight/split, candidate, image,
inputs, configuration, seed vectors, objectives and the physical launch drivers
must match. Added or modified cases cannot use this replay path. Workers validate
all stage and GRT hashes, audit the original source,
mount old outputs read-only and write new analysis/results in a separate attempt.
Native execution time remains historical; the new receipt measures analysis
and verification separately. This is not reuse across changed algorithms or
images. Bind `rescore_sources` by new queue job ID and
`rescore_previous_framework` in the host environment; normal candidates use the
ordinary full-flow path.
Changing mean temperature to peak temperature changes the objective and is not
eligible for this evaluator-fix replay. Preserve the v1 references and qualify a
new task/reference before admitting peak-temperature results to evolution.
The three-objective policy likewise uses a new task/reference. Task schema v2
rejects old five-objective task files. The physical endpoint remains
`post-grt-passive-hbt-peak-v2`, since its raw measurements have not changed.
Old receipts and archives keep their original contracts; the evaluator-fix
replay cannot change timing weights or the power constraint.

## Task and replication

The host freezes task JSON with the evaluator implementation digest, each
case's config SHA and input-manifest digest.
`five_cases.template.json` is the active qualification inventory after Gemmini
was deferred by the user on 2026-10-13. It retains BP_FE ng45, BP ng45, Ariane133
ng45, SweRV ng45 and BP_FE ASAP7, each with three paired seeds.
`six_cases.template.json` retains the original inventory including Gemmini.
Both templates are unbound evaluation inventories, not a committed scientific
train/test split. Change splits before `prepare`.
The active five-case task gives every design/PDK case weight 1, so each contributes
20% to each aggregate objective. BP_FE/ng45 and BP_FE/ASAP7 are separate,
equally weighted cases. The six-case template is a retained legacy inventory,
not the active task. Weights never depend on measured performance.
The three default vectors use matching partition/placement seeds:
`(1001,1001)`, `(1002,1002)` and `(1003,1003)`. I/O stays 1. The retained
`routing: -1` field controls DRT and is unused at the post-GRT endpoint.
Tasks frozen before this change retain their original seeds and identities;
their baseline results cannot be relabeled as the new paired baseline.
The current OPT has no exposed stochastic seed API; inventing an `opt.seed`
would misrepresent replication and is rejected. A future stochastic OPT must
register its seed application and native evidence in a new contract revision.

Every required matrix cell must independently pass. Normalize each raw metric
within its case/seed pair as `(candidate + floor)/(reference + floor)`, then
average seeds and apply frozen case weights. The three objectives are weighted
timing (default: 0.7 normalized WNS deficit + 0.3 normalized TNS deficit), routing
overflow and peak temperature, all minimized. WNS/TNS remain separate raw metrics.
Power is a hard constraint: each case's mean raw power across paired seeds must
be at most 1.10 times its original baseline mean. Another case cannot offset a
violation. No floor or case weighting enters this check. The task freezes
`objectives`, `scoring.timing_weights` and `scoring.power_max_ratio`.
Publish per-case samples, standard deviations and worst samples, the aggregate,
and per-case power limits/excesses. Floors and epsilon are also fixed task values.
Epsilon boxes provide transitive dominance and deterministic duplicate handling.
Test/auxiliary outcomes cannot enter the evolution archive.
Frozen optimization tasks use exact dominance (`epsilon=0`) and retain every
feasible non-dominated source, including equal objective vectors. Over-limit
candidates remain in history but cannot become parents. Their persistent
archive, prompt scope and task-bound admission are documented in
[POST_GRT_OPTIMIZATION.md](POST_GRT_OPTIMIZATION.md).

## Candidate and build boundaries

The immutable store retains full source bytes, executable bits, parent digest
and scope audit. Trusted relative in-tree symlinks are preserved; candidate
symlink changes, escaping links and mode changes are rejected. Host-owned policy may
allow frontend algorithm directories and OpenROAD native algorithm directories;
control-plane, platform, input, HBT and evaluator paths remain protected.
This is a file-boundary verifier, not a proof of arbitrary algorithm semantics.
Native compilation uses the existing quota-limited, private-tree/shared-ccache
builder, single-flighted by native-source/image/recipe identity. A changed runtime
is packaged into a new pinned image and checked there. Python-only candidates
reuse the base image; a Python descendant of a native candidate reuses that
candidate's exact native runtime when the native source fingerprint matches.
Physical optimization uses candidate OpenROAD. The hierarchical scoring view
uses fixed image-installed standalone OpenSTA; its binary hash is retained.
OpenSTA source is protected from candidate edits. A future scoring-engine task
must define and qualify a separate contract. DREAMPlace native source edits
are rejected until an installed candidate runtime builder is registered.

The worker prepares each candidate/case's fixed input and I/O prefix once in a
locked SSD cache, then shares it across seed replicas. The cache identity binds
candidate, task, case, image and I/O seed. Every consumer uses the existing strict
`--frontend-prefix` validator, including full artifact hashes and signatures.
Failed preparations remain for diagnosis and are not silently retried. Set
`cache_frontend_prefix: false` in the host environment to request fresh execution.
An explicit host-bound prefix is also supported. Generalized
cross-image suffix reuse is not implemented and is not bypassed by copying
manifests. Compiler-cache reuse and physical-prefix reuse are separate.

## Scheduler and operation

Keep SQLite on the coordinator's local SSD. Register host CPU/memory/storage,
GPU IDs and build slots conservatively, then start the loopback-only queue
service. Remote workers connect by an SSH tunnel with separate host tokens.
Provision identical candidate snapshots and case-bound inputs on each worker;
The scheduler does not automatically transfer candidate/input data between hosts.

Each worker executes one job at a time; start multiple workers to fill admitted
slots. CLI submission interleaves cases by seed round so replicas do not occupy
all slots waiting for one case's shared prefix. Existing admitted jobs retain
their original order. Transactions reserve all resources atomically. Heartbeats cover source
preparation and native execution. On expiration, jobs become `uncertain` and
keep their reservation until a host-owned termination receipt reconciles them.
Cancellation stops admission and asks active workers to stop their named
container. No ambiguous execution is silently repeated. Queue completion is
distinct from independently verified scoring acceptance.

A Docker-client error is not termination proof. Workers remove and inspect the
named container before acknowledging failure. If the daemon cannot confirm
termination, the worker leaves its lease reserved for host reconciliation.

The revised rafa1 profile admits six concurrent GPU phases across two GPUs, with
`gpu_slots_per_device: 3`. Slots are assigned to the least occupied device;
CPU, memory and scratch limits remain independent admission constraints.
Omitting this field preserves exclusive-device scheduling for older hosts.
Frontend containers reserve 8 logical CPUs, 24 GiB memory and 40 GiB scratch.
They stop after placement and validate the four input-to-placement manifests,
native resume signatures and physical checkpoint. After Docker termination is
confirmed, the authenticated release operation atomically reduces the live
reservation to 4 CPUs, 16 GiB memory and zero GPUs. A new container without a
GPU mount validates the handoff again and starts PDN through GRT and scoring.
The original submitted resource request remains immutable; the database stores
the smaller current allocation separately. Expired or uncertain leases cannot
release resources without host reconciliation. Backend work cannot reacquire a
GPU, and the complete physical attempt retains its six-hour time budget.

`phase_resources.backend` in the host environment enables this mode. A pool of
12 workers lets queued GPU frontends start while previous cells run their CPU
backends. Six is the GPU-phase limit, not an artificial limit on CPU backend
concurrency. Total reservations remain capped at 64 CPUs, 192 GiB memory and
600 GiB scratch. A GPU slot is returned at the placement boundary even while
that cell's CPU backend is still running; dispatch then also requires enough
free CPU, memory and scratch capacity.
Physical algorithms retain the frozen four-thread setting; the container CPU
limit is a ceiling. The earlier two-worker profile remains preserved in its
original experiment receipt for measured comparison.
Source, queues, build caches and experiment outputs live on `$AGENTIC3D_DATA_ROOT`.
Image layers use the host's existing Docker daemon. Prefix and runtime receipts
are separate from measured physical-stage wall time.

`scripts.benchmark_grt_phased` waits for the preceding trial to finish and its
idle workers to be restored, then checks exact process identities before
replacing the service. It does not cancel preceding jobs. The scheduling
change receives a new evaluator/task identity, preserves the scoring modules,
and qualifies all 15 cells freshly from partition. Only validated input and
floorplan prefixes are adopted. It records queue/phase allocations and GPU
telemetry, publishes timing only for a complete verified matrix, and seals a
new reference without relabelling the old results.

Example commands from the AgenticBench directory:

```sh
python -m grt_bench prepare grt_bench/five_cases.template.json bindings.json /data/agentic3d/agentic3d-grt-20261012
python -m grt_bench register-host queue.sqlite rafa1 capacity.json
python -m grt_bench.service --database queue.sqlite --tokens worker-tokens.json
python -m grt_bench.worker --environment environment.json --host rafa1 --token worker-token.txt
python -m grt_bench submit queue.sqlite task.json CANDIDATE_SHA resources.json
python -m grt_bench status queue.sqlite
```

Environment JSON binds `root`, `framework`, `base_image`, and `cases` entries with
`config`, `manifest` and optional `frontend_prefix`. Store prepared tasks under
`control/tasks/<digest>.json`, runtime receipts under
`control/runtimes/<candidate>.json`, and candidate snapshots under `candidates`.
An explicit held-out release under `control/test-releases/<candidate>.json`
contains `candidate`, `task_digest`, and `frozen: true`. Do not feed held-out
results back into parent selection. All plans, logs and receipts are English.

Read `runs/<job>/result.json` only after the queue reports success. Its metrics
come from the independent verifier. The physical `output/run` stops at CTS;
`output/grt` and `output/analysis` own the new GRT endpoint. The older root run
summary is not the matrix score. A compact progress view is available with
`python -m scripts.grt_matrix_status ROOT`.

## Measuring a fresh matrix from partition

`python -m scripts.benchmark_grt_matrix PLAN.json` is a host-only, one-shot timing
driver. It waits for complete native qualification, creates a separately
identified candidate with byte-identical source, and reuses the verified
image/input/floorplan prefix. All partition-through-GRT stages and scoring run
again. The original resource queue admits the trial, with an explicit worker
count and per-cell budget; existing idle workers are handed over and restored.
The trial does not modify the reference or enter the Pareto archive.

The JSON plan binds `root`, a new `output` directory, `framework`, `trial_id`,
`qualification`, `qualification_controller_status`, `environment`,
`worker_processes`, `worker_count`, `host`, the private token **file path**, and
`resources`. The current rafa1 trial uses two workers and the same 16-CPU,
64-GiB, one-GPU cell budget as native qualification, retaining four physical
threads. A failed qualification prevents the timing run from launching.

`timing.json` and `TIMING.md` report submission-to-last-verified wall time,
first-partition-to-last-verified time, per-cell queue/startup/stage/verification
times and each case's three-seed minimum/median/maximum. The report rejects
analysis-only replays, reused partition stages, missing cells and failed cells.
Historical input/synthesis/floorplan and compile durations are not added to a
fresh trial. Setup outside the timed admission window has its own receipt.
Within-batch case spans include interleaving with other cases; neither they nor
summed worker durations establish a separate serial baseline or a speedup.

## Qualification still required

The 2026-10-13 five-case baseline/candidate native qualification covers the
historical mean-temperature v1 endpoint. It does not qualify peak-temperature v2.
Before formal searches: qualify the same new endpoint and matrix for reference
and candidate; verify at least two simultaneous native jobs; exercise failure,
cancellation and restart; qualify native image builds with unchanged and edited
source and record cache statistics. Offline fixtures never establish native
OpenROAD/HotSpot success or a speedup. Keep failed attempts and old references.
