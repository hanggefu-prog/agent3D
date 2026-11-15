# Matched early-timing / thermal co-design batch (2026-09-16)

## Isolated manual workers and resource caps (2026-09-23)

An independent manual worker uses its own `WORK/source`, `WORK/config.yaml`,
`WORK/run`, `WORK/logs`, and scratch directory. Source
`scripts/manual_flow.sh init` from the selected frozen Agentic snapshot, not an
unrelated V2 checkout. `AGENTICBENCH_GPU` selects the visible device at init;
it neither reserves the device nor imposes a GPU memory quota. Shared-device
workers require an explicit, external admission policy and measured headroom.
The worker should export `AGENTICBENCH_ADMITTED_GPU` after its admission check.
When set, manual flow requires an exact match with the GPU selected at init or
saved in `WORK/gpu.txt` for later stages. A mismatch fails before Docker starts
or an existing stage log is rotated; it does not rewrite the workspace. Leaving
the guard unset retains the original standalone manual-flow behavior.

For each stage invocation, optionally export `MANUAL_FLOW_CPUS=4` and
`MANUAL_FLOW_MEMORY=32g`. The launcher passes Docker CPU and memory limits;
the memory-swap limit equals the memory limit, allowing no additional swap.
Unset values retain the prior uncapped launcher behavior. Explicit empty,
nonpositive, or malformed values fail before a container starts. These caps do
not modify the algorithm thread counts or other physical settings in the YAML.

The co-design batch scheduler is separate. By default it still requires an
approved, unoccupied GPU per physical evaluation. Set JSON `"share_gpu": true`
(or `--share-gpu` for a single experiment) to allow other existing GPU processes.
The existing free-device-memory, host-RAM, root-disk and output-disk thresholds
still apply. One batch-local GPU lock remains held for the full physical
evaluation, so this option does not run two native evaluations from the same
batch on one device. Different batches or external workers may share an approved
device; they still require host-level admission to avoid resource races. Docker
device visibility is not a GPU memory quota. The policy is recorded in the
immutable batch/run/evaluation contracts and cannot change during resume. Older
configs without the flag default to exclusive admission.
The positive integer JSON field `"min_gpu_free_mb"` (single-run flag
`--min-gpu-free-mb`) sets the free-memory admission threshold in MiB; its default
remains 16000. A worker pool may explicitly select 2048 as a minimum sanity
guard, but that does not qualify the algorithm's peak memory requirement. The
threshold is recorded alongside the sharing policy in immutable run and
evaluation identities. Neither field limits GPU allocation or changes physical
algorithm parameters, metrics, seeds, or thread counts.
Frozen physical references may be mounted read-only for inspection, but their
settings must pass the endpoint-specific qualification checks before adoption.
Do not rewrite historical artifact paths or substitute the physical reference
for a two-pass co-design reference implicitly.

This deployment snapshot retains the consolidated Agentic stage interfaces,
including timing extraction and authenticated split routing. New English
observability work in the separate V2 snapshot is not part of this snapshot;
do not copy overlapping V2 orchestrator modules over Agentic extensions.

## Case and budget entry points (2026-09-23)

The current launcher accepts `bp_fe`, `bp`, `ariane133`, and `swerv_wrapper`.
For a single method/objective/case run, use `run_codesign_experiment.py --case bp
--rounds N` together with its required frozen-input/runtime arguments. For a
server shard, set the JSON field `"rounds": N` in the config passed to
`launch_codesign_batch.py`. `rounds` is a positive integer **candidate final-attempt
budget per group**, defaults to 10, and includes failed final evaluations. JSON
booleans, strings and fractional numbers are rejected; bootstrap references are
separate. The printed plan reports each group's `final_budget` and their sum.

All four cases × two objectives × two methods produce 16 groups: `16 * N`
candidate final attempts and eight bootstrap references. Case/objective/method
subsets reduce these totals. Both methods remain on `gpt-5.6-sol`; Agentic3D's
inner limit remains at most three physical proxy evaluations per outer round.
The budget is part of the immutable run/batch configuration: `--resume` cannot
increase or otherwise change it. Use a new experiment identity for a new budget.

Adding `bp` to the entry point does not qualify any existing reference, change
physical settings, or adopt a new TP-GNN decoder. Every case must still pass the
unchanged endpoint-specific reference checks. The dated matrix below is retained
as the historical September 16 setup, not a claim that new experiments started.

## CTS-thermal replay interface correction (2026-09-17)

The newer `tp_gnn_placement_cts_thermal` endpoint supersedes the historical
full-route thermal endpoint for the current batch: two OPT passes, CTS,
placement-estimated cell power and HotSpot, with no routing/RCX.

A frozen-input replay records only the six executed stages, partition through
CTS, in `run.json`. It does not claim to have rerun input/floorplan. The metric
extractor now accepts precisely this suffix **only with an explicit trusted
`frozen_input_run`**. The evaluator and reference qualifier supply that argument
from the selected frozen input, not from candidate metadata. Both copied prefix
manifests must match the source byte-for-byte; source artifacts are hash-checked,
and copied artifacts are checked against those same records. The only content
exception is the copier's exact three-field relocation of `design_params.json`
(`def_input`, `verilog_input`, `sdc_input`); every other parameter is unchanged.
Case/platform mismatches, missing/failed/skipped physical stages, extra routing
stages, changed data and arbitrary path rewrites remain errors. Two-pass OPT,
CTS, power, HotSpot and all existing physical gates are unchanged.

Complete eight-stage CTS runs retain their existing interface. The legacy routed
thermal endpoint and the independent full-chain workspace verifier keep their
separate contracts. Historical failed result envelopes are not rewritten by
this fix: a read-only raw-artifact check is not an automatic reference adoption
or permission to restart a frozen experiment. A new frozen controller snapshot
or an explicitly audited migration is required before running with the fix.

This is a **new** experiment contract, not a continuation or relabeling of an old
five-evaluation STA run. User-selected matrix:

| Cases (ng45) | Methods | Endpoints | Candidate final attempts per group |
|---|---|---|---|
| swerv_wrapper, bp_fe, ariane133 | Codex, Agentic3D | early timing, thermal | 10 |

12 groups, **120 candidate final attempts**. Six bootstrap references are separate.
Both methods use `gpt-5.6-sol`, reasoning effort `xhigh`, the same frozen source,
five-file modification boundary and endpoint-specific trusted judge. Reference:
TP-GNN `global-2way` + Concurrent DREAMPlace. All fixed-seed settings remain owned
by the harness; no density/footprint/backend/gate relaxation to make a run pass.

Agentic3D uses enhanced-task-v1 / global-analyzer-only-v1: Local and Reflector
submit advice; only Analyzer publishes tasks/skills. Each group starts with fresh
maintained seeds and no imports. At most three cheap proxy evaluations per outer
round, first strict improvement; a final-only task consumes its single outer
evaluation, not an additional trial. Codex has no physical proxy evaluations.

Timing retains the 0.6 WNS-deficit + 0.4 TNS-deficit normalized cost, exactly two
OPT passes and complete constrained global STA. Thermal retains the maintained
full backend and mean physical-tile temperature in K, score reference/candidate.
It does **not** inherit the two-pass STA score or extra OPT passes. Thermal model,
10x10 grid and 318.15 K ambient are fixed. Complete routing/extraction/power and
thermal qualification remain mandatory. This is not a DRC-clean/signoff claim.

## Execution and isolation

Entry points:

- `scripts/launch_codesign_batch.py`: server-local shard planner / detached tmux
  supervisor. Different cases, objectives and methods run independently.
- `scripts/run_codesign_experiment.py`: one method/objective/case run with
  `--rounds N` final attempts (default 10).
- `scripts/codesign_runtime.py`: credential-free, network-disabled physical
  Docker evaluation, memory/CPU limits, approved GPU pool with shared locks.

Assign each case to **exactly one server**; its two objectives can run concurrently,
and the methods share one reference per objective. Different servers receive
separate manifests and the identical frozen source archive/image contract.
Within a server, all method/reference/input processes share batch GPU locks.
GPU reservation persists through a physical evaluation to bound total RAM and
CPU concurrency even after GPU placement finishes. Model calls can overlap
physical evaluations. No process belonging to another task is stopped or changed.

Default per-slot limits: 32 CPUs, low CPU shares, 96 GiB memory. Admission requires
112 GiB currently available host RAM, 16 GB GPU free memory and 40 GiB free output
disk. GPU IDs must be explicitly approved; idle utilization is not ownership.
These are conservative limits, not a guarantee that every design fits. The pool
should be reduced if another workload needs the memory/CPU. Existing reservations
and real resource usage must be rechecked immediately before deployment.

## Freeze and launch

1. Create a separate source checkout; apply the intended maintained code and new
   harness there, then commit **that isolated snapshot only**. Record base commit,
   overlay checksum and new snapshot commit. The evaluator clones Git HEAD, so a
   dirty checkout is rejected. Do not commit/stash/reset the shared worktree or
   include results, credentials or evolved knowledge in the snapshot. All tracked
   runtime source dependencies must be self-contained at container-visible paths.
2. Use a qualified release image, pin its full `sha256:` ID. The image must provide
   `/usr/bin/python3`, `/opt/dreamplace`, OpenROAD's passive-cutpoint STA contract,
   Yosys and HotSpot. Source/framework paths are mounted read-only; only the batch
   output root is writable. No home-directory/auth/key mount enters EDA containers.
3. Install the same Python controller dependencies and an authorized, working
   Codex configuration on each controller host. Do not copy credentials into a
   manifest or source tree. The launcher probes the requested model once before
   physical preparation; auth/service errors stop the shard without retries.
4. Write a server config (example below), inspect the printed plan, then launch:

```sh
python AgenticBench/scripts/launch_codesign_batch.py --config server.json
python AgenticBench/scripts/launch_codesign_batch.py --config server.json --launch
```

Example **template**, not an already deployed configuration:

```json
{
  "session": "codesign-20260916-amax2-v1",
  "source": "/home/example/agentic3d-runs/codesign-20260916-v1/source",
  "batch_root": "/home/example/agentic3d-runs/codesign-20260916-v1/results",
  "image": "sha256:REPLACE_WITH_64_CHARACTER_QUALIFIED_IMAGE_ID",
  "cases": ["swerv_wrapper", "ariane133"],
  "rounds": 10,
  "gpus": "6,7",
  "cpus": 32,
  "memory_gb": 96,
  "model": "gpt-5.6-sol",
  "codex_bin": "/home/example/.local/bin/codex",
  "prepare_only": false
}
```

The tentative second shard is bp_fe on ta2 (one newly checked nonreserved GPU),
not ta1/ta3 reserved GPUs or the amax2 GPU5 reference job. These assignments are
**provisional**, based on the September 16 resource audit, not a live reservation.
If API access is blocked, `prepare_only: true` is an explicit reference-only run;
never call that evolution. Moving from reference-only to evolution uses a new
explicit launch configuration, with validated reference/input adoption, not an
in-place mutation of an immutable batch manifest.

Inputs are freshly prepared per case. Both references must independently pass
their endpoint's gates and actual frontend-method/decoder checks before any
source-evolution model call. Failed thermal qualification does not block that
case's timing job or other cases; it does block both thermal methods for fairness.

## Accounting, restart and monitoring

Each physical attempt has a durable contract/ledger before it starts. Cached
results require identical candidate/source/image/input/reference/config identity.
Incomplete physical attempts stop for inspection; there is no silent retry or
extra final attempt. Candidate failures returned by the evaluator count toward
the configured final-attempt budget. Missing-result infrastructure failures stop the group, preserving
the attempted evaluation ledger for budget reconciliation. Never delete the ledger
to get a free retry. Model/reflection format repair cannot invoke physical runs.

Inspect `supervisor.log`, `status.json`, `<case>/<objective>/status.json`, method
`history.json` / `summary.json`, and each `physical/**/result.attempt.json`.
`tmux-started-not-yet-qualified` only confirms dispatch, not working optimization.

After real deployment, create a Codex thread heartbeat every 30 minutes. Check
tmux/controller/container state and actual new evaluation evidence, not just PIDs.
Switch to every three hours only after the launched method/endpoint combinations
have produced valid candidate results without repeated API/stage failures; blocked
groups must be clearly reported, not counted as healthy. Report meaningful changes,
failures and final summaries; stay quiet for unchanged state. Pause when the batch
is finished. No monitoring automation has been created merely by writing these
scripts.

## Current verification status

Local mocked integration covers both methods × both endpoints, three and ten attempts,
failed-attempt accounting, cached resume, endpoint-specific gates and role authority.
Runtime command tests cover GPU mapping, memory/network isolation, two-pass injection
only for timing, candidate-cache identity and refusal to repeat an incomplete run.
These tests do **not** constitute real Docker/STA/thermal qualification or measured
optimization gains. Remote deployment and end-to-end execution remain required.
