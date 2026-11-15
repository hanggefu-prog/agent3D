# Small SSH cluster scheduler

`scripts/schedule_codesign_cluster.py` schedules a user-defined DAG without
changing physical tasks, candidate budgets, evaluators, algorithms, or seeds.
It uses longest remaining critical path priority and the earliest estimated
finish among currently admissible hosts. This is a deterministic list-scheduling
heuristic, **not a proof of minimum makespan**. Estimates are planning inputs,
not measured runtimes or timeout limits.

## Interface

Deploy the exact same script on the controller and every worker. Hosts need
Python 3.9+, Linux `/proc`, existing authenticated SSH, and tmux. GPU jobs also
need `nvidia-smi`. No credentials are copied or written by this tool. Use a
dedicated existing output root on each host, outside source and frozen references.

For a persistent server-local shard, set its single host's `ssh` field to
`"local"`; resource probes, dispatch, and reconciliation invoke that host's
configured Python/script directly, without SSH. A deployment tool can first use
the offline multi-host plan to assign case affinities, then generate one pinned
shard per server and start each controller in that server's tmux. Those shards
perform dynamic admission only within their own host; this is static cross-host
assignment, not live migration or globally dynamic rebalancing. The laptop may
disconnect after launch. No new server-to-server authorization is required.

```json
{
  "cluster_id": "codesign-example-v1",
  "hosts": [{
    "id": "worker-a",
    "ssh": "approved-worker-alias",
    "root": "/data/example/jobs",
    "python": "/data/example/venv/bin/python",
    "script": "/data/example/source/AgenticBench/scripts/schedule_codesign_cluster.py",
    "slots": 2,
    "slot_ids": ["01", "02"],
    "cpus": 64,
    "memory_mb": 131072,
    "disk_mb": 204800,
    "gpus": [{"index": 0, "memory_mb": 40000}, {"index": 1, "memory_mb": 40000}]
  }],
  "jobs": [{
    "id": "bp-prepare",
    "deps": [],
    "affinity_group": "bp",
    "hosts": ["worker-a"],
    "estimated_seconds": 1800,
    "resources": {"cpus": 32, "memory_mb": 65536, "disk_mb": 20480, "gpu_memory_mb": 12000},
    "command": ["{python}", "{root}/trusted_wrapper.py", "--case", "bp", "--phase", "prepare", "--gpu", "{gpu}", "--slot", "{slot}", "--cpus", "{cpus}", "--memory-mb", "{memory_mb}", "--output", "{job_dir}"]
  }]
}
```

All memory/disk quantities use MiB. `slots` is a positive integer; optional
`slot_ids` defaults to `01`, `02`, etc. A host's optional `speed` (default 1,
at least 1) divides job duration estimates. Host budgets should cover only the
capacity approved for this batch, not necessarily the whole machine.

`jobs[].hosts` restricts eligible hosts; one element pins a job. An optional
`affinity_group` binds all its jobs to the host chosen for its first dispatch.
For each case, prepare and both objective/reference jobs can share one affinity
group. The simplest deployment instead pins those jobs to the same host.
Do not create conflicting host restrictions inside an affinity group.
Jobs are independent unless their IDs appear in `deps`; dependencies must be
acyclic. Add separate wrapper jobs for data transfer if needed: the scheduler
does not copy reference data or infer filesystem dependencies across hosts.

Commands are **argv arrays**, never shell programs. Available placeholders are
`{root}`, `{python}`, `{script}`, `{job_id}`, `{cluster_id}`, `{job_dir}`, `{gpu}`,
`{slot}`, `{cpus}`, `{memory_mb}`, `{disk_mb}`, and `{gpu_memory_mb}` (plus host
fields). Literal braces must be escaped as `{{` and `}}`. Optional `cwd` uses
the same placeholders and defaults to `{job_dir}`. GPU-free jobs request
`gpu_memory_mb: 0` and receive `{gpu}=none`. Do not put keys/tokens in argv: the
full admitted command is retained as provenance.

```bash
# Offline: no SSH, files, models, or EDA. Also checks static resource feasibility.
python AgenticBench/scripts/schedule_codesign_cluster.py --config cluster.json --plan

# The controller itself should live in a persistent terminal/session.
python AgenticBench/scripts/schedule_codesign_cluster.py --config cluster.json \
  --run --state-dir /data/example/controller-state

# One reconcile/admission cycle; not a continuous supervisor.
python AgenticBench/scripts/schedule_codesign_cluster.py --config cluster.json \
  --run --state-dir /data/example/controller-state --once
```

## Admission and resource responsibility

New `run_scheduled_codesign.py` jobs default to `--retention best`. Use
`--retention full` for diagnostic runs, and explicitly add `--retention best`
to smoke-wrapper argv when using those in a large batch. The mode is recorded
per experiment and never edits a live immutable plan. See
[storage and build-cache retention](experiment_retention.md). Disk admission
thresholds are unchanged; cleanup is not permission to over-admit jobs.

Before dispatch the controller probes actual idle CPU over 0.5 seconds,
`MemAvailable`, root-disk free space, and GPU free memory. Existing GPU processes
do not disqualify a device. Unknown, stale, or failed probes are fail-closed.
Configured budgets and slots account for all admitted nonterminal jobs, including
unknown jobs after an SSH failure. Reservations are also deducted conservatively
from observed headroom, protecting queued starts and later peak allocations;
this may underfill a host when a running job's usage is already reflected in the
probe. Each worker rechecks actual headroom immediately before its command.

Among equally preferred allocations on one host, the scheduler first selects
the GPU with less memory reserved by this batch, then fewer reserved jobs,
then the numeric device index. This avoids unnecessary serialization on the
runtime's per-GPU lock. It still allows sharing an occupied GPU when capacity
permits; this tie-breaker does not change admission rules.

Admission is not a cgroup, GPU-memory quota, or cluster-wide reservation against
other users. **The trusted command wrapper must apply the admitted CPU/RAM limits
to Docker and use the selected GPU and slot.** Do not give a command hidden nested
parallelism that exceeds its reservation. Hardware already occupied by other work
is never stopped or modified. External workloads can still grow after admission;
insufficient measured headroom fails the command before launch, without retries.

This small scheduler owns one batch. Use exactly one controller state directory
for that batch; independent schedulers require disjoint approved budgets or an
external shared resource manager. The user supplies paths and SSH configuration;
these are trusted operator inputs, not an untrusted multi-tenant API.

## Persistence, recovery, and evidence

- `manifest.json` fixes config and scheduler-source SHA. Resume rejects changes.
- `state.json` contains pending/dispatching/queued/running/unknown/success/failed/
  blocked status, reservations, rendered specifications, and remote observations.
- `events.jsonl` retains English state transitions and admission probes.
- A central nonblocking `flock` rejects a second controller using the same state
  directory. Every remote job has its own dispatch and worker locks.
- Remote outputs are `<host.root>/<cluster_id>/jobs/<job.id>/`: immutable
  `spec.json`, one `started.json`, `console.log`, and final `result.json`.
- Detached tmux survives SSH disconnects. The worker executes its argv in the
  foreground; child stdout/stderr are retained unchanged in `console.log`.
- The controller persists dispatch intent **before** SSH. If acknowledgement is
  lost, it retains the reservation and queries the same job; it never relaunches
  automatically. Even a recorded specification with no worker/receipt requires
  investigation rather than a retry. This favors at-most-once execution over
  automatic recovery of work that might never have started.
- A failed job blocks all dependent jobs. Independent jobs may continue.
  An unknown job keeps dependents pending and its reservation occupied until a
  matching receipt or live session is observed. Stop/restart only the controller
  if necessary; it never kills worker jobs. Do not delete a ledger to get a retry.

`success` means the trusted command exited zero, **not** independently proven
placement legality, completed evaluation budget, improved timing, or thermal
validity. The wrapper must validate its own final receipts/metrics and return
nonzero for incomplete or invalid results. Model authorization and endpoint
qualification stay in that wrapper. This scheduler performs no model preflight.

The internal `--probe`, `--dispatch`, `--inspect`, and `--worker` modes are the SSH
transport, not additional user experiment interfaces. Script checksums must match
before any job is admitted. Changing code/config requires a new batch identity;
existing remote jobs and receipts are never overwritten by that change.

## Validation boundary

`tests/test_codesign_cluster.py` uses temporary directories and mocks for all SSH,
tmux, resource probes, and job processes. It checks priority, affinity, slots,
busy-GPU sharing, budgets, failed dependencies, immutable identity, locks,
at-most-once dispatch/worker execution, and SSH-loss reconciliation. Passing these
tests does not establish real model/EDA qualification or measured speedup.
