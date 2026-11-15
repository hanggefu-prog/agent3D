# Workspace-oriented global tasks

Version note: this page describes the generic framework and historical task
examples. The current four-case workspace driver supports explicit one-OPT
pre-CTS timing tasks, protected frontend-only edits, and independent clone
isolation; it is not an unrestricted whole-backend editor. See the
[task settings report](TASK_SETTINGS_REPORT.md) and
[harness design report](HARNESS_DESIGN_REPORT.md) for deployed-versus-generic
behavior and qualification status. Historical thermal tasks are not automatically
qualified by the new diagnostic thermal results.

Global tasks optimize an engineering outcome rather than one bounded function
or parameter record. They are deliberately separate from `local_tasks/` and
its ask/tell candidate schema.

```text
host-owned GlobalTaskSpec + round-0 user command
  -> Global Analyzer produces an Intervention (no file scope)
  -> one isolated Trial workspace per acting agent
  -> agent plans, edits, builds, runs and reads results
  -> source-and-artifact Submission
  -> separate trusted Verifier
  -> adaptive trusted Evaluator
  -> accepted state or another Intervention
```

## Trust boundary

The subject repository is the optimization surface. Inside its isolated
workspace the agent may change any implementation or configuration. Benchmark
instructions, frozen inputs, Skills, verifier definitions and scoring code are
read-only mounts or live only in the host/verifier environment. A protected
path is a control-plane boundary, not an algorithm allowlist.

Each trial uses an exclusive Git worktree and exclusive `build`, `runs`,
`logs`, `scratch`, and `submission` directories. Containers receive no Docker
socket, drop Linux capabilities, use a read-only root filesystem, and are
bounded by CPU, memory, PID, GPU and wall-time settings. Named GPUs and licenses
can be serialized with `global_bench.scheduler.FileResourcePool`; the CLI uses
`GlobalTrialScheduler` to enforce each task's concurrent-trial budget. No
mutable directory is shared across concurrent trials. Git history is mounted
read-only while the current trial's worktree metadata is mounted separately,
so ordinary status/diff/restore operations do not expose writable repository
state belonging to another trial.

## Problem and handoff contracts

`GlobalTaskSpec` defines the instruction, source revision, runtime resources,
read-only inputs, required artifacts, trusted verifier ID, immutable objective,
stage DAG and total budget. It intentionally has no `source_scope`,
`config_scope`, candidate program entry point, or fixed code-output field.
The objective names the evaluator profile, reference input, direction, primary
metric, and exact constants. This keeps scoring fixed while leaving the subject
algorithm space open.

Every task also declares an evaluator-owned `seed_policy`. The maintained LoL
contract fixes I/O placement to 1, partitioning to 1001, placement
initialization to 1000, and detailed routing to -1 (deterministic net order).
The evaluator rewrites these values into its resolved configuration, and the
verifier checks the seeds reported by the physical stages. Seeds are replication
controls, never optimization parameters.

`GlobalSearchContext` records the current accepted revision, objectives, hard
constraints, trusted stage artifacts, observations and remaining budget.
Global Analyzer returns an `Intervention` containing a hypothesis, target
stages, evaluation policy and execution budget. Target stages may widen replay
and focus the acting agent, but never restrict the editable subject files.
`WorkspaceGlobalAnalyzer` reuses the existing stateful read-only Codex runner,
while `GlobalRoundCoordinator` fixes the model → execute → impact resolve →
trusted evaluate order in code.

The acting agent may be an existing terminal-capable CLI. The harness needs
only its launch command and lifecycle; it does not redefine file or shell
tools. `DockerTrialRuntime` supplies the working directory and standard
environment variables and records the complete launch plan.

## Adaptive evaluation

Every stage lists its implementation path patterns and dependencies.
`ImpactResolver` maps the actual repository diff to directly affected stages.
Agent-claimed stages are unioned with that result and cannot narrow it. Unknown
files conservatively invalidate the complete DAG unless the task explicitly
chooses rejection.

- `quick` evaluates directly affected stages;
- `suffix` and `final` evaluate their downstream closure;
- `full` evaluates every stage.

Reuse is allowed only when upstream artifacts, resolved configuration,
implementation inputs, runtime identity and seed reproduce the trusted stage
signature. The existing Open3DBench `--from`, `--to`, `--resume`, stage
manifests and checkpoint chain implement the physical replay. A routing-only
candidate therefore reuses the verified CTS checkpoint and runs route, report
and thermal; an unknown shared-flow change reruns everything.

## Submission and verification

`collect_submission` freezes the base revision, binary patch, complete changed
file payload, and required artifacts into `submission.json`. It rejects
symlinks, artifacts outside the trial, missing logical artifact names and
changes to protected control-plane paths.

The host first verifies all submission hashes and the trusted verifier
identity. A trusted evaluator then replays the affected stages into an
evaluator-owned run directory. The final verifier runs in a separate container
with that run, the submission and benchmark inputs mounted read-only; it admits
metrics only when the submitted artifacts exactly match the independently
validated evaluator-owned run. It writes `verification.json` and `reward.json`
to a verifier-owned directory. Candidate-written metrics remain diagnostic
until this step.

## Commands

```bash
openagentic3dbench-global list
openagentic3dbench-global inspect lol_full_flow_ppa

openagentic3dbench-global prepare lol_full_flow_ppa \
  --source-root /path/to/Open3DBench-V2.0 \
  --trials-root /data/openagentic3dbench/trials \
  --user-command "在合法性门禁通过的前提下降低LoL全流程PPA"

openagentic3dbench-global impact lol_full_flow_ppa \
  --changed-file open3dbench/backend/three_d_route.py \
  --claimed-stage 80_route --policy suffix

openagentic3dbench-global runtime-plan lol_full_flow_ppa \
  --trial-dir /data/openagentic3dbench/trials/... --gpu 0 \
  -- codex exec --full-auto /task/instruction.md

openagentic3dbench-global collect lol_full_flow_ppa \
  --trial-dir /data/openagentic3dbench/trials/... \
  --artifact run_manifest=runs/final/run.json \
  --artifact final_checkpoint=runs/final/95_thermal/checkpoint.json \
  --artifact metrics=runs/final/90_report/metrics.json

openagentic3dbench-global release \
  --trial-dir /data/openagentic3dbench/trials/... \
  --trials-root /data/openagentic3dbench/trials
```

## Checked-in tasks

| Task | Authoritative endpoint | Fixed score |
| --- | --- | --- |
| `lol_full_flow_ppa` | user-defined full-flow PPA | supplied in the round-0 command |
| `lol_partition_placement_timing` | exactly two 3D OPT passes and continuous passive-HBT setup STA | reciprocal of `0.60 WNS deficit ratio + 0.40 TNS deficit ratio` |
| `lol_partition_placement_thermal` | complete backend and HotSpot steady-state analysis | reference average K divided by candidate average K |

The two co-design tasks leave partition and placement algorithms/configuration
editable while protecting benchmark code and all downstream evaluators. Their
case-matched benchmark input and reference are read-only inputs supplied through
`OPENAGENTIC3D_GLOBAL_INPUT` and `OPENAGENTIC3D_GLOBAL_REFERENCE`; task files
contain no server path or generated data. Timing rejects incomplete partitions,
placements, OPT passes, Liberty coverage, or timing graphs. Thermal fixes the
10x10 grid, 318.15 K ambient, shared provisional stack and entire downstream
backend, then averages every physical top/bottom silicon tile rather than the
two die averages or peak temperature.

The thermal reference frontend is explicitly **TP-GNN (default global-2way) +
Concurrent DREAMPlace**, the same methods as the STA co-design baseline, not
the general LoL Mt-KaHyPar default or the separate TA3D experiment. The thermal
profile's initial candidate names both methods; its resolved configuration and
actual run manifests must agree. The independent workspace verifier checks the
reference method manifests and recorded TP-GNN decoder without restricting scope-free candidate algorithm
development. Missing/wrong-method references must be regenerated and qualified;
historical result labels are not rewritten. This frontend alignment does not
change the thermal backend, OPT policy, objective, stack or physical budgets.

`scripts/run_global_codesign_codex.py` is the reproducible single-Codex
baseline adapter for these two tasks. One persistent Codex context proposes a
complete TP-GNN/concurrent-DREAMPlace intervention per round and receives only
the trusted final endpoint; it does not receive inner proxies. The adapter's
focused candidate surface is recorded in `run_config.json` and must not be
confused with the unrestricted workspace surface of the global task itself.
When a compute host cannot reach the Codex service, the same driver supports
`--evaluation-host`: the persistent Codex call and prompt trace stay on the
authorized client, while candidates, Docker execution, run directories and
all EDA artifacts remain on the named compute server.
