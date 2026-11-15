# Agentic3D workspace cooperation baseline v1

`scripts/run_workspace_agentic3d.py` implements the versioned
`agentic3d-container-workspace-v1` adapter. It tests the original Agentic3D
Global Analyzer / Local Analyzer / Task Agent cooperation on the same
workspace-oriented tasks and trusted evaluator as the single-Codex baseline.
It is not the legacy scoped-program adapter or a reproduction of its
knowledge-v2 proxy search, reflector, or task/skill-library evolution.

## Fixed control flow

One fresh reference is independently qualified before any role is launched.
The reference is outside the candidate budget. Each of five candidate rounds
then has exactly this sequence:

1. **Global Analyzer:** read the immutable user task, trusted reference/history
   and source; diagnose one hypothesis and target partition, placement, or both.
   Its first planning call is explicitly global analysis round 0.
2. **Local Analyzer, pre-task:** inspect the same hypothesis and evidence;
   refine implementation guidance and checks without widening the target stages.
3. **Task Agent:** write an explicit plan, then edit and inspect the private
   source workspace. Planning and coding remain in one persistent session.
4. **Trusted harness:** stop the agent container, freeze the actual source
   changes, and run exactly one candidate final evaluation through the existing
   `WorkspaceEvaluator`. Model-reported metrics are never scoring inputs.
5. **Local Analyzer, post-task:** inspect the actual final evidence and audit
   whether the candidate supports the planned hypothesis.
6. **Global Analyzer, adjudication:** review that audit against the immutable
   global objective and state whether the candidate should become the incumbent.

There are no cheap physical proxy evaluations in this adapter. Bounded source
inspection and tests are available to the Task Agent; final physical evaluation
is harness-owned. Stage hints guide cooperation but introduce no file allowlist.
The prohibition on additional physical runs in private scratch is a prompt
instruction, not a syscall-level ban on installed EDA binaries. Only the
harness-owned final ledger and independently verified results count as benchmark
evaluations; terminal traces must be retained to audit other agent activity.
Existing protected paths, fixed inputs, seeds, evaluator and verifier are
unchanged. Python/configuration changes are supported by the shared workspace
evaluator. Native C/C++/CUDA/build-file changes still require a separately
qualified image and are rejected by that evaluator, not silently ignored.

## Isolation and sessions

Each role has its own persistent Codex session and private scratch directory.
Global planning and adjudication share the Global session. Local pre/post audits
share the Local session. Task planning and editing share the Task session across
all rounds. A session cannot be reused for another role or silently replaced.

All model calls use the existing host-authenticated container-only terminal
transport. Credentials remain on the host. Tool containers are offline and do
not receive a Docker socket. Only the Task Agent gets a writable subject mount;
the analyzers get `/workspace` read-only. `/task`, `/results`, frozen input and
reference are always read-only. Analyzer scratch is not shared with the Task.

Before **every** analyzer call, host-owned code qualifies the real mount by
trying to open `/workspace/agentic_candidate.yaml` for writing. It never writes
to that file, even if the mount is wrong. Qualification requires `EROFS`, plus
a successful temporary-file write in private scratch. The receipt is saved as
`analyzer-isolation.json`. A before/after source-diff fingerprint is additional
defense in depth. All containers are stopped before returning control.

Every successful role turn must include at least one completed isolated
terminal call. Analyses contain checked mounted evidence paths and file size /
SHA-256 receipts. Post-task and adjudication analyses must cite at least one
file in the **current** candidate's trusted result directory. These checks
prove that files exist and bind their bytes; they do not prove that an LLM's
reasoning or interpretation of those bytes is correct.

## Task identities and scoring

| Objective argument | Global task | Authoritative final boundary |
| --- | --- | --- |
| `timing` | `lol_partition_placement_timing_one_pass` | Exactly one native OPT pass, pre-CTS passive-HBT setup STA |
| `thermal` | `lol_partition_placement_cts_thermal_one_pass` | Exactly one native OPT pass, CTS, cell power, HotSpot snapshot |
| `routed_thermal` | `lol_partition_placement_thermal` | Native one-pass OPT, CTS, GRT/DRT, report and routed HotSpot |

The driver imports these definitions from the existing `ENDPOINTS` table.
The [2026-09-27 reference alignment](reference_alignment_20260927.md) is versioned;
legacy two-pass tasks remain explicit historical contracts, not rewritten results.
Timing keeps the fixed WNS/TNS objective; thermal variants keep their respective
mean-temperature objective and evaluation boundaries. Case selection is
explicit because the global task templates do not pin a case. For the global
smoke batch, the caller selects `bp_fe` as the recorded smoke-default.

`best` is the highest independently verified score measured so far, including
the reference. `incumbent` is distinct: a candidate can replace it only when
its verified score strictly improves the **prior best**, the Local post-audit
accepts, and Global adjudication accepts. Role statements cannot promote a
failed, invalid or non-improving candidate. Reports retain both values, so a
role-rejected improvement is not presented as an accepted cooperative result.
Private working edits carry forward even after rejection, enabling diagnosis
and repair; they are not thereby accepted as the incumbent. Each evaluated
snapshot remains separately recoverable in the trusted results directory.

## Failure and budget policy

- Exactly five candidate final attempts, including ordinary physical or
  candidate-admission failures. The driver records a final slot durably before
  invoking the evaluator. A thrown evaluator exception is also charged once.
- A model/tool/container failure, malformed analysis, missing evidence or
  missing Task plan before evaluation stops the run without spending a final.
- A post-evaluation analyzer failure preserves the consumed final and trusted
  result; adjudication remains pending rather than being fabricated.
- Explicit disk, Docker and recognized device-environment failures pause
  the run for inspection. The shared evaluator currently returns free-text
  errors rather than a complete typed failure taxonomy. The wrapper checks
  those errors and bounded phase-log excerpts; unknown ordinary failures are
  retained as failed finals, without asserting an infrastructure diagnosis.
  In particular, `FileNotFoundError`, `PermissionError` or `OSError` alone do
  not establish infrastructure failure: candidate code may cause them.
- No automatic retries, reference adoption, hidden physical runs or implicit
  resume. An existing `run_config.json` prevents relaunch into the same output.
  Evidence requests and real blockers stop for human inspection.

## Integration API

```python
from scripts.run_workspace_agentic3d import run_workspace_agentic3d_experiment

summary = run_workspace_agentic3d_experiment(
    runtime,       # existing codesign_runtime.Runtime with shared host locks
    "bp_fe",       # explicitly recorded case
    "timing",      # timing | thermal | routed_thermal
    bundle,        # previously validated, frozen benchmark-input bundle
    output,        # new private run directory within Runtime's batch root
    codex_binary,  # existing authorized host installation/login
)
```

The host launcher owns validated model/tool preflight receipts and process-local
network settings. This module does not probe a model, copy credentials, change
global configuration, schedule competing jobs, or prepare benchmark inputs.

## Reading results

```text
run_config.json                 immutable adapter/task/resource identity
status.json / summary.json      progress, budget, verified best and incumbent
history.json                    reference and every candidate final outcome
sessions.json                   distinct role session IDs
control/round-NN-*.json          host-owned handoffs mounted at /task
rounds/NN/
  global-plan/                  context, analysis, isolation proof, agent trace
  local-pre/                    pre-task diagnosis and evidence
  task/                         task-agent prompt/events/final response
  plan.md                       preserved explicit task plan
  local-post/                   audit against current trusted final evidence
  global-adjudication/          global decision and checked evidence
results/reference/              fresh qualified physical reference
results/round-NN/               independent physical logs, verifier and feedback
scratch/{global-analyzer,local-analyzer,task-agent}/
```

The `agent/` trace within each phase records its real prompt, command, events,
stderr and result. All generated plans, analyses, logs and summaries are
requested in English. Existing native tool output is preserved verbatim.
The mock suite `tests/test_run_workspace_agentic3d.py` validates orchestration
and isolation contracts only; it makes no claim of remote model or EDA success.
