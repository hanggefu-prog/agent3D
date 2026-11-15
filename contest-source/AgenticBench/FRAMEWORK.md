# Agentic3D multi-agent framework

For the current four-case execution contracts, read the
[harness design report](docs/HARNESS_DESIGN_REPORT.md) and
[task settings report](docs/TASK_SETTINGS_REPORT.md) first. This document also
retains generic and historical modes, including two-pass profiles and bounded
stage orchestration. Those capabilities do not imply that the latest workspace
driver activates them, or that all local/thermal tasks have native qualification.

## Implemented architecture

OpenAgentic3DBench now has two complementary execution tracks:

- bounded local tasks use the existing `StageTask` and ask/tell contracts;
- open-ended global tasks use `GlobalTaskSpec`, scope-free `Intervention`
  handoffs, isolated whole-repository Trial workspaces, adaptive stage replay,
and a separate trusted verifier. Two fixed co-design tasks instantiate that
track: LoL partition-placement for two-pass setup WNS/TNS and for full-flow
HotSpot average temperature. Their task YAML fixes the evaluator/reference and
score constants without introducing an algorithm source allowlist.

The global track does not use `scope_mode=direct`. Freedom is provided inside
an ephemeral candidate workspace, while benchmark control files and evaluation
remain outside or read-only. Its code and task example live in `global_bench/`
and `global_tasks/`; see [docs/global_tasks.md](docs/global_tasks.md).

Agentic3D is an independent control plane for Open3DBench. It does not vendor
or import Open3DBench Python internals. A version adapter discovers source,
configuration, logs, results, commands, metrics, and correctness gates and
normalizes them for all agents.

```text
Round 0 user command
        |
        v
Global Analyzer (read-only, persistent thread)
  parse user objective -> collect global PPA -> diagnose -> select one stage
        |
        v
Local Analyzer for selected stage (read-only, persistent per-stage thread)
  inspect logs/results -> one hypothesis -> gates + least-privilege scope proposal
        |
        v
Scope Compiler (code, partition/layout/routing in enforced mode)
  adapter hard boundary ∩ Local proposal -> signed scope contract -> worktree
        |
        v
Task Agent instance (workspace-write, persistent per-task thread)
  stage Skill + frozen infrastructure Skills
  inspect -> persist plan.md -> edit -> compile -> evaluate -> validate
        |
        v
Scope Guard + CandidateUpdate + deterministic Global PPA collector
        |
        +--> Local Analyzer post-analysis
        |
        +--> Global Analyzer accept/reject/pending
                  |
                  +--> dispatch same/next stage
                  +--> complete
                  +--> blocked
```

The collaboration order is implemented by
`agentic3d/orchestrator.py`; agents cannot bypass or reorder it.

That diagram describes the bounded multi-agent baseline. The global workspace
runner uses the following state transition instead:

```text
GlobalSearchContext -> Intervention -> isolated Trial -> Submission
  -> trusted verification -> adaptive evaluation -> accepted state / next round
```

Research tasks, evaluation profiles, and Skills are distinct. A task defines a
bounded evolvable program or parameter space. An evaluation profile replays an
already selected candidate through a fixed downstream contract. A Skill fixes
common agent procedures such as environment qualification, runtime builds,
benchmark-input validation, candidate evaluation, STA/HotSpot promotion, and
result analysis. The code-validated registry is `skills/catalog.yaml`.

## Agents

### Global Analyzer

The Global Analyzer is the only user ingress. On global round 0 it receives the
raw user command, repository context, and current global PPA evidence. In the
bounded multi-agent baseline it produces a structured diagnosis and selects
exactly one of:

- `partition`
- `layout`
- `place_opt`
- `routing`

On later bounded rounds it receives CandidateUpdate, Local Analyzer feedback,
and the refreshed PPA report. It explicitly marks the candidate `accept`,
`reject`, or `pending`, then dispatches another stage, completes, or blocks.

In global-workspace mode it instead emits a scope-free `Intervention`. One or
more target stages express the hypothesis and determine adaptive replay, but do
not limit which subject files the acting agent may edit. The accepted revision,
trusted observations, and remaining budget are carried by
`GlobalSearchContext`.

### Local Analyzer

Each stage has one persistent read-only Local Analyzer thread. It runs before
and after a Task Agent:

- pre-analysis provides one falsifiable hypothesis and predeclared gates;
- post-analysis restates and audits that hypothesis against real logs/results,
  then returns acceptance, rejection,
  or more evidence.

It cannot modify files or launch another agent. The schema and runtime
validator enforce phase-specific decisions: pre-analysis may only recommend or
request evidence; post-analysis may only accept, reject, or request evidence.
For partition, layout, and routing, pre-analysis also proposes exact
repository-relative source/config paths and conservative operation and diff
budgets. It cannot expand the adapter's stage boundary. Post-analysis must
restate that proposal and audit the deterministic scope report.

### Task Agents

The four domain agents are instances of one `TaskAgent` state machine:

| Instance | Main scope | Accepted output |
|---|---|---|
| Partition | partition algorithms/config | frozen partition and cut/balance evidence |
| Layout | Place-3D/legacy placement and DREAMPlace | legal DEF and HPWL/density evidence |
| Place Opt | OpenROAD placement, resizing, timing opt, detailed placement, CTS | `3_place`/`4_cts` ODB+SDC and PPA |
| Routing | GRT, DRT, HBT, die-by-die routing | routed ODB/DEF, guides, DRC/connectivity/PPA |

Planner and coding remain in the same Codex thread. Every round must persist
`plan.md` before edits and return structured plan/execution/validation/decision
JSON. `goal_achieved` is accepted only when all validation results pass and
goal evidence is present. It means that the stage candidate satisfies the
current handoff, not that the end-to-end user task is complete. Only Global
Analyzer may declare global completion.

In enforced mode, every plan lists exact `files_to_modify`, and the Task Agent
runs inside a detached candidate worktree. It may not perform Git lifecycle
operations; framework code owns audit, checkpoint, rollback, and integration.
In direct mode, the Task Agent runs in the canonical checkout with no
framework path restriction. Direct mode deliberately has no automatic merge or
rollback and records that mediation was bypassed.

Task Agent inputs have a fixed precedence: user objective and immutable
constraints; Global Plan; Local Analyzer hypothesis and gates; stage Skill;
supporting infrastructure Skills frozen by the orchestrator; adapter-generated
repository context; current measured evidence; historical state. A newer
handoff supersedes stale historical paths.

## Repository adapter

Adapter manifests are under `adapters/`. The current manifests support:

- unified Open3DBench with `Place-3D`;
- legacy Open3DBench with `Place-MoL`.

Each manifest contains only version-specific facts:

- sentinels used for detection;
- stage source and config paths;
- log and result roots;
- compile and evaluation command hints;
- metrics and correctness gates;
- artifact glob patterns.

To support a future release or a new MoL/LoL/HBT regime, add or update one JSON
manifest. Analyzer, TaskAgent, orchestrator, and schemas do not need to change.

`agentic3d-inspect --repo <path>` validates detection and prints the normalized
context before starting a workflow.

## Global PPA collector

`agentic3d/ppa.py` scans adapter-discovered JSON, CSV, logs, and reports and
extracts evidence for HPWL, routed wirelength, overflow, maximum density, WNS,
TNS, power, area, DRC, vias/HBTs, skew, runtime, partition cut, and balance.

The report is an evidence inventory, not automatic proof. Global and Local
Analyzers still verify design, seed, units, stage completion, artifact counts,
and correctness before comparing candidates.

## Handoff contracts

The orchestrator persists:

```text
StageHandoff
  workflow/cycle/stage
  user task
  repository context
  global plan
  local feedback
  accepted upstream candidates
  immutable constraints
  scope contract (enforced mode)

CandidateUpdate
  task status/state
  round results
  discovered logs/results/artifacts
  scope mode/report, worktree and candidate/integrated commits
```

Only a candidate explicitly accepted by Global Analyzer enters
`accepted_candidates` and is propagated to downstream handoffs.

## Source mediation

`scope_mode` is a workflow-level hyperparameter with two values:

| Mode | Workspace | Enforcement | Acceptance behavior |
|---|---|---|---|
| `enforced` (default) | detached candidate worktree | adapter boundary intersected with Local Analyzer proposal; actual Git diff and budgets checked by code | accepted candidate patch is re-audited and committed into a clean canonical checkout |
| `direct` | canonical checkout | none; stage Skill remains advisory | edits already exist in the canonical checkout; no automatic integration or rollback |

Enforced mediation currently applies to partition, layout, and routing. The
Scope Guard checks exact paths, extensions, file and line budgets, creation,
deletion, rename, mode changes, symlinks, and unexpected HEAD changes. Agent
state under `.agentic3d` is treated as framework control data rather than a
source candidate. Global Analyzer cannot accept an enforced candidate without
a passing scope report, or a direct candidate without an explicit bypass
report.

The Integration Gate runs only after Global Analyzer acceptance. It verifies
that the canonical checkout still points to the baseline commit and has no
tracked modifications, re-audits the complete candidate diff, applies it, and
creates the integration commit. A failed candidate is rolled back only in its
isolated worktree; unrelated user changes are never reset.

## Runtime layout

Workflow evidence defaults to an external per-repository directory under
`$XDG_STATE_HOME/agentic3d` or `~/.local/state/agentic3d`. It may be moved to a
data disk with `AGENTIC3D_RUNS_ROOT` or `--workflow-dir`. Candidate worktrees
default beside that external workflow state and may be moved with
`AGENTIC3D_WORKTREES_ROOT` or `--worktrees-root`:

```text
<agentic3d-state>/repositories/<repository-id>/workflows/<workflow-id>/
├── state.json
├── repository-context.json
├── global-ppa-report.json
├── global-plan-round-000.json
├── global-analyzer/
├── local-analyzers/<stage>/
└── cycles/NNN/
    ├── handoff.json
    ├── local-feedback-pre.json
    ├── scope.json                    # enforced mode
    ├── worktree.json                 # enforced mode
    ├── candidate-repository-context.json
    ├── candidate-scope-report.json   # pass report or direct bypass record
    ├── candidate-update.json
    ├── local-feedback-post.json
    └── global-plan-next.json
```

In enforced mode, Task Agent state, the immutable Skill snapshot, plans, and
round evidence live under `<candidate-worktree>/.agentic3d/task-agent`. In
direct mode, they live under
`<open3dbench>/.agentic3d/direct/<workflow-id>/task-agents`.

Agentic3D source is maintained in a separate repository. The orchestrator
copies the selected Skill, helper scripts, tests, and validation tools into a
candidate runtime workspace, so the target repository must not contain an
`Agentic3D/` source tree. The ignored `.agentic3d` directory used in candidate
worktrees and direct mode is runtime control data only.

## Sandbox and execution

- Global and Local Analyzers use `read-only`.
- Task Agents use `workspace-write` on a candidate worktree in enforced mode,
  or on the target Open3DBench checkout in direct mode.
- Scope and integration decisions are deterministic code checks, not LLM
  judgments. Local and Global Analyzers diagnose and accept evidence but cannot
  override a failed check.
- Commands containing unresolved placeholders must not run.
- Every real run uses an isolated result directory or `FLOW_VARIANT`.
- Prompts, JSONL events, stderr, final responses, schemas, state, and evidence
  are persisted for replay and audit.
