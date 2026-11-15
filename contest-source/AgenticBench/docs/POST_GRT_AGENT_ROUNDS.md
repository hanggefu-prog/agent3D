# Explicit Codex evolution rounds

One evolution round uses **one `codex exec` invocation** to inspect feedback,
choose a parent, edit its source and return a candidate. During that invocation,
the agent calls the bound `checkout_parent` tool; the experiment harness (the host) prepares the selected
source and returns control to the same running agent. Parent selection is an
ordinary tool interaction, not a second prompt or model invocation.

After the agent exits, the harness uses the EDA backend tool to build the candidate
and evaluate one complete optimization matrix on the training cases. The
next round starts only after its outcome is known. By default the controller
resumes the explicit Codex session ID within one campaign; `session_mode: fresh`
starts every round in a new session and repeats the background. Every round still supplies current
source, budget and measured results; conversation history is not authoritative.
A round is distinct from a tool call, case or seed.

The [server controller](../global_bench/grt_campaign.py) connects the renderer,
reading library, parent checkout, source verifier, candidate builder, worker pool
and Pareto archive. The [stdio MCP bridge](../global_bench/grt_round_tools.py)
exposes `exec` in the isolated round container, `checkout_parent` bound
to that round, and the budgeted `gpu_check` diagnostic tool. Codex CLI runs on the same server as the controller; its terminal
commands run in the container. Authentication stays outside the container.

In the prompt, "EDA backend tool" means the build and physical evaluation
services managed by the harness. It is not an agent-callable full-matrix tool.
A real model-generated candidate qualification must be reported separately from
synthetic controller tests and native container/MCP checks.

### Claude Code backend

`agent_backend: claude` selects the same round protocol through
`global_bench/workspace_claude.py`. Supply an explicit `model`, `claude_binary`,
owner-only `provider_config` containing `ANTHROPIC_BASE_URL` and
`ANTHROPIC_AUTH_TOKEN`, and a separate `claude_config_directory`. Credentials stay
outside the candidate container and command arguments. Claude Code runs with
`--bare --restricted`, no built-in tools, and only the three bound MCP tools.
The current integration was checked with Claude Code 2.1.301.

Up to five transport attempts may continue the same Claude session within the
original round timeout. Only recognized connection, rate-limit and server
errors qualify. Authentication errors, invalid models, timeouts and unfinished
tool calls stop recovery. Completed tool results remain in the session; a retry
does not repeat parent checkout or launch an evaluation. The campaign records
client attempts separately. A generation failure spends no matrix budget.
Both backends use the same source verifier, incremental EDA flow and archive.

A campaign can own one or two GPUs with three slots per GPU and up to twelve
workers. Concurrent campaigns must receive disjoint GPU IDs and CPU/memory
capacities whose totals fit the server. Local GPU checks use the same campaign
queue; the frontend releases its slot before the CPU backend starts.

### Optimization cases and final tests

The frozen task assigns each case to `train` or `test`. Only `train` cases enter
optimization matrices, Pareto aggregation, prompts, reading libraries and local
input mounts. Case weights apply within the selected split. A split change
creates a new task identity and archive; historical receipts must not be relabelled.
The original reference can be requalified by analysis replay when the physical
contracts are unchanged and training cases are reserved as final tests. The
replay regenerates verifier receipts and checks equality with original measurements.

`global_bench/grt_final_test.py` freezes the original reference and the entire
final training frontier before any test result is produced. It releases the test
matrices through the same worker/independent-verifier protocol, saves separate
test summaries, and leaves the training archive unchanged. The persistent
`control/final-test-frozen.json` gate prevents further agent rounds. Tests have a
separate final evaluation phase and are not optimization feedback.

The 2026-10-19 model comparison uses three equal-weight optimization cases:
`bp_fe-ng45`, `bp-ng45`, and `ariane133-ng45`. `swerv_wrapper-ng45` and
`bp_fe-asap7` are final tests. Both splits use the same three paired seeds.
The split transition starts fresh model sessions from the original reference,
without old candidate summaries or old test feedback. These cases were observed
in older experiments, so this is a prospective separation, not a claim that they
have never been observed in the project's history.

## Prompts

- [First-round background](../global_tasks/lol_post_grt_pareto/background.md)
  briefly explains the two-die design, existing flow and main methods, related
  changes across stages, the three objectives and power constraint in plain English. The renderer
  includes it only in round 1. It describes the project; it does not expand the
  task's allowed files or permit method switches.
  The default task enables partition, placement, HBT assignment, timing
  optimization, CTS and GRT. The prompts encourage algorithm changes beyond
  early parameter tuning and exploring other stages when progress stalls.
  Power remains a constraint to keep the Pareto objective count at three.
- [Round prompt](../global_tasks/lol_post_grt_pareto/round.md) follows one sequence:
  interpret evidence, choose a hypothesis, prepare its parent, implement, review,
  then return a final response. Scope and budget rules are stated once. The old
  selection/editing prompt files are removed. Each invocation still gets
  the goal, scoring rules and full operating rules; later rounds do not depend
  on remembering the background.
  Every round prioritizes early exploration of partition and placement algorithms,
  including their interaction, and revisits them when later-stage gains stall.
  It points to budgeted GPU local checks for focused tests within the allowed scope.
- [Renderer](../global_bench/grt_round_prompt.py) appends a compact `HOST_CONTEXT`
  and field guides for parent checkout and the final response. The full schemas
  are supplied through the tool interface and structured output, not repeated in
  the prompt. All task-bound enums and host validation remain intact.
  The agent-facing prose lives in the background and round templates. The frozen
  task's long `research_brief` is not repeated as an escaped JSON string in
  `HOST_CONTEXT`. Its text stays in host task records; only audit digests are
  kept in the host bundle, outside the prompt.
  Every round also lists the actual evaluation cases and seed combinations in
  tables and permits those cases in local tests. Raw measurements keep their
  physical units; the three normalized objective scores have no units.
  `check_parent_request` validates the proposed parent/files/hypothesis;
  `check_response` binds the final JSON to the recorded host receipt and rejects
  stale IDs, changed proposals, scope violations and inconsistent statuses.
- [Checkout handler](../global_bench/grt_round_checkout.py) records one live
  frontier selection and materializes its source during the invocation. It
  accepts no model-provided host paths. A lock and persistent state serialize
  concurrent requests: identical delivery returns the original receipt, a changed
  proposal is rejected, and incomplete checkout requires host reconciliation.

These checks do not replace source verification. After freezing, `record_summary`
checks the declared files against the actual parent diff and binds the descriptions
to both source comparisons. A parent checkout does not reserve an evaluation.

Do not reuse the legacy single-case TaskAgent prompt: it expects the same
persistent workspace and allows a different evaluation lifecycle.

Before an agent starts, edit the background/round templates, then use
`grt_campaign revise-prompt --config ... --research-brief ...` to rebuild the
review bundle without rewriting the frozen task or baseline evidence. The brief
argument updates the host audit record only; it does not inject another copy of
instructions into the agent prompt. Its source-scope front matter must stay the same;
numeric objectives, cases, seeds, allowed files and admission rules still come
from the frozen task. This is a presentation revision, not qualification of a
new evaluation contract. Changes to that contract require a new task.
The controller preserves the old bundle and tool configuration under
`prompt-revisions/`, records both brief digests, and binds the replacement prompt
to its review hash. It consumes no model call or evaluation; later rounds use the
same templates and inherit the brief audit record. An interrupted rewrite blocks execution until
reconciled. Revisions after model or checkout activity are rejected.

## Context supplied each round

The prompt includes the task ID, round/budget state, timing weights, power limit,
normalization floors, workspace/scratch paths, check limits, editable components,
reference source, candidate summaries and latest eight completed round summaries.
Cases, weights and paired seeds appear once, in the evaluation tables.
`task_details` points to the reading library's `task-context.json`: exact editable
files, fixed physical settings and a structured copy of the evaluation contract.
The agent must read its `allowed_files` before planning an edit. Legacy sealed
libraries without this file keep the exact scope inline.

The bundle retains complete host `context` separately from the rendered
`agent_context`. Library/reference hashes, reservation flags, duplicate frontier
IDs, empty history and summary-origin metadata are omitted from the reading view;
host checks still use the original data. Nonempty failures and missing results
remain explicit. Numeric scores and deltas retain full precision, since rounding
could hide a small temperature difference. This changes presentation only, not
task identity, source permissions, evaluation or Pareto admission.

An `infeasible` history entry records a complete matrix that
exceeded the power cap; it cannot supply a frontier parent. Older summaries
remain in the host history and the prompt states
how many are omitted. First-round history is empty. Rejected and failed rounds
are visible as failures, never converted into zero or fabricated PPA.

The prompt gives a short entry for every task-admitted candidate, including
superseded parents, dominated candidates and failed or unevaluated legal edits:

- candidate and parent IDs, and status (`frontier` means selectable);
- a few sentences describing changes from the original reference and from the
  immediate parent; description origins remain in the library's candidate records;
- the three normalized objective values, their differences from reference and
  parent, verified cell count, power pass/fail and any failing cases;
- one agent-visible directory for code, diffs, complete results and logs.

Objective differences are **candidate minus comparator**, so negative is better;
all three scores are dimensionless. They are recomputed from admitted training
measurements, not predicted by the agent. Failed, partial and unevaluated matrices
have no aggregate score. Only current frontier members appear in the parent tool's
choices. Rejected source edits remain in round history, outside the source library.

[The library exporter](../global_bench/grt_candidate_library.py) creates a new,
hashed reading directory for each round. It copies all task-editable source files
under their repository paths, both actual diffs, detailed metrics and allowlisted
native training logs/reports. This is an algorithm source view, not a second full
repository checkout. The selected parent's full frozen tree is supplied separately
by `checkout_parent`. Candidate files can be inspected **before selecting a parent**.

```text
/workspace/candidates/
  index.json
  manifest.json
  task-context.json
  <candidate-id>/
    README.md
    candidate.json
    code/<repository-relative editable files>
    from_reference.diff
    from_parent.diff
    evaluation/metrics.json
    evaluation/cells.json
    logs/index.json
    logs/<case>/<seed>/<original log path>
```

`metrics.json` retains absolute per-case/per-seed PPA, mean/sample standard
deviation/worst seed, normalized ratios, the full aggregate and per-case power
means, limits and excess in W. Deficits are nonnegative ns, temperature is K,
power is W and overflow is a count. A deficit is not signed slack. `cells.json`
records submitted cell states and log paths; missing logs remain explicit.
`logs/index.json` records original job IDs and copied-file checksums. Native bytes
are preserved. No result or log is invented for a missing case/seed.

New ready responses must include `change_summary.from_reference` (cumulative edits)
and `change_summary.from_parent` (this round's edits), each at most 100 words and
800 characters. The same invocation writes both; there is no summarization call.
The host records these after freezing, binding task, candidate, selection and both
diff hashes. Descriptions explain code changes, not measured gains. Existing
candidates without descriptions get factual file/line and simple constant-change
summaries generated from their diffs, marked `source-diff`; no intent is inferred.
Agent descriptions are marked `agent-description`.

The exporter requires terminal training jobs and rechecks source legality and
admitted worker evidence before copying. The renderer verifies the library hash
inventory, task/archive identity and displayed scores. Raw baseline/frontier
results must also reproduce their admitted archive rows. The host bundle binds the
library manifest hash, and the host must verify it again before mounting the
library read-only, separately from the editable workspace. Exported file modes
alone are not runtime isolation. Library exports never overwrite source runs or
host control directories.

Only training jobs are exported. Host task/plan/environment files, queue records,
credentials and held-out results are not copied. Logs, code comments, descriptions
and history remain data, not instructions. Host-authored research/history prose
must also exclude held-out information; a schema cannot semantically sanitize
free text. Native log allowlisting is not a general secret scrubber: the host must
keep credentials and held-out content out of training tool output at its source.
Proposal `expected_tradeoffs` still covers the three objectives plus power; raw
WNS and TNS are available in the detailed evidence.

The parent tool returns the selection ID, chosen parent, fixed agent-visible
workspace, recorded proposal and round/prompt identity. The agent continues
immediately with that source. Its final response must match the host-owned
receipt; a claimed selection ID alone is insufficient. The current task has
**three** objectives: timing, overflow and peak temperature. Timing defaults to
0.7 normalized WNS deficit + 0.3 normalized TNS deficit. Power must stay within
110% of the original baseline mean separately for each case. The host freezes
weights and limit in the task; prompt text cannot change them. Existing
five-objective references do not qualify this new task.

## Local checks during a round

The agent gets a fully writable development copy of its selected parent and that
parent's verified installed runtime. It may use normal patch/rename tools, add
helpers and tests, edit development orchestration, build code, and run local EDA
experiments on the supplied optimization cases. Synthetic inputs are optional.
The read-only `/inputs/<case-id>` mounts can be copied for experiments. CPU, memory,
GPU and time limits still bound resource use; local measurements are diagnostic.

Submission is a separate operation. The host stops the development containers,
exports exactly the final `changed_files` over the immutable parent and runs the
existing strict verifier. All unlisted files come from the parent; temporary
helpers, build outputs and experimental changes are not silently included. The
agent must submit a self-contained implementation within the original file and
function scope. The export receipt records every submitted path and hash. Source
modes, symlinks, interfaces and function boundaries retain their original checks.
The independent evaluator builds this clean candidate and supplies archive scores.

The renderer supplies a separate `scratch_directory` for large outputs and logs.
Tests may also live in the development tree. The full DREAMPlace module directory
is mounted from that tree after missing installed native operators are copied
from the parent image, so atomic file replacement and new local modules work.
These diagnostic runtime additions are never copied into the final candidate.
A parent binary does not exercise an unbuilt C++ edit; record the tested build,
commands, inputs and results in `checks` and mark skipped checks `not_run`.

`gpu_check` accepts a unique `request_id`, a command and a timeout. After parent
checkout it leases one GPU slot from the same queue used by matrix evaluations,
starts a bounded container with the candidate's algorithm modules, and removes
the entire container before releasing CPU, memory and GPU reservations. The
ordinary terminal container has no GPU. Default diagnostic limits are two calls,
900 reserved seconds in total and 600 seconds per call; the frozen round context
states the actual limits. The frontend qualification uses three calls to check
normal execution, timeout and background-child cleanup.

Reservations are durable before launch. Repeating an identical request returns
its original result; reusing its ID with other arguments is rejected. If capacity
is unavailable, no command runs and its time reservation is refunded. Uncertain
termination keeps the resource lease and time reservation until host cleanup
confirms removal. Diagnostic jobs and evaluation jobs have separate queue types,
so an evaluation worker cannot claim a local command. Results include the GPU,
elapsed charge, output/log directory and termination receipt. They never count
as a complete matrix or authorize Pareto admission.

`direct_full_evaluation_allowance: 0` replaces the ambiguous
`direct_evaluation_allowance` field; it forbids direct full evaluations, not local
tests. `maximum_matrices_this_round` remains one. Directory validation and prompt
permissions alone do not configure a runtime. The server controller supplies
these mounts and limits; its native isolation checks and a complete real
agent-to-Pareto qualification are recorded separately.

## Host sequence and hard budget

1. Qualify and ingest the task-matched baseline before opening any evolution
   round. Freeze model configuration, template/renderer hashes, round limit `R`,
   candidate-matrix limit `E <= R`, time limits and infrastructure-retry policy
   (none by default) in the host
   campaign record. Baseline and final held-out evaluation budgets are separate.
2. Under a host lock/transaction, reserve and persist the round number **before**
   agent invocation. Require fewer than `E` previously spent/reserved candidate
   matrix attempts. Export a fresh candidate reading library and render one prompt
   from the same archive snapshot. Verify and mount the library read-only before
   invocation, then seal the prompt.
   Bind the parent tool to this prompt, task, round and a fresh workspace. Start
   one isolated `codex exec` with the output schema and bound tool available.
   Mount a separate writable scratch directory for local tests and diagnostic
   outputs, under the round's resource and time limits.
3. During that invocation, `checkout_parent` records the request as pending,
   calls `Archive.select` to recheck the live frontier, and uses `Archive.checkout`
   to prepare the source. Its receipt lets the agent continue editing without
   exiting or resuming. The controller must provide the fixed container path and
   protect all non-allowed files before tool success becomes visible. A stale
   parent or interrupted checkout stops the round without a replacement call.
4. After the single invocation ends, terminate its editing container/process and
   descendants before reading source. Pass `RoundParentCheckout.receipt()` to
   `check_response`.
   A pre-checkout failure may return blocked with null parent/selection and empty
   file lists. Missing final output, no-change, blocked, illegal edits or failed
   checks consume the round and cannot launch a matrix.
   They do not consume an evaluation. A request for ten actual evaluations must
   drive the loop by `evaluations_used`, with a separate model-attempt ceiling;
   it must not stop after ten model invocations if some never submitted a matrix.
5. For a ready legal candidate, freeze and reverify it, record the two change
   summaries against the actual frozen diffs, then build and install its image.
   Build failures do not consume a physical evaluation. After the runtime is
   ready, persist one evaluation reservation and submit the complete training
   matrix through the existing scheduler. A failure before any candidate queue
   row or matrix admission refunds that reservation and stops for reconciliation.
   Dispatched candidate execution failures spend the attempt. No per-seed retries,
   replacement candidates or extra physical runs are implicit in the budget.
6. Wait for the matrix to become terminal; do not start the next evolution round
   while its evidence is partial. Successful, independently verified matrices
   enter `Archive.ingest`; dominated and power-infeasible candidates remain in
   history but cannot be selected as new parents. Failed or
   incomplete matrices never enter the frontier. Record feedback and advance only
   if round and evaluation capacities remain. Ambiguous infrastructure execution
   stops the campaign without automatic retries.

The budget snapshot passed to the renderer is descriptive, not a spending token:

```json
{
  "round_number": 1,
  "round_limit": 10,
  "evaluation_limit": 10,
  "evaluations_used": 0,
  "round_reserved": true
}
```

The renderer checks bounds and consistency but cannot enforce a campaign cap by
itself. The host's durable ledger must charge rejected rounds and prevent two
processes from reusing a reservation. This is necessary even when prompts forbid
extra evaluations. After a crash, reconcile recorded process/job IDs; do not
silently call Codex or submit a matrix again.

For `C` training cases and `S` seeds, `E*C*S` is the ordinary candidate-cell
budget, plus `C*S` for a fresh baseline. For example, ten candidate matrices at
five cases x three seeds cover 150 candidate cells; model invocations have their
own limit `R`. A known infrastructure failure may be excluded only after the
execution is reconciled and an authorized credit receipt preserves its original
reservation, evidence and reason. This is not an automatic retry allowance.
The 2026-10-15 recovery uses ten actual evaluations and a separate ceiling of
30 recorded rounds, stops after three consecutive model-only failures, and
credits the matrix invalidated by regenerated fixed inputs. It reuses the same
model-generated source without another model call. Actual candidate algorithm
failures still spend an evaluation; incomplete matrices cannot enter Pareto.
Six GPU frontend slots and immediate GPU release remain managed
by the existing scheduler; CPU backends overlap within its independent limits.

## Preparing prompt bundles

From `AgenticBench`, the host exports `Archive.show()` to `archive.json` and
passes admitted worker results for exactly the baseline and current frontier
to `outcomes.json`. Do not mount these full host exports into the agent. Each
output directory below must be new. Keep the archive stable during export and
rendering; reconcile a stale snapshot instead of mixing evidence.

```sh
python -m global_bench.grt_candidate_library export \
  /host/harness task.json round-001/library --agent-root /workspace/candidates
python -m global_bench.grt_round_prompt \
  task.json archive.json budget.json outcomes.json round-001/prompt \
  --library round-001/library --history history.json --workspace /workspace/subject
```

After checkout, response validation and legal candidate freezing, bind the same
invocation's descriptions to the frozen candidate:

```sh
python -m global_bench.grt_candidate_library record-summary \
  /host/harness task.json round-001/prompt/bundle.json \
  checkout-receipt.json response.json CANDIDATE_ID
```

The receipt is the host's `RoundParentCheckout.receipt()` result. Run this before
submitting the candidate matrix. Notes stay in host control; later reading exports
include only their source-bound descriptions. Conflicting notes are rejected.
Prompt bundles now use schema version 3; task schema version 2 and physical
scoring are unchanged. Old response bundles without the two summaries are not
accepted by the new round validator.

Every bundle has `prompt.md`, `output.schema.json`, `parent-tool.json`,
`bundle.json` and a hash manifest. The host stores actual prompts, schemas, model settings,
renderer/template digests, JSONL events, final response, source diff, verification
and evaluation receipts outside the candidate tree. Template changes cannot
silently alter an ongoing campaign. `template_digest` covers both the background
and round instructions in every round, even when the background is omitted.
The renderer is outside the physical
evaluator package; changing prompt assembly alone does not relabel PPA evidence.

The controller constructs `RoundParentCheckout(root, task, bundle, control,
workspace)` with a dedicated host control directory under `root/control` and a
new host source directory under `root/workspaces`. Neither directory is a tool
argument. Expose only its `checkout(proposal)` method using `parent-tool.json`.
The tool receipt publishes the agent-visible workspace, not the host path.

The controller uses `global_bench.workspace_codex` to pass the prompt on stdin,
record JSONL events and validate the final JSON schema. It preserves an explicit
session ID for subsequent `codex exec resume` calls (unless `session_mode` is
`fresh`). It does not use `--last`. Token usage (the round's share of the
session total that `turn.completed` reports) and an optional price-table cost
estimate are recorded per round; see
[agent adapters](AGENT_ADAPTERS.md).
The current selected source and scores always come from the round's frozen data.

The container has no network, Docker socket, SSH access, host authentication or
queue/archive mounts. Before checkout it exposes the read-only candidate library
and writable scratch directory. Checkout replaces that container with one that
also exposes a fully writable copy of the selected parent. Final source exports
remain restricted to the task scope; development files may be created or replaced. CPU, memory and turn/tool time limits are visible in HOST_CONTEXT.
No GPU is reserved by the editing container. After the model exits, the controller
confirms removal of its container and all descendants before freezing source.

## Server controller commands

A campaign config pins the task, framework, source baseline, image, input paths,
model, round/matrix budgets and resource limits. This qualification uses two GPUs
with three slots each and twelve worker loops. The queue releases each GPU after
frontend termination; the CPU backend continues under a smaller allocation.
Workers use the existing queue/lease protocol in the controller process, without
exposing an administrative queue endpoint to the agent.

```bash
python3 -m global_bench.grt_campaign baseline --config /path/to/control/config.json
python3 -m global_bench.grt_campaign prepare --config /path/to/control/config.json
# Inspect the complete prompt.md returned by prepare before calling run.
python3 -m global_bench.grt_campaign run --config /path/to/control/config.json \
  --reviewed-prompt-sha256 EXACT_REVIEWED_HASH
python3 -m global_bench.grt_campaign status --config /path/to/control/config.json
```

`prepare` is idempotent for the current prepared round and makes no model call.
`run` checks the reviewed prompt hash (every round by default, or only round 1 and
revised prompts under `prompt_review: first_round`), controller/tool implementation identity and
server authentication before reserving one model invocation. Reservations are
persisted before side effects; neither a failed generation nor an uncertain
execution is silently retried. The build dependency closure is checked before
model invocation; a candidate reserves its matrix only after image construction.
Complete independently verified matrices alone reach
archive admission. A dominated or power-infeasible result stays in history.

`resume-evaluation` continues a reconciled ready candidate or an already reserved
evaluation without another model call. `Campaign.reconcile_undispatched` requires
the exact candidate, a recovery reason, an idle queue, no candidate queue rows and
no matrix admission. It records the old state and refund before restoring the
ready state, preserving the source and model session. It cannot replay a partial
matrix. Failed or uncertain cells and incomplete builds require explicit
host reconciliation. An interrupted generation cannot be replayed as if it had
never happened. The campaign lock prevents concurrent controllers spending the
same budget. Baseline qualification has a separate budget.

Official [Codex command documentation](https://learn.chatgpt.com/docs/developer-commands#codex-exec)
describes non-interactive execution, JSON events and explicit session resume.
No model call or EDA launch is part of prompt preparation or offline tests.

## Objective variants and failure handling

See [objective profiles](POST_GRT_OBJECTIVE_PROFILES.md) for the preserved peak-temperature
profile, the new timing/overflow-penalty profile, response binding, activation preflight
and continuation after completed candidate failures. Both use the same source scopes
and resource broker. New campaigns use `global_bench.grt_campaign_loop`; historical
campaign configurations and driver authorizations are not rewritten.
