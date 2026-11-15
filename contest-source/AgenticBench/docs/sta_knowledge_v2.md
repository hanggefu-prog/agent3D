# Agentic3D STA knowledge-v2

## 2026-09-22: minimal research I/O repair, unchanged algorithm

The algorithm is retained: Global Analyzer plans enhanced tasks and is the only
formal task/skill publisher; partition/placement/joint Local Researchers generate
candidates and advice; the independent Reflector reviews outcomes; Analyzer's
post-evaluation review adopts, rejects or defers the advice. First-improvement,
bounded candidate repairs/replanning, immutable final scoring, seed/imported/
evolved/test-proposals separation, and explicit train-to-test knowledge reuse are
unchanged. No new agent, autonomous environment repair, writable model workspace,
or persistent hidden conversation memory is introduced.

The changes are confined to the existing research I/O boundary:

- Every role's transport wraps its unchanged response schema in
  `{"blocked": null, "response": <normal role object>}`. A role genuinely unable
  to read its source/evidence or use tools returns
  `{"blocked": "specific reason", "response": null}`. It must not create an
  `evidence-access-pause` optimization task or a dummy candidate. This records
  `research-blocked` in `search_status.json` and stops the current invocation;
  it does not turn the infrastructure issue into score 0, spend a candidate
  repair slot, publish knowledge, or trigger automatic environment changes.
  Explicit resume may retry, preserving earlier rejected samples and budgets.
- Before the first uncached research call of each invocation, one read-only,
  ephemeral tool-access probe reads a fresh nonce and computes hashes of exact
  local source, evidence and seed-tool files. It uses the same model, binary,
  workspace and sandbox as research; API/login success alone is insufficient.
  Expected values are not supplied in its prompt. Traces and pass/fail evidence
  live in `research-preflight-v2/calls/`; this is one additional model call,
  included in model-usage accounting, and **zero physical evaluations**.
- The bounded evidence reader covers root/input/floorplan/timing-extraction
  artifacts and the actual `80_route` stage (as well as legacy `80_routing`).
  It prioritizes manifests, failure/status records and metrics over recent
  routine logs. Long artifacts keep head/first-error/tail excerpts; truncation,
  missing evidence and read errors remain explicit. First-error search is
  limited to the first 256 KB, not a claim to have inspected the entire log.
  Readable artifacts retained outside `run/` by compact evaluation, including
  evaluator failures before `run/` exists, are valid evidence too. Only exact
  evaluator-envelope paths and retained-artifact allowlists are inspected;
  other runs are not searched. Indexed text remains checksummed and bounded to
  this experiment; credentials/symlinks are excluded. Evidence never replaces
  physical validity gates or fabricates metrics.
- The final result and budget history are saved **before** evidence collection
  and reflection. Inner results also retain their exact remote run path before
  evidence collection. A read/reflection failure can therefore be resumed by
  retrying that read/review without another physical evaluation. Failed evidence
  reads can refresh on explicit resume; altered, previously valid checksummed
  evidence is not silently replaced.

Candidate syntax/scope/admission failures still follow the existing bounded
bad-candidate path below; a returned physical failure still consumes its physical
budget and supplies normal failure feedback. Infrastructure blocking is separate.
The task schema, method permissions, source limits, scoring and EDA legality
checks are not relaxed. Local and Reflector remain advisory, not knowledge writers.

This changes the implementation fingerprint: freeze a fresh experiment snapshot,
not an in-place edit of a historical/running batch. Local mock tests cover the
transport, tool preflight, evidence and resume logic; they do not prove real
server sandbox access, API availability or physical QoR. This implementation
alone does not deploy, launch or resume experiments.

## 2026-09-17: bounded candidate failure recovery

Candidate admission failure is not an experiment-level infrastructure exception.
Local syntax, schema/advice, scope, global merge, primitive admission and frozen
source patch checks run before any physical submission. Rejected model outputs
receive selection score 0, `valid=false`, `physical_evaluated=false`, and empty
physical metrics. No WNS/TNS/temperature is fabricated. Each candidate slot has
at most three model responses; after rejection the Local may abandon the old
hypothesis and sample a different solution **within the same frozen task**.
After exhaustion, an `invalid_candidate` trial carries the errors into the next
Local attempt instead of terminating the experiment.

If no inner candidate is valid, no final evaluation of the unchanged incumbent
is performed. The rejected task is stored separately, reviewed by the Reflector
and Analyzer, and a new task is planned for the **same remaining final slot**.
There are at most three task plans per final slot (initial + two replans).
At most three cheap physical evaluations total are permitted across those plans;
if they are spent, the Analyzer must choose a final-only task. Invalid model
responses do not spend physical budget. Replanning has distinct paths and IDs,
and cannot overwrite the previous plan, evidence, or learned definitions.
An unchanged candidate (ignoring notes) is rejected before physical evaluation.
First-improvement selection and the trusted final score/gates are unchanged.

If the search bound is exhausted, `search_status.json` records `needs-attention`
and the actual completed/remaining final counts, then the controller pauses with
`SearchBudgetExhausted`. It never labels the unspent final budget as completed.
`history.json` contains only final evaluations; rejected tasks live in
`rounds/NN/rejected.json` or `rounds/NN/replans/KK/rejected.json`. History includes
`round_path` so resumed incumbents can be loaded from a replan. Consumers must
honor this field rather than assume all accepted candidates are in `rounds/NN`.

Each model sampling call persists its prompt contract, all returned responses,
and an in-flight marker in `*.sampling.json`. Resume replays saved responses and
retains consumed validation attempts. An uncertain/failed candidate API call
pauses for inspection, not silent resampling. A known failed non-candidate
planning/review API call also pauses the current invocation; explicit later
resume can retry it. Filesystem, transport and evaluator-dispatch exceptions are
not converted into candidate scores. A structured physical failure still counts
against the corresponding physical budget and can be followed by a new candidate.

The ten-final Codex baseline uses the same admission/error distinction and up to
three groups of three candidate responses per final slot, without proxy calls.
Its rejected groups are under `rounds/NN/resamples/KK`; it does not receive
Agentic3D's task/knowledge/reflector machinery. Summaries separate invalid
candidate groups, rejected tasks, cheap physical evaluations and final results.

This is a fingerprinted controller change: use a newly frozen experiment version
or an explicitly audited migration, **not** an in-place edit of running snapshots.
Historical ledgers and failed experiments are not rewritten. Mock regression
tests do not demonstrate new physical QoR or imply remote deployment.

## 2026-09-16: Analyzer-owned knowledge evolution

`knowledge_authority=global-analyzer-only-v1` separates evidence production from
formal knowledge publication. There are still three conceptual roles. The
Analyzer has planning and post-evaluation review phases, not a fourth autonomous
agent. All calls retain the configured model and read-only, ephemeral execution.

| Role | Outputs | Authority |
| --- | --- | --- |
| Local Researcher (partition/placement/joint) | `hypothesis`, `program`, `observations`, `suggestions` | Candidate within the frozen task; advice only |
| Reflector | `assessment`, `observations`, `suggestions` | Review actual execution/outcome; advice only |
| Global Analyzer | Planning: enhanced task and optional formal updates. Review: `assessment`, `decisions`, `skills`, `tasks` | Sole author/publisher of formal reusable knowledge |

An observation has `finding`, `evidence_paths`, `uncertainty`. A suggestion has
`target` (`task` or `skill`), `change`, `rationale`, `evidence_paths`, `uncertainty`.
These are concise structured advice, not complete task definitions or skill
packages. Each role may return empty arrays; at most six observations and six
suggestions per response, eight evidence paths per item, 2,000 characters per
observation/suggestion text field (8,000 for the assessment/hypothesis) and 24,000
characters for the advice envelope. Candidate source code
is **not** subject to that advice cap; existing patch admission limits still apply.
Schemas and runtime validators reject formal task/skill fields from Local and
Reflector responses, including cached responses.

The round sequence is:

1. Analyzer plans and freezes the enhanced task.
2. Local proposes candidate + advice; the driver evaluates within the existing
   first-improvement inner budget and submits the selected candidate to final STA.
3. Reflector reviews the task, local advice, proxy results, actual final outcome,
   logs and exact candidate patches. A workspace evidence artifact contains all
   attempted merged candidates and the selected candidate with hashes, avoiding
   attribution of the final outcome to an unselected attempt.
4. Analyzer reviews this evidence with reference/incumbent metrics, global
   history, budget and the allowed knowledge catalog. Driver-issued suggestion
   IDs must each receive exactly one `adopt`, `reject` or `defer` decision + reason.
   Adoption links to formal `skill_ids`/`task_ids` in that response; every formal
   update must be linked, and rejected/deferred advice cannot publish anything.
   Advice may be narrowed, rewritten or consolidated. Publication does not prove
   generalization or causal improvement; evidence limitations must remain explicit.
5. Only Analyzer-authored definitions enter the knowledge store. The next plan
   sees recent advice/review decisions as within-run feedback, plus the permitted
   catalog. Method guidance remains integrated into the enhanced task for Local.

Reflection and review run after **every** final attempt, including failures and
the last round. This adds one Analyzer model call per round (excluding bounded
format repairs), **zero additional physical evaluations**. A review interruption
after history is saved resumes from cached candidates/results/reflection without
repeating STA. Model-call accounting includes `global-analyzer-review-v2`.

Raw advice, exact candidates and review decisions remain round/evidence artifacts.
Transfer packages contain formal knowledge/provenance only; they do not copy raw
feedback or candidate evidence. Seed/imported/evolved/test-proposals separation is
unchanged. Frozen tests still use normal within-run outcome/advice feedback, but
new formal task/skill bodies remain audit-only and cannot enter the active catalog.
This is workflow enforcement, not a hostile-code filesystem boundary or a semantic
proof that prose advice contains no embedded instructions.

Use a fresh run directory: the authority policy, role prompts and new validation
module are fingerprinted. Historical runs/imports are not rewritten or retroactively
relabelled as Analyzer-approved. Existing explicit import lineage/seed checks apply.
No remote deployment or experiment launch is implied by this implementation.

## 2026-09-16: enhanced-task handoff

The maintained knowledge-v2 runner now uses `context_mode=enhanced-task-v1`.
This is the default handoff within knowledge-v2, not an ablation switch. The
legacy protocol and frozen historical experiment copies are not rewritten.
Use a fresh output directory: source-contract/implementation fingerprints reject
resuming an old full-library run with this code. No model or physical experiment
is launched by this interface change.

Knowledge is **stored separately, integrated for execution**:

1. Publish a compact `knowledge/catalog.json` from pre-given seed and only the
   records allowed by the existing independent/train/test policy. Descriptors
   include revision-qualified references, provenance, and resource checksums;
   full skill/task bodies stay on disk for on-demand analyzer inspection.
2. Global Analyzer produces the existing eight-field `local_task`, integrating
   the relevant methods, applicability, limitations and failure lessons into
   `procedure`/`validation`. It must be self-contained, not a request to read a
   collection of skills. `specialist_instruction` is only concise round-specific
   emphasis, not another copy of the procedure.
3. The plan adds `knowledge_refs`: zero to four `{ref, reason, attachments}`
   selections from that catalog. Empty selection is valid. Attachments are exact
   catalog paths, at most eight total; bodies are **not** injected into prompts.
   They are for optional long diagnostics/scripts/references, not required
   reading. The analyzer cannot cite excluded test proposals or knowledge first
   proposed in the same response. New ideas can be integrated directly into the
   task, explicitly marked as hypotheses.
4. The driver validates and freezes `rounds/NN/enhanced_task.json`: the original
   task, hypothesis/evidence, concise guidance, selected source provenance,
   optional attachment paths/checksums and inner budget, plus a packet checksum.
   The task is capped at 16,000 characters and the handoff at 24,000; these are
   serialization-character limits, not token budgets. Overflow is rejected for
   bounded format repair, never silently truncated.
5. Local Agent receives this single packet, necessary metrics/budget/evidence,
   recent measured outcomes, its in-scope incumbent and current inner feedback.
   It no longer receives full knowledge records, the analyzer's full state,
   duplicated task/plan bodies, or a mandate to read all seed skills. It may read
   selected attachments on demand; it is instructed not to scan the library.
6. Local proposals and post-evaluation reflection contribute observations and
   suggestions only. Analyzer review is required before they become formal
   reusable knowledge (see above). Empty advice/updates are normal. Reflection
   receives the frozen task, actual outcome, candidate evidence and compact
   catalog; it does not need the full library inline. Updates never change the
   current packet.

The original task schema, physical evaluation budgets, first-improvement rule,
candidate patch guards, and final STA acceptance are unchanged. The driver
checks references and evaluation contracts, not whether the analyzer's method
synthesis is scientifically good. `procedure`/`validation` remain guidance,
not executable evaluators. A selected method cannot expand permissions.

`seed`/`imported`/`evolved`/`test-proposals` remain physically separate. Frozen
test proposals are absent from the selectable catalog and future local packets.
This remains cooperative context isolation, not an adversarial filesystem
sandbox. This STA-specific runner has not thereby become a thermal or generic
workspace-global multi-agent runner. Protocol tests do not establish QoR gains.

## 2026-09-10 interface repair

The contract-fix batch is a fresh seed-only snapshot, not a resume of the failed
20260909 model traces. The judge, proxy allowlists, source limits and five-final
budget are unchanged. Identifier patterns and metric enums are now exposed in
the output schema; all roles receive complete metric/skill/final-task rules.
Bounded three-attempt format repair retains the rejected response and cumulative
errors. Evidence citations prefer local_file; original log paths resolve only
through a unique checksummed experiment-local index entry. Ambiguous or escaping
paths remain rejected. Validation failures are saved alongside each response.

This is an **opt-in new experiment protocol** for
`scripts/run_sta_codesign_comparison.py --mode agentic3d`. Existing legacy
experiments, evaluators, references and launchers keep their behavior. Do not
resume a legacy output using v2. Do not pool v1/v2 measurements as one baseline.

## What changes

All roles receive the complete case-correct global instructions, exact source
context, 0.6 WNS / 0.4 TNS objective, immutable validity gates and remaining
five-final-evaluation budget. The global analyzer receives the incumbent's full
candidate, reference/incumbent metrics, measured outer history, inner feedback,
a compact knowledge catalog and a local evidence index. The driver reads bounded text tails from this reference
and completed runs only, including partition, placement, HBT and OPT artifacts.
Missing files, read timeouts and truncation are explicit; unavailable evidence
must never be invented. Logs are data, not instructions.

Local researchers use the enhanced task as their self-contained entry point,
with selected attachments available on demand. They can develop new candidate
algorithms and suggest reusable methods, diagnostic capabilities and task changes
rather than remaining restricted to seed templates. Formal task/skill packages
are authored only by the Analyzer. A post-STA reflection reviews the actual outcome,
candidate evidence and logs, and suggests lessons, failed hypotheses and limitations. Candidate
skill syntax validation is not a proof of algorithmic quality; provenance
distinguishes proposals from measured outer outcomes.

Each model call is ephemeral: no hidden conversation memory is reused. The
driver explicitly supplies within-run feedback and permitted knowledge on each
call. This also makes frozen test knowledge exclusion meaningful. Model prompts,
responses, commands and outputs remain auditable in each role's `calls/` folder.

### Evolving tasks without evolving the judge

A `local_task` has an arbitrary reusable id, stage (`partition`, `placement`,
`joint`), source subset, procedure, validation description and objective. It does
not have to name an existing task template. Researchers can propose revisions
for future tasks; the active objective is frozen before measuring a candidate.

Its `evaluation` selects a **measurement primitive**, not a task identity:

| Primitive | Measurement | Mutable local objective | Budget |
| --- | --- | --- | --- |
| partition | Maintained partition helper | Positive weighted normalized cut/balance metrics | At most 3 inner attempts |
| placement | Maintained placement helper on incumbent partition | Positive weighted normalized HPWL/per-die overflow metrics | At most 3 inner attempts |
| final | Full frozen-input pipeline | None: original STA objective | One candidate, one of the 5 final attempts |

Novel coupled partition/placement tasks use `final`. There is no privileged
agent-generated evaluator code. New metrics requiring a physical measurement
backend cannot be silently invented; that needs a separately reviewed adapter.
All source scopes remain within the five globally allowed files. Cheap helpers
retain their existing patch limits; final-only tasks get the full global limits.

For a cheap task, cost is the declared weighted sum of
`max(candidate_metric, scale_floor) / max(incumbent_metric, scale_floor)`;
weights sum to one, both values must be finite/nonnegative, score is reciprocal.
Stop on the first score exceeding `1 + inner_tolerance`. If no strict improvement
appears, submit the best valid inner candidate, not a fabricated proxy win.
Final acceptance still depends only on the original two-pass STA score. Failed
final results consume budget. Incomplete remote runs require inspection, not
automatic free re-execution. Proxy/model-call counts are separate from final STA.

## File-level knowledge separation

```text
RUN/
  run_config.json                 protocol, phase, split and input knowledge hashes
  knowledge/
    manifest.json                 immutable experiment knowledge identity
    source_contract.json          source, prompt, implementation fingerprints
    seed/snapshot.json            pre-given skills/tools/task templates only
    imported/package.json         optional checksummed training package (never modified)
    evolved/round-*-plan/          Analyzer-planned tasks/skills and provenance
    evolved/round-*-review/        Analyzer-reviewed tasks/skills and provenance
      record.json                 checksummed authoritative record
      skills/ID/SKILL.md           readable skill plus scripts/references
      tasks/ID.json               readable task definition
    test-proposals/               frozen-test updates retained ONLY for audit
  agent-workspace/
    Place-3D/...                  private snapshot of the allowed source files
    knowledge/seed/               materialized pre-given files
    knowledge/learned/            revision/{skills,tasks}/ID/...; permitted records only
    knowledge/catalog.json       compact descriptors for analyzer selection
    knowledge/active.json         explicit active reusable records
    evidence/                    bounded log/result snapshots and index
  rounds/NN/
    enhanced_task.json           frozen handoff
    inner/MM/response.json       Local candidate + observations/suggestions (not knowledge)
    inner/MM/merged_candidate.json exact merged candidate submitted/considered
    reflection.json             Reflector advice (not knowledge)
    knowledge_review.json       Analyzer decisions + formal updates
    outer-candidate/            selected candidate; results/plan also retained in this round
  *-v2/calls/                    model traces (never exported)
```

Default `independent` mode never discovers previous runs or imports learned
knowledge. A new output directory gets fresh records and fresh model contexts.
Use `--knowledge-seed OLD_RUN/knowledge/seed/snapshot.json` to repeat the exact
pre-given file snapshot, without importing OLD_RUN's evolved information.
Resume rejects changes to seeds, source, prompts, implementation fingerprint,
split, import digest or adaptation policy. Never edit saved snapshot files.

This is controlled knowledge/context isolation, **not a hostile-code filesystem
security sandbox**. Model tool calls run read-only in a private workspace and
are told not to access other runs or personal memories. For adversarial isolation,
run the controller in a separate OS/container identity without other experiment
directories mounted. Do not install experiment-generated skills in user-home
skill directories. Fixed initial information does not promise bit-identical
LLM outputs or GPU numerical results; record multiple independent replicates.

## Training and testing

Provide an explicit JSON split, e.g.:

```json
{"train": ["bp", "swerv_wrapper"], "test": ["ariane133"]}
```

Modes:

- `--knowledge-phase independent`: seed only at initialization; evolve within run.
- `--knowledge-phase train --knowledge-split SPLIT`: learn within a training case.
  Optionally import a previous training package to accumulate across training cases.
- `--knowledge-phase test --knowledge-split SPLIT --knowledge-import PACKAGE`:
  reuse training skills/tasks; default registry stays frozen. Test-generated
  knowledge is stored in `test-proposals/` but excluded from later active knowledge.
  Normal candidate search and feedback within the five-round test still occur.
- `--test-knowledge-adaptation`: explicit alternative allowing within-test
  knowledge evolution. It never writes back to the imported training package and
  cannot export transferable knowledge. Report this as a different test protocol.

Imports check hashes, pre-given seed identity, training provenance, case membership
and disjoint train/test sets. Training packages contain skill/task records and
their provenance only, not candidate code, model sessions, raw logs or physical
run artifacts. Instructions prohibit case-specific memorization in reusable
content; human review is still needed to assess genuine generalization.

Export after a completed training run:

```bash
python scripts/sta_knowledge.py export TRAIN_RUN TRAIN_PACKAGE.json
python scripts/sta_knowledge.py inspect TRAIN_PACKAGE.json
```

Or pass `--knowledge-export TRAIN_PACKAGE.json` to the training driver. Exports
are immutable: a different package cannot overwrite an existing destination.
For multiple training cases, explicitly import the preceding training package
into the next training run. Its cumulative lineage is checked on test import.

## Launch example

Use a new, experiment-specific local output AND remote comparison directory;
keep the selected case's existing qualified frozen bundle/reference and exact
helper/profile. The placeholders below must be replaced with verified paths.
This command makes real model calls and five final STA evaluations; tests below do not.

```bash
python scripts/run_sta_codesign_comparison.py \
  --mode agentic3d --protocol knowledge-v2 \
  --case bp --gpu 0 --rounds 5 --model gpt-5.6-sol \
  --output-dir NEW_TRAIN_RUN --agent-workspace CLEAN_SOURCE_SNAPSHOT \
  --remote-host QUALIFIED_HOST \
  --remote-bundle CASE_FROZEN_BUNDLE --reference-run CASE_REFERENCE_RUN \
  --remote-comparison NEW_REMOTE_COMPARISON \
  --remote-helper QUALIFIED_HELPER --remote-profile-config CASE_PROFILE \
  --knowledge-phase train --knowledge-split SPLIT.json \
  --knowledge-export TRAIN_PACKAGE.json
```

For testing, select the held-out case and **its own** reference/bundle/profile,
use a new output/comparison directory, and replace the last two lines with
`--knowledge-phase test --knowledge-split SPLIT.json --knowledge-import TRAIN_PACKAGE.json`.
Never reuse the training case's reference for a test design.

The v2 runner is currently Agentic3D-specific; Codex legacy is not silently
changed. A matched v2 comparison should supply the same public task/evidence
information to the Codex arm and report extra model/inner evaluation costs.
Knowledge-v2 is a new protocol, not evidence yet of improved physical results.

## Verification

`pytest -q tests/test_research_feedback.py tests/test_enhanced_task.py tests/test_sta_knowledge.py tests/test_sta_codesign_driver.py
tests/test_sta_codesign.py tests/test_codex_exec.py`

The v2 end-to-end test mocks model/physical calls but exercises real candidate
materialization, scope checks, patch application, dynamic task dispatch, merging,
knowledge reuse, five-round accounting and no-repeat resume. Additional tests
cover role authority, advice bounds/evidence, adoption provenance, reject/defer,
final-round review and interrupted-review recovery, training/test separation,
lineage, tampering, bad scopes/metrics, symlink escape in evidence, and legacy
default compatibility. These are protocol tests,
not physical STA evaluations or empirical generalization tests.
