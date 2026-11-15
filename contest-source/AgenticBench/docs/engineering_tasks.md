# Single-submission engineering tasks

This track is independent of the local OpenEvolve tasks, multi-round global
codesign, and the new post-GRT Pareto task. It adds three development tasks,
not three extra search policies. All developer and judge prompts are English.

## Task and evaluation contract

| Task | Editable contribution | Frozen input | Objective (smaller is better) |
|---|---|---|---|
| T2 `clock_compensation` | Function/helpers selecting real branch compensation | Post-CTS | Maximum same-clock/active-edge/analysis-view original-sink skew |
| T3 `multi_hbt_fanout` | Buffered multi-HBT fanout synthesis/materialization | Post-CTS | Mean worst original-receiver delay over the complete fixed target-net set |
| T4 `cross_die_cts` | CTS coordinator/helpers and native CTS sources | Post-OPT | Same skew definition as T2 |

One implementation per task; `bp_fe`, `bp`, `ariane133`, `swerv_wrapper`; fixed
algorithm seed 1001; one candidate final per case (12 total). Reference creation,
qualification and small functional tests are separate from the candidate budget.
A failed/interrupted dispatched final remains in its once-only ledger and is
not silently retried. A fixed algorithm seed does not imply deterministic LLM
sampling or bit-identical multithreaded EDA.

All three endpoints are **current-stage, placement-estimated RC** on the
independent continuous passive two-die assembly. No GRT, DRT, OpenRCX, power,
thermal or repeated upstream flow is required; these are not signoff results.
T2/T3 apply setup/hold WNS/TNS nonregression (1e-6 ns tolerance) and prohibit new
hold-violating endpoints. T4 setup/hold is diagnostic only, never a rejection
criterion. Physical/PG/electrical legality and complete original-sink coverage
remain mandatory. T4 clock area and buffer count cannot exceed 2x reference;
clock electrical violation count/excess cannot worsen.

T2 fixes all original cells and HBTs while inserting selected noninverting
buffers. Its `clock-compensation-context-v2` exposes measured current timing,
native paths, real Liberty tables, geometry and predicted buffer-chain delays;
predictions are explicitly not post-edit measurements. See
[`context-v2.md`](../engineering_tasks/clock_compensation/context-v2.md).
T3 fixes targets and original receivers, preserves one original branch, and
allows at most three isolated HBT branches per target. Native readback proves
complete disjoint coverage, buffer-only equivalence and distinct electrical
links; the fixed-slew driver/buffer/wire/HBT delay is independently measured.
T4 restores original post-OPT data coordinates before protected standard
legalization for both reference and candidate. Macro/HBT/I/O geometry stays
fixed; data displacement is reported, not optimized by the candidate.

For each legal case, improvement is `100*(reference-candidate)/reference`.
The headline value is the arithmetic mean over **all four cases**. Missing,
running or failed cases make that mean unavailable; never substitute a
successful-subset mean or zero. A zero reference is not a valid denominator.
The independent task-specific judge returns a separate 1–5 score for algorithm
reasoning and code quality, with file/line evidence. It sees anonymous source,
diff and legality evidence, not prior reviews, model identity or QoR ranking.
API/tool failure is ungradable, not score 1. Review scores cannot waive legality.

The paragraphs above describe **scoring version 1**, which is frozen. Existing
qualifications and receipts (with no `scoring_version` key) still verify
under it. Version 1 remains the default for development, qualification and final
queues, so existing three-task commands and remote-only T3 contexts keep their
semantics. Version 2 requires an explicit `--scoring-version 2` on these actions.

## Scoring version 2 (explicit opt-in)

Opt-in qualifications and finals use `scoring_version: 2`. Every native queue
(`clock-final`, `multi-final`, `cts-final`) and `native qualify` accepts
`--scoring-version {1,2}`. Version 2 is bound to the task's
`engineering_tasks/<task>/evaluation-v2.json` profile through
`scoring_profile_sha256`. The profile sits outside the task digest, so v1
receipts are not invalidated, but a v2 receipt fails if its profile changes.
A v2 native result must carry `scoring_version: 2` and belong to a v2
qualification. The aggregate rejects a ledger that mixes the two versions.

`native develop` checks that all qualified cases use the selected version before
starting a model. Its effective instruction and runtime interface render the
selected profile's real thresholds, and a host-owned `scoring-contract.json`
binds the version, profile hash, task and gates to the prompt. Final queues check
that binding when evaluating a development packet; changing the profile or
selecting another version requires a new, matching qualification and prompt.
The original task Markdown and v1 task digests remain unchanged. Campaign JSON
may explicitly set `scoring_version: 2`; omission means version 1, and preflight
rejects qualifications from a different version.

**T4 hard gates.** Setup/hold timing is no longer only diagnostic. The
candidate fails if any setup/hold WNS/TNS regresses by more than the absolute
tolerance (0.01 ns for WNS, 0.1 ns for TNS) plus 2% of |reference|. Other
v2 gates:

- the maximum setup-view insertion delay may not exceed 1.25x reference
  plus 0.01 ns;
- clock area and clock buffer count are each capped at 1.25x reference
  (v1 allowed 2x).

Gate failure is a failed case, not a scored one. Two values are reported but
are not gates: the mean, p95 and per-group skew distribution, alongside the
max-skew objective, and the clock power. The independent measurement does not
produce a power report, so clock power is recorded as
`{"status": "not_measured"}` and never counts as a pass. The 1.1x power bound
applies only when both sides are actually measured.

**T3 receivers and buffer area.** The v2 objective covers **all** original
receivers of each target net: remote ones, plus `original_local_receivers`
on the source die. Remote-only and local-only means are reported as
subsets. A T3 context without `original_local_receivers` is rejected under
v2, so rebuild contexts with the current `native multi-context`. Added buffer
area is bounded by 4x the sum, over target nets, of the smallest buffer area
provided on that net's source die. This bound is recorded in the
`hbt_resource_budget_check` capability.

**Calibration (all three tasks).** Each case reports:

- `qor_noop`: the qualified reference metric, i.e. the unmodified input
  measured by the same endpoint;
- the frozen no-op-relative improvement;
- `normalized_score = 100*(qor-qor_noop)/(qor_expert-qor_noop)` when an
  expert QoR is registered in `expert-reference.json`.

The headline counts failed cases as 0 (`headline_*_failed_as_zero`) and
reports `legality_rate` / `legal_cases` separately. Missing or running cases
still make the headline unavailable. All packaged expert entries are `null`,
so `headline_normalized_status` is `expert_unregistered` until expert
receipts are measured. Never fill in an expert value without a native
receipt.

**Per-capability evidence (capability schema 2).** Each required capability
writes its own `capabilities/<name>.json` record. The record names the check,
the measured values, the threshold, its sources (path and sha256) and a
status of `passed`, `failed` or `unverified`. `unverified` never counts as
a pass. Verification re-reads every evidence file and source hash. The
legalization and measurement-export logs must contain the
`ENGINEERING_PROTECTED_LEGALIZATION` / `ENGINEERING_MEASUREMENT_EXPORT`
markers and no ERROR lines (no WARNING lines either for legalization).

**Evaluator identity v2.** v2 qualifications record `identity_version: 2`.
The hash covers `engineering_bench/**` (Python and Tcl), the transitive
closure of in-repo Python modules it imports, and the explicitly
path-referenced runtime files. Edits to unrelated tracks no longer change the
identity. v1 identities (the broad shared-directory hash) still verify for
old receipts.

**Judge v2** (`native judge --review-version 2 --developer-model <model>`). The judge stays
outside the headline (weight 0) and works as follows.

- **Independence.** The judge and developer model families must differ, using
  the default or a `--model-family-map` family map. The check fails closed
  before any container starts: an unknown family fails too. The only way past
  it is `--allow-same-family-reason`, which is recorded.
- **Redaction.** The judge sees a redacted view. Model names, authorship
  trailers, emails and session IDs are replaced in the changed files, the diff
  and the legality evidence. Replacement keeps line positions so citations
  stay valid, and `.git` metadata is excluded.
- **Citations.** Each citation must include a `quote` that matches the cited
  line.
- **Panel.** At least three fresh samples run, each in a fresh container and
  session. The panel reports the median and dispersion. An invalid or failed
  sample is recorded, not resampled. Fewer than three graded samples gives
  `insufficient_samples` with no score.

Review version 1 remains the default compatible single fresh judge call, including
the existing `--model` alias without `--developer-model`. It does not claim the
v2 family-independence or panel guarantees. Version 2 still requires the explicit
developer identity, independent-family checks and at least three real samples;
neither mode fabricates a review or replaces a failed model call with a score.
`recover_judge_citations` remains for historical receipts only.

## Implemented, qualified history, and remaining limits

The migrated code includes contracts, prompts, source isolation/sealing,
once-only ledgers, native T2 context/executor, T3 branch readback/passive timing,
T4 protected CTS replay, metric reducers and independent judge tooling. The
native scripts are in `engineering_bench/physical`; Tcl helpers are packaged.
The T3 final container includes the previously validated private writable
`/scratch`; source and inputs remain read-only and network stays disabled.

Historical 2026-10-09 campaigns completed all 12 effective candidate evaluations
and three independent reviews for each developer model. GPT-5.6-sol passed
T2/T3/T4 legality on 4/4, 2/4, 4/4 cases; GPT-6 Astra on 4/4, 3/4, 4/4.
The independent GPT-5.6-sol judge scores were 3/2/4 and 3/3/4 respectively.
These are historical results, not new runs performed by this migration. The
Astra batch received richer interface documentation at the beginning; do not
describe it as a strictly prompt-identical model comparison. Generated candidate
implementations, raw runs, failed receipts and credentials are not source defaults.

The legacy adapter keeps the historically qualified Open3DBench base
`5d5dd87f1f2fff8b5d8503138bc99140f0fffbd3` check. A new physical release uses an
explicit `engineering-source-release-v2` receipt binding the committed source
inventory and installed runtime verification. It must complete fresh native
reference/capability qualification; old qualification receipts cannot qualify a
new source, image, input or evaluator.

The evaluator identity now covers the engineering Python/Tcl implementation and
shared runtime/build dependencies. Task identities include interface documents.
T3 qualification reuse rechecks the actual immutable prefix, source, installed
binary, normalized ODBs and context. T2 policy timeouts confirm container
termination; uncertainty is propagated rather than silently releasing capacity.

Native CTS edits use `engineering_bench.candidate_runtime`: build an installed
candidate image with the maintained compiler cache, bind its binary to the
sealed source fingerprint and base image, and verify it before dispatch. Only
CTS execution uses that image. Independent readback, normalization and scoring
continue to use the fixed qualified base runtime. A Python-only submission may
reuse the qualified installed base image.

## Explicit three-seed campaigns

The packaged default remains one seed. A host-owned `ENGINEERING_TASK_ROOT` may
select separately identified contracts for fixed physical input seeds. One
implementation can then be tested on four cases at each of three seeds, for 12
finals per task and 36 across the three tasks. Every cell has its own immutable
input binding, fresh qualification and once-only final ledger. Report the seed
vector, not merely a single ambiguous seed number. Never treat these physical
input seeds as independent model implementations.

`python -m engineering_bench.matrix` dispatches a frozen campaign plan through
the shared transactional resource queue. Each stage-only worker reserves CPU,
memory and storage; no GPU is needed for these fixed post-OPT/post-CTS inputs.
Container labels bind cleanup to the owning job. Failed finals remain terminal,
and uncertain termination retains the lease. Use a new campaign directory for
changed inputs or contracts. The current CLI accepts `--phase prepare` or
`--phase final --task <id>`, with case/seed filters and an explicit worker count.
Qualification time, developer time, native build time and final evaluation time
must be reported separately. Incomplete matrices have no all-case headline mean.

The historical `prepare_development` and per-task `development.md` remain an
explicit preliminary-draft path. For a new qualified campaign use `native
develop`, which supplies the completed interface, requires all four matching
qualifications and takes an explicit `--model`. It neither imports old model
source/sessions nor dispatches finals or judges automatically.

## Isolation and deployment paths

Use Linux/Docker for native work. The host verifies the installed image and
runtime-equivalence evidence. Each task/model gets a private recursive source
clone, build/scratch/output directories and a new CLI session. Only permitted
contribution paths are writable. The trusted framework, inputs and references
are read-only; no host shell, credentials, Docker socket, network or GPU is
exposed to the candidate. The host owns source inventory, sealing, native
qualification, scoring and judge dispatch. Model clients remain on the host.

Pass `--root` pointing to a host-owned deployment directory containing
`engineering-layout.json`. Paths may be absolute or relative to that directory:

```json
{
  "schema_version": 1,
  "release": "releases/qualified-release/release.json",
  "platform": "platforms/nangate45_3D",
  "reference_root": "frozen-references"
}
```

`release.json` uses the existing server-release contract (development source,
installed image ID, hash-verified runtime audit). It is not a tag-only substitute
for physical qualification. `reference_root/<case>/run` holds immutable full
input-through-OPT prefixes. No server names or home directories are hardcoded.
The campaign JSON is an example allocation: replace `/srv/...`, worker aliases
and release identities in an external copy before use; it reserves no resources.

The experiment output layout intentionally retains the tested adapter contract:

```text
experiment/
  native-audit-v3/<case>/60_opt/{bottom,top}/
  stage-reference-v1/<case>/
  local-context-v2/<case>/
  multi-context/<case>/
  qualifications/<task>/<case>/qualification.json
  development/<task>/source/
  finals/<task>/{sealed.json,ledger/,qualifications/,<case>/}
```

First audit inputs (`engineering_bench.prepare_native`), then explicitly create
the normalized stage reference (`native reference`), clock context (`native
clock-context`) and T3 context (`native multi-context`). Use the fixed names above
for reference/clock context, and pass the explicit T3 context directory. Each
step publishes evidence; context creation alone is not task qualification.
`native qualify --task ...` creates independent per-case qualifications without
candidate code or model calls. Select a new output directory on each dispatch;
existing outputs and failed ledgers are not overwritten.

## Commands

From `AgenticBench` (or use installed `openagentic3dbench-engineering`):

```sh
python -m engineering_bench list
python -m engineering_bench inspect clock_compensation
python -m engineering_bench audit-input bp_fe /path/to/frozen/bp_fe/run
python -m engineering_bench preflight engineering_tasks/campaign.gpt56.json /path/to/qualifications
python -m engineering_bench check-qualification clock_compensation bp_fe /path/to/qualification.json
python -m engineering_bench native qualify --help
python -m engineering_bench native develop --help
python -m engineering_bench native clock-final --help
python -m engineering_bench native multi-final --help
python -m engineering_bench native cts-final --help
python -m engineering_bench native judge --help
python -m engineering_bench report clock_compensation /path/to/finals/clock_compensation
```

`report` also accepts the `ledger` directory directly. It verifies task/case/seal
identity; missing cases stay missing. It does not merge two server ledgers by
picking the best result. Cross-host collection must reject duplicate cases and
mixed seals. Supply a complete validated four-case `--evidence-bundle` to the
judge rather than calling a partial host-only view a four-case review.

`preflight` exits 2 when blocked and calls neither a model nor EDA. All `--help`
and inspection commands are offline. Native execution and paid development/
judge calls are explicit separate actions, not consequences of installing code.

## Offline tests

```sh
python -m pytest tests/test_engineering*.py tests/engineering_native tests/test_workspace_codex.py
```

Fixtures validate contracts only, never fabricate physical qualification.
`test_engineering_final_queue.py` runs the real T4 final queue end to end, with
a fake native executor standing in for the CTS stage, snapshot readers and
Liberty areas. It covers success, a v2 gate failure, a measurement failure, an
illegal placement, a resource-budget failure, v1 compatibility, out-of-scope
submissions and post-seal tampering.
Packaging checks must include task JSON/English prompts, Tcl helpers and native
scripts. Preserve the current local/global/post-GRT entrypoints when adding this
independent engineering command.
