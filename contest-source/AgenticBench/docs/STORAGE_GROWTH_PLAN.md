# Storage growth plan: keep the best-so-far trace

Revision: 2026-10-21, following the user's selective-retention request.
Status: production policy proposed; an opt-in isolated-copy prototype passed a pilot.
No production results, controllers or frozen evaluators have been changed.
This replaces the earlier default of preserving or cold-archiving every unique result.

The final deliverable is complete logs for the best-so-far trace and small records
for other attempts. Reclaim other bulky data during the campaign after verification,
feedback capture and dependency checks. Scores, permissions, budgets and seeds stay fixed.

## Define the trace

For the current timing/overflow-penalty task, lower frozen training `score` is better.
Retain the baseline and every candidate that strictly improves the best verified,
complete training-matrix score. Persist training admission order and score in an
append-only trace index. Pending, incomplete, illegal and failed matrices cannot
set records; held-out results never select them. Equal scores keep their normal
archive/parent eligibility but do not add another permanent full-log entry.

The chronological record sequence is different from the final winner's parent
chain: parallel children may improve from different selected parents. Preserve
all source ancestry as reconstructible diffs, but promise full original logs only
for record setters. Do not silently reduce the trace to its final winner.
A Pareto task has no unique best: it needs an explicit reporting-trace policy
before selective full-log deletion, without changing objectives or parent rules.
This revision defines the current single-score policy.

`global_bench/grt_trace_retention.py` is a host-only pilot, not an activated
controller hook. It requires an explicit isolated-copy marker outside
`/data-ssd`, authenticates sealed history, follows verified original log producers
and applies exact pin-checked deletion plans. `scripts/pilot_trace_retention.py`
exercises the real Queue/Archive with synthetic three-case, three-seed matrices.
Synthetic files and scores do not establish native PPA or production capacity.
The pilot does not implement the filesystem-wide growth ledger, source-diff CAS,
full development/library compaction, or automatic failure classification below.

The 2026-10-21 pilot passed 96 related tests and 40 subtests, including 28 dedicated
retention tests. A synthetic five-matrix fixture (three cases, three seeds) saved
30.22% after including seal overhead. An isolated replay of three existing native
matrices (27 cells) saved 5.169 GiB of allocated space, 12.62% of the copied cohort.
The retained best then passed the original frozen v5 verifier in all nine cells,
with identical complete receipts. This was artifact replay, not new physical
evaluations or qualification of the merged evaluator. Original experiments and
budgets were unchanged; temporary large copies were removed after evidence export.
These sample savings are not a campaign-wide capacity guarantee.

## Retention classes

| Data | Permanent record | Online retention |
| --- | --- | --- |
| Baseline and record setters | Complete original text logs/reports, prompts, model/tool streams, local-check commands/results and all case/seed metrics | Compress and share immutable bytes once |
| Other candidates, including failures | Reconstructible source/diff, parent, change summary, hypothesis, all cell metrics/statuses, task/seed/runtime identities, verifier/budget receipts and short diagnosis | Delete full logs, scratch, large local experiments and unused physical outputs after checks |
| ODB/SPEF/DEF/guides and detailed analysis | No permanent full-result entitlement, even for record setters | Keep only while an evaluator, permitted parent, incremental prefix, rescore or final test needs them |
| Active/uncertain work and current model session | Operational pins, not a second permanent archive | Keep until termination, reconciliation or session closure is confirmed |
| Reproducible build/cache objects | Small receipts and required source differences | Reclaim inactive unreferenced objects; bound total cache size |

For non-record failures, a proposed default is the structured reason and the last
200 relevant log lines, capped at 256 KiB per cell. Preserve complete numeric
metrics and identities; label truncation. Do not automatically cold-archive every
unique experiment or output over 128 MiB. Explicit diagnostic pins may protect
material unresolved harness failures.
Source history means one baseline plus hash-checked diff/blob data that reconstructs
every submitted candidate, rather than another full source tree per round.
Materialize required source into verified staging before an original source checker
uses it; missing trees or unverified diffs cannot stand in for legal candidates.

## Complete logs do not require complete physical results

A record setter's logs cover every case and seed, including real upstream stages
reused by incremental evaluation. Before reclaiming an upstream run, collect its
original logs/provenance into the record's log manifest or shared log store.
Identify reuse and the real producer; never imply the stage ran again. Keeping
only this round's executed suffix is not a complete flow log.
Preserve manifests, original artifact hashes and execution/verification records.
Mark removed physical files; logs alone cannot rerun the old full physical verifier.
Historical best candidates may lose unused ODB/SPEF/DEF while keeping their logs.

## Clean after each terminal matrix

1. Finish independent verification, archive admission and feedback capture before
   classifying a successful candidate. Cleanup never refunds, requeues or changes
   an evaluation. Failed results retain their established accounting.
   Unclassified outputs stay pinned; a missing retention record is not permission
   to delete them.
2. For a new record, publish the complete log bundle atomically, then commit the
   trace entry. Keep source artifacts pinned across an interrupted publication.
   Earlier record setters keep their logs after another candidate wins.
3. Seal the small historical record for every attempt. Non-record full logs and
   unique local experiments have no default permanent archive.
4. Pin active/uncertain work, current eligible parent checkpoints, selected parents
   with in-flight children, baseline input/floorplan, incremental prefixes,
   recursive rescore `physical_source_job`, and frozen final-test work.
   Track ancestry, prefix reuse and rescore references separately across campaigns.
   Every reusable checkpoint also pins its original stage logs and source manifest.
   An equal-score non-record parent may produce a future record-setting child;
   keep those reusable logs until parent eligibility and in-flight dependencies end.
5. Immediately reclaim bulky data with no remaining references or readers. A
   non-record may temporarily keep checkpoints and reusable stage logs while its
   full model/tool trace and unrelated local-check logs are pruned.
6. Compact stopped development/build copies on failures as well as success.
   Preserve the active Codex/Claude session for continuation. At campaign closure,
   keep trace turn logs and compact research history instead of duplicate client
   transcripts and disposable workspaces.

Use terminal events and low-space admission as triggers, not repeated full-tree
scans during every stage. A small host CPU/I/O pool may compress/collect stopped
outputs. Release GPUs immediately after confirmed frontend/local-check termination;
GPU release must not wait for cleanup and does not prove the CPU backend is done.

## Adapt historical readers before deleting their inputs

`grt_bench/archive.py::completed_outcomes` currently rechecks all analysis-manifest
artifacts. `global_bench/grt_candidate_library.py::export_library` calls it for every
historical evaluation, then copies every candidate's native logs. Directly deleting
those files would break the next round.

Before pruning, seal a host-owned evidence record while the full files still verify.
Bind original queue results, independent verifier receipts, task/candidate/full seed,
raw metrics, source legality, manifests/hashes and the admitted archive row.
Keep original receipts unchanged. A separate retention receipt records removed
paths, original hashes/sizes and the cleanup transaction; never rewrite a manifest
to pretend its deleted inputs remain present.

Add retention-aware history reads for already admitted records. Initial admission
and live incremental evaluation still require their full original verification.
A history record cannot admit a new candidate, recreate a missing checkpoint, prove
independent execution or claim deleted evidence was reverified. Host-owned metadata
is the trust boundary; a caller-supplied hash is not authentication.
All candidates remain visible through summaries, code/diffs, metrics and compact
diagnostics. Only trace entries offer full logs; others say `pruned by retention policy`.
Reading libraries become small per-round indexes over shared data, not all-history copies.
Prefer host-side history adapters while keeping the initial archive verifier strict.
Any change to evaluator-bound code needs a separately frozen, qualified release;
do not patch live frozen campaigns or relabel their existing receipts.

## Space admission and safe sharing

Use one filesystem-wide ledger for preparation, development, local checks, builds,
matrices, final tests, temporary DEF restores and log publication:

`free_now >= safety_floor + outstanding_unwritten_growth + new_unwritten_growth`

Already written bytes are reflected in `free_now`; do not deduct them twice.
Matrix reservations own cell suballocations. Initially keep the 150 GiB floor plus
remaining growth reservations, using measured case/stage peaks with headroom.
Collect reclaimable data before deferring new admission; deferral spends no evaluation.
Uncertain leases retain reservations and artifact pins.

Keep ODB-first and exact DEF restore checks for live checkpoints. Share only sealed,
hash/mode-identical files. Writable development, build-source and live outputs need
tested reflinks or private copies, never hard links into frozen objects. Check mounts,
readers and references under locks; publish hash-bound deletion receipts and stop if
references change. Count actual freed blocks, not overlapping hard-link `du` totals.
Cold storage is optional for pinned expensive checkpoints or explicit diagnostics,
not the default destination for discarded results. Bound all compiler namespaces
together. Inventory Docker/BuildKit's actual filesystem; protect active builds and
needed runtime images before reclaiming unused layers.

## Evidence, implementation order and acceptance

The historical **2026-10-21 10:57 Asia/Shanghai** sample showed 158.64 GiB free on
`/data-ssd` and 2222.46 GiB on the root filesystem; these are not live capacity.
Prior sharing already released 46.540 GiB from twenty reading libraries and
13.382 GiB from nine build-source copies; neither is a new reclaimable estimate.

Implement the trace/evidence record and retention-aware readers first, then online
pruning with dependency guards and shared reading indexes. Add global admission
alongside this work, followed by cache limits. Do not start by deleting files
behind the current readers.
Test improvement/ties, parallel branches, inherited logs, publication crashes,
history reads after pruning, tamper rejection, uncertain pins, paired-prefix replay,
held-out isolation and unchanged budgets. Non-record full logs must not reappear in
later reading copies. Final output includes the exact trace and its log manifests.
Measure new unique GiB/round, peak working space and retained size. If every round
improves, all rounds legitimately keep complete logs: promise neither a fixed saving
percentage nor a constant archive size. A completed trace archive can move to a cold
tier without losing its required logs.
