# Best-so-far storage retention

This is an **opt-in, host-only policy for scalar post-GRT tasks** whose objective
is `score`. Existing campaigns keep their legacy behavior unless a maintainer
initializes an exact root/task/evaluator authorization. Multi-objective Pareto
tasks require a separately defined reporting trace and cannot enable this policy.

## What remains

A record setter is the baseline or a feasible, independently verified candidate
whose score is strictly lower than every earlier feasible admitted score. The
trace follows admission order, rather than the final winner's parent chain.
Equal scores remain in the archive but do not create another full-log record.

Every admitted candidate keeps its frozen source, parent/reference diffs, exact
case/seed measurements, aggregate score and concise diagnostics. Record setters
also keep complete text logs in an immutable content-addressed host store. Native
log provenance identifies the original producer of each reused upstream stage;
copied stage logs are not presented as a new execution. Prompt, model, tool and
safe local-test text can join this trace. Provider credentials, private settings,
host authorizations and seal keys are excluded.

The baseline, current training frontier and their verified physical checkpoint
dependencies remain pinned for incremental evaluation and final testing. Held-out
results, explicit recovery evidence and unknown or unsubmitted failures remain
protected. An older record setter may lose unneeded physical databases while its
complete text trace remains available.

## Historical reading versus fresh verification

`RetainedHistory` authenticates an already admitted result against its original
source, archive prefix, queue rows and small independent-verifier receipts.
Historical records are explicitly marked `compact_history_only` and
`may_admit_new_candidate=false`. They cannot substitute for a fresh native
evaluation or admit a new candidate. Native archive admission remains unchanged.

New reading libraries expose an explicit retention status. Non-record candidates
have code, diffs, exact metrics and short diagnostics; their log directory explains
why bulk logs are absent. Record-setter libraries link immutable log blobs,
including their original reused-stage provenance. Code and logs share read-only
blobs across new libraries, without changing permissions on original sources.
A missing historical seal uses the original strict native checks; a corrupt seal
fails closed.

Inactive old libraries can retire only manifest-listed duplicate bulk logs. Their
original manifests, code, diffs, metrics and indexes remain unchanged. A separate
signed receipt and `RETENTION_NOTICE.md` make the retirement explicit. The old
strict library loader rejects that incomplete library; later rounds receive a new
complete compact library. Intact legacy libraries remain compatible.

## Host lifecycle and activation

Use a reviewed release and pause the controller at a terminal round before
activation. The host must bind authorization to the resolved campaign root,
frozen task digest, evaluator identity and `best-so-far-full-logs-v1` policy.
`RetainedHistory.initialize` creates an owner-only trust directory and private
seal key. Never expose this directory in an agent mount or copy its key into a
prompt, candidate or reading library.

The campaign loop calls `after_round` before preparing another round and after
each terminal result. It preserves the existing verified-matrix compaction helper,
seals newly admitted history, then runs exact native-result and old-library
retirement plans. Failed or unsubmitted attempts receive short diagnostics without
changing admission or evaluation accounting. Unknown failure evidence stays pinned.
Final held-out testing is not followed by an automatic destructive cleanup.

Both retirement services require an idle queue, terminal campaign state,
campaign/storage/history locks and a successful live-container mount inventory.
Each plan binds exact file hashes, modes, inode/link metadata and dependencies;
signed progress and completion receipts record the operation. Reported savings
count allocated blocks released at the last link, rather than apparent file sizes.
Only sealed non-record rounds may additionally retire recognized model-stream,
GPU-check and scratch text logs. Source, result summaries, prompts, budgets,
private configurations and recovery files stay protected.

Missing evidence, changed mounts, modified receipts or an uncertain interrupted
deletion stop maintenance. They never refund, recharge, replay or reclassify an
evaluated matrix. Reconcile the exact operation before continuing; do not weaken
the verification checks or repeat an old launch wrapper.

Implementation: `global_bench/grt_history_retention.py`,
`grt_retention_gc.py`, `grt_library_retention.py` and
`grt_retention_lifecycle.py`. The native evaluator identity is unaffected.
