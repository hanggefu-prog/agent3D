# Experiment storage and build-cache retention

Storage retention is independent of scoring, verifier rules, result reuse,
seeds, budgets, resource admission and timeout limits. It does not turn failed
attempts into successes or silently replay evaluations.

## Experiment modes

| Mode | Use | Retained data |
| --- | --- | --- |
| `full` | Debugging and interface validation | Every candidate's complete physical flow and diagnostics |
| `best` | Large experiment batches | Complete qualified reference, recorded measured best, a distinct accepted incumbent when applicable, and failed/uncertain attempts; compact diagnostics for other successful candidates |
| `best-per-round` | Workspace Codex and Agentic3D searches | The same protection as `best`, plus the latest round for next-turn feedback; compaction runs between rounds rather than waiting for the search to finish |

The qualified reference is retained even if a candidate wins. A reference may
remain the best solution. The measured best and an Agentic3D role-accepted
incumbent are separate records; both are protected. Ties use the existing
recorded best, not a new ranking algorithm. Failed attempts remain available
for diagnosis. A missing metric is never converted to zero.

Local/global smoke wrappers accept `--retention full|best` and default to
`full`, because smoke is a diagnostic workflow. The large-scale scheduled
co-design wrapper defaults to `best`; pass `--retention full` to override it.
Direct Python drivers read `runtime.retention_mode` (default `full`). The
selected mode is recorded in the immutable run configuration.
The global workspace smoke wrapper additionally accepts `--retention best-per-round`.
This mode is not supported by local-task drivers or the standalone cleanup CLI.

```bash
# Append to the existing local or global smoke command:
--retention best

# For a new workspace Codex or Agentic3D search, compact between rounds:
--retention best-per-round

# Preview compaction of one completed experiment; no writes by default:
python AgenticBench/scripts/experiment_retention.py /path/to/experiment --mode best

# Apply after reviewing the preview (never point at a shared batch root):
python AgenticBench/scripts/experiment_retention.py /path/to/experiment --mode best --apply
```

`best` compacts **after the entire search and all role audits finish**. Its peak
disk provisioning must still cover the active search.

`best-per-round` runs synchronously inside the owning workspace driver, with its
exclusive lock held, after evaluation has returned and candidate containers have
stopped. For Agentic3D it also waits for validated local-post and global-adjudication
receipts. No background cleaner guesses whether a run is idle. The current best,
accepted incumbent, reference and most recent round stay intact for the next turn.
After the next round finishes, an older successful loser can be compacted. The
final boundary no longer needs to retain a last-round loser. Failed and uncertain
attempts are never deleted, and a search that stops for attention does not receive
final compaction.

This reduces accumulation of successful losing physical runs, not the temporary
space required to evaluate a candidate. Peak usage still includes selected runs,
the latest result, the next evaluation, agent workspaces, failed attempts and
shared build caches. Admission thresholds and disk quotas are unchanged; this is
not a fixed-space guarantee and is not retroactive cleanup of existing searches.

## Evidence and deletion boundaries

The finalizer reads the trusted completed status, candidate history, recorded
best, measured results and (for globals) independent verification. Only known
large subdirectories of successful non-selected evaluations are removable.
All direct candidate programs, patches, result records, verification receipts,
build logs, image identities and run contracts stay in place. Configuration,
metrics, manifests and native textual diagnostics inside removable trees are
first copied byte-for-byte into `retention/diagnostics/` and hash-checked.

For per-round retention, archives instead live at
`results/retention/round-NN/` (and `results/retention/final/`), making them readable
through the agents' existing read-only `/results` mount. Prompts explain this
location briefly. Each archive has its own plan, inventory and receipt. The exact
submitted patch, added/modified source files and artifact metadata must match their
submission hashes and pass the recorded structural check before a source snapshot
is removed. Prompts, model responses, role receipts, scores and stage timings are
preserved. Previous archived diagnostics are checked again before further pruning.

`retention/plan.json`, `pruned_inventory.json` and `receipt.json` record the
removed paths, sizes and SHA-256 hashes, the protected paths, and the retained
diagnostics. Nested symlinks are inventoried as link text, never followed.
Source/frozen-input overlap, target/ancestor symlinks, nested filesystems and
non-regular artifacts fail closed. A partial cleanup is not automatically
retried; a partial per-round archive blocks later automatic cleanup too. Cleanup
errors produce a separate `retention-error-*.json`; they do
not rewrite a physical evaluation's success or score.

Original native manifests are **not** rewritten to pretend their deleted
artifacts still exist. Compacted losing runs are diagnostic-only and cannot
be resumed or claimed to pass a fresh full artifact audit. The reference and
selected results retain their native physical evidence. Shared inputs, source
snapshots outside an evaluation, the working agent subject, Docker images and
other experiments are not deleted by this finalizer.

## Private build caches

The routing candidate-image builder separately accepts
`--build-cache discard|retain` (default `discard`). This is not a physical-flow
retention or result-cache setting.

In the default cold-build path's `discard` mode, the successful compiler step copies and seals the installed
OpenROAD binary, then removes its private build objects before committing the
Docker layer. Its one-shot private ccache is disabled, avoiding a second copy
of objects that cannot be shared across these isolated builds. CPU/memory
quotas, network isolation and the image-installed binary contract are unchanged.

Only after the installed runtime manifest matches the candidate and a durable
receipt exists, the builder saves `reproduction/` and removes its private
`workspace` and `context/openroad` copies. It retains the build recipe, patch,
source fingerprint, image ID, binary hash, command, native build log and
`cache-retention.json`. Failed builds retain their scratch evidence. `retain`
keeps the old build copies for compiler debugging. No image or shared Docker
cache is automatically pruned in either mode.

For explicit cross-candidate compiler reuse, see
[incremental candidate builds](incremental_candidate_builds.md). Its separate
`--compiler-cache-dir` option keeps shared ccache outside image layers; successful
`discard` additionally removes private `compiler-cache-attempt/build` and
`build-source` while retaining generated-source evidence, staged binary and logs.
It never deletes the shared cache. This does not change physical-flow retention.

These policies apply to new, explicitly versioned code launches. They are not
injected into an already-running frozen experiment. Retrospective cleanup
requires an explicit, path-scoped decision; never use a global Docker prune
as a substitute for ownership checks.
