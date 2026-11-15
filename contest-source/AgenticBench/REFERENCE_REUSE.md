# Reference reuse: ADOPT, EXTEND and FRESH

This interface separates the algorithm starting point, immutable upstream
physical inputs, and the independently measured scoring reference. Sharing a
configuration name is not enough to make two producers interchangeable.

## Lifecycle

| Operation | Physical work | Successful output | Failure behavior |
|---|---|---|---|
| ADOPT | None; re-read and verify evidence | A new adoption receipt referring to an existing immutable record | Stop; no automatic FRESH or EXTEND |
| FRESH | Explicit new reference evaluation | Independently qualified, versioned record eligible for later ADOPT | Preserve attempt; do not register or retry |
| EXTEND | Explicit missing suffix after a qualified native parent | New versioned record, with immutable parent and byte-identical prefix lineage, eligible for ADOPT | Preserve parent and failed suffix separately; do not register or retry |

The registry does not merge physical files or overwrite earlier baselines.
“FRESH/EXTEND becomes ADOPT” means registering a new reusable authority after
independent qualification. A candidate's score, a model statement, or a JSON
`passed` flag cannot create that authority. Candidate finals do not automatically
become references.

The store is a small append-only directory, not a database:

```text
reference-store/
  records/<content-sha256>.json
new-experiment/results/reference/
  attempt.json
  reference_adoption.json   # successful ADOPT
  failure.json              # failed ADOPT, retained if applicable
```

Records bind case, endpoint, task and verifier code, frozen input hashes,
physical configuration, producer source/image, and actual artifact hashes.
Multiple matching records require explicit selection. Requalification occurs
on adoption and before candidate evaluation. Records are content-addressed and
written without replacement. The controller owns the store; never expose it
as candidate-writable storage. Physical results remain in their original
location and are hash-pinned, not rewritten by registration.

Output/store paths cannot overlap physical producer roots, archive roots,
thermal companions, source/input roots, or parent reference authorities.

## Workspace search entry points

`scripts/run_global_task_smoke.py` and `scripts/run_scheduled_codesign.py`
accept:

```text
--reference-policy ADOPT|EXTEND|FRESH
--reference-store /absolute/shared/reference-store
--reference-record /absolute/shared/reference-store/records/<sha256>.json
```

The reference-aligned workspace default is ADOPT. FRESH must be explicit.
EXTEND requires an explicit parent record; a missing ADOPT record is not
permission to compute one. All physical and model limits remain task-owned.
The input bundle must remain inside the configured batch root. A different
bundle import is not silently treated as the same frozen input: its exact
manifest and native physical-prefix hashes must match the selected contract.

Supported native suffixes are post-OPT timing to CTS thermal, post-OPT timing
to routed thermal, and CTS thermal to routed thermal. EXTEND copies only
authenticated parent prefix files into a new run and mounts the copied prefix
read-only. It starts at CTS or global routing respectively; it does not repeat
the frontend or OPT. It never manufactures a missing timing-extract/global-route
manifest or promotes readiness flags. These execution paths require native
qualification before being described as physically validated.

Timing uses the explicit one-pass task and pre-CTS STA, not signoff. CTS
thermal is a separate endpoint with native power extraction fixed to **2
threads**, while the physical backend remains **4 threads**; the CPU container
limit is another setting. Legacy thermal callers retain their original default.

## Read-only catalog CLI

The two versioned placement smoke tasks also default to ADOPT through
`scripts/run_local_task_smoke.py`. They accept `--reference-policy`,
`--reference-store`, `--reference-record`, and optional `--reference-id`.
Their native producer additionally checks the task's unchanged initial program,
the fixed assignment, the controller execution ledger, installed DREAMPlace
Python source identity, the actually loaded initial hook overlay, native
placement logs and DEF completeness. Local placement has no later endpoint to
EXTEND and rejects that choice before dispatch. Legacy local tasks retain their
previous initial-reference behavior rather than silently changing families.

`scripts/reference_catalog.py` offers `qualify`, `register`, `adopt`, and `list`.
It never launches an evaluator, model, container, or suffix. For example:

```text
python AgenticBench/scripts/reference_catalog.py qualify --producer producer.json
python AgenticBench/scripts/reference_catalog.py register --producer producer.json \
  --origin FRESH --contract contract.json --store /absolute/reference-store
python AgenticBench/scripts/reference_catalog.py adopt --record /absolute/record.json \
  --contract contract.json --output /absolute/new-adoption-attempt
```

The physical contract is extracted by a trusted producer adapter, then bound
to a specific consumer task. A caller-provided contract cannot override native
evidence. `list` is an inventory view, not a claim that all records have just
been requalified.

## Historical 2026-09-23 archive

The historical adapter supports placement, native one-pass timing, GRT-only,
and CTS thermal. It binds both the pinned `reference.json` and `FILES.json`,
the requested **work** configuration and separately the expanded **run**
configuration, per-stage artifacts, physical predecessor chains, and the
independent thermal companion. Later DRT failure does not erase a valid earlier
endpoint. Routed thermal remains unavailable in this archive.

Historical endpoint qualification is **not** qualification of today's algorithm
source, Nesterov hooks, or incremental native execution interface. The archive
has multiple stage-specific producer revisions. Current workspace searches
reject historical archive records until a reviewed source-composition or
equivalence bridge exists. They must not replace those algorithms with current
code merely because the settings match. The archive adapter reports this
limitation explicitly and creates no fake stage 15 or stage 75 evidence.

## Accounting and retention

- ADOPT verification wall time is recorded separately from historical native
  runtime. Zero **new** native attempts does not mean the old reference cost zero.
- FRESH/EXTEND reference attempts do not consume the candidate final budget.
- Model requests, generated candidates, dispatched candidate finals, verifier
  rejection, physical attempts, successful evaluations and improvements remain
  separate. Unknown API latency stays unknown.
- Parent lineage remains required evidence. A retained reference must not be
  compacted by a candidate-loss retention policy. Do not relocate or clean its
  physical files without a separate, explicit requalification/import process.
- `best` retention accepts a verified external ADOPT baseline at round zero.
  Its record, parent lineage, adoption receipt and normalized feedback are
  independently checked; none becomes a prune target. Only successful local
  losing candidate directories are eligible, and the authority is checked
  again immediately before applying a plan. `full` remains a no-op.
- A failed or uncertain attempt is never overwritten or automatically retried.

Unit fixtures exercise the interfaces without EDA/model calls. Passing these
tests is not a claim that a new native hook, suffix, or five-round search has
completed successfully on the server.
