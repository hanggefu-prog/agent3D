# Lossless post-GRT result storage

`global_bench.grt_storage` implements the host-side `odb-first` storage policy.
It changes the representation of completed results, not the physical flow,
evaluation budget, task identity, native manifests, metric extraction or verifier.

Scalar campaigns can separately enable [best-so-far trace retention](STORAGE_BEST_TRACE_RETENTION.md)
to retire unneeded historical artifacts and duplicate reading-library logs. That
host-authorized policy keeps strict fresh admission and necessary physical
dependencies; it is not enabled by the lossless `odb-first` setting below.

## What stays on disk

- Keep ODB checkpoints, source snapshots/diffs, configuration, metrics, native
  manifests, logs and verification receipts.
- A DEF may be omitted only after the run's pinned image exports an identical
  file from a retained ODB. SHA-256 must equal the original DEF hash. Store the
  ODB hash, DEF hash, image ID and original file inventory in a durable receipt
  before removing the DEF. DEFs without such a proof remain available as files.
- Deduplicate identical regular output files of at least 64 KiB by hard link.
  This includes SPEF, ODB, netlists and detailed JSON/TSV reports. Compare complete
  bytes and modes; do not normalize timestamps, names, parasitics or numbers.
  There is no separate object-store copy. The index points to retained result
  paths and is rechecked before linking.

Input data and physical constraints remain unchanged. Saving an ODB does not
replace SDC, Liberty or SPEF. A failed or incomplete matrix is not compacted.

## Reading and incremental evaluation

The controller restores the selected parent's omitted DEFs before the unchanged
incremental evaluator validates and copies its prefix. The pinned OpenROAD
exporter must reproduce every original hash before the consumer starts. Missing
ODB/SPEF, changed measurements, a wrong export hash or an incorrect run/image
identity is an error; none receives a weaker verifier path.

The EDA backend still sees the original paths and mounts parent results read-only.
The existing prefix importer copies files into a private candidate directory,
so subsequent algorithm writes cannot mutate shared archived files. The host
removes only reconstructed DEFs when readers finish. Storage operations hold an
exclusive lock and refuse to compact a directory mounted by a live container.
If termination is uncertain, restored files remain available for reconciliation.

The normal verifier remains unaware of storage receipts: it runs against a
fully materialized tree and checks original native evidence. For an external
read-only reader, use `with Store(root).available(jobs): ...`. The context restores
and then re-compacts DEFs. For interactive use, restore a run explicitly:

```bash
python -m global_bench.grt_storage restore --root CAMPAIGN --jobs JOB_ID
python -m global_bench.grt_storage compact --root CAMPAIGN --jobs JOB_ID
# Automatically restore, run the original verifier, compare its full receipt,
# save fresh verification evidence separately, and remove reconstructed DEFs:
python -m global_bench.grt_storage verify --root CAMPAIGN --jobs JOB_ID
```

Only completed, verified queue entries qualify. These commands are host-only;
they are not exposed as agent tools. The candidate reading library's summaries,
code, diffs and text logs remain directly readable.

## Activation and accounting

The controller enables the policy only when `control/result-storage-policy.json`
contains exactly:

```json
{"version": 1, "mode": "odb-first", "deduplicate_min_bytes": 65536}
```

An existing campaign can adopt this storage-only policy with an explicit host
deployment record. Keep the original task, archive and evaluator identity.
Do not change an active campaign's executable deployment or a reviewed in-flight
prompt. New physical results are compacted after the whole matrix has passed
independent verification and archive admission.

The exporter uses the pinned image, two CPU cores, 4 GiB memory, no GPU and a
600-second batch timeout. Export scratch is removed when the batch exits. The
original scheduler's evaluation limits, six GPU slots and early GPU release are
unchanged. This policy reduces retained storage, not the native flow's transient
peak requirement; restore also preserves a 10 GiB free-space reserve.

Receipts live under `control/result-storage/`. Each compaction operation reports
logical DEF bytes omitted, hard links created, filesystem free-space delta and
allocated blocks actually released. The last value counts a removed inode only
when its final hard link is released. Merely dropping a path shared with another
campaign does not count as freed disk space. Storage maintenance errors are
recorded separately and do not relabel a verified evaluation or spend its budget
again. An interrupted operation is recoverable from its durable recipes.
