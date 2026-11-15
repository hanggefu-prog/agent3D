# Post-GRT v14 collaboration baseline

This release describes the qualified five-case, three-seed reference used by the
2026-10-16 formal campaign. The 68 deployed Python modules in
`deployed-module-hashes.json` are byte-identical to the published control plane.
The physical flow in the integration branch also includes current upstream main;
it must not be substituted for the frozen baseline source without a new task and
qualification. The branch does not claim a fresh end-to-end qualification of the
merged physical flow.

## Locate results

Set `AGENTIC3D_CAMPAIGN` to the server deployment directory named
`agentic3d-evolution-20261016-formal`. All paths in `reference-index.json` are
relative to that directory. Collaborators with server access can use:

| Case | Directory under `$AGENTIC3D_CAMPAIGN` |
| --- | --- |
| bp_fe / ng45 | `reference/bp_fe-ng45/` |
| bp / ng45 | `reference/bp-ng45/` |
| ariane133 / ng45 | `reference/ariane133-ng45/` |
| swerv_wrapper / ng45 | `reference/swerv_wrapper-ng45/` |
| bp_fe / ASAP7 | `reference/bp_fe-asap7/` |

Each directory has `s1`, `s2`, and `s3` links to immutable `runs/<job>` directories.
Read `verified/result.json` for measurements, `output/analysis/` for extraction
evidence, and `output/run/` for stage checkpoints. Root `result.json` is the
worker receipt. The index binds each independent receipt by SHA-256 and includes
its raw metrics; the complete unchanged task is in `task.json`.

The fixed baseline source is `candidates/<baseline_candidate>/tree`, using the
candidate ID in the index. It is a frozen source snapshot, not a Git branch name.
The source producer was the frontend-scope qualification campaign; formal-launch
import preserved its original provenance and charged no evaluation budget.

## Comparison contract

- Five equal case weights; partition and placement paired seeds 1001, 1002, 1003.
- Fixed I/O seed 1, one OPT pass, post-GRT measurement endpoint.
- Minimize 0.7 normalized WNS deficit + 0.3 normalized TNS deficit, overflow, and
  peak temperature across the 200 active silicon tiles in each case/seed.
- Every case's mean power over seeds must be at most 110% of its baseline mean.
- Compare task, baseline, evaluator, input and image identities before reusing
  measurements. Neither a newer branch nor an identical Docker tag proves identity.
- This is the optimization set; these results do not establish held-out generalization.

## Storage and replay

ODB is retained. Byte-identical DEF exports may be omitted under the storage
receipt; duplicated SPEF and other immutable files share hardlinks. Read-only
ODB inspection is supported. Use the host storage interface for restoration and
verification; do not manually rewrite manifests or treat a missing archived DEF
as failed physical execution. For an independent recheck, with the v14 framework
on `PYTHONPATH`:

```sh
python -m global_bench.grt_storage verify --root "$AGENTIC3D_CAMPAIGN" --jobs JOB_SHA
```

That command needs host write access for temporary restoration and must be
scheduled when the campaign is idle. It does not authorize concurrent mutation
of checkpoints used by a live evaluation. `reference/` is a convenience index;
the original receipts and manifests remain authoritative.

## Validation evidence

Before publication, v14 passed 182 post-GRT host regressions, all 45 independent
baseline/qualification-candidate rechecks after compaction, and two real
incremental replays on ng45 and ASAP7 with identical metrics. Actual released
blocks were 64.15 GiB. The OpenROAD merged-segment pin RC regression also passed
in the pinned installed image during publication; its implementation hash
matches the frozen baseline. Publication checks on the merged branch are listed
in `validation.json`; they do not replace physical baseline qualification.

Machine-local outputs, credentials, generated builds and historical qualification
run trees are not included in Git. Older documentation may link to qualification
records that are retained separately on the experiment server.

The complete recovered legacy control-plane suite is not green. See
`validation.json` and the repository merge plan for the original-workspace
comparison and remaining compatibility gates. The publication PR remains draft.
