# Maintained local-task compatibility and native HBT qualification

2026-10-16. This release record supports the draft integration proposal. Main
has not been merged. Historical global-task execution is outside the required
compatibility scope; all eight maintained local tasks remain required.

## Native HBT results

Five existing s1 physical inputs were copied and independently authenticated.
Pruned DEF exports were regenerated only in private input views and checked
against their original hashes. Reference and changed-policy runs reused exactly
the same input, floorplan, partition, placement and PDN manifests. They ran HBT
and joint STA only, with no OPT or GPU use.

All **10 native evaluations passed**. The initial policy reproduced the native
default assignment byte for byte on every real problem. The probe reversed
`net_priority` from `pin_count` to `-pin_count`; all five assignments changed and
remained legal. This is an interface probe, not a proposed optimization winner.

| Input | Native reference score | Probe score | Reference + probe container wall time |
| --- | ---: | ---: | ---: |
| bp_fe / ng45 | 1 | 0.972454 | 132.48 s |
| bp_fe / asap7 | 1 | 0.957261 | 128.60 s |
| ariane133 / ng45 | 1 | 0.961378 | 423.07 s |
| bp / ng45 | 1 | 0.979621 | 358.85 s |
| swerv_wrapper / ng45 | 1 | 1.000037 | 358.77 s |

Times include both evaluations, native default-policy comparison and negative
checks. They exclude queue waiting and the prior DEF restoration step. Each
job reserved four CPUs, 32 GiB and zero GPUs in the existing shared resource
queue; every lease completed and every compatibility container terminated.

On each case, the verifier also rejected a changed upstream manifest chain and
a changed native STA value even after its artifact hash was updated. Thus all
ten physical-evidence corruption probes were rejected. Negative copies used
separate metadata inodes; original references and candidate artifacts were
not modified.

`native-qualification.json` contains raw WNS/TNS in ns, scores, timings, original
input jobs and seeds, evidence hashes, image and implementation identities, and
the exact result directories on rafa1-gcr:

```text
/data-ssd/gcr/open3dbench-compat-20261016/hbt-release/<case>/results/
```

The final producer's 446 relevant source/task/platform/harness files were
checked against review-branch implementation commit `23df59ce`. The installed
OpenROAD binary and pinned image are independently included in the identity.
`qualify.py` is the exact in-container driver used for these checks; its mounts
are `/workspace` (committed source), `/trusted` (framework), `/frozen` (verified
input view) and `/output` (private writable results).

These are post-HBT, pre-OPT local timing scores. They must not replace post-GRT
PPA results or enter the current evolution archive. This record qualifies one
physical input per case, not all seeds, other tasks, or a future merged release.

## Regression coverage and merge gate

`validation.json` and the retained compact test logs document passing groups:
285 local contract/prompt/inventory tests; 478 local interface/reference tests
(10 optional skips); 330 current post-GRT/engineering tests; 67 physical-flow
tests; and 93 final HBT/task-modeling tests. Groups overlap and must not be summed
as a unique total. Fixture-based engineering tests are not new native engineering
qualification.

The final merged source/runtime still needs native, case-matched acceptance for
the other seven local tasks and T2/T3/T4 before main can be declared ready.
See [the merge contract](../../../docs/LOCAL_TASK_MERGE_CONTRACT.md). Historical
global-only failures do not waive any maintained local-task requirement.

Superseded qualification attempts were reduced to a compressed diagnostic
archive. The observed free-space increase was about 26 GiB; concurrent formal
jobs may affect that free-space measurement. Final physical qualification data
remain intact. `superseded-retention.json` records the archive and its checksum.
