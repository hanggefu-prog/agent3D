# Global-task merge comparison, 2026-10-16

This check compares the merged AgenticBench controller at `0a1e5056` with the
existing formal campaign on rafa1. The physical algorithm and paired inputs are
held fixed: candidate
`b085f088129ef96ae85d4dc93850bd9a8c98b036e70befd886b33c34e7d23140`
changes only `OpenROAD-3D/openroad/src/grt/src/fastroute/src/FastRoute.cpp` from
the original frozen baseline.

The comparison runs in
`/data-ssd/gcr/agenticbench-integration-20261016/global-compare-v1`.
The original campaign remains read-only except for ordinary shared resource
broker reservations and compiler-cache updates. No model is called and no
formal candidate-evaluation budget is spent.

## Contract

- Rebuild the exact frozen native source with the merged global builder, shared
  ccache and the same pinned base image. Independently verify the installed
  candidate image and its own executable hash; compare the frozen source and
  observed build inputs with the original build. Record cross-build binary hashes
  separately from source/runtime validation.
- Repeat two concurrent build requests against the new store. Verify that they
  reuse the installed runtime and do not append compilation events.
- Copy baseline evidence and archive state into an isolated test campaign;
  retain its baseline qualification and candidate-parent selection. Clear only
  the isolated copy's previous candidate evaluations before replay.
- Run five cases and three paired seeds through the actual global controller,
  workers, independent verifier, archive admission and ODB-first storage policy.
  The trusted impact proof allows reuse through CTS; GRT, RC, STA, power and
  peak-temperature analysis execute again. This CPU suffix reserves no GPU.
- Compare every raw metric and aggregate objective with the same candidate's
  original verified results. Provenance hashes may change because new execution
  evidence and a new installed runtime are recorded.

The perf branch adds ccache-statistic snapshots to the build recipe without
changing compiler options. This intentionally selects a new cache namespace;
cache-key equality is not a requirement for binary or physical equivalence.
The first test driver required identical cache keys, manifests and binaries,
and retains its diagnostic. Investigation found identical original/generated
source inventories, toolchain identities and all 288 compiler/link command files.
Both builds use `-flto=auto`; LTO section identifiers and linked binary bytes
differ, so this build is not bit-reproducible. The corrected comparison reuses
the completed build, preserves both binary identities, verifies each runtime
against its own trusted receipt, and requires exact equality of all measured
physical metrics. This does not establish general binary or algorithmic
equivalence beyond the tested matrix.

The focused control-plane suite passed **213 tests and 129 subtests**; see
`control-tests.log`. The new build took **216.28 seconds**, compared with
**215.26 seconds** in the original receipt. Two concurrent repeat calls completed
in **16.63 seconds** including source and installed-image checks, with no new
compilation events. These are individual observations, not a universal speedup
claim. All **15/15** physical cells passed the independent verifier. All **75** raw
metric values (WNS deficit, TNS deficit, overflow, watts and peak kelvin) equal
the same candidate's original measurements exactly, without a tolerance.

| Normalized objective | Original | Merged controller |
| --- | ---: | ---: |
| Timing | 0.9990647115672642 | 0.9990647115672642 |
| Overflow | 0.8715956558061821 | 0.8715956558061821 |
| Peak temperature | 1.0 | 1.0 |

Per-case distributions, power constraints and the frozen reference digest also
match. The candidate was admitted into the isolated archive and is feasible
and non-dominated relative to its baseline. This is not a claim about its rank
against all later candidates in the original campaign.

The controller's matrix timer was **1,008.36 seconds (16.81 minutes)**, covering
worker execution and archive admission. It excludes the earlier parent-DEF
restoration and subsequent result compaction. Six CPU cells ran concurrently;
no GPU was reserved. Native threads and seeds match the original experiment.
The first seed's original worker preparation took roughly 293–304 seconds;
this test observed the first EDA dispatch after about 272 seconds. The original
campaign used a different matrix concurrency, so total matrix wall time is not
presented as a before/after speedup.
This is a controlled regression comparison, not a new optimization result or a
qualification of changed physical algorithms against old baseline scores.


## Storage and final state

ODB-first sealing/export proofs, DEF removal and duplicate-file linking completed
without a storage error. The operation released **21.22 GiB** and took **315.67
seconds**. Parent DEFs were restored only in the isolated copy and removed again
after the matrix; original physical results and archive rows were not rewritten.
The shared test leases are all terminal and no test container remains running.
The original campaign and `main` remain unchanged.

See `comparison.json` for all 15 paired metrics, deltas, aggregate checks and
storage statistics; `build-comparison.json` and `binary-comparison.json` record
runtime identities and the cross-build binary difference. The executed driver is
retained as `compare.py`; it is a server-specific audit script that consumes the
separately collected build-input comparison and retained initial attempt, not a
general production launcher.
