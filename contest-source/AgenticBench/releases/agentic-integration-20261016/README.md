# AgenticBench integration, 2026-10-16

Review branch: `codex/agenticbench-integration-20261016` in
`gaocrr/Agentic-Open3DBench`. The original branches and `main` are preserved.
This is a review integration, not activation of a new formal campaign.

## Sources and merge decisions

| Source | Treatment |
| --- | --- |
| AgenticBench `main` (`c682d17e`) | Integration base; current local task interfaces retained |
| `perf/incremental-openroad-build` (`3495ee70`) | All five commits retained, including shared cache, integral timeouts, linked-worktree mounts and invoking-user Docker execution |
| `codex/unify-current-agentic-updates` (`d9d65abe`) | Already an ancestor of main; no second import |
| Open3DBench tested integration (`ce4fb8c8`) | Import maintained physical-flow changes, native pin and current harness/engineering modules; retain AgenticBench-specific fixes and assets |
| `openroad-source` (`741a86a`) | Preserve the unrelated historical branch; do not merge its orphan repository root into the benchmark. Its native lineage is covered by the pinned OpenROAD repository |
| OpenROAD / OpenSTA | Pin `e8c2b9f802b28bbf4a07548ee41f7becc758675a` / `5d8d515edfa5a22165faea5897539da1888fe68d` |

The local HBT task now has native assignment, geometry and post-HBT joint STA
validation. The integration retains the existing 0.6 WNS / 0.4 TNS HBT score,
all eight local task IDs, their initial algorithms and exclusive edit scopes.
Routing tasks use the supported standalone GRT command and authenticated CTS
prefix import; they do not run DRT or replace upstream manifests.

The three engineering tasks retain separate endpoints, source seals, fixed-cell
and connectivity guards, and independent native measurements. Their deployment
layout is explicit. Protected output-path checks also work in the server's thin
Python 3.8 host launcher; normal package installation still requires Python 3.9+.

The dated Nesterov task keeps its old reference families. Its additive
[current-source profile](../../docs/current_nesterov_profile.md) requires a
fresh initial-program reference, binds source/runtime/input/recipe identities,
and explicitly uses the maintained deterministic kernels. Old receipts and
non-deterministic kernels are not relabeled as current evidence.

## Validation

- Maintained task regression gate: **1,153 passed, 11 skipped**, plus 180 subtests.
- Physical flow suite: **520 passed, 1 skipped**, plus 210 subtests.
- Initial program materialization, source scope, patch applicability and prompt
  preflight: **8/8 passed**.
- Native tests run on `rafa1-gcr` under
  `/data-ssd/gcr/agenticbench-integration-20261016/`, in isolated checkouts and
  fresh outputs. They use the existing resource broker and spend no model calls
  or formal campaign evaluation budget.
- All **eight local tasks** passed native functional checks. HBT also passed on
  both ng45 and ASAP7. Three routing candidates were separately built, installed,
  and replayed against fresh references on BP_FE and Ariane133.
- All **three engineering tasks** passed native qualification on BP_FE, seed
  1001, in **409.80 seconds**, including independent physical readback and STA.
- An isolated wheel installation exposes all eight local and three engineering
  tasks and the post-GRT CLI. Required current-reference and native-runtime
  assets are present; see `package-smoke.json`.
- `native-summary.json` records the exact tested source revisions, input paths,
  installed-runtime receipt hashes, native calls and verification results.

The local native matrix exercises partition through fixed placement/HBT and
joint STA, both placement components, an initial and a real Nesterov controller,
and HBT reference/candidate pairs on ng45 and ASAP7. The engineering matrix uses
`bp_fe`, seed 1001: CTS replay/readback/normalization/joint measurement, maximum
clock-buffer-chain executor probes on both dies, and an additional-HBT fixture
with independent delay/STA checks. These are functional qualifications, not an
optimization search or a claim of QoR improvement on every case.

BP_FE did not trigger maze-cost updates, so routing acceptance was extended to
its congested Ariane133 peer. Each reference and identity candidate recorded
550,000 rip-up calls, 217,300,000 maze-cost calls and five iteration-controller
calls. All three reference/adoption pairs passed. Installed candidate builds
spent 850.30, 189.17 and 186.24 seconds in their build containers respectively;
the last two reused the verified compiler cache. The native routing pairs took
about 104 seconds each on BP_FE and 244–247 seconds each on Ariane133.

The current Nesterov reference and a real 0.95 step-multiplier candidate both
passed in 173.35 seconds. Qualification retains the original task ID and
requires the explicit current-source profile described above.

All integration resource leases are released and no integration test container
is running. Four failed preparation attempts remain as diagnostics; the final
checks above supersede them after the launch-environment, deterministic-profile
and thin-host compatibility fixes. Successful build scratch trees were removed
by the verified builder. Existing formal experiment processes were not stopped.

## Current post-GRT campaign

The integrated post-GRT evaluator identity exactly matches the frozen task:
`322c79592b539e312d5386d062fbf727a83a39a99c0da43ab52314c7923a0645`.
All **15 baseline cells** (five cases, three paired seeds) were independently
reverified using the merged verifier and their original frozen physical source.
Every result equals the original verified receipt. Total measured verification
work was **392.90 seconds**, excluding queue waits.

Each compacted output was read into a separate hardlinked view. Missing DEFs
were exported from its authenticated ODBs, checked against original hashes and
used only in that private view. Original results, reference receipts, task,
Pareto archive and campaign budget were not rewritten. Temporary views and
regenerated DEFs were removed after successful verification.

This validates continued execution against the current frozen campaign
contract. It does not qualify the merged physical source as a replacement for
that baseline. A new physical baseline still requires its own frozen task and
qualification. The deployed GPU capacity remains two GPUs with three slots per
GPU; the post-GRT phase-release contract and scheduling regressions are retained.

The release's 68-module hash list describes the historical deployed control
plane. The integration changes one legacy analyzer instruction to expose the
now-executable local HBT task. That instruction is outside the current post-GRT
evaluator identity. The post-GRT scoring/verifier/control implementation remains
compatible with the frozen task.

## Historical suite limits

The complete historical harness suite is **not green**: 2,776 passed, 34 failed,
68 setup errors and 47 skipped (222 subtests). See `historical-suite.json` for
module counts and classification. Most unavailable fixtures require the pinned
September physical recipe; six errors concern the retired archive GRT bridge's
old publication API. Old single-case global extension/importer/reference tests
are outside this integration's compatibility gate, as agreed.

These failures are retained and reported. The current task gate does not delete
or skip them in the full suite, fabricate old reference files, relax runtime
identity checks or promote historical metrics to a new source qualification.
The dated local Nesterov profiles still require their separately provisioned
original source/runtime/input assets; the explicit current-source profile is the
supported route for a fresh merged-source deployment.

## Reproduce the CPU checks

From the repository root, with the Python dependencies installed:

```sh
PYTHONPATH=AgenticBench:.:Place-3D python AgenticBench/scripts/check_task_sources.py
python AgenticBench/scripts/test_maintained_tasks.py
PYTHONPATH=.:Place-3D python -m pytest tests
```

The optional full historical suite expects `AgenticBench` as its working
directory: `PYTHONPATH=.:..:../Place-3D python -m pytest tests`.
Native output paths, source commits and compact acceptance receipts are recorded
in `native-summary.json`; unit logs and the source preflight accompany this file.


## Follow-up: physical global regression comparison

The [global merge comparison](../global-merge-compare-20261016/README.md) rebuilt
one existing native GRT candidate with the merged controller and reran its
five-case, three-seed suffix matrix. All 15 cells, 75 raw metric values, aggregate
objectives and power constraints match the original results exactly. Independent
verification, isolated archive admission and result compaction passed. The
comparison also records the LTO binary-hash difference instead of claiming
bit-for-bit reproduction.
