# GitHub source publication — 29 October 2026

This publication keeps the maintained integration, the physical-flow upstream
and the previously deployed V4 controller distinct. Publishing source does not
install a runtime, authorize an experiment or change a campaign's accounting.

## Maintained integration

The integration starts from the committed tree at
`96cabd471b72f88bca1155c01274342c5c1d9d04`. It retains the validity-review
history and its compatibility fixes, optional scoring/engineering contracts,
and tested best-so-far log retention. Existing timing tasks retain their own
frozen scoring and evaluator identities. Optional contracts and retention
policies do not retroactively change historical results.

The physical-flow source already includes the placement seeding, explicit
TP-GNN reference/experimental presets, strict die SDC projection, live GRT
analysis exports and the published OpenROAD interior-pin RC repair. The
separate physical-flow update is
[Open3DBench-V2.0 PR #5](https://github.com/xiaosiCU/Open3DBench-V2.0/pull/5).
Integration-only HBT policy and routing callback support are retained rather
than overwritten by that repository's older physical base.

## Frozen controller archive

The actual V4 rollback is preserved separately on
[`codex/frozen-v4-runtime-20261025`](https://github.com/gaocrr/Agentic-Open3DBench/tree/codex/frozen-v4-runtime-20261025/AgenticBench/releases/frozen-v4-20261025).
Its 178 exported source/documentation/test entries match the original
185-entry manifest byte-for-byte; seven qualification records are omitted.
The archive records the original manifest and the selected-path inventory.
It is a source subset with explicit limits, not a replacement for the mainline
or a complete export of deployment inputs and bindings.

The mainline is not the V4 source identity. The original timing task/evaluator,
baseline, source, cases, seeds, scores, budgets and sessions remain unchanged.
The last preserved Flash audit reports stopped incomplete at 5/20 actual
evaluations: four verified matrices, one ordinary execution failure and 15
unspent. No campaign was restarted by this publication.

## Repository boundaries

- The mainline uses committed source, not the dirty integration workspace.
  Subsequent uncommitted fleet, continuation and soft-deadline experiments are
  retained locally and are not promoted by this publication.
- Historical run trees, one-use launch/recovery/retirement operators,
  activation receipts, private configuration, credentials, sessions, queues,
  result databases and physical outputs remain outside this source update.
- Local qualification directories and separately managed sites are ignored.
  The research website retains its self-hosting-only publication restriction.
- Software/contract tests and source/hash checks do not establish a new
  native EDA, GPU reproducibility or deployed-runtime qualification.

## Publication validation

The maintained software gate passed 1751 tests and 319 subtests, with 26 skips.
Two loopback HTTP protocol tests initially encountered the local sandbox's
listener restriction; both passed on a focused rerun allowing loopback only.
Thus all 1753 executed maintained tests passed across the gate and focused
rerun. The root physical-contract suites separately passed 189 tests and 115
subtests; source preflight passed 8/8 tasks and document consistency passed.
The frozen archive's selected file hashes, permissions and manifest identity
were also checked. No EDA, model evaluation, deployment or campaign action
was performed.
