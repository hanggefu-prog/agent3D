# Review and production v5 integration

This is an isolated integration branch, `codex/merge-validity-v5-20261021`.
It does not activate a release or change the two frozen server campaigns.

## Inputs and merge strategy

- Repository: `gaocrr/Agentic-Open3DBench`; unchanged main base
  `d798d13439d3c30ec612b0423f15ed263ee97154`.
- Claude review: `review/benchmark-validity-20261019`, commit
  `28965ee340db44b9f2b6301a42348628e20c17ed`.
- Production source: the frozen rafa1 v5 framework at
  `/data-ssd/gcr/agentic3d-harness-timing-20261021-v5/AgenticBench`.
  Its sparse runtime export has 170 files, all byte-identical to the retained
  local continuation fixture. The full main checkout supplies the other files;
  the sparse fixture is never treated as a complete replacement repository.
- OpenROAD remains pinned to `e8c2b9f802b28bbf4a07548ee41f7becc758675a`.

Commit `8000e75e` first preserves production's 32 changed/new files on main.
The subsequent real merge retains Claude's history and resolves five conflicting
files: the campaign, round renderer, contract, Pareto aggregation and round docs.
No shared physical-design algorithm or native submodule revision changes.

## Preserved behavior and review corrections

The integration preserves timing-only schema 3, the peak-temperature schema 2
contract, Claude and Codex session continuation, writable selected-parent
development with strict final export, installed candidate runtimes, and the
budgeted GPU tool. Build/deployment failures before admission spend no evaluation.
Unknown or uncertain execution retains the reservation and resources for
reconciliation. Only explicit, terminated candidate execution failures may end
as ordinary failed evaluations. GPUs are released only after confirmed frontend
or local diagnostic termination.

The review additions include opt-in scoring v3 on the original peak endpoint,
local matrix profiles, routing scoring v2, import-derived incremental proofs,
adapter usage/cost, ranking, documentation checks and engineering scoring v2.
The three global scoring contracts dispatch explicitly by schema, endpoint and
scoring fields; old task values are never silently upgraded.

Corrections from the two-agent review are included:

- All patch consumers share canonical path parsing. Noncanonical aliases and
  duplicate wrapper sections are rejected. Wrapper edits require a source root;
  pure native scope-only preflight remains compatible.
- Engineering defaults remain v1. Explicit v2 qualifications drive the actual
  prompt, thresholds and final-queue scoring binding. Old judge invocations
  remain usable; v2 retains independent model review requirements.
- Final hypervolume obeys the same budget limit as AUC.
- Native noise estimation requires separately operator-authorized host roots,
  matching source/task/complete seed/runtime evidence and distinct physical
  leases. Copied paths are not independent experiments. Synthetic inputs are
  explicit and cannot be represented as native qualification evidence.
- An offline held-out digest no longer claims to prove preregistration. The
  production final-test controller retains its real pre-evaluation sealing.
- Exit 137 is not proof of infrastructure OOM and is excluded from automatic
  retry classes. Existing frozen retry authorizations are not expanded.
- A Claude response from an observed wrong session cannot trigger transport
  retry. Exact zero-usage client connection-loss notifications remain accepted
  only within the original strict model, session and pending-tool checks.

Independent overflow remains an optional one-sided audit, not a trusted complete
replacement for routing overflow. A reference-calibrated correction is not
silently introduced into any objective.

## Qualification boundary

Use `python AgenticBench/scripts/test_maintained_tasks.py` for the repository's
existing maintained local, engineering and post-GRT gate. Integration tests add
cross-contract checks, patch attacks and physical-attempt provenance failures,
plus guard tests for Claude identity, free development, build accounting and
uncertain GPU leases. These are local synthetic/contract tests, not fresh EDA
or model optimization experiments.

The optional full historical suite also includes retired workspace/reference
families whose frozen configs and native interfaces are no longer present.
Keep its failures visible and compare them against the clean review branch;
do not weaken current verification or claim the full historical suite is green.

The new evaluator source has a new identity even where old scoring outputs are
unchanged. Deployment needs a separately frozen release and native baseline/
candidate qualification. Do not migrate existing campaigns or relabel old
receipts using this branch's evaluator.

## Storage follow-up

[Storage growth plan](STORAGE_GROWTH_PLAN.md) now proposes complete logs only for
the best-so-far trace, small historical records for other attempts, online pruning
after verification, and temporary physical-checkpoint retention based on dependencies.
Shared reading/source content, writable-copy isolation, filesystem-wide growth
reservations and cache limits support that policy. It replaces the earlier default
of preserving or cold-archiving every unique result. These storage features remain
proposed rather than implemented by this merge.
