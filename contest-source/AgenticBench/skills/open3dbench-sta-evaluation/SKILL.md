---
name: open3dbench-sta-evaluation
description: Promote an already selected Open3DBench-V2.0 partition or placement candidate through the fixed backend cutpoint, exactly two 3D timing-optimization passes, and authoritative continuous passive-HBT STA. Use for STA promotion, timing validation, physical-gate auditing, result collection, or comparison-table generation after search has finished. Do not use it to generate, evolve, rank, or repair optimization candidates.
---

# Open3DBench STA Evaluation

Treat STA promotion as an evaluation operation, never as a benchmark search
task. Read [references/contract.md](references/contract.md) before running a
profile or interpreting its result.

## Workflow

1. Identify the selected candidate, its originating task, case-specific
   benchmark input, frozen upstream run, and reference run. Reject ambiguous or
   cross-case combinations.
2. Inspect the applicable profile with
   `python3 -m local_bench.sta_cli inspect <profile>`.
3. Before launching a long run, verify that the exact OpenROAD/OpenSTA binary
   exposes the passive-HBT cutpoint command used by the evaluator.
4. Run in an isolated directory on local `/data` or `/home` storage. Avoid NAS
   for workspaces and recurring log writes; use it only when a required frozen
   input has no local copy.
5. Preserve the selected candidate and frozen partition verbatim. Never ask an
   LLM to edit code as part of this evaluation.
6. Require die initialization, legalization, PDN, HBT, both OPT passes, and STA
   coverage checks to pass naturally. Record failures as outcomes; never weaken
   or bypass a gate to manufacture a successful result.
7. Verify `opt_passes == 2`, a complete global timing graph, zero unconstrained
   endpoints, finite WNS/TNS, and matching frozen-input provenance. For a fixed
   partition, verify the partition artifact hash before and after promotion.
8. Report the result path, status, initial/pass-one/final WNS and TNS, placement
   HPWL and overflow, HBT count, runtime, and any failed physical gate.

## Maintained entry points

- `scripts/promote_placement_case.sh`: select the recorded best dynamic and
  source candidates for one case, then promote both through two-pass STA.
- `scripts/validate_ta3d_objective.sh`: validate a selected TA3D TP-GNN loss
  implementation through partition, fixed placement, and two-pass STA.
- `scripts/run_remote_sta.sh`: lower-level container runner shared by the
  promotion workflows.
- `scripts/summarize_placement.py`: collect compact placement-promotion results
  as JSON and CSV without changing their success status.

Use the root `scripts/` aliases only for compatibility with an older deployed
framework. New automation should call the Skill entry points above.
