---
name: optimize-mol-partition
description: Optimize and validate Open3DBench partitioning for unified Place-3D, legacy Place-MoL, and LoL-style assignments using GNN, min-cut, max-cut, hMETIS, or Mt-KaHyPar; produce a complete, balanced, reproducible artifact for downstream layout.
---

# Optimize Open3DBench partition

## Version-adaptive repository contract

Before inspecting or changing code, read the `repository_context` path from the
generated `handoff.md`. Use its `stages.partition` object as the authoritative
source for source/config paths, compile/evaluation commands, log/results roots,
metrics, and correctness gates. In unified releases this normally maps to
`Place-3D/src/partition`, `Place-3D/config/partition`, and
`Place-3D/results`; in legacy releases it maps to `Place-MoL/src/partition`.
Never assume the legacy directory exists. Read any `AGENTS.md` governing a
selected source path before edits.

Read `scope_mode` and `scope_contract` from the handoff. In `enforced` mode,
the signed contract is a hard limit: modify only its listed paths and obey its
file types, operation permissions, and line/file budgets. In `direct` mode,
the framework does not restrict repository paths or integrate a candidate
worktree; the stage boundaries in this Skill remain behavioral guidance. In
either mode, invoke declared compile/evaluation commands only after replacing
every `<placeholder>`, and read logs/results listed by the adapter.

Infer the partition representation from the adapter-selected code and current
artifacts; do not infer it from this Skill's folder name. Supported contracts
include legacy upper/bottom macro JSON, unified instance-assignment CSV, and
LoL-style movable-instance assignment. Record the selected contract and schema
in `plan.md`.

Resolve helper scripts relative to this `SKILL.md` (`scripts/...`). The
orchestrator snapshots the complete Skill directory into the target workflow;
do not require an `Agentic3D/` directory at the repository root.

Treat partitioning as a separate stage. Do not change layout parameters in the
same experiment.

## Task Agent protocol

Run this Skill inside one stateful Partition Task Agent so that analysis,
planning, coding, and validation remain in the same Codex thread. Before every
round of edits, persist the requested `plan.md` with one partition hypothesis,
affected components and files, build and evaluation steps, correctness gates,
and a rollback condition. Use prior cut and area evidence when planning the
next round. Claim stage completion only when the handoff success criteria and
every validation gate pass.

List every intended repository-relative edit in `files_to_modify`. In enforced
mode, this list must be inside the signed scope and must match the eventual
diff. Do not run Git lifecycle commands such as commit, reset, clean, checkout,
merge, or rebase; the framework owns candidate checkpoints, rollback, and
integration.

## Inputs and outputs

Read the adapter-selected benchmark netlist, instance areas, die capacities,
fixed seed, partition configuration, and baseline assignment artifact. Produce
a complete die assignment in the repository's public format, together with cut
metrics, area metrics, code/config diff, manifest, and evidence paths. Do not
assume the design contains macros or exactly two legacy macro lists.

Freeze an accepted result before invoking the Layout Agent. Hash the benchmark
inputs and record the seed so that later HPWL changes cannot be attributed to a
different partition.

## Compile and run

- Start with `stages.partition.compile_commands`; resolve design/config/thread placeholders from the handoff and repository docs.
- Unified `Place-3D` normally uses its public stage API and adapter-listed
  partition-then-place entry point. Legacy releases use the adapter-selected
  `Place-MoL` entry point.
- Rebuild only native backends actually changed by the plan.
- Use a copied JSON config and an isolated result directory; never overwrite the baseline.

## Allowed action space

Modify only adapter-listed partition source/config paths. Keep placement,
legalization, DREAMPlace density settings, benchmark, and seed fixed.

## Metrics

Compare cut weight and count, per-die assigned area and instance count,
normalized area imbalance, partition runtime, and downstream fixed-layout HPWL.
Do not accept a cut improvement that violates die capacity or area balance.

## Correctness gates

Require every expected assigned instance to appear exactly once, die sets to be
disjoint, no unknown IDs, every required die to be non-empty, configured die
capacities and balance tolerance to hold, deterministic replay for the same
seed, and successful downstream database loading.

Run:

```bash
python3 <skill-dir>/scripts/validate_partition.py \
  partition_result.json --output partition_validation.json
```

For unified `Place-3D` assignment CSVs, validate the public contract and area
evidence with:

```bash
python3 <skill-dir>/scripts/validate_partition.py \
  assignment.csv --metrics metrics.json \
  --expected-instances expected_instances.txt \
  --output partition_validation.json
```

Use the JSON validator invocation only for the legacy macro contract and the
CSV invocation for unified/LoL assignments. Reject the iteration if any
correctness gate fails. The agent may produce a partition candidate but must
not modify placement or routing code.
