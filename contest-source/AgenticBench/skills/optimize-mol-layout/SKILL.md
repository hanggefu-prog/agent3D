---
name: optimize-mol-layout
description: Run controlled Open3DBench layout experiments for unified Place-3D, legacy Place-MoL, or LoL regimes; optimize HPWL, density overflow, maximum density, overlap, and legality before OpenROAD-3D handoff.
---

# Optimize Open3DBench layout

## Version-adaptive repository contract

Read the `repository_context` referenced by the generated `handoff.md` before
planning. Its `stages.layout` entry is authoritative for editable algorithms
and configurations, build/evaluation command hints, log/results locations,
metrics, and correctness gates. Unified releases use `Place-3D/src/placement`,
`Place-3D/config/placement`, `Place-3D/thirdparty/DREAMPlace`, and
`Place-3D/results`; legacy releases use `Place-MoL`. Do not hard-code either
layout. Read applicable `AGENTS.md` files before edits.

Determine the active regime from the adapter name, selected source paths,
configuration schema, and accepted partition artifact. Record whether the run
is unified Place-3D, legacy MoL, or LoL. Never apply a MoL-only metric name,
benchmark reconstruction, or command to another regime.

Read `scope_mode` and `scope_contract` from the handoff. In `enforced` mode,
the signed contract is a hard limit: modify only its listed paths and obey its
file types, operation permissions, and line/file budgets. In `direct` mode,
the framework does not restrict repository paths or integrate a candidate
worktree; the stage boundaries in this Skill remain behavioral guidance. In
either mode, compile the affected runtime, launch declared local evaluations
with all `<placeholders>` resolved, and parse actual adapter-discovered
logs/results. Keep run directories isolated and never treat a synthetic
parser fixture as a measured layout result.

Resolve parsers, validators, tests, and Docker helpers relative to this
`SKILL.md` (`scripts/`, `tests/`, `docker/`). They are snapshotted into the
workflow, so the target repository does not need to vendor Agentic3D.

Keep the accepted partition, benchmark, seed, code revision, and evaluation
method fixed. Test one coherent layout hypothesis per iteration.

For unified LoL, distinguish a path name from a dependency. The maintained
source flow writes generated contest input under
`Place-3D/binaries/converted_input`, but that directory is produced by the
repository's `scripts/joint_placement/convert_input.sh`; it does not require
the optional `binaries.tar.gz`. Execute adapter-listed `preparation_commands`
before declaring inputs missing. The official contest evaluator may be listed
under `optional_tools`: its absence must be reported, but must not block
source-built DREAMPlace HPWL/density experiments when internal metrics,
overlap checks, and layout correctness gates are available. The released LoL
benchmark data itself is still required, but the unified algorithm accepts the
original Place-LoL benchmark tree. Reuse it at `Place-3D/benchmarks/lol` with
a read-only link/bind mount, or make one controlled copy to the experiment's
data disk; do not require a fresh archive download or extraction. When the
source resides on NAS, prefer the one-time copy so repeated placement rounds
read benchmarks and write all logs/results on the data disk.

## Task Agent protocol

Run this Skill inside one stateful Layout Task Agent; do not hand its analysis
to a separate coding agent. At the beginning of every round, inspect all prior
evidence and persist the requested `plan.md` before changing source or
configuration. State the hypothesis, affected components and files, expected
HPWL and density effects, compile and evaluation steps, correctness gates, and
rollback condition. Claim stage completion only when the handoff success
criteria and every validation gate pass.

List every intended repository-relative edit in `files_to_modify`. In enforced
mode, this list must be inside the signed scope and must match the eventual
diff. Do not run Git lifecycle commands such as commit, reset, clean, checkout,
merge, or rebase; the framework owns candidate checkpoints, rollback, and
integration.

## Compile

- Execute the applicable `stages.layout.compile_commands` from repository context.
- Unified releases compile the shared `Place-3D/thirdparty/DREAMPlace` runtime; legacy releases compile their adapter-listed DREAMPlace tree.
- For Python-only edits, run compileall over the adapter-listed placement source.
- Rebuild C++/CUDA operators only when native sources changed.

## Allowed action space

Modify adapter-listed placement algorithms/configs, including relevant macro
placement/refinement/legalization and DREAMPlace `target_density`,
`density_weight`, `gamma`, `stop_overflow`, and iteration controls. Do not
modify partition output, OpenROAD place opt, or routing in the same loop.

Prefer small experiments in this order:

1. Establish baseline with a fixed seed and isolated result directory.
2. Tune density target/weight while preserving HPWL.
3. Tune macro HPWL/regularity/overlap weights only after cell density is trustworthy.
4. Re-run the best candidate once for reproducibility.

Use the release's documented result-root override or an isolated copied run
directory. Record the absolute output path in evidence.

Create a single-purpose config without editing the baseline:

```bash
python3 <skill-dir>/scripts/make_place_config.py \
  <adapter-selected-base-config.json> <isolated-candidate.json> \
  --set macro_refine_params.regularity_weight=50.0 \
  --set dreamplace_overrides.density_weight=0.00012
```

Use `stages.layout.evaluation_commands` by default. The following helper is
available only when repository context selects the legacy Place-MoL backend and
the handoff supplies matching remote paths:

```bash
<skill-dir>/scripts/run_remote_place_mol.sh \
  --run-id bp-fe-baseline --design bp_fe \
  --remote <user@host> \
  --source-root /path/to/Open3DBench-V2.0/Place-MoL \
  --work-root /data/openagentic3dbench/place-mol \
  --config-path /path/to/Place-MoL/config/or_3D_bp_fe.json \
  --benchmark-root /path/to/benchmark_mol
```

For a legacy Place-MoL engineering run only, if the original benchmark package
is unavailable, reconstruct a placement-only benchmark from an OpenROAD-3D
evaluation DEF. Record the provenance; reconstructed data is not a
bit-for-bit published baseline and cannot prove a published PPA claim.

```bash
python3 <skill-dir>/scripts/prepare_benchmark_from_eval_pack.py \
  --eval-def OpenROAD-3D/flow/evaluation_pack/mol-analytical/bp_fe_top.def \
  --platform OpenROAD-3D/flow/platforms/nangate45 \
  --design bp_fe --output /path/to/reconstructed-benchmark
```

Use the unshrunk `nangate45` macro LEFs for the placement benchmark. The
`nangate45_3D` platform contains geometry transformed for the downstream 3D
flow and is not a valid source of physical macro dimensions for DREAMPlace.
When the evaluation DEF already encodes the die assignment in `_upper` and
`_bottom` master suffixes, freeze that assignment in the reconstruction
manifest and load it in every layout-loop run; do not repartition it.

## Measure

Use the converged placer metric, not the first nested metric sample. Read metric
names from the adapter and active regime. For legacy MoL, `Mem-on-logic HPWL
before legalization`, density overflow, and max density are the primary
objectives. For unified or LoL runs, use the corresponding public Place-3D
metrics. Keep post-legalization HPWL secondary until independently recomputed
from the emitted DEF.

```bash
python3 <skill-dir>/scripts/parse_place_log.py \
  run.log --require-density --output metrics.json

python3 <skill-dir>/scripts/compare_place_runs.py \
  baseline/metrics.json candidate-a/metrics.json candidate-b/metrics.json \
  --output comparison.json
```

## Correctness gates

Require successful completion, finite positive HPWL, recorded overflow and
maximum density, component and net counts identical to the frozen input, all
components inside DIEAREA, fixed macro coordinates preserved during cell
placement, no illegal macro overlap on any die, a parseable output DEF, and
deterministic replay within the declared tolerance.

Accept a candidate only if it Pareto-improves HPWL and density or stays within
the explicitly declared regression budget. Record the accepted DEF and metric
JSON as inputs to the OpenROAD Place Opt Agent.
