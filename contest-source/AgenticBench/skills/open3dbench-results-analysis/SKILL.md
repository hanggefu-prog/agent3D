---
name: open3dbench-results-analysis
description: Read and compare OpenAgentic3DBench evolution histories and Open3DBench-V2.0 evaluation artifacts, classify failures, compute reference-relative improvements, and produce convergence data or tables. Use after evaluations exist. This Skill is read-only and must not rerun stages, change candidates, or relabel failed evaluations as successful.
---

# Open3DBench results analysis

Analyze recorded evidence without changing experiment state.

## Sources of truth

Prefer, in order, `evaluation_result.json`, the task or evaluation profile,
`benchmark_input.json`, `history.jsonl` or `history.json`, `summary.json`, and
the referenced Open3DBench stage manifests and metrics. Logs are supporting
evidence; do not replace a machine-readable metric with a similarly named log
number unless provenance and units are explicit.

## Procedure

1. Group records by design, integration, PDK, frozen-input identity, task,
   evaluator version, seed policy, and reference. Refuse percentage comparisons
   across mismatched groups.
2. Count proposed, verified, physically attempted, successful, failed, and
   accepted candidates separately. `successful evaluation` means all required
   execution and correctness gates passed; it does not mean the candidate
   improved the objective.
3. Recompute improvement from raw values using the metric direction declared
   by the task. For lower-is-better metric `m`, report
   `(reference - candidate) / reference`. Preserve sufficient precision during
   calculation and round only the displayed result.
4. For placement histories, generate normalized convergence artifacts with:

   ```bash
   openagentic3dbench-convergence <run-dir> --output-dir <report-dir>
   ```

   The output includes JSON, CSV, and an SVG curve for evaluated and incumbent
   metrics. For other tasks, emit an equivalent tabular series from the
   recorded history without inventing missing points.
5. Report the reference and best candidate separately, the evaluation count,
   raw HPWL/overflow/timing/routing metrics with units, fitness definition,
   improvement, runtime, and failure distribution. Link every conclusion to
   its result or manifest path.

## Integrity rules

Never mix DREAMPlace native weighted HPWL with a later OpenROAD HPWL under one
column, and never mix density overflow with maximum density. Do not select the
last row as “best” unless its score actually wins. Keep failed legalization,
PDN, HBT, routing, or STA outcomes visible and preserve missing values as
missing rather than zero.
