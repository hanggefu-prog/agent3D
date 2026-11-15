---
name: open3dbench-candidate-evaluation
description: Verify and evaluate one already proposed local-task candidate with the common OpenAgentic3DBench harness and an external Open3DBench-V2.0 checkout. Use for dry-run contract checks, isolated stage evaluation, metric extraction, or failure classification. Do not ask an LLM for a candidate, manage a population, or run evaluator-only STA promotion profiles.
---

# Open3DBench candidate evaluation

Evaluate exactly one candidate against one task contract and one immutable
benchmark input. Candidate generation and the outer ask/tell policy are outside
this Skill.

## Procedure

1. Inspect the optimization task:

   ```bash
   openagentic3dbench-task inspect <task-id>
   ```

   Confirm the task ID, case/profile compatibility, allowed source and config
   scopes, execution boundary, metric objective, and correctness gates.
   For an evaluator-owned inner proxy, inspect it with
   `openagentic3dbench-profile inspect <profile>` instead; a profile accepts an
   already proposed candidate and never becomes an ask/tell search problem.
2. Validate the frozen input with `openagentic3dbench-task validate-input`.
   Confirm that any reference run uses the same case, input identity, PDK,
   seed policy, evaluator contract, and metric units.
3. Run a dry evaluation first when a new candidate format, task adapter, or
   external evolve bridge is being tested:

   ```bash
   openagentic3dbench-task evaluate <task-id> \
     --candidate <candidate> --open3dbench-root <source-root> \
     --benchmark-input <bundle> --output-dir <dry-run-dir> --dry-run
   ```

   Replace `openagentic3dbench-task` with `openagentic3dbench-profile` when
   invoking `tp_gnn_partition_proxy` or `placement_proxy`.

4. For the physical evaluation, remove `--dry-run`, use a new isolated output
   directory, and add the case-matched `--reference-run` when normalized
   fitness requires it. Use `scripts/benchmark_docker.sh` when the release
   image owns the runtime.
5. Preserve `evaluation_result.json`, the resolved plan/configuration, verifier
   report, stage logs, source diff/provenance, and artifact paths. Compact or
   discard workspaces only after the machine-readable result is safely written.

## Gates

Reject scope violations, malformed candidate programs, dirty-source leakage,
missing frozen artifacts, failed stage manifests, non-finite metrics, and
reference mismatches. Never reinterpret a missing metric as zero or score a
candidate that bypassed the task's correctness gates.

Return the output path, task and input IDs, status, fitness and raw metrics,
failure class, runtime, provenance, and artifacts needed by `tell`. Use
`open3dbench-sta-evaluation` instead when a selected candidate is being
promoted through the fixed two-pass STA profiles.
