# Eight local tasks: paired matrix evolution and immediate TP-GNN GPU release

All eight local tasks completed two algorithm proposals on BP_FE and Ariane133,
with three paired seeds per case. The final contracts have 48 qualified reference
cells and 96 candidate cells: 89 succeeded and seven failed. Fourteen candidate
matrices are complete; two correctly receive no aggregate score. All complete
scores were independently recomputed, and all 16 archived programs match their
evaluated source hashes. No candidate replaces a maintained reference automatically.

The implementation is deployed on rafa1-gcr at frozen commit `e951f6df` and pushed
to `codex/agenticbench-integration-20261016`. It has not been merged into main.

## Results

Reference is 1 and higher is better. These are the existing local-task scores,
not percentage improvements in raw timing or wire length, and not the global
post-GRT Pareto objective.

| Task | Round 1 | Round 2 | Best choice including reference |
| --- | ---: | ---: | ---: |
| Placement initialization | 0.983818 | 1.010147 | 1.010147 |
| Placement preconditioner | Invalid: unsupported Torch API | 0.979103 | Reference |
| Current Nesterov controller | 1.011624 | 1.011223 | 1.011624 |
| HBT assignment | 0.999724 | 1.000522 | 1.000522 |
| TP-GNN post-placement timing | 1.018889 | Invalid: top-die VDD connectivity | 1.018889 |
| GRT iteration control | 1.002907 | 1.002907 | 1.002907 |
| GRT rip-up policy | 0.999684 | 0.999991 | Reference |
| GRT maze cost | 0.999615 | 1.001033 | 1.001033 |

Initialization improved after replacing movable-cell quantile redistribution
with seeded low-discrepancy filler initialization. Nesterov's first controller
combined smoothed shared steps with per-die density/gradient balancing; its
second, density-gated revision did not beat that parent. TP-GNN added annealed
weighted graph attraction and an area-centered embedding penalty. Its first
round improved all three Ariane133 seeds and had mixed BP_FE results. HBT's
second policy mildly weighted incremental reach toward the tier with more pins;
its gain is very small. The preconditioner's valid degree transformation
worsened the aggregate score.

The routing iteration controller shortened wire length in all six first-round
cells, trading some overflow for length. Its second program has a different
verified installed binary, but all six cells have identical recorded control
outputs and seven identical raw routing metrics. It adds no observed control
behavior or gain over its parent. The model targeted within-run stalled
regression using differences in final overflow relative to the reference;
these are different observations. Equal recorded controls do not establish
byte-identical unretained route databases or reveal unlogged context values.
See [the independent parent comparison](iteration-parent-comparison.json).

Rip-up round 2 limited extra detour recovery to low-pressure dies. On all three
BP_FE inputs it returned exactly to native decisions and score 1; the aggregate
still fell just below reference. Maze round 2 replaced congested-edge cost
compression with a small length bias and overflow/history surcharges. It improved
the average score but still regressed two of six cells. These are modest,
case-dependent findings, not evidence of out-of-sample generalization.

Candidate code, reference/parent diffs and each model's hypothesis are in
[candidates/](candidates/). Runtime behavior does not branch on benchmark names
or the selected seed IDs. Activity flags compare with native defaults, not
with the selected parent; source changes and activity alone are not improvement.

## Evaluation and budget

Partition and placement seeds are 1001, 1002 and 1003; I/O stays at 1 and routing
at -1. Every candidate uses its own task's six paired initial-program references.
Scores are averaged across seeds, then equally across cases. Deterministic HBT
and GRT policies see three frozen upstream physical states, not three identical
reruns. Single-case interfaces and local objectives remain unchanged.

The audit uses `gpt-5.6-sol` through server-side Codex in the local function-level
ask/tell interface. Codex returns a program and hypothesis; the harness verifies,
builds and evaluates it. Terminal tools are disabled in this small audit. This
is distinct from the global workspace-editing controller.

There were **17 actual model calls**: 16 retained proposals and one discarded
maze-cost second proposal that had received feedback from an invalid routing
setup. The corrected routing searches reuse their first programs unchanged,
then make one new model call each with valid feedback. There are also three
explicit native diagnostic controls: one split/unsplit comparison, one same-seed
PDN replay and one routing-hook preflight. Earlier failed qualifications and
stopped routing attempts remain separately recorded; the 48/96 counts above
refer to the final evaluation contracts, not all diagnostic work.

For the three routing tasks, BP_FE freezes a 50% routing-capacity reduction to
exercise the hooks; Ariane133 retains its already exercised full capacity.
Both use 16 CPU threads and the unchanged low native budget of five maximum
iterations and four congestion iterations. Every candidate uses the same
per-case configuration as its reference. These results must not be compared
with the full-capacity or global post-GRT reference.

## Failures found and corrected

- **Seed binding:** the initial TP-GNN qualification exposed outer seeds that
  did not override native method JSON. The fix reuses
  `grt_bench.flow.bind_method_seeds`, writes private per-cell configs and checks
  executed seeds. All six old cells were rejected before a model proposal.
- **Nesterov validation:** a missing import after the execution-phase refactor
  caused six qualification failures. It was fixed before this task's proposals.
- **Routing qualification:** full-capacity BP_FE skipped the policy loop on
  both dies. Process-success receipts were mistakenly accepted despite the
  existing hook gate returning zero. Matrix qualification and reference loading
  now reject such inputs before evolution. Old zero scores are setup diagnostics,
  not algorithm regressions. New references all pass the hook gate.
- **Failure feedback:** the first preconditioner used `torch.nan_to_num`, absent
  from the installed runtime. Its six failures originally had generic feedback.
  The harness now includes bounded native exceptions and their log paths.
  Boolean activity flags are retained; Boolean scores and non-finite values are
  still rejected. Actual failing logs were checked with server Python 3.8.10.
- **PDN evidence:** compact retention omitted PDN logs and could surface an
  unrelated upstream warning. PDN logs are now retained; feedback follows the
  failing wrapper's referenced native log and does not substitute an older
  warning when that log is missing. External symlinks are ignored.

TP-GNN round 2 failed on Ariane133/1003. One same-candidate/seed replay reproduced
`PSM-0069: Check connectivity failed on VDD` in the top-die PDN check. Partition,
cross-die nets and both dies' DEF/ODB/Verilog hashes matched the original failed
run. This is reproducible physical failure, not evidence of seed nondeterminism.
The original invalid matrix remains unchanged; the other five cells do not
supply a substitute aggregate. See [the replay evidence](tp-pdn-diagnostic.json).

Audit-driver corrections also handled an omitted optional `pins` mapping,
two pre-launch disk rejections, and task-ID versus directory-name lookup after
routing qualification. Qualified evidence was retained; the last lookup repair
restarted only the empty search, without new qualification or model calls.

## GPU lifecycle, storage and validation

TP-GNN runs partition and placement in a GPU container. After confirming its
termination, the host seals the continuation/checkpoint and reduces the same
live lease to CPU-only. A separate container resumes at PDN, then runs HBT and
post-placement STA without a GPU or repeated partition/placement. Both phases
share one evaluation time budget. Unknown termination retains the allocation.

All 18 TP-GNN reference/candidate cells released their GPU before CPU continuation.
The CPU phases totaled **49.26 GPU-slot minutes avoided**, with a median 74.48
seconds per cell. This is occupancy saved, not measured end-to-end speedup.
The unsplit BP_FE/1001 control exactly matches WNS, TNS, timing-weighted cut and
cut count, with score 1.0.

The shared queue reached **six GPU slots, three per device**, and eight live
evaluations. Builds, HBT and routing reserve no GPU. GRT builds once per matrix;
all six cells share its verified installed image. Final checks found no owned
running containers, queued work or unreleased reservations. About **76.7 GiB**
was free. Obsolete failed/stopped runs were compacted, and identical frozen
source/upstream files were deduplicated without changing their content.
Candidates retain code, diffs, scalar receipts and bounded logs; references
retain their physical evidence. Cleanup and resource records are in the evidence.

The maintained integration gate passed **1,267 tests**, with **8 optional skips**
and **180 passing subtests**; 82 focused matrix/phase/transport tests passed.
Skips cover external EvoStage, optional wheel packaging and six unavailable local
Torch tests. The full gate ran with loopback access for its two socket tests.
See [the test log](maintained-tests.log), [summary](summary.json), and
[full evidence](evidence.json). Run `python3 verify_evidence.py` for an offline
check of source hashes, aggregation, completion counts and resource evidence.
The evidence includes obsolete attempts under explicit suffixes; only its eight
unsuffixed tasks are final-contract results.

## Locations and source identities

Audit root (the deployment parent is supplied by the operator):

```text
${AGENTIC_AUDIT_PARENT}/local-matrix-audit-20261018
```

| Task group | Directory below the audit root | Frozen source |
| --- | --- | --- |
| Initialization, preconditioner, HBT | `tasks/<task>/` | `4b1f586b` |
| TP-GNN | `tp-seed-fix/tasks/tp_gnn_post_placement_timing/` | `e08588c8` |
| Current Nesterov | `nesterov-validation-fix/tasks/placement_parameters_reference_20260923/` | `8e01e930` |
| Three GRT tasks | `routing-active-hooks/tasks/<task>/` | `e951f6df` |

Within each task, `references/qualification/qualified_matrix.json` lists the six
qualified native references; `search/evaluations/*/matrix/` holds candidate cell
receipts and retained logs. Source changes across these versions concern phase
execution, seed binding, validation, qualification and diagnostics. Native
algorithm revisions and scoring formulas are unchanged. Routing configuration
changes are disclosed above. Every comparison uses its own frozen contract;
older receipts are never relabeled as qualification of a newer source.

The final clean runtime source is
`${AGENTIC_AUDIT_PARENT}/source-local-phases-e951f6df`.
[audit_runner.py](audit_runner.py) and the routing audit/resume scripts preserve
the server recipes. The exported routing wrappers take `AGENTIC_AUDIT_PARENT`
and `CODEX_BIN` from the environment and locate helper scripts beside themselves;
only path binding differs from the recorded executions. They require host-owned
inputs/runtime settings and a clean initialized source checkout; use fresh
destinations. The resume recipe additionally requires its recorded qualified
references and prior proposals.

Full native receipts and the original report remain on the server. Repository
JSON exports replace the host storage prefix with `${DATA_ROOT}`; this is a
reading-view alias, not a new native receipt. Candidate programs and diffs are
unchanged. [collect_evidence.py](collect_evidence.py) recreates the original
reading view without changing native receipts.
