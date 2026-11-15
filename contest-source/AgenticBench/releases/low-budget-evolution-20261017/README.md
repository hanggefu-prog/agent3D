# Low-budget algorithm evolution audit, 2026-10-17

Status: physical and model tests completed; final resource/storage accounting is
recorded in `budget-and-resources.json`. This is a bounded research audit, not
a claim that every generated candidate improves or that retired tasks are qualified.

This audit exercises actual model-generated changes on the integration branch,
not identity candidates. It leaves the original branches, main, frozen physical
references and formal campaign untouched. The model is `gpt-5.6-sol` through the
server-installed Codex CLI. Native work runs on `rafa1-gcr` under:

`/data-ssd/gcr/agenticbench-integration-20261016/evolution-audit-20261017/`

## Budget and interpretation

- Eight local tasks: two proposals each, one fixed case/seed per task. Partition,
  placement and HBT use BP_FE; GRT uses congested Ariane133, where all three hooks
  are exercised. The local ask/tell interface supplies the complete editable
  program and measured feedback; these model calls have no terminal tools.
- One additional initialization proposal checks the corrected active-path prompt.
  It is a separately recorded diagnostic, not an unreported retry.
- Maze diagnostics add no model calls: each original installed image was replayed
  once; after the context repair, a fresh reference and the same two programs are
  rebuilt, measured and each repeated once. Earlier results are not relabeled.
- Three engineering tasks: one implementation each, at most 1,200 seconds of
  development, then one final evaluation per each of four cases: BP_FE, BP,
  Ariane133 and SweRV. Native reference qualification has a separate ledger.
- Current post-GRT global task: two rounds, each with the unchanged five-case,
  three-paired-seed matrix. Engineering and global agents have the controlled
  workspace tools supplied by their maintained interfaces.

Local task scores are maximized and must be compared with a normalized reference
of **1.0**. Fresh-reference bootstrap scores in the routing and dated Nesterov
adapters are not candidate-relative scores. A successful evaluation is not
necessarily an improvement. The local search API reports the best submitted
candidate; this audit also compares it with the unchanged reference before
calling anything the best solution.

These small local tests check algorithm execution, feedback use and acceptance.
They do not establish generalization across cases/seeds or post-GRT improvement
for an algorithm evaluated at an earlier endpoint.

## Local algorithm results

Local scores below are normalized to the unchanged reference at 1.0. The maze
row uses fresh measurements after the context repair, replaying the two existing
programs rather than claiming a new feedback-driven search after the repair.

| Local task | First proposal | Second proposal | Interpretation |
| --- | ---: | ---: | --- |
| TP-GNN post-placement timing | 1.015354 | 1.033748 | Both improve the fixed-input timing endpoint |
| Placement initialization | 1.000000 | 1.000000 | Inactive branch; separate corrected-prompt check: 1.001587 |
| Placement preconditioner | 1.000000 | 0.975168 | Inactive callback first; active second change regresses |
| Placement dynamic parameters | 0.997684 | 0.978365 | Both regress; retain reference |
| HBT assignment | 0.999197 | 0.999384 | Second recovers some loss, neither beats reference |
| GRT rip-up policy | 1.000258 | 1.001240 | More selective detour rerouting improves the fixed input |
| GRT iteration controller | 1.000671 | 1.000726 | Narrower window; second restores unchanged bottom policy |
| GRT maze cost, repaired context | 1.000490 | 1.000485 | Both repeat exactly; second no longer improves on first |

### Active control paths matter

The first two initialization candidates changed the native random-center branch,
but the maintained concurrent driver sets `random_center_init_flag=0`. Both
candidates were legal and returned exactly the reference metrics. The first
preconditioner candidate similarly depended on `set_overflow`, which is not called
by the maintained two-region gamma-update path. Legal source changes therefore
did not imply an active algorithm change.

The task prompts now explicitly describe these paths. They permit transformation
of the supplied movable starting coordinates without changing the protected flag,
and direct preconditioner changes toward constructor/`__call__` inputs. No physical
algorithm, source boundary, evaluator or reference was weakened for this repair.

The one-proposal initialization recheck used area-weighted coordinate ranks within
the assigned regions. Its HPWL changed from 4,773,003 to 4,764,943, worst overflow
from 0.03081 to 0.03079, and score to 1.0015873248. This verifies that the editable
initializer is effective when the model has accurate execution context.

### Feedback is used, but plausible ideas can regress

The TP-GNN candidates changed timing-weighted graph cohesion and then focused
its gradient on high-weight nets. Their scores were 1.0153538930 and 1.0337484586.
WNS improved from -71.2011874 ns to -69.4246245 ns, and TNS from -106,604.0234 ns
to -101,893.2191 ns. The second candidate's timing-weighted cut proxy increased
slightly relative to the first while downstream timing improved; the proxy alone
would have misranked these candidates.

The second preconditioner proposal executed a different pin/density curvature
combination, but scored 0.9751680672. The two dynamic-step proposals scored
0.9976842081 and 0.9783654409. Both HBT ordering proposals were legal and slightly
worse than the reference. These are measured negative results, not harness
failures or reasons to replace the unchanged reference.

Rip-up round two reduced wirelength from 6,970,147 to 6,965,786 micrometers and
overflow from 37 to 30. Its code lowered the detour threshold while retaining
native rip-ups and clock, NDR, HBT, criticality and congestion guards. The
iteration controller's second round restored bottom wirelength to its reference
3,313,127 micrometers while preserving top's first-round 28 overflow and
3,658,123 micrometer wirelength exactly. This agrees with its localized change.

Two GRT second-round proposals switch their changes off on the bottom die after
observing that only the top die benefited. This is within the supplied interface,
but its rationale is tied to one input. It should be distinguished from a policy
that reacts to measured live congestion; an improvement here would not establish
that die-specific gating generalizes.

### Undefined maze input invalidates the original apparent gain

The maze policy's `wirelength` feature read `FrNet::getNetLength()`. The native
`FrNet::reset` does not initialize its backing member. Its assignment in
`updateSlacks` belongs to the resistance-aware layer-assignment path, after the
2-D maze phase. Thus a candidate using this feature could consume undefined
data even though identity-policy references were repeatable.

This was exposed by a causal inconsistency: adding a bottom-only condition also
changed the top result. Each original installed image was replayed on the same
authenticated checkpoint. Round one's top overflow changed from 30 to 32 and
round two's from 25 to 31; bottom measurements were unchanged. These pre-repair
maze scores are invalid evidence of a stable algorithm gain, not a successful
demonstration of seed variation.

The adapter now snapshots the sum of initialized Steiner-edge lengths once per
maze iteration. Its public scalar normalization is unchanged. A compiled probe
tests sparse net IDs and known tree lengths without permitting the unavailable
FrNet accessor. The repaired adapter has its own fresh reference, source/runtime
receipts and measurements; existing model programs are reused without spending
extra search calls. The new reference equals the old reference exactly. Both
candidate repeats match all seven checked aggregate/per-die metric values exactly.
Top overflow is 29 and top wirelength 3,659,243 micrometers for both candidates.
Round two restores the bottom policy and adds 42 micrometers to total wirelength,
so its original apparent advantage disappears. See `maze-fixed-summary.json` and
`maze-original-repeat.json` for the before/after evidence.

## Integration repairs

- `48bbc9e2`: the thin Python 3.8 host launcher can render placement feedback
  without `str.removeprefix`. Original failures occurred before model invocation;
  replacement placement runs have separate output directories and fresh references.
- `f52a8546`: unified engineering checkouts can mount the protected framework
  over their existing AgenticBench directory only when the subject root is
  read-only. Writable subject overlays and symlinks remain rejected.
- `d6c9f2d4`: initialization and preconditioner prompts describe the active paths.
- `827237cf`: maze wirelength is derived from initialized Steiner edges; its
  focused regression set passed 57 tests and 20 subtests.

The maintained regression gate at `51ef2781` passed **1,159 tests, with 10 skipped and
180 subtests passed**, using the local socket access required by its service tests.
The historical full-suite limitations documented in the integration report remain;
this audit does not claim that retired global tasks were restored.

The post-GRT evaluator identity is still
`322c79592b539e312d5386d062fbf727a83a39a99c0da43ab52314c7923a0645`.
The first three engineering model runs use evaluator identity
`9ef059787e5c36c6538529aaee24f363b0988750d461563ff1b12c027b1709ae`.
Control/runtime fixes change the broader engineering identity, so the CTS
corrective run obtains fresh qualifications instead of relabeling old receipts.

Completed local routing outputs were deduplicated against their byte-identical
immutable upstream input files. All paths and contents remain, including the
diagnostic records; 1,780 file replacements released **47.225 GiB**. The receipt
is `prefix-dedup-1792244254.json` under the audit root. No unrelated experiment
or unique physical result was removed.

## Evidence locations

Local candidates, parent diffs, effective prompts, model events and native result
directories are under `locals/`, `locals-fixed/` and `initializer-prompt-recheck/`.
Engineering qualification/development/finals are under `engineering-v3/`.
The isolated global campaign, exact prompts, budget state, candidate library and
Pareto archive are under `global-v4/`; the blocked `global-v3/` calls remain
separately preserved.

Both global matrices and all three engineering tasks have final measurements.
The model and matrix budget records preserve failed infrastructure attempts
separately from successful candidate evaluations.


## Engineering algorithm checks

Clock compensation selects one or two nonoverlapping branch actions using library
min/max delay estimates, clock-domain membership and predicted timing guards.
BP_FE skew improves from 0.279506 to 0.268249 ns (4.0275%); BP and Ariane133
correctly retain no-op plans and unchanged skew. SweRV fails the independent
new-hold-endpoint guard: despite improved aggregate setup and hold WNS/TNS,
26 endpoints newly violate hold. A predicted safe plan is not a native STA proof.
This candidate is not successful across all four cases.

Multi-HBT constructs physically materialized, isolated extra branches chosen by
geometry and an RC/library proxy. BP_FE is legal but mean target-net delay
regresses 0.5723%; Ariane133 improves 0.0847% and SweRV improves 2.9068%.
BP introduces 16 new hold-violating endpoints and worsens hold TNS from
-405.45147 to -406.06168 ns, so the verifier rejects it. No all-case improvement
is computed with that failure missing.

The multi-HBT model initially left a bytecode cache outside its editable scope.
The verifier rejected this correctly. An independent copy removes only that
cache, keeping the exact model-generated algorithm bytes; the subsequent four
finals are explicitly host-assisted diagnostics, not an autonomous submission
success and not additional model searches.

The first CTS invocation made no changes: its patch tool attempted atomic file
replacement in a read-only parent directory, although the exact file mount was
writable. The old development wrapper confused successful CLI exit with a ready
implementation and launched unchanged-source controls. Those controls do not
count as algorithm improvements. Commit `7144dfe2` rejects empty implementations,
explains in-place writes in both engineering and global prompts, and sends
explicit Python compilation caches to scratch. A native pinned-image probe
confirms allowed-file writes, scratch-only compilation, and protected-write denial.

Two initial global model calls hit the same patching limitation and spent zero
matrix evaluations. One corrective CTS model call and two corrective global
model calls have separate recorded budgets. The prior failures are retained.
Commit `51ef2781` also passes Python `-B` to host MCP tool bridges: a filtered
child environment had otherwise created 53 cache files in frozen controller
source. These cache files were moved to a diagnostic quarantine and the original
source inventory was recovered exactly. Its focused regressions pass 47 tests, followed by the full maintained gate above.

A second content-identical upstream deduplication releases another 28.335 GiB
from the completed repaired-maze runs, for **75.560 GiB** across both operations.
Unique physical results and all artifact paths/hashes are retained.


### CTS smoke-test representativeness

The corrected CTS invocation makes real scoped edits, passes 60 targeted Python
tests (one optional sealed-input test skipped), and exercises an existing native
insertion-delay smoke test. All four final cases are legal and exactly equal to their respective references:
BP_FE 0.279506 ns, BP 0.454953 ns, Ariane133 0.378453 ns and SweRV
0.372858 ns. All four internal selectors choose the baseline because measured
coordinated skew is identical. The candidate is an algorithmic no-gain result. The candidate's internal coordinated and baseline constructions
both report 0.279519 ns before independent normalization, with 118 bottom clock
buffers, so its bounded selector falls back.

A read-only diagnostic of the actual generated HBT Liberty reports
`is_clock=1` and `is_register_clock=0`; its LEF class is COVER. Both
`TritonCTS::isSink` and `LatencyBalancer::isSink` recognize block instances or
register-clock ports. The native smoke fixture instead uses a BLOCK macro.
The delay balancer therefore does not recognize the actual HBT proxy as the
endpoint used by its smoke test; merely setting its clock-tree insertion-delay
arcs does not establish useful receiving-tree balancing. This is an algorithm
integration limitation, not a reason to change the frozen evaluator or mark a
no-op as an improvement. The model's candidate remains unchanged for its finals.
See `cts-endpoint-probe.json` and the model's recorded native-test command.

The obsolete unchanged-source CTS control batch was stopped after confirming
its owned native container stopped. BP_FE and BP had completed with exact reference
metrics; the partial Ariane133 output remains diagnostic and SweRV was not run.
The outer resource lease was returned normally. The corrected implementation
uses its own fresh qualification and four-case final ledger.

The corrected CTS development and four-case evaluation took 2,803.67 seconds
(46.73 minutes), including 497.77 seconds for the model turn. Its final resource
lease returned with `success=true` and `uncertain=false`.

## Global round one: proxy gain does not establish final PPA gain

The model adds two bounded passes of deterministic pair swaps after track-aware
HBT greedy assignment. It searches 24 nearby selected sites per net, accepts only
strict combined HPWL reductions with nonincreasing pin-weighted cost, and retains
the selected site set and escape-track legality. Existing unit tests, a repeated
120-net integration check and a 2,000-net synthetic test pass. All fifteen native
cells are independently verified; all 75 upstream stage-manifest checks match.
The affected suffix starts at HBT and needs no GPU. The worker pool peaks at ten
live CPU evaluations. This is not evidence of six simultaneous GPU executions.

Compared with the original reference vector (1, 1, 1), the candidate scores:

| Candidate | Timing | Overflow | Peak-temperature ratio | Power feasible |
| --- | ---: | ---: | ---: | --- |
| `68a75a22` | 1.007240944 | 1.087802840 | 0.999987783 | Yes, all five cases |

All objectives minimize. Timing regresses 0.7241% and overflow regresses 8.7803%.
Mean peak temperature across case/seed measurements falls only 0.004 K. Ariane133
mean overflow increases from 19 to 28.6667 despite shorter HBT-related HPWL. The
largest per-case mean power ratio is 1.000106, well within the frozen 1.10 limit.
The source is correctly retained as non-dominated because of its tiny temperature
gain, alongside the reference. That is exact-contract compliance, not persuasive
practical improvement. The audit does not introduce an epsilon or change scoring.
An independent standard-library recomputation matches archive vectors, eligibility
and the frontier; `global/analyze_global.py` reproduces this from saved raw cells.

The matrix worker interval is 2,447.24 seconds (40.79 minutes), excluding earlier
parent restoration/build/proof preparation. Post-evaluation storage takes 316.34
seconds and releases 22,787,973,120 bytes (21.22 GiB). This overhead is material:
strict source/checkpoint validation and repeated DEF restoration remain performance
opportunities even when the physical prefix is reused. No validation was weakened.

A separate lossless deduplication of completed workspace source copies releases
30.482 GiB across 22,944 files. It preserves file modes and links only identical
completed copies; frozen read-only snapshots are unchanged. The receipt is
`workspace-dedup-1792250126.json`. Together with the earlier routing-prefix
operations, these manual lossless actions release about 106.04 GiB; the storage
policy's own accounting is reported separately to avoid conflating operations.

### Local HBT context limits

The maintained local HBT v1 task exposes geometric context and explicitly does
not expose per-network timing context. Its endpoint uses placement-estimated RC
and ideal pre-CTS clocks, with no OPT/CTS/GRT. Reordering equal-fanout nets by size
is therefore an indirect timing hypothesis. Neither proposal improves its native
timing score; those endpoint measurements must not be compared numerically with
the global post-GRT endpoint. The global HBT source experiment likewise demonstrates
that shorter HBT-related HPWL is an insufficient acceptance criterion.

## Global round two: feedback-driven stage change and native build

The model resumes the same explicit Codex session, sees both frontier candidates
and their raw case/seed results, and selects the reference rather than the HBT
candidate. Its rationale cites the first round's TNS and Ariane overflow regressions.
This is an observed feedback-driven change of stage, not an inferred intention.
The only source edit is `FastRoute.cpp`; no case names or expected metric values
appear in its activation logic.

The candidate adds at most four 2-D maze iterations when residual overflow is
between 1 and 500, checkpoints the entry and each lower-overflow solution, restores
the best stored route, and refreshes congestion before layer assignment. It also
adds one 3-D medium-edge cleanup pass for small residual congestion. The model
acknowledges that this 3-D pass has no layer-aware rollback. Its focused C++ object
compile passes. Local full CMake configuration is unavailable in the scoped agent
container; the host's separate verified build succeeds in **210.15 seconds**
(3.50 minutes), installs the candidate binary in a new pinned image, and checks
its binary hash and base-image layers. Shared compiler-cache counters are saved
as namespace statistics, not claimed as a per-build hit rate.

Candidate `419705da` uses image
`sha256:59b5cc67c7eeaba428733b7e97de923e3a0c18e4097cc1e109b56a2d768d2a31`.
All 15 independent verifications pass. Every cell starts at GRT and reuses the
eight verified stages through CTS with zero native GPU allocation. Together,
both rounds pass **195** upstream stage-manifest reuse checks. The two complete
matrices use the same task, original normalization reference and power limits.

| Candidate | Timing | Overflow | Peak-temperature ratio | Frontier result |
| --- | ---: | ---: | ---: | --- |
| Original reference | 1.000000000 | 1.000000000 | 1.000000000 | Dominated by round two |
| HBT pair swaps, round one | 1.007240944 | 1.087802840 | 0.999987783 | Retained |
| GRT residual polish, round two | 0.999064712 | 0.871595656 | 1.000000000 | Retained; dominates reference |

Round two improves normalized timing by 0.09353% and the overflow objective by
12.84043%, with unchanged peak temperatures and all five power constraints met.
The largest per-case mean power ratio is exactly 1.0. Physical overflow is zero
in all 15 cells; the normalized overflow objective is not zero because shifted
ratios keep already-zero reference cells neutral. The original reference stays
in history and remains the scoring/power reference after leaving the frontier.

| Ariane seed | Overflow, reference to candidate | GRT wirelength, um | GRT wall time, seconds | Logged extra iterations |
| --- | --- | --- | --- | --- |
| 1001 | 37 to 0 | 6,970,147 to 6,977,098 | 83.01 to 115.16 | 4 to 8 |
| 1002 | 20 to 0 | 7,490,688 to 7,509,963 | 142.54 to 140.29 | 4 to 7 |
| 1003 | 0 to 0 | 6,172,349 to 6,172,349 | 76.90 to 78.76 | 3 to 3 |

The other 13 case/seed cells match all five original raw metrics exactly. The
new code is active where expected: native logs explicitly record residual
polishing at overflow 37 and 7, respectively. Increased routing effort and
slightly greater wirelength are real tradeoffs. Wall times come from different
concurrent runs, so the second seed's shorter time is not proof of speedup.
There is no 2-D-only/3-D-only or equal-iteration control within this small budget;
the audit does not assign the full benefit to either sub-change in isolation or
claim better routing at equal computational effort.

The second matrix worker interval is **1,091.32 seconds (18.19 minutes)**. It starts
at GRT, whereas round one starts at HBT; their matrix times are not an algorithm
speed comparison. Source/provenance preparation precedes native process starts
by roughly nine minutes in round two, a material remaining host overhead.

## Budget accounting and conclusions

The initial search plan was 16 local calls, three engineering calls and two
global calls. Explicit corrective diagnostics add one initializer call, one CTS
call and two replacement global calls: **25 model calls in total**, comprising
17 local, four engineering and four global. The two blocked global calls spend
**zero** physical matrices; the corrected campaign spends exactly two matrices,
30 candidate cells, plus its unchanged imported 15-cell reference. Maze repeat
and repair diagnostics consume native work but no additional model calls. No
unreported search rounds are used to turn negative algorithm results into wins.

The three substantive engineering candidates have 12 final case results: ten
legal and two correctly rejected for new hold violations. Four of the legal
results belong to the CTS candidate and are exact baseline fallbacks. The
original zero-change CTS invocation's two completed controls, one partial
cancelled control and unrun final case are not counted as algorithm results.
The multi-HBT finals use an explicitly disclosed host-assisted clean copy of
the exact model source after cache contamination; they are not represented as
an autonomous submission success.

The maintained regression gate passes 1,159 tests and 180 subtests with ten
optional-dependency/integration skips. The skips concern local Torch/OpenEvolve/
EvoStage availability; native TP-GNN and placement evolution were independently
executed in the server image. Historical global-task/full-suite failures remain
outside the requested compatibility scope and are not hidden by this result.

The round loop behaves as intended: it exposes source changes and measurements,
resumes one explicit session, selects a verified frontier parent, verifies actual
edits, evaluates a complete paired matrix, and admits/removes frontier members
using independently checked measurements. The observed algorithm-quality issues
are more specific than simple pass/fail: inactive editable branches, undefined
native context, unrepresentative smoke fixtures, missing timing context for a
geometric policy, proxy/final-objective disagreement, and gains bought with more
search effort. The first two and the editing/cache/zero-change control failures
were repaired and retested; unsuccessful research algorithms remain negative
results rather than being silently replaced.

Next research checks should use active-path tests on real task context, pair
fast surrogates with focused STA/native checks, and compare the GRT candidate
against equal-iteration controls before claiming an efficiency improvement.
Exact Pareto retention is sensitive to the first candidate's 0.004 K mean peak
temperature change; introducing a meaningful-improvement threshold would be a
separate, reviewed scoring-contract change and is not done here.

The audit wrapper reserves a conservative six-slot outer GPU pool during global
phases. The actual HBT/GRT suffix jobs correctly request zero GPUs. No competing
GPU campaign is active, but this wrapper does not establish fine-grained release
between concurrent campaigns or six-way native GPU utilization. Its coarse
reservation and repeated source/checkpoint preparation are recorded limitations,
not evidence that the production GPU phase-release mechanism was exercised here.

Evidence can be inspected without rerunning the physical tools:

- [Local programs, hypotheses and feedback](local-evidence.json), with source in
  `local-programs/`.
- [Engineering results and source edits](engineering/engineering-evidence.json).
- [CTS internal selection decisions](engineering/cts-selection-evidence.json)
  and [actual HBT endpoint probe](cts-endpoint-probe.json).
- [Global raw matrices and control records](global/global-evidence.json),
  [independent aggregation](global/global-analysis.json), candidate diffs,
  [native build receipt](global/native-build.json), and
  [GRT activation/runtime diagnostics](global/routing-diagnostics.json).
- [Regression gate](unit-tests.json) and
  [model budgets, resource cleanup and storage receipts](budget-and-resources.json).

Run `python3 global/analyze_global.py` in this directory to independently reproduce
the three objective vectors, power feasibility and final non-dominated set.

## Final cleanup

Both rounds finish with status `verified`; the corrected campaign records exactly
two used evaluations. The original blocked global campaign records zero. All
audit outer resource jobs are terminal, no audit containers remain, and unrelated
pre-existing containers are untouched. Available SSD space at completion is
**113.79 GiB**.

Manual, content-and-mode-preserving prefix/source deduplication releases
**138.34 GiB** in total. The two enabled ODB-first storage operations
release a separate **47.91 GiB**; second-round storage takes
323.95 seconds. Receipts and their hashes are listed in `budget-and-resources.json`.
No unique physical checkpoint is removed by the manual actions. Reproducible DEF
removal follows the existing hash-bound ODB-first policy.

The integration fixes and this report are committed only to the draft integration
branch. Model-generated research candidates remain archived experiment artifacts;
they are not silently installed as the maintained reference algorithms.
