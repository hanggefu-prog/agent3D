# BP_FE global workspace endpoints

Both Codex and Agentic3D use the same trusted evaluator and sealed reference.
Select `--reference-baseline reference_20261004 --rounds 10` with an explicit
objective. Each search permits exactly ten candidate finals; reference
qualification, tooling preflight and rejected infrastructure admission are
reported separately. A rejected submitted candidate still consumes its slot.

| Objective | Evaluated flow | Score |
| --- | --- | --- |
| `timing` | Partition, placement, PDN, HBT, exactly one OPT, pre-CTS STA | Reciprocal of the maintained 0.6 WNS / 0.4 TNS normalized deficit cost |
| `thermal` | Same frontend and one OPT, then CTS, vectorless cell power, HotSpot | Mean-temperature ratio minus 0.1 times post-CTS timing-cost excess above 5% degradation |

The thermal task is `lol_partition_placement_cts_thermal_20261008_10`.
Neither endpoint runs detailed routing. Thermal also omits global routing.
The thermal result is not routed power or signoff: power uses placement-estimated
continuous-passive-HBT parasitics, idealized vectorless input activity 0.1, and
the existing provisional shared thermal stack. HotSpot uses 10 by 10 tiles per
die and 318.15 K ambient. The unweighted mean covers all 200 equal-area tiles.

The protected `workspace_cts_thermal` adapter replays the exact CTS STA body,
redirects only reports, and appends power queries. It verifies that timing does
not change, DEF normalization loses no components, all instance power maps to
tiles, power is conserved, and all temperature tiles are finite and complete.
The independent consumer rechecks the checkpoint and file hashes and recomputes
power totals and temperatures. Missing metrics fail qualification; they are not
zero. The old die-local thermal endpoint is not silently relabeled or reused.

New thermal references require a separately published authority containing the
new power/thermal evidence from an immutable qualified CTS checkpoint. The
original reference flow and manifests remain untouched. Candidate and reference
must use the same evaluator digest, power model, stack and one-OPT contract.
The reference API remains ADOPT-only for searches, with no FRESH fallback.

On hosts storing the verified classic serialization of the canonical OCI image,
pass the immutable `--release-manifest` and its `--release-manifest-sha256`.
The launcher checks the installed image and rootfs layers against the existing
equivalence proof, qualifies model tooling on the actual image, and records both
image identities. Every container dispatch rechecks the release binding. An
absent proof or changed image fails closed; no image tag is used as a fallback.

Implementation and offline test success do not establish native qualification.
Run receipts must distinguish native qualification, generated proposals,
candidate finals, accepted evaluations, failures, provider turns and measured
stage wall times. Provider/API latency remains unknown unless measured directly.

## Thermal timing penalty

The current thermal objective maximizes `combined_score`. It uses post-CTS
WNS/TNS on the same checkpoint as power estimation, for both candidate and
reference. The CTS power replay must match the original native CTS STA; its
slacks are also cross-checked against the CTS metrics JSON. Pre-CTS OPT timing
is only an upstream validity check, never the thermal penalty input. The separate
Timing task keeps its pre-CTS endpoint. No additional OPT or routing is performed.

Let W and T be `max(0, -WNS)` and `max(0, -TNS)`, in ns. Normalize each against
the immutable reference using the Timing task's zero-deficit rule: divide by
the reference when it exceeds 1e-12 ns; otherwise return 1 if the candidate is
also at most 1e-12 ns, or `1 + candidate_deficit / 1e-12` otherwise.
The cost is `C = 0.6 * relative_W + 0.4 * relative_T`.

`combined_score = (reference_mean_K / candidate_mean_K) - 0.1 * max(0, C - 1.05)`.
Thus a 10% timing-cost regression subtracts 0.005 from the thermal reward. Timing
improvements earn no extra bonus; this is a soft penalty, not a feasibility gate.
The 5% threshold applies to the combined cost, not to either metric individually.
Costs and penalties are dimensionless fractions, not percentage-point values.
Scores are not clamped; severe timing regressions can produce negative scores.
Missing/nonfinite timing data fail evaluation rather than implying no penalty.

| Feedback field | Meaning |
| --- | --- |
| `wns_setup_ns`, `tns_setup_ns` | Candidate post-CTS setup slack metrics, ns (one upstream OPT) |
| `reference_wns_setup_ns`, `reference_tns_setup_ns` | Matching immutable reference measurements, ns |
| `wns_deficit`, `tns_deficit` | Nonnegative magnitudes of negative setup slack, ns |
| `normalized_wns_cost`, `normalized_tns_cost` | Dimensionless candidate/reference deficit ratios |
| `normalized_sta_cost` | Weighted timing cost C; reference is 1 |
| `timing_degradation_fraction` | `max(0, C - 1)`; 0.10 means 10% worse |
| `timing_tolerance_fraction` | Unpenalized timing regression allowance, 0.05 |
| `timing_penalty_excess` | `max(0, C - 1.05)` |
| `timing_penalty_weight` | Fixed dimensionless weight, 0.1 |
| `timing_penalty` | Score deduction `0.1 * timing_penalty_excess` |
| `timing_penalty_applied` | Whether the cost exceeds 1.05 |
| `unpenalized_thermal_score` | Dimensionless reference/candidate mean temperature ratio, computed in kelvin |
| `combined_score` | Final penalized reward; higher is better |
| `timing_evaluation_boundary` | `post_cts_one_opt`; same physical checkpoint as power estimation |

This scoring revision changes task/evaluator digests and the explicit authority
contract. Publish a newly qualified authority before a new search; do not rewrite
old authorities, experiment feedback or rankings. Sealed physical evidence may
be independently revalidated without rerunning EDA. Legacy thermal profiles keep
their original temperature-only score.

## Prompt presentation

Current timing tasks (five and ten rounds) and the ten-round thermal task explain
the chip context first, the optimization problem second, and measurements last.
Their instructions omit release hashes, directory history, thread settings and
qualification procedures. Task configuration, mounted permissions and the trusted
evaluator still enforce those details; shorter prose does not expand edit scope.

The Codex and Agentic3D Task prompts identify the source checkout as the starting
program and give the configuration entry point, plan path and submission format.
Agentic3D reviewers retain their required JSON fields and evidence references.
The rendered context includes each history/feedback payload once and omits the
duplicate user_task text; complete controller context remains unchanged on disk.
Prompt shortening alone does not change acceptance, budgets or physical settings.
The separately requested thermal timing penalty above changes scoring and adds
the listed feedback fields.

Instruction text participates in the task digest. These edits therefore require
a new release and matching authority qualification before another search; do not
reuse an older record by rewriting its digest. These local revisions do not
restart an existing experiment or resolve the separate model-authentication block.
