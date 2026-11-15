# Benchmark validity review (2026-10-19)

This is the historical review and experiment record. The later
[production v5 integration](MERGE_VALIDITY_V5_20261021.md) documents corrections
to path verification, accounting, profile defaults and evidence claims; it also
preserves production capabilities absent from the original review branch.

Branch `review/benchmark-validity-20261019`, based on `d798d134`. This branch
implements the P0–P2 recommendations of the benchmark review. Every change is
additive or opt-in unless stated otherwise; existing frozen tasks, archives and
release evidence are not relabeled. No model call or agent campaign was
started by this work. Small native experiments were run on rafa1-gcr (E1–E4
below; E5 replays existing logs); everything else reads existing receipts only.

## What changed

### Global post-GRT scoring (opt-in `scoring.version: 3`)

- `grt_bench/scoring_v3.py`: temperature as rise over ambient, geometric or
  arithmetic aggregation across cases, absolute or ratio overflow with per-case
  scales, per-case worst-case and uncertainty reporting.
- Noise-aware dominance: per-objective tolerance `k·sqrt(2·nf² + u_a² + u_b²)`
  plus a strictly lower objective sum. It is order-independent and reduces to
  exact dominance at `k = 0`. Tasks without `version` keep v2 byte-for-byte.
- `pareto.heldout_report` / `freeze_selection`: freeze a selection digest on the
  train split, then report it on a held-out split.
- `analysis/ranking.py`, `scripts/rank_campaigns.py`: hypervolume, HV-vs-budget
  curve and its area, bootstrap confidence intervals, mean log score.
- `scripts/estimate_noise_floor.py`: noise floor from repeated identical
  matrices, or a labeled across-seed proxy.
- Documentation: `AgenticBench/docs/SCORING_V3.md`.

### Independent overflow measurement (opt-in)

- `grt_bench/guide_congestion.py` and `grt_bench/independent_overflow.py`:
  overflow re-estimated from `grt.guide` demand and DEF track capacity,
  independent of FastRoute's own report. The estimator is a one-sided lower
  bound; the rule is `E ≤ 1.05·R + 5`.
- The estimator is part of the trusted evaluator (mounted at `/trusted`). It
  first lived under `open3dbench/backend/`. That is the candidate workspace, so
  a candidate could have edited the code that audits it; the `estimator_sha256`
  pin would have caught a change, but only as an audit failure. The scored
  overflow is still read with the candidate's `_grt_overflow_count`, exactly as
  `three_d_route` scores it.
- `verify.py` runs it only when the task has an `independent_overflow` block.
- `scripts/audit_grt_overflow.py` audits existing GRT outputs.
- Documentation: `AgenticBench/docs/INDEPENDENT_OVERFLOW.md`.

### Harness

- `global_bench/agent_adapters.py`: agent adapter interface, Codex CLI adapter,
  token usage parsing and a cost table; usage/cost are written to each round.
- Campaign options (all opt-in): independent prompt review, fresh vs resumed
  sessions, adapter choice, price table, and classified infrastructure retry
  with `retries.json`, reconciliation guard and a `retry-infrastructure` command.
- `launch_codesign_batch.py --model` is checked against the task allowlist.
- `grt_bench/incremental.py`: stage dependencies derived from the import graph;
  proof v2 records it, proof v1 still verifies.
- Documentation: `AgenticBench/docs/AGENT_ADAPTERS.md`; round/optimization docs fixed.

### Engineering tasks (T2/T3/T4)

- `engineering_bench/scoring.py`: scoring v2 with calibrated profiles and gates;
  `capabilities.py`: capability evidence; evaluator identity v2; judge v2
  (family independence, redaction, quoted citations, median panel).
- `--scoring-version {1,2}`, **default 2**. Version 1 remains available.
- Per-task `evaluation-v2.json` and `expert-reference.json` (expert QoR is
  `null` until measured).
- Documentation: `AgenticBench/docs/engineering_tasks.md`.

### Local track and repository hygiene

- `local_bench/routing_scoring.py`: routing score v2, used only when a
  `task.yaml` declares `scoring.version: 2` (none does yet). Task IDs, prompts,
  formulas and references of the eight local tasks are unchanged.
- `local_bench/routing_wrapper_integrity.py`: the trusted routing wrappers must
  stay byte-identical to the reference. The check is skipped when a caller does
  not pass the Open3DBench root (`build_routing_candidate_image.py`,
  `qualify_archive_grt.py`, `check_task_sources.py`, matrix initial-candidate
  check); wiring those callers is follow-up work.
- `local_bench/matrix_profile.py` and `reference_configs/default_matrix_profile.yaml`:
  a declarative default evaluation matrix and a `render-matrix` command. The
  profile is an unqualified template.
- `benchmark_inputs/ibex_lol_ng45_cpu_smoke`: CPU-only smoke input
  (`qualified: false`) for the "5-minute path" in `docs/tasks.md`. All five
  steps were run in the release image (E2 below).
- `TASK_SETTINGS_REPORT.md` and `HARNESS_DESIGN_REPORT.md` rewritten in English
  against the code; the Chinese originals are archived byte-identical under
  `AgenticBench/docs/archive/`.
- Local tasks keep 0.6 WNS / 0.4 TNS; post-GRT keeps 0.7 / 0.3. A local 0.7/0.3
  variant would need a new task ID or score version (documented in `tasks.md`).
- `scripts/check_doc_consistency.py`: compares the README and task docs with
  task definitions, weights and `DEFAULT_SCORING`; it runs inside the maintained
  gate. `ci/doc-consistency.yml.example` is an untested CI example.
- `docs/REPO_HYGIENE.md`: repository layout and hygiene rules.
- `scripts/openevolve_docker.py` resolves the git directory with
  `rev-parse --git-dir`/`--git-common-dir` relative to the source instead of
  `--path-format=absolute`. The latter needs git ≥ 2.31, and rafa1-gcr has
  2.25.1.

## Tests

Maintained gate (`AgenticBench/scripts/test_maintained_tasks.py`), on this
branch and on the base `d798d134`:

| Environment | Base | Branch | Failures |
| --- | --- | --- | --- |
| local (macOS, Python 3.9) | 1279 passed, 7 failed | 1533 passed, 7 failed | same 7 |
| rafa1-gcr (Python 3.9 venv, git 2.25.1) | 1277 passed, 10 failed | 1531 passed, 10 failed | same 10 |

None of the failures come from this branch:

- In both environments, the three `global_routing_*` tasks fail
  `test_initial_materialization_is_unchanged` and two `test_local_matrix`
  tests. They need the OpenROAD-3D submodule source, which is not initialized
  in these checkouts. After `git submodule update --init --depth 1
  OpenROAD-3D/openroad` locally, the gate passes with no failures (1543
  passed).
- On rafa1-gcr, three packaging and compiler-cache tests also fail on the base.

Before the `openevolve_docker.py` fix, the branch also failed
the launcher tests on git 2.25; those pass now. Root `tests/`: 514 passed,
4 failed, the same 4 as the base. `check_doc_consistency.py`: OK.

## Real-data validation on rafa1-gcr

Run from a separate worktree under `/data-ssd/gcr/review-validity-20261019/`,
reading existing campaign receipts and SQLite archives in read-only mode. The
running campaign and its database were not touched.

**Independent overflow audit.** 12 distinct GRT outputs from
`20261018-partition-placement-20` (10 NG45, 2 ASAP7) are all consistent with
the reported overflow when audited with their own platform directory. Totals
(reported/estimate): ten at 0/0, one at 3/3, one at 73/37 (limit 81.65), and
the two ASAP7 runs at 0/1. The DEF GCELLGRID agrees with the derived grid on
every die. An ASAP7 run audited with the NG45 platform fails to parse
(`unknown adjustment layer range: M2-M9`), so the platform must match.

**v2 replay.** Re-aggregating 24 archived candidates from four campaigns
(`20261015-incremental`, `20261016-formal`, `20261016-frontend-scope`,
`20261018-partition-placement-20`) with this branch's code reproduces all 24
archived v2 rows bit-for-bit, including digests.

**Noise floor.** The identical baseline matrix appears in three campaigns of
task `c060ea39…` with identical metrics. This is **not** repeat evidence: on
rafa1-gcr, all 69 job ids that appear in more than one campaign root are hard
links to a single run directory (same inode), because roots are seeded from
one another. Genuine repeats are E1 and E4 below; they also give exactly 0.
The across-seed spread is much larger (normalized timing ≈ 0.06, overflow ≈
0.16, temperature ≈ 0.003). Using it as a noise floor is a conservative proxy,
not measurement noise.

**v3 replay.**

| Campaign | Candidates | Front v2 | Front v3 |
| --- | ---: | ---: | ---: |
| 20261016-formal (floor 0) | 9 | 2 | 2 |
| 20261016-frontend-scope | 3 | 1 | 1 |
| 20261018-partition-placement-20 | 1 | 1 | 1 |
| 20261015-incremental (across-seed proxy, k = 2) | 11 | 2 | 9 |

With a zero floor v3 dominance equals exact dominance. With the across-seed
proxy most candidates are no longer distinguishable, which is the intended
conservative behavior but too coarse to rank. Ratio overflow and rise-over-
ambient temperature change the vectors (for example formal-front overflow
0.873 → 0.671), not the front.

**E1: a v3 task with an enforced overflow audit, run end to end.** A small
post-GRT task was built from the bounded-halo task:

- Cases: `bp_fe-ng45` (train) and `bp_fe-asap7` (test); seeds s1 and s2.
- Scoring v3: rise-over-ambient temperature, geometric aggregation, ratio
  overflow, noise dominance with k = 2 and floor 0.
- An enforced `independent_overflow` block that pins the trusted estimator's
  SHA-256.

The task was prepared with this branch's `grt_bench` (a new evaluator
identity) in a separate root. It ran through the real queue service and one
worker on rafa1-gcr: GPU frontend, CPU backend, and container verification
with `/trusted`.

| Candidate | Case / seed | Split | Minutes | Status | Overflow R/E |
| --- | --- | --- | ---: | --- | --- |
| baseline `b193b298` | ng45 / s1 | train | 10.6 | verified | 0/0 |
| baseline `b193b298` | ng45 / s2 | train | 10.2 | verified | 0/0 |
| native `317743de` | ng45 / s1 | train | 10.6 | verified | 0/0 |
| native `317743de` | ng45 / s2 | train | 14.4 | verified | 0/0 |
| baseline `b193b298` | asap7 / s1 | test | 14.4 | verified | 0/0 |
| baseline `b193b298` | asap7 / s2 | test | 13.0 | verified | 0/0 |

Results:

- **Overflow audit.** Every receipt is `consistent` and `enforced`. The pinned
  estimator hash matches, and `tech_lef` comes from the protected
  platform of each case: `nangate45_3D` for train and `asap7_3D` for test.
- **Held-out report.** A selection digest was frozen on train, then reported
  on the test split with `heldout_report`. It reports `split: test`,
  `archive_used: false` and `selection_binds_candidates: true`, and the
  baseline is feasible and nondominated.
- **Metrics bit-identical to the original campaign.** All six cells match the
  same candidate, case and seed in the original bounded-halo campaign, where
  ASAP7 was a train case. The new
  evaluator does not change measured metrics. Because E1 ran fresh in its own
  root, this is a genuine repeat, and its noise floor is 0.
- **v3 aggregate.** The native candidate scores timing 1.330, temperature rise
  1.0048, overflow 1.0 (ratio, 0/0 is defined as 1). The front is the baseline
  alone.
- **Hypervolume.** HV(1,1,1) = 0, the degeneracy noted above: the baseline sits
  at exactly (1, 1, 1). HV(1.1,1.1,1.1) = 0.001.
- **Seed sensitivity.** With two seeds, the across-seed proxy floor is timing
  0.080 and temperature 0.0014. Seed s3 of the original campaign gives about
  half the TNS of s1 and s2 for both candidates. Seed sensitivity is large
  compared with the native candidate's effect.

The case has zero overflow, so this run checks the audit wiring and the trust
location on real artifacts. It does not test the
audit's ability to tell real overflow from false overflow; E3 below does.

**E2: the 5-minute path in the release image.** Steps 1–5 of `docs/tasks.md`
were run on `ibex_lol_ng45_cpu_smoke` in the release image (`sha256:a339ce29…`;
4 CPUs, 8 GB, no GPU, no network). The run found two documentation defects,
now fixed in `tasks.md`:

- DREAMPlace is at `/opt/dreamplace` in the image, so `prepare-input` failed
  until `OPEN3D_LOCAL_DREAMPLACE_INSTALL` was set.
- `evaluate` needs a git checkout for its isolated clone.

Measured times:

| Step | Time | Result |
| --- | ---: | --- |
| 3 `prepare-input` | 335 s | input prepared |
| 4 reference | 47 s | `combined_score` 1.0 |
| 5 candidate | 43 s | `combined_score` 0.995, HPWL +0.6 % |

The step-5 candidate adds a 5 % Gaussian jitter to the initial x positions.

An earlier candidate only changed the `random_center_init_flag` branch. The
config sets that flag to 0, so the change had no effect. That run produced a
byte-identical `placement.gp.def`, which shows the path is deterministic.

**Hypervolume degeneracy.** Against reference point (1, 1, 1), hypervolume is
0 in three campaigns because front candidates match the baseline exactly on
temperature. Report HV against a reference point beyond 1 (for example 1.1) or
report per-objective improvement alongside it.

**E3a: offline under-reporting against the overflow audit.** Two retained
`ariane133-ng45` GRT outputs with real overflow (`a9ec4979`: R = 20, E = 13;
`adaf47d9`: R = 37, E = 17, all on the top die) were audited with their
FastRoute per-layer overflow scaled by a factor f before rounding, as a
candidate that edits its own report would do. The DEF was regenerated from the
retained ODB because retention deletes it.

| f | `a9ec4979` R′ | audit | `adaf47d9` R′ | audit |
| ---: | ---: | --- | ---: | --- |
| 1.0 | 20 | consistent | 37 | consistent |
| 0.75 | 11 | consistent | 23 | consistent |
| 0.5 | 7 | **rejected** | 15 | consistent |
| 0.25 | 2 | rejected | 6 | rejected |
| 0.1, 0 | 1, 0 | rejected | 0, 0 | rejected |

Reporting 0, or anything below `(E − 5) / 1.05`, is rejected on both dies and in
total. Because the estimator is a one-sided lower bound (E/R = 0.46–0.65 on
these outputs), a candidate can still hide up to about half of a real
overflow. Ways to close the gap: calibrate a per-case E/R ratio from qualified
references and test `R ≥ E / ratio − tolerance`, score `max(R, E / k)`, or score
the estimate itself.

**E3b: a cheating candidate run through the real queue.** Two candidates were
derived from the honest `adaf47d9` candidate (`ariane133-ng45` s1, R = 37,
E = 17). Each changes only the overflow parser in its own
`open3dbench/backend/`: one reports 0, the other halves each layer before
rounding. Both ran in a separate root through the queue service, one worker on
GPU 1 and container verification with the enforced audit:

| Candidate | Reported | Audit (top / total) | Outcome |
| --- | ---: | --- | --- |
| honest `adaf47d9` | 37 | — | scored 37 |
| reports 0 | 0 | 0 vs E 17, limit 5: **failed** | job `failed`, not scored |
| halves | 15 | 15 vs E 17, limit 20.75: passed | `succeeded`, scored 15 |

- Both cheats leave routing untouched: both guides and both SPEFs are
  byte-identical to the honest run, and power, temperature, WNS and TNS are
  equal. Only the scored overflow changes.
- The zero report is rejected in the verification container
  (`Independent overflow audit failed: top:bound; total:bound`), so it cannot
  enter the archive.
- The halved report passes and is scored 15 instead of 37: a 59 % improvement
  in one objective that does not exist. This is the lower-bound gap from E3a
  reproduced end to end.
- The queue records the rejection only as `Native container failed; inspect
  retained log`. The audit reason is in `verification/container.log`. It should
  be copied to the job result so that operators and agents can tell an audit
  rejection from a crash.

**E4: repeats under real scheduling and thread perturbation.** The baseline
`bp_fe-ng45` s1 cell was run again with the container CPU quota and
OMP/MKL/OpenBLAS/`OPEN3DBENCH_THREADS` set to 2 and to 1 (E1 used 8/4 CPUs in
the frontend/backend and 4 threads). OpenROAD's own `set_thread_count 4` comes
from the case configuration and was not changed, so with one CPU its four
threads are time-sliced. With E1 and the original campaign run (the six
campaign roots that list job `fb683f59` share one hard-linked run), this gives
four independent runs spanning GPU 0 and GPU 1, a solo run and one among 11–14
concurrent jobs, runs up to 17 hours apart, and 1, 2 and 4 BLAS threads:

- Metrics are identical in all four runs.
- `grt.guide` and `global_route.odb.spef` of both dies are byte-identical in all
  four, and so is the routed DEF wherever retention kept it.
- Only `runtime_seconds` in `grt_metrics.json` differs.

The test case `bp_fe-asap7` s1 was repeated the same way with 2 CPUs and 2
threads. With E1 and the original campaign run (`d1df8bfe`, also shared by
hard links), this gives three independent runs. Metrics, both guides and both
SPEFs are identical, as is the routed DEF where it was retained; again only
`runtime_seconds` differs (21.8 s vs 22.0 s).

The repeat noise floor of this flow is therefore exactly 0 for a fixed
candidate, seed and evaluator image, also under these perturbations. A v3
tolerance must come from another source: seeds (the across-seed proxy above) or
a different evaluator image or hardware, neither of which is measured here.

**E5: harness options against real campaign logs.**

- *Usage accounting (a bug, fixed).* Replaying every Codex round of
  `astra-20`, `astra-train3` and `bounded-halo-20` showed that `turn.completed`
  carries the running total of the whole session, including earlier resumed
  rounds. The first version of the adapter summed per round, so the eight
  resumed `bounded-halo-20` rounds recorded 89.8 M input tokens instead of
  18.85 M (4.8× overall; round 8 alone 18.85 M instead of 1.50 M). Rounds now
  record the delta from the same thread's previous total, which is positive and
  plausible for all seven resumed rounds (1.33–4.65 M input, 11–19 k output),
  and keep the raw total as `agent_session_usage`.
- *Event parsing.* Thread ids match the recorded ones in 10 of 10 Codex
  attempts. The adapter counts one terminal call fewer than `tool-calls.jsonl`
  in every round: it counts `exec` only, and the round log also has the
  `checkout_parent` call.
- *Claude campaigns.* `fable5-20` and `fable5-train3` were run with Claude Code
  `stream-json` logs, which the Codex adapter cannot parse (usage `null`). The
  production framework used for those campaigns has its own
  `agent_backend`/`workspace_claude.py` path that records no usage and is not
  on this branch; the two need to be reconciled before a Claude adapter is
  added here.
- *Failure classifier.* The strict classifier was run read-only on every failed
  or interrupted job of the five 2026-10-19 campaigns. All eight were
  classified not retryable, which is correct: two candidate errors
  (`Nonfinite/negative power` in the backend) and six operator stops whose
  lease expiry was followed by `cancelling`. No real infrastructure failure
  occurred, so recall is covered only by the unit tests.

## Not yet qualified

- v3 tolerances need a measured noise floor. Genuine repeats (E1, E4) give
  exactly 0, even with GPU, load and thread perturbation, and the across-seed
  proxy is too coarse. A nonzero floor could only come from changes not
  measured here (evaluator image, hardware) or from treating seeds as noise.
- Repeats must be counted by run directory inode, not by job id: campaign
  roots share hard-linked runs.
- Any `grt_bench` edit changes `evaluator_identity()`. Frozen tasks (for example
  v14) must be re-frozen against this evaluator before v3 or the overflow audit
  is enabled; their old digests are not relabeled.
- Held-out cases, references and expert solutions for engineering scoring v2
  are not yet measured (`expert-reference.json` QoR is `null`).
- Engineering T3 v2 rejects existing contexts without local receivers; rebuild
  them with `native multi-context`. T4 clock power is reported `not_measured`.
- Engineering v2, the harness options and the local routing scoring v2 have
  unit and fake-executor tests but no native qualification run.
- The routing tests that need the OpenROAD-3D submodule source were not run
  because the submodule is not initialized in these checkouts.
- The overflow audit rejects gross under-reporting on real outputs (E3a, E3b),
  but it cannot catch under-reporting by up to about half while the estimator
  is a lower bound. Enabling it as a scored-objective guard needs one of the
  calibrations listed under E3a.
