# Post-GRT scoring contract v3

This document describes the **peak-temperature endpoint's opt-in scoring v3**. It is
separate from the production **timing-only task schema v3**. The number 3 alone does
not identify a contract.

| Task profile | Task schema | Endpoint | Scoring selector | Result vector |
| --- | --- | --- | --- | --- |
| Existing peak v2 | 2 | `post-grt-passive-hbt-peak-v2` | Exact keys `timing_weights`, `power_max_ratio` | timing, overflow, peak temperature |
| Peak with scoring v3 | 2 | `post-grt-passive-hbt-peak-v2` | Explicit `scoring.version: 3` and all fields below | timing, overflow, peak temperature |
| Production timing-only | 3 | `post-grt-passive-hbt-timing-v3` | Exact keys `timing_weights`, `overflow_penalty`; no scoring version | score |

Validation rejects a scoring object borrowed from another profile. No validator
injects defaults or upgrades old tasks. Existing v2 task digests, archive rows and
`aggregate()`/`frontier()` outputs remain unchanged
(`tests/test_grt_scoring_v3.py::FrozenV2Tests`). Timing-only scoring retains its frozen
0.7/0.3 timing plus overflow-penalty formula; it does not use the temperature,
power or noise-aware rules below. See [objective profiles](POST_GRT_OBJECTIVE_PROFILES.md).

Every `grt_bench/*.py` file contributes to evaluator identity. A new integrated
release therefore has a new evaluator digest and requires its own task qualification;
old receipts are not relabelled as evidence for that release.

## Why v3

1. **Diluted temperature.** v2 divides absolute peak Kelvin. The ambient is 318.15 K and
   baseline peaks are 320-329 K, so halving the temperature *rise* moves the ratio by only
   about 0.01. The axis is about 100x less sensitive than the other two.
2. **Asymmetric averaging.** An arithmetic mean of ratios counts a 2x regression on one
   case more than a 2x gain on another case.
3. **No noise model.** Exact dominance with `epsilon=0` treats differences of 0.004 K as
   real. In one real round, a candidate that was 0.7% worse on timing and 8.8% worse on
   overflow entered the front only because of such a temperature difference. The per-case
   `stdev` and `worst` fields were computed but never used.
4. **Reference-sensitive overflow.** When baseline overflow is zero, the floor
   determines the ratio scale. An explicit absolute scale offers a separate ablation.
5. **No ranking metric** for comparing agents or campaigns.
6. **Offline held-out comparison.** This module adds a reporting helper; production
   final-test release and pre-registration controls remain separate host operations.

## Contract

```json
"scoring": {
  "version": 3,
  "timing_weights": {"wns_deficit_ns": 0.7, "tns_deficit_ns": 0.3},
  "power_max_ratio": 1.1,
  "temperature": "rise_over_ambient",
  "aggregation": "geometric",
  "overflow": {"mode": "absolute", "scale": {"bp_fe-ng45": 100, "bp-ng45": 100}},
  "dominance": {"mode": "noise", "k": 2.0,
                "noise_floor": {"timing": 0.002, "overflow": 0.01, "temperature_peak_k": 0.003},
                "seed_uncertainty": false}
}
```

Validation is strict, and every field is required except `seed_uncertainty`, which
defaults to false:

- `temperature`: `"rise_over_ambient"`. `"absolute"` (the v2 behavior) is allowed for ablation.
- `aggregation`: `"geometric"`. `"arithmetic"` is allowed for ablation.
- `overflow`: `{"mode": "ratio"}` (the v2 transform), or `{"mode": "absolute", "scale": {...}}`
  with one positive scale for every case id in the task, across all splits.
- `dominance`: `{"mode": "exact"}`, or `{"mode": "noise", "k": >=0, "noise_floor":
  {timing, overflow, temperature_peak_k: >=0}, "seed_uncertainty": bool}`.
- The task `epsilon` must be 0. The dominance model replaces epsilon boxes, and
  `frontier(rows, epsilon)` rejects a nonzero epsilon for v3 rows.

## Formulas

Here `x` is the candidate value and `r` is the paired baseline value for the same case and
seed. `f` is `normalization_floors[metric]`. Every per-cell value is positive, equals 1 at
equality, and is monotone in `x`.

| metric | per-cell value `v` |
|---|---|
| WNS, TNS, power, and overflow in `ratio` mode | `(x + f) / (r + f)` (the v2 formula) |
| temperature, `rise_over_ambient` | `(max(0, x - A) + f) / (max(0, r - A) + f)`, where `A = ambient_k` |
| temperature, `absolute` | `(x + f) / (r + f)` |
| overflow, `absolute` mode | `exp((x - r) / scale[case])`, so `log v = (x - r) / scale` |

- **Temperature check.** A cell whose candidate or reference peak is more than 1e-6 K below
  ambient is rejected.
- **Why `exp` for overflow.** It is neutral at equality, stays positive, which the geometric
  mean requires, and does not depend on how small `r` is. Under geometric aggregation the
  overflow objective is `exp(weighted mean of (x - r) / scale)`, so an overflow increase of
  one `scale` in every cell multiplies the objective by `e`. Use the scale to state how much
  absolute overflow counts as a large change for each case, for example a fraction of the
  GCell edge count. A cell with `|x - r| / scale > 700` is rejected because the scale is too
  small.

### Aggregation

Let `y` be the value in aggregation space: `y = log v` for geometric, `y = v` for arithmetic.

- `case_c(m) = mean over seeds of y`
- `L(m) = sum_c w_c case_c(m) / sum_c w_c`
- `metric_ratios[m] = exp(L(m))` (geometric) or `L(m)` (arithmetic)
- Timing is combined per cell in aggregation space: `y_timing = 0.7 y_wns + 0.3 y_tns`, with
  the weights normalized. Geometric timing is therefore `ratio_wns^0.7 * ratio_tns^0.3`.
  Arithmetic timing equals the v2 formula.
- `vector[o] = exp(L(o))` or `L(o)`, for `o` in `timing`, `overflow`, `temperature_peak_k`.

The output contains every v2 key, plus:

- `scoring_version: 3`
- `aggregation`
- `dominance`: a copy of the contract's dominance model, which `frontier` reads.
- `per_case_objectives[case][o]`
- `worst_case[o] = {case, value}`: the largest per-case objective. Ties go to the first case
  in task order.
- `uncertainty[o]`: the propagated seed-sampling standard error of the aggregate. Cases are
  treated as fixed and seeds as independent draws within a case:

  `SE_L(o) = sqrt( sum_c (w_c / W)^2 * s_c(o)^2 / n_c )`

  Here `s_c` is the sample standard deviation of `y` over the `n_c` seeds of case `c`, and
  it is 0 when there is one seed. The reported value is `vector[o] * SE_L(o)` for geometric
  aggregation (delta method) and `SE_L(o)` for arithmetic.
- Geometric runs also report `log_mean` and `log_stdev` for each case and metric.

The power constraint is the same as in v2: the per-case seed-mean watts must be no more
than `power_max_ratio` times the baseline mean, and power is not a Pareto axis.

### Noise-aware dominance

Rows `a` and `b` are compared per objective with this tolerance:

`tol_o(a, b) = k * sqrt( 2 * noise_floor_o^2 + [u_a,o^2 + u_b,o^2 if seed_uncertainty] )`

`noise_floor_o` is the standard deviation of one evaluation's `vector[o]` under run-to-run
nondeterminism, as estimated below. A difference of two independent evaluations has
variance `2 sigma^2`. The reference matrix is shared by both rows, so it adds no variance
to the difference. `u` is the `uncertainty` field. It measures sensitivity to seed choice,
which is a different source of variation, so the two are combined as independent variances.

`a` dominates `b` when all three conditions hold:

- `a_o <= b_o + tol_o` for every objective,
- `a_o < b_o - tol_o` for at least one objective, and
- if any `tol_o > 0`, `sum_o a_o < sum_o b_o`.

Tolerant dominance alone can be cyclic. For example, with `tol = 1`, the points
`(0, .75, 1.5)`, `(1.5, 0, .75)` and `(.75, 1.5, 0)` each dominate the next, so every point
would be removed. The sum condition makes the relation acyclic, so the feasible row with the
smallest objective sum is always retained. When `k = 0` or every tolerance is 0, the rule
is exact Pareto dominance.

**Retained set:** every feasible row that no other feasible row dominates. This is computed
pairwise over the whole set, so it does not depend on input order. Rows are returned sorted
by candidate. With `retain_equivalent=False`, a row whose vector is identical to an earlier
row's vector is dropped. Rows within noise of each other do not dominate each other, so
both are kept.

For example, against an incumbent at `(1, 1, 1)` with
`noise_floor = {timing: .002, overflow: .01, temperature_peak_k: .002}` and `k = 2`, the
candidate `(1.007, 1.088, 0.99998)` is noise-dominated. Its temperature gain of 2e-5 is
below the 0.0057 tolerance, and its 8.8% overflow loss exceeds the 0.028 tolerance. Under
exact dominance it would be retained.

`archive.py` is unchanged. Archive rows carry `scoring_version` and `dominance`, so
`Archive.show/select/ingest` use the v3 frontier automatically.

## Held-out report

`pareto.heldout_report(task, {candidate: outcomes}, reference, selection_digest, split="test")`
aggregates a frozen candidate set on the `test` or `auxiliary` split.

- It rejects `train`, and it never reads or writes the archive.
- It returns per-candidate rows and vectors, feasibility, and the noise-aware (or exact)
  non-dominated subset.
- It records `selection_digest` and `selection_frozen_before_evaluation: null`.
  Offline outcomes cannot prove that selection preceded evaluation; the report
  says so in `selection_provenance`.

Commit `pareto.freeze_selection(candidates)` to the evidence store before you submit
held-out jobs, then pass the same value as `selection_digest`. The report then sets
`selection_binds_candidates: true`. Any other pre-registered 64-hex commitment, such as an
archive `frontier_digest`, is also accepted and recorded, but `selection_binds_candidates`
is then false. A matching digest proves only which candidate names it binds. The
host final-test release must supply the actual ordering and frozen-selection evidence.

## Estimating the noise floor

Native repeats require two inputs with different trust roles: outcome envelopes
and a separate operator-supplied allowlist of host campaign roots.

```sh
python AgenticBench/scripts/estimate_noise_floor.py --task task.json \
    --trusted-root /trusted/repeats/run1 --trusted-root /trusted/repeats/run2 \
    run1.json run2.json --k 2 --output noise_floor.json
```

Add one `--trusted-root` and outcome file for each additional repeat. Roots match
exactly after path resolution; authorizing a parent directory does not authorize
all descendants. An envelope cannot authorize its own root.

Each run file has `outcomes` containing the unchanged independent-verifier rows
(or unchanged worker results), plus `provenance` containing the original full task
object and its campaign root. For example, build an envelope from host-exported
files as follows:

```python
import json
from pathlib import Path

source_task = json.loads(Path("source-task.json").read_text())
verified_outcomes = json.loads(Path("run1-outcomes.json").read_text())
envelope = {
    "outcomes": verified_outcomes,
    "provenance": {"root": "/trusted/repeats/run1", "task": source_task},
}
Path("run1.json").write_text(json.dumps(envelope))
```

`provenance.task` is an object, not a filename. Preserve each outcome's candidate,
task digest, endpoint, split, full seed vector, metrics and evidence. Every run must
use the same candidate, normally the original baseline. The script verifies:

- Complete successful host queue cells, matching worker and independent-verifier
  results, and unchanged task, plan and analysis artifact hashes.
- Matching source candidate, full seeds, measurement settings, evaluator and image.
- Different completed physical host leases for repeats of each cell. Analysis
  replays and copied directories cannot create new independent executions. File
  hashes and inodes alone are not used as proof of independence.

The operator-authorized host store is the trust root. These checks detect
inconsistent or copied evidence; they do not authenticate a forged or compromised
host database. Do not authorize model-writable or untrusted directories. Explicit
source-task objects allow old measurements to be rescored under new analysis
settings while retaining their original identities; they cannot change physical
inputs or silently convert timing-only measurements into five-metric peak results.

The estimator computes the per-cell mean as a pseudo-reference, scores each repeat
in the target peak scoring v3 space, then reports sample standard deviations of
run vectors as `noise_floor`. Output also includes `candidate`, `provenance`,
`metric_sigma`, `per_cell_sigma` and a proposed `dominance` block. It never changes
an existing task, archive or evaluation budget.

For synthetic fixtures only, use `--synthetic` with bare complete outcome lists:

```sh
python AgenticBench/scripts/estimate_noise_floor.py --task task.json \
    --synthetic synthetic1.json synthetic2.json --output synthetic-floor.json
```

These outputs say `synthetic: true` and are not native qualification. Full
candidate/task/seed consistency checks still apply. Do not combine `--synthetic`
with `--trusted-root`.

A usable floor needs at least 5 repeated runs of every cell. The script accepts 2 to 4 runs
with a warning. `--across-seeds` treats the seeds of each case as repeats and scales the
result by `1/sqrt(n_seeds)`. This is only a proxy, because it also includes real seed
sensitivity. Prefer true repeats and `seed_uncertainty`.

Estimate the floor with the final scoring transforms, because rise-over-ambient and
absolute-overflow sigmas differ from v2 ratios. The changed noise floor changes the task
digest, so freeze it before the campaign starts.

## Ranking agents

`AgenticBench/analysis/ranking.py` uses only the standard library and provides:

- `hypervolume(points, reference=(1, 1, 1))`: the exact dominated volume of a minimization
  front. A point that is not strictly better than the reference on every axis adds no volume
  and is excluded. This is equivalent to clipping it to the reference box.
- `hv_curve(admissions, budget)`: hypervolume of the feasible admissions after each
  admission, against cumulative cost. The budget is `evaluations` (one per admission
  unless a cost is given), `wall_seconds`, or `tokens`.
- `area_under_curve(curve, limit)`: the step-curve area divided by `limit`. A campaign that
  stopped early keeps its final hypervolume.
- `mean_log_score(scores)`: mean log score for the local track, with a percentile bootstrap
  confidence interval. The RNG is seeded, so results are deterministic.
- `campaign_summary({agent: [campaign, ...]})`: mean, standard deviation, and bootstrap
  interval of the final hypervolume and of the hypervolume-budget area. The budget limit
  shared by all agents defaults to the largest spend. Both final hypervolume and
  AUC obey an explicit limit; admissions after it cannot improve either ranking.
  An admission exactly at the boundary counts toward final hypervolume.

```
python AgenticBench/scripts/rank_campaigns.py --agent A a1.json a2.json a3.json \
    --agent B b1.json b2.json b3.json --budget evaluations --output ranking.json
```

A campaign file is a list of admissions in admission order, or `{"admissions": [...]}`.
Each admission is `{"vector": {...} | [t, o, T], "feasible": bool, "cost": {...}}`.

Run at least 3 independent campaigns per agent, with fresh archives and the same frozen
task and budget. Report the mean and the bootstrap interval, not a single run. The summary
warns when an agent has fewer than 3 campaigns. When intervals overlap, the ranking order is
not a significant difference. Make final claims from the held-out report, not from the
training hypervolume.

## Known limitations

- `optimization.OBJECTIVE` still labels optimization specs
  `paired-ratio-weighted-timing-power-cap-v2`. `scoring.version` is the authoritative
  scoring selector.
- `per_case[...].stdev` in v2 and v3 rows comes from `statistics.stdev`. Its last bit
  differs between CPython 3.9 and 3.12. This predates v3. Re-aggregating an archived
  candidate under a different interpreter version can therefore fail with "Conflicting
  evidence".
