# Post-GRT objective profiles

Two task-bound prompt versions are maintained. The renderer chooses from the
endpoint in the frozen task; an agent cannot switch profiles or change scoring.

| Profile | Prompts | Raw measurements | Retention |
| --- | --- | --- | --- |
| Peak v2 | `global_tasks/lol_post_grt_pareto/{background,round}.md` | WNS/TNS deficits, overflow, peak K, W | Three-objective non-dominated set; per-case mean power ≤110% of original baseline |
| Timing v3 | `global_tasks/lol_post_grt_timing/{background,round}.md` | WNS/TNS deficits, overflow | Minimum score; retain equal-score sources |

## Timing v3

For each case/seed, retain the existing paired normalization of timing:

```
timing = 0.7 * (WNS_deficit + WNS_floor) / (baseline_WNS_deficit + WNS_floor)
       + 0.3 * (TNS_deficit + TNS_floor) / (baseline_TNS_deficit + TNS_floor)
penalty = clip((overflow - 300) / (600 - 300), 0, 1)
score = mean_over_cases(mean_over_seeds(timing + 0.1 * penalty))
```

The current optimization set has three equally weighted ng45 cases: bp_fe, bp
and ariane133, each with seeds 1001/1002/1003. Other splits remain unchanged.
The contract also supports frozen case weights; never infer them from results.

The overflow bounds and lambda are task fields. The default upper bound of 600
is an explicit design choice, not a maximum fitted to the candidate population.
Overflow 300/450/600 contributes 0/0.05/0.1; greater overflow stays at 0.1.
This gives penalty the baseline timing scale (baseline timing is 1), **not** the
same bounded range: timing remains unbounded above so regressions are visible.
Compute each penalty before seed/case averaging. Raw overflow remains visible.

The v3 endpoint does not execute power reports, validate power or invoke HotSpot.
Physical legality, stage identity, routing evidence, joint timing completeness,
cross-die timing paths and latch checks remain required. No missing measurement
is replaced by zero. Old profile defaults and power/thermal checks are retained.

Create a separate task with `python -m grt_bench.profiles OLD_TASK NEW_TASK --id NEW_ID`.
This creates no baseline receipts, archive or evaluations. Requalify the original
baseline under the new endpoint, then start a separate campaign. Analysis replay
accepts only the unchanged original baseline across objective profiles, with
unchanged physical settings, seed pairs and timing normalization. It checks
execution-driver compatibility and immutable physical evidence. Historical
candidates cannot be relabelled into a new task-bound ancestry.

## Harness changes

- The final proposal may refine its hypothesis, planned allowed files, predictions
  and explanation. The authenticated checkout still fixes the parent. Preserve
  `response.raw.json`, the canonical response and the receipt-binding audit.
  A conflicting real parent, task or round remains invalid.
- Activation preflight rejects semantic no-ops and the known disabled cross-die
  weighting branch, including its disconnected optional CSV preparation, only
  when frozen inputs prove multiplier=1 for every optimization case. Unknown
  edits remain eligible. This is not a proof of arbitrary program equivalence;
  local tests should exercise changed paths using the actual configuration.
- A terminated native-process failure is recorded with case, seed and diagnostic
  location. It spends one reserved evaluation, cannot enter the archive, and
  the next round continues. Verification, integrity and uncertain process faults
  block further rounds until reconciled. No automatic physical retry is added.
- `global_bench.grt_campaign_loop` is the reviewed-budget driver for new campaigns.
  Use it instead of older copied launchers that treat every failed round as fatal.
- Successful verified siblings of a failed matrix still receive ODB-first
  compaction. Failed outputs are retained for diagnosis. GPU phase release and
  source verification remain mandatory.
