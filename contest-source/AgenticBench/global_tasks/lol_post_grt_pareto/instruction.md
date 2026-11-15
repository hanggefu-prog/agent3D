---
components: [partition, placement, hbt, timing_opt, cts, grt]
---

Improve the source code and parameters of the 3D-IC backend design tool for the
multi-case, paired-seed LoL task through partition,
placement, HBT assignment, timing optimization, CTS and GRT. Optimize post-GRT setup
timing, total routing overflow and peak silicon temperature. Timing combines
baseline-normalized WNS and TNS deficits using the frozen weights (default:
0.7 WNS + 0.3 TNS). Minimize these three objectives; there is no single overall
score or requirement to improve every objective over the baseline.
We treat power as a constraint to reduce the number of objectives in Pareto
optimization. Each case's mean across paired seeds must be at most
the frozen limit times its original baseline mean (default: 1.10). Use raw watts
without a normalization floor. A gain in another case cannot offset a violation.
Peak temperature is the maximum over all 200 active silicon tiles (100 per die)
for each case/seed, not a mean or a maximum across cases/seeds. Retain average
temperatures as diagnostics only.

You may tune key parameters early, but parameter tuning has limited potential.
Try changes to the algorithms themselves. We focus on multi-stage co-optimization
that changes multiple correlated algorithms or policies in different design
stages, to improve the final PPA objectives. If progress stalls in one stage,
explore other stages and how they work together, within the allowed files.
Keep the fixed flow and its single OPT pass.

The experiment harness (the host) manages rounds, budgets, source checkout and
archive updates. Each round uses one agent invocation. Select one promising
training-frontier parent and state one coherent hypothesis, then use the bound
parent-checkout tool to obtain that source. Continue editing in the same
invocation, return one candidate and stop. The harness verifies and freezes it,
then uses the EDA backend tool to build it and run at most one complete
optimization matrix. It updates the archive before the next round. The EDA
backend tool runs the physical design flow and reports post-GRT measurements;
the harness aggregates scores, accounts for the budget and updates the frontier.
The harness retains every feasible non-dominated candidate, including different
sources with identical objective vectors. Over-limit results stay in history
but cannot enter the frontier or become parents.
Use per-case and per-seed diagnostics to assess variability rather than selecting
a favorable seed. Each round lists its evaluation cases and seeds. You may use
those cases for local tests.

Explore freely in a writable development copy of the selected parent, using its
installed runtime and the supplied optimization cases, within resource limits.
You may add tests, edit development files, build code and run local EDA experiments.
Synthetic inputs are optional. Record local commands and results as diagnostics.
Only the declared changed_files are submitted over a clean parent; all other
files come from that parent. Submitted changes must follow the exact file/function
scope, preserve interfaces, connectivity, physical constraints and supplied seeds,
and must not specialize behavior using benchmark names or known answers.
Final file creation/deletion, executable-mode changes and symlink changes remain
forbidden. The strict verifier and independent evaluator remain authoritative;
local experiments cannot change their inputs, implementation, budgets or scores.

Python partition, placement and HBT algorithms use the source snapshot directly.
OpenROAD C++ edits in Resizer, CTS or GRT require the verified incremental build
and an installed candidate image before evaluation. Each candidate has one
parent; several children can evolve concurrently from previously selected parents.
Failed, incomplete or illegal candidates retain diagnostic artifacts but cannot
enter the frontier.
