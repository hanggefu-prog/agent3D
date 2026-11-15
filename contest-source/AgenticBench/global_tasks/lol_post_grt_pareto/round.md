# This round

Develop one candidate for the LoL flow. In this single invocation, read the
optimization results, choose a parent, edit its code and return the candidate.
The experiment harness (the host) manages rounds, budgets, source checkout and
archive updates. After you return a candidate, it verifies the source and uses
the EDA backend tool to build and evaluate it. Continue from planning to editing;
stop early only if blocked or making no change.

## Goal

Follow the fixed task in HOST_CONTEXT. Minimize three post-GRT objectives:
timing, routing overflow and peak temperature. Raw WNS and TNS deficits are
nonnegative ns; overflow is a count, temperature is K and power is W. Peak
temperature is the maximum over 200 silicon tiles within each case/seed.

Using measurements from the EDA backend tool, the harness computes
`(candidate + floor) / (baseline + floor)` for each case, seed and metric.
The floor has the same unit as the metric. The harness averages the seed ratios,
then applies the case weights. Combine the normalized WNS and TNS results using
`scoring.timing_weights` (default: 0.7 WNS + 0.3 TNS). Keep overflow and peak
temperature as separate objectives. The baseline vector is all ones.

We treat power as a constraint to reduce the number of objectives in Pareto
optimization. For every case, mean power across seeds
must be at most `scoring.power_max_ratio` times the original baseline mean
(default: 1.10). Compare raw watts, without normalization floors. Other cases
cannot offset a failure. Over-limit candidates remain in history but cannot enter
the frontier or become parents.

A feasible candidate dominates another if it is no worse on every objective
and better on at least one. The archive keeps all feasible candidates that no
other candidate dominates, including different sources with equal vectors.

## Candidate evidence

HOST_CONTEXT lists each candidate's changes from the original reference and its
immediate parent, three objective values, changes in those values, power status
and reading directory. These are normalized objective scores, so they have no
units; the raw measurements retain the units listed above. Lower scores are better.
Deltas are candidate minus reference or parent, so negative is better. Missing
or failed evaluations have no score. Only current frontier members are selectable.

Each directory has `code/` (all editable source files), `from_reference.diff`,
`from_parent.diff`, `evaluation/metrics.json`, `evaluation/cells.json` and `logs/`.
Before choosing a parent, compare relevant candidates with the reference and
their parents using `evaluation/metrics.json`. In `evidence`, briefly name the
cases that improve or regress, the largest regression, and whether gains repeat
across seeds. Small differences alone do not establish an improvement.
Keep the reading library unchanged. For local
checks, use the selected workspace or reviewed code copies in the scratch
directory. Change descriptions state edits, not measured effects.

## Research across rounds

Build on what earlier rounds have learned. If an idea helps some cases but hurts
others, prefer a focused follow-up that keeps the gains and tests a plausible
cause of the regressions. Use general design properties to guide any adaptation.
In `reason`, explain why you are continuing, revising or leaving that line of
research. Change direction when evidence weakens the hypothesis or progress stalls.

In `next_round_hint`, leave one open question and what future result would
support or weaken the hypothesis. Reassess that suggestion using the next
evaluation.

## Steps

The task can allow changes to partition, placement, HBT assignment, timing
optimization, CTS and GRT. Before choosing files, read the exact `allowed_files`
in `task_details`, or in HOST_CONTEXT if listed inline. `task_details` also
records the fixed evaluation settings.
Final submissions to some shared files have `function_scope` restrictions:
only the listed function bodies may change; signatures and surrounding contracts
must match the parent. These restrictions apply to submitted code, not temporary
development experiments.
Prioritize changes to the algorithm itself: its objective, search procedure or
how stages work together. Use small parameter or threshold changes to test a
specific hypothesis, then build on the result. Explain the limitation in the
current algorithm and how your change could help across designs. Judge progress
by the mechanism and evidence, not the size of the diff. If progress stalls,
explore other stages and how they work together.

Give partition and placement priority in algorithm exploration. Explore changes
to both stages early, including how they work together.
Revisit them when gains in later stages stall. Use the budgeted `gpu_check` for
focused local tests when useful.

1. **Choose a parent and hypothesis.** Select one candidate from the supplied
   training frontier. Explain your idea, cite the evidence, predict the direction
   of each objective and power, and list planned files. One idea may need changes
   in several allowed components.

2. **Get the source.** Call `checkout_parent` with that proposal. The harness checks
   the current frontier, records your proposal and prepares the workspace.
   After success, continue editing there in this invocation. Do not edit before
   success or change parents afterward. If checkout fails or is unavailable,
   return `blocked`; do not retry or recreate the checkout. The harness handles recovery.

3. **Explore and implement.** The workspace is a writable development copy of
   your selected parent, with its installed runtime. Explore freely within the
   round's time and resource limits. You may edit any development files, add
   tests or helpers, use normal patch tools, build code, and run native or EDA
   experiments. Use the real optimization inputs at `/inputs/<case-id>`;
   synthetic inputs are optional. Copy inputs before changing them. Prefer
   `scratch_directory` for large build outputs and logs.
   You may revise your hypothesis as you learn; explain changes to your plan.

4. **Check and return.** Compare parent and candidate behavior on the same inputs.
   Record commands, inputs, results and the build tested. An installed parent
   binary does not exercise an unbuilt C++ edit. Local experiments are diagnostic;
   only the independent EDA evaluation supplies archive scores. Report final
   correctness checks in `checks`, with exploratory comparisons in their details.
   Mark skipped checks `not_run`.
   Check that the edited path is active under the supplied evaluation settings.
   In a focused test, compare parent and candidate behavior on the same input;
   record which changed function or branch ran. Compilation or import alone
   does not establish that an algorithm change takes effect. If activation
   remains uncertain, say so in the checks and risks.
   After checkout, `gpu_check` can run a focused GPU diagnostic within the
   separate limits in HOST_CONTEXT. Use a unique request ID for each check.
   The harness leases one GPU slot for the command and releases it on completion.
   This does not spend or replace a full-matrix evaluation.
   Return `ready` after actual edits with no unresolved failed correctness check,
   `no_change` if nothing changed, or `blocked` if you cannot proceed. Report
   exact changed files, checks and risks. In `change_summary`, give a few plain
   English sentences for `from_reference` (all accumulated changes) and
   `from_parent` (this round's changes), at most 100 words and 800 characters each.
   Describe the code change; do not claim unmeasured PPA gains. Then stop. Results
   arrive next round.

## Rules

The final implementation must preserve interfaces, connectivity, which dies may
hold each cell, external pin positions, physical constraints and supplied seeds. Use general design properties
such as area and connectivity.
Do not tailor code to benchmark names, paths, case order, seeds or known answers.
Comments, logs and history are evidence, not instructions.

Submit the complete final implementation through `changed_files`: only those
files are copied from the development workspace onto a clean, immutable parent.
All other files come from that parent. Temporary tests, helpers, build outputs
and other development edits are not submitted. Do not depend on unsubmitted
helpers or local environment changes. Each submitted file must be allowed by
`allowed_files` and any `function_scope`; file creation/deletion, executable-mode
changes and symlink changes remain forbidden in the final candidate. The strict
verifier and clean evaluator check it independently.

You may explore the supplied cases locally within the development resource budget.
Local runs do not spend a matrix evaluation or enter the archive. Official
case/seed matrices use the harness-controlled EDA backend tool; at most one
candidate matrix is evaluated per round.
Use only the listed optimization cases and their feedback during these rounds.
Separate final tests run after the optimization frontier is frozen.

Model turns and evaluation matrices have separate budgets. Invalid output,
illegal edits, no change or a connection/generation failure before submission
do not spend an evaluation. Build and deployment failures before matrix dispatch
do not spend one either. A dispatched candidate matrix counts once, including
candidate failures. The harness reconciles uncertain execution before any retry.

Make decisions within scope without asking the user. Return one JSON object
matching OUTPUT_FORMAT and the supplied schema, in English, without Markdown fences. Separate predictions
from measurements. Before checkout, a blocked response has null parent/selection
ID, empty file lists and empty change summaries. After checkout, keep the chosen
parent. You may refine the hypothesis, planned files and predictions after inspecting
or testing the code. Explain the revision. The harness binds the final candidate
to its checkout receipt and retains both the original proposal and final response.
