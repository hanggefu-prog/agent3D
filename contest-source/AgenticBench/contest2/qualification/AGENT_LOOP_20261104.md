# Default GNN Agent feedback loop

Later protocol changes and current validation are recorded in [LOOP_POLICY_20261104.md](LOOP_POLICY_20261104.md).

Both default GNN Agents now repeat Codex edit, return, candidate submission,
native evaluation wait, and feedback delivery. They no longer return immediately
after the first evaluation. Native waits run outside the Codex call.

The outer harness retains authority over the shared evaluation and model-cost
budgets. Ordinary candidate rejections and failed native evaluations become
feedback for the next edit. An uncertain accepted evaluation stops further work.
An error after acceptance does not cause a duplicate submission. The Agent
returns the highest-scoring verified candidate for its own baseline and restores
that function source into program.py.

The controller exposes the validated source from the frozen candidate capsule
when reporting a successful evaluation. History and Codex sessions are separated
by GNN baseline, while both baselines retain the shared budget. Restarting the
Agent restores its verified best source and latest evaluation feedback.

Smoke mode stages the same default Agent as formal mode, with a prescribed small
first edit and a single-cell evaluation. The existing verifier still requires
two code changes, actual evaluator feedback, and no evaluation of the second
draft. No physical algorithm, function template, evaluator metric, reference,
seed matrix, image, or budget-setting file changed.

Validation on 2026-11-04: all 74 contest-2 host regressions passed. Nineteen new
tests execute both shipped Agents with simulated model/native operations and
exercise the real controller, candidate validation, and transactional ledger for
history isolation and source integrity. Coverage includes three-round feedback,
failed/rejected candidates, deduplicated acceptance, budget/model failures,
uncertain evaluations, best-source recovery, and smoke/default Agent identity.

This change has not received a new real-model/native smoke run. Earlier native
smoke receipts belong to their recorded source versions. The already delivered
release archives are not updated by this source commit.
