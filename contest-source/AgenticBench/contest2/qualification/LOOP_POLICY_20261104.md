# Contest loop policy and recovery

This update supersedes the termination and timeout behavior described in
`AGENT_LOOP_20261104.md`. It does not replace earlier native qualification receipts.

- A candidate rejected by the edit-scope or function-interface audit consumes
  one evaluation. The ledger records `rejected` atomically without native jobs.
  Repeating the same request is idempotent, including after the budget closes.
  The compatibility field `evaluations_accepted` counts all charged attempts.
- `uncertain` is nonterminal. The outer harness checks the original controller,
  exact container names and retained results. An active or uninspectable
  execution stays `reconciling`; no replacement native attempt is launched.
  Confirmed termination permits a verified result or an explicit failed result.
  Missing ownership evidence and orphaned setup/build work remain pending host
  inspection rather than being assumed terminated.
- A failed model turn retains evaluator feedback and retries the next round
  while the fee/evaluation budget remains open. Smoke checks allow unsuccessful
  model attempts but still require two successful edits and one native evaluation.
- Agent sessions, container commands, resource waiting and native evaluations
  have no execution deadline. Model API requests retain a 1800-second timeout.
  MCP calls yield running command sessions instead of killing long commands.
  Communication probes, database lock waits and cancellation cleanup safeguards
  are not native execution deadlines.
- Timeout-only adapters preserve the frozen physical/evaluator source bytes.
  Their exact source transformations fail if the expected frozen code differs.
  Execution-owner records include the execution-policy source digest.

Validation on 2026-11-04: 129 contest-specific host regressions passed, with
simulated model/native operations and real transactional ledgers and bridges.
Coverage includes rejection charges and deduplication, model failure before and
after a successful score, real default-Agent feedback loops, uncertainty
inspection, retained resource ownership, asynchronous terminal execution,
credentials isolation and runtime policy binding.

Task 2 now returns case, seed, status, exception type/message and bounded native
log excerpts for failed cells. Partial or truncated summaries retain the
available diagnostics instead of making the feedback API fail. Uncertain cells
prevent additional cells from starting until the outer harness checks ownership.

The reference-admission modules and both task contracts match the preceding
commit, and no physical algorithm or scoring formula changed. The additional
legacy matrix suite had 70 passes and 15 failures; loading the pre-change matrix
implementations in the same checkout produced 69 passes and the identical 15
failures (the additional passing test covers uncertain-cell dispatch). These
failures concern unavailable full physical-source fixtures and legacy task/seed
expectations. The comparison detected no new failures.

No new real-model/native smoke run was performed for this update. Existing
published archives and remote runtime deployments have not been refreshed by
this source change. Previous native receipts apply only to their recorded source.
