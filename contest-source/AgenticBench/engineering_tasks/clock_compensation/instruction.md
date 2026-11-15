# Repair cross-die clock arrival imbalance

Both dies already have legal clock trees. They distribute the same clock, but
their register arrivals may have different offsets. Balancing a tree inside one
die does not necessarily align it with the other die. A hybrid bonding terminal
(HBT) is a passive connection, not an independent clock generator.

Implement `plan_clock_compensation(context)` in
`engineering/clock_compensation.py`. Select useful local changes on exclusive
clock branches: buffer masters, stages, sink groups and legal placement choices.
The supplied executor performs the physical edits; do not reimplement OpenDB,
legalization or the STA engine. Helpers and tests are welcome. There is no code
length limit or required optimization algorithm.

The context includes both dies' branch connectivity, clock/domain/edge identity,
sink arrivals, load/slew, affected setup and hold paths, legal buffer options,
placement opportunities and resource limits. Large reports remain available by
reference. Missing or ambiguous timing information is not zero delay.

The deployed interface is `clock-compensation-context-v2`, documented in
`context-v2.md`. It supplies current native measurements and explicit Liberty
delay estimates, not measurements of hypothetical edits. Select one plan using
that information; the protected executor then measures its actual physical
effect once. Do not require an oracle of post-edit results to make a decision.

Improve global same-domain clock skew without violating the published timing or
resource guards. Delaying a common trunk does not correct relative skew. A new
buffer can reduce wire delay as well as add cell delay, so use physical feedback
rather than assuming a constant delay per buffer. Later capture clocks can help
setup and harm hold; both directions of cross-die data traffic matter.

Keep original clock-tree branch ownership, existing cell locations, data logic,
clock definitions and HBT locations/count unchanged. Use real physical changes,
not `set_clock_latency`, a changed clock period, excluded sinks or edited reports.
Returning an empty plan is correct when no safe improvement exists; an
unconditional no-op is not an implementation of this feature.

Evaluation stops at the edited and legalized post-CTS databases. The protected
evaluator runs placement-estimated RC and joint propagated-clock STA on both
dies, reports global same-domain skew and setup/hold WNS/TNS, and checks the
published current-stage timing guards. It does not rerun synthesis, partition,
placement, OPT or CTS, or run GRT, DRT, extraction, power or thermal analysis.
Reference and candidate use this same stage-only endpoint.

Use seed 1001. Develop and submit one implementation for all four cases.
Each case receives exactly one final physical evaluation, not ten optimization
rounds and not a seed sweep. You may run compilation and supplied small functional
tests during development; do not run extra full-case physical searches. Reference
qualification is separate and does not consume candidate evaluations.

Test a useful compensation, an already aligned tree, a shared-trunk rejection,
and a hold-limited or space-limited case. Finish with an English summary of what
changed, what you tested and what remains unverified. Submit source edits, not
fabricated physical results. The harness, not your summary, establishes success.
