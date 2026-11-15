# Coordinate clock tree synthesis across two dies

The maintained CTS flow builds both local trees and uses a continuous passive
HBT timing assembly to measure their complete paths. It already transfers remote
capacitance to the source side. However, the remaining cross-die latency offset
is mainly reported rather than used to adjust the physical trees. OpenROAD
already provides local clustering, clock buffers and latency balancing; you may
reuse those capabilities.

Starting from the frozen post-OPT checkpoint, improve the CTS implementation so
that construction and adjustment consider the whole two-die clock distribution.
You may change the coordinator, construction order, clustering, topology,
buffer sizing/placement and necessary native CTS code. New related helper files
and tests are allowed. No particular algorithm, solver, number of passes or
local-compensation method is required within the published runtime budget.

Reduce global same-clock-domain skew while satisfying the area, electrical and
physical guards. Setup/hold WNS/TNS, when available from the same STA, are
diagnostics, not a second objective or an acceptance gate for this task.
Keep original data logic, clock source/period/edge
definitions, sink inventory, PDK and clock-HBT positions
and count fixed. Rebuild clock buffers and branches as needed. Preserve clock
gating and auxiliary domains correctly; never mix unrelated clocks into one
skew calculation or silently drop unsupported sinks.

The harness owns standard placement legalization. It may move existing standard
cells as needed to legalize the new clock trees, using the same fixed settings
for the baseline and candidate. You may not optimize data-cell coordinates,
change the placement objective, modify legalization settings, or bypass its
checks. Before the trusted final legalization, the harness restores original
data-cell coordinates and orientations from the frozen post-OPT input; fixed
macros, I/O and HBTs remain fixed. Clock-cell placement is part of your CTS
algorithm. Report displacement as a diagnostic, not as a second objective.

Submit real, readable two-die CTS artifacts with coherent clock and HBT maps.
Do not edit the independent STA/RC evaluator, its reports, the SDC constraints
or legality checks. Local boundary approximations may guide your algorithm,
but the unchanged full passive assembly determines final quality. Never present
placement-estimated RC as routed signoff timing.

Evaluation stops after CTS and harness-owned legalization. Reuse the frozen
post-OPT checkpoint; do not rerun its upstream stages. The independent evaluator
uses placement-estimated RC with propagated clocks solely to score current-stage
global skew. No later OPT, GRT, DRT, extraction, power or thermal run is required.
Do not add downstream-flow qualification to this task's completion criteria.

Use seed 1001. Submit one implementation and evaluate each of the four cases
once; there are no ten-round searches or repeated seeds. Compilation and small
functional/regression tests are available during development; extra full-case
physical searches are not. Native edits must use the provided candidate-image
builder, so the executable is tied to the submitted source.

Test normal cross-die construction, a case with little offset, setup/hold-sensitive
branches and preserved clock-domain ownership. Explain the implementation, tests
and known limits in English. Distinguish evidence from expected improvements.
