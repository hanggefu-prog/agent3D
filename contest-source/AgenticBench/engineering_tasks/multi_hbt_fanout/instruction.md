# Add multi-HBT data fanout synthesis

A high-fanout signal may drive many geographically separated loads on the other
die. The maintained flow normally gives one crossing electrical net one HBT.
For some signals, several separately buffered HBT branches may reduce delivery
delay. This is a physical fanout transformation, not logic duplication or an
assumption that more HBTs always improve timing.

Implement `synthesize(context, output_dir)` in `engineering/multi_hbt.py` and any
needed helpers. Decide which eligible networks to split, partition their remote
loads, choose legal spare HBT sites and buffers, and materialize an executable
two-die result with a complete logical-to-physical branch map. Use the supplied
OpenDB/materialization interfaces; you own the synthesis policy and integration,
not a new router or timing engine. The result must be directly measurable by
joint STA after materialization and legalization; no routing stage is required.

Only single-driver ordinary data networks in the fixed target set are eligible.
Keep one HBT when appropriate; use at most three per logical network. Exclude
clocks, power, tri-state, multi-driver and unsupported exception-bearing nets.
Preserve all original receivers exactly once. Buffer outputs must never be
shorted together. HBTs carrying the same logical value must be isolated electrical
branches where necessary, not parallel links silently fed into a one-link RC
model. Retain the shared-electrical-net rejection in the trusted evaluator.

The context provides drivers/sinks, die ownership, fixed target nets, baseline
timing and input slew, existing and spare HBT sites, library data and exact
resource limits. Original cells, clock trees and existing HBT coordinates stay
fixed. Legalize new cells and preserve PG, layer and connection requirements.

Reduce the mean of each target net's worst original-receiver delivery delay.
The fixed measurement includes the original driver, added buffers, wires and
HBT RC at reference input slew. It includes every preselected target net, even
ones you leave unchanged. Global setup/hold and physical/resource guards still
apply at this step. The endpoint is placement-estimated RC and joint two-die STA,
including setup/hold WNS/TNS. Reuse the frozen post-CTS input: do not rerun
synthesis, partition, placement, OPT or CTS, and do not run GRT, DRT, OpenRCX,
power or thermal analysis. This is not routed or signoff timing. Reference and
candidate must both use this endpoint; old GRT-RC measurements are not comparable.

Use seed 1001 and submit one implementation for all four cases. Each case is
physically evaluated once, with no optimization rounds or seed repetitions.
Use the supplied small tests for development; do not run extra full-case
physical searches. Test one-HBT preservation, real multi-branch materialization,
complete disjoint sink coverage, exhausted sites and unsupported inputs.
Finish with an English implementation/test/limitation summary. A proposed plan
without materialized, independently valid artifacts does not complete the task.
