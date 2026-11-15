# LoL partition-placement co-optimization for setup timing

Jointly improve the partition and two-die placement of the user-selected LoL
design. The authoritative endpoint is pre-CTS continuous passive-HBT global
setup STA after exactly one native coordinated 3D OPT pass.

The fixed lower-is-better cost is

`0.60 * WNS_deficit / reference_WNS_deficit + 0.40 * TNS_deficit / reference_TNS_deficit`,

and the reported `combined_score` is its reciprocal. WNS and TNS deficits are
`max(0, -slack)`. The case-matched reference, frozen input/floorplan, clocks,
timing features, HBT materialization, OPT/STA implementation, evaluator, and
ten-evaluation budget are immutable. Cut metrics, HPWL, overflow, HBT count,
and intermediate timing are diagnostic only.
The evaluator also owns all random seeds (I/O 1, partition 1001, placement
1000, and deterministic routing order -1); they are common-random-number
controls, not optimization variables.

Use one coherent workspace context to reason across partition and placement.
Begin every round with an explicit plan, then edit and test the subject
implementation. Target stages guide adaptive replay but are not a source-file
allowlist. Do not modify or bypass protected benchmark/evaluator files.

A candidate is valid only when partition assignment is complete and within the
10% area-balance gate, both die placements are complete and legal, the single OPT
pass finishes on both dies, the continuous passive-HBT timing graph is complete,
all timing values are finite, and there are no missing Liberty cells,
combinational loops, unclocked endpoints, or unconstrained endpoints. Accept a
candidate only from trusted final evaluation, never from a proxy improvement.

This version follows the reference-20260923 physical baseline. Historical two-pass
references and scores are incompatible. The trusted evaluator does not apply the
two-pass OPT patch. Reference qualification is separate from candidate finals;
shared settings do not by themselves prove physical artifact identity.
