# Placement candidate promotion to two-pass timing optimization

This evaluation profile replays a placement candidate selected by the dynamic
learning-rate or initialization/preconditioner task. It never asks a model for
a new candidate. The harness preserves the case's frozen input, floorplan, and
partition, then advances the candidate through placement, die initialization,
PDN validation, HBT construction, and exactly two coordinated OPT passes.

Final setup WNS/TNS come from continuous passive-HBT global STA. A successful
result requires a complete timing graph, zero unclocked or unconstrained
endpoints, complete Liberty coverage, and two completed passes on both dies.
Legality, PDN, HBT, or timing-validity failures are retained as evaluation
outcomes; the profile must not bypass a physical correctness gate to manufacture
STA numbers.
