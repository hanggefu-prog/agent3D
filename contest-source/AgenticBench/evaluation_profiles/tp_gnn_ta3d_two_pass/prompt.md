# TP-GNN TA3D candidate validation through two-pass STA

This is an evaluation profile, not an optimization task. It accepts a
previously selected maintained or evolved `compute_tp_gnn_loss` implementation
and evaluates that implementation without asking a model for another edit.

The harness reruns the frozen TA3D timing input and TP-GNN partition, uses the
fixed concurrent two-die placement protocol, and performs exactly two
coordinated OpenROAD timing-optimization passes. Final setup WNS and TNS come
from the continuous passive-HBT global STA view. The signed cut, timing-weighted
cut, and cut-net count are retained as diagnostic measurements rather than
substitutes for physical timing.

A physical or timing-contract failure is an evaluation outcome and must remain
visible. Do not bypass partition provenance, legalization, PDN, HBT, timing
coverage, Liberty, graph-completeness, or constrained-endpoint checks in order
to report a successful result.
