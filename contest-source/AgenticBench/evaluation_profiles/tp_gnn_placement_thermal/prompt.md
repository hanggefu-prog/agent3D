# TP-GNN and 3D placement promotion to HotSpot average temperature

This is an evaluator-owned promotion profile, not a search task. It accepts a
candidate produced by a global workspace agent or another external optimizer
and reruns the following fixed pipeline:

`frozen input + floorplan -> TP-GNN partition -> Concurrent DREAMPlace two-die placement -> PDN -> HBT -> OPT -> CTS -> routing -> report -> HotSpot`

The no-patch reference uses the same maintained TP-GNN (default global-2way
decoder) and Concurrent DREAMPlace frontend as the STA co-design baseline.
Both method selectors are explicit in the initial candidate. This is not the
general LoL Mt-KaHyPar default and not the separate TA3D decoder experiment.
The focused adapter accepts source evolution inside these implementations;
the driver enforces the frontend selectors and validates the actual partition/
placement manifests of both reference and candidate. Missing or mismatched
method provenance is an error, never an excuse to relabel historical results.
Existing candidates may omit placement.algorithm only when the resolved base
selects Concurrent DREAMPlace (or leaves it unspecified for explicit injection).

The thermal backend/OPT policy remains unchanged; sharing frontend methods does
not replace the full thermal pipeline with the timing task's two-pass endpoint.

The sole objective is the mean of all physical top- and bottom-silicon HotSpot
grid-tile temperatures. The higher-is-better `combined_score` is the
case-matched reference average in kelvin divided by the candidate average.
Per-die average, peak temperature, power, cut, HPWL, congestion, timing, DRC,
and runtime remain diagnostics.

The evaluator fixes HotSpot to a 10x10 grid, 318.15 K ambient, and the
maintained provisional shared 3D stack. Thermal analysis must finish without
warnings, include both physical die layers, emit finite temperatures no lower
than ambient, and conserve positive mapped power. Partition assignment must be
complete and within the 10% area-balance gate, placement must contain both dies
without missing connection objects, and every backend stage must succeed.
