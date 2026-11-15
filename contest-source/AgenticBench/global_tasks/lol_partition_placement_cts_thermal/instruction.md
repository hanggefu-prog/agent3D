# TP-GNN / placement co-design, post-CTS cell-power thermal v1

Use the `tp_gnn_placement_cts_thermal` trusted evaluation profile. Start from
frozen input/floorplan. Reference is TP-GNN global-2way + Concurrent DREAMPlace.
The pre-CTS pipeline and exactly two OPT passes are identical to timing co-design;
append CTS, fixed vectorless per-instance power with placement-estimated RC,
and HotSpot. Include clock-tree cells. No GRT, DRT, report or RCX is executed.

Minimize mean kelvin over all 200 physical silicon tiles (10x10 per die), ambient
318.15 K and fixed thermal stack. Score is reference mean / candidate mean.
Never use old routed-thermal metrics/reference with this task. Legal frontend,
complete two-pass passive-HBT STA, successful CTS and power/temperature gates are
mandatory; this early cell-power model is not routed or signoff verification.
Submit run.json, 70_cts/checkpoint.json and 95_thermal_snapshot/metrics.json.
Only the trusted profile may generate the power/HotSpot suffix.
