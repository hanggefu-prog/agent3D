# TP-GNN / placement co-design, post-CTS cell-power thermal v1

Use the `tp_gnn_placement_cts_thermal_one_pass` trusted evaluation profile. Start from
frozen input/floorplan. Reference is TP-GNN-flat global-2way + Concurrent DREAMPlace.
The pre-CTS pipeline and exactly one native OPT pass are identical to timing co-design;
append CTS, fixed vectorless per-instance power with placement-estimated RC,
and HotSpot. Include clock-tree cells. No GRT, DRT, report or RCX is executed.

Minimize mean kelvin over all 200 physical silicon tiles (10x10 per die), ambient
318.15 K and fixed thermal stack. Score is reference mean / candidate mean.
Never use old routed-thermal metrics/reference with this task. Legal frontend,
complete one-pass passive-HBT STA, successful CTS and power/temperature gates are
mandatory; this early cell-power model is not routed or signoff verification.
Submit run.json, 70_cts/checkpoint.json and 95_thermal_snapshot/metrics.json.
Only the trusted profile may generate the power/HotSpot suffix.

This version follows the reference-20260923 physical baseline. Historical two-pass
references and scores are incompatible. The trusted evaluator does not apply the
two-pass OPT patch. Reference qualification is separate from candidate finals;
shared settings do not by themselves prove physical artifact identity.
