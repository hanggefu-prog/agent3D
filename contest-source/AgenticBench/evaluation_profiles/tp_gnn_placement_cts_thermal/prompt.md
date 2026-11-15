# Post-CTS cell-power thermal evaluation v1

Start from the same frozen input/floorplan and TP-GNN timing features as timing
co-design. Reference: TP-GNN global-2way plus Concurrent DREAMPlace. Candidate
source may evolve partition/placement within the declared policy. Exactly TWO
OPT passes use the identical trusted passive-HBT STA patch as the timing task.
Then complete CTS on both dies, extract per-instance power using fixed vectorless
input activity 0.1 and placement-estimated RC, and run HotSpot (10x10 physical
tiles per die, ambient 318.15 K, fixed stack). Include CTS-added clock cells.

Score = reference mean temperature K / candidate mean temperature K over ALL
200 physical silicon tiles. Higher is better; trade-offs between dies are allowed.
Power, peak temperature, and pre-CTS WNS/TNS are diagnostics. No GRT, DRT, DRC,
routed SPEF/RCX, or full-flow report is required or claimed. This is an early
cell-power thermal model, not routed/signoff thermal. Do not mix its reference,
score or artifacts with the historical full-route thermal profile.

Two-pass timing completeness, legal frontend, actual TP-GNN/Concurrent method
provenance, CTS completion, power extraction/coverage/conservation, all 200 tiles,
and warning-free extraction/HotSpot remain mandatory. Seeds, input, footprint,
activity, thermal stack, evaluator and backend are immutable.
