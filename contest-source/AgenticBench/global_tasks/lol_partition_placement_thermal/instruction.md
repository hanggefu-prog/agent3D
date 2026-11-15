# LoL partition-placement co-optimization for average temperature

Jointly improve the partition and two-die placement of the user-selected LoL
design. The authoritative endpoint is the maintained full Open3DBench backend
followed by HotSpot steady-state 3D grid analysis.

The fixed no-evolution reference frontend is maintained TP-GNN partitioning
(default global-2way decoding, not TA3D) followed by Concurrent DREAMPlace
two-die placement, matching the STA co-design frontend baseline. Do not use the
general LoL Mt-KaHyPar default as this task's reference. The trusted verifier
checks reference partition/placement method manifests. Missing/mismatched
provenance requires a correctly generated reference, not relabeling an old run.
This baseline choice does not restrict algorithm development in the scope-free
workspace task; the focused candidate-program adapter retains its documented
TP-GNN/Concurrent-DREAMPlace intervention surface.

Minimize `temperature.global_average_k`, defined as the arithmetic mean of all
physical top- and bottom-silicon grid-tile temperatures emitted by HotSpot.
The reported higher-is-better `combined_score` is
`reference_global_average_k / candidate_global_average_k`. Per-die average and
global maximum temperatures, total power, HPWL, congestion, timing, DRC, and
runtime are diagnostic rather than alternate objectives.

The case-matched reference, frozen input/floorplan, downstream HBT/OPT/CTS/
routing/report implementations, HotSpot binary/configuration, 10x10 grid,
318.15 K ambient, provisional shared stack model, evaluator, and five-evaluation
budget are immutable. This task studies thermal gains caused by coordinated
partition and placement decisions, not by changing the thermal model or backend.
The evaluator also owns all random seeds (I/O 1, partition 1001, placement
1000, and deterministic routing order -1); they are common-random-number
controls, not optimization variables.

Use one coherent workspace context to reason across partition and placement.
Begin every round with an explicit plan, then edit and test the subject
implementation. Target stages guide adaptive replay but are not a source-file
allowlist. Do not modify or bypass protected benchmark/evaluator files.

A candidate is valid only when partition and placement are complete and legal,
the full backend reaches thermal evaluation, all stage manifests and artifact
hashes validate, HotSpot reports both physical die layers without warnings,
temperatures are finite and no lower than ambient, and mapped power is positive
and conserved. Accept a candidate only from trusted final evaluation.
