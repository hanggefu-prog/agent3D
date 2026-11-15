# Improve timing through partitioning and placement

## Background

A logic-on-logic (LoL) three-dimensional integrated circuit (3D-IC) spreads logic
across two stacked tiers. Partitioning chooses each instance's tier; placement
chooses its position. Together they affect wire lengths, cross-tier connections,
cell loading and timing.

## Your task

Improve the partition and placement algorithms together, rather than optimizing
an embedding loss or wirelength alone. The starting implementation uses a graph
neural network for timing-driven partitioning (TP-GNN), a global two-way partition
decoder and concurrent two-tier DREAMPlace placement. You may explore algorithm
changes, parameter changes, or both; there is no fixed function to return.

## How results are compared

Each candidate is placed and legalized, followed by power delivery network (PDN)
construction, hybrid bonding terminal (HBT) assignment and exactly one timing
optimization (OPT) pass. Static timing analysis (STA) then measures setup timing
before clock tree synthesis (CTS), using placement-estimated wire resistance and
capacitance with ideal clocks. This is not routed or signoff timing.

The objective balances worst negative slack (WNS) and total negative slack (TNS),
both in nanoseconds. For each, the deficit is `max(0, -slack)`: smaller is better.
The evaluator combines reference-normalized deficits with weights 0.6 for WNS
and 0.4 for TNS, then returns the reciprocal as `combined_score`. The reference
scores 1; a valid score above 1 is an improvement. If a reference deficit is
effectively zero, matching it is neutral and introducing a deficit is penalized.

Use `wns_setup_ns` to assess the worst setup violation and `tns_setup_ns` to
assess the total violation. `normalized_wns_cost` and `normalized_tns_cost`
show their separate contributions; `normalized_sta_cost` is the weighted cost.
Other measurements help explain a result but are not additional rewards.
Missing measurements are unknown; failed evaluations do not establish an
improvement.

Start from the latest feedback and test a clear idea, such as keeping critical
connections close or improving the placement of cells on critical paths.
