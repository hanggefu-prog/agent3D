# First-round background

## The design and flow

A 3D chip stacks silicon dies connected by hybrid bonding terminals (HBTs).
Cells implement logic; macros are larger blocks such as memories. The netlist
defines their connections. Physical design assigns blocks to dies, places them
and routes their wires.

This task optimizes the source code and parameters of a 3D-IC backend design tool
for two-die logic-on-logic (LoL) designs. The flow of the backend tool is:

Input and floorplan → partition → placement → power grid (PDN) → HBT assignment
→ timing optimization (OPT) → clock tree synthesis (CTS) → global routing (GRT)
→ timing, power and thermal reports.

Inputs, die sizes and external pins are fixed. There is one OPT pass. Scores
use estimates after GRT, before detailed routing.

The main methods are:

- **Partition:** TP-GNN uses a graph neural network with timing features to
  assign cells to the two dies, balancing die loads and reducing costly cuts.
- **Placement:** Concurrent DREAMPlace adjusts positions on both dies to reduce
  wirelength and spread cells, then legalizes their positions.
- **HBT assignment:** Compute ideal HBT positions from the pin geometry on both
  dies, then assign legal grid sites. Track-aware greedy assignment handles
  high-fanout nets first, reduces two-die HPWL and avoids nearby escape-track
  conflicts.
- **Timing Opt:** OpenROAD Resizer sizes cells and inserts buffers to repair
  electrical limits and setup timing. Cross-die timing analysis supplies each
  die with boundary arrival times and required times through the HBT model.
- **CTS:** TritonCTS builds a balanced clock tree on each die. The top tree's
  load guides bottom-tree construction; timing analysis through the clock HBT
  then transfers arrival time and slew to the top die.
- **GRT:** OpenROAD FastRoute builds routing trees on each die, reroutes congested
  nets and assigns metal layers. HBT boundary pins connect the two dies.

OpenSTA reports timing and power, and HotSpot estimates temperature.

## Multi-stage co-optimization

Partition affects placement; placement affects routing, timing and heat.
We focus on multi-stage co-optimization that changes multiple correlated
algorithms or policies in different design stages, to improve the final PPA
objectives. Editable stages are partition, placement, HBT assignment, timing
optimization, CTS and GRT, within the task's
allowed files. If progress stalls in one stage, explore changes in other stages
and how they work together.

You may tune key parameters early, but parameter tuning has limited potential.
Go beyond tuning and try changes to the algorithms themselves.

## What the metrics mean

Minimize three objectives: timing, routing overflow and peak temperature.
Timing combines normalized WNS and TNS deficits with the task's fixed weights
(default: 0.7 and 0.3). Setup slack is required arrival time minus actual arrival
time; negative slack means data arrives late. The five raw measurements are:

| Metric | Meaning |
| --- | --- |
| `wns_deficit_ns` | The largest setup timing violation, in ns: `max(0, -WNS)`. |
| `tns_deficit_ns` | The sum of negative setup slack magnitudes across timing endpoints, in ns: `max(0, -TNS)`. |
| `overflow` | Routing demand beyond capacity, summed across both dies after GRT. |
| `temperature_peak_k` | Highest temperature among 200 silicon tiles, 100 per die, in K for each case/seed. |
| `power_w` | Total estimated cell power on both dies, in W, at fixed switching activity. |

We treat power as a constraint to reduce the number of objectives in Pareto
optimization. Each case's mean across seeds must stay within the task's limit
relative to the original baseline (default: at most 10% higher). WNS and
TNS remain visible separately. Die sizes are fixed, so area is not scored separately.
