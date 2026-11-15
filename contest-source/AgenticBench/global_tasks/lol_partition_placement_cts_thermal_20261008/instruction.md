# Reduce temperature through partitioning and placement

## Background

A logic-on-logic (LoL) three-dimensional integrated circuit (3D-IC) spreads logic
across two stacked tiers. Partitioning and placement affect heat distribution,
wire loads, buffering and power consumption.

## Your task

Improve the partition and placement algorithms to lower the mean steady-state
temperature across both silicon layers while limiting timing degradation. The starting implementation uses a graph
neural network for timing-driven partitioning (TP-GNN), a global two-way partition
decoder and concurrent two-tier DREAMPlace placement. You may explore algorithm
changes, parameter changes, or both; there is no fixed function to return.

## How results are compared

Each candidate is placed and legalized, followed by power delivery network (PDN)
construction, hybrid bonding terminal (HBT) assignment, one timing optimization
(OPT) pass and clock tree synthesis (CTS). The evaluator estimates cell power from
placement-based wire resistance and capacitance, including passive cross-tier
connections, then runs HotSpot. Power uses a fixed vectorless primary-input
activity of 0.1 rather than a workload trace. No routing is performed.

HotSpot uses a 10 by 10 grid on each die, an ambient temperature of 318.15 kelvin
(45°C), and the same thermal stack for every candidate. The objective is the mean
over all 200 equal-area silicon tiles, not the hottest tile.
The temperature reward is reference mean temperature divided by candidate mean
temperature, in kelvin. Timing is measured after CTS, on the same design used for
power estimation,
using worst negative slack (WNS) and total negative slack (TNS), in nanoseconds.
For each, the deficit is `max(0, -slack)`. Divide candidate deficits by their
reference deficits and combine them as `C = 0.6 * WNS_ratio + 0.4 * TNS_ratio`.
An effectively zero reference deficit (at most 1e-12 ns) gives ratio 1 if the
candidate is also effectively zero; otherwise use `1 + candidate_deficit / 1e-12`.

The timing cost `C` is 1 at the reference. Up to 5% degradation is allowed without
penalty. Above that, `timing_penalty = 0.1 * max(0, C - 1.05)` and
`combined_score = temperature_reward - timing_penalty`.
For example, 10% timing degradation subtracts 0.005 from the temperature reward.
The reference scores 1; higher is better. This is a soft penalty, not rejection.

`global_average_temperature_k` is the measured mean; `normalized_sta_cost` is `C`.
`bottom_average_temperature_k` and `top_average_temperature_k` show each tier's
mean. `global_maximum_temperature_c` is the hottest tile in degrees Celsius,
and `total_power_w` is cell power in watts. Missing measurements are unknown; failed
evaluations do not establish an improvement. These are pre-route estimates,
not signoff temperatures.

Use feedback to test a clear idea, such as reducing wire loads or redistributing
power-intensive cells.
