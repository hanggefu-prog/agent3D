# TP-GNN and 3D placement promotion to two-pass global STA

This is an evaluator-owned promotion profile, not a search task. It accepts a
candidate that has already been selected by an external evolution framework or
by the Agentic3D inner task loop. The profile never proposes, edits, ranks, or
repairs candidates.

The fixed pipeline is:

`frozen input + floorplan -> TP-GNN partition -> 3D global/legal placement -> passive HBT materialization -> OPT pass 1 -> OPT pass 2 -> continuous global STA`

The evaluator reruns partition through OPT in an isolated checkout. Partition
timing features and delay/slack hyperedge weights are frozen from a
full-coverage partition-last STA snapshot, so every candidate receives the
same timing input. OPT and STA code are evaluator-owned and cannot be edited.

## Authoritative objective

Only final global setup timing after pass 2 determines the promotion score.
Relative to the case-matched frozen reference, the lower-is-better cost is

`0.60 * final_WNS_deficit / reference_WNS_deficit + 0.40 * final_TNS_deficit / reference_TNS_deficit`

and `combined_score` is its reciprocal. The score is not clipped above 1.0.
Cut nets, timing-weighted cut, cut pins, total and cross-die HPWL, per-die
overflow, HBT count, and pass-1 timing are retained only as diagnostic evidence.

Every successful promotion must complete exactly two coordinated OPT passes on
both dies, use the continuous passive-HBT timing view with remote-die load
visibility, pass graph-completeness checks, contain no missing Liberty cells or
combinational loops, and report zero unclocked or unconstrained endpoints. The
partition must be complete and within the 10% area-balance gate, and placement
must contain both dies with no missing connection objects.

This is a pre-CTS, placement-estimated-RC evaluation. Hold, DRC, detailed
routing, and later signoff stages are outside this profile. A failed physical
or timing gate remains a valid failed evaluation record and must not be relabeled
as a successful result.
