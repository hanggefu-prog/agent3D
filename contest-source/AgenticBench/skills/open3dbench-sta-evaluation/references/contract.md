# STA evaluation contract

## Boundary

This Skill begins only after an optimization method has selected a candidate.
Candidate generation, model prompting, mutation, population management, and
accept/reject decisions belong to tasks or evolve frameworks. The Skill may
materialize and replay a selected candidate, but it must not alter it.

The three maintained evaluation profiles live in `evaluation_profiles/`:

- `placement_two_pass`: fixed-partition placement promotion;
- `tp_gnn_placement_two_pass`: a selected TP-GNN plus placement candidate
  promoted through the common two-pass STA objective;
- `tp_gnn_ta3d_two_pass`: TA3D TP-GNN objective validation followed by fixed
  placement.

An evaluation profile uses the same immutable candidate materialization and
Open3DBench adapter code as a task, but it is intentionally absent from
`local_tasks/` and from the search-task inventory.

## Required inputs

- a candidate file or materialized program selected by a completed task;
- a case-matched normalized Open3DBench configuration;
- a baseline run or benchmark-input bundle containing every frozen upstream
  stage named by the profile;
- an optional case-matched reference run when normalized timing scores are
  required;
- the pinned Open3DBench source, runtime image, DREAMPlace install, and an
  OpenROAD/OpenSTA binary with the passive-HBT cutpoint query.

For fixed-partition placement promotion, `00_input`, `10_floorplan`, and
`20_partition` are immutable. For TA3D objective validation, `00_input`,
`10_floorplan`, and `20_partition/timing_input` are immutable while partition
is rerun from the supplied program.

## Execution

The physical cutpoint is:

```text
selected candidate
  -> placement or partition+placement
  -> die initialization and legalization
  -> PDN validation
  -> HBT construction
  -> 3D OPT pass 1
  -> 3D OPT pass 2
  -> continuous passive-HBT global STA
```

Do not continue across a failed prerequisite. Do not silently substitute a
different case, partition, candidate, binary, Liberty set, timing graph, or
number of OPT passes.

## Result

`evaluation_result.json` is the machine-readable result. A successful record
must include `runs_successfully=1`, `opt_passes=2`, finite initial and final
global setup WNS/TNS, `sta_graph_complete=1`, and
`sta_unconstrained_endpoints=0`. It should also retain pass-one timing,
placement HPWL, per-die or maximum overflow, HBT count, OPT runtime, artifact
paths, and provenance.

A failed legalization, PDN, HBT, OPT, or STA gate is still a scientifically
useful completed evaluation record. Keep its logs and failure classification;
do not relabel it as a successful candidate.
