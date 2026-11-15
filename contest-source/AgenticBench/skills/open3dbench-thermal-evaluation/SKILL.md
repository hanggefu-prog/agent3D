---
name: open3dbench-thermal-evaluation
description: Evaluate Open3DBench thermal results with HotSpot at placement, post-CTS, or post-route fidelity, and attribute candidate temperature improvement among partition-placement, OPT/CTS, and routing. Use for thermal candidate promotion, fast placement-level screening, or matched reference/candidate thermal attribution; do not generate candidates or change the thermal model.
---

# Open3DBench thermal evaluation

Use the least expensive fidelity that answers the question. Candidate
generation, search policy, and acceptance decisions remain outside this Skill.

## Placement-level screening

Use P0 when an inner loop needs fast feedback before backend implementation.
The evaluator opens the immutable `30_placement/checkpoint.json`, estimates
die-local placement parasitics with the registered PDK RC layers, assigns 0.1
vectorless primary-input activity, reports instance power, and maps it to the
same HotSpot grid and stack used by the release flow.

```bash
PYTHONPATH=<open3dbench-root> python3 -m open3dbench.backend.thermal_snapshot \
  --checkpoint <run>/30_placement/checkpoint.json \
  --platform-dir <open3dbench-root>/OpenROAD-3D/flow/platforms/<platform> \
  --hotspot-root <hotspot-root> --openroad <openroad> \
  --output-dir <new-output>/p0
```

P0 is a placement-level estimate, not signoff thermal analysis. Use it to rank
candidates only when every candidate uses the identical estimator contract.
The power adapter clips a negative per-instance total to zero only when the
sum clipped is at most 0.1% of positive die power, records the count and watts,
and fails above that bound. Zero-power temporary HBT masters need no thermal
geometry; any positive-power instance missing a platform LEF remains fatal.
OpenROAD warning categories and at most 20 samples are stored in metrics while
the complete log remains an artifact.

## Matched-stage attribution

Use P0/P1/P2 attribution to answer how much of a final thermal change arose
before versus after placement:

- P0: `30_placement`, placement-estimated power and RC;
- P1: `70_cts`, after 3D OPT and CTS, with the same placement-estimated model;
- P2: canonical `95_thermal`, using post-route assembly power and extracted RC.

Run the deterministic adapter:

```bash
PYTHONPATH=<open3dbench-root> python3 \
  <bench-root>/skills/open3dbench-thermal-evaluation/scripts/run_thermal_attribution.py \
  --open3dbench-root <open3dbench-root> \
  --reference-run <reference-run> --candidate-run <candidate-run> \
  --openroad <openroad> --hotspot-root <hotspot-root> \
  --output-dir <new-output> --levels p0,p1,p2
```

On the maintained remote runtime, use the fixed container wrapper instead of
reconstructing mounts and library paths by hand:

```bash
scripts/run_remote_thermal_attribution.sh \
  <reference-run> <candidate-run> <new-output-dir> p0,p1,p2
```

The script writes `thermal_attribution.json` and
`thermal_attribution.csv`. It compares temperature rise over ambient, not
absolute Kelvin percentages. The decomposition is:

```text
partition_and_placement = improvement(P0)
opt_and_cts             = improvement(P1) - improvement(P0)
routing_and_report      = improvement(P2) - improvement(P1)
```

A positive contribution means the candidate became cooler than the reference;
a negative contribution means that stage cancelled an earlier gain. Signed
shares may be negative or exceed 100%. Absolute shares describe magnitude only.
P2-P1 also includes the transition from placement-estimated to routed power/RC,
but that estimator change is matched between reference and candidate.

## Default co-design promotion: two OPT passes + CTS cell power

New co-design experiments use `tp_gnn_placement_cts_thermal`. Reference is
TP-GNN global-2way plus Concurrent DREAMPlace. Reuse the timing task's complete
pre-CTS flow and exactly two OPT passes, then append CTS, vectorless cell power
(activity 0.1, placement-estimated RC) and HotSpot. Include CTS-added cells.
No GRT, DRT, report or RCX is required. Use the trusted profile, not a diagnostic
snapshot alone, to qualify the two-pass history and generate final feedback.
Score is reference mean K / candidate mean K over all 200 physical die tiles,
with grid 10x10/die and ambient 318.15 K. This is not routed/signoff thermal.

`scripts/run_remote_thermal.sh` now defaults to this profile (6-hour timeout).
It retains the common wrapper's qualified tool and HotSpot library checks below.
Never reuse a historical one-pass or full-route thermal reference for this task.
Final thermal artifacts are under `95_thermal_snapshot`; final checkpoint is CTS.

## Historical full-flow promotion

Evaluate one selected candidate with the evaluator-owned
`tp_gnn_placement_thermal` profile:

```bash
openagentic3dbench-profile evaluate tp_gnn_placement_thermal \
  --candidate <candidate> --open3dbench-root <source-root> \
  --benchmark-input <bundle> --reference-run <reference-run> \
  --output-dir <new-output-dir>
```

For an explicitly requested historical replay, set
`OPENAGENTIC3D_THERMAL_PROFILE=tp_gnn_placement_thermal`. The deterministic wrapper is
`scripts/run_remote_thermal.sh CANDIDATE OUTPUT INPUT_KIND INPUT
REFERENCE_OR_DASH GPU RETENTION [PROFILE_CONFIG]`. It shares the qualified
OpenROAD/OpenSTA runtime with STA promotion but extends the run through report
and thermal with a 24-hour candidate timeout. The wrapper also requires
`libsuperlu.so.5` and `libblas.so.3` under
`$OPENAGENTIC3D_HOTSPOT_LIB_DIR`, or under
`$OPENAGENTIC3D_RUNTIME_ROOT/hotspot-libs` by default, and exposes that
directory only inside the evaluation container through `LD_LIBRARY_PATH`.
The maintained compute image also keeps DREAMPlace graphics libraries under
`/opt/legacy-libs`; the wrappers append the HotSpot sidecar to that existing
runtime path instead of replacing it.

Require successful input through thermal, a complete partition within the area
balance gate, legal two-die placement, and `95_thermal/metrics.json` identifying
`hotspot-steady-grid`, the fixed grid/ambient/stack, both physical die layers,
zero HotSpot warnings, finite temperatures no lower than ambient, and positive
conserved power. Rank the canonical task by
`temperature.global_average_k`; retain peak temperature, power, HPWL,
congestion, timing, DRC, and runtime as diagnostics.

Never accept candidate-authored thermal metrics. The trusted evaluator owns
OpenROAD power extraction and HotSpot invocation. Preserve metrics, logs,
manifests, source identity, and reference identity for the outer `tell` step.
