# Independent GRT overflow audit

## Problem

The post-GRT `overflow` objective is the `Total` row of FastRoute's own final
congestion table, read from `grt.log` by
`open3dbench/backend/three_d_route.py:_grt_overflow_count`. When FastRoute is
candidate-editable, the router reports its own score. The verifier checks
artifact hashes, but nothing checks that the reported number matches the
routing the router produced.

## Components

| File | Role |
| --- | --- |
| `AgenticBench/grt_bench/guide_congestion.py` | Estimator, part of the trusted evaluator (mounted at `/trusted`, never imported from the candidate workspace). Runs no router code. Reads `grt.guide`, `global_route.odb.def`, `grt.tcl` and the tech LEF. |
| `AgenticBench/grt_bench/independent_overflow.py` | Audit receipt, consistency rule, task-contract validation, verifier hook, run discovery. |
| `AgenticBench/grt_bench/verify.py` | One opt-in line: when a task declares `independent_overflow`, `verify_cell` adds the audit receipt (and enforces it). |
| `AgenticBench/scripts/audit_grt_overflow.py` | CLI for one or many GRT directories or run roots. |

When no task declares `independent_overflow`, behaviour is unchanged. Metric
extraction, scoring and existing contracts stay byte-identical.

## Estimator model (version 1)

* **GCell grid**: OpenDB's default `getGCellTileSize`. The tile is 15 x the
  median preferred-direction track spacing of routing levels 2-4 (an odd
  spacing is reduced by 1). The grid is anchored at the DIEAREA lower-left
  corner, with `floor(extent / tile)` GCells per axis, and the last row or
  column absorbs the remainder. `--gcell-size` overrides the tile. The DEF
  `GCELLGRID` is written by the router, so it is only compared
  (`def_gcellgrid_agrees`) and never used.
* **Capacity**: preferred-direction `TRACKS` (Y for horizontal layers, X for
  vertical) inside the GCell row or column, counted on `[lo, hi)` like
  OpenROAD's nominal `computeGCellCapacity`. Several patterns on one layer are
  averaged like `getAverageTrackSpacing` (`--track-model exact` uses the exact
  union instead). The `grt.tcl` adjustment gives
  `max(1, floor(cap * (1 - a)))` in single precision. Layers outside the
  `set_routing_layers -signal` window have zero capacity. Obstructions, PDN
  and macro blockages are **not** subtracted.
* **Demand**: each guide box on a layer marks the GCell edges it spans along
  that layer's preferred direction. A net adds at most 1 per edge: overlapping
  boxes and repeated net blocks are de-duplicated. Single-GCell boxes (vias and
  pin access) use no edge.
* **Overflow**: the sum of `max(0, demand - capacity)` over all edges. The
  report also gives each layer's totals, the maximum overflow on any edge, and
  the number of overflowing edges.
* **Layer directions** come from the tech LEF: `--tech-lef`, else
  `<platform>/lef/*tech*.lef`, else the platform recorded in `grt.tcl`
  (`ROUTE_SET_RC`). Without a LEF, a declared built-in table is used (NG45
  `metalN`: odd is H; ASAP7 `MN`: odd is V). It was checked against both
  maintained platform LEFs. `direction_source` in the receipt records which
  one was used.

The result is an **estimate**. Capacity is more generous than FastRoute's,
and demand is at most FastRoute's usage (NDR and wide-wire costs are not
charged). So the estimate is expected to be at or below FastRoute's 3D
overflow. That makes it an audit check against under-reporting. It can also be
an independently measured objective for **new** contracts, once their
references are re-qualified with it. It is not a drop-in replacement for the
frozen objective.

## Consistency rule (audit version 1)

With R = reported overflow and E = estimated overflow, the defaults are
`rel_tolerance=0.05`, `abs_tolerance=5`, `reference_rel_tolerance=0.05` and
`reference_abs_tolerance=5`:

1. **bound**: per die and in total, `E <= R * (1 + rel_tolerance) + abs_tolerance`.
2. **positive** (optional, `require_positive_estimate`): `R > 0` requires `E > 0`.
3. **reference** (when a reference `{reported, estimate}` is supplied): a
   claimed improvement `R < R_ref` requires
   `E <= E_ref * (1 + reference_rel_tolerance) + reference_abs_tolerance`.

`consistent` is true only if every check passes. Any missing or symlinked
artifact, parse error, `grt_manifest.json` hash mismatch, or manifest metric
that differs from the log fails closed: `consistent: false` with an entry in
`errors`. Receipts record both values, every check with its limit, the rule,
the input hashes, and `estimator` (`name`, `version`, `sha256` of
`guide_congestion.py`, and the audit module's sha256).

The tolerances are placeholders. Calibrate them on maintained reference runs
before enforcing them; see the validation steps below.

## Opt-in task contract

Add the following block to a **new** task. It changes the task digest, so
frozen tasks are never affected.

```json
"independent_overflow": {
  "version": 1,
  "enforce": true,
  "estimator_sha256": "<sha256 of AgenticBench/grt_bench/guide_congestion.py>",
  "track_model": "average",
  "rule": {"rel_tolerance": 0.05, "abs_tolerance": 5},
  "reference_estimates": {"<case>": {"<seed id>": {"reported": 1234, "estimate": 1100}}}
}
```

`validate_contract` rejects unknown keys. `verify_cell` raises when
`enforce` is true and the audit is inconsistent. With `enforce: false` it only
records the receipt under `independent_overflow`. `grt_bench/contract.py`
(owned elsewhere) currently accepts unknown top-level keys. A future strict
whitelist must allow `independent_overflow` and should call
`independent_overflow.validate_contract` at task-validation time.

Adding `independent_overflow.py` and the `verify.py` hook changes
`grt_bench.common.evaluator_identity()`. A frozen task whose
`evaluator_sha256` was computed before this change therefore fails
`verify_cell`'s evaluator check until it is re-frozen. This is true of any
`grt_bench/*.py` edit.

## Validating on real artifacts

Each die directory under a GRT output (`<run>/grt/{bottom,top}/`) must hold
`grt.guide`, `global_route.odb.def`, `grt.tcl`, `grt.log` and
`grt_manifest.json`. The tech LEF is
`OpenROAD-3D/flow/platforms/<platform>/lef/NangateOpenCellLibrary.tech.lef`.

```sh
cd <repo>
PYTHONPATH=AgenticBench:.:Place-3D python AgenticBench/scripts/audit_grt_overflow.py \
    <run_root_or_grt_dir> [...] \
    --platform-dir OpenROAD-3D/flow/platforms/nangate45_3D \
    --output /tmp/overflow_audit.json --no-fail
```

1. Run on the frozen reference runs of every maintained case. Compare
   `reported_overflow` and `estimated_overflow` for each die, and check
   `grid.def_gcellgrid_agrees` (it should be true; if not, pass
   `--gcell-size` with the DEF step and report the gap).
2. Set the tolerances from the observed spread (E is expected to be at most R).
   Store each reference's `{reported, estimate}` in `reference_estimates`.
3. Run on candidate runs with `--reference <reference receipt>` to apply the
   reference rule.

## Limitations

* The guide and `GCELLGRID` are written by the routing process. A router that
  writes guides that understate its real routing escapes this audit. Checking
  guide connectivity against net pins (and against the detailed route, where
  one exists) is the natural follow-up.
* Obstruction-reduced capacity, NDR and wide-wire demand, via capacity and
  3D layer-assignment details are not modelled. The estimate is a one-sided
  check, not an equality.
* Tolerances need calibration on real references before enforcement.
