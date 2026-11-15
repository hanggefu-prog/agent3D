---
name: optimize-openroad-place
description: Optimize and validate OpenROAD-3D placement, resizing, detailed placement, timing optimization, and CTS after a validated adapter-selected layout handoff.
---

# Optimize OpenROAD Place Opt and CTS

## Version-adaptive repository contract

Read `repository_context` from `handoff.md` and use `stages.place_opt` for all
paths and commands. This stage intentionally includes OpenROAD global
placement, resizing, detailed placement, timing optimization, and CTS so it
matches the framework's third stage. Typical unified and legacy adapters expose
`openroad/src/gpl`, `dpl`, `rsz`, `cts`, flow Tcl, design configuration,
`3_*`/`4_*` logs, and `3_place`/`4_cts` results. The adapter, not this prose,
is authoritative after a repository update. Read the nested OpenROAD
`AGENTS.md` before changing C++.

The Task Agent may modify adapter-listed C++/Tcl/config files and directly run
listed compile/evaluation commands after resolving every `<placeholder>`.
Always use an isolated `FLOW_VARIANT`; inspect both placement and CTS
logs/results and preserve validated Place-3D input artifacts.

Resolve any Skill helper under the directory containing this `SKILL.md`; never
assume the target repository contains an `Agentic3D/` source checkout.

## Task Agent protocol

Run this Skill inside one stateful Place Opt Task Agent; do not split planning
from OpenROAD or Tcl implementation. Before every round of edits, inspect prior
timing, congestion, legalization, and CTS evidence, then persist the requested
`plan.md`. State one hypothesis, affected components and files, compile and
evaluation steps, correctness gates, and a rollback condition. Resume the same
Codex thread in later rounds. Claim stage completion only when the handoff
success criteria and every validation gate pass.

Consume one validated adapter-selected placement DEF. Keep its partition and
macro placement frozen unless Global Analyzer explicitly returns to layout.

## Compile and stage targets

- For Tcl/config changes, run `bash -n` where applicable and start from an isolated `FLOW_VARIANT`.
- For OpenROAD C++ changes, use the `open3dbench-docker-eval` build helper with incremental `build-codex`; do not use `--clean` unless the cache is corrupt.
- Use adapter-listed targets. Typical flows run the complete `do-place` chain followed by `do-cts`; verify actual targets in the checked-out Makefile.
- Record `OPENROAD_EXE`. A patched C++ experiment is invalid if the log uses `/usr/bin/openroad` instead of the mounted build.

## Allowed action space

Modify only adapter-listed OpenROAD placement, resizer, detailed-placement, and
CTS sources or configurations. These may include C++, global-place, resize,
detail-place, and CTS Tcl files, plus variables such as `GPL_TIMING_DRIVEN`,
`GPL_ROUTABILITY_DRIVEN`, `ENABLE_DPO`, and `DPO_MAX_DISPLACEMENT`. Do not
modify the input DEF or routing settings in the same experiment.

## Metrics

Collect HPWL or wirelength, density overflow or utilization, WNS, TNS, design
area, inserted and removed buffers, displacement, and runtime after each
adapter-selected placement checkpoint, commonly 3_3, 3_4, and 3_5.

```bash
python3 <skill-dir>/scripts/parse_openroad_place.py \
  3_3_place_gp.log 3_4_place_resized.log 3_5_place_dp.log \
  --require-file 3_place.odb --output place_metrics.json
```

Then run the adapter-selected CTS chain and collect clock count, sinks,
inserted clock buffers, skew, latency, WNS/TNS, clock connectivity, runtime,
and the final `4_cts.odb`/SDC. The parser above validates placement only; it
does not replace CTS evidence. If the current adapter exposes a CTS metrics
report or validator, run it and record the path.

## Correctness gates

Require all placement and CTS ODBs/SDCs, zero fatal/error markers, unchanged
macro coordinates unless authorized, legal detailed placement, no missing
instances/nets/clocks/sinks, timing units recorded, connected clock trees,
finite skew/latency, and successful OpenROAD database readback. Reject timing
improvements caused by dropped clocks, nets, instances, or sinks.

Hand the accepted `4_cts.odb`, CTS-compatible SDC, placement and CTS metric
JSON, logs, exact `OPENROAD_EXE`, and `FLOW_VARIANT` to the Routing Agent.
Retain `3_place.odb` as provenance, not as the normal Routing Agent startpoint.
