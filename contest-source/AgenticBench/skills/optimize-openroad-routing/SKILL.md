---
name: optimize-openroad-routing
description: Optimize and validate OpenROAD-3D global and detailed routing, including MoL die-by-die and HBT flows. Use for adapter-selected routing stages, routing Tcl or C++ changes, congestion and wirelength experiments, guide merging, cross-die connectivity, DRC, and final route correctness.
---

# Optimize OpenROAD routing

## Version-adaptive repository contract

Read `repository_context` from `handoff.md`; `stages.routing` defines the
editable GRT/DRT/3D-routing sources and configs, compilation/evaluation command
hints, logs/results roots, metrics, and correctness gates. It supports unified
`Place-3D` releases and legacy repositories without changing this Skill. Read
the OpenROAD `AGENTS.md` that governs any C++ path before edits.

Read `scope_mode` and `scope_contract` from the handoff. In `enforced` mode,
the signed contract is a hard limit: modify only its listed paths and obey its
file types, operation permissions, and line/file budgets. In `direct` mode,
the framework does not restrict repository paths or integrate a candidate
worktree; the stage boundaries in this Skill remain behavioral guidance. In
either mode, build the affected OpenROAD binary and run an isolated
`FLOW_VARIANT` after resolving all `<placeholders>`. Read measured `5_*`/`6_*`
logs, reports, guides, ODB/DEF and metric artifacts from adapter-discovered
locations. Never accept missing-net, layer-leakage, connectivity, DRC, or
readback failures as a wirelength improvement.

Resolve validators and parsers relative to this `SKILL.md`; the orchestrator
copies the entire Skill into the workflow so no target-repository
`Agentic3D/` directory is required.

## Task Agent protocol

Run this Skill inside one stateful Routing Task Agent so that route analysis and
code or Tcl changes share the same Codex thread. Before every round of edits,
persist the requested `plan.md` with one routing hypothesis, affected
components and files, compile and evaluation steps, correctness gates, and a
rollback condition. Carry forward prior wirelength, via, DRC, and connectivity
evidence. Claim stage completion only when the handoff success criteria and
every validation gate pass.

List every intended repository-relative edit in `files_to_modify`. In enforced
mode, this list must be inside the signed scope and must match the eventual
diff. Do not run Git lifecycle commands such as commit, reset, clean, checkout,
merge, or rebase; the framework owns candidate checkpoints, rollback, and
integration.

Start from the accepted adapter-selected CTS artifact—normally `4_cts.odb`
plus its compatible SDC—and a new `FLOW_VARIANT`. Keep the accepted
`3_place.odb` only as provenance. Never overwrite the source variant.

## Compile and run

- Run the adapter-listed compile commands for the files actually changed. For
  Python/Tcl/shell changes, compile or syntax-check only adapter-selected paths.
- For OpenROAD GRT/DRT C++ changes, use the `open3dbench-docker-eval` incremental `build-codex` flow and verify the mounted binary path.
- Use `stages.routing.evaluation_commands` as the default entry point. Run
  `do-5_1_grt`/`do-5_2_route` only when those targets exist in the selected
  revision.
- Enter the HBT/die-by-die branch only when repository context, Global Plan,
  and accepted CTS artifacts identify that regime. In that branch, use the
  repository helper and adapter-selected pre-HBT CTS artifact; do not assume a
  fixed filename. A split-routing run that unexpectedly inserts zero HBT
  terminals is invalid.

## Allowed action space

Modify only adapter-listed routing Tcl/Python/config paths and corresponding
OpenROAD GRT/DRT C++. Do not change placement, CTS, layout, or partition inputs
in the same routing iteration.

## Metrics

Collect routed wirelength, via count, congestion or overflow, antenna count,
DRC count, disconnected nets, guide connected components, cross-die or HBT
counts, WNS, TNS, runtime, and memory.

```bash
python3 <skill-dir>/scripts/parse_openroad_routing.py \
  5_1_grt.log 5_2_route.log 6_report.log \
  --require-file 5_route.odb --output routing_metrics.json
```

## Correctness gates

Require valid merged guides, no missing or extra net guides, permitted layer
ranges on every die, connected cross-die or HBT paths, zero unexpected
disconnected nets, parseable routed ODB and DEF artifacts, and DRC no worse
than the declared budget. For die-by-die routing, also run the repository's
guide-layer, connectivity, cross-die, and DRT geometry validators.

Reject lower wirelength if it results from omitted nets, invalid guides, layer
leakage, or a failed DRT run. Return final PPA and correctness evidence to the
Global Analyzer.
