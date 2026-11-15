# Workspace evaluator / maintained flow interface alignment

Historical contract: this page records the 2026-09-24 adapter repair, including
its two-OPT objective. It is not the current four-case reference specification.
For the explicit new one-OPT and standalone-GRT boundaries, read
[REFERENCE_20261004.md](REFERENCE_20261004.md) and the
[current qualification status](../../docs/REFERENCE_AND_INTERFACE_STATUS.md).

## Contract (2026-09-24)

This repair changes adapters, not agent search or physical algorithms. The
co-design tasks retain their objectives, seeds, resources, exactly two
coordinated OPT passes and final-evaluation budgets. Core Open3DBench OPT still
defaults to one pass. Strict SDC, physical legality and artifact checks remain
authoritative. Historical failed runs and sealed references must not be edited.

| Boundary | Producer / consumer contract | Alignment and regression coverage |
|---|---|---|
| Input preparation | Native `plan` to frozen input bundle | Include real timing-extract and split-routing producers; do not synthesize missing manifests from historical directories. |
| Configuration | Workspace YAML plus method JSON to both execution phases | Resolve repository- or Place-3D-relative JSON once; bind both phases to the same read-only snapshot bytes and record SHA-256. Reject native top-level fields the runner would ignore; arbitrary method knobs remain in editable JSON. |
| Frontend handoff | Fixed netlist, clocks, I/O, partition and legal placement | Preserve and check immutable input/floorplan bytes and partition assignment; candidate frontend code cannot become the backend evaluator. |
| OPT pass boundary | Completed two-die DEF/ODB/Verilog/SDC to next pass | Advance the complete pair atomically. Keep the original base SDC for deriving each round's budgets. A mixed old DEF/new ODB still fails. |
| Historical placement promotion | Frozen source dialect to trusted OPT patch | Select exactly one applicable trusted patch before EDA. Unknown or ambiguous source interfaces fail explicitly. Record patch path/hash; do not relabel historical sources. |
| CTS to thermal | Native CTS identity and ready checkpoint | Accept explicit historical/current identities, require matching checkpoint metadata, and forbid mixed implementation identities in a comparison. |
| Power / HotSpot | Exact CTS checkpoint to 200 physical silicon tiles | Bind snapshot and power provenance to the same checkpoint; verify grid, ambient, positive conserved power and physical-tile mean. |
| Result to agent | Independent verifier to feedback/history | Keep score formulas, failure semantics and final budget unchanged. No model calls are required for interface qualification. |

The post-CTS `thermal_snapshot` module is retained AgenticBench infrastructure,
not the routed `90_report`/`95_thermal` flow. Do not replace it with routed
thermal or a pure upstream checkout lacking that module. Timing and thermal
references must be newly qualified at their actual task boundary.

## Reference-only physical qualification

Run the following from the same clean pinned source snapshot supplied to the
evaluator. `BATCH` must contain both the frozen input bundle and new output;
paths may refer to separate subdirectories of a release's `runs/` root.
The source, runtime image and input identities are recorded automatically.

```sh
python AgenticBench/scripts/validate_workspace_flow.py \
  --source "$SOURCE" --batch-root "$BATCH" \
  --benchmark-input "$FROZEN_INPUT" --output "$NEW_OUTPUT" \
  --case bp_fe --objective thermal --image "$IMAGE_SHA256" --gpu 2 \
  --cpus 16 --memory-gb 48 --min-gpu-free-mb 2048
```

Use `--objective timing` for the exact post-two-OPT stopping point. The command
uses the production workspace evaluator, reuses only validated input/floorplan,
does not call Codex, consumes no candidate budget, and refuses an existing output
directory. See `qualification.json`, `summary.json` and `reference/` for the
attempt, native logs, stage manifests, metrics and verifier result.

Unit/contract tests are not physical qualification. Report real EDA outcomes
separately, including failed gates and partial results. Never claim a reference
is usable until its independent verifier succeeds.
