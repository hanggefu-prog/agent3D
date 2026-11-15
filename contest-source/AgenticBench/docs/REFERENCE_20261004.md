# Reference family 20261004

This is an explicit interface family, not a relabeling of the 20260923
reference store. The historical image allowlist and old contract remain
unchanged. Unknown family identifiers fail closed.

## Maintained source and scope

The root `AgenticBench` directory is a recovered, separately maintained
framework source tree. `RECOVERY.json` records the original snapshot and every
recovered source file hash. Its broken historical worktree pointer is not a
new Git identity. Framework edits are not part of the physical flow commit.

This family pins physical flow `4d882f0b5b9aa98ba2742510dc1b011e43b31657`,
OpenROAD `56ea5f41b1352b937cae368832f156a445097102`, OpenSTA
`5d8d515edfa5a22165faea5897539da1888fe68d`, and canonical image
`sha256:1b5567c86d48effa03b0543089e01fec37786e4f249309b7da627b477d7639e5`.
A different Docker configuration image ID additionally requires an independently
collected, hash-bound equivalence proof for both installed binaries and layers.

The four explicit input profiles are `{bp_fe,bp,ariane133,swerv_wrapper}_lol_ng45_reference_20261004`.
They retain the fixed physical settings and two trusted framework-owned
algorithm JSON files in `local_bench/reference_configs`. Candidate paths do not
gain permission to escape their workspace.

## Stage and objective boundaries

- Input import copies only verified `00_input`, `10_floorplan`, and
  `20_partition`. The new ledger records imported stages and zero executed
  stages. `20_partition/timing_input` is native partition-owned data; there is
  no `15_timing_extract` stage in this family.
- Local partition and the two placement tasks retain their existing algorithm
  scope and scoring. Their profile selection is explicit. An imported physical
  input is not a local-hook scoring authority.
- The three supported GRT source interfaces use the standalone `global-route`
  command, `80_route/global_route_checkpoint.json`, and `grt_metrics.json`.
  They never invoke the combined routing stage to obtain GRT measurements.
  Native hook-use evidence remains mandatory; net ordering remains excluded.
- Global profile `tp_gnn_placement_one_pass_20261004` evaluates exactly one OPT
  pass using pre-CTS STA. A producer may be qualified through standalone GRT,
  but its global timing score is read from the independently checked OPT
  checkpoint. GRT and timing authority endpoints cannot be interchanged.
- GRT completion is not DRT, DRC-clean routing, or signoff. GP metrics are not
  placement legalization proof. No new HBT search algorithm is introduced.

## Read-only input adoption

After an existing native controller has completed successfully, independently
collect its controller plan, dispatch receipt, actual terminal Docker state,
installed runtime proof, and three clean source revisions. Their four leaf
paths and hashes are bound by a `latest-reference-execution-proof-v1` file.
The importer will reject missing, failed, running, or changed execution evidence.

```bash
python AgenticBench/scripts/adopt_reference_latest.py \
  --reference-run /absolute/completed/case/run --case bp_fe \
  --source /absolute/clean/physical/source --output /absolute/new/input-bundle \
  --execution-proof /absolute/execution-proof.json \
  --execution-proof-sha256 VERIFIED_SHA256
```

For an independently equivalent noncanonical Docker configuration ID, add
`--runtime-proof` and `--runtime-proof-sha256`. The importer validates the
scientific config, all eight completed stage manifests and registered
artifacts, GRT checkpoint/metrics/raw logs, and immutable source identity before
publishing the new bundle. It does not execute EDA or a model, or publish a
scoring authority. It refuses an existing output directory; it never retries.

Global qualification/publication is implemented by
`scripts/workspace_latest_reference.py`. Local reference qualification remains
separate and requires the real local task boundary. Both use the existing
reference store rather than silently falling back to a fresh run.

## Validation status

The four real producers are independently qualified through standalone GRT,
and four case-matched authorities are published for
`lol_partition_placement_timing_20261004_10`. See the
[current status report](../../docs/REFERENCE_AND_INTERFACE_STATUS.md) and its
linked reference index for paths, hashes and endpoint limitations.
Local scoring authorities are still null. The separate post-CTS STA/thermal
diagnostics are not a latest-family thermal search qualification.

Offline tests exercise the actual current-flow CLI planning/materialization
boundary for partition, both placement adapters, and all three GRT adapters;
they also exercise importer-to-global-prefix-to-authority using tiny synthetic
native artifacts. These tests are not native physical qualification. The completed
four-case producer qualification is supported by real terminal artifacts and
independent execution proofs, not inferred from these tests.

Some legacy fixture suites assume the old physical source contains
`tp_gnn_reference_20260923.json`; that file is deliberately absent from the new
physical source. Legacy net-order hook tests also target the older combined
router. Neither assumption is repaired by broadening the new family or
relabeling historical evidence.
