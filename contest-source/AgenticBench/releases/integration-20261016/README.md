# Isolated branch integration and test record

The user authorized an integrated version while preserving the current branch.
`codex/local-task-integration-20261016` starts from publication commit `15e08ded`
and merges both collaborator branches with real merge commits. The original
working branch, its uncommitted files, the publication branch and GitHub main
are unchanged. The active campaign keeps its original source and references.

## Integrated behavior

- Merge `1dbac0cf` incorporates the clock branch while retaining the richer
  physical clock classifier already present in main. It adds the real bottom
  primary-clock source regression from the collaborator branch.
- Merge `4dc038de` incorporates explicit `pdn.algorithm: none` passthrough,
  checkpoint chaining, truthful HBT PDN metadata and resume invalidation.
- Fix `cb445971` completes the no-grid HBT/OPT handoff. Native testing exposed
  missing VDD/VSS nets in both PDKs. HBT now creates only logical supply nets
  and connects PG pins for this explicit mode; the typed handoff preserves
  the no-grid state through OPT and CTS. Normal PDN inputs still require
  physical special wires. Missing or malformed mode declarations cannot opt
  a legacy checkpoint out of that check.
- The missing-STA-API zero/empty-cutpoint fallback and blanket STA-0471/0473
  allowances are resolved out. Native STA evidence, physical clock ownership
  and audited SDC projection remain strict.
- OpenROAD remains pinned to `e8c2b9f802b28bbf4a07548ee41f7becc758675a`, including
  the RC pin-segment fix; OpenSTA remains `5d8d515edfa5a22165faea5897539da1888fe68d`.
  There is no new compiled-runtime change in this integration.

The historical contest branch remains separate. The two older physical branches
were already ancestors of main and needed no additional merge.

## Regression results

| Test group | Passed | Skipped |
| --- | ---: | ---: |
| Current local-task, engineering-task and post-GRT interfaces | 1064 | 10 |
| Physical flow and orchestrator | 520 | 1 |
| Place-3D in the installed DREAMPlace image | 198 | 0 |

There were also 156 and 210 passing subtests in the first two groups. The one
physical-flow skip needs optional sealed CTS evidence. The other skips concern
optional package-build, OpenEvolve and host ML dependencies. The initial attempt
to collect the entire Place-3D suite on macOS lacked DREAMPlace/PyTorch; the
maintained suite was instead executed successfully in the installed server image.
The full historical global suite is outside this agreed compatibility scope.

`validation.json`, `tests/`, and `tested-source.json` preserve the evidence.
The tested implementation is `cb445971`; subsequent changes contain only this
record and documentation. All 450 relevant source/task/platform files matched
the server snapshot. The image and installed OpenROAD checksum are recorded.

## Native boundary tests

Four independent tests reused the authenticated input/floorplan/partition/
placement prefix and executed PDN, HBT, OPT and CTS. The original inputs were
read-only. All four validated their new artifacts and checkpoint chain, retained
the exact upstream hashes, and then reused all four new stages without changing
any stage-manifest hash. The explicit PDN state persisted through CTS.

| Input | PDN mode | PDN (s) | HBT (s) | OPT (s) | CTS (s) | Container total (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bp_fe-asap7 | none | 0.22 | 26.90 | 235.40 | 95.37 | 361.73 |
| bp_fe-asap7 | openroad | 62.19 | 27.69 | 236.24 | 96.66 | 426.79 |
| bp_fe-ng45 | none | 0.26 | 26.58 | 205.25 | 69.10 | 305.46 |
| bp_fe-ng45 | openroad | 13.85 | 26.81 | 206.07 | 69.29 | 319.88 |

Container totals include validation and resume; they exclude queue waiting.
These are fixed-placement diagnostic tests, not full optimization matrices,
post-GRT metrics or Pareto candidates. They used four CPU threads and 24 GiB per
job, up to four workers, and zero GPUs. All leases completed and all test
containers terminated. No model call or campaign evaluation budget was spent.

The first uncorrected merge passed both normal-PDN tests and failed both no-grid
HBT tests with a missing VDD connection. Those attempts remain explicit in
`native-evidence.json`. After the fix, all four fresh tests passed. For both PDKs,
the normal-PDN HBT assignments, OPT global metrics and CTS assembly metrics were
exactly equal before and after the fix.

The exact native drivers are retained beside this record. They take explicit
server data/queue paths; the native driver uses separate source and output names
for each immutable attempt. Detailed output lives under the configured data
root's `integration-native-v2/<case>-<mode>/`, and Place-3D logs under
`integration-place/`. This record contains compact evidence rather than physical
databases. `native-evidence.json` binds the original input jobs and hashes.

## Release boundary

Eight maintained local tasks, the three engineering tasks and current post-GRT
interfaces remain the compatibility scope. These passing integration tests do
not replace final native, case-matched qualification of every task endpoint.
Main and the live campaign have not been activated from this branch. Existing
reference receipts remain bound to their original source/runtime.

Use `pdn.algorithm: openroad` for GRT/DRT and downstream reports. The explicit
no-grid variant is supported through CTS and does not claim physical power-grid
connectivity or signoff.
