---
name: open3dbench-benchmark-input
description: Prepare, validate, copy, or audit a frozen OpenAgentic3DBench benchmark-input bundle backed by Open3DBench-V2.0 stage artifacts. Use when establishing the common reference input for a task or moving it between servers. Do not evolve candidates, alter released benchmarks, or treat an unvalidated run directory as a benchmark input.
---

# Open3DBench benchmark input

Create one immutable, case-specific input bundle that every compared method
can consume. Read `docs/benchmark_inputs.md` before preparing a new bundle.

## Prepare

Inspect the task and its benchmark-input profile first. Then run:

```bash
openagentic3dbench-task prepare-input <profile> \
  --open3dbench-root <open3dbench-root> \
  --output-dir <data-root>/<profile>
```

Supply the profile's required Python and DREAMPlace paths when they are not
provided by the selected runtime. `--seed-run` may reuse only the stages that
the preparation contract explicitly validates; downstream stages are
regenerated. Use `--resume` only for the same output and unchanged contract.

## Validate and transfer

Before every experiment, and again after copying a bundle between machines,
run:

```bash
openagentic3dbench-task validate-input <bundle> --task <task-id>
```

Require matching design, integration, PDK, profile and task compatibility,
complete stage manifests, and exact artifact hashes. Copy the self-contained
bundle as data; never rewrite paths inside its checked artifacts. Keep active
workspaces and recurring logs on local `/data` or `/home`, not NAS.

## Result contract

The bundle contains `benchmark_input.json`, preparation provenance, a normalized
configuration, and the frozen stage directories declared by the profile. Its
identity is the manifest plus artifact hashes, not its absolute filesystem
path. Report the bundle path, profile ID, case, upstream stages, source/runtime
provenance, validation status, and any incompatibility.

Do not edit a validated bundle in place. A changed design, PDK, partition,
floorplan, seed, source contract, or evaluator requirement creates a new bundle
and a new experiment identity.
