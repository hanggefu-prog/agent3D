---
name: open3dbench-environment-check
description: Inspect and qualify an OpenAgentic3DBench plus Open3DBench-V2.0 execution environment before an experiment. Use to identify the adapter, source revision, dirty state, runtime image, GPU visibility, required EDA tools, and stage entry points. This is read-only qualification; do not build tools, modify source, prepare inputs, or run a QoR evaluation.
---

# Open3DBench environment check

Qualify the exact environment that will execute a benchmark. Keep the
OpenAgentic3DBench control plane and the external Open3DBench source checkout
as separate paths.

## Procedure

1. Resolve and record the framework root, Open3DBench root, run-data root,
   requested integration type, PDK, design, stage, GPU, and runtime image.
2. Generate version-adaptive repository context:

   ```bash
   openagentic3dbench-inspect --repo <open3dbench-root> --output <report-dir>/repository_context.json
   ```

3. Read the selected stage from `repository_context.json`. Require its source,
   configuration, command, input, metric, and correctness-gate paths to be
   present. Do not guess a legacy path when the adapter reports it missing.
4. Record the source commit, branch, and dirty state. A dirty checkout is not
   automatically invalid, but its status must be explicit and identical across
   reference and candidate runs unless the candidate harness owns the diff.
5. Verify that the selected runtime image exists and record its immutable image
   ID. In that image, check only the tools required by the requested stage.
   Examples include Python and DREAMPlace for placement; OpenROAD/OpenSTA for
   backend stages; Mt-KaHyPar or hMETIS for the selected partition method.
6. When GPU execution is requested, verify device visibility and the imports
   required by the actual method. Do not infer CUDA usability from host
   `nvidia-smi` alone.
7. Write an environment report containing the resolved paths, revisions,
   binary versions, image ID, device information, checks performed, failures,
   and the commands that a later build or evaluator Skill should use.

## Boundary

This Skill never installs dependencies, compiles code, edits configuration,
creates a benchmark-input bundle, launches an optimization candidate, or
repairs a failing environment. Hand an actionable failure report to
`open3dbench-runtime-build` or to the user instead of silently substituting a
different tool, image, method, PDK, or device.
