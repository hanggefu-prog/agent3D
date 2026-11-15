---
name: operate-open3dbench-global-workspace
description: Execute an open-ended Open3DBench global Intervention inside an isolated writable trial workspace. Use when an agent may modify any subject implementation or configuration while trusted inputs, legality checks, and scoring remain outside its control.
---

# Operate an Open3DBench global workspace

Work only in the current trial. Read `$OPENAGENTIC3D_INSTRUCTION`,
`$OPENAGENTIC3D_USER_COMMAND`, and `$OPENAGENTIC3D_TASK` before planning. The
subject repository is writable; `/task`, `/inputs`, `/tools`, and any protected
workspace overlays are benchmark-owned and read-only.

For each Intervention:

1. Inspect the accepted revision, referenced checkpoints, logs, and current
   metrics. State one falsifiable hypothesis and persist a short plan before
   editing.
2. Modify whatever subject source or configuration is necessary. The target
   stage is a diagnosis and replay hint, not a file allowlist.
3. Use `open3dbench-environment-check` and `open3dbench-runtime-build` only for
   affected components. Keep build products under `/build` and runs under
   `$OPENAGENTIC3D_RUNS_ROOT`.
4. Use `open3dbench-stage-replay` for quick and suffix evaluations. Treat
   candidate-written metrics as diagnostics until the trusted evaluator
   accepts them.
5. Use `open3dbench-artifact-handoff` to submit the final repository state and
   required canonical artifacts. Do not edit, replace, or bypass benchmark
   legality checks, reference inputs, verifier definitions, or scoring code.

When an experiment fails, preserve the exact command, source revision, build
receipt, run directory, failing stage, and relevant log tail. Do not report a
QoR improvement from incomplete or incomparable runs.
