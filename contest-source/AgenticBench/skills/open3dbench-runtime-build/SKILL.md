---
name: open3dbench-runtime-build
description: Build or refresh the Open3DBench-V2.0 runtime components required by a benchmark, including DREAMPlace and the pinned OpenROAD-3D/OpenSTA-3D tree, then verify the exact produced binaries. Use after environment qualification or after relevant source changes. Do not generate candidates, prepare benchmark inputs, or interpret QoR.
---

# Open3DBench runtime build

Build only the components affected by the requested task and preserve enough
provenance to prove which binary evaluated the candidate.

## Inputs

- a qualified `repository_context.json`;
- the exact Open3DBench source root and runtime image ID;
- affected repository-relative files and the requested build target;
- an output directory on local `/data` or `/home` storage for build logs and a
  build receipt.

## Procedure

1. Read the current repository `AGENTS.md`, root `Makefile`, and the selected
   adapter stage before choosing commands. Prefer adapter-declared compile
   commands over remembered paths.
2. For maintained source checkouts, use the narrow root targets when present:

   ```bash
   make dreamplace
   make openroad
   make check-openroad
   ```

   Run only the targets required by the changed files. Python, Tcl, YAML, and
   shell-only changes should receive their native syntax or unit checks instead
   of an unrelated full C++ rebuild.
3. For an isolated OpenROAD source candidate, use
   `AgenticBench/scripts/build_openroad_candidate.sh <source-root> <build-dir> <threads>`.
   Keep the build directory outside the immutable benchmark-input bundle.
4. Capture complete stdout/stderr, elapsed time, source commit and dirty diff
   hash, compiler and dependency identity, image ID, build directory, and the
   absolute paths and versions of the resulting binaries.
5. Verify the binary that the next evaluator will actually receive. For local
   OpenROAD selection, require both `build/openroad-3d/cmake/bin/openroad` and
   `build/openroad-3d/cmake/src/sta/sta`, then run the repository's contract
   check before setting `USE_LOCAL_OPENROAD=1`.

## Correctness

Fail on compilation errors, missing runtime libraries, stale or ambiguous
binaries, failed contract checks, or an image/source mismatch. Do not copy a
different server's binary into place without recording and validating its
provenance. A successful build is infrastructure readiness, not evidence of a
QoR improvement.
