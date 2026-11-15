# Server deployment

Every development node uses the same monorepo layout:

```text
/data/gcr/Agentic-Open3DBench/  # clean Git checkout
/data/gcr/agentic3d-runs/       # generated inputs, references, logs and state
/data/gcr/agentic3d-cache/      # optional caches and build downloads
```

`Agentic-Open3DBench` already contains Open3DBench and `AgenticBench/`.
The former split layout (`framework/` plus `Open3DBench-V2.0/`) is obsolete.
Existing `/data/gcr/agentic3d-dev` trees are historical experiment archives;
do not use them for new source changes.

After placing a clean checkout at the canonical path, install the environment
and command wrappers:

```bash
export OPENAGENTIC3DBENCH_DEV_ROOT=/data/gcr
export EXPECTED_AGENTIC_OPEN3DBENCH_REV=<git-commit>
bash AgenticBench/deploy/server/install-server.sh
source ~/.config/agentic3d/env.sh
```

The environment resolves `OPEN3DBENCH_ROOT` to the repository root and
`AGENTIC3D_ROOT` to its `AgenticBench/` directory. Agent state and candidate
worktrees live under `/data/gcr/agentic3d-runs`, keeping the source checkout
clean. Normal runs use `AGENTIC3D_SCOPE_MODE=enforced`. The installer verifies
and extracts the matching vendored PyYAML wheel into the isolated virtual
environment, so deployment does not depend on PyPI or `ensurepip`.

All nodes use `shiyunqi/open3dbench-v2.0:release` for both placement and
backend evaluation. `OPEN3DBENCH_RELEASE_IMAGE` is the single override used to
qualify a future release; the derived `OPEN3DBENCH_PLACE_IMAGE` and
`OPEN3DBENCH_EVAL_IMAGE` must not diverge across nodes.

Check source-only integrity without running EDA tools:

```bash
scripts/check_server_checkout.sh "$PWD" "$EXPECTED_AGENTIC_OPEN3DBENCH_REV"
```

Run the complete node verification after images and DREAMPlace are available:

```bash
verify-agentic3d-server
```

See [`docs/SERVER_CODE_LAYOUT.md`](../../../docs/SERVER_CODE_LAYOUT.md) for
the Git workflow and the boundary between canonical source and legacy runs.
