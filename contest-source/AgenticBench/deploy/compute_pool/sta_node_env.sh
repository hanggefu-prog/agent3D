#!/usr/bin/env bash
# Installed under NODE_ROOT/tools-sta. No credentials live in this environment.
task_node_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$task_node_root" in /data/gcr/agentic3d-compute-20260908|/data/agentic3d/agentic3d-compute-20260908) ;; *) return 2 ;; esac
export OPENAGENTIC3D_NODE_ROOT="$task_node_root"
export OPENAGENTIC3D_STA_DATA_ROOT="$task_node_root/sta-data"
export OPENAGENTIC3D_FRAMEWORK_ROOT="$task_node_root/framework-sta"
export OPEN3DBENCH_ROOT=/data/gcr/agentic3d-dev/open3dbench-source-sta-codesign-20260902-v1
export OPENAGENTIC3D_RUNTIME_ROOT="$task_node_root/sta-data/agentic3d-runs/runtime"
export OPENAGENTIC3D_DREAMPLACE_INSTALL="$OPENAGENTIC3D_RUNTIME_ROOT/dreamplace-overlay"
export OPEN3D_STA_OPENROAD_EXE="$task_node_root/sta-data/agentic3d-dev/docker-sta-20260902/docker/runtime-refresh/openroad"
export OPENAGENTIC3D_EVAL_DOCKER_IMAGE=agentic3d/compute-pool-runtime:20260908
export OPENAGENTIC3D_EVAL_GPU_LOCK_DIR="$task_node_root/gpu-locks"
