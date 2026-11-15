#!/usr/bin/env bash
# One constrained evaluation slot. No API credentials are required or mounted.
set -euo pipefail
node_root=${1:?node root required}
node=${2:?node name required}
action=${3:?qualify or evaluate required}
shift 3
case "$node" in ta2|ta3|amax3) ;; *) exit 2 ;; esac
case "$node_root" in /data/gcr/agentic3d-compute-20260908|/data/agentic3d/agentic3d-compute-20260908) ;; *) exit 2 ;; esac
case "$action" in qualify|evaluate) ;; *) exit 2 ;; esac
exec 8>"$node_root/worker.lock"
flock -n 8 || { echo 'node evaluation slot is already in use'; exit 3; }
available=$(df -Pk "$node_root" | awk 'NR==2 {print $4}')
(( available > 20 * 1024 * 1024 )) || { echo 'insufficient scratch space'; exit 4; }
memory=64g
[[ "$node" != amax3 ]] || memory=40g
exec docker run --rm --init --name "agentic3d-pool-$node-$action" \
  --cpus 32 --memory "$memory" --memory-swap "$memory" \
  --network none \
  -v "$node_root/data:/data/gcr/agentic3d-dev:ro" \
  -v "$node_root/framework:$node_root/framework:ro" \
  -v "$node_root/tools:$node_root/tools:ro" \
  -v "$node_root/inputs:$node_root/inputs:ro" \
  -v "$node_root/references:$node_root/references:ro" \
  -v "$node_root/candidates:$node_root/candidates:ro" \
  -v "$node_root/results:$node_root/results" \
  -v "$node_root/scratch:$node_root/scratch" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e "PYTHONPATH=$node_root/framework" \
  -e OPEN3DBENCH_PYTHON=/opt/conda/bin/python3 \
  -e "OPEN3D_LOCAL_OPENROAD_BUILDER=$node_root/framework/scripts/build_openroad_candidate.sh" \
  -e OPEN3D_LOCAL_OPENROAD_BUILD_THREADS=8 \
  -e OPEN3D_LINK_FROZEN_UPSTREAM=1 \
  -e "OPEN3D_EVALUATION_SCRATCH_ROOT=$node_root/scratch" \
  -e "CCACHE_DIR=$node_root/scratch/ccache" -e CCACHE_MAXSIZE=8G \
  agentic3d/compute-pool-runtime:20260908 \
  /opt/conda/bin/python3 "$node_root/tools/worker.py" \
  --root "$node_root" --node "$node" "$action" "$@"
