#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.config/agentic3d/env.sh"
mirror_ip="$(getent ahostsv4 mirror.nju.edu.cn | awk 'NR == 1 { print $1 }')"
[[ -n "$mirror_ip" ]] || { echo "Cannot resolve mirror.nju.edu.cn on host" >&2; exit 2; }
docker build \
  --no-cache \
  --network host \
  --add-host "mirror.nju.edu.cn:$mirror_ip" \
  --build-arg CUDA_DONOR_IMAGE=shiyunqi/open3dbench:place \
  --build-arg RUNTIME_BASE_IMAGE=shiyunqi/open3dbench:eval \
  --tag "$OPEN3DBENCH_PLACE_IMAGE" \
  --file "$AGENTIC3D_ROOT/deploy/server/Dockerfile.ta-cached-base" \
  "$AGENTIC3D_ROOT/deploy/server"
