#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.config/agentic3d/env.sh"
mkdir -p "$OPEN3DBENCH_RESULTS_ROOT/setup"
log="$OPEN3DBENCH_RESULTS_ROOT/setup/$(hostname)-open3dbench-image-build.log"

exec > >(tee "$log") 2>&1
printf 'build_started=%s\n' "$(date --iso-8601=seconds)"
printf 'repository=%s\n' "$OPEN3DBENCH_ROOT"
printf 'revision=%s\n' "$(git -C "$OPEN3DBENCH_ROOT" rev-parse HEAD)"

if [[ -n "${CUDA_IMAGE:-}" ]]; then
  host_args=()
  for hostname in \
    mirror.nju.edu.cn github.com release-assets.githubusercontent.com \
    objects.githubusercontent.com download.pytorch.org data.pyg.org \
    pypi.org files.pythonhosted.org ppa.launchpad.net \
    ppa.launchpadcontent.net launchpad.net; do
    address="$(getent ahostsv4 "$hostname" | awk 'NR == 1 { print $1 }')"
    if [[ -n "$address" ]]; then
      host_args+=(--add-host "$hostname:$address")
    fi
  done
  docker build \
    --network host \
    "${host_args[@]}" \
    --build-arg "CUDA_IMAGE=$CUDA_IMAGE" \
    --build-arg "OPENROAD_JOBS=${JOBS:-12}" \
    --tag "$OPEN3DBENCH_PLACE_IMAGE" \
    --file "$OPEN3DBENCH_ROOT/docker/Dockerfile" \
    "$OPEN3DBENCH_ROOT"
else
  make -C "$OPEN3DBENCH_ROOT" image IMAGE="$OPEN3DBENCH_PLACE_IMAGE" JOBS="${JOBS:-12}"
fi
make -C "$OPEN3DBENCH_ROOT" check-image IMAGE="$OPEN3DBENCH_PLACE_IMAGE"

printf 'build_completed=%s\n' "$(date --iso-8601=seconds)"
printf 'build_log=%s\n' "$log"
