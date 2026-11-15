#!/usr/bin/env bash
set -euo pipefail
node_root=${1:?node root required}
case "$node_root" in /data/gcr/agentic3d-compute-20260908|/data/agentic3d/agentic3d-compute-20260908) ;; *) exit 2 ;; esac
pool=/nas1/gcr/agentic3d-dev/compute-pool-20260908
bundle=$pool/bundles/sta-dependencies-v1
expected=e4910ddc820f3702f2d82e653508ac2e78e13eee49c212094a5bcf934ce9d503
image=agentic3d/compute-pool-runtime:20260908
[[ -f "$node_root/runtime.ready" && -f "$bundle.ready" ]]
exec 9>"$node_root/sta-dependencies.lock"
flock -n 9 || exit 3
[[ ! -f "$node_root/sta-dependencies.ready" ]] || exit 0
[[ ! -e "$node_root/sta-data" ]] || { echo 'partial STA dependency extraction requires inspection'; exit 4; }
available=$(df -Pk "$node_root" | awk 'NR==2 {print $4}')
(( available > 25 * 1024 * 1024 )) || { echo 'need 25 GiB node-local free space'; exit 5; }
printf '%s  %s\n' "$expected" "$bundle.tar.zst" | sha256sum -c -
mkdir "$node_root/sta-data"
echo "extracting_sta_dependencies $(date -Is)"
zstd -dc "$bundle.tar.zst" | tar --no-same-owner -xf - -C "$node_root/sta-data"
cp "$bundle.exact-files.sha256" "$node_root/sta-exact-files.sha256"
docker run --rm --network none \
  -v "$node_root:$node_root" \
  -v "$node_root/sta-data/agentic3d-dev:/data/gcr/agentic3d-dev:ro" \
  -v "$node_root/sta-data/agentic3d-runs:/data/gcr/agentic3d-runs:ro" \
  "$image" bash -ec '
    sha256sum -c "$1/sta-exact-files.sha256"
    git -c safe.directory="*" -C /data/gcr/agentic3d-dev/open3dbench-source-sta-codesign-20260902-v1 fsck --connectivity-only
    /data/gcr/agentic3d-dev/docker-sta-20260902/docker/runtime-refresh/openroad -no_init -exit /data/gcr/agentic3d-runs/runtime/check_opensta3d_command.tcl
  ' bash "$node_root" > "$node_root/sta-dependencies-preflight.log" 2>&1
grep -q 'get_parasitic_cutpoint_timing=get_parasitic_cutpoint_timing' "$node_root/sta-dependencies-preflight.log"
date -Is > "$node_root/sta-dependencies.ready"
echo "sta_dependencies_ready $(date -Is)"
# Further gates: framework, frozen-input preparation, CUDA placement and per-case reference.
