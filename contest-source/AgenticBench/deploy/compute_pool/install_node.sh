#!/usr/bin/env bash
# Install an isolated CPU evaluation slot without modifying system services.
set -euo pipefail
pool=${1:?shared pool directory required}
node_root=${2:?isolated local node directory required}
node=${3:?node name required}
case "$node" in ta2|ta3|amax3) ;; *) exit 2 ;; esac
case "$node_root" in /data/gcr/agentic3d-compute-20260908|/data/agentic3d/agentic3d-compute-20260908) ;; *) exit 2 ;; esac
[[ "$pool" == /nas1/gcr/agentic3d-dev/compute-pool-* ]] || exit 2
mkdir -p "$node_root" "$pool/nodes/$node"
exec 9>"$node_root/install.lock"
flock -n 9 || exit 3
trap 'echo "installation_failed line=$LINENO"; date -Is > "$pool/nodes/$node/install.failed"' ERR
[[ ! -e "$node_root/installed.ready" ]] || exit 0
available=$(df -Pk "$node_root" | awk 'NR==2 {print $4}')
required_gib=60
[[ ! -f "$node_root/source.ready" ]] || required_gib=20
(( available > required_gib * 1024 * 1024 )) || { echo "need $required_gib GiB local free space"; exit 4; }
docker info >/dev/null
for component in runtime source; do
  until [[ -f "$pool/bundles/$component.ready" || -f "$node_root/$component.ready" ]]; do
    echo "waiting_for=$component $(date -Is)"
    sleep 30
  done
done
if [[ -f "$node_root/runtime.ready" ]]; then
  expected=$(cat "$node_root/runtime.ready")
else
  expected=$(cat "$pool/bundles/runtime.ready")
fi
image=agentic3d/compute-pool-runtime:20260908
actual=$(docker image inspect "$image" --format '{{index .Config.Labels "org.agentic3d.source-image"}}' 2>/dev/null || true)
if [[ -n "$actual" && "$actual" != "$expected" ]]; then
  echo "refusing to replace an existing different runtime image"; exit 5
fi
if [[ "$actual" != "$expected" ]]; then
  docker_root=$(docker info --format '{{.DockerRootDir}}')
  docker_available=$(df -Pk "$docker_root" 2>/dev/null | awk 'NR==2 {print $4}' || true)
  # Some daemons keep their root private; /var/lib/docker is on / on these hosts.
  [[ -n "$docker_available" ]] || docker_available=$(df -Pk / | awk 'NR==2 {print $4}')
  (( docker_available > 25 * 1024 * 1024 )) || { echo "need 25 GiB Docker disk headroom"; exit 4; }
  (cd "$pool/bundles" && sha256sum -c runtime.sha256)
  zstd -dc "$pool/bundles/runtime.tar.zst" | docker import \
    --change "LABEL org.agentic3d.source-image=$expected" \
    --change 'ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
    --change 'ENV LD_LIBRARY_PATH=/opt/legacy-libs' \
    --change 'ENV LANG=C.UTF-8' - "$image"
fi
[[ "$(docker image inspect "$image" --format '{{index .Config.Labels "org.agentic3d.source-image"}}')" == "$expected" ]]
mkdir -p "$node_root/data" "$node_root/inputs" "$node_root/scratch" "$node_root/results" "$node_root/candidates"
if [[ ! -f "$node_root/source.ready" ]]; then
  (cd "$pool/bundles" && sha256sum -c source.sha256)
  zstd -dc "$pool/bundles/source.tar.zst" | tar --no-same-owner -xf - -C "$node_root/data"
  date -Is > "$node_root/source.ready"
fi
until [[ -f "$node_root/git-objects.ready" ]]; do
  echo "waiting_for=git-objects $(date -Is)"
  sleep 30
done
until [[ -f "$pool/bundles/metadata.ready" ]]; do sleep 10; done
rsync -a "$pool/framework/" "$node_root/framework/"
rsync -a "$pool/references/" "$node_root/references/"
for design in bp ariane133; do
  echo "copying_frozen_input=$design $(date -Is)"
  if [[ "$design" == bp ]]; then
    input=/nas1/gcr/agentic3d-dev/runs/grt_heuristics_50round_20260903-v1/inputs/bp
  else
    input=/nas1/gcr/agentic3d-dev/runs/grt_ariane133_repaired_50round_20260907-v2/inputs/ariane133
  fi
  mkdir -p "$node_root/inputs/$design"
  if [[ "$design" == bp ]]; then
    native="$node_root/data/runs/grt_net_order_50round_20260902/inputs/bp"
  else
    native="$node_root/data/runs/grt_net_order_ariane133_20260901_r5/benchmark-input"
  fi
  if [[ -d "$native/run" ]]; then
    rsync -a --link-dest="$native" "$native/" "$node_root/inputs/$design/"
    # The maintained NAS manifest is authoritative; validation below verifies
    # all stage hashes and every artifact against this copied native run.
    cp "$input/benchmark_input.json" "$node_root/inputs/$design/benchmark_input.json"
  else
    rsync -a --exclude=run.external-reference "$input/" "$node_root/inputs/$design/"
  fi
done
rsync -a "$pool/tools/" "$node_root/tools/"
date -Is > "$node_root/installed.ready"
cp "$node_root/installed.ready" "$pool/nodes/$node/installed.ready"
echo "installed=$node; starting independent qualification (not evolution budget)"
bash "$node_root/tools/run_worker.sh" "$node_root" "$node" qualify
cp "$node_root/results/qualification.json" "$pool/nodes/$node/qualification.json"
echo "qualified=$node"
