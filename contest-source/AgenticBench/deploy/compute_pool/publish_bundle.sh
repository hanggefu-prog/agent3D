#!/usr/bin/env bash
# Run on ta1. Publish only named experiment dependencies; never copy credentials.
set -euo pipefail
pool=${1:?pool directory required}
component=${2:?runtime, source, or metadata required}
[[ "$pool" == /nas1/gcr/agentic3d-dev/compute-pool-* ]] || exit 2
mkdir -p "$pool/bundles" "$pool/references" "$pool/logs"
exec 9>"$pool/publish-$component.lock"
flock -n 9 || exit 3
image=agentic3d/open3dbench-v2-runtime:offline
expected=sha256:fe4689919f8f474c77d09bca8210b1755ae4fd862b8d3200d43d690986cd3262
data=/data/gcr/agentic3d-dev
case "$component" in
  runtime)
    [[ ! -e "$pool/bundles/runtime.ready" ]] || exit 0
    [[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$expected" ]]
    # docker save materializes all image layers under DockerRootDir/tmp, which
    # is unsafe on ta1's nearly-full system disk. Export a stopped, never-run
    # container instead: stream its merged filesystem straight to NAS.
    container=agentic3d-pool-export-20260908
    docker create --name "$container" "$image" /bin/true >/dev/null
    trap 'docker rm "$container" >/dev/null' EXIT
    docker export "$container" | zstd -q -T2 -3 -o "$pool/bundles/runtime-fs.tar.zst.partial"
    mv "$pool/bundles/runtime-fs.tar.zst.partial" "$pool/bundles/runtime.tar.zst.partial"
    mv "$pool/bundles/runtime.tar.zst.partial" "$pool/bundles/runtime.tar.zst"
    (cd "$pool/bundles" && sha256sum runtime.tar.zst > runtime.sha256)
    printf '%s\n' "$expected" > "$pool/bundles/runtime.ready"
    ;;
  source)
    [[ ! -e "$pool/bundles/source.ready" ]] || exit 0
    tar -C "$data" -cf - open3dbench-source-grt-order-20260831-5round EvoStage-main \
      | zstd -T2 -3 -o "$pool/bundles/source.tar.zst.partial"
    mv "$pool/bundles/source.tar.zst.partial" "$pool/bundles/source.tar.zst"
    (cd "$pool/bundles" && sha256sum source.tar.zst > source.sha256)
    bash "$0" "$pool" metadata
    date -Is > "$pool/bundles/source.ready"
    ;;
  metadata)
    # Scoring uses reference metrics and exact upstream manifests, not the old
    # reference's reproducible multi-GB physical outputs.
    for design in bp ariane133; do
      if [[ "$design" == bp ]]; then
        reference="$data/runs/grt_net_order_50round_20260902/references/bp/run"
      else
        reference="$data/runs/grt_net_order_ariane133_20260901_r5/reference-v4/run"
      fi
      mkdir -p "$pool/references/$design"
      rsync -a --include='*/' --include=metrics.json --include=stage_manifest.json \
        --exclude='*' "$reference/" "$pool/references/$design/"
    done
    rsync -a --exclude=__pycache__ \
      /nas1/gcr/agentic3d-dev/framework-grt-ariane133-repaired-20260907-v2/ "$pool/framework/"
    date -Is > "$pool/bundles/metadata.ready"
    ;;
  *) exit 2 ;;
esac
echo "published=$component"
