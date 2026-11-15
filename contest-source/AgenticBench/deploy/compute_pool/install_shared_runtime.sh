#!/usr/bin/env bash
# Import the already-published runtime on the destination without desktop relay.
set -euo pipefail
node_root=${1:?isolated node root required}
case "$node_root" in
  /data/gcr/agentic3d-compute-20260908|/data/agentic3d/agentic3d-compute-20260908) ;;
  *) exit 2 ;;
esac
pool=/nas1/gcr/agentic3d-dev/compute-pool-20260908
expected_image=sha256:fe4689919f8f474c77d09bca8210b1755ae4fd862b8d3200d43d690986cd3262
expected_archive=5dd9d13bfb47b2e8f09b38f8c5b017e48692ee441568834daceb3587be4d4734
image=agentic3d/compute-pool-runtime:20260908
mkdir -p "$node_root"
exec 9>"$node_root/runtime-import.lock"
flock -n 9 || exit 3
actual=$(docker image inspect "$image" --format '{{index .Config.Labels "org.agentic3d.source-image"}}' 2>/dev/null || true)
if [[ -n "$actual" && "$actual" != "$expected_image" ]]; then
  echo 'refusing to overwrite a different runtime'; exit 4
fi
if [[ "$actual" != "$expected_image" ]]; then
  # Previous relay clients must be drained before a replacement import starts.
  if pgrep -af 'docker import' | grep -F -- "$image" >/dev/null; then
    echo 'old runtime import is still present; retry after it has exited'; exit 5
  fi
  available=$(df -Pk / | awk 'NR==2 {print $4}')
  (( available > 25 * 1024 * 1024 )) || { echo 'need 25 GiB root free'; exit 6; }
  [[ "$(cat "$pool/bundles/runtime.ready")" == "$expected_image" ]]
  echo "verifying_shared_runtime $(date -Is)"
  printf '%s  %s\n' "$expected_archive" "$pool/bundles/runtime.tar.zst" | sha256sum -c -
  echo "importing_shared_runtime $(date -Is)"
  zstd -dc "$pool/bundles/runtime.tar.zst" | docker import \
    --change "LABEL org.agentic3d.source-image=$expected_image" \
    --change 'ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
    --change 'ENV LD_LIBRARY_PATH=/opt/legacy-libs' \
    --change 'ENV LANG=C.UTF-8' - "$image"
fi
[[ "$(docker image inspect "$image" --format '{{index .Config.Labels "org.agentic3d.source-image"}}')" == "$expected_image" ]]
docker run --rm --network none "$image" /opt/conda/bin/python3.8 -c 'import sys; print(sys.version)'
printf '%s\n' "$expected_image" > "$node_root/runtime.ready"
echo "runtime_ready $(date -Is)"
# This is NOT a GPU placement or STA qualification.
