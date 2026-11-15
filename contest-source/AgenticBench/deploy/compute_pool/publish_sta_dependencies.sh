#!/usr/bin/env bash
# Run on ta1. Preserve absolute-link layout, source worktree metadata and seed evidence.
set -euo pipefail
pool=/nas1/gcr/agentic3d-dev/compute-pool-20260908
bundle=$pool/bundles/sta-dependencies-v1
mkdir -p "$pool/bundles"
exec 9>"$pool/publish-sta-dependencies-v1.lock"
flock -n 9 || exit 3
[[ ! -e "$bundle.ready" ]] || exit 0
[[ ! -e "$bundle.tar.zst" && ! -e "$bundle.tar.zst.partial" ]] || {
  echo 'existing unfinished archive requires inspection; will not overwrite'; exit 4;
}
source=/data/gcr/agentic3d-dev/open3dbench-source-sta-codesign-20260902-v1
[[ "$(git -C "$source" rev-parse HEAD)" == 8a89deb905b96f30996ec2ce4498d80626a250a8 ]]
paths=(
  agentic3d-dev/open3dbench-source-grt-order-20260831-5round
  agentic3d-dev/open3dbench-source-sta-codesign-20260902-v1
  agentic3d-dev/open3dbench-source-tp-paper-reference-20260902-v1/.git
  agentic3d-dev/runs/evolution-20260830/open3dbench-source/.git/objects
  agentic3d-dev/docker-sta-20260902/docker/runtime-refresh/openroad
  agentic3d-runs/runtime/dreamplace-overlay
  agentic3d-runs/runtime/yosys-0.58
  agentic3d-runs/runtime/check_opensta3d_command.tcl
  agentic3d-dev/runs/grt_net_order_50round_20260902/inputs/bp/run/00_input
  agentic3d-dev/runs/grt_net_order_50round_20260902/inputs/bp/run/10_floorplan
  agentic3d-dev/runs/grt_net_order_ariane133_20260901_r5/benchmark-input-v4/run/00_input
  agentic3d-dev/runs/grt_net_order_ariane133_20260901_r5/benchmark-input-v4/run/10_floorplan
  agentic3d-dev/runs/tp_gnn_paper_reference_20260902-v1/references-v6/swerv_wrapper/run/00_input
  agentic3d-dev/runs/tp_gnn_paper_reference_20260902-v1/references-v6/swerv_wrapper/run/10_floorplan
)
for path in "${paths[@]}"; do [[ -e "/data/gcr/$path" ]] || { echo "missing $path"; exit 5; }; done
git -C "$source" status --porcelain > "$bundle.source-status.txt"
git -C "$source" diff --binary > "$bundle.source-working-tree.patch"
files=(
  "$source/Place-3D/src/partition/lol/methods/tp_gnn.py"
  "$source/Place-3D/src/placement/concurrent_dreamplace.py"
  "$source/Place-3D/thirdparty/DREAMPlace/dreamplace/BasicPlace.py"
  "$source/Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"
  "$source/Place-3D/thirdparty/DREAMPlace/dreamplace/NonLinearPlace.py"
  "$source/open3dbench/backend/three_d_opt.py"
  /data/gcr/agentic3d-dev/docker-sta-20260902/docker/runtime-refresh/openroad
)
sha256sum "${files[@]}" > "$bundle.exact-files.sha256"
echo "publishing_sta_dependencies $(date -Is)"
tar -C /data/gcr -cf - "${paths[@]}" | zstd -q -T2 -3 -o "$bundle.tar.zst.partial"
# Refuse to publish if relevant source/binary bytes changed during packaging.
sha256sum -c "$bundle.exact-files.sha256"
mv "$bundle.tar.zst.partial" "$bundle.tar.zst"
sha256sum "$bundle.tar.zst" > "$bundle.sha256"
date -Is > "$bundle.ready"
echo "sta_dependencies_published $(date -Is)"
