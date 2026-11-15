#!/usr/bin/env bash
# One-time retry through the real evaluator-patch channel, preserving failures.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sta_node_env.sh"
[[ "$task_node_root" == /data/agentic3d/agentic3d-compute-20260908 ]]
experiment="$task_node_root/sta-cases/ariane133"
patched="$task_node_root/tools-sta/tp_gnn_placement_sta_guarded.patch"
installed="$OPENAGENTIC3D_FRAMEWORK_ROOT/local_bench/evaluator_patches/tp_gnn_placement_sta.patch"
diagnostic="$experiment/guard-diagnostic-v2-materialized"
[[ -f "$diagnostic/qualification.json" ]]
grep -q '"workspace_materialized": true' "$diagnostic/qualification.json"
[[ "$(sha256sum "$diagnostic/patched/open3dbench/backend/three_d_opt.py" | cut -d ' ' -f1)" == 3c43822c258d1bccf3caee1f722d5e3e8758163abb724c6c344c41562ed76161 ]]
[[ "$(sha256sum "$patched" | cut -d ' ' -f1)" == 27c39e1ddc08e577ca0f56cd6ec4fc8905f177ac5455f131ed2a56996af37f0f ]]
[[ "$(sha256sum "$installed" | cut -d ' ' -f1)" == 344802366e4a7571879eb559c3b1117ee4c6efef65f6765844986d03adce845c ]]
[[ ! -e "$experiment/reference-failed-guard-not-cloned" && ! -e "$installed.before-die-sdc-guard" ]]
[[ -z "$(docker ps --filter name=sta-codesign --format '{{.Names}}')" ]]
free_ram_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
(( free_ram_kib >= 64 * 1024 * 1024 )) || { echo "waiting_for_safe_ram available_kib=$free_ram_kib"; exit 4; }
(( $(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits) >= 12288 ))
exec 8>"$task_node_root/sta-case-preparation.lock"
flock -n 8
cp -p "$installed" "$installed.before-die-sdc-guard"
cp "$patched" "$installed"
sha256sum "$installed" > "$experiment/materialized-die-sdc-guard.sha256"
mv "$experiment/reference" "$experiment/reference-failed-guard-not-cloned"
mv "$experiment/reference.log" "$experiment/reference-failed-guard-not-cloned.log"
echo "materialized_guard_reference_retry_started $(date -Is)"
bash "$task_node_root/tools-sta/sta_eval_node.sh" tp_gnn_placement_sta \
  "$OPENAGENTIC3D_FRAMEWORK_ROOT/evaluation_profiles/tp_gnn_placement_two_pass/initial_candidate" \
  "$experiment/reference" bundle "$experiment/benchmark-input" - 0 full \
  "$OPENAGENTIC3D_FRAMEWORK_ROOT/local_bench/benchmark_inputs/ariane133_lol_ng45_tp_gnn_sta/config.yaml" \
  > "$experiment/reference.log" 2>&1
echo "materialized_guard_reference_retry_finished $(date -Is)"
