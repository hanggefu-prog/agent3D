#!/usr/bin/env bash
# One-time, isolated ariane reference repair; retain failed reference evidence.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sta_node_env.sh"
[[ "$task_node_root" == /data/agentic3d/agentic3d-compute-20260908 ]]
experiment="$task_node_root/sta-cases/ariane133"
source_actual="$OPENAGENTIC3D_STA_DATA_ROOT/agentic3d-dev/open3dbench-source-sta-codesign-20260902-v1"
code="$source_actual/open3dbench/backend/three_d_opt.py"
patch="$task_node_root/tools-sta/guard_absent_die_false_path.patch"
[[ -f "$experiment/guard-diagnostic-v1/qualification.json" ]]
[[ ! -e "$experiment/reference-failed-sta0471" && ! -e "$code.before-die-sdc-guard" ]]
[[ "$(sha256sum "$code" | cut -d ' ' -f1)" == bebaf313e31f3320291652ae5d712b5a17a0b1bacb9c0463e4311d3b4bd7e615 ]]
[[ "$(stat -c %h "$code")" == 1 ]]
[[ -z "$(docker ps --filter name=sta-codesign --format '{{.Names}}')" ]]
(( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) >= 64 * 1024 * 1024 ))
(( $(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits) >= 12288 ))
exec 8>"$task_node_root/sta-case-preparation.lock"
flock -n 8
git apply --check --unsafe-paths --directory "$source_actual" "$patch"
cp -p "$code" "$code.before-die-sdc-guard"
git apply --unsafe-paths --directory "$source_actual" "$patch"
[[ "$(sha256sum "$code" | cut -d ' ' -f1)" == 3c43822c258d1bccf3caee1f722d5e3e8758163abb724c6c344c41562ed76161 ]]
sha256sum "$code" "$patch" > "$experiment/die-sdc-guard.sha256"
mv "$experiment/reference" "$experiment/reference-failed-sta0471"
mv "$experiment/reference.log" "$experiment/reference-failed-sta0471.log"
echo "guard_applied_reference_retry_started $(date -Is)"
bash "$task_node_root/tools-sta/sta_eval_node.sh" tp_gnn_placement_sta \
  "$OPENAGENTIC3D_FRAMEWORK_ROOT/evaluation_profiles/tp_gnn_placement_two_pass/initial_candidate" \
  "$experiment/reference" bundle "$experiment/benchmark-input" - 0 full \
  "$OPENAGENTIC3D_FRAMEWORK_ROOT/local_bench/benchmark_inputs/ariane133_lol_ng45_tp_gnn_sta/config.yaml" \
  > "$experiment/reference.log" 2>&1
echo "guard_reference_retry_finished $(date -Is)"
