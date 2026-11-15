#!/usr/bin/env bash
# Frozen input preparation and one identity reference; no model calls.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sta_node_env.sh"
design=${1:?case required}
gpu=${2:?GPU required}
case "$gpu" in [0-7]) ;; *) exit 2 ;; esac
case "$design" in
  bp) seed=/data/gcr/agentic3d-dev/runs/grt_net_order_50round_20260902/inputs/bp/run; min_gpu_mib=6144 ;;
  ariane133) seed=/data/gcr/agentic3d-dev/runs/grt_net_order_ariane133_20260901_r5/benchmark-input-v4/run; min_gpu_mib=12288 ;;
  swerv_wrapper) seed=/data/gcr/agentic3d-dev/runs/tp_gnn_paper_reference_20260902-v1/references-v6/swerv_wrapper/run; min_gpu_mib=8192 ;;
  *) exit 2 ;;
esac
[[ -f "$task_node_root/sta-dependencies.ready" && -f "$task_node_root/framework-sta.ready" ]]
experiment="$task_node_root/sta-cases/$design"
profile="$OPENAGENTIC3D_FRAMEWORK_ROOT/local_bench/benchmark_inputs/${design}_lol_ng45_tp_gnn_sta"
[[ ! -e "$experiment" ]] || { echo 'existing case requires inspection/resume, not a duplicate run'; exit 3; }
free_gpu=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_gpu >= min_gpu_mib )) || { echo 'insufficient currently free GPU memory'; exit 4; }
free_ram_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
(( free_ram_kib >= 64 * 1024 * 1024 )) || { echo 'need 64 GiB available host RAM'; exit 4; }
mkdir -p "$experiment" "$OPENAGENTIC3D_EVAL_GPU_LOCK_DIR"
exec 8>"$task_node_root/sta-case-preparation.lock"
flock -n 8 || exit 5
exec 9>"$OPENAGENTIC3D_EVAL_GPU_LOCK_DIR/gpu-$gpu.lock"
flock -n 9 || exit 5
docker run --rm --network none --gpus "device=$gpu" --cpus 32 --memory 60g --memory-swap 60g \
  --name "sta-prepare-${design}-$$" \
  -v "$task_node_root:$task_node_root" \
  -v "$OPENAGENTIC3D_STA_DATA_ROOT/agentic3d-dev:/data/gcr/agentic3d-dev:ro" \
  -v "$OPENAGENTIC3D_STA_DATA_ROOT/agentic3d-runs:/data/gcr/agentic3d-runs:ro" \
  -e HOME=/tmp -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e PYTHONDONTWRITEBYTECODE=1 \
  -e OPEN3DBENCH_PYTHON=/opt/conda/bin/python3.8 -e OPEN3D_LINK_FROZEN_UPSTREAM=1 \
  -e OPEN3D_OPT_PASSES=2 \
  -e OPEN3D_LOCAL_DREAMPLACE_INSTALL="$OPENAGENTIC3D_DREAMPLACE_INSTALL" \
  -e PLACE3D_DREAMPLACE_INSTALL="$OPENAGENTIC3D_DREAMPLACE_INSTALL" \
  -e OPENROAD_EXE="$OPEN3D_STA_OPENROAD_EXE" \
  -e YOSYS_EXE="$OPENAGENTIC3D_RUNTIME_ROOT/yosys-0.58/bin/yosys" \
  -e PYTHONPATH="$OPENAGENTIC3D_FRAMEWORK_ROOT:$OPENAGENTIC3D_FRAMEWORK_ROOT/.deps-py38:$OPEN3DBENCH_ROOT:$OPENAGENTIC3D_DREAMPLACE_INSTALL:$OPENAGENTIC3D_DREAMPLACE_INSTALL/dreamplace" \
  "$OPENAGENTIC3D_EVAL_DOCKER_IMAGE" /opt/conda/bin/python3.8 -m local_bench prepare-input "$profile/profile.yaml" \
  --open3dbench-root "$OPEN3DBENCH_ROOT" --output-dir "$experiment/benchmark-input" \
  --python /opt/conda/bin/python3.8 --dreamplace-install "$OPENAGENTIC3D_DREAMPLACE_INSTALL" \
  --seed-run "$seed" --timeout 21600 > "$experiment/prepare.log" 2>&1
flock -u 9
bash "$task_node_root/tools-sta/sta_eval_node.sh" tp_gnn_placement_sta \
  "$OPENAGENTIC3D_FRAMEWORK_ROOT/evaluation_profiles/tp_gnn_placement_two_pass/initial_candidate" \
  "$experiment/reference" bundle "$experiment/benchmark-input" - "$gpu" full "$profile/config.yaml" \
  > "$experiment/reference.log" 2>&1
echo "case_reference_finished=$design $(date -Is)"
# The comparison driver must check STA validity and source/input provenance before API calls.
