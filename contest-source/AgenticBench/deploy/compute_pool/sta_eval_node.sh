#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sta_node_env.sh"
# amax3 is shared and RAM can change while a model generates its candidate.
# Hold a separate admission lock across the helper so each waiting evaluation
# rechecks resources after its predecessor exits. The helper still owns GPU9.
if [[ "$OPENAGENTIC3D_NODE_ROOT" == /data/agentic3d/agentic3d-compute-20260908 ]]; then
  [[ $# -eq 8 || $# -eq 9 ]] || exit 2
  task_gpu=$7
  [[ "$task_gpu" =~ ^[0-7]$ ]] || exit 2
  mkdir -p "$OPENAGENTIC3D_EVAL_GPU_LOCK_DIR"
  exec 8>"$OPENAGENTIC3D_EVAL_GPU_LOCK_DIR/sta-admission.lock"
  flock 8
  task_waiting=0
  while true; do
    task_ram_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    task_gpu_mib=$(nvidia-smi --id="$task_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( task_ram_kib >= 67108864 && task_gpu_mib >= 12288 )); then
      echo "STA_RESOURCE_READY ram_kib=$task_ram_kib gpu_mib=$task_gpu_mib"
      break
    fi
    if (( task_waiting == 0 )); then
      echo "STA_RESOURCE_WAIT ram_kib=$task_ram_kib gpu_mib=$task_gpu_mib required_ram_kib=67108864 required_gpu_mib=12288"
      task_waiting=1
    fi
    sleep 60
  done
fi
exec bash "$OPENAGENTIC3D_FRAMEWORK_ROOT/scripts/run_sta_codesign_remote_eval.sh" "$@"
