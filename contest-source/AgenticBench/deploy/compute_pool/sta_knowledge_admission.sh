#!/usr/bin/env bash
# New batch only. Preserve the qualified helper, GPU locks and evaluator.
set -euo pipefail
node_kind=${1:?ta1 or pool}
shift
[[ $# == 8 || $# == 9 ]] || exit 2
case "$node_kind" in
  ta1)
    batch_root=/data/gcr/agentic3d-dev/sta-knowledge-v2-20260909
    helper=$batch_root/qualified_helper.sh
    ;;
  pool)
    script_parent=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    node_root=$(dirname "$script_parent")
    case "$node_root" in /data/gcr/agentic3d-compute-20260908|/data/agentic3d/agentic3d-compute-20260908) ;; *) exit 2 ;; esac
    batch_root=$node_root/sta-knowledge-v2-20260909
    helper=$node_root/tools-sta/sta_eval_node.sh
    ;;
  *) exit 2 ;;
esac
gpu=$7
[[ "$gpu" =~ ^[0-7]$ ]] || exit 2
mkdir -p "$batch_root"
exec 7>"$batch_root/batch-admission.lock"
flock 7
while true; do
  ram_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  free_kib=$(df -Pk "$batch_root" | awk 'NR==2 {print $4}')
  gpu_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  if (( ram_kib >= 67108864 && free_kib >= 20971520 && gpu_mib >= 12288 )); then
    echo "V2_RESOURCE_READY ram_kib=$ram_kib disk_kib=$free_kib gpu_mib=$gpu_mib"
    break
  fi
  echo "V2_RESOURCE_WAIT ram_kib=$ram_kib disk_kib=$free_kib gpu_mib=$gpu_mib"
  sleep 60
done
exec bash "$helper" "$@"
