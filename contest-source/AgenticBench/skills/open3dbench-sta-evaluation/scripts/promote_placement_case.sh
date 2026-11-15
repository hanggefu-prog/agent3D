#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CASE GPU PLACEMENT_RESULTS_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

case_name=$1
gpu=$2
placement_root=${3%/}
output_root=${4%/}

case "$case_name" in
  bp|bp_multi|ariane133|swerv_wrapper) ;;
  *) echo "unsupported case: $case_name" >&2; exit 2 ;;
esac
case "$gpu" in [0-7]) ;; *) echo "GPU must be between 0 and 7" >&2; exit 2 ;; esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
framework=${OPENAGENTIC3D_FRAMEWORK_ROOT:-$repo_root}
source_root=${OPEN3DBENCH_ROOT:-/data/gcr/agentic3d-dev/runs/evolution-20260830/open3dbench-source}
image=${OPENAGENTIC3D_EVAL_DOCKER_IMAGE:-agentic3d/open3dbench-v2-runtime:offline}
helper=${OPENAGENTIC3D_STA_REMOTE_HELPER:-$script_dir/run_remote_sta.sh}
mounts=(-v /data:/data)
if [[ "$framework $source_root $placement_root $output_root" == *"/nas1/gcr/"* ]]; then
  mounts+=(-v /nas1/gcr:/nas1/gcr)
fi
if [[ -n "${OPENAGENTIC3D_NODE_ROOT:-}" ]]; then
  [[ "$OPENAGENTIC3D_NODE_ROOT" == /data/agentic3d/agentic3d-* || "$OPENAGENTIC3D_NODE_ROOT" == /data/gcr/agentic3d-* ]] || {
    echo "OPENAGENTIC3D_NODE_ROOT must be an isolated agentic3d path" >&2
    exit 2
  }
  mounts+=(-v "$OPENAGENTIC3D_NODE_ROOT:$OPENAGENTIC3D_NODE_ROOT")
fi
case_root="$placement_root/$case_name"
baseline="$case_root/reference/run"
base_config="$case_root/reference/resolved_config.yaml"
case_output="$output_root/$case_name"

for required in "$framework" "$source_root" "$helper" "$baseline/20_partition/stage_manifest.json" "$base_config"; do
  [[ -e "$required" ]] || { echo "required input is missing: $required" >&2; exit 2; }
done
mkdir -p "$case_output"

for method in dynamic source; do
  candidate="$case_output/$method/candidate"
  evaluation="$case_output/$method/evaluation"
  [[ ! -e "$evaluation/evaluation_result.json" ]] || {
    echo "evaluation already exists: $evaluation/evaluation_result.json" >&2
    exit 2
  }
  mkdir -p "$candidate" "$evaluation"

  docker run --rm \
    "${mounts[@]}" \
    -e HOME=/tmp \
    -e PYTHONPATH="$framework:$framework/.deps-py38:$source_root" \
    "$image" \
    /opt/conda/bin/python3.8 "$framework/scripts/select_placement_best_candidate.py" \
      --framework-root "$framework" \
      --case-root "$case_root" \
      --method "$method" \
      --output-dir "$candidate"

  "$helper" \
    placement_two_pass_sta \
    "$candidate" \
    "$evaluation" \
    run \
    "$baseline" \
    - \
    "$gpu" \
    compact \
    "$base_config"
done
