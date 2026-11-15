#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT bp|ariane133 ripup|maze|iteration" >&2
  exit 2
fi

run_root="$1"
case_name="$2"
heuristic="$3"

framework="${OPENAGENTIC3D_FRAMEWORK:-/nas1/gcr/agentic3d-dev/framework-grt-heuristics-20260903-v1}"
source_root="${OPEN3DBENCH_SOURCE:-/data/gcr/agentic3d-dev/open3dbench-source-grt-order-20260831-5round}"
evostage_root="${EVOSTAGE_ROOT:-/data/gcr/agentic3d-dev/EvoStage-main}"
image="${OPEN3DBENCH_IMAGE:-agentic3d/open3dbench-v2-runtime:offline}"
scratch_host="${OPEN3D_SCRATCH_HOST:-/data/agentic3d/agentic3d-grt-scratch}"

case "${case_name}" in
  bp)
    benchmark_input="${run_root}/inputs/bp/benchmark_input.json"
    reference_run="/data/gcr/agentic3d-dev/runs/grt_net_order_50round_20260902/references/bp/run"
    ;;
  ariane133)
    benchmark_input="${run_root}/inputs/ariane133/benchmark_input.json"
    reference_run="/data/gcr/agentic3d-dev/runs/grt_net_order_ariane133_20260901_r5/reference-v4/run"
    ;;
  *)
    echo "unsupported case: ${case_name}" >&2
    exit 2
    ;;
esac

case "${heuristic}" in
  ripup) task="global_routing_ripup_policy" ;;
  maze) task="global_routing_maze_cost" ;;
  iteration) task="global_routing_iteration_controller" ;;
  *)
    echo "unsupported heuristic: ${heuristic}" >&2
    exit 2
    ;;
esac

output="${run_root}/preflight/${case_name}-${heuristic}-full"
if [[ -e "${output}" ]]; then
  echo "preflight output already exists: ${output}" >&2
  exit 2
fi
mkdir -p "${output}" "${scratch_host}/ccache"

container="grt-preflight-${case_name}-${heuristic}"
exec docker run --rm \
  --name "${container}" \
  --network host \
  -v /data/gcr/agentic3d-dev:/data/gcr/agentic3d-dev:ro \
  -v /nas1/gcr:/nas1/gcr \
  -v "${scratch_host}:/scratch" \
  "${image}" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='${framework}:/nas1/gcr/agentic3d-dev/evostage-py38-deps'
    export OPEN3DBENCH_PYTHON=/opt/conda/bin/python3
    export OPEN3D_LOCAL_OPENROAD_BUILDER='${framework}/scripts/build_openroad_candidate.sh'
    export OPEN3D_LOCAL_OPENROAD_BUILD_THREADS='${OPEN3D_BUILD_THREADS:-8}'
    export OPEN3D_LINK_FROZEN_UPSTREAM=1
    export OPEN3D_EVALUATION_SCRATCH_ROOT=/scratch
    export CCACHE_DIR=/scratch/ccache
    export CCACHE_MAXSIZE='${OPEN3D_CCACHE_MAXSIZE:-12G}'
    export CCACHE_SECONDARY_STORAGE='file:/data/gcr/agentic3d-dev/ccache-grt-order-20260831'
    cd '${framework}'
    /opt/conda/bin/python3 -m local_bench.evolve_cli run '${task}' \
      --method randsample \
      --provider replay \
      --evostage-root '${evostage_root}' \
      --evaluation-mode full \
      --open3dbench-root '${source_root}' \
      --benchmark-input '${benchmark_input}' \
      --reference-run '${reference_run}' \
      --output-dir '${output}' \
      --max-samples 1 \
      --max-generations 1 \
      --population-size 1 \
      --compact-evaluations \
      --timeout 21600
  "
