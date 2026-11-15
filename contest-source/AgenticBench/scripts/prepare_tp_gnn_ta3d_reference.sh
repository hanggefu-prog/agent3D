#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: prepare_tp_gnn_ta3d_reference.sh CASE PROFILE SEED_RUN GPU" >&2
  exit 2
fi

CASE=$1
PROFILE=$2
SEED_RUN=$3
GPU=$4

DEV=${OPENAGENTIC3D_DEV_ROOT:-/data/gcr/agentic3d-dev}
EXPERIMENT=${OPENAGENTIC3D_EXPERIMENT_ROOT:-/data/agentic3d/agentic3d-ta3d-20260904}
FRAMEWORK=${OPENAGENTIC3D_FRAMEWORK_ROOT:-${EXPERIMENT}/framework}
SOURCE=${OPEN3DBENCH_ROOT:-${EXPERIMENT}/open3dbench-source}
IMAGE=${OPENAGENTIC3D_EVAL_DOCKER_IMAGE:-agentic3d/open3dbench-v2-runtime:offline}
RUNTIME=${OPENAGENTIC3D_RUNTIME_ROOT:-/data/gcr/agentic3d-runs/runtime}
DREAMPLACE=${OPENAGENTIC3D_DREAMPLACE_INSTALL:-${RUNTIME}/dreamplace-overlay}
COLLATERAL=${SOURCE}/OpenROAD-3D/flow/platforms/nangate45_3D
LOCK_ROOT=${OPENAGENTIC3D_EVAL_GPU_LOCK_DIR:-${EXPERIMENT}/gpu-locks}
PROFILE_PATH=${FRAMEWORK}/local_bench/benchmark_inputs/${PROFILE}/profile.yaml
OUTPUT=${EXPERIMENT}/inputs/${CASE}
SEED=${EXPERIMENT}/seeds/${CASE}
LOG=${EXPERIMENT}/logs/prepare-${CASE}.log

mkdir -p "${SEED}" "${OUTPUT}" "${LOCK_ROOT}" "${EXPERIMENT}/logs"
if [[ ! -e "${SEED}/00_input" ]]; then
  ln -s "${SEED_RUN}/00_input" "${SEED}/00_input"
fi
if [[ ! -e "${SEED}/10_floorplan" ]]; then
  ln -s "${SEED_RUN}/10_floorplan" "${SEED}/10_floorplan"
fi

flock "${LOCK_ROOT}/gpu-${GPU}.lock" docker run --rm \
  --gpus "device=${GPU}" \
  -v "${EXPERIMENT}:${EXPERIMENT}" \
  -v "${DEV}:${DEV}:ro" \
  -v "/data/gcr/agentic3d-runs:/data/gcr/agentic3d-runs:ro" \
  -v "${RUNTIME}:/runtime:ro" \
  -v "${DREAMPLACE}:/dreamplace:ro" \
  -v "${COLLATERAL}:/collateral:ro" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PATH=/runtime/yosys-0.58/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e YOSYS_EXE=/runtime/yosys-0.58/bin/yosys \
  -e PYTHONPATH="${FRAMEWORK}:${FRAMEWORK}/.deps-py38:/dreamplace:/dreamplace/dreamplace" \
  -e OPEN3DBENCH_PYTHON=/opt/conda/bin/python3 \
  -e OPEN3D_LOCAL_DREAMPLACE_INSTALL=/dreamplace \
  -e PLACE3D_DREAMPLACE_INSTALL=/dreamplace \
  -e OPENROAD_EXE=/runtime/openroad-agentic3d \
  -e OPEN3D_LINK_FROZEN_UPSTREAM=1 \
  -w "${FRAMEWORK}" \
  "${IMAGE}" /opt/conda/bin/python3 -m local_bench prepare-input "${PROFILE_PATH}" \
    --open3dbench-root "${SOURCE}" \
    --output-dir "${OUTPUT}" \
    --python /opt/conda/bin/python3 \
    --dreamplace-install /dreamplace \
    --seed-run "${SEED}" \
    --timeout 21600 2>&1 | tee "${LOG}"
