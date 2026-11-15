#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: prepare_placement_input.sh PROFILE OUTPUT_DIR GPU" >&2
  exit 2
fi

PROFILE=$1
OUTPUT=$2
GPU=$3
DEV=${OPENAGENTIC3D_DEV_ROOT:-/data/gcr/agentic3d-dev}
FRAMEWORK=${OPENAGENTIC3D_FRAMEWORK_ROOT:-${DEV}/framework}
SOURCE=${OPEN3DBENCH_ROOT:-${DEV}/Open3DBench-V2.0}
IMAGE=${OPENAGENTIC3D_EVAL_DOCKER_IMAGE:-agentic3d/open3dbench-v2-runtime:offline}
RUNTIME=${OPENAGENTIC3D_RUNTIME_ROOT:-/data/gcr/agentic3d-runs/runtime}
DREAMPLACE=${OPENAGENTIC3D_DREAMPLACE_INSTALL:-${RUNTIME}/dreamplace-overlay}
COLLATERAL=${SOURCE}/OpenROAD-3D/flow/platforms/nangate45_3D
LOCK_ROOT=${OPENAGENTIC3D_EVAL_GPU_LOCK_DIR:-${DEV}/runs/gpu-locks}

mkdir -p "${OUTPUT}" "${LOCK_ROOT}"
flock "${LOCK_ROOT}/gpu-${GPU}.lock" docker run --rm \
  --gpus "device=${GPU}" \
  -v "${DEV}:${DEV}" \
  -v "/data/gcr/agentic3d-runs:/data/gcr/agentic3d-runs" \
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
  -w "${FRAMEWORK}" \
  "${IMAGE}" /opt/conda/bin/python3 -m local_bench prepare-input "${PROFILE}" \
    --open3dbench-root "${SOURCE}" \
    --output-dir "${OUTPUT}" \
    --python /opt/conda/bin/python3 \
    --dreamplace-install /dreamplace \
    --timeout 21600 \
    --resume
