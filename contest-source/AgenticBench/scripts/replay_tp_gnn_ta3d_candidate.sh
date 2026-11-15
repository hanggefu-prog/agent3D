#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: replay_tp_gnn_ta3d_candidate.sh OUTPUT_DIR CASE PROFILE INPUT_DIR PROGRAM GPU" >&2
  exit 2
fi

OUTPUT=$1
CASE=$2
PROFILE=$3
INPUT=$4
PROGRAM=$5
GPU=$6

DEV=${OPENAGENTIC3D_DEV_ROOT:-/data/gcr/agentic3d-dev}
EXPERIMENT=${OPENAGENTIC3D_EXPERIMENT_ROOT:-/data/agentic3d/agentic3d-ta3d-20260904}
FRAMEWORK=${OPENAGENTIC3D_FRAMEWORK_ROOT:-${EXPERIMENT}/framework}
SOURCE=${OPEN3DBENCH_ROOT:-${EXPERIMENT}/open3dbench-source}
IMAGE=${OPENAGENTIC3D_EVAL_DOCKER_IMAGE:-agentic3d/open3dbench-v2-runtime:offline}
RUNTIME=${OPENAGENTIC3D_RUNTIME_ROOT:-/data/gcr/agentic3d-runs/runtime}
DREAMPLACE=${OPENAGENTIC3D_DREAMPLACE_INSTALL:-${RUNTIME}/dreamplace-overlay}
COLLATERAL=${SOURCE}/OpenROAD-3D/flow/platforms/nangate45_3D
TASK=tp_gnn_ta3d_learning_objective
BASE_CONFIG=${FRAMEWORK}/local_bench/benchmark_inputs/${PROFILE}/config.yaml

[[ -f "${INPUT}/benchmark_input.json" ]] || { echo "missing benchmark input" >&2; exit 10; }
[[ -f "${PROGRAM}" ]] || { echo "missing TP-GNN program" >&2; exit 11; }
[[ -f "${BASE_CONFIG}" ]] || { echo "missing case config" >&2; exit 12; }
[[ -z "$(git -C "${SOURCE}" status --short)" ]] || {
  echo "evaluation source is dirty" >&2
  exit 13
}

mkdir -p "${OUTPUT}"
docker run --rm --gpus "device=${GPU}" \
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
  -e PYTHONPATH="${SOURCE}:${FRAMEWORK}:${FRAMEWORK}/.deps-py38:/dreamplace:/dreamplace/dreamplace" \
  -e OPEN3DBENCH_PYTHON=/opt/conda/bin/python3 \
  -e OPEN3D_LOCAL_DREAMPLACE_INSTALL=/dreamplace \
  -e PLACE3D_DREAMPLACE_INSTALL=/dreamplace \
  -e OPENROAD_EXE=/runtime/openroad-agentic3d \
  -e OPEN3D_LINK_FROZEN_UPSTREAM=1 \
  -w "${FRAMEWORK}" \
  "${IMAGE}" /opt/conda/bin/python3 -m local_bench evaluate "${TASK}" \
    --candidate "${PROGRAM}" \
    --open3dbench-root "${SOURCE}" \
    --output-dir "${OUTPUT}" \
    --benchmark-input "${INPUT}" \
    --reference-run "${INPUT}/run" \
    --base-config "${BASE_CONFIG}" \
    --compact-artifacts

python3 - "${OUTPUT}/evaluation_result.json" "${CASE}" <<'PY'
import json
import sys

path, case = sys.argv[1:]
metrics = json.load(open(path, encoding="utf-8")).get("metrics", {})
if float(metrics.get("runs_successfully", 0.0)) != 1.0:
    raise SystemExit("{} replay failed".format(case))
print(
    "case={} fitness={:.12g} signed={:.12g} gain={:.12g}".format(
        case,
        float(metrics["combined_score"]),
        float(metrics["ta3d_signed_cut_score"]),
        float(metrics["ta3d_signed_cut_gain"]),
    )
)
PY
