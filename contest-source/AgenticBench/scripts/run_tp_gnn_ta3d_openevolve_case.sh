#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: run_tp_gnn_ta3d_openevolve_case.sh RUN_ROOT CASE PROFILE INPUT_DIR GPU" >&2
  exit 2
fi

RUN_ROOT=$1
CASE=$2
PROFILE=$3
INPUT=$4
GPU=$5

DEV=${OPENAGENTIC3D_DEV_ROOT:-/data/gcr/agentic3d-dev}
EXPERIMENT=${OPENAGENTIC3D_EXPERIMENT_ROOT:-/data/agentic3d/agentic3d-ta3d-20260904}
FRAMEWORK=${OPENAGENTIC3D_FRAMEWORK_ROOT:-${EXPERIMENT}/framework}
SOURCE=${OPEN3DBENCH_ROOT:-${EXPERIMENT}/open3dbench-source}
OPENEVOLVE=${OPENEVOLVE_ROOT:-${DEV}/OpenEvolve}
IMAGE=${OPENAGENTIC3D_EVAL_DOCKER_IMAGE:-agentic3d/open3dbench-v2-runtime:offline}
RUNTIME=${OPENAGENTIC3D_RUNTIME_ROOT:-/data/gcr/agentic3d-runs/runtime}
DREAMPLACE=${OPENAGENTIC3D_DREAMPLACE_INSTALL:-${RUNTIME}/dreamplace-overlay}
COLLATERAL=${SOURCE}/OpenROAD-3D/flow/platforms/nangate45_3D
LOCK_ROOT=${OPENAGENTIC3D_EVAL_GPU_LOCK_DIR:-${EXPERIMENT}/gpu-locks}
SECRET=${DEEPSEEK_ENV_FILE:-${DEV}/.secrets/deepseek.env}
TASK=tp_gnn_ta3d_learning_objective
TASK_DIR=${FRAMEWORK}/local_tasks/${TASK}
BASE_CONFIG=${FRAMEWORK}/local_bench/benchmark_inputs/${PROFILE}/config.yaml
REFERENCE=${INPUT}/run
CASE_ROOT=${RUN_ROOT}/${CASE}
BUNDLE=${CASE_ROOT}/bundle
OUTPUT=${CASE_ROOT}/openevolve
EVALUATIONS=${CASE_ROOT}/evaluations
PREFLIGHT=${CASE_ROOT}/preflight
LOG=${CASE_ROOT}/pipeline.log

mkdir -p "${CASE_ROOT}" "${LOCK_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
echo "$(stamp) case=${CASE} gpu=${GPU} validating frozen TA3D input"

[[ -f "${SECRET}" ]] || { echo "missing DeepSeek environment file" >&2; exit 10; }
[[ -d "${TASK_DIR}" ]] || { echo "missing task directory: ${TASK_DIR}" >&2; exit 11; }
[[ -f "${BASE_CONFIG}" ]] || { echo "missing base config: ${BASE_CONFIG}" >&2; exit 12; }
[[ -f "${INPUT}/benchmark_input.json" ]] || { echo "missing benchmark input manifest" >&2; exit 13; }
[[ -f "${REFERENCE}/20_partition/partition_metrics.json" ]] || { echo "missing TA3D reference metrics" >&2; exit 14; }
[[ -x "${OPENEVOLVE}/.venv/bin/python" ]] || { echo "missing OpenEvolve virtual environment" >&2; exit 15; }
SOURCE_COMMIT=$(git -C "${SOURCE}" rev-parse HEAD)
[[ -z "$(git -C "${SOURCE}" status --short)" ]] || { echo "evaluation source is dirty" >&2; exit 16; }
echo "source_commit=${SOURCE_COMMIT}"
docker image inspect "${IMAGE}" >/dev/null

PYTHONPATH="${OPENEVOLVE}/.deps:${FRAMEWORK}" python3 -m local_bench validate-input \
  "${INPUT}" --task "${TASK}"

python3 - "${CASE}" "${PROFILE}" "${INPUT}" "${BASE_CONFIG}" <<'PY'
import json
import math
import sys
from pathlib import Path

case, profile, input_value, config_value = sys.argv[1:]
input_root = Path(input_value).resolve()
config = Path(config_value).read_text(encoding="utf-8")
manifest = json.loads((input_root / "benchmark_input.json").read_text(encoding="utf-8"))
metrics = json.loads((input_root / "run/20_partition/partition_metrics.json").read_text(encoding="utf-8"))
timing = json.loads((input_root / "run/20_partition/timing_input/timing_manifest.json").read_text(encoding="utf-8"))
if manifest.get("benchmark_id") != profile:
    raise SystemExit("benchmark profile/input id mismatch")
if "design: {}\n".format(case) not in config:
    raise SystemExit("case/base-config mismatch")
if "config: config/partition/tp_gnn_ta3d.json\n" not in config:
    raise SystemExit("base config does not select the TA3D decoder")
if metrics.get("timing_weight_mode") != "paper_delay_slack_hyperedge_v1":
    raise SystemExit("reference does not use DATE delay/slack weights")
if timing.get("timing_status") != "full_coverage_partition_last":
    raise SystemExit("reference timing input is not full-coverage partition-last")
for name in (
    "timing_critical_total",
    "timing_noncritical_total",
    "critical_cut_reward",
    "noncritical_cut_penalty",
    "ta3d_signed_cut_score",
):
    value = float(metrics[name])
    if not math.isfinite(value):
        raise SystemExit("non-finite TA3D reference metric: {}".format(name))
print("reference_signed_score={:.12g} critical_reward={:.12g} noncritical_penalty={:.12g}".format(
    float(metrics["ta3d_signed_cut_score"]),
    float(metrics["critical_cut_reward"]),
    float(metrics["noncritical_cut_penalty"]),
))
PY

set -a
source "${SECRET}"
set +a
[[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is empty" >&2; exit 17; }

if [[ ! -f "${BUNDLE}/config.yaml" ]]; then
  mkdir -p "${BUNDLE}"
  PYTHONPATH="${OPENEVOLVE}/.deps:${FRAMEWORK}" python3 -m local_bench export-openevolve \
    "${TASK}" \
    --benchmark-profile "${PROFILE}" \
    --open3dbench-root "${SOURCE}" \
    --output-dir "${BUNDLE}"
  PYTHONPATH="${OPENEVOLVE}/.deps:${FRAMEWORK}" python3 \
    "${FRAMEWORK}/scripts/configure_openevolve_deepseek.py" "${BUNDLE}/config.yaml"
fi

PYTHONPATH="${OPENEVOLVE}/.deps:${FRAMEWORK}" python3 - \
  "${BUNDLE}/config.yaml" "${BUNDLE}/evaluator.py" "${BASE_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path, evaluator_path, base_config = map(Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if int(config.get("max_iterations", -1)) != 20:
    raise SystemExit("OpenEvolve max_iterations is not 20")
if config.get("diff_based_evolution") is not False:
    raise SystemExit("function task must use whole-block evolution")
if config.get("llm", {}).get("models", [{}])[0].get("name") != "deepseek-v4-flash":
    raise SystemExit("DeepSeek model is not configured")
message = str(config.get("prompt", {}).get("system_message", ""))
if "signed timing-cut score" not in message or "TA3D-style" not in message:
    raise SystemExit("exported prompt lost the TA3D signed-fitness contract")
if repr(str(base_config.resolve())) not in evaluator_path.read_text(encoding="utf-8"):
    raise SystemExit("exported evaluator is not bound to the case config")
print("bundle_contract=passed iterations=20 provider=deepseek")
PY

if [[ ! -f "${PREFLIGHT}/evaluation_result.json" ]]; then
  mkdir -p "${PREFLIGHT}"
  docker run --rm \
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
    -w "${FRAMEWORK}" \
    "${IMAGE}" /opt/conda/bin/python3 -m local_bench evaluate "${TASK}" \
      --candidate "${BUNDLE}/initial_program.py" \
      --open3dbench-root "${SOURCE}" \
      --output-dir "${PREFLIGHT}" \
      --benchmark-input "${INPUT}" \
      --reference-run "${REFERENCE}" \
      --base-config "${BASE_CONFIG}" \
      --dry-run
fi

python3 - "${PREFLIGHT}/evaluation_result.json" <<'PY'
import json
import sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if float(result.get("metrics", {}).get("dry_run", 0.0)) != 1.0:
    raise SystemExit("candidate preflight did not pass")
print("candidate_preflight=passed")
PY

mkdir -p "${OUTPUT}" "${EVALUATIONS}"
export OPEN3DBENCH_ROOT="${SOURCE}"
export OPEN3DBENCH_REFERENCE_RUN="${REFERENCE}"
export OPENAGENTIC3D_BENCHMARK_INPUT="${INPUT}"
export OPENAGENTIC3D_BASE_CONFIG="${BASE_CONFIG}"
export OPEN3D_LOCAL_EVAL_ROOT="${EVALUATIONS}"
export OPEN3D_LOCAL_EVAL_TIMEOUT=21600
export OPENAGENTIC3D_EVAL_COMPACT_ARTIFACTS=1
export OPENAGENTIC3D_EVAL_DOCKER_IMAGE="${IMAGE}"
export OPENAGENTIC3D_EVAL_DOCKER_GPU="${GPU}"
export OPENAGENTIC3D_EVAL_GPU_LOCK_DIR="${LOCK_ROOT}"
export OPENAGENTIC3D_EVAL_DOCKER_PYTHON=/opt/conda/bin/python3
export OPENAGENTIC3D_EVAL_DOCKER_WORKDIR="${FRAMEWORK}"
export OPENAGENTIC3D_EVAL_DOCKER_MOUNTS_JSON="[\"${EXPERIMENT}:${EXPERIMENT}\",\"${DEV}:${DEV}:ro\",\"/data/gcr/agentic3d-runs:/data/gcr/agentic3d-runs:ro\",\"${RUNTIME}:/runtime:ro\",\"${DREAMPLACE}:/dreamplace:ro\",\"${COLLATERAL}:/collateral:ro\"]"
export OPENAGENTIC3D_EVAL_DOCKER_ENV_JSON="{\"HOME\":\"/tmp\",\"PYTHONDONTWRITEBYTECODE\":\"1\",\"PATH\":\"/runtime/yosys-0.58/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\",\"YOSYS_EXE\":\"/runtime/yosys-0.58/bin/yosys\",\"PYTHONPATH\":\"${SOURCE}:${FRAMEWORK}:${FRAMEWORK}/.deps-py38:/dreamplace:/dreamplace/dreamplace\",\"OPEN3DBENCH_PYTHON\":\"/opt/conda/bin/python3\",\"OPEN3D_LOCAL_DREAMPLACE_INSTALL\":\"/dreamplace\",\"PLACE3D_DREAMPLACE_INSTALL\":\"/dreamplace\",\"OPENROAD_EXE\":\"/runtime/openroad-agentic3d\",\"OPEN3D_LINK_FROZEN_UPSTREAM\":\"1\"}"

echo "$(stamp) starting OpenEvolve case=${CASE}: 20 iterations on GPU ${GPU}"
cd "${OPENEVOLVE}"
PYTHONPATH="${OPENEVOLVE}/.deps:${OPENEVOLVE}:${FRAMEWORK}" \
  "${OPENEVOLVE}/.venv/bin/python" openevolve-run.py \
    "${BUNDLE}/initial_program.py" \
    "${BUNDLE}/evaluator.py" \
    --config "${BUNDLE}/config.yaml" \
    --output "${OUTPUT}" \
    --iterations 20 \
    --log-level INFO

echo "$(stamp) completed case=${CASE} iterations=20"
