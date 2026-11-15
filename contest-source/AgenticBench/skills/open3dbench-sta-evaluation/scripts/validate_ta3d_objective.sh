#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 OUTPUT_DIR CASE PROFILE BASELINE_RUN PROGRAM GPU [REFERENCE_RUN]" >&2
  exit 2
fi

output=$1
case_name=$2
profile=$3
baseline_run=$4
program=$5
gpu=$6
reference_run=${7:-}

dev=${OPENAGENTIC3D_DEV_ROOT:-/data/gcr/agentic3d-dev}
experiment=${OPENAGENTIC3D_EXPERIMENT_ROOT:-/data/agentic3d/agentic3d-ta3d-20260904}
framework=${OPENAGENTIC3D_FRAMEWORK_ROOT:-${experiment}/framework}
source_root=${OPEN3DBENCH_ROOT:-${experiment}/open3dbench-source}
image=${OPENAGENTIC3D_EVAL_DOCKER_IMAGE:-agentic3d/open3dbench-v2-runtime:offline}
runtime=${OPENAGENTIC3D_RUNTIME_ROOT:-/data/gcr/agentic3d-runs/runtime}
dreamplace=${OPENAGENTIC3D_DREAMPLACE_INSTALL:-${runtime}/dreamplace-overlay}
collateral=${source_root}/OpenROAD-3D/flow/platforms/nangate45_3D
profile_dir=${framework}/evaluation_profiles/tp_gnn_ta3d_two_pass
base_config=${framework}/local_bench/benchmark_inputs/${profile}/config.yaml

[[ -f "${profile_dir}/profile.yaml" ]] || { echo "missing validation profile: ${profile_dir}" >&2; exit 10; }
[[ -f "${base_config}" ]] || { echo "missing case config: ${base_config}" >&2; exit 11; }
[[ -d "${baseline_run}/00_input" && -d "${baseline_run}/10_floorplan" ]] || {
  echo "baseline run is missing frozen input/floorplan" >&2
  exit 12
}
[[ -d "${baseline_run}/20_partition/timing_input" ]] || {
  echo "baseline run is missing frozen timing input" >&2
  exit 13
}
[[ -f "${program}" ]] || { echo "missing TP-GNN program: ${program}" >&2; exit 14; }
[[ -z "$(git -C "${source_root}" status --short)" ]] || { echo "evaluation source is dirty" >&2; exit 15; }
if [[ -n "${reference_run}" ]]; then
  [[ -f "${reference_run}/60_opt/metrics.json" ]] || {
    echo "STA reference has no OPT metrics: ${reference_run}" >&2
    exit 16
  }
fi

mkdir -p "${output}"
command=(
  /opt/conda/bin/python3 -m local_bench.sta_cli evaluate tp_gnn_ta3d_two_pass
  --candidate "${program}"
  --open3dbench-root "${source_root}"
  --output-dir "${output}"
  --baseline-run "${baseline_run}"
  --base-config "${base_config}"
  --discard-workspace
)
if [[ -n "${reference_run}" ]]; then
  command+=(--reference-run "${reference_run}")
fi

docker run --rm --gpus "device=${gpu}" \
  -v "${experiment}:${experiment}" \
  -v "${dev}:${dev}:ro" \
  -v "/data/gcr/agentic3d-runs:/data/gcr/agentic3d-runs:ro" \
  -v "${runtime}:/runtime:ro" \
  -v "${dreamplace}:/dreamplace:ro" \
  -v "${collateral}:/collateral:ro" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PATH=/runtime/yosys-0.58/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e YOSYS_EXE=/runtime/yosys-0.58/bin/yosys \
  -e PYTHONPATH="${source_root}:${framework}:${framework}/.deps-py38:/dreamplace:/dreamplace/dreamplace" \
  -e OPEN3DBENCH_PYTHON=/opt/conda/bin/python3 \
  -e OPEN3D_LOCAL_DREAMPLACE_INSTALL=/dreamplace \
  -e PLACE3D_DREAMPLACE_INSTALL=/dreamplace \
  -e OPENROAD_EXE=/runtime/openroad-agentic3d \
  -e OPEN3D_LINK_FROZEN_UPSTREAM=1 \
  -e OPEN3D_OPT_PASSES=2 \
  -w "${framework}" \
  "${image}" "${command[@]}"

python3 - "${output}/evaluation_result.json" "${case_name}" <<'PY'
import json
import math
import sys

path, case = sys.argv[1:]
result = json.load(open(path, encoding="utf-8"))
metrics = result.get("metrics", {})
if float(metrics.get("runs_successfully", 0.0)) != 1.0:
    raise SystemExit("{} STA validation did not complete".format(case))
if int(float(metrics.get("opt_passes", 0.0))) != 2:
    raise SystemExit("{} STA validation did not run exactly two passes".format(case))
if float(metrics.get("sta_graph_complete", 0.0)) != 1.0:
    raise SystemExit("{} STA graph is incomplete".format(case))
if int(float(metrics.get("sta_unconstrained_endpoints", -1.0))) != 0:
    raise SystemExit("{} STA has unconstrained endpoints".format(case))
for key in ("final_global_wns_setup_ns", "final_global_tns_setup_ns"):
    if not math.isfinite(float(metrics[key])):
        raise SystemExit("{} {} is not finite".format(case, key))
print(
    "case={} fitness={:.12g} WNS={:.12g}ns TNS={:.12g}ns".format(
        case,
        float(metrics["combined_score"]),
        float(metrics["final_global_wns_setup_ns"]),
        float(metrics["final_global_tns_setup_ns"]),
    )
)
PY
