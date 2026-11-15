#!/usr/bin/env bash
# Thin, stage-by-stage launcher for the Open3DBench unified flow.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
framework_root="$(cd "${script_dir}/.." && pwd)"
default_source_root="$(cd "${framework_root}/.." && pwd)"
source_root="${OPEN3DBENCH_ROOT:-${default_source_root}}"
data_root="${AGENTICBENCH_DATA_ROOT:-${source_root}/user_runs}"
default_image="${OPEN3DBENCH_IMAGE:-${IMAGE:-shiyunqi/open3dbench-v2.0:release}}"

usage() {
  cat <<'EOF'
Usage:
  manual_flow.sh init <case> [work-dir]
  manual_flow.sh prepare|input|floorplan|timing_extract|partition|placement|pdn|hbt|opt|cts|global_routing|detailed_routing|routing|sta|hotspot|report|thermal <work-dir>
  manual_flow.sh global_routing|grt <work-dir> [new-output-dir]
  manual_flow.sh status <work-dir>

Environment used by init:
  AGENTICBENCH_GPU       GPU index saved for later stages (default: 0; use none for CPU only)
  MANUAL_FLOW_CONFIG     Optional source config; otherwise use the maintained TP-GNN/CTS case config
  OPEN3DBENCH_IMAGE      Release image saved for this run
  AGENTICBENCH_DATA_ROOT Default parent for work-dir

Optional limits used by each stage invocation (unset preserves prior behavior):
  MANUAL_FLOW_CPUS       Positive decimal Docker CPU quota, e.g. 4 or 2.5
  MANUAL_FLOW_MEMORY     Positive Docker memory size, e.g. 16g or 512m
                        Sets equal memory and memory-swap limits (no extra swap).
These limits do not alter algorithm thread settings in config.yaml. GPU selection
does not reserve a device; parallel workers must coordinate resource admission.
  AGENTICBENCH_ADMITTED_GPU Optional worker admission guard. When set, it must
                           match the GPU saved in the workspace (or selected at init).

Examples:
  AGENTICBENCH_GPU=0 manual_flow.sh init bp_fe /data/gcr/runs/manual-bp-fe
  manual_flow.sh prepare   /data/gcr/runs/manual-bp-fe
  manual_flow.sh timing_extract /data/gcr/runs/manual-bp-fe
  manual_flow.sh partition /data/gcr/runs/manual-bp-fe
  manual_flow.sh placement /data/gcr/runs/manual-bp-fe
  manual_flow.sh status    /data/gcr/runs/manual-bp-fe

`global_routing` (or `grt`) stops after FastRoute. On newer V2 sources, it uses
the standalone global-route command without changing the full routing status;
outputs default to 80_route/grt_metrics.json and global_route_checkpoint.json.
For a parallel inspection, explicitly choose a new output directory inside WORK.
Older split-stage sources retain their 75_global_route -> 80_route interface.
`routing` continues to run the source version's complete routing flow.
`sta` writes 90_report through OpenRCX and OpenSTA; `hotspot` writes
95_thermal and requires thermal.enabled: true in config.yaml.  `report` and
`thermal` remain compatibility names for the same two stages.
`timing_extract` is the explicit step 15 between floorplan and partition. It
writes the immutable TP-GNN/OpenSTA feature contract to 15_timing_extract;
`partition` consumes it and never launches timing extraction implicitly.

`init` creates an isolated, reproducible source snapshot at WORK/source.
After initialization, edit algorithm files under WORK/source (or create a new
workspace); later edits to the canonical repository are intentionally not
injected into an existing run.

The terminal shows stage/substep progress and warning samples. Complete tool
output stays in each stage's raw logs; metrics.json and figures/ expose the
results. Set OPEN3DBENCH_LOG_LEVEL=full to stream all direct tool output.
EOF
}

fail() {
  echo "manual_flow: $*" >&2
  exit 2
}

verify_admitted_gpu() {
  local actual_gpu="$1"
  [[ -n "${AGENTICBENCH_ADMITTED_GPU+x}" ]] || return 0
  [[ "${AGENTICBENCH_ADMITTED_GPU}" == none || "${AGENTICBENCH_ADMITTED_GPU}" =~ ^[0-9]+$ ]] \
    || fail "AGENTICBENCH_ADMITTED_GPU must be an index or none"
  [[ "${AGENTICBENCH_ADMITTED_GPU}" == "${actual_gpu}" ]] \
    || fail "GPU admission mismatch: worker admitted ${AGENTICBENCH_ADMITTED_GPU}, workspace selects ${actual_gpu}; use a matching worker GPU. Existing workspace settings were not changed."
}

absolute_parent_path() {
  local value="$1"
  local parent base
  parent="$(dirname "${value}")"
  base="$(basename "${value}")"
  mkdir -p "${parent}"
  parent="$(cd "${parent}" && pwd)"
  printf '%s/%s\n' "${parent}" "${base}"
}

require_workspace() {
  local requested="$1"
  [[ -d "${requested}" ]] || fail "work directory does not exist: ${requested}"
  work_dir="$(cd "${requested}" && pwd)"
  [[ -f "${work_dir}/.manual-flow" ]] || fail "not a manual-flow workspace: ${work_dir}"
  [[ -d "${work_dir}/source" ]] || fail "isolated source is missing: ${work_dir}/source"
  [[ -f "${work_dir}/config.yaml" ]] || fail "config is missing: ${work_dir}/config.yaml"
  [[ -f "${work_dir}/source-root.txt" ]] || fail "source-root record is missing"
  [[ -f "${work_dir}/image.txt" ]] || fail "image record is missing"
  [[ -f "${work_dir}/gpu.txt" ]] || fail "GPU record is missing"
}

initialize() {
  [[ $# -ge 1 && $# -le 2 ]] || fail "init expects <case> [work-dir]"
  local case_name="$1"
  local requested_work="${2:-${data_root}/manual-${case_name}-$(date +%Y%m%d-%H%M%S)}"
  local config_source gpu work

  [[ -d "${source_root}/open3dbench" ]] || fail "Open3DBench root is invalid: ${source_root}"
  [[ -d "${source_root}/AgenticBench" ]] || fail "AgenticBench is missing under: ${source_root}"

  config_source="${MANUAL_FLOW_CONFIG:-${source_root}/AgenticBench/local_bench/benchmark_inputs/${case_name}_lol_ng45_tp_gnn_cts/config.yaml}"
  [[ -f "${config_source}" ]] || fail "case config does not exist: ${config_source}"
  gpu="${AGENTICBENCH_GPU:-0}"
  [[ "${gpu}" == "none" || "${gpu}" =~ ^[0-9]+$ ]] || fail "AGENTICBENCH_GPU must be an index or none"
  verify_admitted_gpu "${gpu}"

  work="$(absolute_parent_path "${requested_work}")"
  [[ ! -e "${work}" ]] || fail "refusing to overwrite existing work directory: ${work}"
  mkdir -p "${work}/logs" "${work}/tmp/home" "${work}/tmp/matplotlib"

  python3 - \
    "${framework_root}/local_bench/workspace.py" \
    "${source_root}" \
    "${work}/source" <<'PY'
import importlib.util
import sys
from pathlib import Path

module_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("manual_flow_workspace", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load workspace helper: {}".format(module_path))
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)
workspace.isolated_clone(
    Path(sys.argv[2]), Path(sys.argv[3]), include_worktree=True
)
PY

  cp "${config_source}" "${work}/config.yaml"
  printf '1\n' > "${work}/.manual-flow"
  printf '%s\n' "${case_name}" > "${work}/case.txt"
  printf '%s\n' "${gpu}" > "${work}/gpu.txt"
  printf '%s\n' "${default_image}" > "${work}/image.txt"
  printf '%s\n' "$(cd "${source_root}" && pwd)" > "${work}/source-root.txt"

  cat <<EOF
Initialized manual Open3DBench run:
  case:       ${case_name}
  workspace:  ${work}
  config:     ${work}/config.yaml
  source:     ${work}/source
  results:    ${work}/run
  GPU:        ${gpu}
  image:      ${default_image}

Next:
  ${BASH_SOURCE[0]} prepare ${work}
  ${BASH_SOURCE[0]} timing_extract ${work}
  ${BASH_SOURCE[0]} partition ${work}

This workspace is isolated.  For this run, edit TP-GNN parameters here:
  ${work}/source/Place-3D/config/partition/tp_gnn.json
EOF
}

stage_bounds() {
  case "$1" in
    prepare) printf 'input floorplan\n' ;;
    input|floorplan|timing_extract|partition|placement|pdn|hbt|opt|cts|global_routing|detailed_routing|routing|report|thermal)
      printf '%s %s\n' "$1" "$1"
      ;;
    grt)
      printf 'global_routing global_routing\n'
      ;;
    sta)
      printf 'sta sta\n'
      ;;
    hotspot)
      printf 'hotspot hotspot\n'
      ;;
    *) fail "unknown stage: $1" ;;
  esac
}

run_stage() {
  [[ $# -ge 2 && $# -le 3 ]] || fail "$1 expects <work-dir> [new-output-dir for GRT]"
  [[ "${USE_LOCAL_OPENROAD:-0}" == "0" ]] \
    || fail "USE_LOCAL_OPENROAD is unsupported: use tools installed in the pinned Docker image; rebuild the image for OpenROAD/OpenSTA changes"
  local stage="$1"
  require_workspace "$2"
  local grt_output="${3:-}"
  if [[ $# -eq 3 ]]; then
    [[ "${stage}" == "global_routing" || "${stage}" == "grt" ]] \
      || fail "a separate output directory is supported only for GRT"
    grt_output="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${grt_output}")"
    [[ "${grt_output}" == "${work_dir}/"* ]] \
      || fail "GRT output must be inside this workspace's Docker mount: ${work_dir}"
  fi

  local from_stage to_stage canonical image image_id gpu uid_gid log_path
  local cpu_limit="${MANUAL_FLOW_CPUS:-}" memory_limit="${MANUAL_FLOW_MEMORY:-}"
  if [[ -n "${MANUAL_FLOW_CPUS+x}" ]]; then
    [[ "${cpu_limit}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
      && [[ -n "${cpu_limit//[.0]/}" ]] \
      || fail "MANUAL_FLOW_CPUS must be a positive decimal number"
  fi
  if [[ -n "${MANUAL_FLOW_MEMORY+x}" ]]; then
    [[ "${memory_limit}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)([kKmMgGtTpP]([iI]?[bB])?|[bB])?$ ]] \
      || fail "MANUAL_FLOW_MEMORY must be a positive Docker size such as 16g"
    [[ -n "${BASH_REMATCH[1]//[.0]/}" ]] \
      || fail "MANUAL_FLOW_MEMORY must be greater than zero"
  fi
  read -r from_stage to_stage < <(stage_bounds "${stage}")
  canonical="$(<"${work_dir}/source-root.txt")"
  image="$(<"${work_dir}/image.txt")"
  gpu="$(<"${work_dir}/gpu.txt")"
  verify_admitted_gpu "${gpu}"
  [[ -d "${canonical}" ]] || fail "recorded source root is unavailable: ${canonical}"
  command -v docker >/dev/null 2>&1 || fail "docker is not installed"
  image_id="$(docker image inspect --format '{{.Id}}' "${image}" 2>/dev/null)" \
    || fail "Docker image is unavailable: ${image}"
  uid_gid="$(id -u):$(id -g)"
  log_path="${work_dir}/logs/${stage}.log"
  if [[ -f "${log_path}" ]]; then
    local previous_log="${log_path%.log}.previous.$(date +%Y%m%d-%H%M%S).$$.log"
    mv "${log_path}" "${previous_log}"
  fi

  local -a docker_args
  docker_args=(
    --rm
    --user "${uid_gid}"
    --shm-size "${MANUAL_FLOW_SHM_SIZE:-8g}"
    --network none
    -e "HOME=${work_dir}/tmp/home"
    -e "TMPDIR=${work_dir}/tmp"
    -e "MPLCONFIGDIR=${work_dir}/tmp/matplotlib"
    -e PYTHONDONTWRITEBYTECODE=1
    -e PYTHONUNBUFFERED=1
    -e "OPEN3DBENCH_LOG_LEVEL=${OPEN3DBENCH_LOG_LEVEL:-summary}"
    -e "OPEN3DBENCH_IMAGE=${image}"
    -e "OPEN3DBENCH_IMAGE_ID=${image_id}"
    -v "${canonical}:${canonical}:ro"
    -v "${work_dir}:${work_dir}:rw"
    -w "${work_dir}/source"
  )
  if [[ "${gpu}" != "none" ]]; then
    docker_args+=(--gpus "device=${gpu}")
  fi
  if [[ -n "${cpu_limit}" ]]; then
    docker_args+=(--cpus "${cpu_limit}")
  fi
  if [[ -n "${memory_limit}" ]]; then
    docker_args+=(--memory "${memory_limit}" --memory-swap "${memory_limit}")
  fi
  echo "Running ${stage}: ${from_stage} -> ${to_stage}"
  echo "Isolated source: ${work_dir}/source"
  if [[ "${stage}" == "partition" ]]; then
    echo "Default TP-GNN config for this workspace: ${work_dir}/source/Place-3D/config/partition/tp_gnn.json"
    echo "The runner will print the resolved config path and all resolved parameters below."
  fi
  echo "Log: ${log_path}"
  docker run "${docker_args[@]}" "${image}" bash -c '
    export PYTHONPATH="$1:$1/AgenticBench:${PYTHONPATH:-}"
    if [[ "$6" == "global_routing" || "$6" == "grt" ]]; then
      if python3 -m open3dbench --help | grep -q -- "global-route"; then
        args=(--repo "$1" global-route "$2" --run-dir "$5")
        if [[ -n "$7" ]]; then args+=(--output-dir "$7"); fi
        exec python3 -m open3dbench "${args[@]}"
      elif [[ -n "$7" ]]; then
        echo "This source version cannot run standalone GRT with a separate output directory" >&2
        exit 2
      fi
    fi
    exec python3 -m open3dbench --repo "$1" run "$2" \
      --from "$3" --to "$4" --resume --run-dir "$5"
  ' bash \
    "${work_dir}/source" \
    "${work_dir}/config.yaml" \
    "${from_stage}" \
    "${to_stage}" \
    "${work_dir}/run" \
    "${stage}" \
    "${grt_output}" \
    2>&1 | tee "${log_path}"
}

show_status() {
  [[ $# -eq 1 ]] || fail "status expects <work-dir>"
  require_workspace "$1"
  local manifest="${work_dir}/run/run.json"
  if [[ ! -f "${manifest}" ]]; then
    echo "initialized; no Open3DBench stage has run"
    echo "workspace: ${work_dir}"
    exit 0
  fi
  python3 - "${manifest}" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
print("run:    {}".format(path))
print("design: {}".format(data.get("design", "unknown")))
print("status: {}".format(data.get("status", "unknown")))
for name, row in data.get("stages", {}).items():
    status = row.get("status", "unknown") if isinstance(row, dict) else str(row)
    runtime = row.get("runtime_seconds") if isinstance(row, dict) else None
    suffix = " ({:.1f}s)".format(runtime) if isinstance(runtime, (int, float)) else ""
    print("  {:10s} {}{}".format(name, status, suffix))
PY
}

command_name="${1:-help}"
case "${command_name}" in
  -h|--help|help)
    usage
    ;;
  init)
    shift
    initialize "$@"
    ;;
  status)
    shift
    show_status "$@"
    ;;
  prepare|input|floorplan|timing_extract|partition|placement|pdn|hbt|opt|cts|global_routing|detailed_routing|routing|sta|hotspot|report|thermal|grt)
    shift
    run_stage "${command_name}" "$@"
    ;;
  *)
    usage >&2
    fail "unknown command: ${command_name}"
    ;;
esac
