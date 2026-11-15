#!/usr/bin/env bash
# New TP-GNN/Concurrent baseline only. Historical runs retain their frozen launchers.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 RUN_ROOT CASE ripup|maze|iteration [--check-only]" >&2
  exit 2
fi
run_root="$1"
case_name="$2"
heuristic="$3"
mode="${4:-run}"
[[ "$mode" == run || "$mode" == --check-only ]] || { echo "unknown mode: $mode" >&2; exit 2; }
[[ "$run_root" == /* && -d "$run_root" ]] || { echo "RUN_ROOT must be an existing absolute path" >&2; exit 2; }
[[ "$case_name" =~ ^[a-zA-Z0-9_]+$ ]] || { echo "invalid case name" >&2; exit 2; }
framework="${OPENAGENTIC3D_FRAMEWORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source_root="${OPEN3DBENCH_SOURCE:-$(cd "${framework}/.." && pwd)}"
image="${OPEN3DBENCH_IMAGE:-shiyunqi/open3dbench-v2.0:release}"
benchmark_input="${OPENAGENTIC3D_BENCHMARK_INPUT:-${run_root}/inputs/${case_name}/benchmark_input.json}"
reference_run="${OPEN3DBENCH_REFERENCE_RUN:-${run_root}/references/${case_name}/run}"
profile="${framework}/local_bench/benchmark_inputs/${case_name}_lol_ng45_tp_gnn_cts"
case "$heuristic" in
  ripup) task=global_routing_ripup_policy ;;
  maze) task=global_routing_maze_cost ;;
  iteration) task=global_routing_iteration_controller ;;
  *) echo "unknown routing heuristic: $heuristic" >&2; exit 2 ;;
esac
route_metrics="${reference_run}/75_global_route/metrics.json"
[[ -f "${route_metrics}" ]] || route_metrics="${reference_run}/80_route/metrics.json"
for required in "${profile}/profile.yaml" "${profile}/config.yaml" \
  "${framework}/local_bench/routing_baseline.py" "${source_root}/open3dbench/__main__.py" \
  "${benchmark_input}" "${route_metrics}"; do
  [[ -f "$required" ]] || { echo "missing input: $required; prepare a new TP-GNN CTS input/reference" >&2; exit 2; }
done
output="${run_root}/evolution/${case_name}/${heuristic}"
preflight="${run_root}/preflight/${case_name}/${heuristic}"
scratch="${OPEN3D_SCRATCH_HOST:-${run_root}/scratch}"
args=(--rm --user "$(id -u):$(id -g)" --cpus 32 --cpu-shares 256 --memory 96g --shm-size 8g
  --log-opt max-size=10m --log-opt max-file=2
  -v "${source_root}:${source_root}:ro" -v "${framework}:${framework}:ro"
  -v "${run_root}:${run_root}" -v "$(dirname "${benchmark_input}"):$(dirname "${benchmark_input}"):ro"
  -v "${reference_run}:${reference_run}:ro"
  -w "$framework" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1
  -e OPEN3DBENCH_IMAGE="$image" -e OPEN3DBENCH_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$image")")
secret=""
evostage=""
if [[ "$mode" == --check-only ]]; then
  args+=(--network none)
else
  : "${EVOSTAGE_ROOT:?Set the external EvoStage checkout explicitly}"
  : "${DEEPSEEK_ENV_FILE:?Set a private shell-format DeepSeek environment file explicitly}"
  evostage="$EVOSTAGE_ROOT"
  secret="$DEEPSEEK_ENV_FILE"
  [[ -d "$evostage" && -f "$secret" ]] || { echo "EvoStage/credential path missing" >&2; exit 2; }
  [[ ! -e "$output" && ! -e "$preflight" ]] || { echo "output/preflight exists; use a fresh run root" >&2; exit 2; }
  args+=(--network host -v "${evostage}:${evostage}:ro" -v "${secret}:${secret}:ro")
fi
mkdir -p "${scratch}/tmp" "${scratch}/ccache"
args+=(-v "${scratch}:/scratch" -v "${scratch}/tmp:/tmp")
exec docker run "${args[@]}" "$image" bash -c '
  set -euo pipefail
  framework=$1 source_root=$2 benchmark_input=$3 reference_run=$4 case_name=$5 task=$6
  output=$7 preflight=$8 mode=$9 secret=${10} evostage=${11} profile=${12}
  export PYTHONPATH="$framework:$source_root:/opt/dreamplace:/opt/dreamplace/dreamplace:${PYTHONPATH:-}"
  export OPEN3DBENCH_PYTHON="$(command -v python3)"
  export OPEN3D_LOCAL_OPENROAD_BUILDER="$framework/scripts/build_openroad_candidate.sh"
  export OPEN3D_LOCAL_OPENROAD_BUILD_THREADS=8 OPEN3D_LINK_FROZEN_UPSTREAM=1
  export OPEN3D_EVALUATION_SCRATCH_ROOT=/scratch TMPDIR=/tmp MPLCONFIGDIR=/tmp/matplotlib
  export CCACHE_DIR=/scratch/ccache CCACHE_MAXSIZE=24G
  checks=(--case "$case_name" --task "$task" --benchmark-input "$benchmark_input" --reference-run "$reference_run")
  python3 -m local_bench.routing_baseline "${checks[@]}"
  [[ "$mode" != --check-only ]] || exit 0
  # Evaluate the unchanged task policy before loading any model credentials.
  python3 -m local_bench evaluate "$task" --open3dbench-root "$source_root" \
    --benchmark-input "$benchmark_input" --base-config "$profile/config.yaml" \
    --reference-run "$reference_run" --output-dir "$preflight" --compact-artifacts --timeout 21600
  python3 -m local_bench.routing_baseline "${checks[@]}" --identity-result "$preflight/evaluation_result.json"
  set -a
  source "$secret"
  set +a
  exec python3 -m local_bench.evolve_cli run "$task" --method evostage --provider deepseek \
    --evostage-root "$evostage" --evaluation-mode full --open3dbench-root "$source_root" \
    --benchmark-input "$benchmark_input" --reference-run "$reference_run" --output-dir "$output" \
    --max-samples 50 --max-generations 50 --population-size 5 --compact-evaluations --timeout 21600
' bash "$framework" "$source_root" "$benchmark_input" "$reference_run" "$case_name" "$task" \
  "$output" "$preflight" "$mode" "$secret" "$evostage" "$profile"
