#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 && $# -ne 9 ]]; then
  echo "usage: $0 TASK_OR_PROFILE CANDIDATE OUTPUT INPUT_KIND INPUT REFERENCE_OR_DASH GPU RETENTION [PROFILE_CONFIG]" >&2
  exit 2
fi

target=$1
candidate=$2
output=$3
input_kind=$4
input_path=$5
reference=$6
gpu=$7
retention=$8

profile=""
case "$target" in
  tp_gnn_placement_sta|tp_gnn_placement_two_pass) profile=tp_gnn_placement_two_pass ;;
  tp_gnn_placement_thermal) profile=tp_gnn_placement_thermal ;;
  tp_gnn_placement_cts_thermal) profile=tp_gnn_placement_cts_thermal ;;
  placement_two_pass_sta) profile=placement_two_pass ;;
  tp_gnn_sta_inner|tp_gnn_partition_proxy) profile=tp_gnn_partition_proxy ;;
  placement_sta_inner|placement_proxy) profile=placement_proxy ;;
  *) echo "unsupported STA task or profile: $target" >&2; exit 2 ;;
esac
case "$input_kind" in bundle|run) ;; *) echo "INPUT_KIND must be bundle or run" >&2; exit 2 ;; esac
case "$gpu" in [0-7]) ;; *) echo "GPU must be between 0 and 7" >&2; exit 2 ;; esac
case "$retention" in full|compact) ;; *) echo "RETENTION must be full or compact" >&2; exit 2 ;; esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
framework=${OPENAGENTIC3D_FRAMEWORK_ROOT:-$repo_root}
source_root=${OPEN3DBENCH_ROOT:-/data/gcr/agentic3d-dev/open3dbench-source-sta-codesign-20260902-v1}
runtime=${OPENAGENTIC3D_RUNTIME_ROOT:-/data/gcr/agentic3d-runs/runtime}
dreamplace=${OPENAGENTIC3D_DREAMPLACE_INSTALL:-$runtime/dreamplace-overlay}
image=${OPENAGENTIC3D_EVAL_DOCKER_IMAGE:-agentic3d/open3dbench-v2-runtime:offline}
sta_openroad=${OPEN3D_STA_OPENROAD_EXE:-/data/gcr/agentic3d-dev/docker-sta-20260902/docker/runtime-refresh/openroad}
sta_contract=$runtime/check_opensta3d_command.tcl
profile_config=${9:-$framework/local_bench/benchmark_inputs/bp_multi_lol_ng45_tp_gnn_sta/config.yaml}
container_name="sta-eval-${target//_/-}-$(basename "$output")-$$"
evaluation_timeout=21600
container_runtime_env=()
if [[ "$target" == "tp_gnn_placement_thermal" || "$target" == "tp_gnn_placement_cts_thermal" ]]; then
  if [[ "$target" == "tp_gnn_placement_thermal" ]]; then evaluation_timeout=86400; fi
  hotspot_lib_dir=${OPENAGENTIC3D_HOTSPOT_LIB_DIR:-$runtime/hotspot-libs}
  [[ -f "$hotspot_lib_dir/libsuperlu.so.5" && -f "$hotspot_lib_dir/libblas.so.3" ]] || {
    echo "HotSpot runtime libraries are missing: $hotspot_lib_dir" >&2
    exit 2
  }
  base_ld_library_path=${OPENAGENTIC3D_BASE_LD_LIBRARY_PATH:-/opt/legacy-libs}
  container_runtime_env+=(-e "LD_LIBRARY_PATH=$hotspot_lib_dir:$base_ld_library_path")
fi

mounts=()
[[ ! -d /data ]] || mounts+=(-v /data:/data)
if [[ "$framework $source_root $runtime $dreamplace $candidate $output $input_path $reference $profile_config" == *"/nas1/gcr/"* ]]; then
  mounts+=(-v /nas1/gcr:/nas1/gcr)
fi
if [[ -n "${OPENAGENTIC3D_NODE_ROOT:-}" ]]; then
  [[ "$OPENAGENTIC3D_NODE_ROOT" == /data/agentic3d/agentic3d-* || "$OPENAGENTIC3D_NODE_ROOT" == /data/gcr/agentic3d-* ]] || exit 2
  mounts+=(-v "$OPENAGENTIC3D_NODE_ROOT:$OPENAGENTIC3D_NODE_ROOT")
fi
if [[ -n "${OPENAGENTIC3D_STA_DATA_ROOT:-}" ]]; then
  [[ -d "$OPENAGENTIC3D_STA_DATA_ROOT/agentic3d-dev" && -d "$OPENAGENTIC3D_STA_DATA_ROOT/agentic3d-runs" ]] || exit 2
  mounts+=(-v "$OPENAGENTIC3D_STA_DATA_ROOT/agentic3d-dev:/data/gcr/agentic3d-dev:ro"
           -v "$OPENAGENTIC3D_STA_DATA_ROOT/agentic3d-runs:/data/gcr/agentic3d-runs:ro")
fi
if [[ -n "${OPENAGENTIC3D_EVAL_GPU_LOCK_DIR:-}" ]]; then
  mkdir -p "$OPENAGENTIC3D_EVAL_GPU_LOCK_DIR"
  exec 9>"$OPENAGENTIC3D_EVAL_GPU_LOCK_DIR/gpu-$gpu.lock"
  flock 9
fi

[[ -f "$profile_config" ]] || { echo "missing profile config: $profile_config" >&2; exit 2; }
[[ -x "$sta_openroad" ]] || { echo "STA OpenROAD binary is missing or not executable: $sta_openroad" >&2; exit 2; }
[[ -f "$sta_contract" ]] || { echo "STA OpenROAD contract check is missing: $sta_contract" >&2; exit 2; }

if ! sta_contract_output=$(docker run --rm \
  "${mounts[@]}" \
  -e HOME=/tmp \
  "$image" \
  "$sta_openroad" -no_init -exit "$sta_contract" 2>&1); then
  echo "$sta_contract_output" >&2
  echo "STA OpenROAD contract check failed: $sta_openroad" >&2
  exit 2
fi
case "$sta_contract_output" in
  *"get_parasitic_cutpoint_timing=get_parasitic_cutpoint_timing"*) ;;
  *)
    echo "$sta_contract_output" >&2
    echo "STA OpenROAD lacks get_parasitic_cutpoint_timing: $sta_openroad" >&2
    exit 2
    ;;
esac

command=(/opt/conda/bin/python3.8 -m local_bench.profile_cli evaluate "$profile")
command+=(
  --candidate "$candidate"
  --open3dbench-root "$source_root"
  --output-dir "$output"
  --base-config "$profile_config"
  --timeout "$evaluation_timeout"
)
if [[ "$input_kind" == bundle ]]; then
  command+=(--benchmark-input "$input_path")
else
  command+=(--baseline-run "$input_path")
fi
if [[ "$reference" != "-" ]]; then
  command+=(--reference-run "$reference")
fi
if [[ "$retention" == compact ]]; then
  command+=(--compact-artifacts)
else
  command+=(--discard-workspace --discard-runtime)
fi

docker run --rm --gpus "device=$gpu" --name "$container_name" \
  "${mounts[@]}" \
  "${container_runtime_env[@]}" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e OPEN3DBENCH_PYTHON=/opt/conda/bin/python3.8 \
  -e OPEN3D_LINK_FROZEN_UPSTREAM=1 \
  -e OPEN3D_OPT_PASSES=2 \
  -e OPEN3D_LOCAL_DREAMPLACE_INSTALL="$dreamplace" \
  -e PLACE3D_DREAMPLACE_INSTALL="$dreamplace" \
  -e OPENROAD_EXE="$sta_openroad" \
  -e YOSYS_EXE="$runtime/yosys-0.58/bin/yosys" \
  -e PYTHONPATH="$framework:$framework/.deps-py38:$source_root:$dreamplace:$dreamplace/dreamplace" \
  "$image" \
  "${command[@]}"
