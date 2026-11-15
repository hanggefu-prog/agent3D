#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 && $# -ne 4 ]]; then
  echo "usage: $0 REFERENCE_RUN CANDIDATE_RUN OUTPUT_DIR [LEVELS]" >&2
  exit 2
fi

reference_run=$1
candidate_run=$2
output_dir=$3
levels=${4:-p0,p1,p2}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
open3dbench_root=${OPEN3DBENCH_ROOT:-$repo_root}
runtime=${OPENAGENTIC3D_RUNTIME_ROOT:-/data/gcr/agentic3d-runs/runtime}
image=${OPENAGENTIC3D_EVAL_DOCKER_IMAGE:-agentic3d/compute-pool-runtime:20260908}
openroad=${OPEN3D_STA_OPENROAD_EXE:-$runtime/openroad}
hotspot_root=${OPEN3DBENCH_HOTSPOT_ROOT:-$open3dbench_root/OpenROAD-3D/flow/HotSpot}
hotspot_lib_dir=${OPENAGENTIC3D_HOTSPOT_LIB_DIR:-$runtime/hotspot-libs}
base_ld_library_path=${OPENAGENTIC3D_BASE_LD_LIBRARY_PATH:-/opt/legacy-libs}
threads=${OPENAGENTIC3D_THERMAL_THREADS:-32}

[[ -d "$reference_run" ]] || { echo "missing reference run: $reference_run" >&2; exit 2; }
[[ -d "$candidate_run" ]] || { echo "missing candidate run: $candidate_run" >&2; exit 2; }
[[ -d "$open3dbench_root/open3dbench" ]] || {
  echo "invalid Open3DBench root: $open3dbench_root" >&2
  exit 2
}
[[ -x "$openroad" ]] || { echo "missing OpenROAD executable: $openroad" >&2; exit 2; }
[[ -x "$hotspot_root/hotspot" ]] || { echo "missing HotSpot executable: $hotspot_root/hotspot" >&2; exit 2; }
[[ -f "$hotspot_lib_dir/libsuperlu.so.5" && -f "$hotspot_lib_dir/libblas.so.3" ]] || {
  echo "missing HotSpot sidecar libraries: $hotspot_lib_dir" >&2
  exit 2
}

for path in "$reference_run" "$candidate_run" "$output_dir" "$open3dbench_root" \
  "$runtime" "$openroad" "$hotspot_root" "$hotspot_lib_dir"; do
  [[ "$path" == /data/* ]] || {
    echo "the maintained remote wrapper accepts only /data paths: $path" >&2
    exit 2
  }
done

mkdir -p "$(dirname -- "$output_dir")"
exec docker run --rm --user "$(id -u):$(id -g)" \
  -v /data:/data \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e "PYTHONPATH=$open3dbench_root" \
  -e "LD_LIBRARY_PATH=$hotspot_lib_dir:$base_ld_library_path" \
  "$image" \
  /opt/conda/bin/python3.8 \
  "$open3dbench_root/AgenticBench/skills/open3dbench-thermal-evaluation/scripts/run_thermal_attribution.py" \
  --open3dbench-root "$open3dbench_root" \
  --reference-run "$reference_run" \
  --candidate-run "$candidate_run" \
  --output-dir "$output_dir" \
  --openroad "$openroad" \
  --hotspot-root "$hotspot_root" \
  --levels "$levels" \
  --threads "$threads" \
  --resume
