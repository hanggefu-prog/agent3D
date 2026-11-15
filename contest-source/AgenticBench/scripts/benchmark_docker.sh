#!/usr/bin/env bash
# Run the task or evaluator-owned STA CLI inside the release environment.
set -euo pipefail
framework="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="${OPEN3DBENCH_ROOT:-$(cd "${framework}/.." && pwd)}"
data_root="${AGENTICBENCH_DATA_ROOT:-${source_root}/user_runs}"
image="${IMAGE:-shiyunqi/open3dbench-v2.0:release}"
gpu="${AGENTICBENCH_GPU:-0}"
if [[ $# -eq 0 ]]; then
  echo "Usage: benchmark_docker.sh [profile|sta] <CLI arguments>" >&2
  exit 2
fi
if [[ "${USE_LOCAL_OPENROAD:-0}" != 0 ]]; then
  echo "USE_LOCAL_OPENROAD is unsupported: use tools installed in the pinned Docker image; rebuild the image for OpenROAD/OpenSTA changes." >&2
  exit 2
fi
module=local_bench
if [[ "$1" == profile ]]; then
  module=local_bench.profile_cli
  shift
  [[ $# -gt 0 ]] || { echo "evaluation profile CLI arguments are required" >&2; exit 2; }
elif [[ "$1" == sta ]]; then
  module=local_bench.sta_cli
  shift
  [[ $# -gt 0 ]] || { echo "STA CLI arguments are required" >&2; exit 2; }
fi
mkdir -p "${data_root}"
data_root="$(cd "${data_root}" && pwd)"
args=(--rm --user "$(id -u):$(id -g)"
  -v "${source_root}:${source_root}" -v "${data_root}:${data_root}"
  -w "${framework}" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1
  -e OPEN3DBENCH_IMAGE="${image}"
  -e OPEN3DBENCH_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${image}")"
  -e OPEN3D_LOCAL_OPENROAD_BUILDER="${framework}/scripts/build_openroad_candidate.sh")
if [[ "${gpu}" != none ]]; then args+=(--gpus "device=${gpu}"); fi
exec docker run "${args[@]}" "${image}" bash -c '
  export PYTHONPATH="$1:$2:${PYTHONPATH:-}"
  export OPEN3DBENCH_PYTHON="$(command -v python3)"
  module=$3
  shift 3
  exec python3 -m "$module" "$@"
' bash "${framework}" "${source_root}" "${module}" "$@"
