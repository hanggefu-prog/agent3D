#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 && $# -ne 8 ]]; then
  echo "usage: $0 CANDIDATE OUTPUT INPUT_KIND INPUT REFERENCE_OR_DASH GPU RETENTION [PROFILE_CONFIG]" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
common="$script_dir/../../open3dbench-sta-evaluation/scripts/run_remote_sta.sh"
target=${OPENAGENTIC3D_THERMAL_PROFILE:-tp_gnn_placement_cts_thermal}
case "$target" in tp_gnn_placement_cts_thermal|tp_gnn_placement_thermal) ;;
  *) echo "unsupported thermal profile: $target" >&2; exit 2 ;;
esac
exec "$common" "$target" "$@"
