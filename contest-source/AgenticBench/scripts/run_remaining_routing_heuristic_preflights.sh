#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root="$1"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
worker="${script_dir}/run_routing_heuristic_preflight.sh"

if docker ps --format '{{.Names}}' | grep -Fxq grt-preflight-bp-ripup; then
  code=$(docker wait grt-preflight-bp-ripup)
  if [[ "${code}" -ne 0 ]]; then
    echo "rip-up preflight exited ${code}" >&2
    exit "${code}"
  fi
fi

"${worker}" "${run_root}" bp maze \
  >"${run_root}/preflight/bp-maze-full.wrapper.log" 2>&1
"${worker}" "${run_root}" bp iteration \
  >"${run_root}/preflight/bp-iteration-full.wrapper.log" 2>&1
