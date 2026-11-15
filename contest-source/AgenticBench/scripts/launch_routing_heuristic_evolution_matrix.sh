#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root="$1"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
worker="${script_dir}/run_routing_heuristic_evolution_case.sh"
mkdir -p "${run_root}/launch"

for case_name in bp ariane133; do
  for heuristic in ripup maze iteration; do
    name="${case_name}-${heuristic}"
    log="${run_root}/launch/${name}.log"
    nohup "${worker}" "${run_root}" "${case_name}" "${heuristic}" \
      >"${log}" 2>&1 &
    pid=$!
    echo "${pid}" >"${run_root}/launch/${name}.pid"
    echo "launched ${name}: pid=${pid} log=${log}"
  done
done
