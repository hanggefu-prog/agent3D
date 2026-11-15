#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 SOURCE_ROOT BUILD_DIR THREADS" >&2
  exit 2
fi

source_root="$1"
build_dir="$2"
threads="$3"

if [[ ! -f "${source_root}/CMakeLists.txt" ]]; then
  echo "OpenROAD source root is incomplete: ${source_root}" >&2
  exit 2
fi

# Each candidate is built under RESULT/{workspace,runtime}.  Rebase both the
# source and generated-build include paths to RESULT so ccache keys do not
# contain the per-candidate evaluation directory.  The preprocessed source is
# still hashed, so only unchanged translation units can be shared.
candidate_result_dir="$(dirname "$(dirname "${build_dir}")")"
export CCACHE_BASEDIR="${candidate_result_dir}"
export CCACHE_NOHASHDIR=true

# Candidate evaluations use a fresh CMake/object tree, but the launcher mounts
# one experiment-owned CCACHE_DIR into every evaluator container.  Capture the
# shared namespace counters on both sides of this build so warm reuse is
# observable in each retained candidate result.  The snapshots are global
# namespace statistics (evaluations are serial in the OpenEvolve launcher),
# not exact per-translation-unit timings.
if [[ -n "${CCACHE_DIR:-}" ]]; then
  # Image builders also invoke this recipe with /candidate/build or /build.
  # Only the workspace evaluator has RESULT/runtime/BUILD; other callers keep
  # snapshots inside their writable build directory, never a parent mount.
  cache_stats_dir="${build_dir}"
  if [[ "$(basename "$(dirname "${build_dir}")")" == runtime ]]; then
    cache_stats_dir="${candidate_result_dir}"
  fi
  mkdir -p "${cache_stats_dir}"
  ccache -s > "${cache_stats_dir}/openroad-ccache-before.log" 2>&1
  capture_ccache_stats() {
    local status=$?
    trap - EXIT
    ccache -s > "${cache_stats_dir}/openroad-ccache-after.log" 2>&1 || true
    exit "${status}"
  }
  trap capture_ccache_stats EXIT
fi

cmake_args=(
  -DBUILD_GUI=OFF
  -DENABLE_TESTS=OFF
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
)
deps_file=/etc/openroad_deps_prefixes.txt
if [[ -f "${deps_file}" ]]; then
  # The maintained OpenROAD image publishes whitespace-separated -D options.
  # shellcheck disable=SC2207
  dependency_args=($(tr '\n' ' ' < "${deps_file}"))
  cmake_args+=("${dependency_args[@]}")
fi

cmake -S "${source_root}" -B "${build_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}" --target openroad -j "${threads}"
