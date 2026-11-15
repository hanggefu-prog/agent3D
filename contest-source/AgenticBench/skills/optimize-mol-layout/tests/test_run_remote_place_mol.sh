#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
helper=$(cd "$script_dir/../scripts" && pwd)/run_remote_place_mol.sh
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/place-mol-remote-test.XXXXXX")
run_id=bp-fe-isolation-test
launcher=/tmp/agentic3d_place_mol_${run_id}.sh

cleanup() {
  chmod -R u+rwX "$tmp_dir" 2>/dev/null || true
  rm -rf "$tmp_dir"
  rm -f "$launcher"
  rm -f "${readonly_launcher:-}"
}
trap cleanup EXIT

source_root="$tmp_dir/shared-place-mol"
benchmark_root="$tmp_dir/reconstructed-benchmark"
work_root="$tmp_dir/work"
fake_bin="$tmp_dir/bin"
mkdir -p \
  "$source_root/src/place_3d" \
  "$source_root/config" \
  "$benchmark_root/nangate45" \
  "$benchmark_root/or_3D/bp_fe" \
  "$fake_bin"
printf '%s\n' 'print("fixture")' > "$source_root/src/place_3d/main.py"
printf '%s\n' '{"method":"mem-on-logic"}' > "$source_root/config/or_3D_bp_fe.json"
printf '%s\n' 'immutable-source' > "$source_root/source-marker"
printf '%s\n' 'immutable-benchmark' > "$benchmark_root/or_3D/bp_fe/input-marker"
chmod -R a-w "$benchmark_root"

cat > "$fake_bin/ssh" <<'FAKE_SSH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-tt" ]]; then
  exit 0
fi
shift
if [[ "${1:-}" == "bash" && "${2:-}" == "-s" && "${3:-}" == "--" ]]; then
  shift 3
  exec bash -s -- "$@"
fi
# The post-run probe is not relevant to this preparation-only harness.
exit 0
FAKE_SSH
chmod +x "$fake_bin/ssh"

source_before=$(cksum "$source_root/source-marker")
benchmark_before=$(cksum "$benchmark_root/or_3D/bp_fe/input-marker")

PATH="$fake_bin:$PATH" \
  "$helper" \
    --remote fake-host \
    --source-root "$source_root" \
    --work-root "$work_root" \
    --run-id "$run_id" \
    --design bp_fe \
    --config-path "$source_root/config/or_3D_bp_fe.json" \
    --benchmark-root "$benchmark_root" \
    --gpu 2 \
    --seed 3 \
    --prepare-only

run_dir="$work_root/$run_id"
test -f "$run_dir/source/src/place_3d/main.py"
test -f "$run_dir/benchmark-work/or_3D/bp_fe/input-marker"
test -f "$run_dir/config.json"
: > "$run_dir/benchmark-work/or_3D/bp_fe/macro_fixed.def"
test -w "$run_dir/benchmark-work/or_3D/bp_fe/macro_fixed.def"

test "$source_before" = "$(cksum "$source_root/source-marker")"
test "$benchmark_before" = "$(cksum "$benchmark_root/or_3D/bp_fe/input-marker")"
test ! -e "$benchmark_root/or_3D/bp_fe/macro_fixed.def"

grep -Fq -- "-v \"$run_dir/source:/workspace\"" "$launcher"
grep -Fq -- "-v \"$run_dir/benchmark-work:/workspace/benchmarks\"" "$launcher"
! grep -Fq -- "-v \"$source_root:/workspace\"" "$launcher"
! grep -Fq -- "-v \"$benchmark_root:/workspace/benchmarks:ro\"" "$launcher"
grep -Fq -- '-e CUDA_VISIBLE_DEVICES="2"' "$launcher"
grep -Fq -- '-e PYTHONUNBUFFERED=1' "$launcher"
grep -Fq -- 'python -u src/place_3d/main.py' "$launcher"
grep -Fq -- '--benchmark="bp_fe" --seed="3"' "$launcher"

readonly_run_id=bp-fe-readonly-source-test
readonly_launcher=/tmp/agentic3d_place_mol_${readonly_run_id}.sh
PATH="$fake_bin:$PATH" \
  "$helper" \
    --remote fake-host \
    --source-root "$source_root" \
    --work-root "$work_root" \
    --run-id "$readonly_run_id" \
    --design bp_fe \
    --config-path "$source_root/config/or_3D_bp_fe.json" \
    --benchmark-root "$benchmark_root" \
    --gpu 2 \
    --seed 3 \
    --prepare-only \
    --read-only-source

test ! -e "$work_root/$readonly_run_id/source"
grep -Fq -- "-v \"$source_root:/workspace:ro\"" "$readonly_launcher"
grep -Fq -- '-e PYTHONDONTWRITEBYTECODE=1' "$readonly_launcher"
grep -Fq -- '-e PYTHONUNBUFFERED=1' "$readonly_launcher"
grep -Fq -- 'python -u src/place_3d/main.py' "$readonly_launcher"

echo "PASS: per-run writable benchmark isolation and optional read-only source"
