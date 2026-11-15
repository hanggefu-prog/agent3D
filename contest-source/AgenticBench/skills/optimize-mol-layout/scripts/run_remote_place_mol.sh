#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-}
SOURCE_ROOT=${SOURCE_ROOT:-}
WORK_ROOT=${WORK_ROOT:-}
IMAGE=${IMAGE:-shiyunqi/open3dbench:place}
RUN_ID=
DESIGN=
CONFIG_PATH=
BENCHMARK_ROOT=
GPU=0
SEED=3
COMPILE=0
PREPARE_ONLY=0
READ_ONLY_SOURCE=0

usage() {
  echo "Usage: $0 --run-id ID --design NAME --config-path REMOTE_JSON --benchmark-root REMOTE_DIR [options]"
  echo "Options: --remote HOST --source-root DIR --work-root DIR --image IMAGE --gpu N --seed N --compile"
  echo "         --prepare-only --read-only-source"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE=$2; shift 2 ;;
    --source-root) SOURCE_ROOT=$2; shift 2 ;;
    --work-root) WORK_ROOT=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --run-id) RUN_ID=$2; shift 2 ;;
    --design) DESIGN=$2; shift 2 ;;
    --config-path) CONFIG_PATH=$2; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --compile) COMPILE=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --read-only-source) READ_ONLY_SOURCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$REMOTE" || -z "$SOURCE_ROOT" || -z "$WORK_ROOT" || -z "$RUN_ID" || -z "$DESIGN" || -z "$CONFIG_PATH" || -z "$BENCHMARK_ROOT" ]]; then
  echo "--remote, --source-root, and --work-root are required unless their environment variables are set" >&2
  usage >&2
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run id may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ "$READ_ONLY_SOURCE" == "1" && "$COMPILE" == "1" ]]; then
  echo "--read-only-source cannot be combined with --compile" >&2
  exit 2
fi
if [[ "$PREPARE_ONLY" != "1" && -z "${OPEN3DBENCH_SUDO_PASSWORD:-}" ]]; then
  echo "OPEN3DBENCH_SUDO_PASSWORD is required for remote Docker access" >&2
  exit 2
fi

RUN_DIR="$WORK_ROOT/$RUN_ID"
REMOTE_LAUNCHER="/tmp/agentic3d_place_mol_${RUN_ID}.sh"

ssh "$REMOTE" bash -s -- \
  "$SOURCE_ROOT" "$RUN_DIR" "$CONFIG_PATH" "$BENCHMARK_ROOT" "$REMOTE_LAUNCHER" \
  "$IMAGE" "$DESIGN" "$GPU" "$SEED" "$COMPILE" "$READ_ONLY_SOURCE" <<'REMOTE_PREP'
set -euo pipefail
source_root=$1
run_dir=$2
config_path=$3
benchmark_root=$4
launcher=$5
image=$6
design=$7
gpu=$8
seed=$9
compile=${10}
read_only_source=${11}

test -f "$source_root/src/place_3d/main.py"
test -f "$config_path"
test -d "$benchmark_root/nangate45"
test -d "$benchmark_root/or_3D/$design"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing run directory: $run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir"
cp "$config_path" "$run_dir/config.json"
source_snapshot="$run_dir/source"
benchmark_work="$run_dir/benchmark-work"
mkdir -p "$benchmark_work"
if [[ "$read_only_source" == "1" ]]; then
  source_snapshot=$source_root
  source_mount_suffix=:ro
else
  mkdir -p "$source_snapshot"
  cp -a "$source_root/." "$source_snapshot/"
  chmod -R u+rwX "$source_snapshot"
  source_mount_suffix=
fi
cp -a "$benchmark_root/." "$benchmark_work/"
# The shared benchmark is immutable provenance. Only its per-run copy may be
# changed by Step 2 when it emits or_3D/<design>/macro_fixed.def.
chmod -R u+rwX "$benchmark_work"
host_uid=$(id -u)
host_gid=$(id -g)

cat > "$launcher" <<SH
#!/usr/bin/env bash
set -euo pipefail
exec docker run --rm --gpus all \
  --user "$host_uid:$host_gid" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_VISIBLE_DEVICES="$gpu" \
  -e PLACE_MOL_RESULT_DIR=/agentic3d-run/results \
  -v "$source_snapshot:/workspace$source_mount_suffix" \
  -v "$benchmark_work:/workspace/benchmarks" \
  -v "$run_dir:/agentic3d-run" \
  "$image" bash -lc '
    set -euo pipefail
    cd /workspace
    if [[ "$compile" == "1" ]]; then
      mkdir -p DREAMPlace/build
      cd DREAMPlace/build
      cmake ..
      make -j 8
      make -j 8 install
      cd /workspace
    fi
    python -u src/place_3d/main.py \
      --benchmark="$design" --seed="$seed" \
      --config_file=/agentic3d-run/config.json 2>&1 | tee /agentic3d-run/run.log
  '
SH
chmod +x "$launcher"
REMOTE_PREP

echo "Prepared remote launcher: $REMOTE:$REMOTE_LAUNCHER"
echo "Remote run directory: $REMOTE:$RUN_DIR"
if [[ "$PREPARE_ONLY" == "1" ]]; then
  exit 0
fi

printf '%s\n' "$OPEN3DBENCH_SUDO_PASSWORD" | \
  ssh -tt "$REMOTE" "sudo -S -p '' bash -lc $(printf '%q' "$REMOTE_LAUNCHER")"

ssh "$REMOTE" "test -s '$RUN_DIR/run.log'; find '$RUN_DIR' -maxdepth 3 -type f | sort"
