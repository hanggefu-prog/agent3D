#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.config/agentic3d/env.sh"

printf 'host=%s\n' "$(hostname)"
printf 'repository=%s\n' "$AGENTIC3D_REPO_ROOT"
printf 'agenticbench=%s\n' "$AGENTIC3D_ROOT"
printf 'results=%s\n' "$OPEN3DBENCH_RESULTS_ROOT"
"$AGENTIC3D_REPO_ROOT/scripts/check_server_checkout.sh" \
  "$AGENTIC3D_REPO_ROOT" "${EXPECTED_AGENTIC_OPEN3DBENCH_REV:-}"
python3 --version
agentic3d-inspect --repo "$OPEN3DBENCH_ROOT" >/tmp/agentic3d-repository-context.json
openagentic3dbench-skill list >/tmp/openagentic3dbench-skills.json
openagentic3dbench-profile list >/tmp/openagentic3dbench-profiles.json
python3 -m unittest discover -s "$AGENTIC3D_ROOT/tests" -v
docker version --format 'docker-server={{.Server.Version}}'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
docker image inspect --format 'placement-image={{.Id}}' "$OPEN3DBENCH_PLACE_IMAGE"
docker image inspect --format 'evaluation-image={{.Id}}' "$OPEN3DBENCH_EVAL_IMAGE"

[[ -f "$OPEN3DBENCH_ROOT/Place-3D/build/dreamplace/source_fingerprint.json" ]] || {
  echo "Missing DREAMPlace build. Run: make -C $OPEN3DBENCH_ROOT dreamplace IMAGE=$OPEN3DBENCH_PLACE_IMAGE" >&2
  exit 2
}

docker run --rm --gpus all \
  -v "$OPEN3DBENCH_ROOT/Place-3D:/workspace" \
  -w /workspace \
  "$OPEN3DBENCH_PLACE_IMAGE" bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/build/dreamplace/install:/workspace/build/dreamplace/install/dreamplace:/workspace
    python3 -c "import torch; import dreamplace.configure; import dreamplace.ops.electric_potential.electric_potential_cuda; assert torch.cuda.is_available(); print(\"torch=\" + torch.__version__); print(\"gpu=\" + torch.cuda.get_device_name(0))"
  '

docker run --rm \
  -v "$OPEN3DBENCH_ROOT:/workspace" \
  -w /workspace/OpenROAD-3D/flow \
  "$OPEN3DBENCH_EVAL_IMAGE" bash -lc '
    set -euo pipefail
    openroad -version
    test -f Makefile
  '
printf 'adapter_context=/tmp/agentic3d-repository-context.json\n'
printf 'skill_catalog=/tmp/openagentic3dbench-skills.json\n'
printf 'evaluation_profiles=/tmp/openagentic3dbench-profiles.json\n'
