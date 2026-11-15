#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${OPENAGENTIC3DBENCH_DEV_ROOT:-/data/gcr}"
REPO_ROOT="${AGENTIC3D_REPO_ROOT:-$DATA_ROOT/Agentic-Open3DBench}"
AGENT_ROOT="$REPO_ROOT/AgenticBench"
RESULTS_ROOT="${OPEN3DBENCH_RESULTS_ROOT:-$DATA_ROOT/agentic3d-runs}"
CACHE_ROOT="${AGENTIC3D_CACHE_ROOT:-$DATA_ROOT/agentic3d-cache}"
EXPECTED_REV="${EXPECTED_AGENTIC_OPEN3DBENCH_REV:-}"

[[ -d "$REPO_ROOT/.git" ]] || {
  echo "Missing canonical Git checkout: $REPO_ROOT" >&2
  echo "Deploy or clone Agentic-Open3DBench before running this installer." >&2
  exit 2
}
[[ -d "$AGENT_ROOT" ]] || {
  echo "Missing AgenticBench directory: $AGENT_ROOT" >&2
  exit 2
}

"$REPO_ROOT/scripts/check_server_checkout.sh" "$REPO_ROOT" "$EXPECTED_REV"
mkdir -p "$HOME/.local/bin" "$HOME/.config/agentic3d" \
  "$RESULTS_ROOT" "$RESULTS_ROOT/agent-state" "$RESULTS_ROOT/worktrees" \
  "$CACHE_ROOT"

if [[ ! -x "$REPO_ROOT/.venv/bin/python3" ]]; then
  python3 -m venv --without-pip "$REPO_ROOT/.venv"
fi

PYTHON_BIN="$REPO_ROOT/.venv/bin/python3"
python_tag="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$python_tag" in
  3.10)
    pyyaml_wheel="$REPO_ROOT/docker/wheels/PyYAML-6.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    pyyaml_sha256="ec031d5d2feb36d1d1a24380e4db6d43695f3748343d99434e6f5f9156aaa2ed"
    ;;
  3.8)
    pyyaml_wheel="$REPO_ROOT/docker/wheels/PyYAML-6.0.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    pyyaml_sha256="9056c1ecd25795207ad294bcf39f2db3d845767be0ea6e6a34d856f006006083"
    ;;
  *)
    echo "Unsupported server Python for offline PyYAML install: $python_tag" >&2
    exit 2
    ;;
esac
[[ -f "$pyyaml_wheel" ]] || {
  echo "Missing offline PyYAML wheel: $pyyaml_wheel" >&2
  exit 2
}
actual_pyyaml_sha256="$(sha256sum "$pyyaml_wheel" | awk '{print $1}')"
[[ "$actual_pyyaml_sha256" == "$pyyaml_sha256" ]] || {
  echo "PyYAML wheel checksum mismatch" >&2
  exit 2
}
if ! "$PYTHON_BIN" -c 'import yaml' >/dev/null 2>&1; then
  site_packages="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  "$PYTHON_BIN" -m zipfile -e "$pyyaml_wheel" "$site_packages"
fi
"$PYTHON_BIN" -c 'import yaml; assert yaml.__version__ == "6.0.2"'

install -m 0644 "$AGENT_ROOT/deploy/server/env.sh" \
  "$HOME/.config/agentic3d/env.sh"
install -m 0755 "$AGENT_ROOT/deploy/server/agentic3d-entrypoint" \
  "$HOME/.local/bin/agentic3d-entrypoint"
for command in \
  agentic3d-run agentic3d-inspect agentic3d-task-agent agentic3d-codex-exec \
  openagentic3dbench-run openagentic3dbench-inspect \
  openagentic3dbench-task openagentic3dbench-evolve \
  openagentic3dbench-convergence openagentic3dbench-profile \
  openagentic3dbench-sta openagentic3dbench-skill; do
  ln -sfn "$HOME/.local/bin/agentic3d-entrypoint" "$HOME/.local/bin/$command"
done
install -m 0755 "$AGENT_ROOT/deploy/server/run-agentic3d" \
  "$HOME/.local/bin/run-agentic3d"
install -m 0755 "$AGENT_ROOT/deploy/server/verify-server.sh" \
  "$HOME/.local/bin/verify-agentic3d-server"

source "$HOME/.config/agentic3d/env.sh"
agentic3d-inspect --repo "$OPEN3DBENCH_ROOT" >/tmp/agentic3d-repository-context.json
openagentic3dbench-task list >/tmp/openagentic3dbench-tasks.json
openagentic3dbench-skill list >/tmp/openagentic3dbench-skills.json
openagentic3dbench-profile list >/tmp/openagentic3dbench-profiles.json
printf 'installed_repository=%s\n' "$AGENTIC3D_REPO_ROOT"
printf 'installed_agenticbench=%s\n' "$AGENTIC3D_ROOT"
printf 'results_root=%s\n' "$OPEN3DBENCH_RESULTS_ROOT"
printf 'revision=%s\n' "$(git -C "$AGENTIC3D_REPO_ROOT" rev-parse HEAD)"
