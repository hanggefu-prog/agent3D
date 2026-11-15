from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "deploy" / "server"


def test_server_environment_uses_one_monorepo_checkout():
    env = (SERVER / "env.sh").read_text()
    assert 'AGENTIC3D_REPO_ROOT' in env
    assert 'AGENTIC3D_ROOT="$AGENTIC3D_REPO_ROOT/AgenticBench"' in env
    assert 'OPEN3DBENCH_ROOT="$AGENTIC3D_REPO_ROOT"' in env
    assert 'AGENTICBENCH_DATA_ROOT' in env
    assert 'OPEN3DBENCH_RELEASE_IMAGE' in env
    assert 'shiyunqi/open3dbench-v2.0:release' in env
    assert 'OPEN3DBENCH_PLACE_IMAGE="$OPEN3DBENCH_RELEASE_IMAGE"' in env
    assert 'OPEN3DBENCH_EVAL_IMAGE="$OPEN3DBENCH_RELEASE_IMAGE"' in env


def test_server_entrypoint_uses_repository_virtual_environment():
    entrypoint = (SERVER / "agentic3d-entrypoint").read_text()
    task_cli = (ROOT / "local_bench" / "cli.py").read_text()
    assert '"$AGENTIC3D_REPO_ROOT/.venv/bin/python3"' in entrypoint
    assert '"$AGENTIC3D_ROOT/.venv/bin/python3"' not in entrypoint
    assert 'openagentic3dbench-profile) module="local_bench.profile_cli"' in entrypoint
    assert 'openagentic3dbench-skill) module="local_bench.skill_cli"' in entrypoint
    assert 'if __name__ == "__main__":' in task_cli


def test_server_installer_requires_clean_checkout_verification():
    installer = (SERVER / "install-server.sh").read_text()
    assert 'scripts/check_server_checkout.sh' in installer
    assert 'EXPECTED_AGENTIC_OPEN3DBENCH_REV' in installer
    assert 'PyYAML-6.0.2-cp310-cp310' in installer
    assert 'python3" -m zipfile -e' not in installer
    assert '"$PYTHON_BIN" -m zipfile -e' in installer
    assert 'openagentic3dbench-profile' in installer
    assert 'openagentic3dbench-skill' in installer
    assert 'openagentic3dbench-task list' in installer


def test_checkout_verifier_requires_main_to_match_recorded_origin():
    verifier = (ROOT.parent / "scripts" / "check_server_checkout.sh").read_text()
    assert 'upstream" == "origin/main' in verifier
    assert 'refs/remotes/origin/main' in verifier
