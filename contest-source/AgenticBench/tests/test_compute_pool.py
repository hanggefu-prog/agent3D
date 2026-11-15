import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

POOL = Path(__file__).resolve().parents[1] / 'deploy/compute_pool'
spec = importlib.util.spec_from_file_location('compute_pool_worker', POOL / 'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_worker_pins_case_and_reference_without_api(monkeypatch, tmp_path):
    seen = []
    output = tmp_path / 'results/test'

    def run(command, **kwargs):
        seen.append(command)
        output.mkdir(parents=True)
        (output / 'evaluation_result.json').write_text(json.dumps({
            'metrics': {'runs_successfully': 1}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, 'run', run)
    worker.evaluate(tmp_path, 'global_routing_ripup_policy', 'bp', tmp_path / 'candidate', output)
    command = seen[0]
    assert command[command.index('--base-config') + 1].endswith('bp_lol_ng45_mtkahypar_cts/config.yaml')
    assert command[command.index('--reference-run') + 1] == str(tmp_path / 'references/bp')
    assert '--compact-artifacts' in command


def test_refuses_existing_evaluation(tmp_path):
    with pytest.raises(FileExistsError):
        worker.evaluate(tmp_path, 'global_routing_ripup_policy', 'bp', tmp_path, tmp_path)


@pytest.mark.parametrize('returncode,success', [(1, 1), (0, 0)])
def test_stage_failure_not_qualified(monkeypatch, tmp_path, returncode, success):
    output = tmp_path / 'failed'

    def run(*args, **kwargs):
        output.mkdir()
        (output / 'evaluation_result.json').write_text(json.dumps({
            'metrics': {'runs_successfully': success}}))
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(worker.subprocess, 'run', run)
    with pytest.raises(RuntimeError):
        worker.evaluate(tmp_path, 'global_routing_ripup_policy', 'bp', tmp_path, output)


def test_node_resources_and_no_credentials():
    text = (POOL / 'run_worker.sh').read_text()
    assert '--network none' in text
    assert '--cpus 32 --memory "$memory" --memory-swap "$memory"' in text
    assert 'flock -n 8' in text
    assert ':ro' in text
    assert 'API_KEY' not in text and '.secrets' not in text


def test_no_docker_daemon_restart_or_prune():
    scripts = '\n'.join(p.read_text() for p in POOL.glob('*.sh'))
    assert 'docker system prune' not in scripts
    assert 'systemctl' not in scripts
    assert 'docker export "$container"' in scripts
    assert 'docker save "$image"' not in scripts


def test_transfer_marker_is_on_same_connection_and_skipped_on_failure():
    import subprocess
    from deploy.compute_pool.stream_install import extract_and_mark, ssh
    command = extract_and_mark('/tmp/compute-pool-test', 'git-objects')
    assert command.startswith('set -e;')
    assert 'git-objects.ready' in command
    assert 'bash -o pipefail' in ssh('example.test', 22, command)[-1]
    # Use a failed extraction and observable marker instead of touching disk.
    failure = command.replace('zstd -dc | tar --no-same-owner -xf - -C /tmp/compute-pool-test/data', 'true | false')
    failure = failure.replace('date -Is > /tmp/compute-pool-test/git-objects.ready', 'echo incorrect-marker')
    result = subprocess.run(['bash', '-o', 'pipefail', '-c', failure], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'incorrect-marker' not in result.stdout
    with pytest.raises(ValueError):
        extract_and_mark('/tmp/compute-pool-test', '../bad')


def test_sta_deployment_pins_provenance_and_separates_reference_from_evolution():
    publisher = (POOL / 'publish_sta_dependencies.sh').read_text()
    installer = (POOL / 'install_sta_dependencies.sh').read_text()
    prepare = (POOL / 'prepare_sta_case.sh').read_text()
    assert '8a89deb905b96f30996ec2ce4498d80626a250a8' in publisher
    assert 'source-working-tree.patch' in publisher
    assert 'sha256sum -c' in installer and 'fsck --connectivity-only' in installer
    assert 'get_parasitic_cutpoint_timing=get_parasitic_cutpoint_timing' in installer
    assert 'framework-sta.ready' in prepare and 'sta-dependencies.ready' in prepare
    assert '--network none' in prepare and '--memory 60g' in prepare
    assert '--seed-run "$seed"' in prepare
    assert 'initial_candidate' in prepare
    assert 'OPEN3D_OPT_PASSES=2' in prepare
    assert 'gpt-' not in prepare
