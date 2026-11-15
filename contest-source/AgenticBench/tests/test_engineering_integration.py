"""Migration/CLI tests; no real model, Docker or physical runs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from engineering_bench.cli import main
from engineering_bench.native_cli import COMMANDS
from engineering_bench.protocol import OnceLedger
from engineering_bench.tasks import load_task
from test_engineering_tasks import fake_qualification

ROOT = Path(__file__).resolve().parents[1]


def adapter(name):
    spec = importlib.util.spec_from_file_location('engineering_test_' + name, ROOT / 'engineering_bench/physical' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('entry', COMMANDS)
def test_all_native_help_commands_are_offline(entry):
    result = subprocess.run([sys.executable, '-m', 'engineering_bench', 'native', entry, '--help'],
        cwd=ROOT, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_layout_requires_explicit_existing_paths(tmp_path):
    module = adapter('layout')
    with pytest.raises(FileNotFoundError):
        module.layout_path(tmp_path, 'release')
    (tmp_path / 'release.json').write_text('{}')
    (tmp_path / 'platform').mkdir()
    (tmp_path / 'references').mkdir()
    value = {'schema_version': 1, 'release': 'release.json', 'platform': 'platform', 'reference_root': 'references'}
    config = tmp_path / 'engineering-layout.json'
    config.write_text(json.dumps(value))
    assert module.layout_path(tmp_path, 'release') == tmp_path / 'release.json'
    assert module.layout_path(tmp_path, 'reference_root') == tmp_path / 'references'
    value['schema_version'] = True
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Invalid'):
        module.layout_path(tmp_path, 'release')
    config.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match='Duplicate'):
        module.layout_path(tmp_path, 'release')


@pytest.mark.parametrize('as_queue', [False, True])
def test_report_accepts_real_queue_or_ledger(tmp_path, capsys, as_queue):
    task = load_task('clock_compensation')
    queue = tmp_path / 'finals'
    ledger = OnceLedger(queue / 'ledger', task, 'e' * 64)
    q = fake_qualification(tmp_path / 'qualification', task, 'bp_fe')
    ledger.claim('bp_fe', q)
    ledger.finish('bp_fe', succeeded=True, legal=True, candidate_metric=.5)
    (queue / 'sealed.json').write_text(json.dumps({'task_digest': task.digest(), 'submission_sha256': 'e' * 64}))
    root = queue if as_queue else queue / 'ledger'
    assert main(['report', task.task_id, str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    # One case may succeed without inventing three missing measurements.
    assert '50.0' in json.dumps(result)
    assert 'missing' in json.dumps(result)
    if as_queue:
        (queue / 'sealed.json').write_text(json.dumps({'task_digest': task.digest(), 'submission_sha256': 'd' * 64}))
        with pytest.raises(ValueError, match='Queue seal'):
            main(['report', task.task_id, str(root)])


def test_task_contracts_do_not_select_the_post_grt_endpoint():
    for name in ('clock_compensation', 'multi_hbt_fanout', 'cross_die_cts'):
        task = load_task(name)
        assert 'grt' not in task.data['endpoint']
        assert task.data['seed'] == 1001
        assert task.data['evaluations_per_case'] == 1


def test_packaging_keeps_old_and_new_entrypoints():
    import tomllib
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    scripts = config['project']['scripts']
    assert scripts['openagentic3dbench-engineering'] == 'engineering_bench.cli:main'
    assert scripts['openagentic3dbench-grt'] == 'grt_bench.__main__:main'
    assert scripts['openagentic3dbench-global'] == 'global_bench.cli:main'
    assert 'native/*.tcl' in config['tool']['setuptools']['package-data']['engineering_bench']
