import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from engineering_bench.identity import evaluator_identity
from engineering_bench.tasks import load_task, TASK_ROOT

PHYSICAL = Path(__file__).resolve().parents[1] / 'engineering_bench/physical'
sys.path.insert(0, str(PHYSICAL))


@pytest.mark.parametrize('success,changes,expected', [
    (True, [], 'no_change'),
    (True, ['open3dbench/backend/three_d_cts.py'], 'implementation_ready'),
    (False, [], 'developer_failed'),
    (False, ['open3dbench/backend/three_d_cts.py'], 'developer_failed'),
])
def test_developer_exit_success_does_not_qualify_an_empty_candidate(success, changes, expected):
    from develop import development_status
    assert development_status({'success': success}, changes) == expected


def test_scoring_and_tcl_and_shared_runtime_invalidate_identity(tmp_path):
    root = tmp_path / 'engineering_bench'
    (root / 'physical').mkdir(parents=True)
    (root / 'native').mkdir()
    (tmp_path / 'grt_bench').mkdir()
    files = [root / 'metrics.py', root / 'interfaces.py', root / 'clock_equivalence.py',
             root / 'native/inspect.tcl', tmp_path / 'grt_bench/runtime.py']
    for p in files:
        p.write_text('original')
    # Frozen identity_version 1 (still used to verify old qualifications).
    # Version 2 is covered by tests/test_engineering_identity_v2.py.
    before = evaluator_identity(root, version=1)
    for p in files:
        p.write_text('changed')
        assert evaluator_identity(root, version=1) != before
        p.write_text('original')
        assert evaluator_identity(root, version=1) == before


def test_clock_timeout_confirms_container_termination(tmp_path):
    from clock_executor import sample_plan
    subject = tmp_path / 'source'
    subject.mkdir()
    context = tmp_path / 'context.json'
    context.write_text('{}')
    with patch('clock_executor.subprocess.run', side_effect=subprocess.TimeoutExpired('docker', 300)), \
         patch('grt_bench.runtime.stop_and_confirm') as cleanup:
        with pytest.raises(subprocess.TimeoutExpired):
            sample_plan(subject, context, 'test', tmp_path / 'output')
    assert cleanup.call_count == 1
    assert cleanup.call_args.args[0].startswith('o3db-engineering-clock-policy-')


def test_uncertain_clock_termination_is_not_hidden(tmp_path):
    from clock_executor import sample_plan
    from grt_bench.runtime import UncertainExecution
    subject = tmp_path / 'source'
    subject.mkdir()
    context = tmp_path / 'context.json'
    context.write_text('{}')
    with patch('clock_executor.subprocess.run', side_effect=subprocess.TimeoutExpired('docker', 300)), \
         patch('grt_bench.runtime.stop_and_confirm', side_effect=UncertainExecution('running')):
        with pytest.raises(UncertainExecution):
            sample_plan(subject, context, 'test', tmp_path / 'output')


def test_native_runtime_requires_exact_source_and_base(tmp_path):
    from engineering_bench.candidate_runtime import verify
    seal = {'source_sha256': 'source', 'submission_sha256': 'submission'}
    for receipt in ({'source_sha256': 'other', 'base_image': 'base'},
                    {'source_sha256': 'source', 'base_image': 'other'}):
        with patch('engineering_bench.candidate_runtime.verify_seal'):
            with pytest.raises(ValueError, match='another source or base'):
                verify(tmp_path, seal, receipt, 'base')


def test_seed_contract_is_separate_and_interface_changes_identity(tmp_path, monkeypatch):
    import shutil
    shutil.copytree(TASK_ROOT, tmp_path / 'tasks')
    old = load_task('clock_compensation')
    path = tmp_path / 'tasks/clock_compensation/task.json'
    data = json.loads(path.read_text()); data['seed'] = 1002
    path.write_text(json.dumps(data))
    monkeypatch.setenv('ENGINEERING_TASK_ROOT', str(tmp_path / 'tasks'))
    changed = load_task('clock_compensation')
    assert changed.data['seed'] == 1002 and changed.digest() != old.digest()
    before = changed.digest()
    (changed.root / 'context-v2.md').write_text('New public interface')
    assert changed.digest() != before
