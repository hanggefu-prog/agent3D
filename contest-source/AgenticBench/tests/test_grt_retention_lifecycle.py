"""Terminal lifecycle tests use synthetic receipts, never EDA qualification."""
import io
from pathlib import Path
import sys
import tarfile

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from global_bench import grt_retention_lifecycle as lifecycle
from global_bench.grt_history_retention import RetainedHistory, CONTROL
from grt_bench.common import digest, read, sha, write
from test_grt_history_retention import SyntheticCampaign, authorization


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    f = SyntheticCampaign(tmp_path / 'lifecycle', artifact_bytes=16)
    f.branches()
    rows = []
    for number, name in enumerate(('improved', 'weak', 'tie', 'derived'), 1):
        rows.append({'round': number, 'status': 'verified', 'candidate': f.names[name],
                     'control': '/original-server/control/rounds/%04d' % number,
                     'matrices': 1, 'history': {'feedback': 'Synthetic completed matrix'}})
        folder = f.root / 'control/rounds' / ('%04d' % number)
        write(folder / 'prompt/bundle.json', {'round': number, 'task': 'synthetic'})
        (folder / 'prompt/prompt.md').write_text('Synthetic prompt\n')
        write(folder / 'agent/attempt-01/result.json', {'status': 'success'})
        (folder / 'agent/attempt-01/events.jsonl').write_text('{"event":"synthetic"}\n')
    write(f.root / 'control/campaign.json', {'evaluations_used': 4, 'rounds': rows})
    f.history = RetainedHistory.initialize(f.root, f.task, authorization=authorization(f))
    f.calls = []
    class Service:
        def __init__(self, root, task):
            self.kind = type(self).__name__
        def plan(self):
            f.calls.append((self.kind, 'plan'))
            relative = 'control/trace-retention/' + self.kind + '.plan.json'
            target = f.root / relative
            if not target.exists():
                write(target, {'fixture': self.kind})
            return {'path': relative, 'sha256': sha(target)}
        def apply(self, path, *, expected_sha256):
            f.calls.append((self.kind, 'apply', path, expected_sha256))
            target = f.root / path
            assert sha(target) == expected_sha256
            receipt = target.with_name(target.name[:-len('.plan.json')] + '.receipt.json')
            if not receipt.exists():
                write(receipt, {'fixture': self.kind, 'completed': True})
            return {'released_blocks_bytes': 4096, 'budget_changed': False}
    class Raw(Service):
        pass
    class Libraries(Service):
        pass
    import global_bench.grt_retention_gc as gc
    import global_bench.grt_library_retention as library_gc
    monkeypatch.setattr(gc, 'RetentionGC', Raw)
    monkeypatch.setattr(library_gc, 'LibraryRetention', Libraries)
    return f


def test_no_policy_keeps_exact_legacy_noop(tmp_path):
    assert lifecycle.after_round(tmp_path, {'unvalidated': 'legacy'}) is None
    assert list(tmp_path.iterdir()) == []


def test_existing_bad_policy_is_never_a_legacy_fallback(tmp_path):
    write(tmp_path / CONTROL / 'authorization.json', {'corrupt': True})
    with pytest.raises((ValueError, FileNotFoundError)):
        lifecycle.after_round(tmp_path, {'unvalidated': 'legacy'})


def test_backfill_preserves_accounting_and_uses_stable_round_events(fixture):
    f = fixture
    before = read(f.root / 'control/campaign.json')
    queue_before = digest(f.queue.rows())
    archive_before = digest(f.archive.show())
    result = lifecycle.after_round(f.root, f.task)
    assert len(result['sealed']) == 5
    assert result['budget_changed'] is False
    assert result['evaluations_used'] == 4
    assert read(f.root / 'control/campaign.json') == before
    assert digest(f.queue.rows()) == queue_before
    assert digest(f.archive.show()) == archive_before
    task = f.history.task_id
    assert f.history.history(f.baseline)['event_id'] == task + ':baseline'
    candidate = f.names['derived']
    proof = f.history.history(candidate)
    assert proof['event_id'] == task + ':round:4:' + candidate
    assert any(log['path'] == 'control/rounds/0004/prompt/prompt.md' for log in proof['logs'])
    assert any(log['path'] == 'control/rounds/0004/agent/attempt-01/events.jsonl' for log in proof['logs'])
    assert [call[:2] for call in f.calls] == [('Raw', 'plan'), ('Raw', 'apply'),
                                          ('Libraries', 'plan'), ('Libraries', 'apply')]


def test_repeated_completed_lifecycle_audits_but_never_replans_gc(fixture):
    first = lifecycle.after_round(fixture.root, fixture.task)
    count = len(fixture.calls)
    second = lifecycle.after_round(fixture.root, fixture.task)
    assert first == second
    assert len(fixture.calls) == count + 2
    assert [call[1] for call in fixture.calls[count:]] == ['apply', 'apply']


def test_idempotent_lifecycle_requires_completed_receipt_before_any_apply(fixture):
    first = lifecycle.after_round(fixture.root, fixture.task)
    plan = fixture.root / first['raw_plan']
    plan.with_name(plan.name[:-len('.plan.json')] + '.receipt.json').unlink()
    count = len(fixture.calls)
    with pytest.raises(ValueError, match='completed receipt'):
        lifecycle.after_round(fixture.root, fixture.task)
    assert len(fixture.calls) == count


def test_strict_full_logs_and_summary_only_record_selection(fixture):
    lifecycle.after_round(fixture.root, fixture.task)
    weak = fixture.history.history(fixture.names['weak'])
    improved = fixture.history.history(fixture.names['improved'])
    assert not weak['record_setter'] and weak['logs'] == []
    assert improved['record_setter'] and any(log['job'] is None for log in improved['logs'])


def test_nonrecords_never_materialize_development_or_full_trace_copies(fixture, monkeypatch):
    seen = []
    original = lifecycle.round_log_paths
    def tracked(root, row, **kwargs):
        seen.append(row['round'])
        return original(root, row, **kwargs)
    monkeypatch.setattr(lifecycle, 'round_log_paths', tracked)
    lifecycle.after_round(fixture.root, fixture.task)
    assert seen == [1, 4]


@pytest.mark.parametrize('status', ['prepared', 'evaluating', 'reconciliation_required', 'unknown'])
def test_unknown_or_live_round_cannot_collect_or_retire(fixture, status):
    path = fixture.root / 'control/campaign.json'
    state = read(path)
    state['rounds'][-1]['status'] = status
    write(path, state, replace=True)
    with pytest.raises(ValueError, match='terminal rounds'):
        lifecycle.after_round(fixture.root, fixture.task)
    assert fixture.calls == []
    assert not fixture.history.has(fixture.baseline)


def test_failed_admitted_and_unsubmitted_rounds_are_diagnostics_only(fixture):
    f = fixture
    failed, job = f.terminal_branch('failed')
    path = f.root / 'control/campaign.json'
    state = read(path)
    state['evaluations_used'] += 1
    state['rounds'].append({'round': 5, 'status': 'failed', 'candidate': failed, 'matrices': 1,
                            'evaluation_failure': [job], 'history': {'feedback': 'Native failure'}})
    state['rounds'].append({'round': 6, 'status': 'generation_failed', 'candidate': None, 'matrices': 0,
                            'history': {'feedback': 'Client failure before admission'}})
    write(path, state, replace=True)
    result = lifecycle.after_round(f.root, f.task, state['rounds'][-1])
    assert not f.history.has(failed)
    assert result['evaluations_used'] == 5
    assert len(result['diagnostics']) == 2
    assert read(path) == state
    for relative in result['diagnostics']:
        value = f.history.verify_host_record(read(f.root / relative), kind='terminal-diagnostics')
        assert value['budget_changed'] is False
    assert f.history.verify_host_record(read(f.root / result['diagnostics'][0]), kind='terminal-diagnostics')['evaluated']


def test_mismatched_passed_round_is_rejected_without_state_edits(fixture):
    state = read(fixture.root / 'control/campaign.json')
    with pytest.raises(ValueError, match='exact persisted'):
        lifecycle.after_round(fixture.root, fixture.task, dict(state['rounds'][-1], round=999))
    assert read(fixture.root / 'control/campaign.json') == state


def test_explicit_trace_allowlist_excludes_credentials_configs_and_workspaces(fixture):
    f = fixture
    row = read(f.root / 'control/campaign.json')['rounds'][-1]
    folder = f.root / 'control/rounds/0004'
    values = {
        'agent/private/result.json': '{}', 'agent/attempt-01/settings.json': '{}',
        'agent/attempt-01/command.json': '{}', 'tool-config.json': '{}',
        'agent/attempt-01/stderr.log': 'Original client diagnostic\n',
        'agent/attempt-01/extra.stdout.log': 'Original stdout\n',
        'tool-calls.jsonl': '{"tool":"checkout_parent"}\n',
        'gpu-checks/req/result.json': '{"status":"success"}',
        'gpu-checks/req/lease.json': '{"token":"host-only"}'}
    for name, data in values.items():
        p = folder / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)
    scratch = f.root / 'scratch/0004'
    for name in ('test.py', 'check.sh', 'params.yaml', 'report.xml', 'private/config.json', 'output.npy',
                 'api-key.yaml', 'apikey.json', 'api.key.xml', 'password.txt', 'token.txt'):
        p = scratch / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('Synthetic local diagnostic\n')
    p = scratch / 'binary.log'
    p.write_bytes(b'\xff\x00')
    outside = f.root / 'workspaces/0004/do-not-copy.py'
    outside.parent.mkdir(parents=True)
    outside.write_text('This is source, not a diagnostic')
    logs = lifecycle.round_log_paths(f.root, row)
    assert 'scratch/0004/test.py' in logs
    assert 'scratch/0004/params.yaml' in logs
    assert 'control/rounds/0004/tool-calls.jsonl' in logs
    assert 'control/rounds/0004/agent/attempt-01/stderr.log' in logs
    assert all(not any(word in path for word in ('private', 'settings', 'command.json', 'tool-config', 'lease', 'workspaces', 'binary', '.npy',
                                                'api-key', 'apikey', 'api.key', 'password', 'token'))
               for path in logs)


def test_direct_pruning_allowlist_keeps_source_results_bindings_and_unknowns(fixture):
    f = fixture
    row = read(f.root / 'control/campaign.json')['rounds'][-1]
    root = f.root
    names = ['control/rounds/0004/agent/attempt-01/events.jsonl',
             'control/rounds/0004/agent/attempt-01/result.json',
             'control/rounds/0004/agent/attempt-01/stderr.log',
             'control/rounds/0004/agent/attempt-01/command.json',
             'control/rounds/0004/gpu-checks/a/container.log',
             'control/rounds/0004/gpu-checks/a/result.json',
             'scratch/0004/check.log', 'scratch/0004/synthetic.py',
             'scratch/0004/params.yaml', 'scratch/0004/source/trace.log',
             'scratch/0004/secret.txt', 'scratch/0004/build/compile.log']
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('Synthetic diagnostics\n')
    assert lifecycle.prunable_round_texts(root, row) == sorted([
        'control/rounds/0004/agent/attempt-01/events.jsonl',
        'control/rounds/0004/agent/attempt-01/stderr.log',
        'control/rounds/0004/gpu-checks/a/container.log', 'scratch/0004/check.log'])


def test_legacy_development_archive_retains_only_unique_safe_utf8_text(fixture):
    f = fixture
    row = read(f.root / 'control/campaign.json')['rounds'][-1]
    development = f.root / 'control/old-launch/retention/0004/development'
    development.mkdir(parents=True)
    files = {'check.py': b'assert 1 + 1 == 2\n', 'private/settings.json': b'{"token":"excluded"}',
             'data.npy': b'\x00\xff', 'results.csv': b'case,result\na,1\n'}
    archive = development / 'experiments.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    write(development / 'receipt.json', {'status': 'removed', 'candidate': row['candidate'],
          'retained_paths': list(files), 'retained_bytes': sum(map(len, files.values())),
          'experiments_sha256': sha(archive), 'workspace': '/old-server/workspaces/0004'})
    paths = lifecycle.round_log_paths(f.root, row)
    copied = [f.root / name for name in paths if '/development-texts/' in name]
    assert sorted(p.name for p in copied) == ['check.py', 'results.csv']
    assert all(p.stat().st_mode & 0o777 == 0o444 for p in copied)
    assert (development / 'receipt.json').exists() and archive.exists()
    assert lifecycle.round_log_paths(f.root, row) == paths


def test_retention_failure_never_reclassifies_or_recharges_matrix(fixture, monkeypatch):
    import global_bench.grt_retention_gc as gc
    before = read(fixture.root / 'control/campaign.json')
    original = gc.RetentionGC.apply
    def failing(*args, **kwargs):
        raise RuntimeError('Maintenance storage failure')
    monkeypatch.setattr(gc.RetentionGC, 'apply', failing)
    with pytest.raises(RuntimeError, match='Maintenance storage'):
        lifecycle.after_round(fixture.root, fixture.task)
    assert read(fixture.root / 'control/campaign.json') == before
    assert before['rounds'][-1]['status'] == 'verified'
    assert before['evaluations_used'] == 4


def loop_fixture(tmp_path, monkeypatch):
    from global_bench import grt_campaign_loop as loop
    root = tmp_path / 'loop-campaign'
    root.mkdir()
    config_path = root / 'control/config.json'
    config = {'model': 'unchanged-model', 'evaluation_limit': 1}
    write(config_path, config)
    state_path = root / 'control/campaign.json'
    write(state_path, {'evaluations_used': 0, 'rounds': []})
    class Campaign:
        def __init__(self, supplied):
            assert supplied == str(config_path)
            self.root, self.config, self.config_path = root, config, config_path
            self.task, self.task_id, self.state_path = {'legacy': 'unused'}, 'task', state_path
        def implementation_digest(self):
            return 'unchanged-framework'
        def prepare(self):
            folder = root / 'control/rounds/0001'
            folder.mkdir(parents=True)
            row = {'round': 1, 'status': 'prepared', 'control': str(folder), 'prompt_sha256': 'prompt'}
            write(state_path, {'evaluations_used': 0, 'rounds': [row]}, replace=True)
            return row
        def run(self, prompt):
            assert prompt == 'prompt'
            row = read(state_path)['rounds'][-1]
            row.update(status='verified', candidate='candidate')
            write(state_path, {'evaluations_used': 1, 'rounds': [row]}, replace=True)
            return row
    monkeypatch.setattr(loop, 'Campaign', Campaign)
    authorization_path = root / 'control/launch/authorization.json'
    write(authorization_path, {'config': str(config_path), 'config_digest': digest(config),
          'task_digest': 'task', 'implementation_digest': 'unchanged-framework',
          'driver_sha256': sha(loop.__file__), 'model': config['model'], 'evaluation_limit': 1,
          'first_prompt_sha256': 'prompt', 'minimum_free_gib': 0,
          'consecutive_pre_submission_failure_limit': 2})
    return loop, state_path, authorization_path


def test_unconfigured_loop_emits_exact_legacy_statuses(tmp_path, monkeypatch):
    loop, state, authorization_path = loop_fixture(tmp_path, monkeypatch)
    loop.run(authorization_path)
    events = [__import__('json').loads(line)['phase'] for line in
              (authorization_path.parent / 'events.jsonl').read_text().splitlines()]
    assert events == ['round_running', 'round_finished', 'completed']
    assert read(state)['evaluations_used'] == 1


def test_loop_calls_retention_before_prepare_and_preserves_admitted_result_on_error(tmp_path, monkeypatch):
    loop, state, authorization_path = loop_fixture(tmp_path, monkeypatch)
    calls = []
    def after(root, task, row=None, **kwargs):
        calls.append(None if row is None else row['status'])
        if row is not None:
            raise RuntimeError('Retention disk failure after completed matrix')
        return None
    monkeypatch.setattr(loop, 'after_round', after)
    with pytest.raises(RuntimeError, match='Retention disk'):
        loop.run(authorization_path)
    assert calls == [None, 'verified']
    assert read(state)['evaluations_used'] == 1
    assert read(state)['rounds'][-1]['status'] == 'verified'
    status = read(authorization_path.parent / 'status.json')
    assert status['phase'] == 'stopped_with_error'
    assert 'Retention disk' in status['error']
