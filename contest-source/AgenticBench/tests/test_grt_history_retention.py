"""Host retention control tests; synthetic artifacts do not qualify native EDA."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from global_bench.grt_history_retention import RetainedHistory, CONTROL, POLICY
from grt_bench.archive import completed_outcomes
from grt_bench.common import digest, read, sha, write
from grt_bench.contract import validate
from scripts import pilot_trace_retention as module
SyntheticCampaign = module.SyntheticCampaign


def authorization(fixture):
    return {'version': 1, 'enabled': True, 'host_only': True,
            'root': str(fixture.root.resolve()), 'task_digest': validate(fixture.task),
            'evaluator_sha256': fixture.task['evaluator_sha256'], 'policy': POLICY}


@pytest.fixture
def campaign(tmp_path):
    fixture = SyntheticCampaign(tmp_path / 'production-campaign', artifact_bytes=16)
    fixture.branches()
    history = RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))
    for index, name in enumerate(('baseline', 'improved', 'weak', 'tie', 'derived')):
        history.seal(fixture.names[name], 'terminal-%d' % index)
    return fixture, history


def test_best_trace_is_strict_score_record_not_ancestry(campaign):
    fixture, history = campaign
    assert [item['candidate'] for item in history.trace()] == [fixture.names[n] for n in ('baseline', 'improved', 'derived')]
    assert history.history(fixture.names['tie'])['record_setter'] is False
    assert history.history(fixture.names['weak'])['logs'] == []
    item = history.history(fixture.names['derived'])
    assert item['source_audit']['parent'] == fixture.names['tie']
    assert item['compact_history_only'] and item['may_admit_new_candidate'] is False
    assert len(item['outcomes']) == 9
    assert item['verification_scope'].endswith('not a fresh physical verification')


def test_original_upstream_logs_are_bound_to_requesting_cells(campaign):
    fixture, history = campaign
    requested = fixture.jobs[fixture.names['derived']][0]
    producer = fixture.jobs[fixture.names['tie']][0]
    item = next(log for log in history.history(fixture.names['derived'])['logs']
                if log['path'] == 'runs/' + producer + '/output/run/30_placement/stage.log')
    assert item['requested_jobs'] == [requested]
    assert item['reused'] is True
    assert stat_mode(fixture.root / item['blob']) == 0o444
    proof = 'runs/' + producer + '/control/plan.json'
    assert proof in history.history(fixture.names['derived'])['original_receipts']


def stat_mode(path):
    return path.stat().st_mode & 0o777


def test_history_survives_raw_retirement_but_is_not_native_admission(campaign):
    fixture, history = campaign
    candidate = fixture.names['weak']
    before_archive = digest(fixture.archive.show())
    before_queue = digest(fixture.queue.rows())
    for job in fixture.jobs[candidate]:
        attempt = fixture.root / 'runs' / job
        (attempt / 'output/analysis/metrics.json').unlink()
        for path in (attempt / 'output/run').rglob('*'):
            if path.is_file() and path.suffix in {'.log', '.odb', '.def', '.spef'}:
                path.unlink()
    assert len(history.history(candidate)['outcomes']) == 9
    assert digest(fixture.archive.show()) == before_archive
    assert digest(fixture.queue.rows()) == before_queue
    with pytest.raises((ValueError, FileNotFoundError)):
        completed_outcomes(fixture.root, fixture.task, candidate, fixture.archive.admit(candidate))


def test_record_setter_full_logs_survive_original_log_retirement(campaign):
    fixture, history = campaign
    candidate = fixture.names['improved']
    record = history.history(candidate)
    for log in record['logs']:
        (fixture.root / log['path']).unlink(missing_ok=True)
    assert len(history.history(candidate)['logs']) == len(record['logs'])


def test_already_retired_physical_files_do_not_block_initial_seal(tmp_path):
    fixture = SyntheticCampaign(tmp_path / 'physical-retired', artifact_bytes=16)
    fixture.branches()
    for path in fixture.root.rglob('*.odb'):
        path.unlink()
    history = RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))
    assert history.seal(fixture.baseline, 'baseline')['record_setter']
    assert history.seal(fixture.names['improved'], 'improved')['record_setter']


@pytest.mark.parametrize('field,value', [('enabled', False), ('host_only', False), ('root', '/other'),
    ('task_digest', 'f' * 64), ('evaluator_sha256', 'e' * 64), ('policy', 'pilot'), ('version', True)])
def test_exact_explicit_host_authorization_required(tmp_path, field, value):
    fixture = SyntheticCampaign(tmp_path / 'auth', artifact_bytes=16)
    policy = authorization(fixture)
    policy[field] = value
    with pytest.raises(ValueError):
        RetainedHistory.initialize(fixture.root, fixture.task, authorization=policy)
    assert not (fixture.root / CONTROL).exists()


def test_scalar_only_policy_rejects_multiobjective_contract(tmp_path, monkeypatch):
    fixture = SyntheticCampaign(tmp_path / 'multi', artifact_bytes=16)
    altered = copy.deepcopy(fixture.task)
    altered['objectives'] = ['timing', 'overflow', 'temperature_peak_k']
    with pytest.raises(ValueError):
        RetainedHistory.initialize(fixture.root, altered, authorization=authorization(fixture))


def test_initialization_blocks_active_queue(tmp_path):
    fixture = SyntheticCampaign(tmp_path / 'active', artifact_bytes=16)
    from grt_bench.contract import matrix
    fixture.queue.submit(matrix(fixture.task, fixture.baseline)[0], fixture.resources)
    with pytest.raises(ValueError, match='Active'):
        RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))
    assert not (fixture.root / CONTROL).exists()


def test_missing_history_does_not_accept_unverified_failed_candidate(campaign):
    fixture, history = campaign
    candidate, _ = fixture.terminal_branch('failed')
    assert not history.has(candidate)
    with pytest.raises(ValueError):
        history.seal(candidate, 'failure')
    with pytest.raises(ValueError, match='no host-sealed'):
        history.history(candidate)


@pytest.mark.parametrize('tamper', ['seal', 'authorization', 'queue', 'receipt', 'source', 'log', 'log-mode', 'key-mode', 'key-link', 'control-mode', 'archive'])
def test_authentication_and_original_evidence_tamper_fail_closed(campaign, tamper):
    fixture, history = campaign
    candidate = fixture.names['improved']
    record = history.history(candidate)
    if tamper in {'seal', 'authorization'}:
        target = fixture.root / CONTROL / ('authorization.json' if tamper == 'authorization' else 'history/' + candidate + '.json')
        value = read(target)
        value['body']['root'] = '/forged'
        write(target, value, replace=True)
    elif tamper == 'queue':
        with sqlite3.connect(fixture.queue.path) as db:
            db.execute('UPDATE jobs SET result=? WHERE id=?', ('{}', fixture.jobs[candidate][0]))
    elif tamper == 'receipt':
        path = fixture.root / 'runs' / fixture.jobs[candidate][0] / 'verified/result.json'
        value = read(path)
        value['metrics']['overflow'] = -99
        write(path, value, replace=True)
    elif tamper == 'source':
        path = fixture.root / 'candidates' / candidate / 'tree' / module.PARTITION
        path.chmod(0o644)
        path.write_text('value = 999\n')
    elif tamper in {'log', 'log-mode'}:
        path = fixture.root / record['logs'][0]['blob']
        path.chmod(0o644)
        if tamper == 'log':
            path.write_bytes(b'forged log')
            path.chmod(0o444)
    elif tamper == 'key-mode':
        (fixture.root / CONTROL / 'host.key').chmod(0o644)
    elif tamper == 'key-link':
        os.link(fixture.root / CONTROL / 'host.key', fixture.root / 'extra-key')
    elif tamper == 'control-mode':
        (fixture.root / CONTROL).chmod(0o755)
    else:
        with sqlite3.connect(fixture.archive.path) as db:
            value = json.loads(db.execute('SELECT row_json FROM evaluations WHERE candidate=?', (fixture.baseline,)).fetchone()[0])
            value['vector']['score'] = 0.0
            db.execute('UPDATE evaluations SET row_json=? WHERE candidate=?', (json.dumps(value), fixture.baseline))
    with pytest.raises((ValueError, FileNotFoundError)):
        history.history(candidate)


@pytest.mark.parametrize('tamper', ['control-proof', 'source-log', 'source-plan'])
def test_missing_upstream_log_or_untrusted_provenance_cannot_seal(tmp_path, tamper):
    fixture = SyntheticCampaign(tmp_path / 'proof', artifact_bytes=16)
    fixture.branches()
    history = RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))
    history.seal(fixture.baseline, 'baseline')
    producer = fixture.jobs[fixture.names['tie']][0]
    dest = fixture.jobs[fixture.names['derived']][0]
    if tamper == 'control-proof':
        path = fixture.root / 'runs' / dest / 'control/incremental.json'
        value = read(path)
        value['source_job'] = fixture.jobs[fixture.baseline][0]
        write(path, value, replace=True)
    elif tamper == 'source-log':
        (fixture.root / 'runs' / producer / 'output/run/30_placement/stage.log').unlink()
    else:
        path = fixture.root / 'runs' / producer / 'control/plan.json'
        value = read(path)
        value['task_digest'] = 'f' * 64
        write(path, value, replace=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        history.seal(fixture.names['derived'], 'derived')


def test_gc_domains_snapshot_and_idempotency(campaign):
    fixture, history = campaign
    item = history.history(fixture.names['weak'])
    assert history.seal(fixture.names['weak'], item['event_id']) == item
    with pytest.raises(ValueError, match='another terminal'):
        history.seal(fixture.names['weak'], 'changed-event')
    snapshot = history.gc_snapshot()
    assert len(snapshot['candidates']) == len(snapshot['archive_candidates']) == 5
    assert fixture.names['derived'] in snapshot['full_log_candidates']
    value = history.sign_host_record({'paths': ['a']}, 'gc-plan')
    assert history.verify_host_record(value, 'gc-plan') == {'paths': ['a']}
    with pytest.raises(ValueError):
        history.verify_host_record(value, 'gc-receipt')
    with pytest.raises(FileExistsError):
        RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))


def test_signed_records_cannot_be_renamed_to_another_candidate(campaign):
    fixture, history = campaign
    first, second = fixture.names['improved'], fixture.names['weak']
    original = fixture.root / CONTROL / 'history' / (first + '.json')
    target = fixture.root / CONTROL / 'history' / (second + '.json')
    write(target, read(original), replace=True)
    with pytest.raises(ValueError, match='path binding'):
        history.history(second)
    with pytest.raises(ValueError, match='path binding'):
        history.trace()


def test_record_setter_retains_explicit_text_local_tests_not_disguised_binary(tmp_path):
    fixture = SyntheticCampaign(tmp_path / 'local-checks', artifact_bytes=16)
    fixture.branches()
    script = fixture.root / 'local-diagnostics' / 'check.py'
    script.parent.mkdir()
    script.write_text('assert 1 + 1 == 2\n')
    history = RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))
    row = history.seal(fixture.baseline, 'baseline', log_paths=['local-diagnostics/check.py'])
    assert any(log['path'] == 'local-diagnostics/check.py' for log in row['logs'])
    script.write_bytes(b'\x80\x81')
    with pytest.raises(UnicodeDecodeError):
        history.seal(fixture.names['improved'], 'improved', log_paths=['local-diagnostics/check.py'])


@pytest.mark.parametrize('kind', ['lifecycle-receipt', 'terminal-diagnostics'])
def test_lifecycle_domains_are_explicit_and_isolated(campaign, kind):
    fixture, history = campaign
    value = history.sign_host_record({'round': 4, 'note': 'diagnostic only'}, kind)
    assert history.verify_host_record(value, kind)['round'] == 4
    with pytest.raises(ValueError):
        history.verify_host_record(value, 'gc-receipt')
    with pytest.raises(ValueError):
        history.sign_host_record({}, 'arbitrary-agent-input')


def test_blocked_unsubmitted_round_allows_prior_verified_backfill_without_accounting_change(tmp_path):
    fixture = SyntheticCampaign(tmp_path / 'blocked-backfill', artifact_bytes=16)
    fixture.branches()
    write(fixture.root / 'control/campaign.json', {'evaluations_used': 5,
        'rounds': [{'round': 1, 'status': 'verified'}, {'round': 2, 'status': 'blocked', 'matrices': 0}]})
    archive, queue, campaign = digest(fixture.archive.show()), digest(fixture.queue.rows()), sha(fixture.root / 'control/campaign.json')
    history = RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))
    history.seal(fixture.baseline, 'baseline')
    item = history.seal(fixture.names['improved'], 'verified-before-block')
    assert item['record_setter'] and len(item['outcomes']) == 9
    assert digest(fixture.archive.show()) == archive and digest(fixture.queue.rows()) == queue
    assert sha(fixture.root / 'control/campaign.json') == campaign
