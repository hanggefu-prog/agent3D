"""Host GC contract tests; not native physical qualification."""
from contextlib import contextmanager
import copy
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import types

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from grt_bench.common import digest, read, sha, write
from global_bench import grt_retention_gc as gc


def ident(label):
    return hashlib.sha256(label.encode()).hexdigest()


class FakeHistory:
    """An authenticated host snapshot, independent of raw paths under test."""
    def __init__(self, root, task):
        self.root, self.task = root, task
        self.key = b'x' * 32
        self.rows = {}

    @contextmanager
    def locked(self):
        yield

    def gc_snapshot(self):
        return {'candidates': list(self.rows.values()), 'archive_candidates': list(self.rows),
                'protected_receipts': ['runs/' + r['queue_rows'][0]['id'] + '/output/analysis/manifest.json' for r in self.rows.values()],
                'protected_blobs': ['control/trace-retention/log-blobs/old.log']}

    def sign_host_record(self, body, *, kind):
        signature = hmac.new(self.key, digest({'kind': kind, 'body': body}).encode(), hashlib.sha256).hexdigest()
        return {'body': body, 'kind': kind, 'signature': signature}

    def verify_host_record(self, envelope, *, kind):
        if envelope != self.sign_host_record(envelope['body'], kind=kind):
            raise ValueError('Bad host seal/domain')
        return envelope['body']


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    root.mkdir()
    task_id = ident('task')
    baseline, old, weak, best = [ident(x) for x in ('baseline', 'old-record', 'weak', 'best')]
    task = {'optimization': {'baseline': baseline}}
    monkeypatch.setattr(gc, 'validate', lambda task: task_id)
    monkeypatch.setattr(gc, 'live_docker_mounts', lambda: [])
    history = FakeHistory(root, task)
    service = gc.RetentionGC.__new__(gc.RetentionGC)
    service.root, service.task, service.task_id = root, task, task_id
    service.history = history
    service.control = root / gc.CONTROL
    service.control.mkdir(parents=True)
    write(root / 'control/campaign.json', {'evaluations_used': 4, 'rounds': [{'status': 'verified', 'parent': best}]})
    database = root / 'control/queue.sqlite'
    con = sqlite3.connect(database)
    con.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,payload TEXT,resources TEXT,state TEXT,result TEXT)')
    archive = root / ('control/archives/' + task_id + '.sqlite')
    archive.parent.mkdir()
    adb = sqlite3.connect(archive)
    adb.execute('CREATE TABLE metadata(key TEXT,value TEXT)')
    adb.execute('INSERT INTO metadata VALUES(?,?)', ('task', json.dumps(task)))
    adb.execute('CREATE TABLE evaluations(candidate TEXT,row_json TEXT)')
    adb.execute('CREATE TABLE selections(id TEXT,receipt TEXT)')
    adb.execute('CREATE TABLE children(candidate TEXT,parent TEXT,selection TEXT)')
    jobs = {}
    for index, (candidate, score) in enumerate(((baseline, 1.), (old, .99), (weak, 1.1), (best, .95))):
        job = ident(candidate)
        jobs[candidate] = job
        payload = {'candidate': candidate, 'task_digest': task_id, 'case': 'case-a',
                   'seed': {'id': 's1', 'placement': 1}, 'split': 'train', 'endpoint': 'post-grt'}
        path = root / 'runs' / job
        plan = {**payload, 'output': '/output'}
        write(path / 'control/plan.json', plan)
        write(path / 'control/task.json', task)
        verified = {**payload, 'status': 'verified', 'evidence': {'plan_sha256': sha(path / 'control/plan.json')}}
        write(path / 'verified/result.json', verified)
        write(path / 'output/analysis/manifest.json', {'artifacts': {'detail.rpt': {}}})
        raw = path / 'output/run/60_opt/design.odb'
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'physical-data' * 1000)
        log = raw.with_name('tool.log')
        log.write_text('important algorithm diagnostics\n' * 200)
        raw.with_name('metrics.json').write_text('{}')
        (path / 'output/analysis/detail.rpt').write_text('raw analysis\n' * 100)
        (path / 'output/analysis/unclassified.bin').write_bytes(b'unknown')
        con.execute('INSERT INTO jobs VALUES(?,?,?,?,?)', (job, json.dumps(payload), '{}', 'succeeded', json.dumps(verified)))
        evaluation = {'candidate': candidate, 'feasible': True, 'vector': {'score': score}, 'split': 'train', 'task_digest': task_id, 'reference_digest': baseline}
        adb.execute('INSERT INTO evaluations VALUES(?,?)', (candidate, json.dumps(evaluation)))
        record = candidate != weak
        logs = [{'path': log.relative_to(root).as_posix()}] if record else []
        history.rows[candidate] = {'candidate': candidate, 'record_setter': record, 'logs': logs,
                                   'queue_rows': [{'id': job}]}
        write(root / 'control/trace-retention/history' / (candidate + '.json'), {'sealed-fixture': candidate})
    con.commit()
    con.close()
    adb.commit()
    adb.close()
    blob = root / 'control/trace-retention/log-blobs/old.log'
    blob.parent.mkdir()
    blob.write_text('sealed full log bytes')
    return types.SimpleNamespace(root=root, service=service, history=history, task=task, task_id=task_id,
                                 baseline=baseline, old=old, weak=weak, best=best, jobs=jobs)


def raw(f, candidate, filename='design.odb'):
    return 'runs/' + f.jobs[candidate] + '/output/run/60_opt/' + filename


def apply(f, plan):
    return f.service.apply(plan['path'], expected_sha256=plan['sha256'])


def test_only_nonessential_raw_artifacts_retire_and_current_best_stays(fixture):
    f = fixture
    plan = f.service.plan()
    paths = plan['plan']['body']['files']
    assert raw(f, f.weak) in paths
    assert raw(f, f.old) in paths  # past records need log CAS, not permanent physical runs
    assert raw(f, f.baseline) not in paths
    assert raw(f, f.best) not in paths
    before = sha(f.root / 'control/campaign.json')
    outcome = apply(f, plan)
    assert outcome['released_blocks_bytes'] > 0
    assert not (f.root / raw(f, f.weak)).exists()
    assert (f.root / raw(f, f.best)).exists()
    assert (f.root / raw(f, f.baseline)).exists()
    assert sha(f.root / 'control/campaign.json') == before
    assert (f.root / raw(f, f.weak, 'metrics.json')).exists()
    assert (f.root / 'control/trace-retention/log-blobs/old.log').exists()
    for job in f.jobs.values():
        assert (f.root / 'runs' / job / 'output/analysis/manifest.json').exists()
        assert (f.root / 'runs' / job / 'output/analysis/unclassified.bin').exists()
    assert apply(f, plan) == outcome


@pytest.mark.parametrize('kind', ['uncertain', 'queued', 'running', 'cancelling', 'unknown'])
def test_nonterminal_or_unknown_queue_fails_closed(fixture, kind):
    f = fixture
    con = sqlite3.connect(f.root / 'control/queue.sqlite')
    con.execute('UPDATE jobs SET state=? WHERE id=?', (kind, f.jobs[f.weak]))
    con.commit()
    con.close()
    with pytest.raises(ValueError, match='queue work'):
        f.service.plan()


def test_live_mount_pins_exact_directory_and_mount_change_invalidates_plan(fixture, monkeypatch):
    f = fixture
    mount = {'container': 'host-test', 'source': str(f.root / 'runs' / f.jobs[f.weak] / 'output')}
    monkeypatch.setattr(gc, 'live_docker_mounts', lambda: [mount])
    plan = f.service.plan()
    assert raw(f, f.weak) not in plan['plan']['body']['files']
    monkeypatch.setattr(gc, 'live_docker_mounts', lambda: [])
    with pytest.raises(ValueError, match='changed after planning'):
        apply(f, plan)


def test_whole_root_mount_and_unavailable_docker_block(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(gc, 'live_docker_mounts', lambda: [{'container': 'x', 'source': str(f.root)}])
    with pytest.raises(ValueError, match='whole campaign'):
        f.service.plan()
    def unavailable():
        raise RuntimeError('Docker inventory unavailable')
    monkeypatch.setattr(gc, 'live_docker_mounts', unavailable)
    with pytest.raises(RuntimeError, match='unavailable'):
        f.service.plan()


@pytest.mark.parametrize('tamper', ['plan', 'verified', 'queue', 'history', 'artifact', 'pins'])
def test_tamper_or_changed_dependencies_rejected_before_any_unlink(fixture, tamper):
    f = fixture
    plan = f.service.plan()
    if tamper in {'plan', 'verified'}:
        relative = 'control/plan.json' if tamper == 'plan' else 'verified/result.json'
        path = f.root / 'runs' / f.jobs[f.weak] / relative
        value = read(path)
        value['case'] = 'wrong-case'
        write(path, value, replace=True)
    elif tamper == 'queue':
        con = sqlite3.connect(f.root / 'control/queue.sqlite')
        con.execute('UPDATE jobs SET resources=? WHERE id=?', ('{"memory_gb":99}', f.jobs[f.weak]))
        con.commit()
        con.close()
    elif tamper == 'history':
        path = f.root / 'control/trace-retention/history' / (f.weak + '.json')
        write(path, {'changed': True}, replace=True)
    elif tamper == 'artifact':
        (f.root / raw(f, f.weak)).write_bytes(b'changed')
    else:
        f.history.gc_snapshot = lambda: {'candidates': list(f.history.rows.values()),
            'archive_candidates': list(f.history.rows), 'protected_receipts': [raw(f, f.weak)], 'protected_blobs': []}
    with pytest.raises((ValueError, FileNotFoundError)):
        apply(f, plan)
    assert (f.root / raw(f, f.old)).exists()


def test_forged_plan_and_authority_deletion_are_rejected(fixture):
    f = fixture
    with pytest.raises(ValueError, match='Pinned or unclassified'):
        f.service.plan(paths=['control/campaign.json'])
    with pytest.raises(ValueError, match='Pinned or unclassified'):
        f.service.plan(paths=['runs/' + f.jobs[f.weak] + '/output/analysis/manifest.json'])
    plan = f.service.plan()
    path = f.root / plan['path']
    value = read(path)
    value['body']['created_at'] = 0
    write(path, value, replace=True)
    with pytest.raises(ValueError, match='seal'):
        f.service.apply(plan['path'], expected_sha256=sha(path))


def test_unresolved_crash_intent_never_repeats_or_assumes_success(fixture):
    f = fixture
    plan = f.service.plan()
    key = digest(plan['plan']['body'])
    relative = raw(f, f.weak)
    intent = f.root / gc.CONTROL / (key + '.intent.json')
    write(intent, f.history.sign_host_record({'path': relative, 'plan_digest': key}, kind='gc-intent'))
    with pytest.raises(ValueError, match='Unresolved retirement intent'):
        apply(f, plan)
    assert (f.root / relative).exists()
    assert intent.exists()


def test_last_link_accounting_does_not_claim_kept_hardlink_blocks(fixture):
    f = fixture
    source = f.root / raw(f, f.weak)
    protected = f.root / 'preserved-hardlink.odb'
    os.link(source, protected)
    plan = f.service.plan(paths=[raw(f, f.weak)])
    outcome = apply(f, plan)
    assert outcome['logical_bytes'] > 0
    assert outcome['released_blocks_bytes'] == 0
    assert protected.exists()


def test_unsealed_candidate_and_unknown_file_remain_pinned(fixture):
    f = fixture
    del f.history.rows[f.weak]
    plan = f.service.plan()
    assert raw(f, f.weak) not in plan['plan']['body']['files']
    assert (f.root / raw(f, f.weak)).exists()


def test_paired_incremental_stage_producer_stays_pinned(fixture):
    f = fixture
    source, child = f.jobs[f.weak], f.jobs[f.best]
    for job in (source, child):
        path = f.root / 'runs' / job / 'output/run/00_input/stage_manifest.json'
        write(path, {'stage': 'input', 'artifacts': {}})
        path.with_name('design.odb').write_bytes(b'prefix')
    parent_plan = read(f.root / 'runs' / source / 'control/plan.json')
    parent_verified = read(f.root / 'runs' / source / 'verified/result.json')
    proof = {'version': 1, 'task_digest': f.task_id, 'source_job': source, 'physical_source_job': source,
             'parent': f.weak, 'parent_plan_digest': digest(parent_plan),
             'parent_verified_digest': digest(parent_verified), 'impact': {'start': 'floorplan'},
             'stages': {'input': sha(f.root / 'runs' / source / 'output/run/00_input/stage_manifest.json')}}
    write(f.root / 'runs' / child / 'control/incremental.json', proof)
    planpath = f.root / 'runs' / child / 'control/plan.json'
    p = read(planpath)
    p['incremental'] = '/control/incremental.json'
    write(planpath, p, replace=True)
    verifiedpath = f.root / 'runs' / child / 'verified/result.json'
    v = read(verifiedpath)
    v['incremental'] = proof
    v['evidence']['plan_sha256'] = sha(planpath)
    write(verifiedpath, v, replace=True)
    plan = f.service.plan()
    prefix = 'runs/' + source + '/output/run/00_input/design.odb'
    assert prefix not in plan['plan']['body']['files']
    assert raw(f, f.weak) in plan['plan']['body']['files']
    apply(f, plan)
    assert (f.root / prefix).exists()
    # Even a well-paired malicious sidecar cannot replace the verifier's proof.
    proof['stages'] = {}
    write(f.root / 'runs' / child / 'control/incremental.json', proof, replace=True)
    with pytest.raises(ValueError, match='original verifier'):
        f.service.plan()


def test_real_retained_history_integration_preserves_nine_cell_evidence(tmp_path, monkeypatch):
    # The real frozen Queue/Archive/source verifier and production HMAC service
    # qualify the compact history; GC may never manufacture admission authority.
    from test_grt_history_retention import SyntheticCampaign, authorization
    from global_bench.grt_history_retention import RetainedHistory
    fixture = SyntheticCampaign(tmp_path / 'real-host-controls', artifact_bytes=128)
    fixture.complete(fixture.baseline, 'baseline', 1.)
    old = fixture.child(fixture.baseline, 'old')
    weak = fixture.child(fixture.baseline, 'weak')
    best = fixture.child(fixture.baseline, 'best')
    fixture.complete(old, 'old', .9)
    fixture.complete(weak, 'weak', 1.2)
    fixture.complete(best, 'best', .8)
    write(fixture.root / 'control/campaign.json', {'evaluations_used': 4, 'rounds': [{'status': 'verified', 'parent': best}]})
    history = RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))
    for i, candidate in enumerate((fixture.baseline, old, weak, best)):
        history.seal(candidate, 'terminal-' + str(i))
    monkeypatch.setattr(gc, 'live_docker_mounts', lambda: [])
    service = gc.RetentionGC(fixture.root, fixture.task)
    original = history.history(weak)
    archived = digest(fixture.archive.show())
    queued = digest(fixture.queue.rows())
    plan = service.plan()
    outcome = service.apply(plan['path'], expected_sha256=plan['sha256'])
    assert outcome['removed']
    assert history.history(weak)['outcomes'] == original['outcomes']
    assert len(history.history(weak)['outcomes']) == 9
    assert digest(fixture.archive.show()) == archived
    assert digest(fixture.queue.rows()) == queued
    assert len(history.trace()) == 3
    for record in history.trace():
        for log in record['logs']:
            assert sha(fixture.root / log['blob']) == log['file']['sha256']
    assert service.apply(plan['path'], expected_sha256=plan['sha256']) == outcome


@pytest.mark.parametrize('name,counts', [('fable5', (150, 18, 117, 252)), ('astra', (224, 24, 179, 880))])
def test_original_native_metadata_dependency_replay(name, counts):
    """Read-only replay of copied native receipts, not a fresh EDA execution."""
    metadata = Path(__file__).resolve().parents[1] / 'native-metadata' / name
    if not metadata.exists():
        pytest.skip('Audited native metadata bundle is not part of the framework release')
    rows = read(metadata / 'queue-rows.json')
    service = gc.RetentionGC.__new__(gc.RetentionGC)
    service.root = metadata
    def metadata_path(relative):
        parts = Path(relative).parts
        assert parts[0] == 'runs'
        if len(parts) == 6 and parts[2:4] == ('output', 'run') and parts[-1] == 'stage_manifest.json':
            return metadata / 'jobs' / parts[1] / 'stage-manifests' / (parts[4] + '.json')
        return metadata / 'jobs' / parts[1] / Path(*parts[2:])
    service._path = metadata_path
    trusted, aliases, edges, manifests = service._dependencies(rows)
    assert (len(trusted), sum(k != v for k, v in aliases.items()), len(edges), len(manifests)) == counts


@pytest.mark.parametrize('target_kind', ['internal', 'external', 'dangling', 'external-directory'])
def test_native_aliases_are_preserved_and_never_delete_targets(fixture, tmp_path, target_kind):
    f = fixture
    parent = (f.root / raw(f, f.weak)).parent
    external = tmp_path / 'external-target'
    external.mkdir()
    (external / 'keep.odb').write_text('external immutable data')
    if target_kind == 'internal':
        target = parent / 'design.odb'
    elif target_kind == 'external':
        target = external / 'keep.odb'
    elif target_kind == 'dangling':
        target = parent / 'already-retired.odb'
    else:
        target = external
    alias = parent / 'native-alias.odb'
    alias.symlink_to(target, target_is_directory=target_kind == 'external-directory')
    with pytest.raises(ValueError, match='Symlink'):
        f.service.plan(paths=[alias.relative_to(f.root).as_posix()])
    for _ in range(2):
        plan = f.service.plan()
        assert alias.relative_to(f.root).as_posix() not in plan['plan']['body']['files']
        apply(f, plan)
        assert alias.is_symlink()  # Including aliases whose original raw target retired.
        assert (external / 'keep.odb').read_text() == 'external immutable data'


def test_signed_linear_journal_resumes_completed_unlinks_and_counts_last_link(fixture, monkeypatch):
    f = fixture
    first = f.root / raw(f, f.weak)
    second = first.with_name('second.odb')
    os.link(first, second)
    allocated = first.stat().st_blocks * 512
    paths = [first.relative_to(f.root).as_posix(), second.relative_to(f.root).as_posix()]
    plan = f.service.plan(paths=paths)
    original_write = gc.write
    intents = []
    def crash_before_second_intent(path, *args, **kwargs):
        if str(path).endswith('.intent.json'):
            intents.append(str(path))
            if len(intents) == 2:
                raise RuntimeError('Simulated crash after the first durable journal step')
        return original_write(path, *args, **kwargs)
    monkeypatch.setattr(gc, 'write', crash_before_second_intent)
    with pytest.raises(RuntimeError, match='Simulated crash'):
        apply(f, plan)
    assert not first.exists()
    assert second.stat().st_nlink == 1
    key = digest(plan['plan']['body'])
    journal = f.root / gc.CONTROL / (key + '.steps')
    assert len(list(journal.iterdir())) == 1
    monkeypatch.setattr(gc, 'write', original_write)
    outcome = apply(f, plan)
    assert outcome['removed'] == sorted(paths)
    assert outcome['released_blocks_bytes'] == allocated
    assert len(list(journal.iterdir())) == 2
    assert not (f.root / gc.CONTROL / (key + '.progress.json')).exists()


def test_only_sealed_nonrecord_round_bulk_text_retires(fixture):
    f = fixture
    rounds = [{'round': i + 1, 'status': 'verified', 'candidate': c, 'parent': f.best}
              for i, c in enumerate((f.baseline, f.old, f.weak, f.best))]
    rounds.append({'round': 5, 'status': 'generation_failed', 'parent': f.best})
    write(f.root / 'control/campaign.json', {'evaluations_used': 4, 'rounds': rounds}, replace=True)
    removable = ['control/rounds/0003/agent/attempt-01/events.jsonl',
                 'control/rounds/0003/agent/stdout.log',
                 'control/rounds/0003/gpu-checks/local-check/container.log',
                 'scratch/0003/local-check.rpt']
    protected = ['control/rounds/0003/agent/result.json',
                 'control/rounds/0003/agent/attempt-01/result.json',
                 'control/rounds/0003/agent/prompt.md',
                 'control/rounds/0003/tool-config.json',
                 'control/rounds/0003/agent/command.json',
                 'control/rounds/0003/agent/private/stdout.log',
                 'control/rounds/0002/agent/attempt-01/events.jsonl',
                 'control/rounds/0004/agent/stdout.log',
                 'control/rounds/0005/agent/stdout.log',
                 'control/recoveries/0003/complete-review.log',
                 'scratch/0003/source/algorithm.log',
                 'scratch/0003/build/output.log',
                 'scratch/0003/test-source.py',
                 'scratch/0003/secret.txt']
    for relative in removable + protected:
        path = f.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('safe synthetic diagnostic\n')
    secret = f.root / 'scratch/0003/unsafe-content.log'
    secret.write_text('sk-' + 'x' * 35)
    protected.append(secret.relative_to(f.root).as_posix())
    with pytest.raises(ValueError, match='Pinned or unclassified'):
        f.service.plan(paths=['control/rounds/0003/agent/result.json'])
    plan = f.service.plan()
    selected = plan['plan']['body']['files']
    assert all(p in selected for p in removable)
    assert all(p not in selected for p in protected)
    apply(f, plan)
    assert all(not (f.root / p).exists() for p in removable)
    assert all((f.root / p).exists() for p in protected)


@pytest.mark.parametrize('tamper', ['missing-journal', 'bad-receipt', 'reappeared'])
def test_completed_receipt_read_requires_exact_completed_journal(fixture, tamper):
    f = fixture
    plan = f.service.plan(paths=[raw(f, f.weak)])
    apply(f, plan)
    body = plan['plan']['body']
    key = digest(body)
    if tamper == 'missing-journal':
        (f.root / gc.CONTROL / (key + '.steps') / '00000001.json').unlink()
    elif tamper == 'bad-receipt':
        path = f.root / gc.CONTROL / (key + '.receipt.json')
        envelope = read(path)
        receipt = f.history.verify_host_record(envelope, kind='gc-receipt')
        receipt['released_blocks_bytes'] += 1
        write(path, f.history.sign_host_record(receipt, kind='gc-receipt'), replace=True)
    else:
        (f.root / raw(f, f.weak)).write_text('unexpected reappearance')
    with pytest.raises(ValueError):
        apply(f, plan)
