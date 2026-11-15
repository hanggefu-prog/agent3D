"""Historical-library retirement regressions; no real EDA or production cleanup."""
import copy
import json
import os
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_grt_history_retention import SyntheticCampaign, authorization, module
from global_bench.grt_candidate_library import export_library, load_library
from global_bench.grt_history_retention import RetainedHistory
from global_bench.grt_library_retention import LibraryRetention, CONTROL, NOTICE
from grt_bench.common import digest, read, records, sha, write
from grt_bench.contract import validate


def bind_library(fixture, folder):
    prompt = fixture.root / 'control/rounds' / folder.name / 'prompt'
    bundle = {'context': {'task_digest': validate(fixture.task),
        'candidate_library': {'manifest_sha256': sha(folder / 'manifest.json')}}}
    write(prompt / 'bundle.json', bundle)
    write(prompt / 'manifest.json', {'bundle_digest': digest(bundle), 'artifacts': records(prompt)})
    write(prompt.parent / 'tool-config.json', {'bundle_digest': digest(bundle)})


@pytest.fixture
def libraries(tmp_path):
    # Three cases and one seed keep filesystem tests small; core tests separately
    # cover the complete three-seed independent matrix and native log provenance.
    original = module.DEFAULT_SEEDS
    module.DEFAULT_SEEDS = original[:1]
    try:
        fixture = SyntheticCampaign(tmp_path / 'library-campaign', artifact_bytes=16)
    finally:
        module.DEFAULT_SEEDS = original
    fixture.branches()
    old1, old2 = fixture.root / 'libraries/0001', fixture.root / 'libraries/0002'
    export_library(fixture.root, fixture.task, old1)
    export_library(fixture.root, fixture.task, old2)
    bind_library(fixture, old1)
    bind_library(fixture, old2)
    write(fixture.root / 'control/campaign.json', {'evaluations_used': 4,
        'rounds': [{'round': 1, 'status': 'verified'}, {'round': 2, 'status': 'verified'}]})
    history = RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))
    for index, name in enumerate(('baseline', 'improved', 'weak', 'tie', 'derived')):
        history.seal(fixture.names[name], 'terminal-' + str(index))
    owner = LibraryRetention(fixture.root, fixture.task)
    with patch('global_bench.grt_library_retention.live_docker_mounts', return_value=[]):
        yield fixture, history, owner


def _protected(folder):
    return {name: sha(folder / name) for name in read(folder / 'manifest.json')['artifacts']
            if '/logs/' not in name or name.endswith(('/index.json', '/README.md'))} | {'manifest.json': sha(folder / 'manifest.json')}


def test_exact_old_libraries_retire_only_duplicate_bulk_logs(libraries):
    fixture, history, owner = libraries
    before = {p.name: _protected(p) for p in (fixture.root / 'libraries').iterdir()}
    archive = digest(fixture.archive.show())
    queue = digest(fixture.queue.rows())
    campaign = sha(fixture.root / 'control/campaign.json')
    blobs = {log['blob']: sha(fixture.root / log['blob']) for record in history.trace() for log in record['logs']}
    plan = owner.plan()
    assert plan['library_count'] == 2 and plan['file_count'] > 0
    result = owner.apply(plan['path'], expected_sha256=plan['sha256'])
    assert result['removed_count'] == plan['file_count']
    assert result['released_blocks_bytes'] > 0
    assert digest(fixture.archive.show()) == archive and digest(fixture.queue.rows()) == queue
    assert sha(fixture.root / 'control/campaign.json') == campaign
    assert all(sha(fixture.root / path) == checksum for path, checksum in blobs.items())
    for folder in (fixture.root / 'libraries').iterdir():
        assert _protected(folder) == before[folder.name]
        assert (folder / NOTICE).is_file()
        assert 'intentionally incomplete' in (folder / NOTICE).read_text()
        with pytest.raises((ValueError, FileNotFoundError)):
            load_library(folder, fixture.task, fixture.archive.show())
    assert len(list((fixture.root / CONTROL / 'receipts').glob('*.json'))) == 2
    assert owner.apply(plan['path'], expected_sha256=plan['sha256']) == result
    assert owner.plan()['file_count'] == 0


def test_new_compact_library_works_after_old_retirement_and_cas_is_pinned(libraries):
    fixture, history, owner = libraries
    new = fixture.root / 'libraries/0003'
    export_library(fixture.root, fixture.task, new)
    bind_library(fixture, new)
    state = read(fixture.root / 'control/campaign.json')
    state['rounds'].append({'round': 3, 'status': 'verified'})
    write(fixture.root / 'control/campaign.json', state, replace=True)
    plan = owner.plan()
    result = owner.apply(plan['path'], expected_sha256=plan['sha256'])
    assert result['library_count'] == 3
    for row in history.trace():
        for log in row['logs']:
            assert sha(fixture.root / log['blob']) == log['file']['sha256']
    current = fixture.root / 'libraries/0004'
    export_library(fixture.root, fixture.task, current)
    assert load_library(current, fixture.task, fixture.archive.show())[0]['schema_version'] == 2


@pytest.mark.parametrize('status', ['prepared', 'generating', 'evaluating', 'reconciliation_required'])
def test_active_round_blocks_library_retirement(libraries, status):
    fixture, history, owner = libraries
    state = read(fixture.root / 'control/campaign.json')
    state['rounds'][-1]['status'] = status
    write(fixture.root / 'control/campaign.json', state, replace=True)
    with pytest.raises(ValueError, match='round blocks'):
        owner.plan()
    assert not (fixture.root / 'libraries/0001' / NOTICE).exists()


def test_unknown_or_active_queue_blocks_library_retirement(libraries):
    fixture, history, owner = libraries
    with sqlite3.connect(fixture.queue.path) as db:
        db.execute('UPDATE jobs SET state=? WHERE id=?', ('uncertain', fixture.jobs[fixture.baseline][0]))
    with pytest.raises(ValueError, match='queue work'):
        owner.plan()


def test_any_live_campaign_mount_blocks(libraries):
    fixture, history, owner = libraries
    with patch('global_bench.grt_library_retention.live_docker_mounts', return_value=[{'container': 'x', 'source': str(fixture.root / 'workspaces')} ]):
        with pytest.raises(ValueError, match='Docker campaign mount'):
            owner.plan()


def test_missing_or_corrupt_history_blocks(libraries):
    fixture, history, owner = libraries
    path = fixture.root / 'control/trace-retention/history' / (fixture.names['weak'] + '.json')
    path.unlink()
    with pytest.raises(ValueError, match='Every admitted'):
        owner.plan()


@pytest.mark.parametrize('tamper', ['content', 'inode', 'links', 'symlink', 'unclassified', 'manifest', 'index'])
def test_tamper_and_unclassified_evidence_fail_closed(libraries, tamper):
    fixture, history, owner = libraries
    plan = owner.plan()
    body = history.verify_host_record(read(fixture.root / plan['path']), 'gc-plan')
    relative = next(iter(body['files']))
    target = fixture.root / relative
    if tamper == 'content':
        target.chmod(0o644)
        target.write_bytes(b'changed')
        target.chmod(0o444)
    elif tamper == 'inode':
        data = target.read_bytes()
        target.unlink()
        target.write_bytes(data)
        target.chmod(0o444)
    elif tamper == 'links':
        os.link(target, fixture.root / 'external-link')
    elif tamper == 'symlink':
        target.unlink()
        target.symlink_to(fixture.root / 'candidates')
    elif tamper == 'unclassified':
        (fixture.root / 'libraries/0001/new-unknown.log').write_text('unknown')
    else:
        path = fixture.root / 'libraries/0001' / (tamper + '.json')
        path.chmod(0o644)
        path.write_text('{}')
        path.chmod(0o444)
    with pytest.raises((ValueError, FileNotFoundError)):
        owner.apply(plan['path'], expected_sha256=plan['sha256'])
    assert not list((fixture.root / CONTROL).glob('*.journal.jsonl'))


def test_exact_hash_and_changed_campaign_snapshot_blocks(libraries):
    fixture, history, owner = libraries
    plan = owner.plan()
    with pytest.raises(ValueError, match='hash authorization'):
        owner.apply(plan['path'], expected_sha256='0' * 64)
    state = read(fixture.root / 'control/campaign.json')
    state['evaluations_used'] += 1
    write(fixture.root / 'control/campaign.json', state, replace=True)
    with pytest.raises(ValueError, match='changed after planning'):
        owner.apply(plan['path'], expected_sha256=plan['sha256'])


def test_crash_after_intent_is_not_replayed(libraries):
    fixture, history, owner = libraries
    plan = owner.plan()
    original = Path.unlink
    def fail(target, *args, **kwargs):
        if '/libraries/0001/' in str(target) and '/logs/' in str(target):
            raise OSError('fixture unlink failure')
        return original(target, *args, **kwargs)
    with patch.object(Path, 'unlink', fail):
        with pytest.raises(OSError, match='fixture unlink'):
            owner.apply(plan['path'], expected_sha256=plan['sha256'])
    assert list((fixture.root / CONTROL).glob('*.intent.json'))
    with pytest.raises(ValueError, match='Unresolved'):
        owner.apply(plan['path'], expected_sha256=plan['sha256'])
    with pytest.raises(ValueError, match='Unresolved'):
        owner.plan()


def test_signed_journal_is_append_only_and_tamper_detected(libraries):
    fixture, history, owner = libraries
    plan = owner.plan()
    result = owner.apply(plan['path'], expected_sha256=plan['sha256'])
    journal = next((fixture.root / CONTROL).glob('*.journal.jsonl'))
    events = [history.verify_host_record(json.loads(line), 'gc-progress') for line in journal.read_text().splitlines()]
    assert len(events) == result['removed_count']
    assert [event['sequence'] for event in events] == list(range(1, len(events) + 1))
    with journal.open('ab') as stream:
        stream.write(b'{}\n')
    with pytest.raises(ValueError, match='receipt|journal'):
        owner.apply(plan['path'], expected_sha256=plan['sha256'])


def test_unregistered_library_is_skipped_not_deleted(libraries):
    fixture, history, owner = libraries
    extra = fixture.root / 'libraries/0099'
    export_library(fixture.root, fixture.task, extra)
    before = records(extra)
    plan = owner.plan()
    assert plan['library_count'] == 2
    owner.apply(plan['path'], expected_sha256=plan['sha256'])
    assert records(extra) == before


@pytest.mark.parametrize('tamper', ['missing', 'bundle', 'prompt-manifest', 'tool-config'])
def test_original_host_prompt_binding_required(libraries, tamper):
    fixture, history, owner = libraries
    folder = fixture.root / 'control/rounds/0001'
    if tamper == 'missing':
        (folder / 'prompt/bundle.json').unlink()
    elif tamper == 'bundle':
        path = folder / 'prompt/bundle.json'
        value = read(path)
        value['context']['candidate_library']['manifest_sha256'] = 'f' * 64
        write(path, value, replace=True)
    elif tamper == 'prompt-manifest':
        path = folder / 'prompt/manifest.json'
        value = read(path)
        value['bundle_digest'] = 'f' * 64
        write(path, value, replace=True)
    else:
        write(folder / 'tool-config.json', {'bundle_digest': 'f' * 64}, replace=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        owner.plan()
