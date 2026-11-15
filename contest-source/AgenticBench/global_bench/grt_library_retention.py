"""Exact retirement of duplicate logs from inactive historical reading libraries.

Only manifest-inventoried ``<candidate>/logs/**`` bulk files can retire. Original
source, diffs, metrics, manifest, index and log indexes/READMEs retain their exact
bytes. A separate signed host receipt and an explicit new reading notice mark a
retired library as incomplete; its original strict loader must reject it. New
compact reading libraries remain independently sealed and usable.
"""
from contextlib import contextmanager, ExitStack
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import time

from grt_bench.common import confined, digest, read, sha, write
from grt_bench.contract import validate
from .grt_history_retention import RetainedHistory
from .grt_retention_gc import _file, _relative, live_docker_mounts

CONTROL = 'control/trace-retention/libraries'
NOTICE = 'RETENTION_NOTICE.md'
HEX = re.compile(r'[0-9a-f]{64}')
LIBRARY = re.compile(r'[0-9]{4}')
TERMINAL = {'succeeded', 'failed', 'cancelled', 'interrupted'}
ROUND_TERMINAL = {'verified', 'infeasible', 'failed', 'rejected', 'generation_failed',
                  'no_change', 'stalled', 'blocked'}


def _bulk_log(name):
    parts = Path(name).parts
    return (len(parts) >= 3 and HEX.fullmatch(parts[0]) and parts[1] == 'logs'
            and parts[2:] not in {('index.json',), ('README.md',)}
            and Path(name).name not in {'README.md', 'index.json'})


def _fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class LibraryRetention:
    """Host-only service; no EDA, native evidence removal or budget mutation."""

    def __init__(self, root, task):
        self.history = RetainedHistory(root, task)
        self.root, self.task, self.task_id = self.history.root, copy.deepcopy(task), validate(task)
        self.control = self._path(CONTROL)
        self.control.mkdir(parents=True, exist_ok=True)

    def _path(self, relative):
        path = confined(self.root, _relative(relative))
        if path.exists() and path.stat().st_dev != self.root.stat().st_dev:
            raise ValueError('Nested filesystem in library retention evidence')
        return path

    @contextmanager
    def _locked(self):
        with ExitStack() as stack:
            for relative in ('control/campaign.lock', 'control/result-storage/storage.lock'):
                path = self._path(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stack.enter_context(self.history.locked())
            yield

    def _binding(self):
        rows = self.history._queue()
        if any(row['state'] not in TERMINAL for row in rows):
            raise ValueError('Active, uncertain or unknown queue work blocks library retirement')
        state = read(self._path('control/campaign.json'))
        rounds = state.get('rounds')
        if not isinstance(rounds, list) or any(not isinstance(r, dict) for r in rounds):
            raise ValueError('Missing classified campaign rounds')
        if rounds and rounds[-1].get('status') not in ROUND_TERMINAL:
            raise ValueError('Prepared, generating or nonterminal round blocks library retirement')
        if len({r.get('round') for r in rounds}) != len(rounds):
            raise ValueError('Duplicate round identities')
        snapshot = self.history.gc_snapshot()
        candidates = {item['candidate'] for item in snapshot['candidates']}
        if (candidates != set(snapshot['archive_candidates'])
                or self.task['optimization']['baseline'] not in candidates):
            raise ValueError('Every admitted candidate must have authenticated retained history')
        mounts = live_docker_mounts()
        for mount in mounts:
            source = Path(mount['source'])
            if source == self.root or source in self.root.parents or self.root in source.parents:
                raise ValueError('Live Docker campaign mount blocks library retirement')
        db_path = self._path('control/archives/' + self.task_id + '.sqlite')
        with sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True) as db:
            selections = [list(r) for r in db.execute('SELECT * FROM selections ORDER BY id')]
            children = [list(r) for r in db.execute('SELECT * FROM children ORDER BY candidate')]
        return {'root': str(self.root), 'task_digest': self.task_id,
                'evaluator_sha256': self.task['evaluator_sha256'],
                'campaign_digest': digest(state), 'queue_digest': digest(rows),
                'history_digest': digest(snapshot), 'selections_digest': digest(selections),
                'children_digest': digest(children), 'mounts': mounts}, state

    def _manifest(self, relative, *, removed=(), notice=None):
        folder = self._path(relative)
        if not LIBRARY.fullmatch(folder.name) or folder.parent != self._path('libraries'):
            raise ValueError('Only exact historical round libraries may retire')
        manifest_path = self._path(relative + '/manifest.json')
        manifest = read(manifest_path)
        inventory = manifest.get('artifacts')
        if manifest.get('schema_version') != 1 or not isinstance(inventory, dict) or not inventory:
            raise ValueError('Invalid original reading manifest')
        index = read(self._path(relative + '/index.json'))
        prompt = 'control/rounds/' + folder.name + '/prompt'
        bundle_path = self._path(prompt + '/bundle.json')
        bundle = read(bundle_path)
        context = bundle.get('context', {})
        prompt_manifest_path = self._path(prompt + '/manifest.json')
        prompt_manifest = read(prompt_manifest_path)
        tool_path = self._path('control/rounds/' + folder.name + '/tool-config.json')
        tool = read(tool_path)
        if (context.get('task_digest') != self.task_id
                or context.get('candidate_library', {}).get('manifest_sha256') != sha(manifest_path)
                or prompt_manifest.get('bundle_digest') != digest(bundle)
                or prompt_manifest.get('artifacts', {}).get('bundle.json')
                   != {'bytes': bundle_path.stat().st_size, 'sha256': sha(bundle_path)}
                or tool.get('bundle_digest') != digest(bundle)):
            raise ValueError('Historical library differs from its original host prompt binding')
        if index.get('schema_version') not in {1, 2} or index.get('task_digest') != self.task_id:
            raise ValueError('Historical library task/schema differs')
        entries = index.get('candidates')
        if (not isinstance(entries, list) or not entries
                or len({e.get('candidate') for e in entries}) != len(entries)
                or any(not isinstance(e.get('candidate'), str) or not HEX.fullmatch(e['candidate']) for e in entries)):
            raise ValueError('Invalid historical candidate index')
        candidates = {e['candidate'] for e in entries}
        removed = set(removed)
        if any(name not in inventory or not _bulk_log(name) for name in removed):
            raise ValueError('Retired path is not an original bulk log')
        actual = set()
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symlink in historical library')
            if path.is_file():
                actual.add(path.relative_to(folder).as_posix())
        expected = set(inventory) - removed | {'manifest.json'}
        if notice is not None:
            expected.add(NOTICE)
        if actual != expected:
            raise ValueError('Historical library inventory differs or has unclassified files')
        for name, metadata in inventory.items():
            _relative(name)
            parts = Path(name).parts
            if len(parts) > 1 and parts[0] not in candidates:
                raise ValueError('Manifest contains an unindexed candidate or unknown subtree')
            if name in removed:
                if self._path(relative + '/' + name).exists():
                    raise ValueError('Retired log unexpectedly reappeared')
                continue
            path = self._path(relative + '/' + name)
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444
                    or metadata != {'bytes': info.st_size, 'sha256': sha(path)}):
                raise ValueError('Original historical artifact changed: ' + name)
        if notice is not None and self._path(relative + '/' + NOTICE).read_text() != notice:
            raise ValueError('Historical reading notice changed')
        return {'manifest_sha256': sha(manifest_path), 'index_sha256': sha(self._path(relative + '/index.json')),
                'schema_version': index['schema_version'], 'inventory_digest': digest(inventory),
                'artifact_count': len(inventory), 'candidates': sorted(candidates),
                'prompt_bundle_sha256': sha(bundle_path), 'prompt_manifest_sha256': sha(prompt_manifest_path),
                'tool_config_sha256': sha(tool_path)}, inventory

    def _notice(self, library, plan_key, receipt):
        return ('# Historical library logs retired\n\n'
                'This historical reading library is intentionally incomplete and must not be used as a current reading bundle.\n'
                'The original manifest, index, source, diffs, metrics and log indexes remain unchanged.\n'
                'Duplicate bulk logs were retired after authenticated history sealing. Full original best-so-far logs remain in the host trace.\n'
                'The original strict library loader is expected to reject this directory. Use the newly prepared compact reading library.\n\n'
                'Library: ' + library + '\nPlan: ' + plan_key + '\nHost receipt: ' + receipt + '\n')

    def _journal(self, path, key, files):
        count, released, chain = 0, 0, None
        names = list(files)
        by_inode = {}
        with path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    raise ValueError('Incomplete library retirement journal line')
                envelope = json.loads(line)
                event = self.history.verify_host_record(envelope, 'gc-progress')
                if count >= len(names):
                    raise ValueError('Extra library retirement journal event')
                name = names[count]
                expected = files[name]
                pair = (expected['device'], expected['inode'])
                adjusted = dict(expected, links=expected['links'] - by_inode.get(pair, 0))
                freed = adjusted['blocks'] * 512 if adjusted['links'] == 1 else 0
                if (event.get('kind') != 'library-log-unlinked' or event.get('plan_digest') != key
                        or event.get('sequence') != count + 1 or event.get('path') != name
                        or event.get('file') != adjusted or event.get('released_blocks_bytes') != freed
                        or event.get('previous_event_sha256') != chain):
                    raise ValueError('Library retirement journal event/chain differs')
                chain = digest(envelope)
                by_inode[pair] = by_inode.get(pair, 0) + 1
                count += 1
                released += freed
        if count != len(names):
            raise ValueError('Incomplete library retirement journal inventory')
        return {'count': count, 'released_blocks_bytes': released, 'last_event_sha256': chain}

    def _existing(self):
        retired, checked_journals = {}, {}
        folder = self._path(CONTROL + '/receipts')
        if not folder.exists():
            return retired
        for path in sorted(folder.glob('*.json')):
            receipt = self.history.verify_host_record(read(path), 'gc-receipt')
            if receipt.get('kind') != 'historical-reading-library-retired':
                raise ValueError('Unexpected library receipt domain')
            relative = receipt['library']
            if relative in retired:
                raise ValueError('Duplicate library retirement receipts')
            plan_path = self._path(receipt['plan_path'])
            if sha(plan_path) != receipt['plan_sha256']:
                raise ValueError('Original library retirement plan changed')
            plan = self.history.verify_host_record(read(plan_path), 'gc-plan')
            if plan.get('kind') != 'exact-library-log-retirement' or digest(plan) != receipt['plan_digest']:
                raise ValueError('Historical library plan binding changed')
            selected = {p[len(relative) + 1:] for p in plan['files'] if p.startswith(relative + '/')}
            notice = self._notice(relative, receipt['plan_digest'], path.relative_to(self.root).as_posix())
            metadata, _ = self._manifest(relative, removed=selected, notice=notice)
            if metadata != receipt['original_library']:
                raise ValueError('Preserved historical library changed')
            journal = self._path(receipt['journal_path'])
            if sha(journal) != receipt['journal_sha256']:
                raise ValueError('Library retirement journal changed')
            if str(journal) not in checked_journals:
                checked_journals[str(journal)] = self._journal(journal, receipt['plan_digest'], plan['files'])
            retired[relative] = {'receipt_path': path.relative_to(self.root).as_posix(),
                                 'receipt_sha256': sha(path), 'metadata': metadata}
        return retired

    def plan(self):
        """Plan only original manifest-listed duplicate logs in completed rounds."""
        with self._locked():
            binding, state = self._binding()
            retired = self._existing()
            # A crash is not permission to guess which unlink committed.
            if any(self.control.glob('*.intent.json')):
                raise ValueError('Unresolved library retirement intent requires audited reconciliation')
            for journal in self.control.glob('*.journal.jsonl'):
                if not journal.with_name(journal.name[:-len('.journal.jsonl')] + '.receipt.json').exists():
                    raise ValueError('Incomplete library retirement journal requires audited reconciliation')
            rounds = {r['round']: r for r in state['rounds']}
            libraries, files, skipped = {}, {}, []
            parent = self._path('libraries')
            for folder in sorted(parent.iterdir()) if parent.exists() else []:
                relative = folder.relative_to(self.root).as_posix()
                if not folder.is_dir() or folder.is_symlink() or not LIBRARY.fullmatch(folder.name):
                    raise ValueError('Unclassified library entry stays pinned')
                row = rounds.get(int(folder.name))
                if row is None or row.get('status') not in ROUND_TERMINAL:
                    skipped.append(relative)
                    continue
                if relative in retired:
                    continue
                metadata, inventory = self._manifest(relative)
                selected = {name: _file(self._path(relative + '/' + name)) for name in inventory if _bulk_log(name)}
                if not selected:
                    skipped.append(relative)
                    continue
                libraries[relative] = metadata
                for name, item in selected.items():
                    files[relative + '/' + name] = item
            body = {'version': 1, 'kind': 'exact-library-log-retirement', 'snapshot': binding,
                    'libraries': libraries, 'files': dict(sorted(files.items())),
                    'previous_retirements': retired, 'skipped': skipped, 'created_at': time.time(),
                    'budget_changed': False, 'source_changed': False}
            target = self._path(CONTROL + '/' + digest(body) + '.plan.json')
            write(target, self.history.sign_host_record(body, 'gc-plan'))
            return {'path': target.relative_to(self.root).as_posix(), 'sha256': sha(target),
                    'file_count': len(files), 'library_count': len(libraries),
                    'logical_bytes': sum(r['bytes'] for r in files.values())}

    def apply(self, relative, *, expected_sha256):
        """One exact plan; append-only signed journal, no quadratic progress rewrite."""
        with self._locked():
            target = self._path(relative)
            if target.parent != self.control or not target.name.endswith('.plan.json') or sha(target) != expected_sha256:
                raise ValueError('Published library plan/hash authorization differs')
            body = self.history.verify_host_record(read(target), 'gc-plan')
            key = digest(body)
            if body.get('kind') != 'exact-library-log-retirement' or target.name != key + '.plan.json':
                raise ValueError('Library plan content address/domain differs')
            receipt_path = self._path(CONTROL + '/' + key + '.receipt.json')
            journal_path = self._path(CONTROL + '/' + key + '.journal.jsonl')
            intent_path = self._path(CONTROL + '/' + key + '.intent.json')
            if receipt_path.exists():
                result = self.history.verify_host_record(read(receipt_path), 'gc-receipt')
                if (result.get('kind') != 'exact-library-log-retirement-complete'
                        or result.get('plan_sha256') != expected_sha256
                        or sha(journal_path) != result['journal_sha256']):
                    raise ValueError('Completed library retirement receipt differs')
                self._existing()
                return result
            if intent_path.exists() or journal_path.exists():
                raise ValueError('Unresolved library retirement intent/journal; reconcile before a new attempt')
            binding, _ = self._binding()
            if binding != body['snapshot'] or self._existing() != body['previous_retirements']:
                raise ValueError('Campaign, queue, history or prior retirement changed after planning')
            for library, original in body['libraries'].items():
                if self._manifest(library)[0] != original:
                    raise ValueError('Original library changed after planning')
            for path, expected in body['files'].items():
                library = '/'.join(Path(path).parts[:2])
                internal = '/'.join(Path(path).parts[2:])
                if library not in body['libraries'] or not _bulk_log(internal) or _file(self._path(path)) != expected:
                    raise ValueError('Planned library log identity/content/link count changed')
            # New notices are explicitly outside the original inventory. They do
            # not amend its manifest and make incompleteness visible immediately.
            for library in body['libraries']:
                receipt = CONTROL + '/receipts/' + library.split('/')[-1] + '-' + key + '.json'
                notice = self._path(library + '/' + NOTICE)
                with notice.open('x') as stream:
                    stream.write(self._notice(library, key, receipt))
                    stream.flush()
                    os.fsync(stream.fileno())
                notice.chmod(0o444)
            removed, released, chain = 0, 0, None
            by_inode = {}
            for item in body['files'].values():
                pair = (item['device'], item['inode'])
                by_inode.setdefault(pair, 0)
            with journal_path.open('xb', buffering=0) as journal:
                for path, expected in body['files'].items():
                    current = _file(self._path(path))
                    pair = (expected['device'], expected['inode'])
                    adjusted = dict(expected, links=expected['links'] - by_inode[pair])
                    if current != adjusted:
                        raise ValueError('Library log identity/content/link count changed during retirement')
                    operation = {'kind': 'library-log-unlink-intent', 'plan_digest': key,
                                 'sequence': removed + 1, 'path': path, 'file': current}
                    write(intent_path, self.history.sign_host_record(operation, 'gc-intent'))
                    self._path(path).unlink()
                    _fsync_directory(self._path(path).parent)
                    freed = current['blocks'] * 512 if current['links'] == 1 else 0
                    event = {'kind': 'library-log-unlinked', 'plan_digest': key,
                             'sequence': removed + 1, 'path': path, 'file': current,
                             'released_blocks_bytes': freed, 'previous_event_sha256': chain}
                    envelope = self.history.sign_host_record(event, 'gc-progress')
                    line = json.dumps(envelope, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
                    if journal.write(line) != len(line):
                        raise OSError('Incomplete retirement journal write; preserve intent for reconciliation')
                    os.fsync(journal.fileno())
                    chain = digest(envelope)
                    removed += 1
                    released += freed
                    by_inode[pair] += 1
                    intent_path.unlink()
                    _fsync_directory(self.control)
            journal_state = self._journal(journal_path, key, body['files'])
            if journal_state['count'] != removed or journal_state['released_blocks_bytes'] != released:
                raise ValueError('Library retirement journal totals differ')
            journal_sha = sha(journal_path)
            # A final snapshot re-authenticates all independent history/source
            # evidence; only historical library duplicate paths changed.
            if self._binding()[0] != binding:
                raise ValueError('Host evidence changed during library retirement')
            for library, original in body['libraries'].items():
                name = library.split('/')[-1] + '-' + key + '.json'
                receipt = CONTROL + '/receipts/' + name
                selected = {p[len(library) + 1:] for p in body['files'] if p.startswith(library + '/')}
                notice = self._notice(library, key, receipt)
                if self._manifest(library, removed=selected, notice=notice)[0] != original:
                    raise ValueError('Preserved library artifacts changed during retirement')
                per_library = {'version': 1, 'kind': 'historical-reading-library-retired',
                               'library': library, 'original_library': original, 'removed_count': len(selected),
                               'plan_path': relative, 'plan_sha256': expected_sha256, 'plan_digest': key,
                               'journal_path': journal_path.relative_to(self.root).as_posix(),
                               'journal_sha256': journal_sha, 'completed_at': time.time(),
                               'verification_scope': 'historical library intentionally incomplete; original strict loader must fail'}
                write(self._path(receipt), self.history.sign_host_record(per_library, 'gc-receipt'))
            result = {'version': 1, 'kind': 'exact-library-log-retirement-complete',
                      'plan_digest': key, 'plan_sha256': expected_sha256, 'removed_count': removed,
                      'library_count': len(body['libraries']), 'released_blocks_bytes': released,
                      'journal_sha256': journal_sha, 'completed_at': time.time(),
                      'budget_changed': False, 'source_changed': False}
            write(receipt_path, self.history.sign_host_record(result, 'gc-receipt'))
            return result
