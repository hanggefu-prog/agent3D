"""Host-only, exact-file retirement after authenticated historical compaction.

This module does not run an evaluator, admit a candidate or alter a receipt. It
requires the production retention authorization, a fully idle campaign queue,
and a successful live Docker mount inventory. Plans are HMAC sealed, bound to
the full queue/archive/selection/pin snapshot, and rechecked before unlinking.
Unknown evidence and unsupported artifacts remain pinned. Original manifests
are retained even when their explicitly inventoried raw artifacts are retired.
"""
from contextlib import contextmanager, ExitStack, closing
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import time

from grt_bench.common import confined, digest, read, sha, write
from grt_bench.contract import validate
from grt_bench.pareto import frontier

CONTROL = 'control/trace-retention/gc'
TERMINAL = {'succeeded', 'failed', 'cancelled', 'interrupted'}
ROUND_TERMINAL = {'verified', 'infeasible', 'failed', 'rejected', 'generation_failed',
                  'no_change', 'stalled', 'blocked'}
STAGES = {'input': '00_input', 'floorplan': '10_floorplan', 'partition': '20_partition',
          'placement': '30_placement', 'pdn': '40_pdn', 'hbt': '50_hbt',
          'opt': '60_opt', 'cts': '70_cts', 'grt': '80_route'}
ORDER = list(STAGES)
HEX = re.compile(r'[0-9a-f]{64}')


def _identity(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError('Invalid immutable job/candidate identity')
    return value


def _relative(value):
    if (not isinstance(value, str) or not value or '\\' in value
            or value.startswith('/') or Path(value).as_posix() != value
            or any(p in {'', '.', '..'} for p in value.split('/'))
            or any(ord(c) < 32 for c in value)):
        raise ValueError('A canonical relative artifact path is required')
    return value


def _file(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('Only regular immutable files may retire: ' + str(path))
    return {'bytes': s.st_size, 'sha256': sha(path), 'mode': stat.S_IMODE(s.st_mode),
            'device': s.st_dev, 'inode': s.st_ino, 'links': s.st_nlink,
            'blocks': s.st_blocks}


def live_docker_mounts():
    """An unavailable or malformed inventory is never permission to delete."""
    query = subprocess.run(['docker', 'ps', '-q'], check=True, capture_output=True,
                           text=True, timeout=30)
    ids = query.stdout.split()
    if not ids:
        return []
    data = subprocess.run(['docker', 'inspect', *ids], check=True, capture_output=True,
                          text=True, timeout=30)
    values = json.loads(data.stdout)
    if not isinstance(values, list) or len(values) != len(ids):
        raise ValueError('Incomplete live Docker container inventory')
    mounts = []
    for item in values:
        if not isinstance(item.get('Mounts'), list) or not item.get('Id'):
            raise ValueError('Malformed live Docker mount inventory')
        for mount in item['Mounts']:
            source = mount.get('Source')
            if not isinstance(source, str) or not source.startswith('/'):
                raise ValueError('Unknown live Docker mount source')
            mounts.append({'container': item['Id'], 'source': str(Path(source).resolve())})
    return sorted(mounts, key=lambda r: (r['container'], r['source']))


class RetentionGC:
    """Production host service; use only between terminal rounds, never in workers."""

    def __init__(self, root, task):
        from .grt_history_retention import RetainedHistory
        self.history = RetainedHistory(root, task)
        self.root = self.history.root
        self.task = copy.deepcopy(task)
        self.task_id = validate(task)
        self.control = confined(self.root, CONTROL)
        self.control.mkdir(parents=True, exist_ok=True)

    def _path(self, relative):
        path = confined(self.root, _relative(relative))
        if path.exists() and path.stat().st_dev != self.root.stat().st_dev:
            raise ValueError('Nested mounted filesystem in retirement evidence')
        return path

    @contextmanager
    def _locked(self):
        with ExitStack() as stack:
            for relative in ('control/campaign.lock', 'control/result-storage/storage.lock'):
                p = self._path(relative)
                p.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(p.open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stack.enter_context(self.history.locked())
            yield

    def _queue(self):
        with closing(sqlite3.connect(self._path('control/queue.sqlite').as_uri() + '?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY id')]

    def _archive(self):
        p = self._path('control/archives/' + self.task_id + '.sqlite')
        with closing(sqlite3.connect(p.as_uri() + '?mode=ro', uri=True)) as db:
            task = db.execute("SELECT value FROM metadata WHERE key='task'").fetchone()
            if not task or json.loads(task[0]) != self.task:
                raise ValueError('Archive task identity differs')
            evaluations = [(r[0], json.loads(r[1])) for r in db.execute('SELECT rowid,row_json FROM evaluations ORDER BY rowid')]
            selections = [list(r) for r in db.execute('SELECT * FROM selections ORDER BY id')]
            children = [list(r) for r in db.execute('SELECT * FROM children ORDER BY candidate')]
        return {'evaluations': evaluations, 'selections': selections, 'children': children}

    def _idle(self, rows):
        if any(row['state'] not in TERMINAL for row in rows):
            raise ValueError('Active, uncertain or unknown queue work blocks retirement')
        state = read(self._path('control/campaign.json'))
        rounds = state.get('rounds')
        if not isinstance(rounds, list) or (rounds and rounds[-1].get('status') not in ROUND_TERMINAL):
            raise ValueError('Campaign has prepared, live or unclassified work')
        return state

    def _trusted(self, job, row):
        """Check immutable original independent receipts, never a guessed call graph."""
        plan_path = self._path('runs/' + job + '/control/plan.json')
        plan = read(plan_path)
        verified = read(self._path('runs/' + job + '/verified/result.json'))
        payload = json.loads(row['payload'])
        for key in ('candidate', 'case', 'seed', 'task_digest', 'split', 'endpoint'):
            if plan.get(key) != payload.get(key) or verified.get(key) != plan.get(key):
                raise ValueError('Queue/plan/independent verifier identity differs: ' + job)
        if (verified.get('status') != 'verified'
                or verified.get('evidence', {}).get('plan_sha256') != sha(plan_path)):
            raise ValueError('Independent verifier plan hash differs: ' + job)
        proof_path = self._path('runs/' + job + '/control/incremental.json')
        if bool(plan.get('incremental')) != proof_path.exists():
            raise ValueError('Unplanned or missing incremental proof: ' + job)
        proof = read(proof_path) if proof_path.exists() else None
        if (proof is not None and verified.get('incremental') != proof
                or proof is None and 'incremental' in verified):
            raise ValueError('Incremental proof differs from original verifier: ' + job)
        if bool(plan.get('rescore')) != ('physical_replay' in verified):
            raise ValueError('Unverified physical replay: ' + job)
        if plan.get('rescore') and verified['physical_replay'] != plan['rescore']:
            raise ValueError('Physical replay differs from original verifier: ' + job)
        return {'plan': plan, 'verified': verified, 'proof': proof}

    def _dependencies(self, rows):
        """Return paired, verified replay aliases and exact stage-prefix edges."""
        by_id = {row['id']: row for row in rows}
        trusted = {}
        for job, row in by_id.items():
            _identity(job)
            if row['state'] == 'succeeded' and json.loads(row['payload']).get('kind', 'evaluation') == 'evaluation':
                trusted[job] = self._trusted(job, row)
        resolving = set()
        aliases = {}
        def physical(job):
            _identity(job)
            if job in aliases:
                return aliases[job]
            if job not in trusted or job in resolving:
                raise ValueError('Missing or cyclic verified physical producer')
            resolving.add(job)
            value = trusted[job]
            replay = value['plan'].get('rescore')
            result = job
            if replay:
                old = _identity(replay['source_job'])
                p = self._path('runs/' + old + '/control/plan.json')
                if sha(p) != replay.get('previous_plan_sha256'):
                    raise ValueError('Physical replay producer hash differs')
                # Old-task replay is allowed only when the original transition
                # validator confirms the exact measured case/seed/endpoint.
                from grt_bench.rescore import validate_transition
                validate_transition(read(self._path('runs/' + job + '/control/task.json')),
                    read(self._path('runs/' + old + '/control/task.json')),
                    value['plan'], trusted[old]['plan'])
                result = physical(old)
            resolving.remove(job)
            aliases[job] = result
            return result
        edges = {}
        evidence = {}
        for job, value in trusted.items():
            physical(job)
            proof = value['proof']
            if not proof:
                continue
            source = _identity(proof['source_job'])
            producer = _identity(proof['physical_source_job'])
            if source == job or producer == job or source not in trusted or physical(source) != producer:
                raise ValueError('Incremental physical source alias differs')
            parent = trusted[source]
            if (proof.get('version') != 1 or proof.get('task_digest') != value['plan']['task_digest']
                    or proof.get('parent') != parent['plan']['candidate']
                    or proof.get('parent_plan_digest') != digest(parent['plan'])
                    or proof.get('parent_verified_digest') != digest(parent['verified'])
                    or any(parent['plan'].get(k) != value['plan'].get(k)
                           for k in ('task_digest', 'case', 'seed', 'split', 'endpoint'))):
                raise ValueError('Incremental parent is not the verified paired cell')
            start = proof.get('impact', {}).get('start')
            if start not in ORDER or set(proof.get('stages', {})) != set(ORDER[:ORDER.index(start)]):
                raise ValueError('Incremental prefix is incomplete or unknown')
            edges[job] = {STAGES[s]: producer for s in proof['stages']}
            for stage, checksum in proof['stages'].items():
                for origin in (job, producer):
                    relative = 'runs/' + origin + '/output/run/' + STAGES[stage] + '/stage_manifest.json'
                    if sha(self._path(relative)) != checksum:
                        raise ValueError('Incremental stage producer manifest differs')
                    evidence[relative] = checksum
        def acyclic(job, folder, active):
            key = (job, folder)
            if key in active:
                raise ValueError('Cyclic incremental stage provenance')
            parent = edges.get(job, {}).get(folder)
            if parent is not None:
                acyclic(parent, folder, active | {key})
        for job, folders in edges.items():
            for folder in folders:
                acyclic(job, folder, set())
        return trusted, aliases, edges, evidence

    def _snapshot(self, selected_candidates=(), extra_pins=()):
        rows = self._queue()
        campaign = self._idle(rows)
        archive = self._archive()
        compact = self.history.gc_snapshot()
        histories = {r['candidate']: r for r in compact['candidates']}
        if set(compact['archive_candidates']) - set(histories):
            raise ValueError('Seal every admitted matrix before retirement')
        trusted, aliases, edges, dependencies = self._dependencies(rows)
        keep_candidates = {self.task['optimization']['baseline']}
        keep_candidates.update(r['candidate'] for r in frontier([r for _, r in archive['evaluations']], retain_equivalent=True))
        keep_candidates.update(_identity(c) for c in selected_candidates)
        if campaign['rounds'] and campaign['rounds'][-1].get('parent'):
            keep_candidates.add(_identity(campaign['rounds'][-1]['parent']))
        pins = set(compact['protected_receipts']) | set(compact['protected_blobs'])
        pins.update(_relative(p) for p in extra_pins)
        mounts = live_docker_mounts()
        for mount in mounts:
            source = Path(mount['source'])
            if source == self.root or source in self.root.parents:
                raise ValueError('A live container mounts the whole campaign')
            if self.root in source.parents:
                pins.add(source.relative_to(self.root).as_posix())
        # Fully retain baseline/frontier/held-out, failed, diagnostic, unknown
        # and unsealed candidates. Compacting a successful candidate is the
        # only authority permitting its raw output to retire.
        roots = set()
        for row in rows:
            payload = json.loads(row['payload'])
            if (row['state'] != 'succeeded' or payload.get('kind', 'evaluation') != 'evaluation'
                    or payload.get('task_digest') != self.task_id or payload.get('split') != 'train'
                    or payload.get('candidate') in keep_candidates or payload.get('candidate') not in histories):
                pins.add('runs/' + row['id'])
                if row['id'] in trusted:
                    roots.add((aliases[row['id']], None))
        # A retained current checkpoint also needs its exact reusable upstream
        # stage producers, including recursive aliases. Pin whole stages, not
        # just JSON/ODB: their declared artifact hashes must remain verifiable.
        seen = set()
        def visit(job, folders):
            key = (job, None if folders is None else tuple(sorted(folders)))
            if key in seen:
                return
            seen.add(key)
            if job not in trusted:
                raise ValueError('Retained checkpoint has no verified producer')
            if aliases[job] != job:
                visit(aliases[job], folders)
            targets = set(STAGES.values()) if folders is None else set(folders)
            pins.add('runs/' + job + '/output' if folders is None else 'runs/' + job + '/control')
            for folder in targets:
                if folders is not None:
                    pins.add('runs/' + job + '/output/run/' + folder)
                producer = edges.get(job, {}).get(folder)
                if producer is not None:
                    visit(producer, {folder})
        for job, folders in roots:
            visit(job, folders)
        compact_rounds = [{'number': r['round'], 'candidate': r['candidate']}
                          for r in campaign['rounds']
                          if type(r.get('round')) is int and r.get('status') in {'verified', 'infeasible'}
                          and r.get('candidate') in histories
                          and not histories[r['candidate']]['record_setter']]
        history_hashes = {candidate: sha(self._path('control/trace-retention/history/' + candidate + '.json'))
                          for candidate in histories}
        return {'version': 1, 'root': str(self.root), 'task_digest': self.task_id,
                'queue_digest': digest(rows), 'archive_digest': digest(archive),
                'campaign_digest': digest(campaign), 'history_hashes': history_hashes,
                'pins': sorted(pins), 'keep_candidates': sorted(keep_candidates),
                'dependencies': dependencies, 'mounts': mounts, 'compact_rounds': compact_rounds,
                'selected_candidates': sorted(selected_candidates), 'extra_pins': sorted(extra_pins)}, rows, histories

    def _eligible(self, relative, row, snapshot, histories):
        path = self._path(relative)
        parts = Path(relative).parts
        if (len(parts) < 4 or parts[0] != 'runs' or parts[1] != row['id']
                or parts[2] not in {'output', 'execution'}):
            return False
        if any(relative == p or relative.startswith(p + '/') for p in snapshot['pins']):
            return False
        candidate = json.loads(row['payload']).get('candidate')
        history = histories.get(candidate)
        if not history or row['id'] not in {r['id'] for r in history['queue_rows']}:
            return False
        raw = path.suffix.lower() in {'.log', '.rpt', '.odb', '.def', '.spef', '.guide'} or path.name.endswith('.spef.gz')
        if not raw or (parts[2] == 'execution' and path.suffix.lower() != '.log'):
            return False
        if history['record_setter'] and path.suffix.lower() in {'.log', '.rpt'}:
            if relative not in {r['path'] for r in history['logs']}:
                return False
        return True

    def _round_paths(self, snapshot):
        from .grt_retention_lifecycle import prunable_round_texts
        result = set()
        for row in snapshot['compact_rounds']:
            result.update(prunable_round_texts(self.root, {'round': row['number'],
                'candidate': row['candidate'], 'status': 'verified'}))
        return result

    def _eligible_path(self, relative, by_job, snapshot, histories):
        parts = Path(relative).parts
        if parts and parts[0] == 'runs':
            row = by_job.get(parts[1]) if len(parts) > 1 else None
            return row is not None and self._eligible(relative, row, snapshot, histories)
        if any(relative == p or relative.startswith(p + '/') for p in snapshot['pins']):
            return False
        # Round paths are selected by the shared tight diagnostic collector at
        # planning time. Its exact bytes are then bound by the host plan; only
        # non-record, sealed terminal rounds may use these two namespaces.
        for row in snapshot['compact_rounds']:
            number = '%04d' % row['number']
            prefixes = ('control/rounds/' + number + '/agent/',
                        'control/rounds/' + number + '/gpu-checks/',
                        'scratch/' + number + '/')
            if any(relative.startswith(prefix) for prefix in prefixes):
                return True
        return False

    def plan(self, *, selected_candidates=(), extra_pins=(), paths=None):
        """Publish a sealed exact plan; None enumerates only recognized raw files."""
        with self._locked():
            snapshot, rows, histories = self._snapshot(selected_candidates, extra_pins)
            by_job = {r['id']: r for r in rows}
            explicit = paths is not None
            round_paths = self._round_paths(snapshot)
            if paths is None:
                paths = []
                for row in rows:
                    if json.loads(row['payload']).get('candidate') not in histories:
                        continue
                    for sub in ('output', 'execution'):
                        parent = self._path('runs/' + row['id'] + '/' + sub)
                        if not parent.exists():
                            continue
                        for p in sorted(parent.rglob('*')):
                            if p.is_symlink():
                                continue  # Preserve aliases verbatim, including dangling/external ones.
                            if p.is_file():
                                paths.append(p.relative_to(self.root).as_posix())
                paths.extend(sorted(round_paths))
            if len(paths) != len(set(paths)):
                raise ValueError('Duplicate explicit retirement path')
            files = {}
            for relative in sorted(paths):
                _relative(relative)
                if (not self._eligible_path(relative, by_job, snapshot, histories)
                        or (not relative.startswith('runs/') and relative not in round_paths)):
                    if explicit:
                        raise ValueError('Pinned or unclassified retirement path: ' + relative)
                    continue
                files[relative] = _file(self._path(relative))
            body = {'version': 1, 'kind': 'exact-raw-retirement', 'snapshot': snapshot,
                    'files': files, 'created_at': time.time(), 'budget_changed': False}
            envelope = self.history.sign_host_record(body, kind='gc-plan')
            target = self._path(CONTROL + '/' + digest(body) + '.plan.json')
            write(target, envelope)
            return {'path': target.relative_to(self.root).as_posix(), 'sha256': sha(target),
                    'plan': envelope, 'file_count': len(files),
                    'logical_bytes': sum(f['bytes'] for f in files.values())}

    def _progress(self, key, files):
        """Linear signed per-path journal; no quadratic rewriting of path lists."""
        folder = self._path(CONTROL + '/' + key + '.steps')
        removed, released, previous = [], 0, None
        if not folder.exists():
            return removed, released, previous
        expected_order = list(files)
        for number, path in enumerate(sorted(folder.iterdir()), 1):
            if path.name != '%08d.json' % number or path.is_symlink():
                raise ValueError('Unclassified or nonconsecutive retirement journal step')
            envelope = read(path)
            step = self.history.verify_host_record(envelope, kind='gc-progress')
            if (number > len(expected_order) or step.get('number') != number
                    or step.get('plan_digest') != key or step.get('previous') != previous
                    or step.get('removed') != expected_order[number - 1]
                    or type(step.get('released_blocks_bytes')) is not int
                    or step['released_blocks_bytes'] < 0):
                raise ValueError('Retirement journal chain or exact path differs')
            removed.append(step['removed'])
            released += step['released_blocks_bytes']
            previous = digest(envelope)
        return removed, released, previous

    def apply(self, plan_path, *, expected_sha256):
        """Apply once under identical dependencies; uncertain crashes stay blocked."""
        with self._locked():
            plan_path = self._path(plan_path)
            if plan_path.parent != self.control or not plan_path.name.endswith('.plan.json') or sha(plan_path) != expected_sha256:
                raise ValueError('Exact published plan/hash authorization differs')
            body = self.history.verify_host_record(read(plan_path), kind='gc-plan')
            key = digest(body)
            if plan_path.name != key + '.plan.json':
                raise ValueError('Host plan content address differs')
            receipt = self._path(CONTROL + '/' + key + '.receipt.json')
            if receipt.exists():
                result = self.history.verify_host_record(read(receipt), kind='gc-receipt')
                removed, released, _ = self._progress(key, body['files'])
                if (result.get('kind') != 'exact-raw-retirement-complete'
                        or result.get('plan_digest') != key or result.get('plan_sha256') != expected_sha256
                        or removed != list(body['files']) or result.get('removed') != removed
                        or result.get('released_blocks_bytes') != released
                        or result.get('logical_bytes') != sum(r['bytes'] for r in body['files'].values())
                        or result.get('budget_changed') is not False or result.get('source_changed') is not False
                        or self._path(CONTROL + '/' + key + '.intent.json').exists()):
                    raise ValueError('Completed retirement receipt/journal binding differs')
                if any(self._path(p).exists() for p in removed):
                    raise ValueError('Retired raw evidence reappeared after completion')
                return result
            previous = body['snapshot']
            snapshot, rows, histories = self._snapshot(previous['selected_candidates'], previous['extra_pins'])
            if snapshot != previous:
                raise ValueError('Queue, pins, mounts, history or dependency evidence changed after planning')
            by_job = {r['id']: r for r in rows}
            intent = self._path(CONTROL + '/' + key + '.intent.json')
            journal = self._path(CONTROL + '/' + key + '.steps')
            removed, released, previous_step = self._progress(key, body['files'])
            removed_set = set(removed)
            if intent.exists():
                uncertain = self.history.verify_host_record(read(intent), kind='gc-intent')
                if uncertain.get('plan_digest') != key:
                    raise ValueError('Retirement intent belongs to another plan')
                target = uncertain['path']
                if target not in removed_set:
                    raise ValueError('Unresolved retirement intent; preserve evidence and reconcile before continuing')
                if self._path(target).exists():
                    raise ValueError('Completed retirement intent path unexpectedly exists')
                intent.unlink()
            if len(removed) != len(removed_set) or any(p not in body['files'] for p in removed):
                raise ValueError('Invalid signed retirement progress')
            unlinked_inodes = Counter((body['files'][p]['device'], body['files'][p]['inode']) for p in removed)
            for relative, expected in body['files'].items():
                if not self._eligible_path(relative, by_job, snapshot, histories):
                    raise ValueError('Retirement target is now pinned')
                path = self._path(relative)
                if relative in removed_set:
                    if path.exists():
                        raise ValueError('Previously retired raw evidence reappeared')
                else:
                    adjusted = dict(expected)
                    adjusted['links'] -= unlinked_inodes[(expected['device'], expected['inode'])]
                    if _file(path) != adjusted:
                        raise ValueError('Planned raw artifact identity/content/link count changed')
            journal.mkdir(exist_ok=True)
            for relative, expected in body['files'].items():
                if relative in removed_set:
                    continue
                path = self._path(relative)
                current = _file(path)
                # Previous unlinks can lower nlink for another target in the
                # same exact plan. Count only the actual last link at unlink.
                if any(current[k] != expected[k] for k in ('bytes', 'sha256', 'mode', 'device', 'inode', 'blocks')):
                    raise ValueError('Artifact changed during retirement')
                write(intent, self.history.sign_host_record({'path': relative, 'file': current, 'plan_digest': key}, kind='gc-intent'))
                path.unlink()
                directory_fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                released_here = current['blocks'] * 512 if current['links'] == 1 else 0
                step = self.history.sign_host_record({'plan_digest': key,
                    'number': len(removed) + 1, 'removed': relative,
                    'released_blocks_bytes': released_here, 'previous': previous_step}, kind='gc-progress')
                write(journal / ('%08d.json' % (len(removed) + 1)), step)
                previous_step = digest(step)
                removed.append(relative)
                removed_set.add(relative)
                released += released_here
                intent.unlink()
            result = {'version': 1, 'kind': 'exact-raw-retirement-complete', 'plan_digest': key,
                      'plan_sha256': expected_sha256, 'removed': removed,
                      'logical_bytes': sum(body['files'][p]['bytes'] for p in removed),
                      'released_blocks_bytes': released, 'completed_at': time.time(),
                      'budget_changed': False, 'source_changed': False,
                      'preserved_manifest_scope': 'original manifests remain immutable; retired raw evidence is not freshly verifiable'}
            write(receipt, self.history.sign_host_record(result, kind='gc-receipt'))
            return result
