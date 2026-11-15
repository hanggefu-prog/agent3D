"""Opt-in, isolated-copy PROTOTYPE for best-so-far log retention.

This is a host-side pilot, NOT a replacement for Archive admission, a production
cleaner, a physical verifier, or a filesystem reservation service. No controller
imports it. A root must first be explicitly marked as an isolated pilot copy.

API: initialize -> seal terminal candidates -> plan_prune -> apply -> history.
The caller supplies a complete host pin snapshot, for example::

    {'version': 1, 'complete': True, 'root': str(root.resolve()),
     'task_digest': validate(task), 'generation': 'terminal-event-5',
     'paths': [{'path': 'runs/<job>/output/run/30_placement',
                'reason': 'reusable checkpoint and original stage logs'}],
     'candidates': [], 'jobs': []}

All pins, the archive and authoritative queue are rechecked before deletion.
Unclassified jobs and unknown queue states are never deletion permissions.
The host-owned HMAC key authenticates pilot records; a supplied content hash is
not accepted as authority. Keep this control directory out of agent mounts.
"""
from contextlib import contextmanager, closing
import copy
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import time
import uuid

from grt_bench.archive import Archive, completed_outcomes
from grt_bench.common import confined, digest, read, sha, write
from grt_bench.contract import validate
from grt_bench.pareto import aggregate
from .grt_candidate_library import _log_paths


MARKER = '.trace-retention-pilot.json'
CONTROL = 'control/trace-retention-pilot'
TERMINAL = {'succeeded', 'failed', 'cancelled', 'interrupted'}
STAGES = {'input': '00_input', 'floorplan': '10_floorplan',
          'partition': '20_partition', 'placement': '30_placement',
          'pdn': '40_pdn', 'hbt': '50_hbt', 'opt': '60_opt',
          'cts': '70_cts', 'grt': '80_route'}
HEX = re.compile(r'[0-9a-f]{64}')


def _identifier(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError('Invalid candidate/job identity')
    return value


def _root(value):
    path = Path(value).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Pilot root cannot traverse symlinks')
    path = path.resolve(strict=True)
    # A copy must be deliberately placed outside the actual campaign filesystem.
    if ('pilot' not in str(path).lower() or path == Path('/')
            or path == Path('/data-ssd') or Path('/data-ssd') in path.parents):
        raise ValueError('Use a clearly named isolated pilot copy outside /data-ssd')
    return path


def _canonical_relative(value):
    if (not isinstance(value, str) or not value or any(ord(c) < 32 for c in value)
            or '\\' in value or Path(value).as_posix() != value
            or value.startswith('/') or any(p in {'', '.', '..'} for p in value.split('/'))):
        raise ValueError('Use a canonical relative pilot path')
    return value


def _row_file(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('Only regular files can be sealed/pruned: ' + str(path))
    return {'bytes': s.st_size, 'sha256': sha(path), 'mode': stat.S_IMODE(s.st_mode)}


def _check(path, expected):
    if _row_file(path) != expected:
        raise ValueError('Sealed artifact changed: ' + str(path))


class TraceRetention:
    """One host, one stopped isolated copy; never starts EDA or spends budget."""

    @classmethod
    def initialize(cls, root, source_root, task, *, acknowledge_isolated_copy=False):
        if acknowledge_isolated_copy is not True:
            raise ValueError('Explicit isolated-copy acknowledgement is required')
        root, source = _root(root), Path(source_root).resolve(strict=True)
        if root == source or root in source.parents or source in root.parents:
            raise ValueError('Pilot and original campaign must be separate trees')
        task_id = validate(task)
        target = confined(root, CONTROL)
        if (root / MARKER).exists() or target.exists():
            raise FileExistsError('Pilot has already been initialized')
        for name in ('control/queue.sqlite', 'control/campaign.json'):
            a, b = root / name, source / name
            if a.exists() and b.exists() and os.path.samefile(a, b):
                raise ValueError('Mutable pilot controls cannot hard-link original controls')
        target.mkdir(parents=True)
        os.chmod(target, 0o700)
        key = target / 'host.key'
        with key.open('xb') as stream:
            stream.write(os.urandom(32))
        key.chmod(0o600)
        write(root / MARKER, {'version': 1, 'prototype': True, 'root': str(root),
            'source_root': str(source), 'task_digest': task_id, 'id': uuid.uuid4().hex})
        return cls(root, task)

    def __init__(self, root, task):
        self.root, self.task = _root(root), copy.deepcopy(task)
        self.task_id = validate(task)
        marker = read(confined(self.root, MARKER))
        if (marker.get('version') != 1 or marker.get('prototype') is not True
                or marker.get('root') != str(self.root)
                or marker.get('task_digest') != self.task_id
                or not marker.get('source_root') or marker['source_root'] == str(self.root)):
            raise ValueError('Missing or mismatched isolated pilot marker')
        self.control = confined(self.root, CONTROL)
        key = confined(self.root, CONTROL + '/host.key')
        if not key.is_file() or key.stat().st_mode & 0o077 or key.stat().st_nlink != 1:
            raise ValueError('Pilot requires an owner-only, unshared host seal key')
        self.key = key.read_bytes()
        if len(self.key) != 32:
            raise ValueError('Invalid host seal key')
        # Metadata itself is host-owned and not accepted from candidate input.
        if self.control.stat().st_mode & 0o077:
            raise ValueError('Host retention control must not be agent-readable')

    @contextmanager
    def _locked(self):
        self.__init__(self.root, self.task)  # recheck path and host trust boundary
        path = confined(self.root, CONTROL + '/retention.lock')
        with path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def _sign(self, body):
        return {'body': body, 'hmac_sha256': hmac.new(self.key, digest(body).encode(), hashlib.sha256).hexdigest()}

    def _verify(self, value):
        if not isinstance(value, dict) or set(value) != {'body', 'hmac_sha256'}:
            raise ValueError('Missing host retention seal')
        if not hmac.compare_digest(self._sign(value['body'])['hmac_sha256'], value['hmac_sha256']):
            raise ValueError('Host retention seal differs')
        return value['body']

    def _path(self, name):
        path = confined(self.root, _canonical_relative(name))
        if path.exists() and path.stat().st_dev != self.root.stat().st_dev:
            raise ValueError('Nested filesystem in pilot evidence')
        return path

    def _queue(self):
        path = self._path('control/queue.sqlite')
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY id')]

    def _archive(self):
        path = self._path('control/archives/' + self.task_id + '.sqlite')
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            task = db.execute("SELECT value FROM metadata WHERE key='task'").fetchone()
            if not task or json.loads(task[0]) != self.task:
                raise ValueError('Archive task differs')
            return [(r[0], json.loads(r[1])) for r in db.execute('SELECT rowid,row_json FROM evaluations ORDER BY rowid')]

    def _idle(self, rows):
        if any(r['state'] not in TERMINAL for r in rows):
            raise ValueError('Active, uncertain or unknown queue work stays pinned')
        campaign = self._path('control/campaign.json')
        if campaign.exists():
            state = read(campaign)
            terminal_rounds = {'verified', 'infeasible', 'failed', 'rejected',
                               'generation_failed', 'no_change', 'stalled'}
            rounds = state.get('rounds', [])
            if rounds and rounds[-1].get('status') not in terminal_rounds:
                raise ValueError('Active, uncertain or unclassified campaign work stays pinned')

    def _selected(self, rows, candidate):
        return [r for r in rows if json.loads(r['payload']).get('candidate') == candidate
                and json.loads(r['payload']).get('task_digest') == self.task_id
                and json.loads(r['payload']).get('split') == 'train']

    def _record_path(self, candidate):
        return self._path(CONTROL + '/history/' + _identifier(candidate) + '.json')

    def _baseline_record(self):
        baseline = self.task['optimization']['baseline']
        path = self._record_path(baseline)
        if not path.exists():
            raise ValueError('Seal the original baseline first')
        return self._verify(read(path))

    @staticmethod
    def _score(row):
        value = row.get('vector', {}).get('score')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('This prototype requires an explicit single training score')
        if set(row['vector']) != {'score'}:
            raise ValueError('Pareto tasks require a separately defined reporting trace')
        return value

    def _origins(self, job, stages=None, seen=None):
        """Original prefix producers, never copied-prefix execution claims."""
        job = _identifier(job)
        seen = set() if seen is None else set(seen)
        if job in seen:
            raise ValueError('Cyclic physical/log provenance')
        seen.add(job)
        attempt = self._path('runs/' + job)
        plan_path = self._path('runs/' + job + '/control/plan.json')
        plan = read(plan_path)
        verified = read(self._path('runs/' + job + '/verified/result.json'))
        if (verified.get('status') != 'verified'
                or verified.get('evidence', {}).get('plan_sha256') != sha(plan_path)
                or any(verified.get(k) != plan.get(k)
                       for k in ('candidate', 'case', 'seed', 'task_digest', 'split'))):
            raise ValueError('Original log producer plan/verifier receipt binding changed')
        proof_path = self._path('runs/' + job + '/control/incremental.json')
        planned_incremental = bool(plan.get('incremental'))
        if planned_incremental != proof_path.exists():
            raise ValueError('Incremental sidecar presence differs from the verified plan')
        proof = read(proof_path) if planned_incremental else None
        if (planned_incremental and (not isinstance(proof, dict) or verified.get('incremental') != proof)
                or not planned_incremental and 'incremental' in verified):
            raise ValueError('Incremental sidecar differs from the independently verified proof')
        if plan.get('rescore'):
            from grt_bench.incremental import physical_source
            if verified.get('physical_replay') != plan['rescore']:
                raise ValueError('Rescore provenance differs from the independent verifier receipt')
            physical, _ = physical_source(self.root, job)
            if physical == job:
                raise ValueError('Rescore did not resolve an original physical producer')
            yield from self._origins(physical, stages, seen)
            return
        reused = {}
        if proof is not None:
            physical = _identifier(proof['physical_source_job'])
            source = _identifier(proof['source_job'])
            source_plan = read(self._path('runs/' + source + '/control/plan.json'))
            if any(source_plan.get(k) != plan.get(k) for k in ('case', 'seed', 'split')):
                raise ValueError('Incremental prefix is not paired by full case/seed')
            for stage, checksum in proof['stages'].items():
                if stage not in STAGES:
                    raise ValueError('Unknown incremental prefix stage')
                folder = STAGES[stage]
                for producer in (job, physical):
                    manifest = self._path('runs/' + producer + '/output/run/' + folder + '/stage_manifest.json')
                    if sha(manifest) != checksum:
                        raise ValueError('Incremental stage manifest differs from its original producer')
                reused[folder] = physical
        requested = None if stages is None else set(stages)
        # Enumerating surviving logs is insufficient: an old producer may have
        # already lost exactly the reused stage's logs. Check every log/report
        # explicitly declared by its stage manifest before promising completeness.
        # Physical artifacts are deliberately not required by this history view.
        folders = (set(STAGES.values()) | {'90_report', '95_thermal'}) if requested is None else requested
        for folder in sorted(folders - set(reused)):
            stage_relative = 'runs/' + job + '/output/run/' + folder
            stage = self._path(stage_relative)
            manifest_path = self._path(stage_relative + '/stage_manifest.json')
            if not manifest_path.exists():
                if requested is not None:
                    raise ValueError('Original reused producer stage manifest is missing')
                continue
            artifacts = read(manifest_path).get('artifacts')
            if not isinstance(artifacts, dict):
                raise ValueError('Original producer stage artifact inventory is invalid')
            for name, expected in artifacts.items():
                if not isinstance(expected, dict):
                    raise ValueError('Original producer stage artifact record is invalid')
                relative = expected.get('path', name)
                if not isinstance(relative, str):
                    raise ValueError('Original producer stage artifact path is invalid')
                if Path(relative).suffix not in {'.log', '.rpt'}:
                    continue
                path = self._path(stage_relative + '/' + _canonical_relative(relative))
                if (not path.is_file() or _row_file(path)['bytes'] != expected.get('bytes')
                        or sha(path) != expected.get('sha256')):
                    raise ValueError('Original producer stage log/report is missing or changed')
        for source in _log_paths(attempt):
            relative = source.relative_to(self.root).as_posix()
            self._path(relative)
            parts = source.relative_to(attempt).parts
            stage = parts[2] if len(parts) > 3 and parts[:2] == ('output', 'run') else None
            if stage in reused or (requested is not None and stage not in requested):
                continue
            yield {'path': relative, 'job': job, 'stage': stage, 'reused': stages is not None}
        for producer in sorted(set(reused.values())):
            selected = {s for s, p in reused.items() if p == producer}
            if requested is not None:
                selected &= requested
            if selected:
                yield from self._origins(producer, selected, seen)

    def seal(self, candidate, event_id, *, receipt_paths=(), log_paths=()):
        """Strictly verify an already admitted matrix, then seal historical facts.

        Non-records store no new full-log copies. For record setters, explicit
        log_paths adds prompt/model/tool/local-check files to the native bundle.
        This function neither ingests candidates nor changes budget receipts.
        """
        _identifier(candidate)
        if not isinstance(event_id, str) or not event_id.strip() or len(event_id) > 200:
            raise ValueError('Record one explicit terminal event identity')
        with self._locked():
            target = self._record_path(candidate)
            if target.exists():
                existing = self._verify(read(target))
                if existing['event_id'] != event_id:
                    raise ValueError('Candidate already sealed for another terminal event')
                return self._history(existing)
            archive_rows, queue_rows = self._archive(), self._queue()
            self._idle(queue_rows)
            matches = [(order, row) for order, row in archive_rows if row['candidate'] == candidate]
            if len(matches) != 1:
                raise ValueError('Only a previously admitted verified candidate can be sealed')
            order, evaluation = matches[0]
            score = self._score(evaluation)
            baseline = self.task['optimization']['baseline']
            if candidate != baseline:
                self._baseline_record()
            archive = Archive(self.root, self.task)
            audit = archive.admit(candidate)
            outcomes = completed_outcomes(self.root, self.task, candidate, audit)
            reference = outcomes if candidate == baseline else self._baseline_record()['outcomes']
            if aggregate(self.task, candidate, outcomes, reference) != evaluation:
                raise ValueError('Archived row does not match the independently verified measurements')
            earlier = [r for n, r in archive_rows if n < order and r.get('feasible')]
            record = candidate == baseline or (evaluation.get('feasible') is True
                and all(score < self._score(r) for r in earlier))
            rows = self._selected(queue_rows, candidate)
            originals = set(receipt_paths)
            originals.add('candidates/' + candidate + '/candidate.json')
            for row in rows:
                for name in ('control/plan.json', 'control/task.json', 'result.json',
                             'verified/result.json', 'candidate-verification.json', 'output/analysis/manifest.json'):
                    originals.add('runs/' + row['id'] + '/' + name)
                for optional in ('control/incremental.json', 'control/parent-plan.json',
                                 'control/parent-verified.json'):
                    name = 'runs/' + row['id'] + '/' + optional
                    if self._path(name).exists():
                        originals.add(name)
            receipts = {}
            for name in sorted(originals):
                path = self._path(name)
                receipts[name] = {'file': _row_file(path), 'value': read(path)}
            accounting = {}
            for name in ('control/campaign.json', 'control/budget.json'):
                path = self._path(name)
                if path.exists():
                    accounting[name] = {'sha256': sha(path), 'value': read(path)}
            logs = []
            if record:
                sources = {}
                for row in rows:
                    for origin in self._origins(row['id']):
                        sources[(origin['path'], origin['job'])] = origin
                for name in log_paths:
                    path = self._path(name)
                    if path.suffix not in {'.log', '.rpt', '.md', '.jsonl', '.txt', '.json'}:
                        raise ValueError('Explicit trace logs must be text artifacts')
                    sources[(name, None)] = {'path': name, 'job': None, 'stage': None, 'reused': False}
                if not sources:
                    raise ValueError('A full-log trace cannot be published without original logs')
                for origin in sorted(sources.values(), key=lambda r: r['path']):
                    path = self._path(origin['path'])
                    file_row = _row_file(path)
                    blob = CONTROL + '/log-blobs/' + file_row['sha256']
                    destination = self._path(blob)
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_name('.' + uuid.uuid4().hex)
                        try:
                            shutil.copyfile(path, temporary)
                            if sha(temporary) != file_row['sha256']:
                                raise ValueError('Original log changed during publication')
                            temporary.chmod(0o444)
                            os.link(temporary, destination)
                        finally:
                            temporary.unlink(missing_ok=True)
                    if sha(destination) != file_row['sha256']:
                        raise ValueError('Shared original log blob changed')
                    logs.append(dict(origin, file=file_row, blob=blob))
            body = {'version': 1, 'prototype': True, 'root': str(self.root), 'task_digest': self.task_id,
                'candidate': candidate, 'event_id': event_id, 'admission_order': order,
                'evaluation': evaluation, 'score': score, 'record_setter': record,
                'source_audit': audit, 'outcomes': outcomes, 'queue_rows': rows,
                'original_receipts': receipts, 'accounting_snapshots': accounting,
                'logs': logs, 'sealed_at': time.time()}
            # This atomic record is the trace entry. Interrupted blobs are harmless
            # and never authorize pruning; no separate index can claim publication.
            write(target, self._sign(body))
            return self._history(body)

    def _history(self, body):
        candidate = body['candidate']
        if body['root'] != str(self.root) or body['task_digest'] != self.task_id:
            raise ValueError('History belongs to another isolated copy/task')
        expected = next((r for n, r in self._archive() if n == body['admission_order']), None)
        if expected != body['evaluation']:
            raise ValueError('Original admitted archive row changed')
        if self._selected(self._queue(), candidate) != body['queue_rows']:
            raise ValueError('Original queue evidence changed')
        for name, evidence in body['original_receipts'].items():
            path = self._path(name)
            _check(path, evidence['file'])
            if read(path) != evidence['value']:
                raise ValueError('Original receipt changed')
        # Preserve readable/reconstructible source, independently of raw physics.
        from grt_bench.candidates import verify_snapshot
        verify_snapshot(self.root / 'candidates' / candidate)
        for item in body['logs']:
            blob = self._path(item['blob'])
            if blob.stat().st_size != item['file']['bytes'] or sha(blob) != item['file']['sha256']:
                raise ValueError('Published full-log trace changed')
        pruned, incomplete = [], []
        folder = self._path(CONTROL + '/pruning')
        if folder.exists():
            for p in sorted(folder.glob('*.receipt.json')):
                outcome = self._verify(read(p))
                if outcome['candidate'] == candidate:
                    pruned.extend(outcome['removed'])
            for p in sorted(folder.glob('*.plan.json')):
                planned = self._verify(read(p))
                stem = p.name[:-len('.plan.json')]
                if (planned['candidate'] == candidate
                        and not (folder / (stem + '.receipt.json')).exists()
                        and any((folder / (stem + suffix)).exists()
                                for suffix in ('.intent.json', '.progress.json'))):
                    incomplete.append(stem)
        result = copy.deepcopy(body)
        result.update(raw_evidence_state=('pruning incomplete; reconciliation required' if incomplete
                      else 'pruned by retention policy' if pruned else 'retained'),
            removed_paths=pruned, verification_scope='host-sealed historical evidence; not a fresh physical verification',
            incomplete_pruning_plans=incomplete, may_admit_new_candidate=False)
        return result

    def history(self, candidate):
        """Read sealed history only; never substitutes for completed_outcomes."""
        path = self._record_path(candidate)
        if not path.exists():
            raise ValueError('Candidate has no host-sealed history; unclassified outputs stay pinned')
        return self._history(self._verify(read(path)))

    def trace(self):
        records = []
        folder = self._path(CONTROL + '/history')
        if folder.exists():
            for path in sorted(folder.glob('*.json')):
                body = self._verify(read(path))
                if body['record_setter']:
                    records.append(self._history(body))
        return sorted(records, key=lambda r: r['admission_order'])

    def reading_index(self):
        """Small historical adapter, not the original strict candidate library.

        Code paths stay in the immutable source store; this creates no reading
        copies and offers full original log links only for record setters.
        """
        folder = self._path(CONTROL + '/history')
        entries = []
        for path in sorted(folder.glob('*.json')) if folder.exists() else []:
            item = self.history(path.stem)
            entries.append({k: item[k] for k in ('candidate', 'admission_order', 'score',
                'record_setter', 'evaluation', 'raw_evidence_state', 'verification_scope')})
            entries[-1]['code'] = 'candidates/' + item['candidate'] + '/tree'
            entries[-1]['logs'] = item['logs'] if item['record_setter'] else 'pruned by retention policy'
        return {'prototype': True, 'task_digest': self.task_id,
            'entries': sorted(entries, key=lambda r: r['admission_order']), 'can_admit': False}

    def _pins(self, pins):
        if (not isinstance(pins, dict) or pins.get('version') != 1 or pins.get('complete') is not True
                or pins.get('root') != str(self.root) or pins.get('task_digest') != self.task_id
                or not isinstance(pins.get('generation'), str) or not pins['generation']
                or not all(isinstance(pins.get(k), list) for k in ('paths', 'candidates', 'jobs'))):
            raise ValueError('Supply a complete host-owned pin generation for this isolated copy')
        protected = set()
        for item in pins['paths']:
            if not isinstance(item, dict) or not isinstance(item.get('reason'), str) or not item['reason'].strip():
                raise ValueError('Every pin needs a reason')
            path = self._path(item['path'])
            protected.add(path)
            # Any checkpoint artifact pins its whole stage, including stage logs.
            parts = Path(item['path']).parts
            if len(parts) > 5 and parts[0] == 'runs' and parts[2:4] == ('output', 'run'):
                protected.add(self._path('/'.join(parts[:5])))
        for candidate in pins['candidates']:
            _identifier(candidate)
        for job in pins['jobs']:
            protected.add(self._path('runs/' + _identifier(job)))
        from grt_bench.pareto import frontier
        rows = [r for _, r in self._archive()]
        parents = {r['candidate'] for r in frontier(rows, retain_equivalent=True)} | set(pins['candidates'])
        for row in self._queue():
            payload = json.loads(row['payload'])
            if payload.get('candidate') in parents:
                protected.add(self._path('runs/' + row['id'] + '/output'))
            if payload.get('candidate') == self.task['optimization']['baseline']:
                for stage in ('00_input', '10_floorplan'):
                    protected.add(self._path('runs/' + row['id'] + '/output/run/' + stage))
        return protected

    def _prunable(self, name, candidate, protected, history=None):
        path = self._path(name)
        parts = Path(name).parts
        if len(parts) < 4 or parts[0] != 'runs' or not HEX.fullmatch(parts[1]):
            raise ValueError('Only explicit raw run output/log paths may be pruned')
        if any(path == p or p in path.parents for p in protected):
            raise ValueError('Pinned physical checkpoint/stage log: ' + name)
        if parts[2] not in {'output', 'execution'}:
            raise ValueError('Original receipts, source and budgets are never prune targets')
        if parts[2] == 'execution' and path.suffix != '.log':
            raise ValueError('Only execution text logs are disposable')
        if parts[2] == 'output' and not (path.suffix in {'.odb', '.def', '.spef', '.guide', '.log', '.rpt'}
                or path.name.endswith('.spef.gz')):
            raise ValueError('Unclassified raw artifact stays pinned')
        history = self.history(candidate) if history is None else history
        if parts[1] not in {r['id'] for r in history['queue_rows']}:
            raise ValueError('Cannot prune another candidate/unclassified producer')
        if history['record_setter'] and path.suffix in {'.log', '.rpt'}:
            if name not in {r['path'] for r in history['logs']}:
                raise ValueError('Record-setter log has not been published')
        return path

    def publish_pins(self, pins):
        """Host terminal-event hook; a new generation invalidates older plans."""
        with self._locked():
            self._pins(pins)
            path = self._path(CONTROL + '/current-pins.json')
            write(path, self._sign(copy.deepcopy(pins)), replace=path.exists())
            return digest(pins)

    def plan_prune(self, candidate, paths, *, pins):
        """Seal explicit file deletions. No directory globs or implicit outputs."""
        _identifier(candidate)
        with self._locked():
            queue = self._queue()
            self._idle(queue)
            history = self.history(candidate)
            protected = self._pins(pins)
            if len(set(paths)) != len(paths) or not paths:
                raise ValueError('Use a nonempty unique explicit path list')
            files = {name: _row_file(self._prunable(name, candidate, protected, history)) for name in paths}
            current = self._path(CONTROL + '/current-pins.json')
            write(current, self._sign(copy.deepcopy(pins)), replace=current.exists())
            body = {'version': 1, 'prototype': True, 'root': str(self.root), 'task_digest': self.task_id,
                'candidate': candidate, 'pins': copy.deepcopy(pins), 'files': files,
                'queue_digest': digest(queue), 'archive_digest': digest(self._archive()),
                'history_sha256': sha(self._record_path(candidate)), 'created_at': time.time()}
            plan = self._sign(body)
            write(self._path(CONTROL + '/pruning/' + digest(body) + '.plan.json'), plan)
            return plan

    def apply(self, plan):
        """Idempotent/resumable application under an unchanged host pin snapshot."""
        with self._locked():
            body = self._verify(plan)
            if body['root'] != str(self.root) or body['task_digest'] != self.task_id:
                raise ValueError('Plan belongs to another isolated copy/task')
            key = digest(body)
            stored = self._path(CONTROL + '/pruning/' + key + '.plan.json')
            if not stored.exists() or read(stored) != plan:
                raise ValueError('Plan was not published by this host pilot')
            result = self._path(CONTROL + '/pruning/' + key + '.receipt.json')
            if result.exists():
                return self._verify(read(result))
            queue = self._queue()
            self._idle(queue)
            if digest(queue) != body['queue_digest'] or digest(self._archive()) != body['archive_digest']:
                raise ValueError('References changed after deletion planning')
            if sha(self._record_path(body['candidate'])) != body['history_sha256']:
                raise ValueError('Historical seal changed after planning')
            if self._verify(read(self._path(CONTROL + '/current-pins.json'))) != body['pins']:
                raise ValueError('Operational pin generation changed after planning')
            protected = self._pins(body['pins'])
            history = self.history(body['candidate'])
            # Check the entire batch before the first removal.
            progress = self._path(CONTROL + '/pruning/' + key + '.progress.json')
            removed = self._verify(read(progress))['removed'] if progress.exists() else []
            if any(name not in body['files'] for name in removed):
                raise ValueError('Unexpected deletion progress')
            for name, row in body['files'].items():
                path = self._prunable(name, body['candidate'], protected, history)
                if name in removed:
                    if path.exists():
                        raise ValueError('Deleted raw evidence unexpectedly reappeared')
                else:
                    _check(path, row)
            logical = sum(body['files'][n]['bytes'] for n in removed)
            released = 0
            for name, row in body['files'].items():
                if name in removed:
                    continue
                path = self._path(name)
                s = path.stat()
                # Intent is durable BEFORE unlink. Crash after this point is
                # deliberately uncertain, never assumed deleted without proof.
                intent = self._path(CONTROL + '/pruning/' + key + '.intent.json')
                write(intent, self._sign({'path': name, 'file': row}), replace=intent.exists())
                path.unlink()
                removed.append(name)
                logical += s.st_size
                released += s.st_blocks * 512 if s.st_nlink == 1 else 0
                write(progress, self._sign({'removed': removed}), replace=progress.exists())
                intent.unlink()
            outcome = {'version': 1, 'prototype': True, 'plan_digest': key, 'candidate': body['candidate'],
                'removed': removed, 'logical_bytes': logical, 'released_blocks_bytes_this_apply': released,
                'original_files': body['files'], 'completed_at': time.time(), 'budget_changed': False}
            write(result, self._sign(outcome))
            return outcome
