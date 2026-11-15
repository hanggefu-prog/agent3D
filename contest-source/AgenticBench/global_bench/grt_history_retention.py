"""Opt-in production historical retention, outside native admission and scoring.

Strict native source admission, completed_outcomes and aggregation run before a
candidate is sealed. Signed compact records are read-only historical evidence;
they never stand in for a new physical verification or admit a new candidate.
Only the baseline and strict score record setters retain full original logs.
No function in this module deletes files or changes budgets, queue or archive.
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


CONTROL = 'control/trace-retention'
POLICY = 'best-so-far-full-logs-v1'
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
            raise ValueError('Campaign root cannot traverse symlinks')
    path = path.resolve(strict=True)
    if path == Path('/'):
        raise ValueError('A campaign root is required')
    return path


def _canonical_relative(value):
    if (not isinstance(value, str) or not value or any(ord(c) < 32 for c in value)
            or '\\' in value or Path(value).as_posix() != value
            or value.startswith('/') or any(p in {'', '.', '..'} for p in value.split('/'))):
        raise ValueError('Use a canonical relative evidence path')
    return value


def _row_file(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('Only regular files can be sealed/pruned: ' + str(path))
    return {'bytes': s.st_size, 'sha256': sha(path), 'mode': stat.S_IMODE(s.st_mode)}


def _check(path, expected):
    if _row_file(path) != expected:
        raise ValueError('Sealed artifact changed: ' + str(path))


class RetainedHistory:
    """Host-authenticated history; never an admission or fresh EDA verifier.

    The entire ``control`` directory, including the private seal key, MUST stay
    outside agent mounts. Authorization is an explicit host-side policy, not a
    candidate file or an agent-supplied checksum. Native admission is unchanged.
    """

    @classmethod
    def initialize(cls, root, task, *, authorization):
        root = _root(root)
        task_id = validate(task)
        cls._authorization(authorization, root, task, task_id)
        # Reject queue/campaign activity before publishing the private trust root.
        temporary = object.__new__(cls)
        temporary.root, temporary.task, temporary.task_id = root, copy.deepcopy(task), task_id
        temporary._idle(temporary._queue())
        target = confined(root, CONTROL)
        target.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(target, 0o700)
        key = target / 'host.key'
        with key.open('xb') as stream:
            stream.write(os.urandom(32))
            stream.flush()
            os.fsync(stream.fileno())
        key.chmod(0o600)
        temporary.control, temporary.key = target, key.read_bytes()
        body = {'version': 1, 'policy': POLICY, 'root': str(root),
                'task_digest': task_id, 'evaluator_sha256': task['evaluator_sha256'],
                'authorization': copy.deepcopy(authorization), 'initialized_at': time.time()}
        write(target / 'authorization.json', temporary._sign(body))
        return cls(root, task)

    @staticmethod
    def _authorization(value, root, task, task_id):
        required = {'version': 1, 'enabled': True, 'host_only': True,
                    'root': str(root), 'task_digest': task_id,
                    'evaluator_sha256': task['evaluator_sha256'], 'policy': POLICY}
        if (not isinstance(value, dict)
                or any(type(value.get(k)) is not type(v) or value.get(k) != v
                       for k, v in required.items())):
            raise ValueError('Explicit exact root/task/evaluator host retention authorization is required')
        if task.get('objectives') != ['score']:
            raise ValueError('Best-so-far retention is opt-in only for the scalar score task')

    def __init__(self, root, task):
        self.root, self.task = _root(root), copy.deepcopy(task)
        self.task_id = validate(task)
        self.control = confined(self.root, CONTROL)
        info = self.control.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077
                or info.st_uid != os.geteuid()):
            raise ValueError('Host retention controls require owner-only host ownership')
        key = confined(self.root, CONTROL + '/host.key')
        info = key.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077
                or info.st_nlink != 1 or info.st_uid != os.geteuid()):
            raise ValueError('Retention requires an owner-only, unshared host seal key')
        self.key = key.read_bytes()
        if len(self.key) != 32:
            raise ValueError('Invalid host seal key')
        binding = self._verify(read(confined(self.root, CONTROL + '/authorization.json')))
        if (binding.get('version') != 1 or binding.get('policy') != POLICY
                or binding.get('root') != str(self.root)
                or binding.get('task_digest') != self.task_id
                or binding.get('evaluator_sha256') != task['evaluator_sha256']):
            raise ValueError('Host retention authorization binding differs')
        self._authorization(binding.get('authorization'), self.root, task, self.task_id)

    @contextmanager
    def locked(self):
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
        if (not isinstance(value['hmac_sha256'], str)
                or not hmac.compare_digest(self._sign(value['body'])['hmac_sha256'], value['hmac_sha256'])):
            raise ValueError('Host retention seal differs')
        return value['body']

    def _path(self, name):
        path = confined(self.root, _canonical_relative(name))
        if path.exists() and path.stat().st_dev != self.root.stat().st_dev:
            raise ValueError('Nested filesystem in retention evidence')
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
                               'generation_failed', 'no_change', 'stalled', 'blocked'}
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
            raise ValueError('This retention policy requires an explicit single training score')
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
        from .grt_candidate_library import _log_paths
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
        with self.locked():
            target = self._record_path(candidate)
            if target.exists():
                existing = self._verify(read(target))
                if existing.get('candidate') != candidate:
                    raise ValueError('History candidate path binding differs')
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
                self.history(baseline)
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
            if not rows or any(r['state'] != 'succeeded' for r in rows):
                raise ValueError('Only a complete independently verified matrix can be sealed')
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
                        key = (origin['path'], origin['job'])
                        if key not in sources:
                            sources[key] = dict(origin, requested_jobs=[])
                        sources[key]['requested_jobs'].append(row['id'])
                for name in log_paths:
                    path = self._path(name)
                    if path.suffix not in {'.log', '.rpt', '.md', '.jsonl', '.txt', '.json',
                            '.py', '.sh', '.tcl', '.cc', '.cpp', '.h', '.csv', '.yaml', '.yml', '.xml'}:
                        raise ValueError('Explicit trace logs must be text artifacts')
                    # Explicit extras come from the host's bounded allowlist;
                    # reject disguised binaries before a reading bundle can expose them.
                    path.read_bytes().decode('utf-8')
                    sources[(name, None)] = {'path': name, 'job': None, 'stage': None, 'reused': False, 'requested_jobs': []}
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
                    if (not stat.S_ISREG(destination.lstat().st_mode)
                            or stat.S_IMODE(destination.lstat().st_mode) != 0o444
                            or destination.stat().st_size != file_row['bytes']
                            or sha(destination) != file_row['sha256']):
                        raise ValueError('Shared original log blob changed')
                    logs.append(dict(origin, file=file_row, blob=blob))
            # Preserve original producer control/manifest bindings even when a
            # record setter reused an earlier candidate's stage logs. These
            # receipts are small and remain immutable after raw retirement.
            for origin in logs:
                if origin['job'] is None:
                    continue
                producer = 'runs/' + origin['job'] + '/'
                names = [producer + 'control/plan.json', producer + 'verified/result.json']
                for optional in ('control/incremental.json',):
                    if self._path(producer + optional).exists():
                        names.append(producer + optional)
                if origin['stage'] is not None:
                    names.append(producer + 'output/run/' + origin['stage'] + '/stage_manifest.json')
                for name in names:
                    if name not in receipts:
                        path = self._path(name)
                        receipts[name] = {'file': _row_file(path), 'value': read(path)}
            body = {'version': 1, 'policy': POLICY, 'root': str(self.root), 'task_digest': self.task_id,
                'evaluator_sha256': self.task['evaluator_sha256'],
                'archive_prefix': [(n, r) for n, r in archive_rows if n <= order],
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
        candidate = _identifier(body['candidate'])
        if (body.get('version') != 1 or body.get('policy') != POLICY
                or body['root'] != str(self.root) or body['task_digest'] != self.task_id
                or body.get('evaluator_sha256') != self.task['evaluator_sha256']):
            raise ValueError('History belongs to another root/task/evaluator/policy')
        current = self._archive()
        expected = next((r for n, r in current if n == body['admission_order']), None)
        prefix = [(n, r) for n, r in current if n <= body['admission_order']]
        if expected != body['evaluation'] or digest(prefix) != digest(body['archive_prefix']):
            raise ValueError('Original admitted archive evidence changed')
        if self._selected(self._queue(), candidate) != body['queue_rows']:
            raise ValueError('Original queue evidence changed')
        for name, evidence in body['original_receipts'].items():
            path = self._path(name)
            _check(path, evidence['file'])
            if read(path) != evidence['value']:
                raise ValueError('Original receipt changed')
        # This checks source identity, cumulative legality and task-bound ancestry.
        # It deliberately does NOT re-admit old raw measurement artifacts.
        audit = Archive(self.root, self.task).admit(candidate)
        if audit != body['source_audit']:
            raise ValueError('Historical candidate source or ancestry changed')
        for item in body['logs']:
            blob = self._path(item['blob'])
            info = blob.lstat()
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444
                    or info.st_size != item['file']['bytes']
                    or sha(blob) != item['file']['sha256']):
                raise ValueError('Published full-log trace changed')
        result = copy.deepcopy(body)
        result.update(raw_evidence_state='compact history; raw artifacts may be retired',
            compact_history_only=True,
            verification_scope='host-sealed historical evidence; not a fresh physical verification',
            may_admit_new_candidate=False)
        return result

    def history(self, candidate):
        """Read sealed history only; never substitutes for completed_outcomes."""
        self.__init__(self.root, self.task)
        path = self._record_path(candidate)
        if not path.exists():
            raise ValueError('Candidate has no host-sealed history; unclassified outputs stay pinned')
        body = self._verify(read(path))
        if body.get('candidate') != candidate:
            raise ValueError('History candidate path binding differs')
        return self._history(body)

    def trace(self):
        records = []
        folder = self._path(CONTROL + '/history')
        if folder.exists():
            for path in sorted(folder.glob('*.json')):
                body = self.history(path.stem)
                if body['record_setter']:
                    records.append(body)
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
        return {'policy': POLICY, 'task_digest': self.task_id,
            'entries': sorted(entries, key=lambda r: r['admission_order']), 'can_admit': False}


    def has(self, candidate):
        """Presence only; malformed existing records must never trigger fallback."""
        self.__init__(self.root, self.task)
        return self._record_path(candidate).exists()

    def gc_snapshot(self):
        """Authenticated historical bindings; not a physical deletion policy."""
        folder = self._path(CONTROL + '/history')
        histories = [self.history(p.stem) for p in sorted(folder.glob('*.json'))] if folder.exists() else []
        return {'version': 1, 'policy': POLICY, 'root': str(self.root),
                'task_digest': self.task_id, 'evaluator_sha256': self.task['evaluator_sha256'],
                'candidates': histories,
                'archive_candidates': [r['candidate'] for _, r in self._archive()],
                'full_log_candidates': sorted(r['candidate'] for r in histories if r['record_setter']),
                'protected_receipts': sorted({name for r in histories for name in r['original_receipts']}),
                'protected_blobs': sorted({log['blob'] for r in histories for log in r['logs']})}

    def sign_host_record(self, body, kind):
        if kind not in {'gc-plan', 'gc-receipt', 'gc-intent', 'gc-progress', 'lifecycle-receipt', 'terminal-diagnostics'}:
            raise ValueError('Unsupported host retention record domain')
        return self._sign({'domain': kind, 'root': str(self.root), 'task_digest': self.task_id,
                           'evaluator_sha256': self.task['evaluator_sha256'], 'value': copy.deepcopy(body)})

    def verify_host_record(self, value, kind):
        body = self._verify(value)
        if (kind not in {'gc-plan', 'gc-receipt', 'gc-intent', 'gc-progress', 'lifecycle-receipt', 'terminal-diagnostics'}
                or body.get('domain') != kind or body.get('root') != str(self.root)
                or body.get('task_digest') != self.task_id
                or body.get('evaluator_sha256') != self.task['evaluator_sha256']):
            raise ValueError('Host retention record domain/binding differs')
        return copy.deepcopy(body['value'])
