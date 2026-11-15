"""Lossless storage for completed post-GRT results; evaluator bytes stay frozen.

The host restores byte-identical DEFs before an unmodified reader runs. Neither
the native manifests nor scoring/verifier code understand or trust this index.
Only the pinned image may export a DEF, and its original hash must match.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid

from grt_bench.common import confined, digest, read, sha, write
from grt_bench.runtime import docker_command, run_container

POLICY = {'version': 1, 'mode': 'odb-first', 'deduplicate_min_bytes': 65536}


def regular_files(root):
    """Never traverse directory symlinks or another filesystem."""
    device = root.stat().st_dev
    for base, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            p = Path(base) / name
            if p.is_symlink() or p.stat().st_dev != device:
                raise ValueError('Linked directory or nested filesystem: ' + str(p))
        for name in names:
            p = Path(base) / name
            if p.is_symlink():
                p.resolve(strict=True).relative_to(root)
                continue
            if not stat.S_ISREG(p.stat().st_mode) or p.stat().st_dev != device:
                raise ValueError('Non-regular result: ' + str(p))
            yield p


def snapshot(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p),
                                      'mode': stat.S_IMODE(p.stat().st_mode)}
            for p in regular_files(root)}


def check_file(path, row):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('Stored artifact changed: ' + str(path))


def assert_idle(outputs):
    ids = subprocess.check_output(['docker', 'ps', '-q'], text=True, timeout=30).split()
    containers = json.loads(subprocess.check_output(['docker', 'inspect', *ids], text=True, timeout=30)) if ids else []
    for c in containers:
        for m in c['Mounts']:
            path = Path(m['Source']).resolve()
            if any(path == out or path in out.parents or out in path.parents for out in outputs):
                raise ValueError('A live container still mounts a result: ' + c['Name'])


# This program is run by the installed OpenROAD, never by a host-built binary.
EXPORT_SCRIPT = '''import json
import odb
from pathlib import Path
for item in json.loads(Path('/storage/requests.json').read_text()):
    db = odb.dbDatabase.create()
    odb.read_db(db, '/stored/' + item['odb'])
    odb.write_def(db.getChip().getBlock(), '/storage/' + item['target'])
    odb.dbDatabase.destroy(db)
'''


def export_defs(output, image, requests, scratch):
    """One bounded CPU process per batch. Returned files live only in scratch."""
    if not requests:
        return
    write(scratch / 'requests.json', requests)
    (scratch / 'export.py').write_text(EXPORT_SCRIPT)
    name = 'grt-storage-' + uuid.uuid4().hex[:16]
    command = docker_command(image, name, {'cpu': 2, 'memory_gb': 4, 'gpu_count': 0}, [],
                             [(output, '/stored', True), (scratch, '/storage', False)],
                             ['-c', "import os; os.execv('/usr/local/bin/openroad', ['openroad', '-no_init', '-exit', '-python', '/storage/export.py'])"])
    # Storage has no source tree. The standard worker's default /trusted does
    # not exist here; the only writable work area is the private scratch mount.
    command[command.index('--workdir') + 1] = '/storage'
    run_container(command, scratch / 'execution', name, timeout=600)


class Store:
    def __init__(self, root, *, exporter=export_defs, idle=assert_idle):
        self.root = Path(root).resolve(strict=True)
        self.control = self.root / 'control/result-storage'
        self.control.mkdir(parents=True, exist_ok=True)
        self.exporter, self.idle = exporter, idle

    @contextmanager
    def locked(self):
        with (self.control / 'storage.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def paths(self, job):
        if not re.fullmatch('[0-9a-f]{64}', job):
            raise ValueError('Invalid result job')
        attempt = confined(self.root, 'runs/' + job)
        output = confined(attempt, 'output').resolve(strict=True)
        return attempt, output, self.control / 'jobs' / (job + '.json')

    def qualified(self, job):
        attempt, output, receipt = self.paths(job)
        plan = read(attempt / 'control/plan.json')
        result = read(attempt / 'verified/result.json')
        if result['status'] != 'verified' or result['evidence']['plan_sha256'] != sha(attempt / 'control/plan.json'):
            raise ValueError('Only complete verified results can be compacted')
        if any(result[k] != plan[k] for k in ('candidate', 'case', 'seed', 'task_digest', 'split')):
            raise ValueError('Result identity changed')
        runtime = read(self.root / 'control/runtimes' / (plan['candidate'] + '.json'))
        image = runtime['image']
        if runtime['candidate'] != plan['candidate'] or not re.fullmatch('sha256:[0-9a-f]{64}', image):
            raise ValueError('Result runtime changed')
        commands = list((attempt / 'execution').rglob('command.json'))
        if not commands or any('OPEN3DBENCH_IMAGE_ID=' + image not in read(p) for p in commands):
            raise ValueError('Result was executed with another image')
        queue = self.root / 'control/queue.sqlite'
        if queue.exists():
            import sqlite3
            from contextlib import closing
            with closing(sqlite3.connect(queue.as_uri() + '?mode=ro', uri=True)) as db:
                state = db.execute('SELECT state FROM jobs WHERE id=?', (job,)).fetchone()
            if state is None or state[0] != 'succeeded':
                raise ValueError('Result queue entry is not successfully terminated')
        return attempt, output, receipt, image

    def receipt(self, job):
        attempt, output, path, image = self.qualified(job)
        if not path.exists():
            return None
        value = read(path)
        if (value['version'] != 1 or value['job'] != job or value['image'] != image
                or value['plan_sha256'] != sha(attempt / 'control/plan.json')
                or value['verified_sha256'] != sha(attempt / 'verified/result.json')):
            raise ValueError('Storage receipt is not bound to this result')
        for name, recipe in value['defs'].items():
            confined(output, name)
            odb = confined(output, recipe['odb'])
            if not name.endswith('.def') or odb.suffix != '.odb' or recipe['odb'] not in value['files']:
                raise ValueError('Invalid DEF recipe')
            if recipe['odb_sha256'] != value['files'][recipe['odb']]['sha256']:
                raise ValueError('DEF recipe ODB binding changed')
            if recipe['def_sha256'] != value['files'][name]['sha256']:
                raise ValueError('DEF recipe output binding changed')
        return value

    def seal(self, job):
        attempt, output, path, image = self.qualified(job)
        existing = self.receipt(job)
        if existing is not None:
            return existing
        files = snapshot(output)
        proofs_path = self.control / 'export-proofs.json'
        proofs = read(proofs_path) if proofs_path.exists() else {}
        candidates, requests, recipes, unmatched = {}, [], {}, []
        for name, row in files.items():
            p = Path(name)
            if p.suffix != '.def':
                continue
            for odb in (p.with_suffix('.odb'), p.with_suffix('')):
                odb = str(odb)
                if odb.endswith('.odb') and odb in files:
                    key = digest({'version': 1, 'image': image, 'odb': files[odb]['sha256'], 'def': row['sha256']})
                    recipe = {'odb': odb, 'odb_sha256': files[odb]['sha256'], 'def_sha256': row['sha256']}
                    candidates[name] = (key, recipe)
                    if key not in proofs and key not in {r['key'] for r in requests}:
                        requests.append({'odb': odb, 'target': key + '.def', 'key': key})
                    break
        with tempfile.TemporaryDirectory(prefix='export-', dir=self.control) as tmp:
            scratch = Path(tmp)
            self.exporter(output, image, requests, scratch)
            by_key = {r['key']: r for r in requests}
            for name, (key, recipe) in candidates.items():
                if key in by_key:
                    exported = scratch / by_key[key]['target']
                    proofs[key] = {'identical': exported.is_file() and sha(exported) == files[name]['sha256']}
                if proofs[key]['identical']:
                    recipes[name] = recipe
                else:
                    unmatched.append(name)
        # The durable recipe precedes any removal. A crash leaves either the
        # original DEF or an independently reproducible, hash-bound DEF.
        write(proofs_path, proofs, replace=proofs_path.exists())
        value = {'version': 1, 'job': job, 'image': image,
                 'plan_sha256': sha(attempt / 'control/plan.json'),
                 'verified_sha256': sha(attempt / 'verified/result.json'),
                 'files': files, 'defs': recipes, 'nonidentical_defs_retained': unmatched}
        write(path, value)
        return value

    def prune_defs(self, job):
        value = self.receipt(job)
        if value is None:
            return {'removed_paths': 0, 'logical_bytes': 0, 'released_blocks_bytes': 0}
        _, output, _ = self.paths(job)
        result = {'removed_paths': 0, 'logical_bytes': 0, 'released_blocks_bytes': 0}
        checked = set()
        for name, recipe in value['defs'].items():
            odb = confined(output, recipe['odb'])
            if recipe['odb'] not in checked:
                check_file(odb, value['files'][recipe['odb']]); checked.add(recipe['odb'])
            path = confined(output, name)
            if path.exists():
                check_file(path, value['files'][name]); s = path.stat()
                path.unlink()
                result['removed_paths'] += 1
                result['logical_bytes'] += s.st_size
                if s.st_nlink == 1:
                    result['released_blocks_bytes'] += s.st_blocks * 512
        return result

    def restore(self, job, cache=None):
        value = self.receipt(job)
        if value is None:
            return
        _, output, _ = self.paths(job)
        cache = {} if cache is None else cache
        requests, pending, request_keys = [], [], {}
        for name, row in value['files'].items():
            path = confined(output, name)
            if path.exists():
                check_file(path, row)
            elif name not in value['defs']:
                raise ValueError('Missing retained artifact: ' + name)
            else:
                recipe = value['defs'][name]
                check_file(confined(output, recipe['odb']), value['files'][recipe['odb']])
                key = (value['image'], recipe['odb_sha256'], row['sha256'], row['mode'])
                if key not in cache and key not in request_keys:
                    request_keys[key] = str(len(requests)) + '.def'
                    requests.append({'odb': recipe['odb'], 'target': request_keys[key], 'name': name})
                pending.append((name, key))
        with tempfile.TemporaryDirectory(prefix='restore-', dir=self.control) as tmp:
            scratch = Path(tmp)
            required = sum(value['files'][r['name']]['bytes'] for r in requests)
            if shutil.disk_usage(self.root).free < required + 10 * 1024**3:
                raise ValueError('Insufficient space to restore DEFs with the reserve intact')
            self.exporter(output, value['image'], requests, scratch)
            # Check the entire batch before publishing any restored paths.
            for name, key in pending:
                source = cache[key] if key in cache else scratch / request_keys[key]
                check_file(source, value['files'][name])
            for name, key in pending:
                target = confined(output, name)
                source = cache[key] if key in cache else scratch / request_keys[key]
                os.chmod(source, value['files'][name]['mode'])
                os.link(source, target)  # no overwrite of concurrent changes
                cache[key] = target

    def deduplicate(self, jobs):
        index_path = self.control / 'dedup-index.json'
        index = read(index_path) if index_path.exists() else {}
        checked = set(); result = {'linked_paths': 0, 'released_blocks_bytes': 0, 'by_extension': {}}
        for job in jobs:
            value = self.receipt(job)
            _, output, _ = self.paths(job)
            for name, row in value['files'].items():
                path = confined(output, name)
                if name in value['defs'] or row['bytes'] < POLICY['deduplicate_min_bytes']:
                    continue
                check_file(path, row)
                key = digest(row)
                other = index.get(key)
                if other:
                    source = confined(self.root, other)
                    if not source.is_file():
                        other = None
                if not other:
                    index[key] = str(path.relative_to(self.root)); continue
                if other not in checked:
                    check_file(source, row); checked.add(other)
                s, t = path.stat(), source.stat()
                if stat.S_IMODE(s.st_mode) != row['mode'] or stat.S_IMODE(t.st_mode) != row['mode']:
                    raise ValueError('Artifact mode changed before deduplication')
                if s.st_dev != t.st_dev:
                    raise ValueError('Result deduplication crossed a filesystem')
                if s.st_ino == t.st_ino:
                    continue
                temporary = path.with_name('.' + path.name + '.link-' + uuid.uuid4().hex)
                try:
                    os.link(source, temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                freed = s.st_blocks * 512 if s.st_nlink == 1 else 0
                result['linked_paths'] += 1; result['released_blocks_bytes'] += freed
                ext = path.suffix or '(none)'
                result['by_extension'][ext] = result['by_extension'].get(ext, 0) + freed
        write(index_path, index, replace=index_path.exists())
        return result

    def compact(self, jobs):
        jobs = list(dict.fromkeys(jobs))
        with self.locked():
            outputs = [self.paths(job)[1] for job in jobs]
            self.idle(outputs)
            free = shutil.disk_usage(self.root).free
            start = time.monotonic(); removals = {}
            for job in jobs:
                self.seal(job)
                removals[job] = self.prune_defs(job)
            dedup = self.deduplicate(jobs)
            result = {'jobs': jobs, 'defs': removals, 'dedup': dedup,
                      'released_blocks_bytes': sum(v['released_blocks_bytes'] for v in removals.values()) + dedup['released_blocks_bytes'],
                      'filesystem_free_delta_bytes': shutil.disk_usage(self.root).free - free,
                      'seconds': time.monotonic() - start}
            write(self.control / 'operations' / (uuid.uuid4().hex + '.json'), result)
            return result

    @contextmanager
    def available(self, jobs):
        """Restore exact bytes while original readers run; discard only exports.

        The storage lock spans the reader lifetime. The EDA containers still
        mount reference inputs read-only. An uncertain live reader prevents
        cleanup, leaving complete files available for reconciliation.
        """
        jobs = sorted(set(jobs))
        with self.locked():
            self.idle([self.paths(j)[1] for j in jobs])
            cache = {}
            try:
                for job in jobs:
                    self.restore(job, cache)
                yield
            finally:
                self.idle([self.paths(j)[1] for j in jobs])
                for job in jobs:
                    self.prune_defs(job)


def policy_enabled(root):
    path = Path(root) / 'control/result-storage-policy.json'
    if not path.exists():
        return False
    if read(path) != POLICY:
        raise ValueError('Unsupported result storage policy')
    return True


def verify_available(root, job, evidence_dir):
    """Run the original independent verifier against a restored result view."""
    store = Store(root)
    attempt, output, _, image = store.qualified(job)
    plan = read(attempt / 'control/plan.json')
    source = confined(store.root, 'candidates/' + plan['candidate'] + '/tree')
    evidence_dir = Path(evidence_dir).resolve()
    if evidence_dir == attempt or attempt in evidence_dir.parents:
        raise ValueError('Fresh verification evidence must be outside the saved result')
    evidence_dir.mkdir(parents=True, exist_ok=False)
    verified = evidence_dir / 'verified'; verified.mkdir()
    args = read(attempt / 'verification/command.json')
    if (args[-4:] != ['-m', 'grt_bench.verify', '/control/plan.json', '/verified/result.json']
            or '--gpus' in args or image not in args or 'OPEN3DBENCH_IMAGE_ID=' + image not in args):
        raise ValueError('Unexpected independent verifier command')
    name = 'grt-storage-verify-' + uuid.uuid4().hex[:16]
    args[args.index('--name') + 1] = name
    for i, arg in enumerate(args):
        if arg.startswith('type=bind,source='):
            fields = arg.split(',')
            original = next(field for field in fields if field.startswith('source='))
            path = Path(original[len('source='):]).resolve()
            workspace = attempt / 'workspace'
            if path == workspace or workspace in path.parents:
                fields[fields.index(original)] = 'source=' + str(source / path.relative_to(workspace))
                arg = ','.join(fields)
            if ',target=/trusted' in arg:
                arg = 'type=bind,source=' + str(Path(__file__).resolve().parents[1]) + ',target=/trusted,readonly'
            if ',target=/verified' in arg:
                arg = 'type=bind,source=' + str(verified) + ',target=/verified'
            args[i] = arg
    terminal = run_container(args, evidence_dir / 'execution', name, timeout=1200)
    result = read(verified / 'result.json')
    if result != read(attempt / 'verified/result.json'):
        raise ValueError('Fresh verifier receipt differs from the original')
    return {'job': job, 'metrics_equal': True, 'original_verifier': True,
            'seconds': terminal['wall_seconds'], 'original_verified_sha256': sha(attempt / 'verified/result.json')}


def parent_jobs(config, task, candidate):
    from grt_bench.contract import validate
    from grt_bench.incremental import physical_source
    from grt_bench.queue import Queue
    root = Path(config['root'])
    if candidate == task['optimization']['baseline']:
        return []
    parent = read(root / 'candidates' / candidate / 'candidate.json')['parent']
    jobs = []
    for row in Queue(root / 'control/queue.sqlite').rows():
        payload = json.loads(row['payload'])
        if (payload.get('candidate') == parent and payload.get('task_digest') == validate(task)
                and row['state'] == 'succeeded'):
            physical, _ = physical_source(root, row['id']); jobs.append(physical)
    return jobs


def evaluate_matrix(evaluate, config, task, candidate, folder):
    if not policy_enabled(config['root']):
        return evaluate(config, task, candidate, folder)
    store = Store(config['root'])
    failure = None
    from .grt_campaign import CandidateEvaluationFailed
    try:
        with store.available(parent_jobs(config, task, candidate)):
            result = evaluate(config, task, candidate, folder)
    except CandidateEvaluationFailed as error:
        failure = error
    # Only complete verified cells are compacted. Failed native outputs remain
    # intact for diagnosis, and uncertain/integrity failures never enter here.
    jobs = read(Path(folder) / 'admission.json')['jobs']
    if failure is not None:
        from grt_bench.queue import Queue
        succeeded = {r['id'] for r in Queue(Path(config['root']) / 'control/queue.sqlite').rows()
                     if r['state'] == 'succeeded'}
        jobs = [j for j in jobs if j in succeeded]
        if not jobs:
            raise failure
    try:
        receipt = store.compact(jobs)
        write(Path(folder) / 'storage.json', receipt, replace=(Path(folder) / 'storage.json').exists())
    except Exception as error:
        # Storage maintenance must not relabel an admitted physical evaluation
        # as failed or spend another evaluation budget. Keep its durable recipe.
        write(Path(folder) / ('storage-error-' + uuid.uuid4().hex + '.json'),
              {'error': type(error).__name__ + ': ' + str(error), 'evaluation_remains_verified': failure is None})
        print('Result storage needs attention: ' + str(error), file=sys.stderr)
    if failure is not None:
        raise failure
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('compact', 'restore', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--jobs', nargs='+', required=True)
    args = parser.parse_args(); store = Store(args.root)
    if args.action == 'compact':
        print(json.dumps(store.compact(args.jobs), indent=2))
    elif args.action == 'restore':
        with store.locked():
            store.idle([store.paths(j)[1] for j in args.jobs])
            for job in args.jobs:
                store.restore(job)
        print(json.dumps({'restored': args.jobs}))
    else:
        evidence = store.control / 'verifications' / uuid.uuid4().hex
        with store.available(args.jobs):
            results = [verify_available(store.root, job, evidence / job) for job in args.jobs]
        write(evidence / 'results.json', results)
        print(json.dumps({'evidence': str(evidence), 'verified': len(results)}))


if __name__ == '__main__':
    main()
