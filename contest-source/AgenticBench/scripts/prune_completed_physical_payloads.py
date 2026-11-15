#!/usr/bin/env python3
"""Audited deletion of explicitly mounted, completed non-best run payloads.

Operator must verify the explicit targets are inactive and not references/best
runs. Only large regenerable geometry/parasitic/netlist files are eligible.
Never removes directories, source, candidates, logs, reports or constraints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

SUFFIXES = {'.odb', '.def', '.spef', '.v'}
STAGES = {'30_placement', '40_pdn', '50_hbt', '60_opt', '70_cts', '80_route'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('Not a regular file: ' + str(path))
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
            'inode': s.st_ino, 'device': s.st_dev, 'links': s.st_nlink,
            'allocated_bytes': s.st_blocks * 512}


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['plan', 'apply'])
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--target', type=Path, action='append', default=[])
    ap.add_argument('--expect-sha256')
    args = ap.parse_args()
    if args.mode == 'plan':
        if not args.target:
            raise ValueError('Explicit targets required')
        plan = {'targets': [], 'files': [], 'protected': {}}
        for target in args.target:
            if target.parent != Path('/targets') or target.resolve() != target:
                raise ValueError('Expected explicit /targets/NAME mount')
            result = target / 'result'
            record = result / 'evaluation_result.json'
            candidate = target / 'candidate/candidate.json'
            data = json.loads(record.read_text())
            if 'metrics' not in data or not candidate.is_file():
                raise ValueError('Missing completed evaluation/candidate')
            plan['targets'].append(str(target))
            for path in (record, candidate):
                plan['protected'][str(path)] = sha(path)
            for directory, subdirs, names in os.walk(result / 'run', followlinks=False):
                subdirs[:] = [d for d in subdirs if not (Path(directory)/d).is_symlink()]
                for name in names:
                    path = Path(directory) / name
                    relative = path.relative_to(result / 'run')
                    if path.is_symlink():
                        continue
                    if (relative.parts[0] in STAGES and path.suffix in SUFFIXES
                            and path.stat().st_size >= 1024 * 1024):
                        if path.resolve() != path:
                            raise ValueError('Symlink parent: ' + str(path))
                        plan['files'].append({'path': str(path), **identity(path)})
                    else:
                        plan['protected'][str(path)] = sha(path)
            # Preserve top-level diagnostic/config/patch files as well.
            for path in result.iterdir():
                if path.is_file() and not path.is_symlink():
                    plan['protected'][str(path)] = sha(path)
        inodes = {}
        for item in plan['files']:
            key = (item['device'], item['inode'])
            count, first = inodes.get(key, (0, item))
            inodes[key] = (count + 1, first)
        plan['logical_bytes'] = sum(x['size'] for x in plan['files'])
        plan['estimated_reclaim_bytes'] = sum(x['allocated_bytes'] for count, x
            in inodes.values() if count == x['links'])
        save_new(args.audit, plan)
        print(json.dumps({'files': len(plan['files']),
            'logical_gib': plan['logical_bytes'] / 2**30,
            'estimated_reclaim_gib': plan['estimated_reclaim_bytes'] / 2**30,
            'protected_files': len(plan['protected']), 'audit_sha256': sha(args.audit)}))
        return
    if not args.expect_sha256 or sha(args.audit) != args.expect_sha256:
        raise ValueError('Reviewed plan hash required')
    plan = json.loads(args.audit.read_text())
    for name, checksum in plan['protected'].items():
        if sha(Path(name)) != checksum:
            raise ValueError('Protected file changed: ' + name)
    # Validate the entire list before deleting even one file.
    for item in plan['files']:
        path = Path(item['path'])
        if path.resolve() != path or identity(path) != {k:v for k,v in item.items() if k != 'path'}:
            raise ValueError('Target changed: ' + str(path))
        if not any(Path(t)/'result/run' in path.parents for t in plan['targets']):
            raise ValueError('Out-of-scope target')
        rel = path.relative_to(next(Path(t)/'result/run' for t in plan['targets']
                                   if Path(t)/'result/run' in path.parents))
        if rel.parts[0] not in STAGES or path.suffix not in SUFFIXES:
            raise ValueError('Ineligible file')
    completed = args.audit.with_suffix('.deleted.jsonl')
    with completed.open('x') as journal:
        for item in plan['files']:
            Path(item['path']).unlink()
            journal.write(json.dumps(item) + '\n')
            journal.flush()
        os.fsync(journal.fileno())
    for name, checksum in plan['protected'].items():
        if sha(Path(name)) != checksum:
            raise ValueError('Protected file verification failed: ' + name)
    print(json.dumps({'deleted_files': len(plan['files']),
                      'protected_files_verified': len(plan['protected'])}))


if __name__ == '__main__':
    main()
