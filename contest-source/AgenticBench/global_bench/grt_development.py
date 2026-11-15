"""Writable development copies; explicit, clean exports for the strict evaluator."""
import os
from pathlib import Path
import shutil
import stat
import tempfile

from grt_bench.candidates import verify_snapshot
from grt_bench.algorithm_scope import DREAMPLACE
from grt_bench.common import confined, sha, write


def export_submission(archive, workspace, receipt, changed_files, destination):
    """Export only declared files over the immutable parent, then verify normally.

    Development-only edits, tests and build outputs are deliberately not source
    submissions. The export receipt makes this boundary explicit. Immutable
    parent files can share read-only inodes in this host-only temporary tree;
    submitted bytes are always copied and no model process can reach the tree.
    """
    workspace, destination = Path(workspace), Path(destination)
    parent = archive.root / 'candidates' / receipt['parent']
    archive.admit(receipt['parent'])
    original = verify_snapshot(parent)
    allowed = set(archive.task['optimization']['allowed_files'])
    names = set(changed_files)
    if not names or len(names) != len(changed_files) or not names <= allowed:
        raise ValueError('Submission requires distinct task-allowed changed_files')
    submitted = {}
    for name in sorted(names):
        try:
            path = confined(workspace, name)
            mode = path.stat().st_mode
        except OSError as error:
            raise ValueError('Submitted source is missing: ' + name) from error
        # Reject symlinked ancestors too, even when they point within the copy.
        relative = Path(name)
        if any((workspace / p).is_symlink() for p in (relative, *relative.parents)):
            raise ValueError('Submitted source may not use symlinks: ' + name)
        if not stat.S_ISREG(mode):
            raise ValueError('Submitted source must be a regular file: ' + name)
        entry = {'sha256': sha(path), 'bytes': path.stat().st_size,
                 'executable': bool(mode & 0o111)}
        if entry == original['files'][name]:
            raise ValueError('Declared source has no change: ' + name)
        submitted[name] = entry
    # Export is intentionally explicit. Other development edits may be experiments
    # the agent discarded; do not infer a submission from the whole writable tree.
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / 'selection.json', {'parent': receipt['parent'],
        'selection_id': receipt['selection_id'], 'files': submitted,
        'policy': 'Only changed_files are submitted; all other files come from the immutable parent.'})
    with tempfile.TemporaryDirectory(prefix='source-', dir=destination) as temporary:
        tree = Path(temporary)
        for name, entry in original['files'].items():
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name in names:
                shutil.copyfile(confined(workspace, name), target)
                target.chmod(0o555 if submitted[name]['executable'] else 0o444)
                if sha(target) != submitted[name]['sha256']:
                    raise ValueError('Development source changed during export')
            elif 'symlink' in entry:
                target.symlink_to(entry['symlink'])
            else:
                source = parent / 'tree' / name
                if source.stat().st_mode & 0o222:
                    raise ValueError('Immutable parent is writable')
                os.link(source, target)
        candidate = archive.freeze(tree, receipt['selection_id'])
    write(destination / 'result.json', {'candidate': candidate,
        'verification': archive.admit(candidate), 'submitted_files': sorted(names)})
    return candidate


# Runs inside the parent image, before the development container starts. Populate
# missing installed modules/native operators in the writable copy, then mount the
# entire directory. Atomic renames and newly created modules are visible to Python;
# per-file bind mounts would continue to point at stale inodes after a rename.
POPULATE_DREAMPLACE = 'DREAMPLACE = ' + repr(DREAMPLACE) + '\n' + '''
import hashlib,json,pathlib,shutil
source=pathlib.Path('/opt/dreamplace/dreamplace')
target=pathlib.Path('/workspace/subject')/DREAMPLACE
added={}
if not source.is_dir() or not target.is_dir():
    raise RuntimeError('Parent DREAMPlace installation/source is missing')
for path in sorted(source.rglob('*')):
    relative=path.relative_to(source)
    if '__pycache__' in relative.parts: continue
    out=target/relative
    if out.exists() or out.is_symlink(): continue
    if path.is_dir(): out.mkdir(parents=True,exist_ok=True)
    elif path.is_file():
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,out)
        out.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
        added[str(relative)]={'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size}
    else: raise RuntimeError('Unsupported installed DREAMPlace entry')
print(json.dumps({'installed_files_added':added,'source_files_preserved':True}))
'''


def retain_development(campaign, row, destination):
    """Retain unique experiments, then remove a stopped disposable development copy.

    Immutable source and installed runtime duplicates need not be retained again.
    Large or special unique outputs are kept in place for explicit housekeeping.
    """
    import json
    import subprocess
    import tarfile
    from grt_bench.common import read
    workspace = campaign.root / 'workspaces' / ('%04d' % row['round'])
    control = Path(row['control'])
    if not (control/'submission/result.json').exists() or not workspace.exists():
        return {'status': 'not_applicable'}
    if workspace.is_symlink() or workspace.resolve() != workspace:
        raise ValueError('Development workspace is redirected')
    submitted = read(control/'submission/result.json')
    if submitted['candidate'] != row['candidate'] or read(control/'container-state.json')['running']:
        raise ValueError('Development workspace is not a stopped submitted round')
    campaign.archive.admit(row['candidate'])
    candidate = verify_snapshot(campaign.root/'candidates'/row['candidate'])
    parent = verify_snapshot(campaign.root/'candidates'/candidate['parent'])
    ids = subprocess.check_output(['docker','ps','-q'], text=True).split()
    if ids:
        containers = json.loads(subprocess.check_output(['docker','inspect',*ids], text=True))
        for container in containers:
            for mount in container['Mounts']:
                path = Path(mount['Source'])
                if path == workspace or path in workspace.parents or workspace in path.parents:
                    raise ValueError('A live container uses the development copy')
    installed = json.loads(read(control/'container-editing/prepare-result.json')['stdout'])['installed_files_added']
    installed = {DREAMPLACE + name: value for name, value in installed.items()}
    keep = []; total = 0; observations = {}
    for path in sorted(workspace.rglob('*')):
        name = path.relative_to(workspace).as_posix(); info = path.lstat()
        observations[name] = (info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)
        if path.is_symlink():
            keep.append(name); continue
        if path.is_dir(): continue
        if not stat.S_ISREG(info.st_mode):
            return {'status': 'kept_unique_special_files'}
        checksum = sha(path)
        expected = [candidate['files'].get(name, {}), parent['files'].get(name, {}), installed.get(name, {})]
        if any(checksum == value.get('sha256') for value in expected): continue
        total += info.st_size
        if total > 128*1024**2:
            return {'status': 'kept_large_unique_outputs'}
        keep.append(name)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(destination/'experiments.tar.gz', 'w:gz', dereference=False) as bundle:
        for name in keep: bundle.add(workspace/name, arcname=name, recursive=False)
    after = {p.relative_to(workspace).as_posix(): (s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns)
             for p in workspace.rglob('*') for s in [p.lstat()]}
    if after != observations:
        raise ValueError('Development workspace changed while retaining experiments')
    result = {'status': 'removed', 'workspace': str(workspace), 'candidate': row['candidate'],
        'retained_paths': keep, 'retained_bytes': total,
        'experiments_sha256': sha(destination/'experiments.tar.gz')}
    write(destination/'receipt.json', result)
    shutil.rmtree(workspace)
    return result
