"""Reconstitute an identical private T4 draft; never change the shared release."""
import argparse
from pathlib import Path
import tarfile

from engineering_bench.prepare_development import verify_release
from engineering_bench.submission import clone_source, inventory
from engineering_bench.tasks import read_json, load_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', default='cross_die_cts')
    args = p.parse_args()
    task = load_task(args.task)
    base, _ = verify_release(args.release)
    args.output.mkdir(parents=True, exist_ok=False)
    clone_source(base, args.output / 'source')
    with tarfile.open(args.archive) as archive:
        for member in archive.getmembers():
            name = member.name
            if Path(name).is_absolute() or '..' in Path(name).parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe transfer member')
            allowed = name in ('baseline-inventory.json', 'draft-inventory.json', 'result.json')
            if name.startswith('source/'):
                local = name[len('source/'):].rstrip('/')
                allowed = task.allows(local) if member.isfile() else any(p.startswith(local + '/') for p in task.data['editable_paths']) or task.allows(local + '/file')
            if not allowed:
                raise ValueError('Unexpected transfer member')
        archive.extractall(args.output)
    if inventory(args.output / 'source') != read_json(args.output / 'draft-inventory.json'):
        raise ValueError('Transferred draft differs from the model-produced source')
    print('Verified identical full draft inventory')


if __name__ == '__main__':
    main()
