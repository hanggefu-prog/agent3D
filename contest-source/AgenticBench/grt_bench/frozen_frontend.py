"""Task-pinned physical inputs for a benchmark that starts at partition.

These are benchmark fixtures, not a claim that candidate code reproduces the
producer's floorplan. Original manifests/provenance survive byte-for-byte;
candidate stages retain their normal source/runtime signatures.
"""
import argparse
from pathlib import Path
import shutil
import tempfile
import os

from .common import digest, read, sha, write

STAGES = {'input': '00_input', 'floorplan': '10_floorplan'}


def validate_contract(task):
    import re
    contract = task['frontend_inputs']
    if set(contract) != {'boundary', 'cases'} or contract['boundary'] != 'partition':
        raise ValueError('Frozen frontend must end before partition')
    if set(contract['cases']) != {c['id'] for c in task['cases']}:
        raise ValueError('Frozen frontend must bind every case exactly once')
    for entry in contract['cases'].values():
        if set(entry) != {'stages', 'producer_image'} or set(entry['stages']) != set(STAGES):
            raise ValueError('Frozen frontend requires both original stage manifests')
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', entry['producer_image']):
            raise ValueError('Unbound frontend producer image')
        if any(not re.fullmatch(r'[0-9a-f]{64}', v) for v in entry['stages'].values()):
            raise ValueError('Unbound frontend manifest')


def verify(task, case, root, config=None):
    from open3dbench.stage_manifest import validate_stage_manifest, digest_json
    validate_contract(task)
    root = Path(root)
    entry = task['frontend_inputs']['cases'][case['id']]
    for stage, directory in STAGES.items():
        path = root / directory / 'stage_manifest.json'
        if sha(path) != entry['stages'][stage]:
            raise ValueError('Frozen frontend manifest differs: ' + stage)
        manifest = validate_stage_manifest(path, expected_stage=stage, verify_artifacts=True)
        if any(manifest[k] != v for k, v in {'design': case['design'], 'pdk': case['pdk'], 'integration': 'lol'}.items()):
            raise ValueError('Frozen frontend design/platform differs')
        if manifest['provenance']['container_image_id'] != entry['producer_image']:
            raise ValueError('Frozen frontend producer image differs')
        upstream = 'none' if stage == 'input' else entry['stages']['input']
        if manifest['signature']['upstream'] != upstream:
            raise ValueError('Frozen frontend upstream chain differs')
        if config is not None:
            keys = ('input',) if stage == 'input' else ('floorplan', 'pins')
            expected = {k: config[k] for k in ('design', 'type', 'pdk')}
            expected.update(threads=config.get('threads', 32))
            expected.update({k: config[k] for k in keys if k in config})
            if manifest['signature']['config'] != digest_json(expected):
                raise ValueError('Frozen frontend configuration differs: ' + stage)
    if read(root / '10_floorplan/io_metrics.json')['seed'] != 1:
        raise ValueError('Frozen frontend I/O seed differs')
    return entry


def materialize(task, case, source, destination, config):
    source, destination = Path(source), Path(destination)
    entry = verify(task, case, source, config)
    if destination.exists():
        raise FileExistsError('Fixed input import requires a fresh run')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.frozen-frontend-', dir=destination.parent))
    try:
        for directory in STAGES.values():
            shutil.copytree(source / directory, temporary / directory)
        verify(task, case, temporary, config)
        write(temporary / 'run.json', {
            'design': case['design'], 'pdk': case['pdk'], 'type': 'lol', 'status': 'running',
            'frontend_prefix': {'source': str(source), 'boundary': 'partition',
                'kind': 'task-frozen-input', 'contract_sha256': digest(task['frontend_inputs']),
                'producer_image': entry['producer_image']},
            'stages': {stage: {'directory': directory, 'status': 'success',
                'execution_mode': 'imported-task-input', 'source_manifest_sha256': entry['stages'][stage]}
                for stage, directory in STAGES.items()}})
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    from .contract import validate
    from .common import evaluator_identity
    from open3dbench.config import load_config
    from open3dbench.runner import FlowRunner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--phase', choices=('frontend', 'backend'), required=True)
    args = parser.parse_args()
    plan = read(args.plan); task = read(plan['task'])
    if validate(task) != plan['task_digest'] or task['evaluator_sha256'] != evaluator_identity():
        raise ValueError('Unbound fixed-input execution')
    case = next(c for c in task['cases'] if c['id'] == plan['case'])
    config = load_config(Path(plan['output']) / 'resolved.yaml')
    root = Path(plan['output']) / 'run'; config['run_dir'] = str(root)
    if args.phase == 'frontend':
        materialize(task, case, plan['frontend_prefix'], root, config)
    else:
        verify(task, case, root, config)
    runner = FlowRunner(Path(plan['source']), config, resume=True)
    runner.run(start='partition' if args.phase == 'frontend' else 'pdn',
               stop='placement' if args.phase == 'frontend' else 'cts')


if __name__ == '__main__':
    main()
