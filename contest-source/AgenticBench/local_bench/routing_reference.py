"""Read-only identity verification for a controller-owned initial GRT hook run."""
import hashlib
import json
from pathlib import Path
import re

from . import reference_latest as latest
from .model import load_candidate
from .program_bridge import initial_program_text, materialize_program
from .runtime_equivalence import validate_latest_build_base
from .workspace import openroad_source_fingerprint, PREBUILT_OPENROAD_BINARY


def validate_initial_hook_image(run, spec, source_root):
    run = Path(run).resolve(strict=True)
    evaluation, parent = run.parent, run.parent.parent
    def read(path):
        path = Path(path)
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ValueError('Initial routing reference evidence must be canonical regular files')
        return latest.read_json(path)
    candidate_path = parent / 'candidate/candidate.json'
    candidate = read(candidate_path)
    expected = materialize_program(spec, initial_program_text(spec, load_candidate(spec.initial_candidate)), source_root)
    if candidate != expected:
        raise ValueError('Routing normalization reference must use the trusted initial hook')
    canonical = hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    build = read(parent / 'image-build/receipt.json')
    verified = read(evaluation / 'openroad-prebuilt-receipt.json')
    if verified != {**build, 'verification': 'passed'}:
        raise ValueError('Initial hook runtime receipt differs from its verified installed binary')
    validate_latest_build_base(build)
    image = build.get('image_id')
    if (build.get('schema_version') != 1 or build.get('kind') != 'routing-source-installed-openroad'
            or not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image)
            or build.get('source_commit') != latest.SOURCE_REVISION
            or build.get('openroad_commit') != latest.NATIVE_REVISION
            or build.get('task') != spec.task_id or build.get('candidate_sha256') != canonical
            or build.get('binary_path') != str(PREBUILT_OPENROAD_BINARY)
            or not re.fullmatch(r'[0-9a-f]{64}', str(build.get('binary_sha256', '')))):
        raise ValueError('Initial hook build identity differs from the exact task/source/image')
    measured = build.get('runtime_verification', {})
    if (measured.get('binary_path') != build['binary_path']
            or measured.get('binary_sha256') != build['binary_sha256']
            or measured.get('resolved_command') != build['binary_path']
            or measured.get('version_exit_code') != 0):
        raise ValueError('Initial hook image-installed binary verification is missing')
    workspace = evaluation / 'workspace/Open3DBench-V2.0'
    if openroad_source_fingerprint(workspace / 'OpenROAD-3D/openroad') != build.get('source_fingerprint'):
        raise ValueError('Initial hook executed source differs from the built runtime')
    ledger = read(parent / (evaluation.name + '.attempt.json'))
    required = {'kind': 'local-task-smoke-evaluation', 'task': spec.task_id,
        'source_commit': latest.SOURCE_REVISION, 'image': latest.IMAGE_ID,
        'candidate_sha256': latest.sha256(candidate_path), 'create_routing_reference': True, 'reference': None}
    if (ledger.get('schema_version') != 2 or ledger.get('state') != 'exited'
            or ledger.get('exit_code') != 0 or ledger.get('evaluator_process_started') is not True
            or any(ledger.get('contract', {}).get(k) != v for k, v in required.items())):
        raise ValueError('Initial hook lacks successful controller-owned bootstrap provenance')
    result = read(evaluation / 'evaluation_result.json')
    artifacts = result.get('artifacts', {})
    if (result.get('metrics', {}).get('runs_successfully') != 1
            or artifacts.get('task') != spec.task_id or artifacts.get('run_dir') != str(run)
            or artifacts.get('workspace') != str(workspace) or artifacts.get('exit_code') != 0
            or artifacts.get('openroad_runtime') != build['binary_path']):
        raise ValueError('Initial hook native evaluation provenance differs')
    native = artifacts.get('native_flow_execution', [])
    expected_command = ['/usr/bin/python3', '-m', 'open3dbench', '--repo', str(workspace),
        'global-route', str(evaluation / 'resolved_config.yaml'), '--run-dir', str(run)]
    if (len(native) != 1 or native[0].get('command') != expected_command
            or native[0].get('process_started') is not True or native[0].get('returncode') != 0
            or native[0].get('status') != 'completed' or artifacts.get('command') != expected_command):
        raise ValueError('Initial hook did not execute exactly one native standalone GRT')
    return {'image_id': image, 'build_receipt_sha256': latest.sha256(parent/'image-build/receipt.json'),
        'verified_receipt_sha256': latest.sha256(evaluation/'openroad-prebuilt-receipt.json'),
        'candidate_sha256': canonical, 'binary_sha256': build['binary_sha256']}
