"""Read-only, hash-pinned deployment admission; never starts EDA or a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_bench import reference_latest as latest
from local_bench.runtime_equivalence import validate_base_equivalence
from scripts.workspace_framework_identity import _identity, validate_latest_framework


def checked_json(binding):
    path = Path(binding['path']).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Release binding must be a regular, non-symlink file')
    if latest.sha256(path) != binding['sha256']:
        raise ValueError('Release binding hash changed: ' + str(path))
    return latest.read_json(path)


def validate_contract(manifest, lock):
    if manifest.get('schema_version') != 1 or manifest.get('kind') != 'agenticbench-server-release-v1':
        raise ValueError('Unsupported deployment manifest')
    expected = {'baseline_id': latest.BASELINE_ID, 'source_revision': latest.SOURCE_REVISION,
                'native_revision': latest.NATIVE_REVISION, 'sta_revision': latest.STA_REVISION,
                'canonical_image': latest.IMAGE_ID}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Reference identity differs from the framework contract')
    if manifest.get('development_revision') != lock.get('tested_commit'):
        raise ValueError('Development checkout differs from the tested upstream commit')
    if lock.get('qualified_reference', {}).get('source_revision') != latest.SOURCE_REVISION:
        raise ValueError('Dependency lock and reference producer differ')
    if manifest.get('reference_policy') != 'ADOPT' or manifest.get('fallback') != 'none':
        raise ValueError('Deployment must reject silent fallback')


def verify(path, expected_sha256, *, requested_source=None):
    data = checked_json({'path': str(path), 'sha256': expected_sha256})
    framework = Path(data['framework'])
    if framework.resolve() != Path(__file__).resolve().parents[1]:
        raise ValueError('Verifier was imported from a different framework')
    validate_contract(data, latest.read_json(framework/'open3dbench.lock.json'))
    identity = validate_latest_framework(data['reference_source'], framework,
                                         expected_sha256=data['framework_sha256'])
    _identity(data['development_source'], data['development_revision'])
    if requested_source is not None and Path(requested_source).absolute() != Path(data['reference_source']).absolute():
        raise ValueError('Requested execution source is not the qualified reference producer')
    index = checked_json(data['reference_index'])
    if (index.get('source_revision') != latest.SOURCE_REVISION or index.get('baseline_id') != latest.BASELINE_ID
            or set(index.get('cases', {})) != set(latest.CASES)):
        raise ValueError('Reference catalog identity differs')
    actual = data['runtime']['actual_image']
    inspected = json.loads(subprocess.check_output(['docker','image','inspect',actual], text=True))
    if actual != latest.IMAGE_ID:
        validate_base_equivalence(data['runtime']['equivalence'], actual, inspected)
    elif len(inspected) != 1 or inspected[0].get('Id') != actual:
        raise ValueError('Installed canonical image changed')
    return {'status': 'verified', 'scope': 'deployment identities, not new physical qualification',
            'framework_sha256': identity['framework_sha256'],
            'development_revision': data['development_revision'],
            'reference_source_revision': latest.SOURCE_REVISION,
            'runtime_image': actual, 'reference_catalog_cases': sorted(index['cases']),
            'model_calls': 0, 'native_dispatches': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--requested-source', type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.manifest, a.manifest_sha256, requested_source=a.requested_source), sort_keys=True))


if __name__ == '__main__':
    main()
