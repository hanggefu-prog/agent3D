"""Explicit proof for the maintained OCI/classic serialization of one runtime.

This never rewrites either image identity. Build receipts retain the actual
base ID and bind the independently audited canonical identity separately.
"""
from pathlib import Path
from . import reference_latest as latest

CLASSIC_ID = 'sha256:851d20609627ed92e97b9489d49a86885d58f06426492b1ab848997fbfbd0784'


def validate_base_equivalence(binding, actual_base, inspected=None):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError('Missing explicit base runtime equivalence binding')
    path = Path(binding['path']).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Runtime proof must be a regular file')
    if latest.sha256(path) != binding['sha256']:
        raise ValueError('Runtime equivalence proof changed')
    proof = latest.read_json(path)
    if (actual_base != CLASSIC_ID or proof.get('status') != 'verified'
            or proof.get('canonical_oci_digest') != latest.IMAGE_ID
            or proof.get('docker_config_image_id') != actual_base
            or proof.get('binaries_sha256') != latest.BINARY_SHA256
            or not isinstance(proof.get('rootfs_layers'), list) or not proof['rootfs_layers']):
        raise ValueError('Runtime proof does not bind the exact maintained runtime')
    if inspected is not None and (len(inspected) != 1 or inspected[0].get('Id') != actual_base
            or inspected[0].get('RootFS', {}).get('Layers') != proof['rootfs_layers']):
        raise ValueError('Installed base layers differ from the independent proof')
    return {'path': str(path), 'sha256': binding['sha256']}


def validate_latest_build_base(receipt):
    base = receipt.get('base_image_id')
    binding = receipt.get('base_runtime_equivalence')
    if base == latest.IMAGE_ID:
        if binding is not None:
            raise ValueError('Canonical base must not claim a classic runtime proof')
    else:
        validate_base_equivalence(binding, base)
    return base
