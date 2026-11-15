"""Explicit classic-Docker address for an independently audited OCI runtime.

This does not change the canonical physical identity or accept arbitrary images.
Both image identities and the proof hash enter every attempt receipt. Native
candidate builds are deliberately unsupported by this local-task-only adapter.
"""
import hashlib
import json
from pathlib import Path
import subprocess

from local_bench import reference_latest as latest
from scripts.run_local_task_smoke import LocalSmokeRuntime

CLASSIC_ID = 'sha256:851d20609627ed92e97b9489d49a86885d58f06426492b1ab848997fbfbd0784'


def validate_proof(path, expected_sha256, inspect):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Runtime proof must be a regular, non-symlink file')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Runtime equivalence proof hash mismatch')
    proof = json.loads(path.read_text())
    if (proof.get('status') != 'verified' or proof.get('canonical_oci_digest') != latest.IMAGE_ID
            or proof.get('docker_config_image_id') != CLASSIC_ID
            or proof.get('binaries_sha256') != latest.BINARY_SHA256
            or not proof.get('rootfs_layers')):
        raise ValueError('Runtime proof does not bind the exact approved runtime')
    if (len(inspect) != 1 or inspect[0].get('Id') != CLASSIC_ID
            or inspect[0].get('RootFS', {}).get('Layers') != proof['rootfs_layers']):
        raise ValueError('Installed image differs from independently audited layers')
    return {'canonical_image': latest.IMAGE_ID, 'actual_docker_image_id': CLASSIC_ID,
            'equivalence_proof': str(path), 'equivalence_proof_sha256': expected_sha256,
            'scope': 'Explicit OCI/classic identity equivalence; no algorithm or physical change'}


class QualifiedLocalRuntime(LocalSmokeRuntime):
    def __init__(self, *args, equivalence_proof, equivalence_proof_sha256, **kwargs):
        self.equivalence_proof = Path(equivalence_proof)
        self.equivalence_proof_sha256 = equivalence_proof_sha256
        super().__init__(*args, **kwargs)
        if self.reference_baseline != latest.BASELINE_ID or self.image != latest.IMAGE_ID:
            raise ValueError('Classic adapter is restricted to the approved latest family')
        self._validate_address()

    def _validate_address(self):
        inspected = json.loads(subprocess.check_output(
            ['docker', 'image', 'inspect', CLASSIC_ID], text=True))
        return validate_proof(self.equivalence_proof, self.equivalence_proof_sha256, inspected)

    def execute(self, command, output, log, contract):
        identity = self._validate_address()
        return super().execute(command, output, log, {**contract, 'runtime_identity': identity})

    def docker(self, command, gpu, scratch=None):
        if self.image_preparation:
            raise ValueError('Classic local-task adapter cannot build routing candidate images')
        identity = self._validate_address()
        line = super().docker(command, gpu, scratch)
        index = line.index(self.image)
        line[index:index + 1] = ['-e', 'OPEN3DBENCH_ACTUAL_DOCKER_IMAGE_ID=' + CLASSIC_ID,
                                '-e', 'OPEN3DBENCH_RUNTIME_EQUIVALENCE_SHA256=' + identity['equivalence_proof_sha256'],
                                CLASSIC_ID]
        return line
