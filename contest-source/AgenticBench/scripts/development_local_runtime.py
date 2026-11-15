"""Explicit fresh-qualification runtime for three new task contracts.

The qualified producer is never relabeled as the development revision. Reuse
the maintained resource locks, attempt accounting and credential-free Docker
transport, while binding this diagnostic execution to a verified fleet release.
"""
import hashlib
import json
from pathlib import Path

from scripts.run_local_task_smoke import LocalSmokeRuntime, DockerEvaluationTransport
from scripts.verify_release import verify
from local_bench.model import load_task
from local_bench import reference_latest as latest

TASKS = frozenset({'tp_gnn_post_placement_timing', 'placement_initialization', 'placement_preconditioner'})


class DevelopmentRuntime(LocalSmokeRuntime):
    def __init__(self, manifest, expected_sha256, root, gpu, task):
        if task not in TASKS or not str(gpu).isdigit():
            raise ValueError('Development execution outside approved task/device scope')
        self.manifest, self.manifest_sha256 = Path(manifest), expected_sha256
        self.task = task
        self.release = json.loads(self.manifest.read_text())
        self.framework = Path(self.release['framework'])
        self.source = Path(self.release['development_source'])
        self.source_commit = self.release['development_revision']
        self.batch = Path(root).resolve(strict=True) / 'runs'
        self.image = latest.IMAGE_ID
        self.gpus = [int(gpu)]
        self.cpus, self.memory_gb = 16, 48
        self.share_gpu, self.min_gpu_free_mb, self.min_data_free_gb = True, 4096, 64
        self.reference_baseline = None
        self.audit_framework_identity = self.latest_framework_identity = None
        if any(self.batch == p or self.batch in p.parents or p in self.batch.parents
               for p in (self.framework, self.source)):
            raise ValueError('Writable batch overlaps immutable release')
        self._verify()
        (self.batch / 'locks').mkdir(exist_ok=True)

    def _verify(self):
        receipt = verify(self.manifest, self.manifest_sha256)
        if (receipt['development_revision'] != self.source_commit
                or receipt['framework_sha256'] != self.release['framework_sha256']):
            raise ValueError('Release changed after admission')
        return receipt

    def execute(self, command, output, log, contract):
        checked = self._verify()
        return super().execute(command, output, log, {**contract,
            'execution_identity': checked, 'execution_scope': 'development fresh task qualification/candidates',
            'reference_promotion': False})

    def docker(self, command, gpu, scratch=None):
        self._verify()
        if self.image_preparation:
            raise ValueError('Development adapter does not build native routing images')
        line = super().docker(command, gpu, scratch)
        line[line.index('OPEN3D_LINK_FROZEN_UPSTREAM=1')] = 'OPEN3D_LINK_FROZEN_UPSTREAM=0'
        actual = self.release['runtime']['actual_image']
        index = line.index(self.image)
        line[index:index+1] = ['-v', f'{self.framework}:{self.framework}:ro',
            '-e', 'OMP_NUM_THREADS=4', '-e', 'MKL_NUM_THREADS=4',
            '-e', 'OPEN3DBENCH_IMAGE=' + self.image,
            '-e', 'OPEN3DBENCH_ACTUAL_DOCKER_IMAGE_ID=' + actual,
            *(['-e', 'OPEN3DBENCH_RUNTIME_EQUIVALENCE_SHA256=' + self.release['runtime']['equivalence']['sha256']] if actual != self.image else []),
            actual]
        return line


class DevelopmentTransport(DockerEvaluationTransport):
    def __call__(self, task_dir, candidate_dir, source, output_dir, **kwargs):
        spec = load_task(task_dir)
        if spec.task_id != self.runtime.task or spec.task_id not in TASKS:
            raise ValueError('Development transport task mismatch')
        config = kwargs.get('base_config')
        if config is not None:
            if Path(config).resolve(strict=True) != spec.base_config.resolve(strict=True):
                raise ValueError('Only the released task default configuration is allowed')
            # Omitting an identical default avoids pretending it is a writable
            # batch input. The native CLI loads this exact read-only task file.
            kwargs['base_config'] = None
        return super().__call__(task_dir, candidate_dir, source, output_dir, **kwargs)
