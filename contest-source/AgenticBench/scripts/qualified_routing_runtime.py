"""Read-only build-base proof transport; no scientific or image identity change."""
from pathlib import Path

from scripts.qualified_local_runtime import QualifiedLocalRuntime, CLASSIC_ID
from scripts.run_local_task_smoke import LocalSmokeRuntime


class QualifiedRoutingRuntime(QualifiedLocalRuntime):
    def _proof_mount(self, scratch=None):
        self._validate_address()
        proof = Path(self.equivalence_proof).absolute()
        if any(c in str(proof) for c in (':', ',', '\n', '\r')):
            raise ValueError('Unsafe runtime proof mount path')
        protected = [self.batch, self.source, self.framework, self.write_root]
        if scratch is not None:
            protected.append(scratch)
        for entry in protected:
            if entry is None:
                continue
            root = Path(entry).resolve()
            if proof == root or root in proof.parents or proof in root.parents:
                raise ValueError('Runtime proof overlaps execution or source state')
        return f'type=bind,src={proof},dst={proof},readonly'

    def routing_build_identity(self):
        self._proof_mount()
        return CLASSIC_ID, dict(path=str(self.equivalence_proof),
                               sha256=self.equivalence_proof_sha256)

    def docker(self, command, gpu, scratch=None):
        mount = self._proof_mount(scratch)
        if self.image_preparation is None:
            return super().docker(command, gpu, scratch)
        # The unchanged transport invokes the builder once and selects its
        # actual candidate image. Mount only the exact independently checked
        # proof file, not the release directory or a writable replacement.
        line = LocalSmokeRuntime.docker(self, command, gpu, scratch)
        self._validate_address()
        if line[:2] != ['docker', 'run']:
            raise ValueError('Unexpected evaluation command')
        line[2:2] = ['--mount', mount]
        return line
