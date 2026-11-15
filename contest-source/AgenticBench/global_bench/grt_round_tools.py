"""One round's stdio MCP bridge: isolated terminal and a bound parent checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from grt_bench.common import confined, digest, read, sha, verify_records, write
from grt_bench.runtime import mount, stop_and_confirm
from .container_terminal import ContainerTerminal, TOOL, serve
from .grt_round_checkout import RoundParentCheckout


class RoundRuntime:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['root']).resolve(strict=True)
        self.control = Path(config['control']).resolve(strict=True)
        self.control.relative_to(self.root / 'control/rounds')
        self.bundle = read(self.control / 'prompt/bundle.json')
        if digest(self.bundle) != config['bundle_digest']:
            raise ValueError('Prompt binding changed')
        self.task = read(config['task'])
        self.workspace = Path(config['workspace'])
        self.scratch = Path(config['scratch']).resolve(strict=True)
        self.library = Path(config['library']).resolve(strict=True)
        for path in (self.workspace, self.scratch, self.library):
            path.resolve().relative_to(self.root)
        if len({self.workspace.resolve(), self.scratch, self.library}) != 3:
            raise ValueError('Round mounts must be separate')
        self.name = config['container']
        if self.name != 'o3db-agent-round-' + digest(str(self.control))[:20]:
            raise ValueError('Container name differs from bound round')
        self.handler = RoundParentCheckout(self.root, self.task, self.bundle, self.control, self.workspace)

    def verify_library(self):
        manifest = self.library / 'manifest.json'
        if sha(manifest) != self.bundle['context']['candidate_library']['manifest_sha256']:
            raise ValueError('Reading library binding changed')
        verify_records(self.library, read(manifest)['artifacts'])

    def command(self, editing, *, name=None, gpus=None):
        image = self.parent_runtime()['image'] if editing else self.config['image']
        import re
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('Agent runtime must be pinned')
        cpu, memory = self.config['agent_cpu'], self.config['agent_memory_gb']
        if type(cpu) is not int or not 1 <= cpu <= 16 or type(memory) is not int or not 1 <= memory <= 64:
            raise ValueError('Invalid local-check resource bound')
        command = ['docker', 'run', '--detach', '--rm', '--pull=never', '--name', name or self.name,
            '--label', 'agentic3d.round=' + digest(str(self.control)), '--network=none', '--read-only',
            '--cap-drop=ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '512',
            '--cpus', str(cpu), '--memory', str(memory)+'g', '--memory-swap', str(memory)+'g',
            '--user', '{}:{}'.format(os.getuid(), os.getgid()), '--workdir', '/workspace/checks']
        command += mount(self.library, '/workspace/candidates', True)
        command += mount(self.scratch, '/workspace/checks', False)
        command += mount(self.scratch / 'tmp', '/tmp', False)
        # The development copy is writable. Final submission is a separate clean
        # export, never this mutable development environment or its build outputs.
        if editing:
            command += mount(self.workspace, '/workspace/subject', False)
            from grt_bench.algorithm_scope import DREAMPLACE
            command += mount(confined(self.workspace, DREAMPLACE), '/opt/dreamplace/dreamplace', False)
        if gpus:
            if len(gpus) != 1 or not str(gpus[0]).isdigit():
                raise ValueError('One scheduler-assigned GPU is required')
            command += ['--gpus', 'device=' + str(gpus[0])]
        for name, host_path in sorted(self.config.get('training_inputs', {}).items()):
            from grt_bench.contract import identifier
            identifier(name)
            if name not in {c['id'] for c in self.task['cases'] if c['split'] == 'train'}:
                raise ValueError('Only training input mounts are allowed')
            command += mount(host_path, '/inputs/' + name, True)
        for value in ['HOME=/workspace/checks/home', 'TMPDIR=/tmp', 'PYTHONDONTWRITEBYTECODE=1',
                      'PYTHONPYCACHEPREFIX=/tmp/python-cache',
                      'PYTHONPATH=/workspace/subject:/opt/dreamplace:/opt/dreamplace/dreamplace',
                      'OMP_NUM_THREADS='+str(cpu), 'OPENBLAS_NUM_THREADS='+str(cpu),
                      'MKL_NUM_THREADS='+str(cpu), 'XDG_CACHE_HOME=/workspace/checks/cache']:
            command += ['--env', value]
        return command + ['--entrypoint', 'sleep', image, 'infinity']

    def parent_runtime(self):
        receipt = self.handler.receipt()
        if receipt is None:
            raise ValueError('Development runtime requires parent checkout')
        parent = receipt['parent']
        from grt_bench.candidates import verify_snapshot
        from grt_bench.incremental import check_runtime
        path = self.root / 'control/runtimes' / (parent + '.json')
        if path.exists():
            value = read(path)
        else:
            source = verify_snapshot(self.root / 'candidates' / parent)
            if (source.get('scope') or {}).get('native_build_required'):
                raise ValueError('Selected native parent has no installed runtime receipt')
            value = {'candidate': parent, 'image': self.config['image'], 'native': False}
        check_runtime(value, parent, self.config['image'])
        return value

    def prepare_development(self, folder):
        from grt_bench.builder import verify_runtime
        from .grt_development import POPULATE_DREAMPLACE
        value = self.parent_runtime()
        if value.get('native'):
            verify_runtime(value)
        write(folder / 'parent-runtime.json', value)
        # Bootstrap uses the installed parent image and the same CPU/memory limits.
        # It sees no controller state, evaluator, credentials or Docker socket.
        command = self.command(False)
        command.remove('--detach')
        command[command.index('--name') + 1] = self.name + '-prepare'
        command[-3:] = ['python3', value['image'], '-c', POPULATE_DREAMPLACE]
        command[command.index('--entrypoint'):command.index('--entrypoint')] = mount(
            self.workspace, '/workspace/subject', False)
        write(folder / 'prepare-command.json', command)
        write(self.control / 'preparation-state.json', {'running': True}, replace=True)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            write(folder / 'prepare-result.json', {'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr})
            if result.returncode:
                raise RuntimeError('Parent development environment preparation failed')
        finally:
            stop_and_confirm(self.name + '-prepare', folder / 'prepare-stop')
            write(self.control / 'preparation-state.json', {'running': False}, replace=True)

    def start(self, editing=False):
        self.verify_library()
        phase = 'editing' if editing else 'reading'
        folder = self.control / ('container-' + phase)
        folder.mkdir(exist_ok=False)
        for name in ('tmp', 'home', 'cache'):
            (self.scratch / name).mkdir(exist_ok=True)
        if editing:
            self.prepare_development(folder)
        command = self.command(editing)
        write(folder / 'command.json', command)
        # Record uncertainty before Docker can create a container. Even a timed-out
        # client is followed by a confirmed removal of this exact owned name.
        write(self.control / 'container-state.json', {'phase': phase, 'running': True}, replace=True)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except BaseException:
            self.stop()
            raise
        write(folder / 'startup.json', {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            self.stop()
            raise RuntimeError('Agent container startup failed; see host startup receipt')
        write(self.control / 'container-state.json', {'phase': phase, 'running': True}, replace=True)

    def stop(self):
        preparation = self.control / 'preparation-state.json'
        if preparation.exists() and read(preparation)['running']:
            stop_and_confirm(self.name + '-prepare', self.control / 'preparation-stop')
            write(preparation, {'running': False}, replace=True)
        if (self.control / 'gpu-checks').exists():
            from .grt_gpu_checks import GPUChecks
            GPUChecks(self).cleanup()
        path = self.control / 'container-state.json'
        if path.exists() and read(path)['running']:
            phase = read(path)['phase']
            stop_and_confirm(self.name, self.control / ('container-' + phase))
            write(path, {'phase': phase, 'running': False}, replace=True)

    def checkout(self, proposal):
        receipt = self.handler.checkout(proposal)
        state = read(self.control / 'container-state.json')
        if state == {'phase': 'reading', 'running': True}:
            self.stop()
            self.start(editing=True)
        elif state != {'phase': 'editing', 'running': True}:
            raise ValueError('Uncertain editing container requires controller reconciliation')
        return receipt


class RoundTools(ContainerTerminal):
    def __init__(self, runtime):
        self.runtime = runtime
        super().__init__(runtime.name)

    def tools(self):
        from .grt_gpu_checks import TOOL as GPU_TOOL
        return [TOOL, self.runtime.bundle['parent_tool'], GPU_TOOL]

    def call_tool(self, name, arguments):
        started = time.time()
        if name == 'checkout_parent':
            result = {'content': [{'type': 'text', 'text': json.dumps(self.runtime.checkout(arguments))}], 'isError': False}
        elif name == 'gpu_check':
            from .grt_gpu_checks import GPUChecks
            value = GPUChecks(self.runtime).run(arguments)
            result = {'content': [{'type': 'text', 'text': json.dumps(value)}], 'isError': value['status'] != 'success'}
        else:
            result = super().call_tool(name, arguments)
        with (self.runtime.control / 'tool-calls.jsonl').open('a') as stream:
            stream.write(json.dumps({'tool': name, 'arguments': arguments, 'started': started,
                                     'finished': time.time(), 'result': result}) + '\n')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    runtime = RoundRuntime(read(args.config))
    serve(runtime.name, sys.stdin.buffer, sys.stdout.buffer, terminal=RoundTools(runtime))


if __name__ == '__main__':
    main()
