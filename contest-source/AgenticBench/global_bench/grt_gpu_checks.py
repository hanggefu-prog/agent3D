"""Budgeted diagnostic GPU commands using the matrix scheduler's resource leases."""
import fcntl
import math
from pathlib import Path
import time

from grt_bench.common import digest, read, sha, write
from grt_bench.contract import identifier
from grt_bench.queue import Queue
from grt_bench.runtime import admission, run_container, stop_and_confirm, UncertainExecution

TOOL = {
    'name': 'gpu_check',
    'description': 'Run one bounded diagnostic command with one leased GPU after parent checkout. '
        'Use a unique request_id. The host enforces the round GPU budget and removes all processes '
        'before releasing the slot. Results are local diagnostics, not a full evaluation or archive score.',
    'inputSchema': {'type': 'object', 'additionalProperties': False,
        'required': ['request_id', 'command', 'timeout_seconds'], 'properties': {
            'request_id': {'type': 'string', 'pattern': '^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$'},
            'command': {'type': 'string', 'minLength': 1},
            'timeout_seconds': {'type': 'integer', 'minimum': 1, 'maximum': 600}}},
}


def limits(config):
    value = config.get('gpu_checks', {'max_calls': 2, 'total_seconds': 900,
                                    'max_seconds': 600, 'storage_gb': 8})
    if set(value) != {'max_calls', 'total_seconds', 'max_seconds', 'storage_gb'}:
        raise ValueError('Explicit GPU check limits required')
    for key, maximum in [('max_calls', 8), ('total_seconds', 3600), ('max_seconds', 600), ('storage_gb', 32)]:
        if type(value[key]) is not int or not 1 <= value[key] <= maximum:
            raise ValueError('Invalid GPU check limit: ' + key)
    return dict(value)


class GPUChecks:
    def __init__(self, runtime):
        self.runtime = runtime
        self.directory = runtime.control / 'gpu-checks'
        self.directory.mkdir(exist_ok=True)
        self.policy = limits(runtime.config)
        self.queue = Queue(runtime.root / 'control/queue.sqlite')

    def run(self, arguments):
        if set(arguments) != {'request_id', 'command', 'timeout_seconds'}:
            raise ValueError('GPU check requires request_id, command and timeout_seconds')
        key, command, seconds = (arguments[k] for k in ('request_id', 'command', 'timeout_seconds'))
        identifier(key)
        if len(key) > 64 or not isinstance(command, str) or not command.strip() or '\0' in command or len(command.encode()) > 65536:
            raise ValueError('Invalid GPU diagnostic command')
        if type(seconds) is not int or not 1 <= seconds <= self.policy['max_seconds']:
            raise ValueError('GPU check exceeds the per-command limit')
        if self.runtime.handler.receipt() is None or not self.runtime.workspace.is_dir():
            raise ValueError('GPU checks require a completed parent checkout')
        with (self.directory / 'budget.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger_path = self.directory / 'budget.json'
            ledger = read(ledger_path) if ledger_path.exists() else {'calls': [], 'seconds_charged': 0}
            for row in ledger['calls']:
                if row['request_id'] == key:
                    if row['arguments_digest'] != digest(arguments):
                        raise ValueError('GPU request_id reused with different arguments')
                    result = self.directory / key / 'result.json'
                    if result.exists():
                        return read(result)
                    raise ValueError('GPU call is incomplete; host reconciliation required')
            if len(ledger['calls']) >= self.policy['max_calls'] or ledger['seconds_charged'] + seconds > self.policy['total_seconds']:
                raise ValueError('Round GPU diagnostic budget exhausted')
            folder = self.directory / key
            folder.mkdir(exist_ok=False)
            resources = {'cpu': self.runtime.config['agent_cpu'],
                'memory_gb': self.runtime.config['agent_memory_gb'], 'storage_gb': self.policy['storage_gb'],
                'gpu_count': 1, 'build_slots': 0}
            payload = {'kind': 'local-check', 'host': self.runtime.config['host'],
                'round': str(self.runtime.control), 'request_id': key, 'arguments_digest': digest(arguments)}
            # Reserve before submitting or launching. Uncertain calls retain the entire reservation.
            row = {'request_id': key, 'arguments_digest': digest(arguments), 'reserved_seconds': seconds}
            ledger['calls'].append(row); ledger['seconds_charged'] += seconds
            write(ledger_path, ledger, replace=True)
            write(folder / 'request.json', {'arguments': arguments, 'resources': resources})
            job = self.queue.submit(payload, resources)
            name = 'grt-local-' + job[:24]
            write(folder / 'job.json', {'job': job, 'container': name})
            lease = self.queue.claim(payload['host'], kind='local-check', job_id=job)
            if lease is None:
                self.queue.cancel(job)
                result = {'status': 'capacity_unavailable', 'request_id': key, 'gpu_seconds': 0,
                          'message': 'No GPU slot is available. No diagnostic was executed.'}
                ledger['seconds_charged'] -= seconds
                write(ledger_path, ledger, replace=True)
                write(folder / 'result.json', result)
                return result
            write(folder / 'lease.json', lease)
            started = time.monotonic()
            result = None
            try:
                admission(self.runtime.root, resources, lease['gpus'])
                cmd = self.runtime.command(True, name=name, gpus=lease['gpus'])
                cmd.remove('--detach')
                entrypoint = cmd.index('--entrypoint')
                image = cmd[entrypoint + 2]  # The selected parent's installed runtime.
                cmd = cmd[:entrypoint] + ['--entrypoint', 'timeout', image,
                    '-k', '5', str(seconds), 'bash', '-lc', command]
                run_container(cmd, folder, name, timeout=seconds + 10,
                              heartbeat=lambda: self.queue.heartbeat(job, lease['token']))
                result = {'status': 'success', 'request_id': key}
            except UncertainExecution:
                raise
            except Exception as error:
                result = {'status': 'failed', 'request_id': key, 'error': str(error)}
            finally:
                # Removal kills background children as well. Failure retains the lease.
                termination = folder / 'termination'; termination.mkdir(exist_ok=True)
                stop_and_confirm(name, termination)
            elapsed = min(seconds, max(1, math.ceil(time.monotonic() - started)))
            result.update(gpu_seconds=elapsed, gpus=lease['gpus'], terminated=True,
                          artifacts=str(folder), command_sha256=digest(command))
            log = folder / 'container.log'
            result['output'] = log.read_text(errors='replace')[-24000:] if log.exists() else ''
            self.queue.finish(job, lease['token'], result, result['status'] == 'success')
            write(folder / 'result.json', result)
            ledger['seconds_charged'] += elapsed - seconds
            write(ledger_path, ledger, replace=True)
            return result

    def cleanup(self):
        """Controller cleanup after CLI exit, interruption, or a lost tool connection."""
        for path in self.directory.glob('*/job.json'):
            value = read(path)
            rows = [r for r in self.queue.rows() if r['id'] == value['job']]
            if not rows or rows[0]['state'] not in {'queued', 'running', 'uncertain', 'cancelling'}:
                continue
            self.queue.cancel(value['job'])
            if rows[0]['state'] == 'queued':
                continue
            folder = path.parent / 'cleanup'; folder.mkdir(exist_ok=True)
            stop_and_confirm(value['container'], folder)
            self.queue.reconcile(value['job'], {'job': value['job'], 'terminated': True,
                                               'evidence_sha256': sha(folder / 'stop.json')})
