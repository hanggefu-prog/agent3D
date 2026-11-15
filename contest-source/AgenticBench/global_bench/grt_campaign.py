"""Server-side, reviewed Codex evolution rounds and a bounded post-GRT matrix pool.

No model is invoked by baseline, prepare, revise-prompt or status. By default run
requires the exact reviewed prompt hash every round (``prompt_review``). Persistent
reservations never replay an uncertain model call or build. A physical cell is
resubmitted only under an explicit ``infrastructure_retry`` policy (or the
``retry-infrastructure`` operator action) after a strict classifier proves the
failure came from the host, never from the candidate; see retries.json.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from grt_bench.archive import Archive, completed_outcomes
from grt_bench.builder import build
from grt_bench.common import digest, evaluator_identity, read, records, sha, verify_records, write
from grt_bench.contract import validate
from grt_bench.queue import Queue
from grt_bench.worker import loop as worker_loop, UncertainExecution
from grt_bench.__main__ import submission_order
from .grt_candidate_library import export_library, record_summary
from .grt_round_prompt import bind_response, check_response, round_prompt, save_bundle
from .grt_round_tools import RoundRuntime
from .workspace_codex import run_workspace_turn
from .agent_adapters import adapter_for, validate_price_table

ACTIVE = {'queued', 'running', 'uncertain', 'cancelling'}
PROMPT_REVIEW = ('every_round', 'first_round')
SESSION_MODES = ('resume', 'fresh')
RETRY_CLASSES = ('disk_full', 'host_memory', 'docker_daemon', 'gpu_admission',
                 'container_launch', 'lease_lost', 'lease_timeout')
# Host-authored worker messages (grt_bench.runtime/worker). Full matches only;
# a candidate cannot author these strings, and anything else fails closed.
_DOCKER = r"\['docker', 'info', '--format', '\{\{\.ServerVersion\}\}'\]"
_SMI = r"\['nvidia-smi', '-i', '[0-9]+', '--query-gpu=memory\.free', '--format=csv,noheader,nounits'\]"
ADMISSION_FAILURES = (
    ('disk_full', r'RuntimeError: Insufficient free disk: [0-9]+ < [0-9]+ bytes'),
    ('host_memory', r'RuntimeError: Insufficient available memory'),
    ('docker_daemon', r"CalledProcessError: Command '" + _DOCKER + r"' returned non-zero exit status [0-9]+\."
                      r"|TimeoutExpired: Command '" + _DOCKER + r"' timed out after 20 seconds"),
    ('gpu_admission', r'RuntimeError: Insufficient GPU memory'
                      r"|CalledProcessError: Command '" + _SMI + r"' returned non-zero exit status [0-9]+\."
                      r"|TimeoutExpired: Command '" + _SMI + r"' timed out after 10 seconds"),
)
CONTAINER_FAILED = 'RuntimeError: Native container failed; inspect retained log'
LEASE_LOST = 'RuntimeError: Lease heartbeat lost; stop container and reconcile'
LEASE_EXPIRED = 'Lease expired; resources stay reserved until reconciliation'
DOCKER_CLIENT_LINE = (r"docker: .+|Error response from daemon: .+|"
                      r"See 'docker run --help'\.|Run 'docker run --help' for more information")


class ReconciliationRequired(ValueError):
    """Host reconciliation must precede any further matrix progress."""


def retry_policy(config):
    """``None`` unless the campaign explicitly opts in."""
    if 'infrastructure_retry' not in config:
        return None
    value = config['infrastructure_retry']
    if not isinstance(value, dict) or set(value) != {'max_attempts_per_cell', 'retryable'}:
        raise ValueError('infrastructure_retry needs exactly max_attempts_per_cell and retryable')
    limit = value['max_attempts_per_cell']
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError('max_attempts_per_cell must be an integer in 1..3')
    classes = value['retryable']
    if (not isinstance(classes, list) or not classes or len(set(classes)) != len(classes)
            or not all(isinstance(c, str) and c in RETRY_CLASSES for c in classes)):
        raise ValueError('retryable must list unique known infrastructure classes')
    return {'max_attempts_per_cell': limit, 'retryable': sorted(classes)}


def campaign_options(config):
    """Strictly validate optional fields; absent fields keep historical behavior."""
    review = config.get('prompt_review', 'every_round')
    if review not in PROMPT_REVIEW:
        raise ValueError('prompt_review must be every_round or first_round')
    session = config.get('session_mode', 'resume')
    if session not in SESSION_MODES:
        raise ValueError('session_mode must be resume or fresh')
    adapter = config.get('agent_adapter', 'codex-cli')
    if not isinstance(adapter, str):
        raise ValueError('Unknown agent adapter')
    adapter_for(adapter)
    if 'price_table' in config:
        validate_price_table(config['price_table'])
    return {'prompt_review': review, 'session_mode': session, 'agent_adapter': adapter,
            'price_table': config.get('price_table'), 'infrastructure_retry': retry_policy(config)}


def _job_events(db, job):
    """Events of the current queue row only (after its latest submission)."""
    rows = [dict(r) for r in db.execute('SELECT seq,state,detail FROM events WHERE job=? ORDER BY seq', (job,))]
    starts = [i for i, r in enumerate(rows) if r['state'] == 'queued']
    return rows[starts[-1]:] if starts else rows


def _terminals(attempt):
    found = []
    for path in sorted(attempt.rglob('terminal.json')) if attempt.is_dir() else []:
        try:
            value = read(path)
        except ValueError:
            return None
        if not isinstance(value, dict) or type(value.get('exit_code')) is not int:
            return None
        found.append((path, value['exit_code']))
    return found


def classify_failure(row, root, events):
    """Return ``(class, evidence)`` for a proven infrastructure failure, else ``(None, reason)``.

    Fail-closed: unknown messages, timeouts, verifier rejections and operator
    cancellations are candidate/terminal outcomes and are never retried.
    """
    import re
    attempt = Path(root) / 'runs' / row['id']
    try:
        result = json.loads(row['result'] or 'null')
    except ValueError:
        return None, 'unreadable result'
    if row['state'] == 'interrupted':
        states = [e['state'] for e in events]
        expired = any(e['state'] == 'uncertain' and json.loads(e['detail'] or 'null') == LEASE_EXPIRED for e in events)
        if ('cancelling' in states or not expired or not isinstance(result, dict)
                or result.get('terminated') is not True or states[-1:] != ['interrupted']):
            return None, 'interruption is not a reconciled lease expiry'
        return 'lease_timeout', {'termination_receipt': result}
    if row['state'] != 'failed' or not isinstance(result, dict) \
            or not {'status', 'error'} <= set(result) <= {'status', 'error', 'failure_kind'} \
            or result['status'] != 'failed' or not isinstance(result['error'], str):
        return None, 'not a worker-reported failure'
    error = result['error']
    for name, pattern in ADMISSION_FAILURES:
        if re.fullmatch(pattern, error):
            # Host admission precedes the attempt directory: nothing ran.
            if attempt.exists():
                return None, 'admission failure with an attempt directory'
            return name, {'error': error}
    if error == LEASE_LOST:
        return 'lease_lost', {'error': error}
    if error != CONTAINER_FAILED:
        return None, 'candidate or unknown failure'
    terminals = _terminals(attempt)
    failing = [(p, c) for p, c in terminals or () if c != 0]
    if len(failing) != 1:
        return None, 'container failure without exactly one failing terminal receipt'
    path, code = failing[0]
    log = path.parent / 'container.log'
    evidence = {'error': error, 'exit_code': code, 'terminal': str(path.relative_to(attempt)),
                'terminal_sha256': sha(path), 'log_sha256': sha(log) if log.is_file() else None}
    if code == 137:
        return None, 'SIGKILL is not proof of host OOM; candidate memory regression remains possible'
    if code == 125 and log.is_file():
        lines = [line for line in log.read_text(errors='replace').splitlines() if line.strip()]
        if lines and all(re.fullmatch(DOCKER_CLIENT_LINE, line) for line in lines):
            return 'container_launch', evidence
    return None, 'container exit is a candidate outcome'


def _receipt(folder):
    path = Path(folder) / 'retries.json'
    value = read(path) if path.exists() else {'version': 1, 'entries': []}
    if value.get('version') != 1 or not isinstance(value.get('entries'), list):
        raise ValueError('Invalid retry receipt')
    return path, value


def recover_retries(root, queue, folder):
    """Finish pending retirements idempotently (crash between steps)."""
    path, value = _receipt(folder)
    for entry in value['entries']:
        if entry['status'] != 'pending':
            continue
        if entry['retained']:
            source, target = Path(root) / 'runs' / entry['job'], Path(root) / entry['retained']
            if source.exists():
                if target.exists():
                    raise ValueError('Retained retry attempt already exists')
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, target)
            elif not target.exists():
                raise ValueError('Retry evidence disappeared')
        with queue.transaction() as db:
            current = db.execute('SELECT * FROM jobs WHERE id=?', (entry['job'],)).fetchone()
            if current is not None:
                if dict(current) != entry['row']:
                    raise ValueError('Queue row changed during retry; reconcile manually')
                Queue.event(db, entry['job'], 'retired', {'attempt': entry['attempt'], 'class': entry['class'],
                            'row': entry['row'], 'retained': entry['retained']})
                db.execute('DELETE FROM jobs WHERE id=?', (entry['job'],))
        entry['status'] = 'retired'
        write(path, value, replace=True)
    return value


def retry_cells(root, queue, folder, ids, policy):
    """Retire every classified failed cell within its limit, or none at all.

    Returns ``(retired, blocked)``; ``blocked`` lists cells that must not be
    retried (candidate failure, exhausted budget or unclassified state).
    """
    recover_retries(root, queue, folder)
    path, value = _receipt(folder)
    rows = [r for r in queue.rows() if r['id'] in ids and r['state'] != 'succeeded']
    planned, blocked = [], []
    with queue.transaction() as db:
        histories = {r['id']: _job_events(db, r['id']) for r in rows}
    for row in rows:
        if row['state'] in {'running', 'queued'}:
            continue  # In flight; the pool owns it.
        if row['state'] in {'uncertain', 'cancelling'}:
            blocked.append({'job': row['id'], 'state': row['state'], 'reason': 'not terminal'})
            continue
        name, evidence = classify_failure(row, root, histories[row['id']])
        attempts = 1 + sum(e['job'] == row['id'] for e in value['entries'])
        if name is None or name not in policy['retryable']:
            blocked.append({'job': row['id'], 'state': row['state'], 'class': name,
                            'reason': evidence if name is None else 'class not retryable'})
        elif attempts >= policy['max_attempts_per_cell']:
            blocked.append({'job': row['id'], 'state': row['state'], 'class': name, 'reason': 'attempt limit reached'})
        else:
            planned.append((row, name, evidence, attempts))
    if blocked or not planned:
        return [], blocked + [{'job': r['id'], 'state': r['state'], 'class': name,
                               'reason': 'withheld: another cell is not retryable'} for r, name, _, _ in planned]
    for row, name, evidence, attempts in planned:
        retained = None
        if (Path(root) / 'runs' / row['id']).exists():
            retained = 'runs-retired/%s/attempt-%d' % (row['id'], attempts)
        value['entries'].append({'job': row['id'], 'attempt': attempts, 'class': name, 'status': 'pending',
            'row': row, 'evidence': evidence, 'policy': policy, 'retained': retained, 'at': time.time()})
    write(path, value, replace=path.exists())
    recover_retries(root, queue, folder)
    return [r['id'] for r, _, _, _ in planned], []


class CandidateEvaluationFailed(ValueError):
    """A finished matrix has failed cells and no possibly live work."""
    def __init__(self, failures):
        self.failures = failures
        super().__init__(json.dumps({"kind": "candidate_evaluation_failed", "cells": failures}))


class PoolFinished(Exception):
    pass


class LocalWorkerClient:
    """Same worker/lease protocol in one trusted server process; no agent access."""
    def __init__(self, queue, host, abort):
        self.queue, self.host, self.abort = queue, host, abort

    def call(self, action, value):
        if action == 'claim':
            if self.abort.is_set() or not any(r['state'] in ACTIVE and
                    json.loads(r['payload']).get('kind', 'evaluation') == 'evaluation' for r in self.queue.rows()):
                raise PoolFinished()
            return self.queue.claim(self.host)
        if action == 'heartbeat':
            if self.abort.is_set():
                raise RuntimeError('Controller is stopping; terminate the owned evaluation')
            self.queue.heartbeat(value['job'], value['token'])
            return {'ok': True}
        if action == 'release':
            return self.queue.release(value['job'], value['token'], value['resources'], value['evidence'])
        if action == 'finish':
            self.queue.finish(value['job'], value['token'], value['result'], value['success'])
            return {'ok': True}
        raise ValueError('Unknown worker operation')


def prepare_runtime(config, candidate):
    """Finish source verification and image installation before an EDA reservation."""
    from .grt_deployment import preflight_build
    preflight_build(config['framework'])
    root = Path(config['root'])
    runtime_path = root / 'control/runtimes' / (candidate + '.json')
    if runtime_path.exists():
        runtime = read(runtime_path)
    else:
        runtime = build(root / 'candidates' / candidate, config['environment']['base_image'],
                        root / 'builds', Path(config.get('compiler_cache', root / 'compiler-cache')),
                        threads=config.get('build_threads', 8))
        write(runtime_path, runtime)
    if runtime['candidate'] != candidate:
        raise ValueError('Runtime candidate changed')
    return runtime


def _run_pool(config, queue, environment, jobs, folder):
    """Drain the queued cells; release GPU as soon as the frontend terminates."""
    abort = threading.Event()
    def work():
        try:
            worker_loop(LocalWorkerClient(queue, config['host'], abort), environment)
        except PoolFinished:
            return
        except BaseException:
            abort.set()
            raise
    with ThreadPoolExecutor(max_workers=config['worker_count']) as pool:
        futures = [pool.submit(work) for _ in range(config['worker_count'])]
        try:
            while not all(f.done() for f in futures):
                rows = [r for r in queue.rows() if r['id'] in jobs]
                allocations = []
                for r in rows:
                    if r['state'] in {'running', 'uncertain', 'cancelling'}:
                        allocations.append({'job': r['id'], 'gpus': json.loads(r['result'])['gpus'],
                            'resources': json.loads(r['allocation'] or r['resources'])})
                counts = {s: sum(r['state'] == s for r in rows) for s in sorted({r['state'] for r in rows})}
                sample = {'at': time.time(), 'counts': counts, 'allocations': allocations}
                with (folder / 'telemetry.jsonl').open('a') as f:
                    f.write(json.dumps(sample) + '\n')
                write(folder / 'status.json', sample, replace=True)
                if any(r['state'] in {'uncertain', 'cancelling'} for r in rows):
                    abort.set()
                time.sleep(2)
            for f in futures:
                f.result()
        except BaseException:
            abort.set()
            raise
    rows = [r for r in queue.rows() if r['id'] in jobs]
    terminal = {'at': time.time(), 'counts': {s: sum(r['state'] == s for r in rows)
                for s in sorted({r['state'] for r in rows})}, 'allocations': []}
    write(folder / 'status.json', terminal, replace=True)
    with (folder / 'telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps(terminal) + '\n')
    return rows


def evaluate_matrix(config, task, candidate, folder):
    from .grt_storage import evaluate_matrix as with_result_storage
    return with_result_storage(_evaluate_matrix, config, task, candidate, folder)


def _evaluate_matrix(config, task, candidate, folder):
    """Submit each cell once; release GPU as soon as the frontend terminates.

    Only an explicit ``infrastructure_retry`` policy retires and resubmits a
    classified cell under the same job identity (one verified outcome per cell).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    root = Path(config['root'])
    archive = Archive(root, task)
    archive.admit(candidate)
    queue = Queue(root / 'control/queue.sqlite')
    expected = submission_order(task, candidate)
    queue.register_host(config['host'], config['capacity'])
    runtime = prepare_runtime(config, candidate)
    from grt_bench.incremental import prepare_matrix, resources_for
    version = 2
    if (folder / 'admission.json').exists():
        # A matrix admitted with v1 proofs resumes with byte-identical v1 proofs.
        admitted = (read(folder / 'admission.json').get('incremental') or {}).values()
        if admitted and all(p.get('version') == 1 for p in admitted):
            version = 1
    proofs = prepare_matrix(root, task, candidate, runtime, version=version)
    backend = (config['environment'].get('phase_resources') or {}).get('backend', config['resources'])
    allocations = [resources_for(proofs.get((p['case'], p['seed']['id'])), config['resources'], backend) for p in expected]
    ids = [digest({'payload': p, 'resources': r}) for p, r in zip(expected, allocations)]
    if any(r['state'] in ACTIVE and r['id'] not in ids and
           json.loads(r['payload']).get('kind', 'evaluation') == 'evaluation' for r in queue.rows()):
        raise ValueError('Another matrix is active in this campaign queue')
    policy = retry_policy(config)
    if policy is not None:
        retry_cells(root, queue, folder, ids, policy)
    if any(r['state'] in {'uncertain', 'cancelling'} and r['id'] in ids for r in queue.rows()):
        raise ReconciliationRequired('Uncertain cells require host reconciliation before resume-evaluation')
    if any(r['state'] in {'uncertain', 'cancelling', 'failed', 'interrupted', 'cancelled'}
           and r['id'] in ids for r in queue.rows()):
        raise ValueError('Failed/uncertain cells require explicit reconciliation; no automatic retry')
    jobs = [queue.submit(p, r) for p, r in zip(expected, allocations)]
    if jobs != ids:
        raise ValueError('Queue job identity differs')
    admission = {'candidate': candidate, 'task_digest': validate(task), 'jobs': jobs,
                 'resources': config['resources'],
                 'backend_resources': (config['environment'].get('phase_resources') or {}).get('backend', config['resources'])}
    if proofs:
        admission['incremental'] = {job: proofs[(p['case'], p['seed']['id'])] for job, p in zip(jobs, expected)}
        admission['allocations'] = dict(zip(jobs, allocations))
    if (folder / 'admission.json').exists():
        if read(folder / 'admission.json') != admission:
            raise ValueError('Matrix reservation changed')
    else:
        write(folder / 'admission.json', admission)
    environment = copy.deepcopy(config['environment'])
    if proofs:
        environment['incremental_plans'] = admission['incremental']
    if candidate != task['optimization']['baseline'] and 'frontend_inputs' not in task:
        for case in environment['cases'].values():
            case.pop('frontend_prefix', None)
    started = time.monotonic()
    while True:
        rows = _run_pool(config, queue, environment, jobs, folder)
        if len(rows) == len(jobs) and all(r['state'] == 'succeeded' for r in rows):
            break
        blocked = []
        if policy is not None:
            retired, blocked = retry_cells(root, queue, folder, jobs, policy)
            if retired:
                cells = dict(zip(ids, zip(expected, allocations)))
                if [queue.submit(*cells[job]) for job in retired] != retired:
                    raise ValueError('Queue job identity differs')
                continue
        failures = [{'job': r['id'], 'state': r['state'], 'result': json.loads(r['result'] or 'null')}
                    for r in rows if r['state'] != 'succeeded']
        if policy is not None:
            reasons = {b['job']: b for b in blocked}
            failures = [dict(f, retry=reasons.get(f['job'])) for f in failures]
        write(folder / 'failures.json', failures, replace=True)
        if any(r['state'] in {'uncertain', 'cancelling', 'running', 'queued'} for r in rows):
            raise ReconciliationRequired('Uncertain cells require host reconciliation before resume-evaluation')
        if len(rows) == len(jobs) and all(r['state'] in {'succeeded', 'failed'} for r in rows):
            if all((f['result'] or {}).get('failure_kind') == 'candidate_execution' for f in failures):
                raise CandidateEvaluationFailed(failures)
        raise ValueError('Incomplete verified matrix; archive admission was not attempted')
    result = archive.ingest(candidate)
    if (folder / 'archive-admission.json').exists():
        if read(folder / 'archive-admission.json')['evaluation'] != result['evaluation']:
            raise ValueError('Archive readback changed')
    else:
        write(folder / 'archive-admission.json', result)
    write(folder / 'completion.json', {'wall_seconds_this_invocation': time.monotonic()-started,
          'verified_cells': len(rows), 'completed_at': time.time()}, replace=True)
    return result


class Campaign:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve(strict=True)
        self.config = read(self.config_path)
        self.root = Path(self.config['root']).resolve(strict=True)
        self.task = read(self.config['task'])
        self.task_id = validate(self.task)
        self.framework = Path(self.config['framework']).resolve(strict=True)
        if self.framework != Path(__file__).resolve().parents[1]:
            raise ValueError('Run the controller from its frozen deployment')
        if evaluator_identity() != self.task['evaluator_sha256']:
            raise ValueError('Evaluator changed since task freeze')
        if self.config['environment']['root'] != str(self.root) or self.config['environment']['framework'] != str(self.framework):
            raise ValueError('Worker environment differs from campaign')
        cap = self.config['capacity']
        if (not 1 <= len(cap['gpus']) <= 2 or len(set(cap['gpus'])) != len(cap['gpus'])
                or cap['gpu_slots_per_device'] != 3):
            raise ValueError('Assign one or two distinct GPUs with three slots each')
        if (type(self.config['worker_count']) is not int or not 1 <= self.config['worker_count'] <= 12
                or self.config['environment']['phase_resources']['backend']['gpu_count'] != 0):
            raise ValueError('Use at most twelve workers and a GPU-free CPU backend')
        if self.config.get('agent_backend', 'codex') not in {'codex', 'claude'}:
            raise ValueError('Unknown agent backend')
        if not 1 <= self.config['evaluation_limit'] <= self.config['round_limit']:
            raise ValueError('Invalid campaign budget')
        self.options = campaign_options(self.config)
        self.control = self.root / 'control'
        self.state_path = self.control / 'campaign.json'
        self.archive = Archive(self.root, self.task)

    @contextmanager
    def locked(self):
        with (self.control / 'campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = read(self.state_path) if self.state_path.exists() else {
                'config_digest': digest(self.config), 'task_digest': self.task_id,
                'evaluations_used': 0, 'rounds': [], 'thread_id': None}
            if state['config_digest'] != digest(self.config) or state['task_digest'] != self.task_id:
                raise ValueError('Campaign configuration is immutable')
            yield state

    def save(self, state):
        write(self.state_path, state, replace=True)

    def baseline(self):
        with self.locked() as state:
            self.save(state)
            return evaluate_matrix(self.config, self.task, self.task['optimization']['baseline'], self.root / 'baseline')

    def prepare(self):
        with self.locked() as state:
            from .grt_deployment import preflight_build
            preflight_build(self.framework)
            if (self.control/'final-test-frozen.json').exists():
                raise ValueError('Final tests released; optimization is sealed')
            if state['rounds'] and state['rounds'][-1]['status'] == 'prepared':
                return state['rounds'][-1]
            if state['rounds'] and state['rounds'][-1]['status'] in {'preparing', 'revising_prompt', 'generating', 'generated', 'candidate_ready', 'building', 'evaluating', 'reconciliation_required'}:
                raise ValueError('Reconcile the incomplete round before preparing another')
            number = len(state['rounds']) + 1
            if number > self.config['round_limit'] or state['evaluations_used'] >= self.config['evaluation_limit']:
                raise ValueError('Campaign budget exhausted')
            view = self.archive.show()
            required = {self.task['optimization']['baseline'], *(r['candidate'] for r in view['frontier'])}
            outcomes = []
            for c in sorted(required):
                outcomes.extend(completed_outcomes(self.root, self.task, c, self.archive.admit(c)))
            folder = self.control / 'rounds' / ('%04d' % number)
            folder.mkdir(parents=True, exist_ok=False)
            row = {'round': number, 'status': 'preparing', 'control': str(folder)}
            state['rounds'].append(row); self.save(state)
            library = self.root / 'libraries' / ('%04d' % number)
            export_library(self.root, self.task, library)
            budget = {'round_number': number, 'round_limit': self.config['round_limit'],
                'evaluation_limit': self.config['evaluation_limit'], 'evaluations_used': state['evaluations_used'],
                'round_reserved': True}
            history = [r['history'] for r in state['rounds'][:-1]]
            from .grt_gpu_checks import limits as gpu_limits
            gpu_budget = gpu_limits(self.config)
            queue = Queue(self.root / 'control/queue.sqlite')
            queue.register_host(self.config['host'], self.config['capacity'])
            bundle = round_prompt(self.task, view, budget, outcomes, history, library=library,
                research_brief=state.get('research_brief'), background=self.background(),
                local_check_limits={'cpu': self.config['agent_cpu'], 'memory_gb': self.config['agent_memory_gb'],
                                    'gpu_count': 0, 'gpu_check': dict(gpu_budget, gpu_count=1),
                                    'round_seconds': self.config['agent_timeout_seconds'],
                                    'tool_seconds': 300, 'source_write_mode': 'development-copy'})
            save_bundle(folder / 'prompt', bundle)
            scratch = self.root / 'scratch' / ('%04d' % number)
            scratch.mkdir(parents=True, exist_ok=False)
            config = {'root': str(self.root), 'task': self.config['task'], 'control': str(folder),
                'workspace': str(self.root / 'workspaces' / ('%04d' % number)), 'scratch': str(scratch),
                'library': str(library), 'bundle_digest': digest(bundle),
                'container': 'o3db-agent-round-' + digest(str(folder))[:20],
                'image': self.config['environment']['base_image'],
                'agent_cpu': self.config['agent_cpu'], 'agent_memory_gb': self.config['agent_memory_gb'],
                'host': self.config['host'], 'gpu_checks': gpu_budget,
                'training_inputs': {k: str(Path(v['manifest']).parent) for k,v in self.config['environment']['cases'].items()
                                    if k in {c['id'] for c in self.task['cases'] if c['split']=='train'}}}
            write(folder / 'tool-config.json', config)
            row.update(status='prepared', implementation_digest=self.implementation_digest(), prompt=str(folder/'prompt/prompt.md'),
                       prompt_sha256=sha(folder/'prompt/prompt.md'), model_invocations=0, matrices=0)
            self.save(state)
            return row

    def revise_prompt(self, research_brief):
        """Revise an unstarted round's presentation; preserve task and evaluation evidence."""
        with self.locked() as state:
            if not state['rounds'] or state['rounds'][-1]['status'] != 'prepared':
                raise ValueError('Only a prepared, unstarted round may revise its prompt')
            row = state['rounds'][-1]
            folder = Path(row['control'])
            if row['model_invocations'] or row['matrices'] or any(
                    (folder / name).exists() for name in ('parent.json', 'container-state.json', 'agent', 'public')):
                raise ValueError('Cannot revise a round after agent or checkout activity')
            old = read(folder / 'prompt/bundle.json')
            verify_records(folder / 'prompt', read(folder / 'prompt/manifest.json')['artifacts'])
            if sha(folder / 'prompt/prompt.md') != row['prompt_sha256']:
                raise ValueError('Prepared prompt changed outside the revision workflow')
            tool_config = read(folder / 'tool-config.json')
            if tool_config['bundle_digest'] != digest(old):
                raise ValueError('Prepared tool binding changed')
            view = self.archive.show()
            required = {self.task['optimization']['baseline'], *(r['candidate'] for r in view['frontier'])}
            outcomes = []
            for candidate in sorted(required):
                outcomes.extend(completed_outcomes(self.root, self.task, candidate, self.archive.admit(candidate)))
            bundle = round_prompt(self.task, view, old['context']['budget'], outcomes,
                [r['history'] for r in state['rounds'][:-1]], library=tool_config['library'],
                workspace=old['context']['workspace'], scratch_directory=old['context']['scratch_directory'],
                local_check_limits=old['context'].get('local_check_limits'), research_brief=research_brief,
                background=self.background())
            # Validate everything before mutating. An interrupted rewrite remains blocked,
            # never silently treated as prepared or retried as a model invocation.
            revision = row.get('prompt_revision', 1)
            history = folder / 'prompt-revisions' / ('%04d' % revision)
            history.mkdir(parents=True, exist_ok=False)
            write(history / 'round.json', row)
            row['status'] = 'revising_prompt'; self.save(state)
            shutil.move(str(folder / 'prompt'), str(history / 'prompt'))
            shutil.move(str(folder / 'tool-config.json'), str(history / 'tool-config.json'))
            save_bundle(folder / 'prompt', bundle)
            tool_config['bundle_digest'] = digest(bundle)
            write(folder / 'tool-config.json', tool_config)
            state['research_brief'] = research_brief
            row.update(status='prepared', prompt_revision=revision + 1,
                       implementation_digest=self.implementation_digest(),
                       prompt_sha256=sha(folder / 'prompt/prompt.md'))
            write(history / 'revision.json', {'at': time.time(), 'task_digest': self.task_id,
                'previous_prompt_sha256': sha(history / 'prompt/prompt.md'),
                'replacement_prompt_sha256': row['prompt_sha256'],
                'research_brief_provenance': bundle['research_brief_provenance']})
            self.save(state)
            return row

    def background(self):
        """Fresh sessions repeat the background; resumed sessions keep round-1 only."""
        return True if self.options['session_mode'] == 'fresh' else None

    def review_required(self, row):
        """``first_round`` waives review after round 1 unless the prompt was revised."""
        return (self.options['prompt_review'] == 'every_round' or row['round'] == 1
                or 'prompt_revision' in row)

    def implementation_digest(self):
        inputs = {}
        for group in ('global_bench', 'grt_bench', 'agentic3d', 'scripts', 'global_tasks/lol_post_grt_pareto', 'global_tasks/lol_post_grt_timing'):
            for path in sorted((self.framework / group).rglob('*')):
                if path.is_file() and path.suffix in {'.py', '.md', '.sh'} and '__pycache__' not in path.parts:
                    inputs[path.relative_to(self.framework).as_posix()] = sha(path)
        return digest(inputs)

    def preflight_codex(self):
        if self.config.get('agent_backend', 'codex') == 'claude':
            private = Path(self.config['provider_config'])
            value = read(private)
            if private.stat().st_mode & 0o077 or not value.get('ANTHROPIC_AUTH_TOKEN') or not value.get('ANTHROPIC_BASE_URL'):
                raise ValueError('Claude requires an owner-only provider configuration')
            subprocess.run([self.config['claude_binary'], '--version'], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            return
        probe = subprocess.run([self.config['codex_binary'], 'login', 'status'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        if probe.returncode:
            raise ValueError('Codex must be authenticated on this server before reserving a model call')

    def run(self, reviewed_prompt_sha256):
        with self.locked() as state:
            if (self.control/'final-test-frozen.json').exists():
                raise ValueError('Final tests released; optimization is sealed')
            if not state['rounds'] or state['rounds'][-1]['status'] != 'prepared':
                raise ValueError('Prepare a new round first; uncertain model calls are never replayed')
            row = state['rounds'][-1]
            folder = Path(row['control'])
            bundle = read(folder / 'prompt/bundle.json')
            verify_records(folder/'prompt', read(folder/'prompt/manifest.json')['artifacts'])
            if sha(row['prompt']) != row['prompt_sha256']:
                raise ValueError('Prepared prompt changed outside the revision workflow')
            required = self.review_required(row)
            if (required or reviewed_prompt_sha256 is not None) and reviewed_prompt_sha256 != row['prompt_sha256']:
                raise ValueError('The exact rendered prompt must be reviewed before run')
            if self.implementation_digest() != row['implementation_digest']:
                raise ValueError('Controller/tool implementation changed after prompt preparation')
            self.preflight_codex()
            from .grt_deployment import preflight_build
            preflight_build(self.framework)
            runtime = RoundRuntime(read(folder/'tool-config.json'))
            runtime.verify_library()
            runtime.start()
            # Reserve before subprocess creation. A crash here cannot spend a second call.
            row.update(status='generating', model_invocations=1, reviewed_prompt_sha256=reviewed_prompt_sha256)
            if not required:
                row['prompt_review'] = 'waived:first_round'
            if self.options['session_mode'] != 'resume':
                row['session_mode'] = self.options['session_mode']
            self.save(state)
            public = folder/'public'; public.mkdir()
            shutil.copyfile(folder/'prompt/output.schema.json', public/'output.schema.json')
            result = None
            try:
                extra = ('mcp_servers.terminal.args=' + json.dumps(['-B', '-m', 'global_bench.grt_round_tools', '--config', str(folder/'tool-config.json')]),
                         'mcp_servers.terminal.env.PYTHONPATH=' + json.dumps(str(self.framework)),
                         'mcp_servers.terminal.enabled_tools=["exec","checkout_parent","gpu_check"]',
                         'mcp_servers.terminal.tool_call_timeout_sec=660',
                         'mcp_servers.terminal.tools.checkout_parent.approval_mode="approve"',
                         'mcp_servers.terminal.tools.gpu_check.approval_mode="approve"')
                environment = dict(os.environ, PYTHONPATH=str(self.framework))
                resume = state['thread_id'] if self.options['session_mode'] == 'resume' else None
                previous = next((r['agent_session_usage'] for r in reversed(state['rounds'][:-1])
                                 if resume and r.get('agent_thread_id') == resume
                                 and (r.get('agent_session_usage') or {}).get('input_tokens') is not None), None)
                if self.config.get('agent_backend', 'codex') == 'claude':
                    from .workspace_claude import run_claude_turn
                    result = run_claude_turn(binary=self.config['claude_binary'], model=self.config['model'],
                        control=public, output=folder/'agent', framework=self.framework,
                        tool_config=folder/'tool-config.json', output_schema=public/'output.schema.json',
                        provider_config=self.config['provider_config'], config_directory=self.config['claude_config_directory'],
                        python=sys.executable, prompt=bundle['prompt'], thread_id=resume,
                        timeout_seconds=self.config['agent_timeout_seconds'], environment=environment,
                        max_attempts=self.config.get('connection_attempts', 5))
                else:
                    result = run_workspace_turn(binary=self.config['codex_binary'], control=public,
                        output=folder/'agent', container=runtime.name, prompt=bundle['prompt'],
                        terminal_script=self.framework/'global_bench/container_terminal.py', python=sys.executable,
                        thread_id=resume, timeout_seconds=self.config['agent_timeout_seconds'],
                        environment=environment, model=self.config['model'], extra_overrides=extra,
                        output_schema=public/'output.schema.json', previous_session_usage=previous,
                        adapter=adapter_for(self.options['agent_adapter']), price_table=self.options['price_table'])
            finally:
                # Remove the entire cgroup, including background descendants, before freezing.
                runtime.stop()
            if result and 'client_attempts' in result:
                row['client_attempts'] = result['client_attempts']
            if isinstance(result, dict) and 'usage' in result:
                row.update(agent_usage=result['usage'], agent_cost=result.get('cost'),
                           agent_session_usage=result.get('session_usage'), agent_thread_id=result.get('thread_id'))
            if result is None or not result['success']:
                self.finish(state, row, 'generation_failed', None, None, '',
                    (result or {}).get('failure_reason') or 'Agent did not complete one valid turn')
                return row
            state['thread_id'] = result['thread_id']
            row['status'] = 'generated'; self.save(state)
            try:
                response = json.loads(result['final_message'])
                receipt = runtime.handler.receipt()
                write(folder/'response.raw.json', response)
                response, binding = bind_response(bundle, response, receipt)
                write(folder/'response-binding.json', binding)
                write(folder/'response.json', response)
                if response['status'] != 'ready':
                    self.finish(state,row,response['status'],response['parent'],None,response['hypothesis'],response['summary'])
                    return row
                from .grt_development import export_submission
                candidate = export_submission(self.archive, runtime.workspace, receipt,
                    response['changed_files'], folder / 'submission')
                note = record_summary(self.root,self.task,candidate,bundle,receipt,response)
                write(folder/'source-verification.json',self.archive.admit(candidate))
                runtime.verify_library()
                from .grt_activation import inspect_activation
                activation = inspect_activation(self.config, self.task, candidate)
                write(folder/'activation.json', activation)
                if activation['status'] == 'inactive':
                    self.finish(state, row, 'rejected', receipt['parent'], candidate,
                        response['hypothesis'], activation['reason'])
                    return row
            except (ValueError, KeyError, TypeError, SyntaxError) as error:
                self.finish(state,row,'rejected',None,None,'',str(error))
                return row
            row.update(status='candidate_ready', candidate=candidate, parent=receipt['parent'], matrices=0,
                       hypothesis=response['hypothesis'])
            self.save(state)
            return self.complete_evaluation(state,row)

    def finish(self,state,row,status,parent,candidate,hypothesis,feedback):
        row['status'] = status
        row['history'] = {'round':row['round'],'split':'train','status':status,'parent':parent,
                          'candidate':candidate,'hypothesis':hypothesis,'feedback':feedback}
        row['finished_at'] = time.time()
        self.save(state)

    def complete_evaluation(self,state,row):
        try:
            if not row['matrices']:
                if state['evaluations_used'] >= self.config['evaluation_limit']:
                    raise ValueError('Campaign evaluation budget exhausted')
                row['status'] = 'building'; self.save(state)
                self.archive.admit(row['candidate'])
                prepare_runtime(self.config, row['candidate'])
                state['evaluations_used'] += 1
                row.update(status='evaluating', matrices=1)
                self.save(state)
            result = evaluate_matrix(self.config,self.task,row['candidate'],Path(row['control'])/'matrix')
        except (ReconciliationRequired, UncertainExecution) as error:
            # Keep reservations and queue leases until termination is proved.
            row['reconciliation_required'] = str(error)
            self.save(state)
            raise
        except CandidateEvaluationFailed as error:
            row["evaluation_failure"] = error.failures
            self.finish(state,row,'failed',row['parent'],row['candidate'],row['hypothesis'],str(error))
            return row
        except Exception as error:
            # Never refund a potentially dispatched or physically running matrix.
            # Build/deployment failures with no queue admission are safe to refund.
            if row['matrices'] and self.undispatched(row):
                row.setdefault('budget_adjustments', []).append({'at': time.time(), 'delta': -1,
                    'reason': 'Failure before any candidate queue admission', 'error': str(error)})
                state['evaluations_used'] -= 1
                row['matrices'] = 0
            self.finish(state,row,'reconciliation_required',row['parent'],row['candidate'],row['hypothesis'],str(error))
            raise
        row.pop('reconciliation_required', None)
        row['archive_admission'] = result
        self.finish(state,row,'verified' if result['feasible'] else 'infeasible',row['parent'],row['candidate'],
                    row['hypothesis'],json.dumps({'objectives':result['evaluation']['vector'],'non_dominated':result['non_dominated']}))
        return row

    def retry_infrastructure(self, retryable=None, baseline=False):
        """Operator fallback: retire classified infrastructure cells, then re-evaluate.

        Uses the configured policy, or an explicit operator class list (attempt
        limit 3). A failed round reopens without spending another evaluation.
        """
        policy = self.options['infrastructure_retry']
        if retryable is not None:
            policy = retry_policy({'infrastructure_retry': {'retryable': retryable,
                'max_attempts_per_cell': policy['max_attempts_per_cell'] if policy else 3}})
        if policy is None:
            raise ValueError('Name retryable infrastructure classes or configure infrastructure_retry')
        with self.locked() as state:
            if baseline:
                candidate, folder, row = self.task['optimization']['baseline'], self.root / 'baseline', None
            else:
                if not state['rounds'] or state['rounds'][-1]['status'] not in {'failed', 'evaluating', 'reconciliation_required'}:
                    raise ValueError('Only the latest failed or evaluating round may retry infrastructure')
                row = state['rounds'][-1]
                if row.get('matrices') != 1 or 'candidate' not in row:
                    raise ValueError('Round has no reserved matrix')
                candidate, folder = row['candidate'], Path(row['control']) / 'matrix'
            if not (folder / 'admission.json').exists():
                raise ValueError('Matrix was never admitted')
            admission = read(folder / 'admission.json')
            if admission['candidate'] != candidate:
                raise ValueError('Matrix reservation changed')
            queue = Queue(self.root / 'control/queue.sqlite')
            retired, blocked = retry_cells(self.root, queue, folder, admission['jobs'], policy)
            if blocked or not retired:
                raise ValueError('No retryable infrastructure failure: ' + json.dumps(blocked, sort_keys=True))
            config = dict(self.config, infrastructure_retry=policy)
            if row is None:
                return evaluate_matrix(config, self.task, candidate, folder)
            row.setdefault('infrastructure_retries', []).append(
                {'at': time.time(), 'jobs': retired, 'previous_status': row['status'], 'policy': policy})
            row['status'] = 'evaluating'; row.pop('history', None); row.pop('finished_at', None)
            self.save(state)
            try:
                result = evaluate_matrix(config, self.task, candidate, folder)
            except (ReconciliationRequired, UncertainExecution) as error:
                row['reconciliation_required'] = str(error); self.save(state)
                raise
            except CandidateEvaluationFailed as error:
                row['evaluation_failure'] = error.failures
                self.finish(state,row,'failed',row['parent'],row['candidate'],row['hypothesis'],str(error))
                return row
            except Exception as error:
                self.finish(state,row,'reconciliation_required',row['parent'],row['candidate'],row['hypothesis'],str(error))
                raise
            row.pop('reconciliation_required', None)
            row['archive_admission'] = result
            self.finish(state,row,'verified' if result['feasible'] else 'infeasible',row['parent'],row['candidate'],
                        row['hypothesis'],json.dumps({'objectives':result['evaluation']['vector'],'non_dominated':result['non_dominated']}))
            return row

    def resume_evaluation(self):
        with self.locked() as state:
            if not state['rounds'] or state['rounds'][-1]['status'] not in {'candidate_ready', 'evaluating'}:
                raise ValueError('Only a ready candidate or reserved evaluation may resume, never the model call')
            return self.complete_evaluation(state,state['rounds'][-1])

    def undispatched(self, row):
        return (not (Path(row['control'])/'matrix/admission.json').exists() and
                not any(json.loads(r['payload']).get('candidate') == row['candidate']
                        for r in Queue(self.root/'control/queue.sqlite').rows()))

    def reconcile_undispatched(self, candidate, reason):
        """Explicit host recovery only; preserve the frozen source and model session."""
        if not reason or not reason.strip():
            raise ValueError('An explicit recovery reason is required')
        with self.locked() as state:
            row = state['rounds'][-1]
            if row['status'] != 'reconciliation_required' or row.get('candidate') != candidate:
                raise ValueError('Recovery must identify the exact blocked candidate')
            if not self.undispatched(row):
                raise ValueError('Candidate queue admission exists; no undispatched refund or replay')
            if any(r['state'] in ACTIVE for r in Queue(self.root/'control/queue.sqlite').rows()):
                raise ValueError('Campaign queue is not idle')
            self.archive.admit(candidate)
            from .grt_deployment import preflight_build
            dependencies = preflight_build(self.framework)
            recovery = self.control/'recoveries'/('%04d-%04d' % (row['round'], len(row.get('recoveries', []))+1))
            write(recovery/'before.json', state)
            refund = row['matrices']
            if refund not in (0, 1) or state['evaluations_used'] < refund:
                raise ValueError('Invalid reservation accounting')
            receipt = {'at': time.time(), 'candidate': candidate, 'task_digest': self.task_id,
                'reason': reason, 'previous_state_sha256': sha(recovery/'before.json'),
                'evaluation_refund': refund, 'candidate_queue_rows': 0, 'build_dependencies': dependencies,
                'implementation_digest': self.implementation_digest()}
            write(recovery/'receipt.json', receipt)
            state['evaluations_used'] -= refund
            row.update(status='candidate_ready', matrices=0)
            row.setdefault('recoveries', []).append(str(recovery/'receipt.json'))
            self.save(state)
            return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['status','baseline','prepare','revise-prompt','run','resume-evaluation',
                                     'retry-infrastructure'])
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--reviewed-prompt-sha256')
    p.add_argument('--research-brief', type=Path)
    p.add_argument('--retryable', help='Comma-separated infrastructure classes (retry-infrastructure only)')
    p.add_argument('--baseline', action='store_true', help='retry-infrastructure on the baseline matrix')
    a = p.parse_args(); c = Campaign(a.config)
    if (a.action == 'revise-prompt') != bool(a.research_brief):
        p.error('--research-brief is required only for revise-prompt')
    if a.action != 'retry-infrastructure' and (a.retryable or a.baseline):
        p.error('--retryable/--baseline apply only to retry-infrastructure')
    if a.action == 'status':
        result = read(c.state_path) if c.state_path.exists() else {'status':'not_started'}
    elif a.action == 'baseline': result = c.baseline()
    elif a.action == 'prepare': result = c.prepare()
    elif a.action == 'revise-prompt': result = c.revise_prompt(a.research_brief.read_text())
    elif a.action == 'run': result = c.run(a.reviewed_prompt_sha256)
    elif a.action == 'retry-infrastructure':
        result = c.retry_infrastructure(a.retryable.split(',') if a.retryable else None, a.baseline)
    else: result = c.resume_evaluation()
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
