"""Campaign harness options and infrastructure retry, with synthetic receipts only."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.common import read, records, sha, write
from grt_bench.contract import METRICS, OBJECTIVES, matrix, validate
from grt_bench.queue import Queue
from global_bench import grt_campaign as campaign
from global_bench.grt_campaign import Campaign, ReconciliationRequired, classify_failure, retry_cells, retry_policy
from global_bench.grt_round_tools import RoundRuntime
import test_grt_campaign as base
import test_grt_optimization as fixtures

DISK = 'RuntimeError: Insufficient free disk: 1 < 2 bytes'
THREAD = '00000000-0000-0000-0000-000000000001'


def failed(error):
    return json.dumps({'status': 'failed', 'error': error})


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def row(self, error=None, state='failed', result=None):
        return {'id': 'a' * 64, 'state': state, 'result': failed(error) if result is None else result}

    def terminal(self, code, log):
        output = self.root / 'runs' / ('a' * 64) / 'execution/backend'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'terminal.json').write_text(json.dumps({'exit_code': code, 'wall_seconds': 1.0, 'container': 'c'}))
        (output / 'container.log').write_text(log)

    def test_host_admission_failures_are_classified_only_without_attempt(self):
        docker = ['docker', 'info', '--format', '{{.ServerVersion}}']
        smi = ['nvidia-smi', '-i', '1', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        cases = {DISK: 'disk_full', 'RuntimeError: Insufficient available memory': 'host_memory',
                 'CalledProcessError: ' + str(subprocess.CalledProcessError(1, docker)): 'docker_daemon',
                 'TimeoutExpired: ' + str(subprocess.TimeoutExpired(docker, 20)): 'docker_daemon',
                 'RuntimeError: Insufficient GPU memory': 'gpu_admission',
                 'CalledProcessError: ' + str(subprocess.CalledProcessError(9, smi)): 'gpu_admission',
                 'TimeoutExpired: ' + str(subprocess.TimeoutExpired(smi, 10)): 'gpu_admission',
                 campaign.LEASE_LOST: 'lease_lost'}
        for error, name in cases.items():
            with self.subTest(error=error):
                self.assertEqual(classify_failure(self.row(error), self.root, [])[0], name)
        (self.root / 'runs' / ('a' * 64)).mkdir(parents=True)
        self.assertIsNone(classify_failure(self.row(DISK), self.root, [])[0])

    def test_candidate_and_unknown_failures_fail_closed(self):
        for error in ('TimeoutError: Physical evaluation exceeded its frozen budget', DISK + ' ',
                      'ValueError: Verifier rejected metrics', 'RuntimeError: Insufficient free disk',
                      'CalledProcessError: ' + str(subprocess.CalledProcessError(-9, ['docker', 'info'])),
                      campaign.CONTAINER_FAILED):
            with self.subTest(error=error):
                self.assertIsNone(classify_failure(self.row(error), self.root, [])[0])
        for result in ('null', json.dumps({'status': 'failed', 'error': DISK, 'extra': 1}), 'not json'):
            with self.subTest(result=result):
                self.assertIsNone(classify_failure(self.row(result=result), self.root, [])[0])
        self.assertIsNone(classify_failure(self.row(DISK, state='cancelled'), self.root, [])[0])

    def test_container_exit_codes(self):
        self.terminal(125, 'docker: Error response from daemon: failed to create task.\n'
                           "Run 'docker run --help' for more information\n")
        name, evidence = classify_failure(self.row(campaign.CONTAINER_FAILED), self.root, [])
        self.assertEqual((name, evidence['exit_code']), ('container_launch', 125))
        self.terminal(125, 'Traceback (most recent call last):\ndocker: forged\n')
        self.assertIsNone(classify_failure(self.row(campaign.CONTAINER_FAILED), self.root, [])[0])
        self.terminal(1, 'docker: Error response from daemon: x\n')
        self.assertIsNone(classify_failure(self.row(campaign.CONTAINER_FAILED), self.root, [])[0])
        self.terminal(137, '')
        self.assertIsNone(classify_failure(self.row(campaign.CONTAINER_FAILED), self.root, [])[0])

    def test_lease_timeout_requires_expiry_and_termination_receipt(self):
        receipt = json.dumps({'job': 'a' * 64, 'terminated': True, 'evidence_sha256': 'b' * 64})
        events = [{'state': 'queued', 'detail': 'null'}, {'state': 'running', 'detail': '{}'},
                  {'state': 'uncertain', 'detail': json.dumps(campaign.LEASE_EXPIRED)},
                  {'state': 'interrupted', 'detail': receipt}]
        row = self.row(state='interrupted', result=receipt)
        self.assertEqual(classify_failure(row, self.root, events)[0], 'lease_timeout')
        cancelled = events[:2] + [{'state': 'cancelling', 'detail': 'null'}] + events[2:]
        self.assertIsNone(classify_failure(row, self.root, cancelled)[0])
        self.assertIsNone(classify_failure(row, self.root, [events[0], events[1], events[3]])[0])

    def test_policy_and_options_are_strict(self):
        self.assertIsNone(retry_policy({}))
        self.assertEqual(retry_policy({'infrastructure_retry': {'max_attempts_per_cell': 2, 'retryable': ['host_memory', 'disk_full']}}),
                         {'max_attempts_per_cell': 2, 'retryable': ['disk_full', 'host_memory']})
        with self.assertRaises(ValueError):
            retry_policy({'infrastructure_retry': {'max_attempts_per_cell': 2, 'retryable': ['oom_kill']}})
        for bad in ([], {'max_attempts_per_cell': 2}, {'max_attempts_per_cell': 4, 'retryable': ['disk_full']},
                    {'max_attempts_per_cell': 0, 'retryable': ['disk_full']},
                    {'max_attempts_per_cell': True, 'retryable': ['disk_full']},
                    {'max_attempts_per_cell': 2, 'retryable': []}, {'max_attempts_per_cell': 2, 'retryable': ['timeout']},
                    {'max_attempts_per_cell': 2, 'retryable': ['disk_full', 'disk_full']},
                    {'max_attempts_per_cell': 2, 'retryable': ['disk_full'], 'extra': 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                retry_policy({'infrastructure_retry': bad})
        for key, value in (('prompt_review', 'never'), ('session_mode', 'new'), ('agent_adapter', 'http'),
                           ('price_table', {}), ('agent_adapter', None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                campaign.campaign_options({key: value})
        defaults = campaign.campaign_options({})
        self.assertEqual((defaults['prompt_review'], defaults['session_mode'], defaults['infrastructure_retry']),
                         ('every_round', 'resume', None))


class HarnessCampaignTests(unittest.TestCase):
    setUp = base.CampaignTests.setUp
    fake_matrix = base.CampaignTests.fake_matrix

    def campaign(self, **options):
        self.config = dict(self.config, **options)
        path = self.f.root / 'campaign-options.json'
        write(path, self.config, replace=path.exists())
        self.c = Campaign(path)
        return self.c

    def fake_turn(self, value=2, **kwargs):
        self.turns.append(kwargs)
        row = read(self.c.state_path)['rounds'][-1]
        runtime = RoundRuntime(read(Path(row['control']) / 'tool-config.json'))
        parent = runtime.bundle['parent_tool']['inputSchema']['properties']['parent']['enum'][0]
        proposal = {'parent': parent, 'reason': 'frontier', 'hypothesis': 'change value',
            'evidence': ['parent measurements'], 'files_to_modify': [fixtures.PARTITION],
            'expected_tradeoffs': dict.fromkeys((*OBJECTIVES, 'power_w'), 'uncertain')}
        receipt = runtime.handler.checkout(proposal)
        (runtime.workspace / fixtures.PARTITION).write_text('value = %d\n' % value)
        response = {**proposal, 'task_digest': validate(self.f.task), 'round': row['round'],
            'selection_id': receipt['selection_id'], 'status': 'ready', 'summary': 'Changed value',
            'change_summary': {'from_reference': 'Changed value.', 'from_parent': 'Changed value.'},
            'changed_files': [fixtures.PARTITION], 'checks': [], 'risks': [], 'next_round_hint': 'Inspect timing'}
        total = {'turns': 1, 'input_tokens': 10 * len(self.turns), 'cached_input_tokens': 0, 'output_tokens': 2}
        return {'success': True, 'thread_id': THREAD, 'final_message': json.dumps(response),
                'usage': {'turns': 1, 'input_tokens': 10, 'cached_input_tokens': 0, 'output_tokens': 2},
                'session_usage': total, 'cost': None}

    def round(self, reviewed, value=2, matrix=None):
        self.turns = getattr(self, 'turns', [])
        with patch.object(RoundRuntime, 'start'), patch.object(RoundRuntime, 'stop'), \
             patch('global_bench.grt_campaign.run_workspace_turn', side_effect=lambda **kw: self.fake_turn(value, **kw)), \
             patch('global_bench.grt_campaign.evaluate_matrix', side_effect=matrix or self.fake_matrix):
            return self.c.run(reviewed)

    def background(self):
        from global_bench import grt_round_prompt as prompts
        return (prompts.TEMPLATES / 'background.md').read_text().strip()

    def test_invalid_options_rejected_at_construction(self):
        for key, value in (('prompt_review', 'none'), ('session_mode', 'x'), ('price_table', {'m': {}}),
                           ('infrastructure_retry', {'max_attempts_per_cell': 9, 'retryable': ['disk_full']})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.campaign(**{key: value})
            self.config.pop(key)

    def test_default_reviews_every_round_and_resumes_session(self):
        first = self.c.prepare(); self.assertEqual(self.round(first['prompt_sha256'])['status'], 'verified')
        second = self.c.prepare()
        self.assertNotIn(self.background(), Path(second['prompt']).read_text())
        with patch('global_bench.grt_campaign.run_workspace_turn') as call:
            with self.assertRaisesRegex(ValueError, 'reviewed'): self.c.run(None)
            call.assert_not_called()
        self.assertEqual(self.round(second['prompt_sha256'], 3)['status'], 'verified')
        self.assertEqual([t['thread_id'] for t in self.turns], [None, THREAD])
        row = read(self.c.state_path)['rounds'][-1]
        self.assertNotIn('prompt_review', row); self.assertNotIn('session_mode', row)
        self.assertEqual(row['agent_usage']['input_tokens'], 10)
        self.assertEqual((row['agent_thread_id'], row['agent_session_usage']['input_tokens']), (THREAD, 20))
        # The resumed turn is measured against the same thread's previous running total.
        self.assertIsNone(self.turns[0]['previous_session_usage'])
        self.assertEqual(self.turns[1]['previous_session_usage']['input_tokens'], 10)
        self.assertEqual(self.turns[-1]['adapter'].name, 'codex-cli')
        self.assertIsNone(self.turns[-1]['price_table'])

    def test_first_round_review_then_waived_but_never_mismatched(self):
        self.campaign(prompt_review='first_round')
        first = self.c.prepare()
        with self.assertRaisesRegex(ValueError, 'reviewed'): self.c.run(None)
        self.round(first['prompt_sha256'])
        self.c.prepare()
        with self.assertRaisesRegex(ValueError, 'reviewed'): self.c.run('0' * 64)
        row = self.round(None, 3)
        self.assertEqual((row['status'], row['prompt_review']), ('verified', 'waived:first_round'))

    def test_revised_later_prompt_requires_review_under_first_round(self):
        self.campaign(prompt_review='first_round', round_limit=3, evaluation_limit=3)
        self.round(self.c.prepare()['prompt_sha256'])
        self.c.prepare()
        revised = self.c.revise_prompt(self.f.task['optimization']['prompt'])
        with self.assertRaisesRegex(ValueError, 'reviewed'): self.c.run(None)
        self.assertEqual(self.round(revised['prompt_sha256'], 3)['status'], 'verified')

    def test_fresh_session_mode_repeats_background_and_never_resumes(self):
        self.campaign(session_mode='fresh')
        first = self.c.prepare(); self.round(first['prompt_sha256'])
        second = self.c.prepare()
        self.assertIn(self.background(), Path(second['prompt']).read_text())
        self.round(second['prompt_sha256'], 3)
        self.assertEqual([t['thread_id'] for t in self.turns], [None, None])
        self.assertEqual([t['previous_session_usage'] for t in self.turns], [None, None])
        self.assertEqual(read(self.c.state_path)['rounds'][-1]['session_mode'], 'fresh')

    def test_reconciliation_keeps_reserved_evaluation_open(self):
        row = self.c.prepare()
        def blocked(*args):
            raise ReconciliationRequired('reconcile')
        with self.assertRaises(ReconciliationRequired):
            self.round(row['prompt_sha256'], matrix=blocked)
        state = read(self.c.state_path)
        self.assertEqual((state['rounds'][-1]['status'], state['evaluations_used']), ('evaluating', 1))
        with patch('global_bench.grt_campaign.evaluate_matrix', side_effect=self.fake_matrix):
            self.assertEqual(self.c.resume_evaluation()['status'], 'verified')
        self.assertNotIn('reconciliation_required', read(self.c.state_path)['rounds'][-1])

    # Real queue and controller; only the physical worker evaluation is synthetic.
    def succeed(self, lease, release, config):
        if lease['resources']['gpu_count']:
            release(config['environment']['phase_resources']['backend'],
                    {'terminated': True, 'container': 'grt-' + lease['id'][:32] + '-frontend', 'checkpoint_sha256': 'a' * 64})
        payload = lease['payload']; attempt = self.f.root / 'runs' / lease['id']
        write(attempt / 'control/task.json', self.f.task)
        write(attempt / 'control/plan.json', payload)
        analysis = attempt / 'output/analysis'; write(analysis / 'metrics.json', dict.fromkeys(METRICS, 9.))
        write(analysis / 'manifest.json', {'artifacts': records(analysis)})
        write(attempt / 'candidate-verification.json', self.f.archive.admit(payload['candidate']))
        verified = {**payload, 'status': 'verified', 'metrics': dict.fromkeys(METRICS, 9.),
            'evidence': {'analysis_manifest_sha256': sha(analysis / 'manifest.json'), 'plan_sha256': sha(attempt / 'control/plan.json')}}
        write(attempt / 'verified/result.json', verified)
        result = {**verified, 'accounting': {'synthetic_test': True},
                  'candidate_verification_sha256': sha(attempt / 'candidate-verification.json')}
        write(attempt / 'result.json', result)
        return result

    def physical(self, config, failures):
        """``failures`` maps a call index to an exception (or factory) for that call."""
        lock, calls = threading.Lock(), []
        def evaluate(lease, environment, heartbeat, release):
            with lock:
                index = len(calls); calls.append(lease['id'])
            if index in failures:
                failure = failures[index]
                raise failure(lease) if callable(failure) and not isinstance(failure, BaseException) else failure
            return self.succeed(lease, release, config)
        return evaluate, calls

    def evaluate(self, config, failures, folder='matrix-test'):
        child = self.f.child()
        if callable(failures):
            failures = failures(len(matrix(self.f.task, child)))
        evaluate, calls = self.physical(config, failures)
        with patch('grt_bench.worker.evaluate', side_effect=evaluate):
            try:
                return campaign.evaluate_matrix(config, self.f.task, child, self.f.root / folder), calls, child
            except ValueError as error:
                error.calls, error.size = calls, len(matrix(self.f.task, child))
                raise

    def retry_config(self, **policy):
        policy = dict({'max_attempts_per_cell': 2, 'retryable': ['disk_full', 'container_launch']}, **policy)
        return dict(self.config, infrastructure_retry=policy,
                    resources={'cpu': 8, 'memory_gb': 24, 'storage_gb': 40, 'gpu_count': 1, 'build_slots': 0})

    def test_classified_cell_is_retired_and_resubmitted_once_under_same_identity(self):
        result, calls, child = self.evaluate(self.retry_config(), {0: RuntimeError('Insufficient free disk: 1 < 2 bytes')})
        size = len(matrix(self.f.task, child))
        self.assertTrue(result['non_dominated'])
        self.assertEqual((len(calls), calls.count(calls[0])), (size + 1, 2))
        rows = [r for r in self.f.queue.rows() if json.loads(r['payload'])['candidate'] == child]
        self.assertEqual(len(rows), size)
        self.assertTrue(all(r['state'] == 'succeeded' for r in rows))
        self.assertEqual(set(read(self.f.root / 'matrix-test/admission.json')['jobs']), {r['id'] for r in rows})
        [entry] = read(self.f.root / 'matrix-test/retries.json')['entries']
        self.assertEqual((entry['job'], entry['class'], entry['status'], entry['attempt'], entry['retained']),
                         (calls[0], 'disk_full', 'retired', 1, None))
        self.assertEqual(json.loads(entry['row']['result'])['error'], DISK)
        with sqlite3.connect(self.f.queue.path) as db:
            retired = db.execute("SELECT detail FROM events WHERE state='retired' AND job=?", (calls[0],)).fetchall()
        self.assertEqual(json.loads(retired[0][0])['row'], entry['row'])
        self.assertEqual(len(retired), 1)

    def test_container_launch_attempt_evidence_is_retained(self):
        def launch_failure(lease):
            output = self.f.root / 'runs' / lease['id'] / 'execution'
            write(output / 'terminal.json', {'exit_code': 125, 'wall_seconds': 0.1, 'container': 'c'})
            (output / 'container.log').write_text('docker: Error response from daemon: OCI runtime create failed.\n')
            return RuntimeError('Native container failed; inspect retained log')
        result, calls, _ = self.evaluate(self.retry_config(), {0: launch_failure})
        self.assertTrue(result['non_dominated'])
        retained = self.f.root / 'runs-retired' / calls[0] / 'attempt-1/execution/terminal.json'
        self.assertEqual(read(retained)['exit_code'], 125)
        self.assertEqual(read(self.f.root / 'matrix-test/retries.json')['entries'][0]['retained'],
                         'runs-retired/%s/attempt-1' % calls[0])

    def test_attempt_limit_stops_retry(self):
        disk = RuntimeError('Insufficient free disk: 1 < 2 bytes')
        with self.assertRaisesRegex(ValueError, 'Incomplete verified matrix') as caught:
            self.evaluate(self.retry_config(), lambda size: {0: disk, size: disk})
        self.assertEqual(len(caught.exception.calls), caught.exception.size + 1)
        [failure] = read(self.f.root / 'matrix-test/failures.json')
        self.assertEqual(failure['retry']['reason'], 'attempt limit reached')
        self.assertEqual(len(read(self.f.root / 'matrix-test/retries.json')['entries']), 1)

    def test_candidate_failure_blocks_all_retries(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete verified matrix'):
            self.evaluate(self.retry_config(), {0: RuntimeError('Insufficient free disk: 1 < 2 bytes'),
                                                1: TimeoutError('Physical evaluation exceeded its frozen budget')})
        self.assertFalse((self.f.root / 'matrix-test/retries.json').exists())
        reasons = [f['retry']['reason'] for f in read(self.f.root / 'matrix-test/failures.json')]
        self.assertEqual(len(reasons), 2)
        self.assertIn('withheld: another cell is not retryable', reasons)

    def test_default_config_never_retries(self):
        config = self.retry_config(); config.pop('infrastructure_retry')
        with self.assertRaisesRegex(ValueError, 'Incomplete verified matrix') as caught:
            self.evaluate(config, {0: RuntimeError('Insufficient free disk: 1 < 2 bytes')})
        self.assertEqual(len(caught.exception.calls), caught.exception.size)
        self.assertFalse((self.f.root / 'matrix-test/retries.json').exists())
        self.assertEqual(set(read(self.f.root / 'matrix-test/failures.json')[0]), {'job', 'state', 'result'})
        child = read(self.f.root / 'matrix-test/admission.json')['candidate']
        with patch('grt_bench.worker.evaluate', side_effect=AssertionError('must not run')):
            with self.assertRaisesRegex(ValueError, 'no automatic retry'):
                campaign.evaluate_matrix(config, self.f.task, child, self.f.root / 'matrix-test')

    def test_uncertain_cell_requires_reconciliation_under_policy(self):
        child = self.f.child()
        queue = Queue(self.f.root / 'control/queue.sqlite')
        job = queue.submit(matrix(self.f.task, child)[0], self.f.resources)
        with queue.transaction() as db:
            db.execute("UPDATE jobs SET state='uncertain',result=? WHERE id=?", (json.dumps({'gpus': []}), job))
        retired, blocked = retry_cells(self.f.root, queue, self.f.root / 'm', [job], retry_policy(self.retry_config()))
        self.assertEqual((retired, blocked[0]['reason']), ([], 'not terminal'))

    def test_operator_retry_reopens_failed_round_without_new_evaluation(self):
        row = self.c.prepare()
        evaluate, calls = self.physical(self.config, {0: RuntimeError('Insufficient free disk: 1 < 2 bytes')})
        with patch('grt_bench.worker.evaluate', side_effect=evaluate):
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                self.round(row['prompt_sha256'], matrix=campaign.evaluate_matrix)
            self.assertEqual(read(self.c.state_path)['rounds'][-1]['status'], 'reconciliation_required')
            with self.assertRaisesRegex(ValueError, 'retryable'):
                self.c.retry_infrastructure(['oom_kill'])
            with self.assertRaisesRegex(ValueError, 'configure infrastructure_retry'):
                self.c.retry_infrastructure()
            result = self.c.retry_infrastructure(['disk_full'])
        self.assertEqual(result['status'], 'verified')
        state = read(self.c.state_path)
        self.assertEqual((state['evaluations_used'], len(state['rounds'][-1]['infrastructure_retries'])), (1, 1))
        self.assertEqual(calls.count(calls[0]), 2)


if __name__ == '__main__':
    unittest.main()
