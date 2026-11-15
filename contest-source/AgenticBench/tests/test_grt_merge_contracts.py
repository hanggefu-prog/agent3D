"""Integration regressions for task profiles and host-bound repeat analysis; no EDA."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from grt_bench import TIMING_ENDPOINT
from grt_bench.common import digest, sha, write
from grt_bench.contract import matrix, validate
from grt_bench.pareto import freeze_selection, heldout_report
from grt_bench.profiles import timing_task
from grt_bench.queue import Queue
from scripts.estimate_noise_floor import estimate
from test_grt_scoring_v3 import v3_task, outcomes
from test_grt_matrix import task_fixture


class ProfileMergeTests(unittest.TestCase):
    def test_peak_v2_peak_scoring_v3_and_timing_schema3_remain_separate(self):
        peak = task_fixture()
        scored = v3_task()
        timing = timing_task(peak, 'timing-merge-fixture')
        self.assertEqual(timing['endpoint'], TIMING_ENDPOINT)
        self.assertEqual(len({validate(x) for x in (peak, scored, timing)}), 3)
        for source, other in ((peak, timing), (scored, timing), (timing, peak), (timing, scored)):
            hybrid = dict(source, scoring=copy.deepcopy(other['scoring']))
            with self.assertRaises(ValueError): validate(hybrid)
        # Normal validation is read-only: no defaults/version upgrades are injected.
        original = copy.deepcopy(peak)
        validate(peak)
        self.assertEqual(peak, original)

    def test_timing_heldout_helper_handles_single_score_without_false_freeze_claim(self):
        source = task_fixture()
        source['cases'].append(dict(source['cases'][0], id='held', split='test'))
        task = timing_task(source, 'held-timing-fixture')
        def rows(candidate):
            return [{**p, 'status': 'verified', 'metrics': dict.fromkeys(task['metrics'], 1.)}
                    for p in matrix(task, candidate, 'test')]
        candidates = {'c'*64: rows('c'*64)}
        result = heldout_report(task, candidates, rows('b'*64), freeze_selection(candidates))
        self.assertIsNone(result['selection_frozen_before_evaluation'])
        self.assertEqual(result['nondominated'], ['c'*64])
        self.assertEqual(set(result['vectors']['c'*64]), {'score'})


class NoiseIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = v3_task()
        self.trusted_roots = []

    def native_run(self, name, source_task=None, image='sha256:'+'d'*64):
        source_task = source_task or self.task
        root = self.root/name
        self.trusted_roots.append(root)
        queue = Queue(root/'control/queue.sqlite')
        queue.register_host('test-host', {'cpu': 4, 'memory_gb': 4, 'storage_gb': 4,
                                         'gpus': [], 'build_slots': 1})
        resources = {'cpu': 1, 'memory_gb': 1, 'storage_gb': 1, 'gpu_count': 0, 'build_slots': 0}
        rows = []
        for value in outcomes(source_task, 'b'*64):
            payload = {k: value[k] for k in ('task_digest','candidate','case','seed','endpoint','split')}
            job = queue.submit(payload, resources)
            lease = queue.claim('test-host')
            self.assertEqual(lease['id'], job)
            attempt = root/'runs'/job
            runtime = {'candidate': 'b'*64, 'image': image}
            plan = {**payload, 'python_runtime_digest': digest(runtime)}
            write(attempt/'control/plan.json', plan)
            write(attempt/'control/task.json', source_task)
            write(attempt/'control/python-runtime.json', runtime)
            analysis = attempt/'output/analysis'
            write(analysis/'metrics.json', value['metrics'])
            write(analysis/'manifest.json', {'artifacts': {'metrics.json': {
                'bytes': (analysis/'metrics.json').stat().st_size, 'sha256': sha(analysis/'metrics.json')}}})
            write(attempt/'output/execution.json', {'candidate': 'b'*64, 'case': value['case']})
            verified = {**value, 'evidence': {'plan_sha256': sha(attempt/'control/plan.json'),
                                            'analysis_manifest_sha256': sha(analysis/'manifest.json')}}
            write(attempt/'verified/result.json', verified)
            result = {**verified, 'accounting': {'execution': {'exit_code': 0}}}
            write(attempt/'result.json', result)
            queue.finish(job, lease['token'], result, True)
            rows.append(verified)
        return {'outcomes': rows, 'provenance': {'root': str(root), 'task': source_task}}

    def test_run_json_cannot_authorize_its_own_evidence_root(self):
        a,b = self.native_run('a'), self.native_run('b')
        with self.assertRaisesRegex(ValueError, 'operator-authorized'):
            estimate(self.task, [a,b])
        # Authorizing a parent directory also does not authorize every child.
        with self.assertRaisesRegex(ValueError, 'operator-authorized'):
            estimate(self.task, [a,b], trusted_roots=[self.root])
        copied = self.root/'untrusted-copy'
        shutil.copytree(a['provenance']['root'], copied)
        fake = copy.deepcopy(a); fake['provenance']['root'] = str(copied)
        with self.assertRaisesRegex(ValueError, 'operator-authorized'):
            estimate(self.task, [a,fake], trusted_roots=self.trusted_roots)

    def test_native_cli_envelopes_use_separate_trusted_roots(self):
        from scripts.estimate_noise_floor import main
        runs = [self.native_run('a'), self.native_run('b')]
        task_path = self.root/'target.json'; task_path.write_text(json.dumps(self.task))
        paths = []
        for index, run in enumerate(runs):
            path = self.root/('run%d.json' % index)
            path.write_text(json.dumps(run)); paths.append(str(path))
        output = self.root/'noise.json'
        args = ['--task', str(task_path), '--output', str(output)]
        for root in self.trusted_roots: args.extend(['--trusted-root', str(root)])
        result = main(args + paths)
        self.assertFalse(result['synthetic'])
        self.assertEqual(json.loads(output.read_text())['noise_floor'], result['noise_floor'])
        with self.assertRaisesRegex(ValueError, 'cannot declare'):
            main(args + ['--synthetic'] + paths)

    def test_symlinked_host_database_cannot_redirect_trusted_root(self):
        a,b = self.native_run('a'), self.native_run('b')
        path = Path(b['provenance']['root'])/'control/queue.sqlite'
        original = self.root/'outside.sqlite'
        shutil.copyfile(path, original); path.unlink(); path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            estimate(self.task, [a,b], trusted_roots=self.trusted_roots)

    def test_bare_metrics_need_explicit_synthetic_mode(self):
        runs = [outcomes(self.task, 'b'*64), outcomes(self.task, 'b'*64)]
        with self.assertRaisesRegex(ValueError, 'provenance'): estimate(self.task, runs)
        result = estimate(self.task, runs, synthetic=True)
        self.assertTrue(result['synthetic'])
        self.assertEqual(result['candidate'], 'b'*64)
        self.assertEqual(result['provenance'], [{'task_digest': validate(self.task)}]*2)

    def test_synthetic_still_rejects_candidate_task_and_full_seed_mixing(self):
        a = outcomes(self.task, 'b'*64)
        for field, value in (('candidate','c'*64), ('task_digest','f'*64),
                             ('seed', dict(a[0]['seed'], partition=999999))):
            b = copy.deepcopy(a)
            b[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                estimate(self.task, [a,b], synthetic=True)
        with self.assertRaisesRegex(ValueError, 'same candidate'):
            estimate(self.task, [a, outcomes(self.task, 'c'*64)], synthetic=True)

    def test_trusted_host_attempts_are_distinct_even_for_identical_artifacts(self):
        a,b = self.native_run('a'), self.native_run('b')
        first = next((Path(a['provenance']['root'])/'runs').glob('*/output/analysis/metrics.json'))
        second = Path(b['provenance']['root'])/first.relative_to(Path(a['provenance']['root']))
        second.unlink()
        second.hardlink_to(first)
        self.assertEqual(first.stat().st_ino, second.stat().st_ino)
        result = estimate(self.task, [a,b], trusted_roots=self.trusted_roots)
        self.assertFalse(result['synthetic'])
        self.assertEqual(result['noise_floor'], dict.fromkeys(('timing','overflow','temperature_peak_k'),0.))

    def test_copying_attempts_to_new_directories_does_not_create_repeats(self):
        a = self.native_run('a')
        copy_root = self.root/'copied'
        shutil.copytree(a['provenance']['root'], copy_root)
        b = copy.deepcopy(a); b['provenance']['root'] = str(copy_root)
        self.trusted_roots.append(copy_root)
        with self.assertRaisesRegex(ValueError, 'same physical attempt'):
            estimate(self.task, [a,b], trusted_roots=self.trusted_roots)

    def test_explicit_old_scoring_can_be_rescored_without_changing_physical_identity(self):
        old = task_fixture()
        # Match all measurement settings while preserving the old scoring digest.
        old['normalization_floors'] = copy.deepcopy(self.task['normalization_floors'])
        a,b = self.native_run('a',old),self.native_run('b',old)
        result = estimate(self.task, [a,b], trusted_roots=self.trusted_roots)
        self.assertNotEqual(result['task_digest'], result['provenance'][0]['task_digest'])
        self.assertEqual(result['provenance'][0]['task_digest'], validate(old))
        bad = copy.deepcopy(self.task); bad['cases'][0]['input_identity'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'measurement contract'):
            estimate(bad, [a,b], trusted_roots=self.trusted_roots)

    def test_altered_metrics_or_runtime_are_not_noise(self):
        a,b = self.native_run('a'),self.native_run('b')
        altered = copy.deepcopy(b); altered['outcomes'][0]['metrics']['overflow'] += 1
        with self.assertRaisesRegex(ValueError, 'differ from verified'): estimate(self.task,[a,altered], trusted_roots=self.trusted_roots)
        c = self.native_run('c', image='sha256:'+'a'*64)
        with self.assertRaisesRegex(ValueError, 'runtime image'): estimate(self.task,[a,c], trusted_roots=self.trusted_roots)
        changed = copy.deepcopy(self.task); changed['evaluator_sha256'] = 'e'*64
        d = self.native_run('d', source_task=changed)
        with self.assertRaisesRegex(ValueError, 'evaluator implementation'): estimate(self.task,[a,d], trusted_roots=self.trusted_roots)


if __name__ == '__main__': unittest.main()
