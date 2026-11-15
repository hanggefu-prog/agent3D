"""Storage changes must not weaken the existing immutable artifact checks."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.common import read, sha, write
from global_bench.grt_storage import Store, POLICY, evaluate_matrix, policy_enabled, verify_available


class StorageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = 'sha256:' + 'f' * 64
        self.calls = []
        self.store = Store(self.root, exporter=self.export, idle=lambda paths: None)

    def export(self, output, image, requests, scratch):
        self.assertEqual(image, self.image)
        self.calls.extend(requests)
        for r in requests:
            (scratch / r['target']).write_bytes((output / r['odb']).read_bytes()[4:])

    def job(self, character):
        job = character * 64; attempt = self.root / 'runs' / job
        out = attempt / 'output'; out.mkdir(parents=True)
        data = b'VERSION 5.8 ;\n' + b'component\n' * 10000
        (out / 'design.def').write_bytes(data)
        (out / 'design.odb').write_bytes(b'ODB:' + data)
        (out / 'design.spef').write_bytes(b'RC' * 50000)
        (out / 'report.json').write_text('{"metric": 1}\n')
        plan = dict(candidate=job, case='case', seed={'id': 's1'}, task_digest='task', split='train')
        write(attempt / 'control/plan.json', plan)
        write(attempt / 'verified/result.json', dict(plan, status='verified', evidence={'plan_sha256': sha(attempt / 'control/plan.json')}))
        write(self.root / 'control/runtimes' / (job + '.json'), {'candidate': job, 'image': self.image})
        write(attempt / 'execution/command.json', ['docker', 'run', 'OPEN3DBENCH_IMAGE_ID=' + self.image, self.image])
        return job, out

    def test_compact_restore_and_recompact_keep_original_bytes(self):
        job, out = self.job('a'); expected = {p.name: p.read_bytes() for p in out.iterdir()}
        result = self.store.compact([job])
        self.assertFalse((out / 'design.def').exists())
        self.assertEqual(result['defs'][job]['removed_paths'], 1)
        with self.store.available([job]):
            self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, expected)
        self.assertFalse((out / 'design.def').exists())
        self.assertEqual(self.store.compact([job])['released_blocks_bytes'], 0)

    def test_generic_deduplication_is_exact_and_scoped(self):
        a, x = self.job('a'); b, y = self.job('b')
        outside = self.root / 'outside.spef'; outside.write_bytes((x / 'design.spef').read_bytes())
        result = self.store.compact([a, b])
        self.assertGreater(result['dedup']['released_blocks_bytes'], 0)
        self.assertEqual((x / 'design.spef').stat().st_ino, (y / 'design.spef').stat().st_ino)
        self.assertNotEqual(outside.stat().st_ino, (x / 'design.spef').stat().st_ino)
        # Existing native prefix import uses copytree/copy2, not writable links.
        import shutil
        copy = self.root / 'editable'; shutil.copytree(y, copy)
        (copy / 'design.spef').write_bytes(b'changed')
        self.assertEqual((x / 'design.spef').read_bytes(), b'RC' * 50000)

    def test_different_spef_bytes_are_not_deduplicated(self):
        a, x = self.job('a'); b, y = self.job('b')
        (y / 'design.spef').write_bytes(b'XX' * 50000)
        self.store.compact([a, b])
        self.assertNotEqual((x / 'design.spef').stat().st_ino, (y / 'design.spef').stat().st_ino)

    def test_reader_batch_exports_identical_def_once(self):
        a, x = self.job('a'); b, y = self.job('b')
        self.store.compact([a, b]); self.calls.clear()
        with self.store.available([a, b]):
            self.assertEqual(len(self.calls), 1)
            self.assertEqual((x / 'design.def').stat().st_ino, (y / 'design.def').stat().st_ino)
        self.assertFalse((x / 'design.def').exists())
        self.assertFalse((y / 'design.def').exists())

    def test_changed_mode_blocks_hardlink_replacement(self):
        a, x = self.job('a'); self.store.seal(a)
        os.chmod(x / 'design.spef', 0o600)
        b, y = self.job('b')
        with self.assertRaisesRegex(ValueError, 'mode changed'):
            self.store.compact([b, a])

    def test_nonidentical_export_is_kept(self):
        a, x = self.job('a'); (x / 'design.def').write_bytes(b'different')
        self.store.compact([a])
        self.assertEqual((x / 'design.def').read_bytes(), b'different')
        self.assertEqual(self.store.receipt(a)['nonidentical_defs_retained'], ['design.def'])

    def test_corrupt_odb_fails_before_restore(self):
        a, x = self.job('a'); self.store.compact([a]); self.calls.clear()
        (x / 'design.odb').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.store.restore(a)
        self.assertFalse(self.calls)
        self.assertFalse((x / 'design.def').exists())

    def test_corrupt_retained_metric_is_not_hidden(self):
        a, x = self.job('a'); self.store.compact([a])
        (x / 'report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.store.restore(a)

    def test_missing_nonregenerable_artifact_is_rejected(self):
        a, x = self.job('a'); self.store.compact([a]); (x / 'design.spef').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing retained'):
            self.store.restore(a)

    def test_wrong_reexport_hash_does_not_publish_def(self):
        a, x = self.job('a'); self.store.compact([a])
        def wrong(output, image, requests, scratch):
            for r in requests: (scratch / r['target']).write_bytes(b'bad')
        self.store.exporter = wrong
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.store.restore(a)
        self.assertFalse((x / 'design.def').exists())

    def test_partial_compaction_with_durable_receipt_recovers(self):
        a, x = self.job('a')
        self.store.seal(a)  # interrupted after receipt, before removing DEF
        self.store.compact([a])
        with self.store.available([a]): self.assertTrue((x / 'design.def').exists())

    def test_live_reader_blocks_compaction(self):
        a, x = self.job('a')
        self.store.idle = lambda paths: (_ for _ in ()).throw(ValueError('live container'))
        with self.assertRaisesRegex(ValueError, 'live'):
            self.store.compact([a])
        self.assertTrue((x / 'design.def').exists())

    def test_storage_lock_excludes_a_concurrent_writer(self):
        other = Store(self.root, exporter=self.export, idle=lambda paths: None)
        with self.store.locked():
            with self.assertRaises(BlockingIOError):
                with other.locked(): self.fail('acquired a busy lock')

    def test_unverified_or_changed_identity_is_rejected(self):
        a, x = self.job('a'); p = x.parent / 'verified/result.json'
        value = read(p); value['status'] = 'failed'; write(p, value, replace=True)
        with self.assertRaisesRegex(ValueError, 'verified'):
            self.store.compact([a])
        self.assertTrue((x / 'design.def').exists())

    def test_path_escape_and_external_symlink_are_rejected(self):
        with self.assertRaises(ValueError): self.store.paths('../outside')
        a, x = self.job('a'); target = self.root / 'elsewhere'; target.write_text('do not touch')
        (x / 'escape').symlink_to(target)
        with self.assertRaises(ValueError): self.store.compact([a])
        self.assertEqual(target.read_text(), 'do not touch')

    def test_storage_is_opt_in_and_does_not_change_evaluator(self):
        self.assertFalse(policy_enabled(self.root))
        write(self.root / 'control/result-storage-policy.json', POLICY)
        self.assertTrue(policy_enabled(self.root))
        write(self.root / 'control/result-storage-policy.json', {'mode': 'unknown'}, replace=True)
        with self.assertRaises(ValueError): policy_enabled(self.root)

    def verifier_command(self, job, out):
        command = ['docker', 'run', '--name', 'grt-original-verify', '-e', 'OPEN3DBENCH_IMAGE_ID=' + self.image,
                   '--mount', 'type=bind,source=' + str(out.parent / 'workspace') + ',target=/workspace,readonly',
                   '--mount', 'type=bind,source=/original/framework,target=/trusted,readonly',
                   '--mount', 'type=bind,source=' + str(out.parent / 'verified') + ',target=/verified',
                   self.image, '-m', 'grt_bench.verify', '/control/plan.json', '/verified/result.json']
        write(out.parent / 'verification/command.json', command)
        return command

    def test_verifier_uses_original_receipt_and_separate_evidence(self):
        job, out = self.job('a'); self.verifier_command(job, out)
        before = (out.parent / 'verified/result.json').read_bytes(); destination = self.root / 'reverification'
        def run(args, folder, name, timeout):
            self.assertIn('type=bind,source=' + str((self.root / 'candidates' / job / 'tree').resolve()) + ',target=/workspace,readonly', args)
            write(destination / 'verified/result.json', read(out.parent / 'verified/result.json'))
            return {'wall_seconds': 1}
        with patch('global_bench.grt_storage.run_container', side_effect=run):
            self.assertTrue(verify_available(self.root, job, destination)['metrics_equal'])
        self.assertEqual((out.parent / 'verified/result.json').read_bytes(), before)

    def test_verifier_rejects_mismatched_receipt(self):
        job, out = self.job('a'); self.verifier_command(job, out); destination = self.root / 'reverification'
        def run(*args, **kwargs):
            write(destination / 'verified/result.json', {'status': 'verified', 'wrong': True})
            return {'wall_seconds': 1}
        with patch('global_bench.grt_storage.run_container', side_effect=run):
            with self.assertRaisesRegex(ValueError, 'differs'):
                verify_available(self.root, job, destination)

    def test_verifier_cannot_overwrite_saved_result(self):
        job, out = self.job('a'); self.verifier_command(job, out)
        with self.assertRaisesRegex(ValueError, 'outside'):
            verify_available(self.root, job, out / 'new-verification')

    def test_verifier_cannot_launch_another_command(self):
        job, out = self.job('a'); command = self.verifier_command(job, out)
        command[-3] = 'grt_bench.flow'
        write(out.parent / 'verification/command.json', command, replace=True)
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            verify_available(self.root, job, self.root / 'reverification')

    def test_matrix_reader_restores_parent_then_compacts_success(self):
        a, x = self.job('a'); self.store.compact([a])
        write(self.root / 'control/result-storage-policy.json', POLICY)
        folder = self.root / 'matrix'
        def evaluate(config, task, candidate, output):
            self.assertTrue((x / 'design.def').exists())
            b, y = self.job('b'); write(output / 'admission.json', {'jobs': [b]})
            return {'verified': True}
        with patch('global_bench.grt_storage.Store', return_value=self.store), patch('global_bench.grt_storage.parent_jobs', return_value=[a]):
            value = evaluate_matrix(evaluate, {'root': str(self.root)}, {}, 'b' * 64, folder)
        self.assertTrue(value['verified']); self.assertFalse((x / 'design.def').exists())
        self.assertFalse((self.root / 'runs' / ('b' * 64) / 'output/design.def').exists())

    def test_failed_matrix_keeps_new_diagnostics_and_restores_parent_archive(self):
        a, x = self.job('a'); self.store.compact([a])
        write(self.root / 'control/result-storage-policy.json', POLICY)
        def evaluate(*args):
            self.job('b'); raise RuntimeError('native failure')
        with patch('global_bench.grt_storage.Store', return_value=self.store), patch('global_bench.grt_storage.parent_jobs', return_value=[a]):
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                evaluate_matrix(evaluate, {'root': str(self.root)}, {}, 'b' * 64, self.root / 'matrix')
        self.assertFalse((x / 'design.def').exists())
        self.assertTrue((self.root / 'runs' / ('b' * 64) / 'output/design.def').exists())

    def test_storage_error_does_not_spend_another_evaluation(self):
        write(self.root / 'control/result-storage-policy.json', POLICY)
        folder = self.root / 'matrix'; write(folder / 'admission.json', {'jobs': ['b' * 64]})
        with patch('global_bench.grt_storage.Store', return_value=self.store), \
                patch('global_bench.grt_storage.parent_jobs', return_value=[]), \
                patch.object(self.store, 'compact', side_effect=ValueError('storage unavailable')):
            value = evaluate_matrix(lambda *args: {'verified': True}, {'root': str(self.root)}, {}, 'b' * 64, folder)
        self.assertEqual(value, {'verified': True})
        self.assertTrue(read(next(folder.glob('storage-error-*.json')))['evaluation_remains_verified'])


if __name__ == '__main__':
    unittest.main()
