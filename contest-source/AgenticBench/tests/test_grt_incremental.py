"""Dependency, pairing and immutable-checkpoint rejection regressions."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grt_bench.common import digest, read, sha, write
from grt_bench.incremental import (DRIVERS, ORDER, STAGES, impact, check_pair,
    check_runtime, resources_for, validate_proof, verify_prefix, materialize, physical_source)
from open3dbench.stage_manifest import write_stage_manifest, digest_json
from test_grt_frozen_frontend import FrozenFrontendTests

HBT = 'Place-3D/src/hbt/assignment.py'
PLACE = 'Place-3D/src/placement/concurrent_dreamplace.py'
RSZ = 'OpenROAD-3D/openroad/src/rsz/src/'


class ImpactTests(unittest.TestCase):
    def test_actual_diff_union_reversion_and_unknown_dependencies(self):
        before = {HBT: 1, PLACE: 1, 'other': 1}
        self.assertEqual(impact(before, dict(before, **{HBT: 2}))['start'], 'hbt')
        self.assertEqual(impact(before, dict(before, **{HBT: 2, PLACE: 2}))['start'], 'placement')
        # A parent's earlier placement change is unchanged in its HBT child.
        parent = dict(before, **{PLACE: 2})
        child = dict(parent, **{HBT: 2})
        self.assertEqual(impact(parent, child)['changes'], [HBT])
        self.assertEqual(impact(child, before)['start'], 'placement')
        self.assertEqual(impact(before, dict(before, other=2))['start'], 'partition')
        self.assertEqual(impact(before, before)['start'], 'grt')

    def test_modified_upstream_ancestor_invalidates_unproven_dependency_graph(self):
        baseline = {HBT: 1, PLACE: 1}
        parent = dict(baseline, **{PLACE: 2})
        child = dict(parent, **{HBT: 2})
        proof = impact(parent, child, baseline)
        self.assertEqual(proof['changes'], [HBT])
        self.assertEqual(proof['start'], 'placement')
        self.assertEqual(proof['uncertified_upstream_dependencies'], {PLACE: 'placement'})
        parent = dict(baseline, **{HBT: 2})
        self.assertEqual(impact(parent, dict(parent, **{HBT: 3}), baseline)['start'], 'hbt')

    def test_native_first_consumers_are_not_component_labels(self):
        expected = {RSZ+'Resizer.cc': 'partition', RSZ+'RepairDesign.cc': 'partition',
            RSZ+'RepairSetup.cc': 'opt', RSZ+'RepairHold.hh': 'opt',
            'OpenROAD-3D/openroad/src/cts/src/HTreeBuilder.cpp': 'cts',
            'OpenROAD-3D/openroad/src/grt/src/GlobalRouter.cpp': 'partition',
            'OpenROAD-3D/openroad/src/grt/src/fastroute/src/maze.cpp': 'grt'}
        for name, stage in expected.items():
            with self.subTest(name=name):
                self.assertEqual(impact({name: 1}, {name: 2})['start'], stage)

    def test_cpu_suffix_does_not_wait_for_gpu(self):
        front = {'cpu': 8, 'gpu_count': 1}; back = {'cpu': 4, 'gpu_count': 0}
        for stage in ('hbt', 'opt', 'cts', 'grt'):
            self.assertEqual(resources_for({'impact': {'start': stage}}, front, back), back)
        for stage in ('partition', 'placement'):
            self.assertEqual(resources_for({'impact': {'start': stage}}, front, back), front)

    def test_pairing_checks_full_seed_not_only_label(self):
        p = dict(task_digest='task', case='case', seed={'id': 's1', 'placement': 1001}, split='train', endpoint='grt')
        check_pair(p, p)
        for key in p:
            q = copy.deepcopy(p); q[key] = 'different'
            with self.assertRaises(ValueError): check_pair(p, q)
        q = copy.deepcopy(p); q['seed']['placement'] = 1002
        with self.assertRaises(ValueError): check_pair(p, q)

    def test_runtime_must_preserve_task_base(self):
        check_runtime({'candidate': 'a', 'image': 'base', 'native': False}, 'a', 'base')
        with self.assertRaises(ValueError):
            check_runtime({'candidate': 'a', 'image': 'other', 'native': False}, 'a', 'base')
        with self.assertRaises(ValueError):
            check_runtime({'candidate': 'a', 'native': True, 'manifest': {'base_image': 'other'}}, 'a', 'base')


class PhysicalSourceTests(unittest.TestCase):
    def setUp(self):
        from grt_bench.contract import matrix
        from test_grt_matrix import task_fixture
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.original, self.replay = 'a'*64, 'b'*64
        before = task_fixture(); after = copy.deepcopy(before)
        after['evaluator_sha256'] = 'e'*64
        self.old = self.root/'runs'/self.original/'control'
        self.new = self.root/'runs'/self.replay/'control'
        write(self.old/'task.json', before)
        write(self.old/'plan.json', matrix(before, 'c'*64)[0])
        write(self.new/'task.json', after)
        self.plan = matrix(after, 'c'*64)[0]
        self.plan['rescore'] = {'source_job': self.original,
                              'previous_plan_sha256': sha(self.old/'plan.json')}
        write(self.new/'plan.json', self.plan)

    def test_replay_resolves_to_original_physical_files(self):
        self.assertEqual(physical_source(self.root, self.replay),
                         (self.original, self.old.parent/'output'))

    def test_multiple_replays_resolve_without_losing_cell_validation(self):
        from grt_bench.contract import matrix
        third = self.root/'runs'/('d'*64)/'control'
        task = read(self.new/'task.json'); task['id'] = 'second-requalification'
        plan = matrix(task, 'c'*64)[0]
        plan['rescore'] = {'source_job': self.replay,
                          'previous_plan_sha256': sha(self.new/'plan.json')}
        write(third/'task.json', task); write(third/'plan.json', plan)
        self.assertEqual(physical_source(self.root, 'd'*64),
                         (self.original, self.old.parent/'output'))
        plan['seed']['placement'] += 1
        write(third/'plan.json', plan, replace=True)
        with self.assertRaisesRegex(ValueError, 'changed seed'):
            physical_source(self.root, 'd'*64)

    def test_changed_previous_plan_is_rejected(self):
        value = read(self.old/'plan.json'); value['extra'] = 1
        write(self.old/'plan.json', value, replace=True)
        with self.assertRaisesRegex(ValueError, 'plan changed'):
            physical_source(self.root, self.replay)

    def test_replay_cannot_switch_seed_or_escape_run_directory(self):
        value = copy.deepcopy(self.plan); value['seed']['placement'] += 1
        write(self.new/'plan.json', value, replace=True)
        with self.assertRaisesRegex(ValueError, 'changed seed'):
            physical_source(self.root, self.replay)
        value = copy.deepcopy(self.plan); value['rescore']['source_job'] = '../outside'
        write(self.new/'plan.json', value, replace=True)
        with self.assertRaises(ValueError): physical_source(self.root, self.replay)


class PrefixTests(unittest.TestCase):
    def setUp(self):
        f = FrozenFrontendTests(); f.setUp(); self.addCleanup(f.doCleanups)
        self.f = f; self.root = f.root; self.source = f.source
        f.config.update(partition={'seed': 1001}, placement={'seed': 1001}, pdn={})
        self.task = f.task
        base = 'sha256:' + 'a'*64
        self.task['incremental'] = {'version': 1, 'base_image': base, 'drivers': {p: 'b'*64 for p in DRIVERS}}
        files = {p: {'sha256': 'b'*64} for p in DRIVERS}; files[HBT] = {'sha256': 'c'*64}
        parent = {'parent': None, 'files': files}; parent_id = digest(parent)
        self.task['optimization'] = {'baseline': parent_id}
        current = copy.deepcopy(parent); current['parent'] = parent_id; current['files'][HBT]['sha256'] = 'd'*64
        payload = {'candidate': digest(current), 'task_digest': 'task', 'case': 'case',
                   'seed': {'id': 's1', 'partition': 1001, 'placement': 1001, 'io': 1, 'routing': -1},
                   'split': 'train', 'endpoint': 'grt'}
        parent_plan = dict(payload, candidate=parent_id)
        verified = dict(parent_plan, status='verified')
        stages = {s: sha(self.source/STAGES[s]/'stage_manifest.json') for s in ('input', 'floorplan')}
        upstream = stages['floorplan']
        for stage in ('partition', 'placement', 'pdn'):
            d = self.source / STAGES[stage]; d.mkdir()
            write(d/'partition_manifest.json', {'seed': 1001})
            write(d/'stages/02_global_place/stage_manifest.json', {'initialization_seed': 1001})
            write(d/'checkpoint.json', {'test': 'checkpoint'})
            path = write_stage_manifest(d, stage=stage, method='fixture', design='bp_fe', integration='lol', pdk='ng45',
                signature={'implementation': 'b'*64, 'runtime': 'c'*64, 'upstream': upstream,
                           'config': digest_json({k: f.config[k] for k in ('design','type','pdk','threads',stage)})},
                artifacts={'checkpoint': d/'checkpoint.json', 'partition_seed': d/'partition_manifest.json',
                           'placement_seed': d/'stages/02_global_place/stage_manifest.json'},
                provenance={'container_image_id': base})
            stages[stage] = upstream = sha(path)
        self.proof = {'version': 1, 'task_digest': 'task', 'parent': parent_id, 'source_job': 'e'*64,
                      'impact': impact(parent['files'], current['files']), 'stages': stages,
                      'parent_plan_digest': digest(parent_plan), 'parent_verified_digest': digest(verified),
                      'runtime': {'candidate': digest(current), 'image': base, 'native': False},
                      'parent_runtime': {'candidate': parent_id, 'image': base, 'native': False}}
        control = self.root/'control'
        for name, value in [('incremental', self.proof), ('parent-source', parent), ('baseline-source', parent), ('candidate-source', current),
                            ('parent-plan', parent_plan), ('parent-verified', verified)]:
            write(control/(name+'.json'), value)
        self.plan = dict(payload, incremental=str(control/'incremental.json'),
                         incremental_source=str(self.source.parent), output=str(self.root/'new'))
        # Parent mount has a run/ child in production.
        self.source.rename(self.root/'run'); self.source = self.root/'run'
        Path(self.plan['output']).mkdir()
        self.values = (current, parent, parent_plan, verified, parent)
        p = patch('grt_bench.contract.validate', return_value='task'); p.start(); self.addCleanup(p.stop)
        # Stage artifact hashes are real. Physical checkpoint geometry has its
        # own native contract suite; this fixture only observes its invocation.
        p = patch('open3dbench.backend.contracts.validate_checkpoint'); self.checkpoint = p.start(); self.addCleanup(p.stop)

    def test_import_preserves_all_original_bytes_and_is_independent(self):
        materialize(self.plan, self.task, self.f.case, self.f.config)
        dest = Path(self.plan['output'])/'run'
        for p in self.source.rglob('*'):
            if p.is_file():
                q = dest/p.relative_to(self.source)
                self.assertEqual(p.read_bytes(), q.read_bytes())
                self.assertNotEqual(p.stat().st_ino, q.stat().st_ino)
        self.assertGreaterEqual(self.checkpoint.call_count, 4)
        self.assertFalse((dest/'50_hbt').exists())
        with self.assertRaises(FileExistsError): materialize(self.plan, self.task, self.f.case, self.f.config)

    def test_tampered_artifact_manifest_config_and_missing_stage_fail_closed(self):
        for name in ('40_pdn/checkpoint.json', '20_partition/stage_manifest.json'):
            p = self.source/name; old = p.read_bytes(); p.write_bytes(old+b' ')
            with self.assertRaises(ValueError): verify_prefix(self.plan, self.task, self.f.case, self.source, self.f.config)
            p.write_bytes(old)
        with self.assertRaises(ValueError):
            verify_prefix(self.plan, self.task, self.f.case, self.source, dict(self.f.config, threads=8))
        p = self.source/'40_pdn/stage_manifest.json'; p.unlink()
        with self.assertRaises(FileNotFoundError): verify_prefix(self.plan, self.task, self.f.case, self.source, self.f.config)

    def test_forged_stage_parent_pair_and_evidence_are_rejected(self):
        for key, value in [('impact', dict(self.proof['impact'], start='cts')),
                           ('parent', 'f'*64), ('stages', {'input': 'f'*64}),
                           ('parent_verified_digest', 'f'*64), ('version', 2)]:
            bad = copy.deepcopy(self.proof); bad[key] = value
            with self.assertRaises(ValueError): validate_proof(bad, self.task, self.plan, *self.values)
        bad = copy.deepcopy(self.plan); bad['seed']['placement'] = 1002
        with self.assertRaises(ValueError): validate_proof(self.proof, self.task, bad, *self.values)


if __name__ == '__main__':
    unittest.main()
