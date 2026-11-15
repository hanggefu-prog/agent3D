"""Fixed-input adoption must preserve provenance and reject physical drift."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grt_bench.common import sha, write
from grt_bench.frozen_frontend import STAGES, materialize, verify, validate_contract
from grt_bench.rescore import validate_transition
from grt_bench.contract import matrix
from open3dbench.stage_manifest import write_stage_manifest, digest_json
from test_grt_matrix import task_fixture


class FrozenFrontendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.source = self.root / 'source'
        self.case = {'id': 'case', 'design': 'bp_fe', 'pdk': 'ng45'}
        self.config = {'design': 'bp_fe', 'pdk': 'ng45', 'type': 'lol', 'threads': 4,
                       'input': {'mode': 'checkpoint'}, 'floorplan': {}, 'pins': {'seed': 1}}
        image = 'sha256:' + 'a' * 64
        entry = {'stages': {}, 'producer_image': image}
        for stage, directory in STAGES.items():
            d = self.source / directory; d.mkdir(parents=True)
            write(d / 'io_metrics.json', {'seed': 1})
            (d / 'design.def').write_text('physical checkpoint\n')
            keys = ('input',) if stage == 'input' else ('floorplan', 'pins')
            signature = {'implementation': 'b'*64, 'runtime': 'c'*64,
                'upstream': 'none' if stage == 'input' else entry['stages']['input'],
                'config': digest_json({k: self.config[k] for k in ('design', 'pdk', 'type', 'threads', *keys)})}
            path = write_stage_manifest(d, stage=stage, method='fixture', design='bp_fe',
                integration='lol', pdk='ng45', signature=signature,
                artifacts={'def': d/'design.def', 'io': d/'io_metrics.json'},
                provenance={'container_image_id': image})
            entry['stages'][stage] = sha(path)
        self.task = {'cases': [self.case], 'frontend_inputs': {'boundary': 'partition', 'cases': {'case': entry}}}

    def test_exact_import_keeps_original_signatures_and_physical_bytes(self):
        destination = self.root / 'run'
        materialize(self.task, self.case, self.source, destination, self.config)
        for directory in STAGES.values():
            for name in ('design.def', 'stage_manifest.json'):
                self.assertEqual((destination/directory/name).read_bytes(), (self.source/directory/name).read_bytes())
        self.assertEqual(json.loads((destination/'run.json').read_text())['frontend_prefix']['kind'], 'task-frozen-input')
        verify(self.task, self.case, destination, self.config)
        with self.assertRaises(FileExistsError): materialize(self.task, self.case, self.source, destination, self.config)

    def test_seed_changes_after_partition_do_not_change_the_frozen_input(self):
        for seed in (1001, 1002, 1003):
            config = dict(self.config, partition={'seed': seed}, placement={'seed': seed})
            verify(self.task, self.case, self.source, config)

    def test_changed_artifact_or_manifest_is_rejected(self):
        f = self.source/'10_floorplan/design.def'; f.write_text('different pins\n')
        with self.assertRaises(ValueError): verify(self.task, self.case, self.source)
        f.write_text('physical checkpoint\n')
        p = self.source/'10_floorplan/stage_manifest.json'; p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'manifest differs'): verify(self.task, self.case, self.source)

    def test_wrong_case_runtime_config_and_incomplete_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, 'design/platform'): verify(self.task, dict(self.case,pdk='asap7'), self.source)
        task=copy.deepcopy(self.task);task['frontend_inputs']['cases']['case']['producer_image']='sha256:'+'d'*64
        with self.assertRaisesRegex(ValueError, 'producer image'): verify(task,self.case,self.source)
        with self.assertRaisesRegex(ValueError, 'configuration'): verify(self.task,self.case,self.source,dict(self.config,threads=8))
        task=copy.deepcopy(self.task);task['frontend_inputs']['cases']={}
        with self.assertRaises(ValueError): validate_contract(task)

    def test_qualification_adds_only_fixed_inputs_to_original_baseline(self):
        # Use the fully validated optimization fixture rather than inventing a scope.
        from test_grt_optimization import OptimizationTests
        fixture=OptimizationTests();fixture.setUp();self.addCleanup(fixture.doCleanups);fixture.bootstrap()
        old=fixture.task;new=copy.deepcopy(old)
        entry=next(iter(self.task['frontend_inputs']['cases'].values()))
        new['frontend_inputs']={'boundary':'partition','cases':{c['id']:entry for c in old['cases']}}
        new['evaluator_sha256']='e'*64
        before=matrix(old,fixture.baseline)[0];after=matrix(new,fixture.baseline)[0]
        validate_transition(new,old,after,before)
        with self.assertRaisesRegex(ValueError,'original baseline'):
            validate_transition(new,old,dict(after,candidate='f'*64),dict(before,candidate='f'*64))


if __name__ == '__main__':
    unittest.main()
