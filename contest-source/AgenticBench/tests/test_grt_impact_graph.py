"""Derived stage-dependency graph (proof v2) on synthetic and real source trees."""
import copy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench import incremental
from grt_bench.common import digest
from grt_bench.incremental import (DRIVERS, ORDER, check_dependencies, dependency_graph, derive_dependencies,
    derived_stage, impact, proof_names, table_stage, validate_proof)

REPO = Path(__file__).resolve().parents[2]
HBT = 'Place-3D/src/hbt/assignment.py'
LEGALIZE = 'Place-3D/src/hbt/legalize.py'
SPLIT = 'Place-3D/src/partition/split.py'
OTHER = 'Place-3D/src/partition/other_integration.py'
REPAIR = 'OpenROAD-3D/openroad/src/rsz/src/RepairSetup.cc'
CTS = 'OpenROAD-3D/openroad/src/cts/src/HTreeBuilder.cpp'
UNUSED = 'Place-3D/src/unused.py'

DRIVER = '''
import json


class FlowRunner:
    def _run_input(self):
        return json

    def _run_floorplan(self):
        return 'scripts/floorplan.tcl'

    def _run_partition(self):
        from src.partition import split
        if self.config["type"] != "lol":
            from src.partition import other_integration
        return split, "scripts/partition.tcl"

    def _run_placement(self):
        return self._helper()

    def _helper(self):
        return None

    def _run_pdn(self):
        return None

    def _run_hbt(self):
        from src.hbt import assignment, legalize
        return assignment, legalize

    def _run_opt(self):
        return "scripts/opt.tcl"

    def _run_cts(self):
        return "scripts/cts.tcl"

    def global_route(self):
        return None

    def _run_routing(self):
        return None

    def _run_report(self):
        return None

    def _run_thermal(self):
        return None
'''

FILES = {
    DRIVERS[0]: DRIVER,
    DRIVERS[1]: 'VALUE = 1\n',
    'Place-3D/src/__init__.py': '',
    'Place-3D/src/partition/__init__.py': '',
    'Place-3D/src/hbt/__init__.py': '',
    SPLIT: 'from src.hbt.assignment import helper\n',
    OTHER: 'VALUE = 2\n',
    HBT: 'def helper():\n    return 1\n',
    LEGALIZE: 'VALUE = 3\n',
    UNUSED: 'VALUE = 4\n',
    'scripts/floorplan.tcl': 'repair_timing -setup\n',
    'scripts/partition.tcl': 'estimate_parasitics -placement\n',
    'scripts/opt.tcl': 'repair_timing -setup\n# clock_tree_synthesis\n',
    'scripts/cts.tcl': 'clock_tree_synthesis -root_buf BUF\n',
}


def make_tree(root, files=FILES):
    for name, text in files.items():
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Path(root)


class GraphTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.tree = make_tree(self.root / 'tree')

    def test_stages_follow_driver_closure_and_prune_other_integrations(self):
        graph = dependency_graph(self.tree)
        self.assertEqual(graph['stages'][SPLIT], 'partition')
        self.assertEqual(graph['stages'][HBT], 'partition')  # Imported by partition code.
        self.assertEqual(graph['stages'][LEGALIZE], 'hbt')
        self.assertNotIn(OTHER, graph['stages'])
        self.assertNotIn(UNUSED, graph['stages'])
        self.assertNotIn('scripts/floorplan.tcl', graph['stages'])  # Frozen input stage never replays.
        self.assertEqual(graph['native_consumers'], {'repair_timing': 'opt', 'clock_tree_synthesis': 'cts'})
        self.assertEqual(derived_stage(graph, REPAIR), 'opt')
        self.assertEqual(derived_stage(graph, CTS), 'cts')
        self.assertEqual(derived_stage(graph, UNUSED), 'grt')

    def test_native_consumer_moves_earlier_with_driver_tcl(self):
        (self.tree / 'scripts/partition.tcl').write_text('estimate_parasitics\n[repair_timing -hold]\n')
        graph = dependency_graph(self.tree)
        self.assertEqual(graph['native_consumers']['repair_timing'], 'partition')
        self.assertEqual(derived_stage(graph, REPAIR), 'partition')

    def test_impact_is_min_of_table_and_derived(self):
        derived = derive_dependencies([dependency_graph(self.tree)], [HBT, LEGALIZE, UNUSED])
        self.assertEqual(table_stage(HBT), 'hbt')
        self.assertEqual(impact({HBT: 1}, {HBT: 2})['start'], 'hbt')
        value = impact({HBT: 1}, {HBT: 2}, dependencies=derived)
        self.assertEqual((value['start'], value['table'], value['derived']),
                         ('partition', {HBT: 'hbt'}, {HBT: 'partition'}))
        # A derived later stage never moves an unknown file's restart later.
        self.assertEqual(impact({LEGALIZE: 1}, {LEGALIZE: 2}, dependencies=derived)['start'], 'partition')
        self.assertEqual(impact({UNUSED: 1}, {UNUSED: 2}, dependencies=derived)['affected'], {UNUSED: 'partition'})
        with self.assertRaisesRegex(ValueError, 'missing'):
            impact({SPLIT: 1}, {SPLIT: 2}, dependencies=derived)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            impact({HBT: 1}, {HBT: 2}, dependencies={HBT: 'floorplan'})

    def test_graph_is_deterministic_cached_and_invalidated(self):
        first = dependency_graph(self.tree)
        self.assertIs(dependency_graph(self.tree), first)
        copied = make_tree(self.root / 'copy')
        self.assertEqual(dependency_graph(copied)['digest'], first['digest'])
        (self.tree / SPLIT).write_text('VALUE = 5\n')
        changed = dependency_graph(self.tree)
        self.assertNotEqual(changed['digest'], first['digest'])
        self.assertEqual(changed['stages'][HBT], 'hbt')

    def test_missing_stage_method_fails_closed(self):
        (self.tree / DRIVERS[0]).write_text(DRIVER.replace('def _run_cts', 'def _run_ctx'))
        with self.assertRaisesRegex(ValueError, 'stage methods'):
            dependency_graph(self.tree)

    def test_real_tree_partition_closure_never_starts_later_than_partition(self):
        graph = dependency_graph(REPO)
        self.assertEqual(graph, dependency_graph(REPO))
        for name in graph['closures']['partition']:
            with self.subTest(name=name):
                self.assertEqual(derived_stage(graph, name), 'partition')
        self.assertEqual(derived_stage(graph, HBT), 'hbt')
        self.assertEqual(set(graph['native_consumers']), {'repair_timing', 'clock_tree_synthesis', 'global_route'})


class ProofTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.trees = {'baseline': make_tree(root / 'baseline'), 'parent': make_tree(root / 'parent')}
        self.trees['current'] = make_tree(root / 'current', dict(FILES, **{HBT: 'def helper():\n    return 2\n'}))
        files = {p: {'sha256': 'b' * 64} for p in DRIVERS}
        files[HBT] = {'sha256': 'c' * 64}
        self.baseline = {'parent': None, 'files': files}
        self.parent = copy.deepcopy(self.baseline)
        self.current = copy.deepcopy(self.parent)
        self.current['parent'] = digest(self.parent)
        self.current['files'][HBT] = {'sha256': 'd' * 64}
        base = 'sha256:' + 'a' * 64
        self.task = {'incremental': {'version': 1, 'base_image': base, 'drivers': {p: 'b' * 64 for p in DRIVERS}},
                     'frontend_inputs': {}, 'optimization': {'baseline': digest(self.baseline)}}
        self.payload = {'candidate': digest(self.current), 'task_digest': 'task', 'case': 'case',
                        'seed': {'id': 's1'}, 'split': 'train', 'endpoint': 'grt'}
        self.parent_plan = dict(self.payload, candidate=digest(self.parent))
        self.verified = dict(self.parent_plan, status='verified')
        self.base = base
        p = patch('grt_bench.contract.validate', return_value='task'); p.start(); self.addCleanup(p.stop)

    def proof(self, version):
        deps = None
        value = {'version': version, 'task_digest': 'task', 'parent': digest(self.parent),
                 'parent_plan_digest': digest(self.parent_plan), 'parent_verified_digest': digest(self.verified),
                 'runtime': {'candidate': digest(self.current), 'image': self.base, 'native': False},
                 'parent_runtime': {'candidate': digest(self.parent), 'image': self.base, 'native': False}}
        if version == 2:
            graphs = {r: dependency_graph(t) for r, t in self.trees.items()}
            names = proof_names(self.parent['files'], self.current['files'], self.baseline['files'])
            deps = derive_dependencies([graphs[r] for r in sorted(graphs)], names)
            value.update(dependencies=deps, graphs={r: g['digest'] for r, g in graphs.items()})
        value['impact'] = impact(self.parent['files'], self.current['files'], self.baseline['files'], dependencies=deps)
        value['stages'] = {s: 'e' * 64 for s in ORDER[:ORDER.index(value['impact']['start'])]}
        return value

    def validate(self, proof, trees=None):
        return validate_proof(proof, self.task, self.payload, self.current, self.parent,
                              self.parent_plan, self.verified, self.baseline, trees=trees)

    def test_version_one_remains_verifiable_and_carries_no_graph(self):
        proof = self.proof(1)
        self.assertEqual(self.validate(proof)['start'], 'hbt')
        bad = dict(proof, dependencies={HBT: 'partition'})
        with self.assertRaisesRegex(ValueError, 'Version 1'):
            self.validate(bad)

    def test_version_two_binds_derived_map_to_source_trees(self):
        proof = self.proof(2)
        self.assertEqual(proof['dependencies'], {HBT: 'partition'})
        self.assertEqual(self.validate(proof, self.trees)['start'], 'partition')
        self.assertEqual(self.validate(proof, {'current': self.trees['current']})['start'], 'partition')
        self.assertEqual(self.validate(proof)['start'], 'partition')

    def test_version_two_rejects_forgery(self):
        proof = self.proof(2)
        later = copy.deepcopy(proof); later['dependencies'][HBT] = 'hbt'
        later['impact'] = impact(self.parent['files'], self.current['files'], self.baseline['files'],
                                 dependencies=later['dependencies'])
        later['stages'] = {s: 'e' * 64 for s in ORDER[:ORDER.index('hbt')]}
        for trees in (self.trees, {'current': self.trees['current']}):
            with self.subTest(trees=sorted(trees)), self.assertRaises(ValueError):
                self.validate(later, trees)
        v1_impact = dict(proof, impact=self.proof(1)['impact'])
        with self.assertRaisesRegex(ValueError, 'affected stage'):
            self.validate(v1_impact)
        for key, value in (('graphs', {'current': 'f' * 64}), ('dependencies', {}),
                           ('graphs', dict(proof['graphs'], current='F' * 64))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(dict(proof, **{key: value}))
        stale = dict(proof, graphs=dict(proof['graphs'], parent='f' * 64))
        with self.assertRaisesRegex(ValueError, 'graph differs'):
            self.validate(stale, self.trees)
        with self.assertRaisesRegex(ValueError, 'role'):
            check_dependencies(proof, self.current, self.parent, self.baseline, {'other': self.trees['current']})

    def test_prepare_defaults_to_version_two(self):
        import inspect
        self.assertEqual(inspect.signature(incremental.prepare_matrix).parameters['version'].default, 2)
        self.assertEqual(incremental.PROOF_VERSIONS, (1, 2))


if __name__ == '__main__':
    unittest.main()
