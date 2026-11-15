"""Frontend permissions, protected contracts and candidate runtime binding."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.algorithm_scope import (DREAMPLACE, function_scope, placement_python,
    verify_functions, python_mounts, verify_runtime)
from grt_bench.common import digest, sha, write
from grt_bench.incremental import impact
from grt_bench.optimization import belongs


class FrontendScopeTests(unittest.TestCase):
    def test_core_algorithms_open_but_evaluation_io_and_native_ops_stay_protected(self):
        for name in ['NonLinearPlace.py', 'PlaceObj.py', 'BasicPlace.py',
                     'NesterovAcceleratedGradientOptimizer.py', 'HBTConcurrentPlace.py',
                     'ops/electric_potential/electric_potential.py']:
            self.assertTrue(belongs(DREAMPLACE + name, 'placement'))
            self.assertEqual(impact({DREAMPLACE + name: 1}, {DREAMPLACE + name: 2})['start'], 'placement')
        for name in ['PlaceDB.py', 'EvalMetrics.py', 'Placer.py', 'params.json',
                     'ops/place_io/place_io.py', 'ops/electric_potential/src/electric_potential_cuda.cu']:
            self.assertFalse(belongs(DREAMPLACE + name, 'placement'))
        self.assertTrue(belongs('Place-3D/src/partition/lol/core.py', 'partition'))
        self.assertEqual(impact({DREAMPLACE+'concurrent_lol.py': 1}, {DREAMPLACE+'concurrent_lol.py': 2})['start'], 'partition')

    def test_mixed_module_allows_algorithm_body_only(self):
        with tempfile.TemporaryDirectory() as temp:
            before, after = Path(temp)/'before', Path(temp)/'after'
            name = 'Place-3D/src/partition/lol/core.py'
            original = 'import math\ndef build_unique_nets(x):\n return x\ndef move_delta_cut(x):\n return 1\ndef compute_metrics(x):\n return 2\n'
            for root in (before, after):
                p=root/name; p.parent.mkdir(parents=True); p.write_text(original)
            policy = {'function_scope': function_scope([name])}
            (after/name).write_text(original.replace('return 1', 'return 3'))
            verify_functions(before, after, policy)
            for changed in [original.replace('return 2','return 0'), original.replace('import math','import os'),
                            original.replace('move_delta_cut(x)', 'move_delta_cut(x, seed=5)')]:
                (after/name).write_text(changed)
                with self.assertRaises(ValueError): verify_functions(before, after, policy)

    def test_runtime_bytes_and_complete_module_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); name=DREAMPLACE+'PlaceObj.py'; p=root/name
            p.parent.mkdir(parents=True); p.write_text('candidate_algorithm = True\n')
            mounts=python_mounts(root,[name]); self.assertEqual(len(mounts),1)
            target=mounts[0][1]
            receipt={'candidate':'a'*64,'image':'sha256:'+'b'*64,
                     'modules':{target:{'source':name,'sha256':sha(p)}}}
            path=root/'receipt.json'; write(path,receipt)
            plan={'candidate':'a'*64,'source':str(root),'python_runtime':str(path),
                  'python_runtime_digest':digest(receipt)}
            real_sha=sha
            with patch('grt_bench.common.sha',side_effect=lambda q: real_sha(p) if str(q)==target else real_sha(q)):
                verify_runtime(plan)
                p.write_text('stale_algorithm = True\n')
                with self.assertRaisesRegex(ValueError,'differs'): verify_runtime(plan)
            empty=copy.deepcopy(receipt);empty['modules']={};write(path,empty,replace=True)
            plan['python_runtime_digest']=digest(empty)
            with self.assertRaisesRegex(ValueError,'Incomplete'):verify_runtime(plan)


if __name__ == '__main__': unittest.main()
