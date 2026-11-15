"""A scheduling migration must qualify fresh cells without relabelling scores."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grt_bench.common import evaluator_identity,write,sha
from grt_bench.contract import validate
from scripts.benchmark_grt_phased import prepare_contract
from test_grt_matrix import task_fixture

class PhasedTrialTests(unittest.TestCase):
    def test_failed_canary_cannot_touch_production_queue_or_services(self):
        from scripts.benchmark_grt_phased import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);framework=root/'framework';task=task_fixture()
            write(root/'control/tasks'/(validate(task)+'.json'),task)
            write(root/'qualified.json',{'task':validate(task)})
            write(root/'canary/status.json',{'status':'failed'})
            write(root/'plan.json',dict(root=str(root),output=str(root/'trial'),framework=str(framework),
                previous_framework=str(root/'previous'),worker_count=12,
                capacity={'gpu_slots_per_device':3,'gpus':['0','1']},qualification=str(root/'qualified.json'),
                canary=str(root/'canary')))
            with patch('scripts.benchmark_grt_phased.__file__',str(framework/'scripts/benchmark_grt_phased.py')), \
                    patch('scripts.benchmark_grt_phased.prepare_contract',return_value=(task,['flow.py'])), \
                    patch('scripts.benchmark_grt_phased.Queue') as queue, \
                    patch('scripts.benchmark_grt_phased.launch') as launch, \
                    patch('scripts.benchmark_grt_phased.os.kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'canary failed'):run(root/'plan.json')
                queue.assert_not_called();launch.assert_not_called();kill.assert_not_called()

    def test_only_scheduling_modules_can_change_and_reference_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';new=root/'new';reference=root/'reference'
            for framework in (old,new):
                (framework/'grt_bench').mkdir(parents=True)
                (framework/'scripts').mkdir()
                for name in ('flow','worker','queue','service','verify','endpoint'):
                    (framework/'grt_bench'/(name+'.py')).write_text('# original\n')
                (framework/'scripts/workspace_cts_thermal.py').write_text('# original\n')
            task=task_fixture();task['evaluator_sha256']=evaluator_identity(old/'grt_bench')
            write(reference/'outcomes.json',{'fixture':True})
            receipt=dict(status='qualified',task=validate(task),evaluator_sha256=task['evaluator_sha256'],
                reference=str(reference),artifacts={'outcomes.json':sha(reference/'outcomes.json')})
            for name in ('flow','worker','queue','service'):
                (new/'grt_bench'/(name+'.py')).write_text('# scheduling change\n')
            updated,changes=prepare_contract(receipt,old,new,task)
            self.assertEqual(dict(updated,evaluator_sha256=task['evaluator_sha256']),task)
            self.assertNotEqual(validate(updated),validate(task))
            self.assertEqual(set(changes),{'flow.py','worker.py','queue.py','service.py'})
            (new/'grt_bench/verify.py').write_text('# changed verifier\n')
            with self.assertRaisesRegex(ValueError,'Scoring/verification'): prepare_contract(receipt,old,new,task)
            (new/'grt_bench/verify.py').write_text('# original\n')
            (reference/'outcomes.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'reference changed'): prepare_contract(receipt,old,new,task)
