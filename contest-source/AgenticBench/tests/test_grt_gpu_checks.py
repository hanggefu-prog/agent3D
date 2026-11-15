"""Resource/budget failure modes for diagnostics; no GPU result is fabricated here."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grt_bench.common import read, write
from grt_bench.queue import Queue
from grt_bench.runtime import UncertainExecution
from global_bench.grt_gpu_checks import GPUChecks


class GPUCheckTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);control=root/'control/rounds/0001';control.mkdir(parents=True)
        workspace=root/'workspace';workspace.mkdir()
        self.runtime=SimpleNamespace(root=root,control=control,workspace=workspace,
            config={'host':'test','image':'sha256:'+'a'*64,'agent_cpu':1,'agent_memory_gb':1,
                    'gpu_checks':{'max_calls':2,'total_seconds':20,'max_seconds':15,'storage_gb':1}},
            handler=SimpleNamespace(receipt=lambda:{'parent':'a'*64}),
            command=lambda *args,**kwargs:['docker','run','--detach','--entrypoint','sleep','image','infinity'])
        self.checks=GPUChecks(self.runtime);self.q=self.checks.queue
        self.q.register_host('test',{'cpu':8,'memory_gb':8,'storage_gb':20,'build_slots':1,
                                     'gpus':['0','1'],'gpu_slots_per_device':3})
        self.resources={'cpu':1,'memory_gb':1,'storage_gb':1,'build_slots':0,'gpu_count':1}
        self.args={'request_id':'test1','command':'true','timeout_seconds':10}

    def execute(self,side_effect=None):
        def stop(name,folder):write(Path(folder)/'stop.json',{'terminated':True})
        with patch('global_bench.grt_gpu_checks.admission'), \
             patch('global_bench.grt_gpu_checks.run_container',side_effect=side_effect) as run, \
             patch('global_bench.grt_gpu_checks.stop_and_confirm',side_effect=stop):
            return self.checks.run(self.args),run

    def test_no_checkout_and_excess_timeout_rejected_before_reservation(self):
        self.runtime.handler.receipt=lambda:None
        with self.assertRaisesRegex(ValueError,'checkout'):self.checks.run(self.args)
        self.runtime.handler.receipt=lambda:{}
        with self.assertRaisesRegex(ValueError,'per-command'):self.checks.run(dict(self.args,timeout_seconds=16))
        self.assertEqual(self.q.rows(),[])

    def test_shared_six_slots_and_evaluation_workers_do_not_steal_diagnostics(self):
        jobs=[self.q.submit({'sequence':n},self.resources) for n in range(6)]
        leases=[self.q.claim('test') for _ in jobs]
        result,_=self.execute();self.assertEqual(result['status'],'capacity_unavailable')
        self.assertEqual(read(self.checks.directory/'budget.json')['seconds_charged'],0)
        self.q.finish(jobs[0],leases[0]['token'],{},True)
        self.args['request_id']='test2';result,_=self.execute()
        self.assertEqual(result['status'],'success');self.assertEqual(result['gpus'],leases[0]['gpus'])
        local=self.q.submit({'kind':'local-check','request_id':'pending'},self.resources)
        self.assertIsNone(self.q.claim('test'))
        self.assertIsNotNone(self.q.claim('test',kind='local-check',job_id=local))

    def test_idempotency_budget_and_failure_release(self):
        result,run=self.execute(RuntimeError('diagnostic failed'))
        self.assertEqual(result['status'],'failed');self.assertEqual(self.q.rows()[0]['state'],'failed')
        self.assertEqual(self.checks.run(self.args),result);self.assertEqual(run.call_count,1)
        with self.assertRaisesRegex(ValueError,'different'):self.checks.run(dict(self.args,command='false'))
        self.args['request_id']='test2';self.execute()
        self.args['request_id']='test3'
        with self.assertRaisesRegex(ValueError,'budget exhausted'):self.checks.run(self.args)

    def test_unconfirmed_termination_retains_lease_until_cleanup(self):
        with patch('global_bench.grt_gpu_checks.admission'),patch('global_bench.grt_gpu_checks.run_container'), \
             patch('global_bench.grt_gpu_checks.stop_and_confirm',side_effect=UncertainExecution('unknown')):
            with self.assertRaises(UncertainExecution):self.checks.run(self.args)
        self.assertEqual(self.q.rows()[0]['state'],'running')
        self.assertEqual(read(self.checks.directory/'budget.json')['seconds_charged'],10)
        def stopped(name,folder):write(Path(folder)/'stop.json',{'terminated':True})
        with patch('global_bench.grt_gpu_checks.stop_and_confirm',side_effect=stopped):self.checks.cleanup()
        self.assertEqual(self.q.rows()[0]['state'],'interrupted')


if __name__=='__main__':unittest.main()
