"""Shared GPU slot admission and safe mid-run resource release."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.queue import Queue
from grt_bench.worker import run_phased
from grt_bench.common import write
from grt_bench.runtime import UncertainExecution


class SharedGPUQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.q = Queue(Path(self.tmp.name) / 'queue.sqlite')
        self.capacity = dict(cpu=64,memory_gb=192,storage_gb=600,gpus=['0','1'],build_slots=1,gpu_slots_per_device=3)
        self.front = dict(cpu=8,memory_gb=24,storage_gb=40,gpu_count=1,build_slots=0)
        self.back = dict(self.front,cpu=4,memory_gb=16,gpu_count=0)
        self.q.register_host('rafa1',self.capacity)
        for i in range(8): self.q.submit({'cell':i},self.front)

    def proof(self,lease):
        return dict(container='grt-'+lease['id'][:32]+'-frontend',terminated=True,checkpoint_sha256='a'*64)

    def test_six_parallel_claims_balance_three_per_gpu_and_release_admits_next(self):
        with ThreadPoolExecutor(max_workers=12) as pool:
            leases=[r for r in pool.map(lambda _:self.q.claim('rafa1'),range(12)) if r]
        self.assertEqual(len(leases),6)
        self.assertEqual(Counter(r['gpus'][0] for r in leases),{'0':3,'1':3})
        first=leases[0]
        result=self.q.release(first['id'],first['token'],self.back,self.proof(first))
        self.assertEqual(result['gpus'],[])
        row=next(r for r in self.q.rows() if r['id']==first['id'])
        self.assertEqual(json.loads(row['resources']),self.front)
        self.assertEqual(json.loads(row['allocation']),self.back)
        reopened=Queue(self.q.path)
        following=reopened.claim('rafa1')
        self.assertEqual(following['gpus'],first['gpus'])
        self.assertIsNone(reopened.claim('rafa1'))
        # Duplicate acknowledgements do not release somebody else's GPU slot.
        reopened.release(first['id'],first['token'],self.back,self.proof(first))
        self.assertIsNone(reopened.claim('rafa1'))
        reopened.finish(first['id'],first['token'],{'ok':True},True)

    def test_failed_proof_growth_stale_and_cancelled_lease_do_not_release(self):
        lease=self.q.claim('rafa1')
        variants=[(self.back,dict(self.proof(lease),terminated=False)),
                  (self.back,dict(self.proof(lease),container='wrong')),
                  (self.back,dict(self.proof(lease),checkpoint_sha256='wrong')),
                  (dict(self.back,memory_gb=32),self.proof(lease)),
                  (dict(self.back,gpu_count=1),self.proof(lease))]
        for resources,evidence in variants:
            with self.subTest(resources=resources,evidence=evidence),self.assertRaises(ValueError):
                self.q.release(lease['id'],lease['token'],resources,evidence)
        with self.assertRaises(ValueError):self.q.release(lease['id'],'wrong',self.back,self.proof(lease))
        self.q.cancel(lease['id'])
        with self.assertRaises(ValueError):self.q.release(lease['id'],lease['token'],self.back,self.proof(lease))
        row=next(r for r in self.q.rows() if r['id']==lease['id'])
        self.assertEqual(json.loads(row['allocation']),self.front)

    def test_expiry_retains_shared_slots_until_reconciled(self):
        leases=[self.q.claim('rafa1') for _ in range(6)]
        with patch('grt_bench.queue.time.time',return_value=10**12):
            self.assertIsNone(self.q.claim('rafa1'))
            with self.assertRaises(ValueError):self.q.release(leases[0]['id'],leases[0]['token'],self.back,self.proof(leases[0]))
        self.assertEqual({r['state'] for r in self.q.rows() if r['started']},{'uncertain'})
        self.q.reconcile(leases[0]['id'],dict(job=leases[0]['id'],terminated=True,evidence_sha256='a'*64))
        self.assertEqual(self.q.claim('rafa1')['gpus'],leases[0]['gpus'])

    def test_memory_budget_still_limits_gpu_sharing_and_capacity_cannot_change_live(self):
        with tempfile.TemporaryDirectory() as d:
            q=Queue(Path(d)/'q.sqlite')
            q.register_host('small',dict(self.capacity,memory_gb=48))
            for i in range(6):q.submit({'cell':i},self.front)
            self.assertIsNotNone(q.claim('small'));self.assertIsNotNone(q.claim('small'))
            self.assertIsNone(q.claim('small'))
            with self.assertRaises(ValueError):q.register_host('small',self.capacity)


class PhaseReleaseTests(unittest.TestCase):
    def test_gpu_container_must_exit_before_release_and_cpu_backend_has_no_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write(root/'output/run/30_placement/checkpoint.json',{'fixture':True})
            front=dict(cpu=8,memory_gb=24,storage_gb=40,gpu_count=1,build_slots=0)
            back=dict(front,cpu=4,memory_gb=16,gpu_count=0)
            lease={'id':'a'*64,'resources':front,'gpus':['1']}
            calls=[]
            def execute(command,output,name,**kwargs):
                calls.append(('execute',name,command))
                return {'exit_code':0,'wall_seconds':1}
            def stopped(name,path):calls.append(('terminated',name))
            def release(resources,evidence):
                calls.append(('release',evidence))
                return {'resources':resources,'gpus':[]}
            with patch('grt_bench.worker.run_container',side_effect=execute),patch('grt_bench.worker.stop_and_confirm',side_effect=stopped),patch('grt_bench.worker.admission'):
                result=run_phased(lease,'sha256:'+'b'*64,[],root,back,lambda:None,release)
            self.assertEqual([r[0] for r in calls],['execute','terminated','release','execute'])
            self.assertIn('--gpus',calls[0][2]);self.assertNotIn('--gpus',calls[3][2])
            self.assertEqual(result['execution_mode'],'gpu-frontend-cpu-backend')
            with patch('grt_bench.worker.run_container',return_value={'exit_code':0}),patch('grt_bench.worker.stop_and_confirm',side_effect=UncertainExecution('unknown')),patch('grt_bench.worker.admission'),patch('grt_bench.worker.write'),patch('grt_bench.worker.sha',return_value='a'*64):
                calls.clear()
                with self.assertRaises(UncertainExecution):run_phased(lease,'sha256:'+'b'*64,[],root,back,lambda:None,release)
                self.assertEqual(calls,[])
