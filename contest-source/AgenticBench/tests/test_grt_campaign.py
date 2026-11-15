"""Controller state/bridge integration using synthetic receipts, never EDA evidence."""
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grt_bench.common import digest,read,write
from grt_bench.contract import OBJECTIVES,validate
from global_bench.grt_campaign import Campaign
from global_bench.grt_round_tools import RoundRuntime,RoundTools
import test_grt_optimization as fixtures


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.OptimizationTests();self.f.setUp();self.addCleanup(self.f.doCleanups);self.f.bootstrap()
        f=self.f
        fw=Path(__file__).resolve().parents[1]
        self.config={'root':str(f.root.resolve()),'task':str(f.root/'control/tasks'/(validate(f.task)+'.json')),
            'framework':str(fw),'host':'fixture','capacity':{'gpus':['0','1'],'gpu_slots_per_device':3,
            'cpu':64,'memory_gb':160,'storage_gb':320,'build_slots':1},'resources':f.resources,
            'worker_count':12,'round_limit':2,'evaluation_limit':2,'codex_binary':'codex',
            'model':'gpt-6.1-sol','agent_cpu':2,'agent_memory_gb':4,'agent_timeout_seconds':60,
            'environment':{'root':str(f.root.resolve()),'framework':str(fw),'cases':{},'base_image':'sha256:'+'a'*64,
                           'phase_resources':{'backend':dict(f.resources,gpu_count=0)}}}
        self.path=f.root/'campaign-config.json';write(self.path,self.config)
        self.c=Campaign(self.path)
        auth=patch.object(Campaign,'preflight_codex');auth.start();self.addCleanup(auth.stop)

    def fake_turn(self,**kwargs):
        row=read(self.c.state_path)['rounds'][-1]
        runtime=RoundRuntime(read(Path(row['control'])/'tool-config.json'))
        proposal={'parent':self.f.baseline,'reason':'baseline','hypothesis':'change value',
            'evidence':['baseline measurements'],'files_to_modify':[fixtures.PARTITION],
            'expected_tradeoffs':dict.fromkeys((*OBJECTIVES,'power_w'),'uncertain')}
        receipt=runtime.handler.checkout(proposal)
        (runtime.workspace/fixtures.PARTITION).write_text('value = 2\n')
        response={**proposal,'task_digest':validate(self.f.task),'round':row['round'],
            'selection_id':receipt['selection_id'],'status':'ready','summary':'Changed value',
            'change_summary':{'from_reference':'Changed value to 2.','from_parent':'Changed value to 2.'},
            'changed_files':[fixtures.PARTITION],'checks':[],'risks':[],'next_round_hint':'Inspect timing'}
        return {'success':True,'thread_id':'00000000-0000-0000-0000-000000000001','final_message':json.dumps(response)}

    def fake_matrix(self,config,task,candidate,folder):
        self.f.receipts(candidate,[9.,9.,9.,9.,10.])
        return self.f.archive.ingest(candidate)

    def test_prepare_is_idempotent_and_review_hash_required(self):
        r=self.c.prepare();self.assertEqual(r,self.c.prepare())
        with patch('global_bench.grt_campaign.run_workspace_turn') as call:
            with self.assertRaisesRegex(ValueError,'reviewed'):self.c.run('0'*64)
            call.assert_not_called()
        self.assertEqual(read(self.c.state_path)['evaluations_used'],0)

    def test_one_turn_edit_verify_archive_and_next_round_resume(self):
        row=self.c.prepare()
        with patch.object(RoundRuntime,'start'),patch.object(RoundRuntime,'stop'), \
             patch('global_bench.grt_campaign.run_workspace_turn',side_effect=self.fake_turn) as call, \
             patch('global_bench.grt_campaign.evaluate_matrix',side_effect=self.fake_matrix):
            result=self.c.run(row['prompt_sha256'])
        self.assertEqual(result['status'],'verified');self.assertTrue(result['archive_admission']['non_dominated'])
        self.assertEqual(call.call_count,1)
        state=read(self.c.state_path);self.assertEqual(state['evaluations_used'],1)
        self.assertEqual(state['thread_id'],'00000000-0000-0000-0000-000000000001')
        with self.assertRaises(ValueError):self.c.run(row['prompt_sha256'])
        next_row=self.c.prepare();self.assertEqual(next_row['round'],2)
        self.assertEqual(len(read(Path(next_row['control'])/'prompt/bundle.json')['context']['recent_history']),1)

    def test_partial_or_failed_matrix_cannot_enter_archive(self):
        row=self.c.prepare()
        def partial(config,task,candidate,folder):
            self.f.receipts(candidate,skip_last=True)
            return self.f.archive.ingest(candidate)
        with patch.object(RoundRuntime,'start'),patch.object(RoundRuntime,'stop'), \
             patch('global_bench.grt_campaign.run_workspace_turn',side_effect=self.fake_turn), \
             patch('global_bench.grt_campaign.evaluate_matrix',side_effect=partial):
            with self.assertRaisesRegex(ValueError,'incomplete'):self.c.run(row['prompt_sha256'])
        state=read(self.c.state_path);self.assertEqual(state['evaluations_used'],1)
        self.assertEqual(state['rounds'][-1]['status'],'reconciliation_required')
        self.assertEqual(len(self.f.archive.show()['evaluated']),1)

    def test_uncertain_call_never_invoked_again(self):
        row=self.c.prepare()
        with patch.object(RoundRuntime,'start'),patch.object(RoundRuntime,'stop'), \
             patch('global_bench.grt_campaign.run_workspace_turn',side_effect=RuntimeError('crash')):
            with self.assertRaisesRegex(RuntimeError,'crash'):self.c.run(row['prompt_sha256'])
        self.assertEqual(read(self.c.state_path)['rounds'][-1]['status'],'generating')
        with self.assertRaises(ValueError):self.c.run(row['prompt_sha256'])
        with self.assertRaises(ValueError):self.c.prepare()

    def test_auth_and_implementation_checks_do_not_spend_the_round(self):
        row=self.c.prepare()
        with patch.object(self.c,'preflight_codex',side_effect=ValueError('not authenticated')):
            with self.assertRaisesRegex(ValueError,'authenticated'):self.c.run(row['prompt_sha256'])
        self.assertEqual(read(self.c.state_path)['rounds'][-1]['model_invocations'],0)
        with patch.object(self.c,'implementation_digest',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'implementation changed'):self.c.run(row['prompt_sha256'])
        self.assertEqual(read(self.c.state_path)['rounds'][-1]['status'],'prepared')

    def test_prompt_revision_preserves_evidence_and_requires_fresh_review(self):
        first=self.c.prepare()
        control=Path(first['control'])
        original=(control/'prompt/prompt.md').read_bytes()
        task=Path(self.config['task']).read_bytes()
        archive=self.f.archive.show()
        from global_bench import grt_round_prompt as prompts
        templates=self.f.root/'revised-templates';templates.mkdir()
        for name in ('background.md','round.md'):
            shutil.copyfile(prompts.TEMPLATES/name,templates/name)
        with (templates/'round.md').open('a') as stream:
            stream.write('\nUse the listed cases for local tests.\n')
        template_patch=patch.object(prompts,'TEMPLATES',templates)
        template_patch.start();self.addCleanup(template_patch.stop)
        brief=self.f.task['optimization']['prompt']+'\nUse the listed cases for local tests.\n'
        revised=self.c.revise_prompt(brief)
        self.assertNotEqual(first['prompt_sha256'],revised['prompt_sha256'])
        self.assertEqual((control/'prompt-revisions/0001/prompt/prompt.md').read_bytes(),original)
        self.assertEqual(Path(self.config['task']).read_bytes(),task)
        self.assertEqual(self.f.archive.show(),archive)
        state=read(self.c.state_path)
        self.assertEqual(state['evaluations_used'],0)
        self.assertEqual((revised['model_invocations'],revised['matrices']),(0,0))
        runtime=RoundRuntime(read(control/'tool-config.json'))
        self.assertNotIn('research_brief',runtime.bundle['context'])
        self.assertEqual(runtime.bundle['research_brief_provenance']['audit_digest'],digest(brief))
        with patch('global_bench.grt_campaign.run_workspace_turn') as call:
            with self.assertRaisesRegex(ValueError,'reviewed'):self.c.run(first['prompt_sha256'])
            call.assert_not_called()
        with patch.object(RoundRuntime,'start'),patch.object(RoundRuntime,'stop'), \
             patch('global_bench.grt_campaign.run_workspace_turn',side_effect=self.fake_turn), \
             patch('global_bench.grt_campaign.evaluate_matrix',side_effect=self.fake_matrix):
            result=self.c.run(revised['prompt_sha256'])
        self.assertEqual(result['status'],'verified')
        with self.assertRaisesRegex(ValueError,'unstarted'):self.c.revise_prompt(brief)
        next_round=self.c.prepare()
        self.assertEqual(read(Path(next_round['control'])/'prompt/bundle.json')['research_brief_provenance']['audit_digest'],digest(brief))

    def test_invalid_prompt_scope_and_started_round_cannot_be_revised(self):
        row=self.c.prepare()
        initial=read(self.c.state_path)
        brief='---\ncomponents: [hbt]\n---\nDifferent permissions.\n'
        with self.assertRaisesRegex(ValueError,'frozen source scope'):self.c.revise_prompt(brief)
        self.assertEqual(read(self.c.state_path),initial)
        self.assertFalse((Path(row['control'])/'prompt-revisions').exists())
        write(Path(row['control'])/'container-state.json',{'running':False})
        with self.assertRaisesRegex(ValueError,'agent or checkout'):self.c.revise_prompt(self.f.task['optimization']['prompt'])
        self.assertEqual(read(self.c.state_path),initial)

    def test_interrupted_prompt_revision_blocks_run_and_new_preparation(self):
        self.c.prepare()
        with patch('global_bench.grt_campaign.save_bundle',side_effect=OSError('interrupted write')):
            with self.assertRaises(OSError):self.c.revise_prompt(self.f.task['optimization']['prompt'])
        self.assertEqual(read(self.c.state_path)['rounds'][-1]['status'],'revising_prompt')
        with self.assertRaises(ValueError):self.c.prepare()
        with self.assertRaises(ValueError):self.c.run('0'*64)
        self.assertEqual(read(self.c.state_path)['evaluations_used'],0)

    def test_real_queue_pool_releases_six_gpu_leases_before_archive(self):
        import threading
        from grt_bench.common import records,sha
        from grt_bench.contract import METRICS
        from global_bench.grt_campaign import evaluate_matrix
        child=self.f.child()
        config=dict(self.config,resources={'cpu':8,'memory_gb':24,'storage_gb':40,'gpu_count':1,'build_slots':0})
        barrier=threading.Barrier(6)
        observed=[]
        def evaluate(lease,environment,heartbeat,release):
            barrier.wait(timeout=15)
            rows=self.f.queue.rows()
            active=[r for r in rows if r['state']=='running']
            observed.append(sum(len(json.loads(r['result'])['gpus']) for r in active))
            release(config['environment']['phase_resources']['backend'],
                    {'terminated':True,'container':'grt-'+lease['id'][:32]+'-frontend','checkpoint_sha256':'a'*64})
            payload=lease['payload'];attempt=self.f.root/'runs'/lease['id']
            write(attempt/'control/task.json',self.f.task)
            write(attempt/'control/plan.json',payload)
            analysis=attempt/'output/analysis';write(analysis/'metrics.json',dict.fromkeys(METRICS,9.))
            write(analysis/'manifest.json',{'artifacts':records(analysis)})
            write(attempt/'candidate-verification.json',self.f.archive.admit(child))
            verified={**payload,'status':'verified','metrics':dict.fromkeys(METRICS,9.),
                'evidence':{'analysis_manifest_sha256':sha(analysis/'manifest.json'),'plan_sha256':sha(attempt/'control/plan.json')}}
            write(attempt/'verified/result.json',verified)
            result={**verified,'accounting':{'synthetic_test':True},'candidate_verification_sha256':sha(attempt/'candidate-verification.json')}
            write(attempt/'result.json',result)
            return result
        with patch('grt_bench.worker.evaluate',side_effect=evaluate):
            result=evaluate_matrix(config,self.f.task,child,self.f.root/'matrix-test')
        self.assertTrue(result['non_dominated']);self.assertEqual(max(observed),6)
        import sqlite3
        with sqlite3.connect(self.f.queue.path) as db:
            count=db.execute("SELECT COUNT(*) FROM events WHERE state='resources_released'").fetchone()[0]
        self.assertEqual(count,6)
        self.assertTrue(all(r['state']=='succeeded' for r in self.f.queue.rows()))

    def test_bridge_lists_diagnostic_tools_and_binds_checkout(self):
        row=self.c.prepare();runtime=RoundRuntime(read(Path(row['control'])/'tool-config.json'))
        bridge=RoundTools(runtime)
        bridge.handle({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-06-18'}})
        bridge.handle({'jsonrpc':'2.0','method':'notifications/initialized'})
        result=bridge.handle({'jsonrpc':'2.0','id':2,'method':'tools/list'})
        self.assertEqual([t['name'] for t in result['result']['tools']],['exec','checkout_parent','gpu_check'])
        bad=bridge.handle({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'checkout_parent','arguments':{'parent':'0'*64}}})
        self.assertIn('error',bad);self.assertFalse(runtime.workspace.exists())
        for d in ['tmp','home','cache']:(runtime.scratch/d).mkdir()
        command=runtime.command(False)
        self.assertNotIn('--gpus',command);self.assertIn('--network=none',command)
        self.assertNotIn('/var/run/docker.sock',' '.join(command))
        self.assertNotIn(str(self.f.root/'control'), ' '.join(command))

    def test_narrow_edit_scope_keeps_parent_algorithms_in_writable_development(self):
        from grt_bench.algorithm_scope import DREAMPLACE
        row = self.c.prepare(); self.fake_turn()
        runtime = RoundRuntime(read(Path(row['control'])/'tool-config.json'))
        for name in ['tmp','home','cache']:
            (runtime.scratch/name).mkdir()
        # Local exploration is writable; final export still enforces the edit grant.
        name = DREAMPLACE+'BasicPlace.py'
        path = runtime.workspace/name;path.parent.mkdir(parents=True);path.write_text('baseline_version = 2\n')
        self.assertNotIn(name, runtime.task['optimization']['allowed_files'])
        command = runtime.command(True)
        target = '/opt/dreamplace/dreamplace'
        mounted = [s for s in command if 'target='+target in s.split(',')]
        self.assertEqual(len(mounted), 1)
        self.assertFalse(mounted[0].endswith(',readonly'))
        self.assertIn('source='+str(runtime.workspace / DREAMPLACE.rstrip('/')), mounted[0])


if __name__=='__main__':unittest.main()
