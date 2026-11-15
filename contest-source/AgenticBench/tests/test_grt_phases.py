"""Container phase boundaries preserve configuration and native resume checks."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grt_bench.common import read,write,sha,digest
from grt_bench.contract import matrix
from grt_bench.flow import execute,validate_frontend_handoff
from test_grt_matrix import task_fixture

class FlowPhaseTests(unittest.TestCase):
    def test_frontend_stops_at_placement_and_backend_continues_from_pdn(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'output';out.mkdir()
            task=task_fixture();write(root/'partition.json',{'partition':{'seed':42}})
            write(root/'placement.json',{'concurrent_dreamplace':{'initialization_seed':42}})
            config=dict(design='bp_fe',pdk='ng45',type='lol',partition={'algorithm':'tp-gnn-flat','config':str(root/'partition.json')},placement={'algorithm':'concurrent-dreamplace','config':str(root/'placement.json')})
            (root/'case.yaml').write_text(yaml.safe_dump(config))
            task['cases'][0]['config_sha256']=sha(root/'case.yaml');write(root/'task.json',task)
            cell=matrix(task,'c'*64)[0]
            plan=dict(cell,task=str(root/'task.json'),config=str(root/'case.yaml'),output=str(out),source=str(root),hotspot='/fixture',execution_mode='gpu-frontend-cpu-backend')
            runtime={'candidate':cell['candidate'],'modules':{}}
            write(root/'python-runtime.json',runtime)
            plan.update(python_runtime=str(root/'python-runtime.json'),python_runtime_digest=digest(runtime))
            write(root/'plan.json',plan)
            commands=[]
            def native(command,**kwargs):
                commands.append(command)
                if '--to' in command and command[command.index('--to')+1]=='placement':
                    for stage in ('00_input','10_floorplan','20_partition','30_placement'):
                        write(out/'run'/stage/'stage_manifest.json',{'fixture':stage})
                    write(out/'run/30_placement/checkpoint.json',{'fixture':'placement'})
            with patch('grt_bench.flow.evaluator_identity',return_value=task['evaluator_sha256']),patch('grt_bench.flow.subprocess.run',side_effect=native),patch('grt_bench.flow.run') as scoring,patch('grt_bench.flow.validate_frontend_handoff',side_effect=lambda *args:read(out/'gpu_frontend.json')):
                execute(root/'plan.json',phase='frontend')
                self.assertEqual(len(commands),1);scoring.assert_not_called()
                config_sha=sha(out/'resolved.yaml')
                execute(root/'plan.json',phase='backend')
                self.assertEqual(sha(out/'resolved.yaml'),config_sha)
                self.assertEqual(len(commands),3)
                self.assertIn('--resume',commands[1]);self.assertEqual(commands[1][commands[1].index('--from')+1],'pdn')
                self.assertIn('global-route',commands[2]);scoring.assert_called_once()
                self.assertEqual(len(read(out/'execution.json')['commands']),3)

    def test_handoff_rejects_changed_config_manifest_or_native_signature(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';write(root/'task.json',task_fixture())
            write(root/'plan.json',{'fixture':1,'task':str(root/'task.json')})
            (root/'config.yaml').write_text('fixture')
            manifests={}
            for stage in ('00_input','10_floorplan','20_partition','30_placement'):
                write(run/stage/'stage_manifest.json',{'fixture':stage})
                manifests[stage]=sha(run/stage/'stage_manifest.json')
            write(root/'gpu_frontend.json',{'plan_sha256':sha(root/'plan.json'),'config_sha256':sha(root/'config.yaml'),'stage_manifests':manifests})
            runner=Mock();runner._resume_is_valid.return_value=True
            with patch('open3dbench.config.load_config',return_value={}),patch('open3dbench.runner.FlowRunner',return_value=runner),patch('open3dbench.backend.contracts.validate_checkpoint'):
                validate_frontend_handoff(root/'plan.json',root/'config.yaml',root,run)
                self.assertEqual(runner._resume_is_valid.call_count,4)
                runner._resume_is_valid.return_value=False
                with self.assertRaisesRegex(ValueError,'signature/artifact'):validate_frontend_handoff(root/'plan.json',root/'config.yaml',root,run)
                runner._resume_is_valid.return_value=True
                (run/'30_placement/stage_manifest.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'signature/artifact'):validate_frontend_handoff(root/'plan.json',root/'config.yaml',root,run)
                (root/'config.yaml').write_text('changed')
                with self.assertRaisesRegex(ValueError,'plan/config'):validate_frontend_handoff(root/'plan.json',root/'config.yaml',root,run)
