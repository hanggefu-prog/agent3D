"""Phase split and host-sealed checkpoint continuation, without physical tools."""
import os
from pathlib import Path

import pytest

from grt_bench.common import read, sha, write
from local_bench.model import load_task
from local_bench.phased import save_continuation, load_continuation, request_binding, split_command

ROOT = Path(__file__).resolve().parents[1]


def state(tmp_path):
    spec = load_task(ROOT/'local_tasks/tp_gnn_post_placement_timing')
    run = tmp_path/'run'
    for p in (run/'20_partition/stage_manifest.json', run/'30_placement/stage_manifest.json',
              run/'30_placement/checkpoint.json', tmp_path/'resolved_config.yaml',
              tmp_path/'candidate_snapshot.json'):
        write(p, {'fixture': True})
    for name in ('partition','placement'):
        write(tmp_path/'method_seeds'/(name+'.json'),{'seed':1001})
    return dict(spec=spec, workspace=tmp_path/'workspace', run_dir=run, command_cwd=tmp_path,
        output_dir=tmp_path, reference_run=None, baseline_run=tmp_path/'baseline',
        dreamplace_overlay=None, run_environment=dict(os.environ), phase_binding='bound',
        artifacts={}, frontend_runtime_seconds=10,
        resolved_config={name:{'config':str(tmp_path/'method_seeds'/(name+'.json'))}
                         for name in ('partition','placement')})


def test_stage_split_never_repeats_partition_or_placement_in_cpu_phase():
    original=['python3','-m','open3dbench','run','config.yaml','--from','partition','--to','hbt','--resume']
    front=split_command(original,'frontend');back=split_command(original,'backend')
    assert front[front.index('--to')+1]=='placement'
    assert back[back.index('--from')+1]=='pdn' and back[back.index('--to')+1]=='hbt'
    assert original[original.index('--to')+1]=='hbt'
    with pytest.raises(ValueError): split_command(['unsupported'],'frontend')


@pytest.mark.parametrize('corruption',['seal','request','checkpoint','config','candidate','method'])
def test_modified_continuation_or_inputs_are_rejected(tmp_path, corruption):
    s=state(tmp_path);p=save_continuation(s);seal=sha(p);binding='bound'
    if corruption=='seal': p.write_text(p.read_text()+' ')
    elif corruption=='request': binding='other'
    else:
        target={'checkpoint':s['run_dir']/'30_placement/checkpoint.json',
                'config':tmp_path/'resolved_config.yaml','candidate':tmp_path/'candidate_snapshot.json',
                'method':tmp_path/'method_seeds/partition.json'}[corruption]
        target.write_text('changed')
    with pytest.raises(ValueError): load_continuation(tmp_path,seal,binding)


def test_environment_does_not_serialize_secrets_or_preserve_gpu_visibility(tmp_path,monkeypatch):
    monkeypatch.setenv('EXAMPLE_AUTH_SECRET','private-fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    s=state(tmp_path);s['run_environment']['PLACE3D_DREAMPLACE_INSTALL']='/runtime/overlay'
    p=save_continuation(s)
    assert 'private-fixture' not in p.read_text() and 'CUDA_VISIBLE_DEVICES' not in p.read_text()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    loaded=load_continuation(tmp_path,sha(p),'bound')
    assert 'CUDA_VISIBLE_DEVICES' not in loaded['run_environment']
    assert loaded['run_environment']['PLACE3D_DREAMPLACE_INSTALL']=='/runtime/overlay'


def test_request_binding_includes_candidate_seed_recipe_and_timeout(tmp_path):
    spec=state(tmp_path)['spec'];candidate=tmp_path/'candidate';write(candidate/'candidate.json',{})
    base=tmp_path/'base.yaml';base.write_text('design: bp_fe')
    kwargs=dict(base_config=base,seed_schedule={'placement':1001},timeout_seconds=30)
    def binding(**overrides):
        return request_binding(spec,candidate,ROOT.parent,tmp_path,**{**kwargs,**overrides})
    initial=binding()
    assert initial!=binding(timeout_seconds=31)
    assert initial!=binding(seed_schedule={'placement':1002})
    (candidate/'candidate.json').write_text('{"changed":true}');assert initial!=binding()


def test_compact_results_retain_algorithm_feedback_evidence_without_physical_databases(tmp_path):
    from local_bench.evaluation import _discard_heavy_evaluation_artifacts
    run=tmp_path/'run'
    write(run/'20_partition/tp_gnn_report.json',{'training':{'loss':.3}})
    log=run/'20_partition/partition.log';log.write_text('native training and decoder diagnostics')
    physical=run/'30_placement/design.odb';physical.parent.mkdir();physical.write_bytes(b'physical')
    metrics=run/'55_post_placement_sta/metrics.json';write(metrics,{'wns_setup':-.1})
    pdn_log=run/'40_pdn/top/run.log';pdn_log.parent.mkdir(parents=True)
    pdn_log.write_text('[ERROR PDN-0001] Disconnected supply stripe\n')
    artifacts={}
    _discard_heavy_evaluation_artifacts(tmp_path,artifacts)
    evidence=tmp_path/'retained_artifacts/algorithm'
    assert read(evidence/'20_partition/tp_gnn_report.json')['training']['loss']==.3
    assert (evidence/'20_partition/partition.log').read_text()=='native training and decoder diagnostics'
    assert read(evidence/'55_post_placement_sta/metrics.json')['wns_setup']==-.1
    assert (evidence/'40_pdn/top/run.log').read_text()=='[ERROR PDN-0001] Disconnected supply stripe\n'
    assert not run.exists() and not list(evidence.rglob('*.odb'))
    assert len(artifacts['algorithm_evidence']['files'])==4


@pytest.mark.parametrize('seed',[1001,1002,1003])
def test_full_tp_flow_binds_native_method_json_seeds_without_changing_originals(tmp_path,seed):
    import yaml
    from local_bench.evaluation import _bind_full_flow_method_seeds
    spec=load_task(ROOT/'local_tasks/tp_gnn_post_placement_timing')
    config=yaml.safe_load(spec.base_config.read_text())
    originals={k:ROOT.parent/'Place-3D'/config[k]['config'] for k in ('partition','placement')}
    hashes={k:sha(p) for k,p in originals.items()}
    _bind_full_flow_method_seeds(config,{'partition':seed,'placement':seed},tmp_path,ROOT.parent)
    assert read(config['partition']['config'])['partition']['seed']==seed
    assert read(config['placement']['config'])['concurrent_dreamplace']['initialization_seed']==seed
    assert {k:sha(p) for k,p in originals.items()}==hashes
