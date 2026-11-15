import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_bench import routing_reference as hook, reference_latest as latest
from local_bench.workspace import openroad_source_fingerprint
from test_reference_latest import fixture, write


def make_hook(tmp_path, monkeypatch):
    parent = tmp_path/'reference'
    evaluation = parent/'result'
    run = evaluation/'run'
    run.mkdir(parents=True)
    candidate = {'source_patch': 'trusted initial patch'}
    spec = SimpleNamespace(task_id='global_routing_net_ordering_source', initial_candidate=Path('/trusted'))
    monkeypatch.setattr(hook, 'load_candidate', lambda p: {})
    monkeypatch.setattr(hook, 'initial_program_text', lambda *a: 'initial')
    monkeypatch.setattr(hook, 'materialize_program', lambda *a: candidate)
    write(parent/'candidate/candidate.json', candidate)
    workspace = evaluation/'workspace/Open3DBench-V2.0'
    native = workspace/'OpenROAD-3D/openroad'
    native.mkdir(parents=True)
    (native/'CMakeLists.txt').write_text('trusted source')
    binary = str(hook.PREBUILT_OPENROAD_BINARY)
    build = dict(schema_version=1, kind='routing-source-installed-openroad', image_id='sha256:'+'a'*64,
        base_image_id=latest.IMAGE_ID, source_commit=latest.SOURCE_REVISION, openroad_commit=latest.NATIVE_REVISION,
        task=spec.task_id, candidate_sha256=hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        binary_path=binary, binary_sha256='b'*64, source_fingerprint=openroad_source_fingerprint(native),
        runtime_verification=dict(binary_path=binary,binary_sha256='b'*64,resolved_command=binary,version_exit_code=0))
    write(parent/'image-build/receipt.json',build)
    write(evaluation/'openroad-prebuilt-receipt.json',{**build,'verification':'passed'})
    write(parent/'result.attempt.json',dict(schema_version=2,state='exited',exit_code=0,evaluator_process_started=True,
        contract=dict(kind='local-task-smoke-evaluation',task=spec.task_id,source_commit=latest.SOURCE_REVISION,
            image=latest.IMAGE_ID,candidate_sha256=latest.sha256(parent/'candidate/candidate.json'),create_routing_reference=True,reference=None)))
    command=['/usr/bin/python3','-m','open3dbench','--repo',str(workspace),'global-route',str(evaluation/'resolved_config.yaml'),'--run-dir',str(run)]
    write(evaluation/'evaluation_result.json',dict(metrics={'runs_successfully':1},artifacts=dict(
        task=spec.task_id,run_dir=str(run),workspace=str(workspace),exit_code=0,openroad_runtime=binary,command=command,
        native_flow_execution=[dict(command=command,process_started=True,returncode=0,status='completed')])))
    return run,spec,build


def test_verified_initial_hook_identity(tmp_path,monkeypatch):
    run,spec,build=make_hook(tmp_path,monkeypatch)
    assert hook.validate_initial_hook_image(run,spec,tmp_path)['image_id']==build['image_id']


@pytest.mark.parametrize('mutation', ['candidate','image_receipt','source','task','base','binary','native','ledger','missing'])
def test_mutations_rejected(tmp_path,monkeypatch,mutation):
    run,spec,build=make_hook(tmp_path,monkeypatch)
    evaluation,parent=run.parent,run.parent.parent
    if mutation=='candidate': write(parent/'candidate/candidate.json',{'source_patch':'evolved patch'})
    elif mutation=='source': (evaluation/'workspace/Open3DBench-V2.0/OpenROAD-3D/openroad/CMakeLists.txt').write_text('changed')
    elif mutation=='missing': (evaluation/'openroad-prebuilt-receipt.json').unlink()
    elif mutation=='ledger':
        p=parent/'result.attempt.json'; d=json.loads(p.read_text()); d['contract']['create_routing_reference']=False; write(p,d)
    elif mutation=='native':
        p=evaluation/'evaluation_result.json'; d=json.loads(p.read_text()); d['artifacts']['native_flow_execution'][0]['returncode']=1; write(p,d)
    else:
        if mutation=='image_receipt': build['image_id']='sha256:'+'c'*64
        if mutation=='task': build['task']='another_task'
        if mutation=='base': build['base_image_id']='sha256:'+'d'*64
        if mutation=='binary': build['runtime_verification']['binary_sha256']='e'*64
        write(parent/'image-build/receipt.json',build)
        if mutation!='image_receipt': write(evaluation/'openroad-prebuilt-receipt.json',{**build,'verification':'passed'})
    with pytest.raises((ValueError,FileNotFoundError)):
        hook.validate_initial_hook_image(run,spec,tmp_path)


def test_canonical_grt_does_not_silently_accept_hook_image(tmp_path,monkeypatch):
    config=fixture(tmp_path)
    p=tmp_path/'80_route/global_route_checkpoint.json'
    d=json.loads(p.read_text());d['image_id']='sha256:'+'a'*64;write(p,d)
    with pytest.raises(ValueError,match='exact image'):
        latest.validate_grt_reference(tmp_path,config)
    monkeypatch.setattr(hook,'validate_initial_hook_image',lambda *a:{'image_id':d['image_id']})
    result=latest.validate_grt_reference(tmp_path,config,hook_spec=object(),source_root=tmp_path)
    assert result['initial_hook_identity']['image_id']==d['image_id']
    monkeypatch.setattr(hook,'validate_initial_hook_image',lambda *a:{'image_id':'sha256:'+'b'*64})
    with pytest.raises(ValueError,match='checkpoint image differs'):
        latest.validate_grt_reference(tmp_path,config,hook_spec=object(),source_root=tmp_path)
