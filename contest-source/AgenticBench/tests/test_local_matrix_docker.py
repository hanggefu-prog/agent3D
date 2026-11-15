"""Resource and build boundaries of the shared local matrix transport."""
from pathlib import Path
from unittest import mock

import pytest

from grt_bench.common import read, sha, write
from grt_bench.queue import Queue
from grt_bench.runtime import UncertainExecution
from local_bench.matrix_docker import DockerMatrixTransport
from local_bench.model import load_task
from local_bench.task_inventory import MATRIX_TASK_DIRECTORIES

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:' + 'a'*64


def frontend_checkpoint(output):
    checkpoint = output/'run/30_placement/checkpoint.json'
    write(checkpoint, {'test': 'completed placement'})
    write(output/'gpu-continuation.json', {'inventory': {str(checkpoint): sha(checkpoint)}})


def transport(tmp):
    queue = tmp/'queue.sqlite'
    Queue(queue).register_host('test',dict(cpu=64,memory_gb=208,storage_gb=320,
        gpus=['0','1'],gpu_slots_per_device=3,build_slots=1))
    return DockerMatrixTransport(queue=queue,host='test',image=IMAGE,admission_timeout=1)


@pytest.mark.parametrize('name', MATRIX_TASK_DIRECTORIES)
def test_only_frontend_cells_reserve_gpu_and_lease_ends(tmp_path, name):
    t = transport(tmp_path)
    spec = load_task(ROOT/'local_tasks'/name)
    cand=tmp_path/'candidate';cand.mkdir()
    baseline=tmp_path/'input';baseline.mkdir()
    ref=tmp_path/'reference/run';ref.mkdir(parents=True)
    base=tmp_path/'base.yaml';base.write_text('{}')
    output=tmp_path/'cell/result'
    commands=[]
    def run(command, *args, **kwargs):
        commands.append(command)
        if '-frontend' in command[command.index('--name') + 1]:
            frontend_checkpoint(output)
            return
        write(output/'evaluation_result.json',{'metrics':{'runs_successfully':1,'combined_score':1},'artifacts':{}})
    with mock.patch('local_bench.matrix_docker.admission'), mock.patch('local_bench.matrix_docker.stop_and_confirm'), mock.patch('local_bench.matrix_docker.run_container',side_effect=run):
        t.evaluate_cell(spec.root,cand,ROOT.parent,output,image=IMAGE,baseline_run=baseline,
            reference_run=ref,base_config=base,seed_schedule=dict(io=1,partition=1001,placement=1001,routing=-1),
            timeout_seconds=30)
    assert ('--gpus' in commands[0]) == (spec.start_stage in {'partition','placement'})
    assert '--network=none' in commands[0] and '--read-only' in commands[0]
    assert t.queue.rows()[0]['state']=='succeeded'
    assert (output.parent/'container/lease.json').exists()
    if spec.task_id == 'tp_gnn_post_placement_timing':
        assert len(commands) == 2 and '--gpus' not in commands[1]
        assert read(output.parent/'container/resource-release.json')['allocation']['gpus'] == []


def test_native_runtime_is_prepared_once_without_gpu(tmp_path):
    t=transport(tmp_path)
    spec=load_task(ROOT/'local_tasks/global_routing_maze_cost')
    runtime=tmp_path/'receipt.json';runtime.write_text('{}')
    with mock.patch('local_bench.matrix_docker.admission'), mock.patch(
            'scripts.build_routing_candidate_image.build_candidate_image',
            return_value={'image':IMAGE,'receipt':str(runtime)}) as build:
        bound=t.prepare_candidate(spec.root,tmp_path,ROOT.parent,tmp_path/'build')
    assert build.call_count==1 and callable(bound)
    row=t.queue.rows()[0]
    import json
    assert json.loads(row['resources'])['gpu_count']==0
    assert json.loads(row['resources'])['build_slots']==1
    assert row['state']=='succeeded'


def test_uncertain_container_does_not_release_resources(tmp_path):
    t=transport(tmp_path)
    with mock.patch('local_bench.matrix_docker.admission'), pytest.raises(UncertainExecution):
        with t.lease(tmp_path/'cell',dict(cpu=4,memory_gb=24,storage_gb=16,gpu_count=1,build_slots=0)):
            raise UncertainExecution('injected disconnect')
    assert t.queue.rows()[0]['state']=='running'


def test_known_failure_releases_resources(tmp_path):
    t=transport(tmp_path)
    with mock.patch('local_bench.matrix_docker.admission'), pytest.raises(ValueError):
        with t.lease(tmp_path/'cell',dict(cpu=4,memory_gb=24,storage_gb=16,gpu_count=1,build_slots=0)):
            raise ValueError('injected failure after confirmed termination')
    assert t.queue.rows()[0]['state']=='failed'


def test_native_failure_receipt_marks_resource_job_failed(tmp_path):
    t=transport(tmp_path)
    spec=load_task(ROOT/'local_tasks/placement_initialization')
    candidate=tmp_path/'candidate';candidate.mkdir()
    baseline=tmp_path/'input';baseline.mkdir()
    base=tmp_path/'base.yaml';base.write_text('{}')
    output=tmp_path/'cell/result'
    def run(*args,**kwargs):
        write(output/'evaluation_result.json',{'metrics':{'runs_successfully':0,'combined_score':0},
                                              'artifacts':{'error_message':'injected native error'}})
    with mock.patch('local_bench.matrix_docker.admission'), mock.patch('local_bench.matrix_docker.run_container',side_effect=run):
        with pytest.raises(ValueError,match='injected native error'):
            t.evaluate_cell(spec.root,candidate,ROOT.parent,output,image=IMAGE,baseline_run=baseline,
                            reference_run=None,base_config=base,timeout_seconds=30)
    assert t.queue.rows()[0]['state']=='failed'


@pytest.mark.parametrize('uncertain', ['none', 'frontend', 'backend'])
def test_tp_gpu_release_precedes_cpu_backend_and_waiting_gpu_can_start(tmp_path, uncertain):
    import json
    t = transport(tmp_path)
    t.queue.register_host('test', dict(cpu=64, memory_gb=208, storage_gb=320,
        gpus=['0'], gpu_slots_per_device=1, build_slots=1))
    output = tmp_path/'cell/result'
    candidate = tmp_path/'candidate'; candidate.mkdir()
    baseline = tmp_path/'baseline'; baseline.mkdir()
    base = tmp_path/'base.yaml'; base.write_text('{}')
    sequence = []
    competing = None
    def run(command, folder, name, **kwargs):
        nonlocal competing
        if name.endswith('-frontend'):
            sequence.append('gpu')
            frontend_checkpoint(output)
        else:
            sequence.append('cpu')
            assert '--gpus' not in command
            row = t.queue.rows()[0]
            assert row['state'] == 'running'
            assert json.loads(row['allocation'])['gpu_count'] == 0
            assert json.loads(row['allocation'])['cpu'] == 4
            job = t.queue.submit({'kind':'waiting-gpu'}, dict(cpu=4,memory_gb=24,
                storage_gb=16,gpu_count=1,build_slots=0))
            competing = t.queue.claim('test',kind='waiting-gpu',job_id=job)
            assert competing['gpus'] == ['0']
            if uncertain == 'backend':
                raise UncertainExecution('backend termination unknown')
            write(output/'evaluation_result.json', {'metrics':{'runs_successfully':1},'artifacts':{}})
    def stop(*args):
        sequence.append('confirmed-stop')
        if uncertain == 'frontend':
            raise UncertainExecution('frontend termination unknown')
    with mock.patch('local_bench.matrix_docker.admission'), mock.patch(
            'local_bench.matrix_docker.stop_and_confirm',side_effect=stop), mock.patch(
            'local_bench.matrix_docker.run_container',side_effect=run):
        args=(ROOT/'local_tasks/tp_gnn_post_placement_timing',candidate,ROOT.parent,output)
        kwargs=dict(image=IMAGE,baseline_run=baseline,reference_run=None,base_config=base,timeout_seconds=30)
        if uncertain == 'none':
            t.evaluate_cell(*args,**kwargs)
        else:
            with pytest.raises(UncertainExecution): t.evaluate_cell(*args,**kwargs)
    row = next(r for r in t.queue.rows() if 'local-matrix' in r['payload'])
    assert sequence == ['gpu','confirmed-stop'] + ([] if uncertain == 'frontend' else ['cpu'])
    assert json.loads(row['allocation'])['gpu_count'] == int(uncertain == 'frontend')
    assert row['state'] == ('succeeded' if uncertain == 'none' else 'running')
