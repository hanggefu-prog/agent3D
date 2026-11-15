"""Host-only final tests after freezing the complete optimization frontier.

Test results never update the optimization archive or return to an agent round.
"""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

from grt_bench.archive import Archive
from grt_bench.common import digest, read, sha, verify_records, write
from grt_bench.contract import matrix, validate
from grt_bench.pareto import aggregate
from grt_bench.queue import Queue
from grt_bench.worker import loop
from .grt_campaign import ACTIVE, LocalWorkerClient, PoolFinished
from .grt_storage import Store


def freeze_finalists(config, task):
    root=Path(config['root']);state=read(root/'control/campaign.json')
    if state['evaluations_used'] != config['evaluation_limit']:
        raise ValueError('Complete the optimization budget before releasing final tests')
    if any(r['state'] in ACTIVE for r in Queue(root/'control/queue.sqlite').rows()):
        raise ValueError('Finish all active optimization jobs before freezing finalists')
    if not matrix(task, task['optimization']['baseline'], 'test'):
        raise ValueError('No final test cases defined')
    archive=Archive(root,task);view=archive.show()
    candidates=sorted({task['optimization']['baseline'],*(r['candidate'] for r in view['frontier'])})
    for candidate in candidates: archive.admit(candidate)
    frozen={'task_digest':validate(task),'candidates':candidates,
            'training_archive_digest':digest(view),'evaluations_used':state['evaluations_used'],
            'selection':'original baseline plus entire final optimization frontier',
            'at':time.time()}
    write(root/'control/final-test-frozen.json',frozen)
    return frozen


def verified_test_outcomes(root, task, candidate):
    root=Path(root);archive=Archive(root,task);audit=archive.admit(candidate)
    expected={digest(p):p for p in matrix(task,candidate,'test')};results=[]
    for row in Queue(root/'control/queue.sqlite').rows():
        p=json.loads(row['payload'])
        if p.get('task_digest')!=validate(task) or p.get('candidate')!=candidate or p.get('split')!='test':continue
        if digest(p) not in expected or row['state']!='succeeded':raise ValueError('Incomplete or duplicate final test cell')
        expected.pop(digest(p));attempt=root/'runs'/row['id'];result=json.loads(row['result'])
        if digest({'payload':p,'resources':json.loads(row['resources'])})!=row['id']:raise ValueError('Test job identity changed')
        if result!=read(attempt/'result.json'):raise ValueError('Test worker receipt changed')
        verified=read(attempt/'verified/result.json')
        if {k:v for k,v in result.items() if k not in {'accounting','candidate_verification_sha256'}}!=verified:raise ValueError('Test verifier receipt changed')
        if any(verified.get(k)!=v for k,v in p.items()) or verified['status']!='verified':raise ValueError('Test result contract changed')
        analysis=attempt/'output/analysis';evidence=verified['evidence']
        if sha(attempt/'control/plan.json')!=evidence['plan_sha256'] or sha(analysis/'manifest.json')!=evidence['analysis_manifest_sha256']:raise ValueError('Test evidence changed')
        if read(attempt/'control/task.json')!=task:raise ValueError('Test task changed')
        if read(attempt/'candidate-verification.json')!=audit or sha(attempt/'candidate-verification.json')!=result['candidate_verification_sha256']:raise ValueError('Test source audit changed')
        verify_records(analysis,read(analysis/'manifest.json')['artifacts']);results.append(result)
    if expected:raise ValueError('Incomplete final test matrix')
    return sorted(results,key=lambda r:(r['case'],r['seed']['id']))


def evaluate_test_candidate(config, task, candidate, folder):
    root=Path(config['root']);folder=Path(folder);folder.mkdir(parents=True)
    q=Queue(root/'control/queue.sqlite');q.register_host(config['host'],config['capacity'])
    environment=copy.deepcopy(config['environment']);resources=config['resources']
    write(root/'control/test-releases'/(candidate+'.json'),{'candidate':candidate,'task_digest':validate(task),'frozen':True})
    payloads=matrix(task,candidate,'test');sources={}
    if candidate==task['optimization']['baseline'] and config.get('final_test_reference_sources'):
        resources=environment.pop('phase_resources')['backend']
        environment['rescore_previous_framework']=config['final_test_previous_framework']
        for p in payloads:
            source=config['final_test_reference_sources'][p['case']][p['seed']['id']]
            sources[digest({'payload':p,'resources':resources})]=source
        environment['rescore_sources']=sources
    jobs=[q.submit(p,resources) for p in payloads]
    write(folder/'admission.json',{'candidate':candidate,'jobs':jobs,'split':'test','resources':resources})
    abort=threading.Event()
    def work():
        try:loop(LocalWorkerClient(q,config['host'],abort),environment)
        except PoolFinished:return
        except BaseException:abort.set();raise
    with Store(root).available(list(sources.values())):
        with ThreadPoolExecutor(max_workers=config['worker_count']) as pool:
            futures=[pool.submit(work) for _ in range(config['worker_count'])]
            for future in futures:future.result()
    outcomes=verified_test_outcomes(root,task,candidate)
    write(folder/'outcomes.json',outcomes)
    Store(root).compact(jobs)
    return outcomes


def run_final_tests(config, task):
    root=Path(config['root']);frozen=freeze_finalists(config,task)
    archive=Archive(root,task);baseline=task['optimization']['baseline'];folder=root/'final-tests'
    reference=evaluate_test_candidate(config,task,baseline,folder/baseline)
    results=[]
    for candidate in [baseline]+[c for c in frozen['candidates'] if c!=baseline]:
        if digest(archive.show())!=frozen['training_archive_digest']:raise ValueError('Optimization archive changed after final freeze')
        outcomes=reference if candidate==baseline else evaluate_test_candidate(config,task,candidate,folder/candidate)
        result=aggregate(task,candidate,outcomes,reference,split='test');results.append(result)
        write(folder/candidate/'summary.json',result)
    if digest(archive.show())!=frozen['training_archive_digest']:raise ValueError('Final tests changed optimization archive')
    write(folder/'completed.json',{'frozen':frozen,'results':results,'at':time.time()})
    return results
