"""Resource-bounded engineering matrix using the qualified type3 queue leases.

Each worker owns a complete case/seed stage-only job. It does not reserve GPUs.
Failures are terminal; uncertain container termination retains the lease.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from grt_bench.queue import Queue
from grt_bench.runtime import stop_and_confirm, UncertainExecution
from .identity import evaluator_identity
from .protocol import read_json, write_new, now, sha


def run(root, phase, cases, seeds, workers, task=None):
    root = Path(root).resolve()
    plan = read_json(root / 'plan.json')
    label = phase + ('-' + task if task else '') + '-' + '-'.join(seeds) + '-' + '-'.join(cases)
    output = root / 'dispatches' / label
    output.mkdir(parents=True, exist_ok=False)
    queue = Queue(root / 'control/engineering-queue.sqlite')
    # Register once, never silently change capacity underneath live evaluations.
    if not (root / 'control/capacity.json').exists():
        queue.register_host('rafa1-engineering',plan['capacity'])
        write_new(root / 'control/capacity.json',plan['capacity'])
    identity = evaluator_identity()
    framework = Path(__file__).resolve().parents[1]
    commands = {}
    tasks = {'clock_compensation':'clock_final_queue.py', 'multi_hbt_fanout':'multi_final_queue.py',
             'cross_die_cts':'cts_final_queue.py'}
    for cell in plan['cells']:
        if cell['case'] not in cases or cell['seed'] not in seeds: continue
        base = Path(cell['root']); experiment=base/'experiment'
        payload={'phase':phase,'task':task,'case':cell['case'],'seed':cell['seed'],
                 'root':str(base),'evaluator':identity}
        if phase=='prepare':
            cmd=[sys.executable,str(framework/'engineering_bench/physical/prepare_cell.py'),
                 '--root',str(base),'--case',cell['case'],'--experiment',str(experiment)]
            receipt=experiment/'preparation-result.json'
        else:
            development=root/'development'/task
            subject=development/'source'
            payload['source_inventory']=sha(development/'draft-inventory.json')
            final=root/'finals'/task/cell['seed']/cell['case']
            cmd=[sys.executable,str(framework/'engineering_bench/physical'/tasks[task]),
                '--root',str(base),'--experiment',str(experiment),'--output',str(final),
                '--subject',str(subject),'--baseline-inventory',str(development/'baseline-inventory.json'),
                '--cases',cell['case']]
            if task=='multi_hbt_fanout':
                cmd+=['--context-root',str(experiment/'multi-context'),
                      '--qualification-root',str(experiment/'qualifications'/task)]
            runtime=root/'builds'/task/'receipt.json'
            if task=='cross_die_cts' and runtime.exists(): cmd+=['--candidate-runtime',str(runtime)]
            receipt=final/'ledger'/cell['case']/'result.json'
        job=queue.submit(payload,plan['resources'])
        commands[job]=(cmd,receipt,cell['seed'])
    write_new(output/'admission.json',{'jobs':list(commands),'evaluator':identity,'at':now(),
        'resources':plan['resources'],'capacity':plan['capacity'],'gpu_reserved':False})
    started=time.monotonic()
    def worker(index):
        while True:
            claim=queue.claim('rafa1-engineering',ttl=120)
            if claim is None: return
            job=claim['id'] if 'id' in claim else claim['job']
            if job not in commands: raise ValueError('Do not dispatch another campaign phase')
            cmd,receipt,seed=commands[job]
            folder=output/job;folder.mkdir()
            env=os.environ.copy();env.update(PYTHONPATH=str(framework),
                ENGINEERING_TASK_ROOT=str(root/'contracts'/seed),ENGINEERING_JOB_ID=job,
                PATH=str(root/'bin')+os.pathsep+env.get('PATH',''))
            begin=time.monotonic();code=None;process=None
            try:
                if evaluator_identity()!=identity: raise ValueError('Frozen evaluator changed before dispatch')
                write_new(folder/'command.json',{'command':cmd,'seed':seed,'started_at':now()})
                with (folder/'console.log').open('x') as log:
                    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    while True:
                        queue.heartbeat(job,claim['token'],ttl=120)
                        try: code=process.wait(timeout=10);break
                        except subprocess.TimeoutExpired:
                            if time.monotonic()-begin>14400: raise TimeoutError('Engineering case budget exceeded')
                row=read_json(receipt)
                success=code==0 and row['status'] in ('qualified','succeeded')
                if evaluator_identity()!=identity: raise ValueError('Frozen evaluator changed during execution')
                result={'status':'succeeded' if success else 'failed','receipt':str(receipt),
                    'receipt_sha256':sha(receipt),'returncode':code,'wall_seconds':time.monotonic()-begin}
            except Exception as error:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                success=False;result={'status':'failed','error':repr(error),'wall_seconds':time.monotonic()-begin}
            try:
                live=subprocess.check_output(['/usr/bin/docker','ps','-q','--filter','label=engineering.job='+job],text=True,timeout=30).split()
                for name in live: stop_and_confirm(name,folder)
            except Exception as error:
                write_new(folder/'uncertain.json',{'error':repr(error),'lease_retained':True})
                raise UncertainExecution('Engineering containers have uncertain termination') from error
            write_new(folder/'result.json',result)
            queue.finish(job,claim['token'],result,success)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker,range(workers)))
    write_new(output/'timing.json',{'wall_seconds':time.monotonic()-started,'completed_at':now(),'jobs':len(commands)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','final'),required=True)
    p.add_argument('--task',choices=('clock_compensation','multi_hbt_fanout','cross_die_cts'))
    p.add_argument('--cases',nargs='+',default=['bp_fe','bp','ariane133','swerv_wrapper'])
    p.add_argument('--seeds',nargs='+',default=['s1','s2','s3'])
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.phase=='final' and not a.task: p.error('--task required for finals')
    run(a.root,a.phase,a.cases,a.seeds,a.workers,a.task)


if __name__=='__main__': main()
