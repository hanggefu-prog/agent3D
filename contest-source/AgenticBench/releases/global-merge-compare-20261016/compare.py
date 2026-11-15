"""Compare the merged global controller against an existing frozen GRT candidate.

Uses an isolated copy of baseline evidence and archive state. Shared scheduler
leases bound the test; no model invocation or formal campaign budget is used.
"""
import json,os,shutil,sqlite3,subprocess,sys,threading,time,traceback
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
home=Path('/data-ssd/gcr/agenticbench-integration-20261016')
source=home/'source-global-compare';sys.path.insert(0,str(source/'AgenticBench'))
from grt_bench.common import read,write,sha,digest,evaluator_identity
from grt_bench.queue import Queue
from grt_bench.builder import build,verify_runtime
from grt_bench.candidates import verify_snapshot
from grt_bench.contract import validate
from grt_bench.archive import Archive
from grt_bench.runtime import admission
from global_bench.grt_campaign import evaluate_matrix
original=Path('/data-ssd/gcr/agentic3d-evolution-20261016-formal')
root=home/'global-compare-v1';root.mkdir(exist_ok=True)
candidate='b085f088129ef96ae85d4dc93850bd9a8c98b036e70befd886b33c34e7d23140'
old_build=original/'builds/735c292adf459770e0c4c663376440cb7b64d2e93cbd296553a000453bfb0d88/receipt.json'
old_runtime=read(old_build);config=read(original/'control/config.json')
baseline=read(original/'candidates'/candidate/'candidate.json')['parent']
shared=Queue(original/'control/queue.sqlite')
@contextmanager
def reservation(stage,resources):
 payload={'kind':'merge-global-comparison','stage':stage,'output':str(root)}
 job=shared.submit(payload,resources);lease=None;start=time.monotonic()
 while lease is None:
  lease=shared.claim(config['host'],kind=payload['kind'],job_id=job)
  if lease is None:time.sleep(5)
 done=threading.Event();fail=[]
 def keep():
  while not done.wait(15):
   try:shared.heartbeat(job,lease['token'])
   except Exception as e:fail.append(repr(e));return
 thread=threading.Thread(target=keep,daemon=True);thread.start();ok=False
 try:
  admission(root,resources,[]);yield;assert not fail,fail;ok=True
 finally:
  done.set();thread.join()
  # Native builders and matrix workers are synchronous and have terminated
  # before a normal return or an ordinary failure reaches this boundary.
  uncertain=list(root.glob('matrix/failures.json'))
  if not uncertain:
   shared.finish(job,lease['token'],{'status':'succeeded' if ok else 'failed','output':str(root),'seconds':time.monotonic()-start},ok)
  write(root/(stage+'-reservation.json'),{'job':job,'status':'succeeded' if ok else 'failed','resources':resources,'seconds':time.monotonic()-start,'heartbeat_errors':fail})
def backup(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 with sqlite3.connect('file:'+str(src)+'?mode=ro',uri=True) as a,sqlite3.connect(str(dst)) as b:a.backup(b)
def clone_tree(src,dst):shutil.copytree(src,dst,copy_function=os.link,symlinks=True)
started=time.monotonic()
try:
 with reservation('build-v2',dict(cpu=16,memory_gb=48,storage_gb=45,gpu_count=0,build_slots=1)):
  for name in (baseline,candidate):
   target=root/'candidates'/name
   if not target.exists():clone_tree(original/'candidates'/name,target)
   assert verify_snapshot(target)==verify_snapshot(original/'candidates'/name)
  new_runtime=build(root/'candidates'/candidate,old_runtime['base_image'],root/'builds',Path(config['compiler_cache']),threads=8)
  checks={k:new_runtime[k]==old_runtime[k] for k in ('base_image',)}
  checks['source_fingerprint']=new_runtime['manifest']['source_fingerprint']==old_runtime['manifest']['source_fingerprint']
  checks['manifest_contract']={k:v for k,v in new_runtime['manifest'].items() if k not in ('cache_key','binary_sha256')}=={k:v for k,v in old_runtime['manifest'].items() if k not in ('cache_key','binary_sha256')}
  new_build=next((root/'builds').glob('*/receipt.json')).parent
  checks['generated_sources']=read(new_build/'compile/source-generation.json')==read(old_build.parent/'compile/source-generation.json')
  flags=read(root/'build-input-comparison.json')['flags_and_link_commands']
  checks['compiler_and_link_commands']=flags['old_count']==flags['new_count']==288 and not flags['changed']
  old_key=read(old_build.parent/'compile/cache-identity.json');new_key=read(new_build/'compile/cache-identity.json')
  checks['toolchain_and_cache_policy']={k:v for k,v in old_key.items() if k not in ('cache_key','builder_sha256')}=={k:v for k,v in new_key.items() if k not in ('cache_key','builder_sha256')}
  assert all(checks.values()),checks
  before={str(p.relative_to(root/'builds')):sha(p) for p in (root/'builds').rglob('phases.jsonl')}
  reuse_started=time.monotonic()
  with ThreadPoolExecutor(max_workers=2) as pool:
   reused=list(pool.map(lambda _:build(root/'candidates'/candidate,old_runtime['base_image'],root/'builds',Path(config['compiler_cache']),threads=8),range(2)))
  assert all(x==new_runtime for x in reused)
  assert before=={str(p.relative_to(root/'builds')):sha(p) for p in (root/'builds').rglob('phases.jsonl')}
  build_comparison={'status':'verified','checks':checks,'old_image':old_runtime['image'],'new_image':new_runtime['image'],'old_build_seconds':old_runtime['wall_seconds'],'new_build_seconds':new_runtime['wall_seconds'],'concurrent_repeat_calls':2,'repeat_wall_seconds':time.monotonic()-reuse_started,'repeat_recompiled':False,'binary_sha256':new_runtime['binary_sha256'],'old_binary_sha256':old_runtime['binary_sha256'],'binary_equal':new_runtime['binary_sha256']==old_runtime['binary_sha256'],'cache_key':new_runtime['cache_key'],'old_cache_key':old_runtime['cache_key'],'cache_key_changed':new_runtime['cache_key']!=old_runtime['cache_key'],'recipe_change':'perf branch adds ccache statistic snapshots; compiler options unchanged'}
  write(root/'build-comparison.json',build_comparison);print(json.dumps(build_comparison),flush=True)
  control=root/'control';control.mkdir()
  write(control/'runtimes'/(candidate+'.json'),new_runtime)
  shutil.copyfile(original/'control/runtimes'/(baseline+'.json'),control/'runtimes'/(baseline+'.json'))
  backup(original/'control/queue.sqlite',control/'queue.sqlite')
  with sqlite3.connect(str(control/'queue.sqlite')) as db:
   rows=db.execute('select id,payload from jobs').fetchall()
   keep=[j for j,p in rows if json.loads(p).get('candidate')==baseline and json.loads(p).get('kind','evaluation')=='evaluation']
   assert len(keep)==15
   db.executemany('delete from jobs where id=?',[(j,) for j,p in rows if j not in keep]);db.execute('delete from events');db.execute('delete from hosts')
  sample=read(original/'runs'/keep[0]/'control/plan.json');task=read(original/'runs'/keep[0]/'control/task.json');task_id=validate(task)
  assert task['evaluator_sha256']==evaluator_identity()
  write(control/'tasks'/(task_id+'.json'),task)
  backup(original/'control/archives'/(task_id+'.sqlite'),control/'archives'/(task_id+'.sqlite'))
  with sqlite3.connect(str(control/'archives'/(task_id+'.sqlite'))) as db:
   old_evaluation=json.loads(db.execute('select row_json from evaluations where candidate=?',(candidate,)).fetchone()[0])
   db.execute('delete from evaluations where candidate<>?',(baseline,))
  write(root/'old-evaluation.json',old_evaluation)
  for job in keep:
   dest=root/'runs'/job;dest.mkdir(parents=True)
   for name in ('control','output','verified','execution','verification'):
    if (original/'runs'/job/name).exists():clone_tree(original/'runs'/job/name,dest/name)
   for name in ('result.json','candidate-verification.json'):
    shutil.copyfile(original/'runs'/job/name,dest/name)
   target=control/'result-storage/jobs'/(job+'.json');target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(original/'control/result-storage/jobs'/(job+'.json'),target)
  shutil.copyfile(original/'control/result-storage-policy.json',control/'result-storage-policy.json')
  config.update(root=str(root),framework=str(source/'AgenticBench'),worker_count=6,capacity=dict(cpu=24,memory_gb=96,storage_gb=160,gpus=[],gpu_slots_per_device=3,build_slots=1))
  config['environment'].update(root=str(root),framework=str(source/'AgenticBench'))
  write(control/'config.json',config)
 print('Independent baseline evidence and archive prepared; starting 15 GRT suffixes.',flush=True)
 with reservation('matrix',dict(cpu=24,memory_gb=96,storage_gb=160,gpu_count=0,build_slots=0)):
  result=evaluate_matrix(config,task,candidate,root/'matrix')
  fields=('vector','metric_ratios','per_case','cells','feasible','constraints','reference_digest')
  aggregate_checks={k:result['evaluation'][k]==old_evaluation[k] for k in fields}
  comparisons=[]
  old_jobs={}
  with sqlite3.connect('file:'+str(original/'control/queue.sqlite')+'?mode=ro',uri=True) as db:
   for job,p in db.execute('select id,payload from jobs'):
    p=json.loads(p)
    if p.get('candidate')==candidate and p.get('kind','evaluation')=='evaluation':old_jobs[(p['case'],p['seed']['id'])]=job
  for job in read(root/'matrix/admission.json')['jobs']:
   got=read(root/'runs'/job/'verified/result.json');old_job=old_jobs[(got['case'],got['seed']['id'])];want=read(original/'runs'/old_job/'verified/result.json')
   deltas={k:got['metrics'][k]-v for k,v in want['metrics'].items()}
   comparisons.append({'case':got['case'],'seed':got['seed']['id'],'job':job,'old_job':old_job,'old_metrics':want['metrics'],'new_metrics':got['metrics'],'deltas':deltas,'exact_equal':got['metrics']==want['metrics'],'reused_stages':list(got['incremental']['stages']),'earliest_executed_stage':got['incremental']['impact']['start']})
  summary={'status':'verified' if all(aggregate_checks.values()) and all(x['exact_equal'] for x in comparisons) else 'difference','source_revision':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'candidate':candidate,'old_root':str(original),'new_root':str(root),'build':build_comparison,'aggregate_checks':aggregate_checks,'old_objectives':old_evaluation['vector'],'new_objectives':result['evaluation']['vector'],'cells':comparisons,'archive_admission':{k:result[k] for k in ('feasible','non_dominated','new_frontier_members','now_dominated')},'wall_seconds':time.monotonic()-started,'formal_model_calls':0,'formal_evaluation_budget_spent':0,'storage':read(root/'matrix/storage.json') if (root/'matrix/storage.json').exists() else None}
  write(root/'comparison.json',summary);print(json.dumps({k:summary[k] for k in ('status','aggregate_checks','old_objectives','new_objectives','wall_seconds')}),flush=True)
  assert summary['status']=='verified','Physical/objective difference requires investigation'
except BaseException as e:
 write(root/'failure-v2.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
