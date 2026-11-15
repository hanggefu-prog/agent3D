from pathlib import Path
import os,sys,runpy,json,shutil,time,difflib,copy,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
H=Path(os.environ['AGENTIC_AUDIT_PARENT']).expanduser().resolve();R=H/'local-matrix-audit-20261018';S=H/'source-local-phases-e951f6df';N=R/'routing-active-hooks'
sys.argv=['audit-helper','--source',str(S),'--root',str(N),'--codex',os.environ['CODEX_BIN']]
a=runpy.run_path(str(Path(__file__).with_name('routing_audit_runner.py')),run_name='audit_helper')
from grt_bench.common import read,write,sha
from local_bench.search_api import CandidateProposal,TaskEvaluationOptions
from local_bench.search_session import run_task_search
from local_bench.model import load_task,EvaluationResult
from local_bench.matrix import LocalEvaluationMatrix
class Backend(a['Backend']):
 def ask(self,ctx):
  if ctx.evaluations:return super().ask(ctx)
  old=R/('routing-config-fix' if self.folder.name=='global_routing_iteration_controller' else 'routing-reference-recovery')/'tasks'/self.folder.name
  p=old/'model-01/result.json';data=json.loads(read(p)['final_message']);code=data['program'];self.programs['round-01']=code
  (self.folder/'round-01.py').write_text(code)
  for label in ('reference','parent'):(self.folder/('round-01-vs-'+label+'.diff')).write_text(''.join(difflib.unified_diff(ctx.problem.initial_program.splitlines(True),code.splitlines(True),fromfile=label,tofile='round-01')))
  write(self.folder/'round-01-provenance.json',{'reused_model_result':str(p),'model_result_sha256':sha(p),'new_model_call':False,'reason':'Reuse the same first algorithm hypothesis on the corrected reference contract; discard old setup feedback'})
  return [CandidateProposal('round-01',code,metadata={'hypothesis':data['hypothesis'],'feedback_use':'First proposal reused unchanged; evaluate anew under the corrected, exercised reference contract before making any improvement claim.','parent':'initial'})]
class Transport(a['AuditTransport']):
 @contextmanager
 def lease(self,folder,resources):
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
  while shutil.disk_usage(folder).free<(resources['storage_gb']+10)*1024**3:
   if time.monotonic()-start>self.admission_timeout:raise TimeoutError('Disk preflight wait exceeded audit admission budget')
   time.sleep(5)
  while True:
   entered=False
   try:
    with super().lease(folder,resources) as claim:
     entered=True;yield claim
    return
   except ValueError as error:
    if entered or not str(error).startswith('Insufficient free disk:') or time.monotonic()-start>self.admission_timeout:raise
    time.sleep(5)

def resume(name):
 folder=N/'tasks'/name;deadline=time.monotonic()+3600
 while not (folder/'summary.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Reference qualification wait expired')
  time.sleep(10)
 old=read(folder/'summary.json');assert old['status']=='failed' and 'FileNotFoundError' in old['error'],old
 assert not list((folder/'search').glob('evaluations/*'))
 spec=load_task(S/'AgenticBench/local_tasks'/name)
 matrix=folder/'references/qualification/qualified_matrix.json';LocalEvaluationMatrix(matrix,spec).assert_unchanged()
 reference=read(folder/'references/qualification/evaluation_result.json');assert reference['metrics']['matrix']['complete']
 (folder/'summary.json').rename(folder/'pre-search-path-error.json')
 if (folder/'search').exists():(folder/'search').rename(folder/'search-path-error')
 start=time.monotonic()
 try:
  options=TaskEvaluationOptions(spec.root,folder/'search',open3dbench_root=S,evaluation_matrix=matrix,matrix_workers=2,timeout_seconds=3600,compact_evaluations=True)
  result=run_task_search(Backend(folder,a['reference_view'](EvaluationResult(reference['metrics'],reference['artifacts']))),options,max_evaluations=2,evaluation_transport=Transport(**a['runtime']))
  summary={'task':name,'status':'completed','evolution':result.to_dict(),'wall_seconds':time.monotonic()-start,'qualification_reused':True,'pre_search_diagnostic':str(folder/'pre-search-path-error.json')}
 except BaseException as error:
  import traceback
  summary={'task':name,'status':'failed','error':repr(error),'traceback':traceback.format_exc()}
 write(folder/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('evolution','traceback')}),flush=True)
 return summary
names=['global_routing_maze_cost','global_routing_ripup_policy','global_routing_iteration_controller']
with ThreadPoolExecutor(max_workers=3) as pool:rows=[f.result() for f in as_completed([pool.submit(resume,n) for n in names])]
write(N/'search-resume-summary.json',rows)
