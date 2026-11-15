"""Restart GRT with exercised, frozen routing capacity; reuse first proposals."""
from pathlib import Path
import os,sys,runpy,json,shutil,time,difflib,copy,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
H=Path(os.environ['AGENTIC_AUDIT_PARENT']).expanduser().resolve();R=H/'local-matrix-audit-20261018';S=H/'source-local-phases-e951f6df';N=R/'routing-active-hooks'
pre=json.loads((R/'routing-hook-preflight/summary.json').read_text())
assert pre['metrics']['grt_complete']==1, 'Stop before models: BP_FE still does not exercise hook'
N.mkdir(exist_ok=False)
for name in ('inputs.json','runtime.json'):shutil.copy2(R/name,N/name)
r=json.loads((N/'runtime.json').read_text());r['admission_timeout']=1800;(N/'runtime.json').write_text(json.dumps(r,indent=2))
sys.argv=['audit-helper','--source',str(S),'--root',str(N),'--codex',os.environ['CODEX_BIN']]
a=runpy.run_path(str(Path(__file__).with_name('routing_audit_runner.py')),run_name='audit_helper')
from grt_bench.common import read,write,sha
from local_bench.search_api import CandidateProposal
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
run=a['run'];run.__globals__.update(Backend=Backend,AuditTransport=Transport)
names=['global_routing_maze_cost','global_routing_ripup_policy','global_routing_iteration_controller']
write(N/'budget.json',{'tasks':names,'valid_rounds_per_task':2,'new_model_calls_max':3,'first_proposals_reused':3,'candidate_cells_max':36,'reference_cells':18,'routing_resource_reduction':{'bp_fe-ng45':0.5,'ariane133-ng45':0.0},'native_loop_budget_unchanged':True,'source':str(S),'source_commit':subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()})
with ThreadPoolExecutor(max_workers=3) as pool:rows=[f.result() for f in as_completed([pool.submit(run,n) for n in names])]
write(N/'summary.json',rows)
