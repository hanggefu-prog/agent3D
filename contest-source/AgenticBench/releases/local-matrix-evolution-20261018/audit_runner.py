"""Two real model proposals per task, six paired native cells per proposal."""
import argparse,copy,difflib,hashlib,json,os,shutil,subprocess,sys,time,traceback,uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--task',action='append');parser.add_argument('--codex',required=True);parser.add_argument('--proxy',default='http://127.0.0.1:18890');args=parser.parse_args()
S=args.source.resolve();R=args.root.resolve();sys.path[:0]=[str(S/'AgenticBench'),str(S)]
import yaml
from grt_bench.common import write,read,sha,digest
from grt_bench.runtime import admission,docker_command,run_container
from local_bench.matrix import qualify_matrix
from local_bench.matrix_docker import DockerMatrixTransport
from local_bench.model import load_task,EvaluationResult
from local_bench.search_api import CandidateProposal,TaskEvaluationOptions
from local_bench.search_session import run_task_search
from local_bench.task_inventory import CORE_TASK_DIRECTORIES
from local_bench.prompt_context import feedback_text
from local_bench.evaluation import UPSTREAM_STAGE_DIRS
from global_bench.workspace_codex import run_workspace_turn
runtime=read(R/'runtime.json');inputs=read(R/'inputs.json')
MODEL='gpt-5.6-sol';TASKS=args.task or list(CORE_TASK_DIRECTORIES)

def file_hash(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def dedup_prefix(result,baseline,stages):
 """Lossless sharing of completed immutable prefixes; no unique file is removed."""
 records=[];freed=0
 for stage in stages:
  d=UPSTREAM_STAGE_DIRS[stage]
  for target in (result/'run'/d).rglob('*'):
   source=baseline/d/target.relative_to(result/'run'/d)
   if target.is_symlink() or source.is_symlink() or not target.is_file() or not source.is_file():continue
   a,b=target.stat(),source.stat()
   if a.st_size<1024*1024 or a.st_size!=b.st_size or a.st_ino==b.st_ino or a.st_dev!=b.st_dev or a.st_mode!=b.st_mode:continue
   x=file_hash(target)
   if x!=file_hash(source):continue
   temp=target.with_name('.share-'+uuid.uuid4().hex)
   os.link(source,temp);os.replace(temp,target)
   freed+=a.st_size if a.st_nlink==1 else 0
   records.append({'relative_path':str(target.relative_to(result)),'sha256':x,'bytes':a.st_size})
 if records:write(result/'shared-prefix.json',{'freed_bytes':freed,'files':records,'source':str(baseline)})
 return freed

class AuditTransport(DockerMatrixTransport):
 def evaluate_cell(self,task,candidate,source,output,**options):
  result=super().evaluate_cell(task,candidate,source,output,**options)
  if options.get('reference_run') is None and (Path(output)/'run').exists():
   dedup_prefix(Path(output),Path(options['baseline_run']),load_task(task).upstream_stages)
  return result

class UnsplitTransport(AuditTransport):
 def run_phased(self,request,run,image,resources,claim,mounts):
  name='grt-unsplit-check-'+claim['id'][:24]
  cmd=docker_command(image,name,resources,claim['gpus'],mounts+[(run/'request.json','/request.json',True)],['-m','local_bench.matrix_docker','--cell','/request.json'])
  run_container(cmd,run/'execution',name,timeout=request['options']['timeout_seconds']+120)

def reference_view(result):
 assert result.metrics['matrix']['complete']
 for row in result.metrics['matrix']['cells']:
  assert row['metrics'].get('grt_complete',1)==1, 'Unexercised routing reference'
 metrics=copy.deepcopy(result.metrics);m=metrics['matrix'];m['qualification']=False
 metrics['combined_score']=1.0
 for c in m['cases']:
  c.update(mean=1.0,stddev=0.0,min=1.0,max=1.0,scores={k:1.0 for k in c['scores']})
 for c in m['cells']:c['score']=1.0;c['metrics']['combined_score']=1.0
 return metrics

class Backend:
 name='codex-six-cell-two-round-audit'
 def __init__(self,folder,reference):self.folder=folder;self.programs={};self.reference=reference
 def ask(self,ctx):
  n=ctx.evaluations+1
  parent=ctx.incumbent if ctx.incumbent and ctx.incumbent.score>1.0 else None
  code=self.programs[parent.proposal_id] if parent else ctx.problem.initial_program
  feedback=[{'proposal':f.proposal_id,'status':f.status,'score':f.score,'feedback':feedback_text(f.metrics,ctx.problem.task_id),'error':f.error_message,'hypothesis':f.metadata.get('hypothesis')} for f in ctx.history]
  prompt=ctx.problem.system_message+'\n\n'+ctx.problem.description+'\n\n'+f'''This is round {n} of a two-round algorithm audit. Return one coherent algorithm improvement within the declared interface, a short hypothesis, and how it uses the feedback. Make a real source change, not a replay. Use the live features exposed by the interface; do not specialize to case names, seeds or observed answers. A plausible idea is not evidence of improvement. Check the per-case and per-seed feedback before claiming a gain. The harness will evaluate the same program on all six cells. You have no terminal tools in this ask/tell interface. Return a complete program, not a diff, with at most {ctx.problem.max_program_chars} characters.

Qualified initial-program reference: the normalized scores below are 1 by definition; raw measurements come from native qualification, not an additional replay.
{feedback_text(self.reference,ctx.problem.task_id)}

Selected parent: {parent.proposal_id if parent else 'initial program (no submitted candidate has beaten its score of 1)'}
```python
{code}
```

Chronological optimization feedback:
{json.dumps(feedback,indent=2)}
'''
  control=self.folder/'model-control';control.mkdir(exist_ok=True);schema=control/'schema.json'
  if not schema.exists():write(schema,{'type':'object','additionalProperties':False,'properties':{k:{'type':'string'} for k in ('program','hypothesis','feedback_use')},'required':['program','hypothesis','feedback_use']})
  environment=dict(os.environ,HTTP_PROXY=args.proxy,HTTPS_PROXY=args.proxy,NO_PROXY='localhost,127.0.0.1')
  result=run_workspace_turn(binary=args.codex,control=control,output=self.folder/('model-%02d'%n),container='unused',prompt=prompt,terminal_script=S/'AgenticBench/global_bench/container_terminal.py',python=sys.executable,timeout_seconds=600,environment=environment,model=MODEL,extra_overrides=('mcp_servers.terminal.enabled=false','mcp_servers.terminal.required=false','model_reasoning_effort="high"'),output_schema=schema)
  if not result['success']:raise RuntimeError('Model invocation failed before proposal; no native candidate matrix spent')
  data=json.loads(result['final_message']);pid='round-%02d'%n;self.programs[pid]=data['program']
  (self.folder/(pid+'.py')).write_text(data['program'])
  for label,baseline in [('parent',code),('reference',ctx.problem.initial_program)]:
   (self.folder/(pid+'-vs-'+label+'.diff')).write_text(''.join(difflib.unified_diff(baseline.splitlines(True),data['program'].splitlines(True),fromfile=label,tofile=pid)))
  return [CandidateProposal(pid,data['program'],metadata={'hypothesis':data['hypothesis'],'feedback_use':data['feedback_use'],'parent':parent.proposal_id if parent else 'initial'})]
 def tell(self,feedback):
  for f in feedback:print(json.dumps({'task_output':str(self.folder),'proposal':f.proposal_id,'status':f.status,'score':f.score,'error':f.error_message}),flush=True)

def run(name):
 folder=R/'tasks'/name;folder.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 try:
  spec=load_task(S/'AgenticBench/local_tasks'/name);cells=[]
  for inp in inputs:
   config=yaml.safe_load((S/'AgenticBench/local_bench/reference_configs/nesterov_current.yaml' if name=='placement_parameters_reference_20260923' else spec.base_config).read_text())
   config['design']=inp['case'][:-5]
   for stage,key in [('pins','io'),('partition','partition'),('placement','placement'),('routing','routing')]:config.setdefault(stage,{})['seed']=inp['seed'][key]
   base=folder/'configs'/inp['case']/(inp['seed']['id']+'.yaml');base.parent.mkdir(parents=True,exist_ok=True);base.write_text(yaml.safe_dump(config))
   cells.append(dict(case=inp['case'],seed=inp['seed']['id'],base_config=str(base),baseline_run=inp['baseline_run'],reference_run=None))
  seeds=[dict(id='s'+str(n),io=1,partition=1000+n,placement=1000+n,routing=-1) for n in (1,2,3)]
  write(folder/'matrix.json',dict(schema_version=1,task=spec.task_id,cases=['bp_fe-ng45','ariane133-ng45'],seeds=seeds,cells=cells))
  transport=AuditTransport(**runtime)
  workers=2 if name.startswith('global_routing_') else 6
  reference=qualify_matrix(folder/'matrix.json',spec.root,S,folder/'references',transport=transport,workers=workers,timeout_seconds=3600)
  print(json.dumps({'task':name,'reference_complete':reference.metrics['matrix']['complete'],'cells':reference.metrics['matrix']['succeeded_cells'],'free_gib':shutil.disk_usage(R).free/1024**3}),flush=True)
  if reference.metrics['runs_successfully']!=1:raise RuntimeError('Incomplete reference matrix')
  if name=='tp_gnn_post_placement_timing':
   # Independent control: same initial program and input, one unsplit container.
   cell=next(c for c in cells if c['case']=='bp_fe-ng45' and c['seed']=='s1')
   ref=next(c for c in reference.metrics['matrix']['cells'] if c['case']==cell['case'] and c['seed']==cell['seed'])
   result=UnsplitTransport(**runtime).evaluate_cell(spec.root,folder/'references/initial',S,folder/'unsplit-control/result',image=runtime['image'],baseline_run=Path(cell['baseline_run']),reference_run=Path(ref['result_dir'])/'run',base_config=Path(cell['base_config']),seed_schedule={k:v for k,v in seeds[0].items() if k!='id'},timeout_seconds=3600,keep_workspace=False,keep_runtime=False,compact_artifacts=True)
   names=['final_global_wns_setup_ns','final_global_tns_setup_ns','timing_weighted_cut','cut_net_count']
   comparison={k:{'split':ref['metrics'][k],'unsplit':result.metrics[k]} for k in names if k in ref['metrics']}
   equal=all(v['split']==v['unsplit'] for v in comparison.values()) and result.metrics['combined_score']==1.0
   write(folder/'phase-equivalence.json',{'identical':equal,'score':result.metrics['combined_score'],'metrics':comparison,'result':result.to_dict()})
   if not equal:raise RuntimeError('Split/unsplit physical comparison differs; stop before model search')
  options=TaskEvaluationOptions(spec.root,folder/'search',open3dbench_root=S,evaluation_matrix=folder/'references/qualification/qualified_matrix.json',matrix_workers=workers,timeout_seconds=3600,compact_evaluations=True)
  result=run_task_search(Backend(folder,reference_view(reference)),options,max_evaluations=2,evaluation_transport=transport)
  report={'task':spec.task_id,'status':'completed','evolution':result.to_dict(),'wall_seconds':time.monotonic()-start}
 except BaseException as e:
  report={'task':name,'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start}
 write(folder/'summary.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('evolution','traceback')}),flush=True);return report

if __name__=='__main__':
 if not (R/'budget.json').exists():write(R/'budget.json',{'model':MODEL,'tasks':list(CORE_TASK_DIRECTORIES),'cases':['bp_fe-ng45','ariane133-ng45'],'paired_seeds':[1001,1002,1003],'rounds_per_task':2,'max_model_calls':16,'reference_cells':48,'candidate_cells_max':96,'phase_equivalence_extra_cells':1,'gpu_slots_total':6,'native_cell_timeout_seconds':3600,'source':str(S),'source_commit':subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()})
 # The first TP-GNN native gate is run separately before the rest of the suite.
 with ThreadPoolExecutor(max_workers=2) as pool:
  rows=[f.result() for f in as_completed([pool.submit(run,n) for n in TASKS])]
 write(R/('summary-'+digest(TASKS)[:12]+'.json'),rows)
