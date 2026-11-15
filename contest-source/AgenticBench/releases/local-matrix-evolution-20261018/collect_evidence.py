import json,sqlite3,sys,shutil,statistics,hashlib
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[sys.argv[2]+"/AgenticBench",sys.argv[2]]
from local_bench.prompt_context import prompt_metrics
from local_bench.failure_feedback import native_failure_diagnostics
def read(p):return json.loads(p.read_text())
def cell(path,task):
 x=read(path); out=Path(x['result_dir']);receipt=out/'evaluation_result.json'
 a=read(receipt).get('artifacts',{}) if receipt.exists() else {}
 d={k:x[k] for k in ('case','seed','status','score','error_message','result_dir') if k in x}
 d['metrics']=prompt_metrics(x.get('metrics',{}),task)
 d['algorithm_activity']={k:v for k,v in x.get('metrics',{}).items() if k.startswith(('nesterov_','tp_gnn_')) or '_policy_' in k or k.startswith('policy_')}
 d['native_execution']=a.get('native_flow_execution',[])
 d['runtime_seconds']=a.get('runtime_seconds')
 d['patch']=a.get('patch')
 if x['status']=='failed':d['native_failure_diagnostics']=native_failure_diagnostics(out)
 release=out.parent/'container/resource-release.json'
 if release.exists():
  d['gpu_release']=read(release)
  d['container_phases']={}
  for phase in ('frontend','backend'):
   phase_dir=out.parent/'container/execution'/phase
   if (phase_dir/'terminal.json').exists():
    command=read(phase_dir/'command.json')
    d['container_phases'][phase]={**read(phase_dir/'terminal.json'),'has_gpu_flag':'--gpus' in command}
 return d
report={'root':str(root),'free_gib':shutil.disk_usage(root).free/1024**3,'tasks':{}}
for p in sorted((root/'tasks').glob('*'))+sorted((root/'tp-seed-fix/tasks').glob('*'))+sorted((root/'nesterov-validation-fix/tasks').glob('*'))+sorted((root/'routing-config-fix/tasks').glob('*'))+sorted((root/'routing-reference-recovery/tasks').glob('*'))+sorted((root/'routing-active-hooks/tasks').glob('*')):
 name=p.name;key=name+(':obsolete-seed-attempt' if name=='tp_gnn_post_placement_timing' and 'tp-seed-fix' not in p.parts else ':obsolete-validation-attempt' if name=='placement_parameters_reference_20260923' and 'nesterov-validation-fix' not in p.parts else ':obsolete-setup-attempt' if name.startswith('global_routing_') and p.parent==root/'tasks' else ':incomplete-disk-attempt' if name in ('global_routing_ripup_policy','global_routing_maze_cost') and 'routing-config-fix' in p.parts else '')
 if name.startswith('global_routing_') and 'routing-active-hooks' not in p.parts and not key.endswith('attempt'):key+=':unexercised-reference-attempt'
 d={'directory':str(p),'reference_cells':[cell(f,name) for f in sorted(p.glob('references/qualification/cells/*/*/cell.json'))],'rounds':[]}
 history=p/'search/search_history.jsonl'
 if history.exists():
  d['proposal_history']=[{k:v for k,v in json.loads(line).items() if k in ('proposal_id','program_sha256','status','score','error_message','metadata','evaluation_id')} for line in history.read_text().splitlines()]
 if (p/'summary.json').exists():d['summary']={k:v for k,v in read(p/'summary.json').items() if k not in ('evolution','traceback')}
 if (p/'phase-equivalence.json').exists():d['phase_equivalence']={k:v for k,v in read(p/'phase-equivalence.json').items() if k!='result'}
 for n,f in enumerate(sorted(p.glob('search/evaluations/*')),1):
  q={'number':n,'directory':str(f),'cells':[cell(c,name) for c in sorted(f.glob('matrix/cells/*/*/cell.json'))]}
  r=f/'matrix/evaluation_result.json'
  if r.exists():
   m=read(r)['metrics'];q.update(score=m.get('combined_score'),complete=m['matrix']['complete'],cases=m['matrix']['cases'])
   if q['complete']:
    means=[statistics.mean(c['score'] for c in q['cells'] if c['case']==case['case']) for case in q['cases']]
    q['aggregate_recomputed']=statistics.mean(means)
  program=p/('round-%02d.py'%n)
  if program.exists():q['program_sha256']=hashlib.sha256(program.read_bytes()).hexdigest()
  model=p/('model-%02d'%n)/'result.json'
  provenance=p/('round-%02d-provenance.json'%n)
  if provenance.exists():
   q['proposal_provenance']=read(provenance);model=Path(q['proposal_provenance']['reused_model_result'])
  if model.exists():
   data=json.loads(read(model)['final_message']);q['proposal']={k:v for k,v in data.items() if k!='program'}
  d['rounds'].append(q)
 report['tasks'][key]=d
db=sqlite3.connect('file:'+sys.argv[3]+'?mode=ro',uri=True)
jobs={}
for jid,state,payload in db.execute('SELECT id,state,payload FROM jobs'):
 p=json.loads(payload)
 if str(root) in p.get('output',''):jobs[jid]={'state':state,'output':p['output']}
live={};cards={};peak_cards={};events=[];peak_gpu=peak_live=0
for at,jid,state,detail in db.execute('SELECT at,job,state,detail FROM events ORDER BY at,rowid'):
 if jid not in jobs:continue
 detail=json.loads(detail or 'null')
 if state=='running':live[jid]=len(detail.get('gpus',[]));cards[jid]=detail.get('gpus',[])
 elif state=='resources_released':live[jid]=detail['after']['gpu_count'];cards[jid]=[]
 elif state in ('succeeded','failed','cancelled','interrupted'):live.pop(jid,None);cards.pop(jid,None)
 for gpu,count in Counter(g for groups in cards.values() for g in groups).items():peak_cards[gpu]=max(peak_cards.get(gpu,0),count)
 peak_gpu=max(peak_gpu,sum(live.values()));peak_live=max(peak_live,len(live))
 events.append({'at':at,'job':jid,'state':state,'output':jobs[jid]['output'],'gpu_slots_live':sum(live.values())})
report['resources']={'peak_gpu_slots':peak_gpu,'peak_gpu_slots_per_device':peak_cards,'peak_live_evaluations':peak_live,'jobs':jobs,'events':events}
report['extra_diagnostics']={}
for relative in ('final-resource-audit.json','iteration-parent-comparison.json','tp-pdn-diagnostic/diagnostic.json','routing-hook-preflight/summary.json','routing-active-hooks/budget.json','routing-hook-audit-stop.json','source-sharing.json','space-sharing-old-prefixes.json','failed-reference-compaction.json'):
 p=root/relative
 if p.exists():report['extra_diagnostics'][relative]=read(p)
report['model_call_receipts']=[str(p) for p in sorted(root.glob('**/model-*/result.json'))]
print(json.dumps(report,indent=2,allow_nan=False))
