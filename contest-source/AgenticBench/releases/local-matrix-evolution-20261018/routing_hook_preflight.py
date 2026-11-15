from pathlib import Path
import os,sys,runpy,json,yaml,shutil
H=Path(os.environ['AGENTIC_AUDIT_PARENT']).expanduser().resolve();R=H/'local-matrix-audit-20261018';S=H/'source-local-phases-e951f6df';N=R/'routing-hook-preflight';N.mkdir(exist_ok=False)
for name in ('inputs.json','runtime.json'):shutil.copy2(R/name,N/name)
sys.argv=['audit-helper','--source',str(S),'--root',str(N),'--codex',os.environ['CODEX_BIN']]
a=runpy.run_path(str(Path(__file__).with_name('audit_runner.py')),run_name='audit_helper')
from local_bench.model import load_task
from local_bench.program_bridge import materialize_program
from grt_bench.common import read,write
spec=load_task(S/'AgenticBench/local_tasks/global_routing_maze_cost');write(N/'candidate/candidate.json',materialize_program(spec,spec.program_template.read_text(),S))
c=yaml.safe_load(spec.base_config.read_text());c['design']='bp_fe';c['threads']=16;c['routing']['threads']=16;c['routing']['resource_reduction']=0.5
for st,k in [('pins','io'),('partition','partition'),('placement','placement'),('routing','routing')]:c.setdefault(st,{})['seed']={'io':1,'partition':1001,'placement':1001,'routing':-1}[k]
base=N/'base.yaml';base.write_text(yaml.safe_dump(c))
t=a['AuditTransport'](**a['runtime']);evaluate=t.prepare_candidate(spec.root,N/'candidate',S,N/'build')
r=evaluate(spec.root,N/'candidate',S,N/'cell/result',baseline_run=R/'inputs/bp_fe-ng45/s1/run',reference_run=None,base_config=base,seed_schedule={'io':1,'partition':1001,'placement':1001,'routing':-1},create_routing_reference=True,timeout_seconds=3600,keep_workspace=False,keep_runtime=False,compact_artifacts=False)
write(N/'summary.json',r.to_dict());print(json.dumps(r.metrics),flush=True)
