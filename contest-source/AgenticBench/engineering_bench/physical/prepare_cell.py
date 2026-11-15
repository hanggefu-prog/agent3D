"""Prepare one immutable case/seed through independent engineering qualification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from layout import layout_path
from engineering_bench.protocol import write_new, sha, now
from engineering_bench.prepare_development import verify_release
from engineering_bench.native_snapshot import snapshot
from engineering_bench.tasks import TASK_IDS, load_task
from cts_reference_queue import reference_case
from prepare_local_contexts import feature_container


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--experiment', type=Path, required=True)
    a = p.parse_args()
    a.experiment.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    times = {}
    source, image = verify_release(layout_path(a.root, 'release'))
    platform = layout_path(a.root, 'platform')
    run = layout_path(a.root, 'reference_root') / a.case / 'run'
    def stage(name, call):
        before = time.monotonic()
        print(json.dumps({'stage':name,'status':'running','at':now()}),flush=True)
        result = call()
        times[name] = time.monotonic()-before
        write_new(a.experiment/(name+'-timing.json'),{'wall_seconds':times[name],'completed_at':now()})
        return result
    def audit():
        for die in ('bottom','top'):
            row = snapshot(run=run,platform=platform,image=image,stage='60_opt',die=die,
                clock_sdc=run/'60_opt'/die/'design.sdc',output=a.experiment/'native-audit-v3'/a.case/'60_opt'/die)
            if row['status'] != 'succeeded': raise ValueError('Post-OPT input audit failed: '+die)
    try:
        stage('input-audit',audit)
        ref=a.experiment/'stage-reference-v1'/a.case
        ref.parent.mkdir()
        row=stage('reference-cts',lambda:reference_case(a.case,root=a.root,experiment=a.experiment,
            output=ref,release_path=layout_path(a.root,'release'),source=source,image=image,platform=platform))
        if row['status'] != 'stage_reference_measured': raise ValueError('Reference CTS qualification failed')
        features=a.experiment/'local-context-v2'/a.case
        features.mkdir(parents=True)
        context=stage('clock-context',lambda:feature_container(root=a.root,experiment=a.experiment,case=a.case,
            output=features,image=image,source=source,platform=platform))
        write_new(features/'clock-context.json',context)
        write_new(features/'result.json',{'status':'current_context_ready','context_sha256':sha(features/'clock-context.json')})
        def multi():
            subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_multi_contexts.py')),
                '--root',str(a.root),'--experiment',str(a.experiment),'--output',str(a.experiment/'multi-context'),
                '--cases',a.case],check=True)
            row=json.loads((a.experiment/'multi-context'/a.case/'result.json').read_text())
            if row['status'] != 'current_context_ready': raise ValueError('Multi-HBT context failed: '+str(row))
        stage('multi-context',multi)
        from cts_final_queue import qualify_reference
        from clock_final_queue import qualify as qualify_clock
        from multi_final_queue import qualify as qualify_multi
        args=SimpleNamespace(root=a.root,experiment=a.experiment,context_root=a.experiment/'multi-context')
        for name in TASK_IDS:
            task=load_task(name); target=a.experiment/'qualifications'/name/a.case
            if name=='cross_die_cts':
                stage(name+'-qualification',lambda:qualify_reference(task,a.case,ref,experiment=a.experiment,
                    root=a.root,output=target,platform=platform,image=image,source=source,sealed={'changed_paths':[]}))
            elif name=='clock_compensation':
                stage(name+'-qualification',lambda:qualify_clock(task,a.case,args,source,image,platform,target))
            else:
                stage(name+'-qualification',lambda:qualify_multi(task,a.case,args,source,image,platform,target))
        result={'status':'qualified','case':a.case,'seed':context['seed'],'tasks':list(TASK_IDS)}
    except Exception as error:
        result={'status':'failed','case':a.case,'error':repr(error)}
    result.update(wall_seconds=time.monotonic()-start,stages_seconds=times,completed_at=now(),candidate_finals=0)
    write_new(a.experiment/'preparation-result.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['status']=='qualified' else 1


if __name__=='__main__':
    raise SystemExit(main())
