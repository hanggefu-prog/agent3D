"""Host-only creation of a separate timing/overflow task; no archive relabelling."""
import argparse
import copy
from pathlib import Path
from . import TIMING_ENDPOINT
from .common import read, write, evaluator_identity
from .contract import validate, TIMING_METRICS, TIMING_OBJECTIVES, DEFAULT_TIMING_SCORING
from .optimization import TIMING_OBJECTIVE


def timing_task(task, task_id):
    validate(task)
    if task_id == task['id']:
        raise ValueError('Use a distinct task ID; never rewrite an existing campaign')
    result = copy.deepcopy(task)
    result.update(id=task_id, schema_version=3, endpoint=TIMING_ENDPOINT,
                  evaluator_sha256=evaluator_identity(), metrics=list(TIMING_METRICS),
                  objectives=list(TIMING_OBJECTIVES), scoring=copy.deepcopy(DEFAULT_TIMING_SCORING))
    result['normalization_floors'] = {k: result['normalization_floors'][k] for k in TIMING_METRICS}
    for key in ('activity', 'ambient_k', 'grid_size'):
        result.pop(key, None)
    if 'optimization' in result:
        spec=result['optimization']
        # Preserve the exact front matter, including optional narrower file scope.
        lines=spec['prompt'].splitlines(); end=lines.index('---',1)
        body=(Path(__file__).resolve().parents[1]/'global_tasks/lol_post_grt_timing/instruction.md').read_text().split('---',2)[2]
        spec['prompt']='\n'.join(lines[:end+1])+body
        spec['objective']=TIMING_OBJECTIVE
    validate(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--id',required=True)
    args=parser.parse_args()
    write(args.output,timing_task(read(args.source),args.id))


if __name__=='__main__':main()
