"""Qualify the task adapters without a candidate or a model call."""
import argparse
import fcntl
import time

from pathlib import Path
from layout import layout_path
from cts_reference_queue import admission
from engineering_bench.prepare_development import verify_release
from engineering_bench.protocol import write_new, now
from engineering_bench.tasks import TASK_IDS, load_task
from engineering_bench.scoring import CURRENT_SCORING_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'experiment', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--task', choices=TASK_IDS, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--context-root', type=Path)
    p.add_argument('--scoring-version', type=int, choices=(1, 2), default=CURRENT_SCORING_VERSION,
                   help='1 (default): frozen scoring; 2: explicitly opt in to evaluation-v2.json')
    args = p.parse_args()
    task = load_task(args.task)
    if len(set(args.cases)) != len(args.cases) or not set(args.cases) <= set(task.data['cases']):
        p.error('Invalid or repeated cases')
    if args.task == 'multi_hbt_fanout' and args.context_root is None:
        p.error('T3 requires --context-root')
    source, image = verify_release(layout_path(args.root, 'release'))
    platform = layout_path(args.root, 'platform')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with (args.experiment / 'engineering-physical.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for case in args.cases:
            while not admission(args.output):
                time.sleep(30)
            try:
                if args.task == 'cross_die_cts':
                    from cts_final_queue import qualify_reference
                    qualify_reference(task, case, args.experiment / 'stage-reference-v1' / case,
                        experiment=args.experiment, root=args.root, output=args.output / case,
                        platform=platform, image=image, source=source, sealed={'changed_paths': []},
                        scoring_version=args.scoring_version)
                elif args.task == 'clock_compensation':
                    from clock_final_queue import qualify
                    qualify(task, case, args, source, image, platform, args.output / case, args.scoring_version)
                else:
                    from multi_final_queue import qualify
                    qualify(task, case, args, source, image, platform, args.output / case, args.scoring_version)
                row = {'case': case, 'status': 'qualified'}
            except Exception as exc:
                row = {'case': case, 'status': 'qualification_failed', 'error': repr(exc)}
            row.update(completed_at=now(), candidate_finals=0, model_calls=0)
            rows.append(row)
            write_new(args.output / ('status-' + case + '.json'), row)
    write_new(args.output / 'result.json', {'cases': rows, 'candidate_finals': 0, 'model_calls': 0})
    return 0 if all(row['status'] == 'qualified' for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
