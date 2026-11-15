"""Run protected T3 fixtures once, separately from candidate submissions."""
from layout import layout_path
import argparse
import fcntl
import json
from pathlib import Path
import time

from multi_final_queue import qualify
from cts_reference_queue import admission
from engineering_bench.protocol import write_new, now
from engineering_bench.tasks import load_task
from engineering_bench.prepare_development import verify_release


def main():
    p = argparse.ArgumentParser()
    for k in ('root', 'experiment', 'output', 'context_root'):
        p.add_argument('--' + k.replace('_', '-'), type=Path, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    task = load_task('multi_hbt_fanout')
    source, image = verify_release(layout_path(args.root, 'release'))
    platform = layout_path(args.root, 'platform')
    rows = []
    with (args.experiment / 'engineering-physical.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for case in args.cases:
            while not admission(args.output):
                time.sleep(30)
            try:
                qualify(task, case, args, source, image, platform, args.output / case)
                row = {'case': case, 'status': 'qualified', 'completed_at': now(), 'candidate_finals': 0}
            except Exception as exc:
                row = {'case': case, 'status': 'qualification_failed', 'error': repr(exc), 'completed_at': now(), 'candidate_finals': 0}
            rows.append(row)
            write_new(args.output / ('status-' + case + '.json'), row)
            print(json.dumps(row), flush=True)
    write_new(args.output / 'result.json', {'cases': rows, 'candidate_finals': 0})


if __name__ == '__main__':
    main()
