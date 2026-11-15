"""Serial reference-diagnostic queue; no model calls or candidate finals."""
from layout import layout_path
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from cts_measure import MEASUREMENT_ENDPOINTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('root', 'experiment', 'output'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--measurement-mode', choices=tuple(MEASUREMENT_ENDPOINTS), default='skew_sta')
    args = p.parse_args()
    root, experiment, output = map(lambda x: Path(x).resolve(), (args.root, args.experiment, args.output))
    if len(args.cases) != len(set(args.cases)) or set(args.cases) - {'bp_fe', 'bp', 'ariane133', 'swerv_wrapper'}:
        p.error('Invalid or repeated case')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in args.cases:
        cmd = [sys.executable, str(Path(__file__).with_name('measure_host.py')),
            '--release', str(layout_path(root, 'release')),
            '--platform', str(layout_path(root, 'platform')),
            '--run', str(layout_path(root, 'reference_root') / case / 'run'),
            '--output', str(output / case), '--measurement-mode', args.measurement_mode]
        for die in ('bottom', 'top'):
            cmd += ['--' + die + '-odb', str(experiment / 'normalization-audit-v2' / case / die / 'native/design.odb')]
        with (output / (case + '.console.log')).open('x') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        row = dict(case=case, exit_code=result.returncode, status='succeeded' if result.returncode == 0 else 'failed')
        rows.append(row)
        print(json.dumps(row), flush=True)
    record = dict(cases=rows, measurement_mode=args.measurement_mode,
                  endpoint=MEASUREMENT_ENDPOINTS[args.measurement_mode],
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  model_calls=0, candidate_finals=0, task_qualification=False)
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    return 0 if all(r['exit_code'] == 0 for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
