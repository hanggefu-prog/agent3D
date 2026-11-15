"""One bounded CTS reference job per host; no candidate dispatch or model calls.

Host orchestration communicates with the physical source only by containers and
artifacts. Each successful stage is independently read back, normalized and
measured. Failed cases are retained and never retried by this queue.
"""
from layout import layout_path
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from cts_measure import sha
from measure_host import bind, launch as measure


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def admission(output):
    fields = dict(row.split(':', 1) for row in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) >= 44 * 1024 * 1024 and shutil.disk_usage(output).free >= 35 * 1024**3


def reference_case(case, *, root, experiment, output, release_path, source, image, platform, role='reference', execution_image=None):
    from engineering_bench.native_snapshot import snapshot
    from engineering_bench.legalization import normalize, load_snapshot, ports_geometry, is_fixed
    from engineering_bench.clock_equivalence import compare

    run = layout_path(root, 'reference_root') / case / 'run'
    original = {d: experiment / 'native-audit-v3' / case / '60_opt' / d for d in ('bottom', 'top')}
    output.mkdir()
    # All support and reference mounts are RO. Only this fresh output is RW.
    if role not in ('reference', 'candidate'):
        raise ValueError('Unknown replay role')
    name = 'o3db-engineering-cts-' + role + '-' + uuid.uuid4().hex[:12]
    command = ['docker', 'run', '--rm', '--name', name, '--read-only', '--network', 'none',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '8',
               '--memory', '32g', '--pids-limit', '2048', '--tmpfs', '/tmp:rw,size=2g',
               '--user', '{}:{}'.format(os.getuid(), os.getgid())]
    for host, target, ro in [(source, '/physical', True), (run, '/input', True),
                              (platform, '/platform', True), (Path(__file__).parent, '/adapter', True),
                              (output, '/out', False)]:
        command += bind(host, target, ro)
    for env in ('HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=/physical'):
        command += ['--env', env]
    command += ['--entrypoint', 'python3', execution_image or image, '/adapter/cts_reference_stage.py',
                '--source', '/physical', '--run', '/input', '--platform', '/platform', '--output', '/out/replay', '--role', role]
    write_new(output / 'dispatch.json', {'role': role, 'case': case, 'started_at': now(),
              'image': image, 'release_sha256': sha(release_path), 'command': command,
              'model_calls': 0, 'candidate_finals': int(role == 'candidate'), 'task_qualification': False})
    try:
        with (output / 'cts.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=False)
        replay = json.loads((output / 'replay/result.json').read_text())
        if result.returncode or replay['status'] != 'succeeded':
            raise ValueError('Reference CTS replay failed; inspect cts.log and replay/result.json')
        normalized, raw, displacement = {}, {}, {}
        for die in ('bottom', 'top'):
            odb = output / 'replay/70_cts' / die / 'design.odb'
            raw[die] = output / ('raw-' + die)
            observed = snapshot(run=run, platform=platform, image=image, stage='70_cts', die=die,
                output=raw[die], clock_sdc=run / '60_opt' / die / 'design.sdc', odb_path=odb)
            if observed['status'] != 'succeeded':
                raise ValueError('Reference independent snapshot failed: ' + die)
            norm = output / ('normalized-' + die)
            receipt = normalize(original_snapshot=original[die], candidate_snapshot=raw[die],
                candidate_odb=odb, output=norm, image=image)
            if receipt['status'] != 'succeeded':
                raise ValueError('Reference protected normalization failed: ' + die)
            normalized[die] = output / ('readback-' + die)
            receipt = snapshot(run=run, platform=platform, image=image, stage='70_cts', die=die,
                output=normalized[die], clock_sdc=run / '60_opt' / die / 'design.sdc',
                odb_path=norm / 'native/design.odb')
            if receipt['status'] != 'succeeded':
                raise ValueError('Reference normalized readback failed: ' + die)
            if ports_geometry(original[die] / 'ports.tsv') != ports_geometry(normalized[die] / 'ports.tsv'):
                raise ValueError('Reference fixed port geometry changed')
            _, _, before = load_snapshot(original[die])
            _, _, after = load_snapshot(normalized[die])
            moved = 0
            for inst, cell in before.items():
                if cell['clock_cell']:
                    continue
                changed = any(cell[k] != after[inst][k] for k in ('x_dbu', 'y_dbu', 'orient'))
                if changed and is_fixed(cell):
                    raise ValueError('Fixed reference macro/HBT moved')
                moved += int(changed)
            displacement[die] = moved
        proof = compare(original, normalized, run / '60_opt/hbt.csv')
        args = argparse.Namespace(release=str(release_path), platform=str(platform), run=str(run),
            bottom_odb=str(output / 'normalized-bottom/native/design.odb'),
            top_odb=str(output / 'normalized-top/native/design.odb'),
            output=str(output / 'independent-skew'), measurement_mode='skew')
        if measure(args):
            raise ValueError('Reference independent skew measurement failed')
        measured = json.loads((output / 'independent-skew/native/measurement/result.json').read_text())
        row = {'case': case, 'status': 'stage_' + role + '_measured', 'clock_skew_ns': measured['clock_skew_ns'],
               'physical_equivalence': proof, 'displacement': displacement,
               'measurement_sha256': sha(output / 'independent-skew/native/measurement/result.json'),
               'remaining': ['complete task capability qualification', 'candidate final', 'independent judge']}
    except Exception as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            subprocess.run(['docker', 'stop', '--time', '5', name], capture_output=True, timeout=20, check=False)
        row = {'case': case, 'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)}
    row.update(completed_at=now(), candidate_finals=int(role == 'candidate'), model_calls=0, task_qualification=False)
    write_new(output / 'result.json', row)
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'experiment', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--cases', nargs='+', choices=('bp_fe', 'bp', 'ariane133', 'swerv_wrapper'), required=True)
    args = p.parse_args()
    if len(args.cases) != len(set(args.cases)):
        p.error('Repeated cases are forbidden')
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / 'queue.json', {'role': 'reference', 'cases': args.cases, 'started_at': now(),
        'pid': os.getpid(), 'model_calls': 0, 'candidate_finals': 0, 'automatic_candidate_dispatch': False})
    from engineering_bench.prepare_development import verify_release
    release_path = layout_path(args.root, 'release')
    source, image = verify_release(release_path)
    platform = layout_path(args.root, 'platform')
    rows = []
    with (args.experiment / 'engineering-physical.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in args.cases:
            while not admission(args.output):
                print(json.dumps({'case': case, 'status': 'waiting_for_resources', 'time': now()}), flush=True)
                time.sleep(30)
            rows.append(reference_case(case, root=args.root, experiment=args.experiment,
                output=args.output / case, release_path=release_path, source=source, image=image, platform=platform))
    write_new(args.output / 'result.json', {'cases': rows, 'completed_at': now(),
              'candidate_finals': 0, 'model_calls': 0, 'task_qualification': False})
    return 0 if all(r['status'] == 'stage_reference_measured' for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
