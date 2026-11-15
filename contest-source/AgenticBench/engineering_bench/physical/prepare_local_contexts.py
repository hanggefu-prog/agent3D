"""Qualify local current-state interfaces from frozen, legal post-CTS baselines."""
from layout import layout_path
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from cts_reference_queue import admission
from measure_host import launch, bind
from engineering_bench.prepare_development import verify_release
from engineering_bench.protocol import write_new, now, sha
from local_context import clock_context


def feature_container(*, root, experiment, case, output, image, source, platform, reuse_measurement=None):
    ref = experiment / 'stage-reference-v1' / case
    run = layout_path(root, 'reference_root') / case / 'run'
    measurement = output / 'reference-measurement'
    args = argparse.Namespace(release=str(layout_path(root, 'release')),
        platform=str(platform), run=str(run), output=str(measurement), measurement_mode='skew_sta',
        bottom_odb=str(ref / 'normalized-bottom/native/design.odb'), top_odb=str(ref / 'normalized-top/native/design.odb'))
    if reuse_measurement:
        from cts_final_queue import check_artifacts
        old = Path(reuse_measurement) / 'native/measurement'
        payload = json.loads((old / 'result.json').read_text())
        if payload['status'] != 'succeeded' or payload['measurement_mode'] != 'skew_sta':
            raise ValueError('Cannot reuse unsuccessful/wrong-mode measurement')
        check_artifacts(old, payload)
        dispatch = json.loads((Path(reuse_measurement) / 'dispatch.json').read_text())
        if any(dispatch['input_odb_sha256'][d] != sha(ref / ('normalized-' + d) / 'native/design.odb') for d in ('bottom', 'top')):
            raise ValueError('Reused measurement input differs')
        shutil.copytree(reuse_measurement, measurement)
        write_new(output / 'measurement-reuse.json', {'source': str(reuse_measurement), 'sha256': sha(old / 'result.json')})
    elif launch(args):
        raise ValueError('Current-state reference STA failed')
    # Replay writes native reports. Use a separate copy, never alter the hashed baseline.
    work = output / 'feature-work'
    shutil.copytree(measurement / 'native', work)
    name = 'o3db-engineering-context-' + uuid.uuid4().hex[:12]
    command = ['docker', 'run', '--rm', '--name', name, '--read-only', '--network', 'none',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '8', '--memory', '32g',
        '--pids-limit', '2048', '--tmpfs', '/tmp:rw,size=2g', '--user', str(os.getuid()) + ':' + str(os.getgid())]
    for host, target, ro in [(source, '/physical', True), (run, '/input', True), (platform, '/platform', True),
        (Path(__file__).parent, '/adapter', True), (work, '/out', False), (output, '/features-root', False),
        (ref / 'normalized-bottom/native/design.odb', '/candidate/bottom.odb', True),
        (ref / 'normalized-top/native/design.odb', '/candidate/top.odb', True)]:
        command += bind(host, target, ro)
    command += ['--env', 'HOME=/tmp', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONPATH=/physical',
        '--env', 'OPEN3DBENCH_THREADS=8', '--entrypoint', 'python3', image, '/adapter/local_context_native.py',
        '--flow', '/physical', '--platform', '/platform', '--bottom-odb', '/candidate/bottom.odb',
        '--top-odb', '/candidate/top.odb', '--measurement', '/out/measurement', '--output', '/features-root/features']
    write_new(output / 'feature-dispatch.json', {'command': command, 'started_at': now(), 'image': image, 'candidate_finals': 0})
    with (output / 'feature-console.log').open('x') as log:
        try:
            p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '--time', '5', name], capture_output=True, timeout=20)
            raise
    if p.returncode:
        raise ValueError('Native feature extraction failed; inspect feature-console.log')
    context = clock_context(ref, output / 'features', platform)
    context['seed'] = json.loads((ref / 'replay/input-replay.json').read_text())['seed']
    return context


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'experiment', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--reuse-measurement-root', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source, image = verify_release(layout_path(args.root, 'release'))
    platform = layout_path(args.root, 'platform')
    with (args.experiment / 'engineering-physical.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for case in args.cases:
            while not admission(args.output):
                print(json.dumps({'case': case, 'status': 'waiting_for_resources', 'time': now()}), flush=True)
                time.sleep(30)
            output = args.output / case
            output.mkdir()
            try:
                context = feature_container(root=args.root, experiment=args.experiment, case=case,
                    output=output, image=image, source=source, platform=platform,
                    reuse_measurement=args.reuse_measurement_root / case / 'reference-measurement' if args.reuse_measurement_root else None)
                write_new(output / 'clock-context.json', context)
                row = {'status': 'current_context_ready', 'case': case, 'actions': len(context['actions']),
                    'branches': len(context['branches']), 'paths': len(context['paths']),
                    'context_sha256': sha(output / 'clock-context.json')}
            except Exception as exc:
                row = {'status': 'failed', 'case': case, 'error': type(exc).__name__ + ': ' + str(exc)}
            row.update(completed_at=now(), candidate_finals=0)
            write_new(output / 'result.json', row)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
