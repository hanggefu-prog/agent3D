"""Bounded host launcher for the protected external physical adapter."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from cts_measure import sha, MEASUREMENT_ENDPOINTS


def bind(path, target, readonly=True):
    path = Path(path).resolve(strict=True)
    if any(x in str(path) for x in (',', '\n', '\0')):
        raise ValueError('Unsafe mount path')
    return ['--mount', 'type=bind,src={},dst={},readonly={}'.format(path, target, str(readonly).lower())]


def launch(args):
    release_path = Path(args.release).resolve(strict=True)
    release = json.loads(release_path.read_text())
    audit = release['runtime']['equivalence']
    if sha(audit['path']) != audit['sha256']:
        raise ValueError('Runtime equivalence evidence changed')
    image = release['runtime']['actual_image']
    if subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip() != image:
        raise ValueError('Installed image changed')
    # No GPU and one 8-CPU/32-GiB physical run per launcher. Refuse low memory or
    # disk before dispatch rather than changing resource settings or other jobs.
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(info['MemAvailable'].split()[0]) < 44 * 1024 * 1024:
        raise ValueError('Insufficient free host memory for 32-GiB physical slot')
    output = Path(args.output).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 25 * 1024**3:
        raise ValueError('Insufficient disk for independent STA')
    output.mkdir()
    native = output / 'native'
    native.mkdir()
    source = Path(release['development_source'])
    platform = Path(args.platform).resolve(strict=True)
    run = Path(args.run).resolve(strict=True)
    odbs = {d: Path(getattr(args, d + '_odb')).resolve(strict=True) for d in ('bottom', 'top')}
    name = 'o3db-engineering-sta-' + uuid.uuid4().hex[:16]
    cmd = ['docker', 'run', '--rm', '--name', name, '--read-only', '--network', 'none',
           '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '8',
           '--memory', '32g', '--pids-limit', '2048', '--tmpfs', '/tmp:rw,size=2g',
           '--user', '{}:{}'.format(os.getuid(), os.getgid())]
    mounts = [(source, '/physical', True), (Path(__file__).parent, '/adapter', True),
              (run, '/input', True), (platform, '/platform', True), (native, '/out', False)]
    mounts += [(p, '/candidate/' + d + '.odb', True) for d, p in odbs.items()]
    extra = getattr(args, 'additional_hbt_links', None)
    if extra:
        mounts += [(Path(extra), '/trusted-additional-links.json', True)]
    for p, target, ro in mounts:
        cmd += bind(p, target, ro)
    for env in ('HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=/physical', 'OPEN3DBENCH_THREADS=8'):
        cmd += ['--env', env]
    cmd += ['--entrypoint', 'python3', image, '/adapter/cts_measure.py',
            '--flow-source', '/physical', '--checkpoint', '/input/60_opt/checkpoint.json',
            '--platform', '/platform', '--bottom-odb', '/candidate/bottom.odb',
            '--top-odb', '/candidate/top.odb', '--output', '/out/measurement',
            '--measurement-mode', args.measurement_mode]
    if extra:
        cmd += ['--additional-hbt-links', '/trusted-additional-links.json']
    dispatch = dict(kind='engineering-joint-sta-diagnostic-v1', command=cmd, image=image,
        release_sha256=sha(release_path), runtime_audit_sha256=audit['sha256'],
        input_odb_sha256={d: sha(p) for d, p in odbs.items()},
        adapter_sha256=sha(Path(__file__).with_name('cts_measure.py')),
        measurement_mode=args.measurement_mode, endpoint=MEASUREMENT_ENDPOINTS[args.measurement_mode],
        model_calls=0, candidate_finals=0)
    (output / 'dispatch.json').write_text(json.dumps(dispatch, indent=2) + '\n')
    start = time.monotonic()
    code = None
    try:
        with (output / 'console.log').open('x') as log:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
        code = p.returncode
        result = json.loads((native / 'measurement/result.json').read_text())
        if code or result['status'] != 'succeeded':
            raise ValueError('Native joint STA failed; inspect console and native logs')
        result = dict(status='succeeded', native_result_sha256=sha(native / 'measurement/result.json'))
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            subprocess.run(['docker', 'stop', '--time', '5', name], capture_output=True, timeout=20, check=False)
        result = dict(status='failed', error=str(exc))
    result.update(exit_code=code, duration_seconds=time.monotonic() - start,
                  candidate_finals=0, model_calls=0, task_qualification=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'succeeded' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('release', 'platform', 'run', 'bottom_odb', 'top_odb', 'output'):
        parser.add_argument('--' + key.replace('_', '-'), required=True)
    parser.add_argument('--measurement-mode', choices=tuple(MEASUREMENT_ENDPOINTS), default='skew_sta')
    raise SystemExit(launch(parser.parse_args()))
