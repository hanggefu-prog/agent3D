"""Protected container entry: replay the unchanged CTS reference from post-OPT.

This is a fixed-checkpoint reference, not normal full-flow resume or a candidate
final. No upstream signatures are rewritten and no downstream flow is invoked.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from cts_measure import sha


def verify_prefix(run):
    """Verify every retained artifact of the immutable prefix before reuse."""
    run = Path(run).resolve(strict=True)
    folders = ('00_input', '10_floorplan', '15_timing_extract', '20_partition',
               '30_placement', '40_pdn', '50_hbt', '60_opt')
    records = {}
    for name in folders:
        stage = run / name
        manifest = stage / 'stage_manifest.json'
        if not manifest.exists() and name == '15_timing_extract':
            continue  # Optional older prefix stage; never pretend to execute it.
        data = json.loads(manifest.read_text())
        artifacts = data.get('artifacts')
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError('Missing immutable stage artifact inventory: ' + name)
        verified = {}
        for key, row in artifacts.items():
            raw = row['path']
            if Path(raw).is_absolute() or '..' in Path(raw).parts:
                raise ValueError('Unsafe manifest artifact path')
            path = (stage / raw).resolve(strict=True)
            path.relative_to(stage.resolve())
            if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('Changed prefix artifact: ' + name + '/' + key)
            verified[key] = row['sha256']
        records[name] = {'manifest_sha256': sha(manifest), 'artifacts': verified}
    return records


def cts_arguments(config, *, input_root, platform, output):
    cts = config['cts']
    if cts.get('algorithm') != 'openroad' or config.get('pdk') != 'ng45':
        raise ValueError('This reference adapter requires the pinned ng45 OpenROAD case')
    if cts.get('repair_hold') is not False:
        raise ValueError('Stage-only reference must not enable later data hold repair')
    if config.get('partition', {}).get('seed') not in (1001, 1002, 1003):
        raise ValueError('Unexpected frozen benchmark seed')
    threads = config.get('threads')
    if type(threads) is not int or threads <= 0:
        raise ValueError('Missing frozen physical thread count')
    flags = ['--checkpoint', str(Path(input_root) / '60_opt/checkpoint.json'),
             '--platform-dir', str(platform), '--output-dir', str(output),
             '--openroad', '/usr/local/bin/openroad',
             '--bottom-buffer', 'BUF_X4_bottom', '--top-buffer', 'BUF_X4_top']
    for key, flag in [('target_skew_ns', '--target-skew-ns'), ('cluster_size', '--cluster-size'),
                      ('cluster_diameter_um', '--cluster-diameter'),
                      ('delay_buffer_derate', '--delay-buffer-derate')]:
        flags += [flag, str(cts[key])]
    return flags, threads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'run', 'platform', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--role', choices=('reference', 'candidate'), default='reference')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'kind': 'engineering-cts-stage-replay-v1', 'role': args.role, 'candidate_finals': int(args.role == 'candidate'),
              'model_calls': 0, 'task_qualification': False}
    try:
        import yaml
        config_path = Path(args.run) / 'config.yaml'
        config = yaml.safe_load(config_path.read_text())
        prefix = verify_prefix(args.run)
        flags, threads = cts_arguments(config, input_root=args.run, platform=args.platform,
                                       output=output / '70_cts')
        source = Path(args.source).resolve(strict=True)
        sys.path.insert(0, str(source))
        from open3dbench.backend.three_d_cts import build_parser, run_cts
        from open3dbench.backend.opt_handoff import load_cts_handoff
        handoff = load_cts_handoff(Path(args.run) / '60_opt/checkpoint.json')
        os.environ['OPEN3DBENCH_THREADS'] = str(threads)
        receipt = {'role': args.role, 'config_sha256': sha(config_path), 'prefix': prefix,
                   'frozen_checkpoint_sha256': sha(handoff.checkpoint), 'flags': flags,
                   'physical_threads': threads, 'seed': config['partition']['seed'],
                   'source_sha256': {str(f.relative_to(source)): sha(f)
                                     for f in sorted((source / 'open3dbench').rglob('*.py'))},
                   'native_binary_sha256': sha('/usr/local/bin/openroad'),
                   'reused_stages': list(prefix), 'executed_stages': ['cts'],
                   'not_executed': ['synthesis', 'partition', 'placement', 'opt', 'grt', 'drt', 'rcx', 'thermal']}
        (output / 'input-replay.json').write_text(json.dumps(receipt, indent=2) + '\n')
        checkpoint = run_cts(build_parser().parse_args(flags))
        if any(sha(Path(args.run) / name / 'stage_manifest.json') != row['manifest_sha256']
               for name, row in prefix.items()) or sha(config_path) != receipt['config_sha256']:
            raise ValueError('Frozen prefix metadata changed')
        result.update(status='succeeded', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
    except Exception as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    result['duration_seconds'] = time.monotonic() - started
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'succeeded' else 1


if __name__ == '__main__':
    raise SystemExit(main())
