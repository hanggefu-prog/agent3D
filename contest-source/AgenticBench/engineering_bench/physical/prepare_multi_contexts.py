from layout import layout_path
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import time

from local_query import query, reusable_exception
from multi_context import create_context
from target_delay import parse_delays
from cts_reference_queue import admission
from engineering_bench.prepare_development import verify_release
from engineering_bench.protocol import write_new, sha, now


def main():
    p = argparse.ArgumentParser()
    for k in ('root', 'experiment', 'output'):
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--reuse-root', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source, image = verify_release(layout_path(args.root, 'release'))
    platform = layout_path(args.root, 'platform')
    with (args.experiment / 'engineering-physical.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for case in args.cases:
            output = args.output / case
            output.mkdir()
            while not admission(output):
                time.sleep(30)
            try:
                run = layout_path(args.root, 'reference_root') / case / 'run'
                sdc = (run / '60_opt/assembly/design.sdc').read_text()
                for line in sdc.splitlines():
                    if re.match(r'\s*set_(?:false_path|multicycle_path|max_delay|min_delay|disable_timing|case_analysis|clock_groups)\b', line):
                        if not any(x in line for x in ('-from [get_ports -quiet rst_ni]', '-to $latch_data', '-asynchronous -group $clk_name -group jtag_clock')):
                            raise ValueError('Unsupported frozen SDC exception: ' + line)
                features = args.experiment / 'local-context-v2' / case
                old = args.reuse_root / case if args.reuse_root else None
                old_excluded = old / 'exceptions/native/excluded-pins.tsv' if old else None
                receipt = json.loads((old / 'exceptions/result.json').read_text()) if old and (old / 'exceptions/result.json').exists() else {}
                if old_excluded and old_excluded.exists() and reusable_exception(receipt, old_excluded):
                    excluded = old_excluded
                    write_new(output / 'exception-reuse.json', {'path': str(excluded), 'sha256': sha(excluded)})
                else:
                    excluded = query(source=source, platform=platform, run=run,
                        measurement_root=features / 'reference-measurement', image=image, output=output / 'exceptions', mode='exceptions')
                context = create_context(run, args.experiment / 'stage-reference-v1' / case, features, platform, excluded)
                context['seed'] = json.loads((args.experiment / 'stage-reference-v1' / case / 'replay/input-replay.json').read_text())['seed']
                write_new(output / 'premeasurement-context.json', context)
                old_delays = old / 'delays/native/target-delays.tsv' if old else None
                if old and (old / 'delays/result.json').exists() and old_delays.exists():
                    receipt = json.loads((old / 'delays/result.json').read_text())
                    previous = json.loads((old / 'premeasurement-context.json').read_text())
                    for key in ('target_nets', 'driver_input_pins', 'reference_input_slew_ns'):
                        if previous[key] != context[key]:
                            raise ValueError('Target selection changed; cannot reuse old delay measurement')
                    if receipt['status'] != 'succeeded' or receipt['result_sha256'] != sha(old_delays):
                        raise ValueError('Cannot reuse changed delay evidence')
                    (output / 'delays/native').mkdir(parents=True)
                    delays = output / 'delays/native/target-delays.tsv'
                    shutil.copy2(old_delays, delays)
                    write_new(output / 'delay-reuse.json', {'path': str(old_delays), 'sha256': sha(delays)})
                else:
                    delays = query(source=source, platform=platform, run=run,
                        measurement_root=features / 'reference-measurement', image=image, output=output / 'delays', mode='delays', context=context)
                values = parse_delays(delays, context)
                context['baseline_timing'] = {n: {'receiver_delays_ns': v} for n, v in values.items()}
                context['baseline_delay_pending_native_measurement'] = False
                context['baseline_delay_sha256'] = sha(delays)
                write_new(output / 'multi-context.json', context)
                row = {'status': 'current_context_ready', 'case': case, 'targets': len(context['target_nets']),
                    'spare_sites': len(context['spare_site_ids']), 'context_sha256': sha(output / 'multi-context.json')}
            except Exception as exc:
                row = {'status': 'failed', 'case': case, 'error': repr(exc)}
            row.update(completed_at=now(), candidate_finals=0)
            write_new(output / 'result.json', row)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
