"""Run isolated integration diagnostics under the existing resource broker."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--queue', type=Path, required=True)
parser.add_argument('--host', default='rafa1-gcr')
parser.add_argument('--source-name', default='integration-source')
parser.add_argument('--output-name', default='integration-native')
parser.add_argument('--release', required=True)
args = parser.parse_args()
root = args.root.resolve()
source = root / args.source_name
sys.path.insert(0, str(source / 'AgenticBench'))
from grt_bench.queue import Queue
from grt_bench.runtime import admission, docker_command, run_container, UncertainExecution
from grt_bench.common import write

def run(case, mode):
    label = case + '-' + mode
    out = root / args.output_name / label
    out.mkdir(parents=True, exist_ok=False)
    queue = Queue(args.queue)
    resources = dict(cpu=4, memory_gb=24, storage_gb=14, gpu_count=0, build_slots=0)
    job = queue.submit({'kind':'integration-test', 'case':label, 'source_release':args.release,
                        'output':str(out)}, resources)
    lease = None
    deadline = time.monotonic() + 1800
    while lease is None and time.monotonic() < deadline:
        lease = queue.claim(args.host, kind='integration-test', job_id=job)
        if lease is None:
            time.sleep(3)
    if lease is None:
        queue.cancel(job)
        raise RuntimeError('Capacity unavailable for ' + label)
    heartbeat = lambda: queue.heartbeat(job, lease['token'])
    write(out / 'lease.json', {key:value for key,value in lease.items() if key != 'token'})
    try:
        admission(root, resources, [])
        qualified = root / 'hbt-release' / case
        image = json.loads((qualified / 'input-provenance.json').read_text())['image']
        results = out / 'results'
        results.mkdir()
        name = 'grt-integration-' + job[:20]
        command = docker_command(image, name, resources, [],
            [(source, '/workspace', True), (source / 'AgenticBench', '/trusted', True),
             (qualified / 'input', '/frozen', True), (results, '/output', False),
             (root / 'integration-native-cell.py', '/check.py', True)], ['/check.py', mode])
        runtime = run_container(command, out / 'execution', name, timeout=7200, heartbeat=heartbeat)
        receipt = json.loads((results / 'qualification.json').read_text())
        assert receipt['status'] == 'verified'
        queue.finish(job, lease['token'], {'qualification':str(results / 'qualification.json'), **runtime}, True)
        return {'case':label, **runtime, 'status':'verified'}
    except UncertainExecution:
        raise
    except BaseException as error:
        queue.finish(job, lease['token'], {'error':str(error)}, False)
        raise

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(run, case, mode):(case, mode)
               for case in ('bp_fe-ng45', 'bp_fe-asap7') for mode in ('openroad', 'none')}
    results = []
    for future in as_completed(futures):
        case, mode = futures[future]
        try:
            result = future.result()
        except Exception as error:
            result = {'case':case, 'mode':mode, 'error':str(error), 'status':'failed'}
        results.append(result)
        print(json.dumps(result), flush=True)
    write(root / args.output_name / 'summary.json', results)
    if any(row['status'] != 'verified' for row in results):
        sys.exit(1)
