"""Run the maintained Place-3D suite in the installed runtime."""
import json
from pathlib import Path
import sys
import time

root = Path(sys.argv[1]).resolve()
source = root / 'integration-source'
sys.path.insert(0, str(source / 'AgenticBench'))
from grt_bench.queue import Queue
from grt_bench.runtime import admission, docker_command, run_container, UncertainExecution
from grt_bench.common import write

queue = Queue(sys.argv[2])
out = root / 'integration-place'
out.mkdir(exist_ok=False)
resources = dict(cpu=4, memory_gb=16, storage_gb=2, gpu_count=0, build_slots=0)
job = queue.submit({'kind':'integration-test', 'case':'place-unit-suite', 'output':str(out)}, resources)
lease = None
deadline = time.monotonic() + 1800
while lease is None and time.monotonic() < deadline:
    lease = queue.claim('rafa1-gcr', kind='integration-test', job_id=job)
    if lease is None:
        time.sleep(3)
if lease is None:
    queue.cancel(job)
    raise RuntimeError('Capacity unavailable')
heartbeat = lambda: queue.heartbeat(job, lease['token'])
write(out / 'lease.json', {key:value for key,value in lease.items() if key != 'token'})
try:
    admission(root, resources, [])
    image = json.loads((root / 'hbt-release/bp_fe-ng45/input-provenance.json').read_text())['image']
    name = 'grt-integration-place-' + job[:16]
    script = ('import os,sys,unittest; os.chdir("/workspace/Place-3D"); '
              'sys.path.insert(0,"/workspace/Place-3D"); '
              'suite=unittest.defaultTestLoader.discover("unit_tests",pattern="test_*.py"); '
              'result=unittest.TextTestRunner(verbosity=1).run(suite); '
              'sys.exit(0 if result.wasSuccessful() else 1)')
    command = docker_command(image, name, resources, [],
              [(source, '/workspace', True), (source / 'AgenticBench', '/trusted', True)], ['-c', script])
    result = run_container(command, out / 'execution', name, timeout=1800, heartbeat=heartbeat)
    queue.finish(job, lease['token'], result, True)
    print(json.dumps(result))
except UncertainExecution:
    raise
except BaseException as error:
    queue.finish(job, lease['token'], {'error':str(error)}, False)
    raise
