"""Diagnostic integration smoke test; never produces campaign scores."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from local_bench.hbt_native import prefix_identity
from open3dbench.backend.contracts import validate_checkpoint
from open3dbench.config import load_config
from open3dbench.release_ablation import copy_prefix, validate_reference
from open3dbench.runner import FlowRunner, STAGE_DIRS
from open3dbench.stage_manifest import validate_stage_manifest

source, frozen, output = Path('/workspace'), Path('/frozen/run'), Path('/output')
mode = sys.argv[1]
original = prefix_identity(frozen)
validate_reference(frozen, 'placement')
config = load_config(frozen / 'config.yaml')
config['pdn']['algorithm'] = mode
config['threads'] = 4
config['opt']['threads'] = 4
config['routing']['threads'] = 4
config['run_dir'] = str(output / 'run')
copy_prefix(frozen, output / 'run', 'pdn', config)
runner = FlowRunner(source, config, resume=True)
start = time.monotonic()
runner.run(start='pdn', stop='cts')
stage_times = {stage: runner.status['stages'][stage]['runtime_seconds']
               for stage in ('pdn', 'hbt', 'opt', 'cts')}
hashes = {}
for stage in stage_times:
    directory = output / 'run' / STAGE_DIRS[stage]
    validate_stage_manifest(directory / 'stage_manifest.json', expected_stage=stage, verify_artifacts=True)
    validate_checkpoint(directory / 'checkpoint.json', require_files=True)
    hashes[stage] = hashlib.sha256((directory / 'stage_manifest.json').read_bytes()).hexdigest()
pdn = json.loads((output / 'run/40_pdn/checkpoint.json').read_text())
assert pdn['pdn_status'] == ('generated' if mode == 'openroad' else 'passthrough_not_generated')
hbt_manifest = json.loads((output / 'run/50_hbt/stage_manifest.json').read_text())
# The HBT producer publishes its semantic state alongside the physical checkpoint.
hbt_checkpoint = json.loads((output / 'run/50_hbt/checkpoint.json').read_text())
assert hbt_checkpoint['source_checkpoint'] == '../40_pdn/checkpoint.json'
assert original == prefix_identity(frozen)
resumed = FlowRunner(source, config, resume=True)
resumed.run(start='pdn', stop='cts')
for stage, old in hashes.items():
    path = output / 'run' / STAGE_DIRS[stage] / 'stage_manifest.json'
    assert old == hashlib.sha256(path.read_bytes()).hexdigest(), stage
    assert resumed.status['stages'][stage]['execution_mode'] == 'reused', stage
receipt = {'status': 'verified', 'kind': 'diagnostic-pdn-to-cts',
           'case': config['design'] + '-' + config['pdk'], 'pdn_algorithm': mode,
           'prefix_unchanged': True, 'resume_unchanged': True,
           'reused': ['input', 'floorplan', 'partition', 'placement'],
           'executed': list(stage_times), 'stage_seconds': stage_times,
           'wall_seconds': time.monotonic() - start, 'manifests': hashes,
           'image': os.environ['OPEN3DBENCH_IMAGE_ID']}
(output / 'qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
