import json
import hashlib
import os
import shutil
import sys
from pathlib import Path

import yaml
from local_bench.evaluation import evaluate_task
from local_bench.hbt_native import prefix_identity, validate_result
from local_bench.hbt_policy import materialize_policy
from local_bench.model import load_task
from local_bench.verifier import verify_candidate

source = Path('/workspace')
output = Path('/output')
frozen = Path('/frozen/run')
task = source / 'AgenticBench/local_tasks/hbt_assignment_policy'
spec = load_task(task)
config = yaml.safe_load(spec.base_config.read_text())
original = yaml.safe_load((frozen / 'config.yaml').read_text())
for key in ('design', 'type', 'pdk'):
    config[key] = original[key]
config['placement']['macro_halo_um'] = original['placement']['macro_halo_um']
(output / 'base.yaml').write_text(yaml.safe_dump(config))
expected = prefix_identity(frozen)

initial = spec.program_template.read_text()
changed = initial.replace('return context["pin_count"]', 'return -context["pin_count"]')
assert changed != initial
for name, program in [('reference', initial), ('candidate', changed)]:
    path = output / (name + '-source')
    path.mkdir()
    candidate = materialize_policy(program)
    verify_candidate(spec, candidate)
    (path / 'candidate.json').write_text(json.dumps(candidate, indent=2))
    (path / 'program.py').write_text(program)
    result = evaluate_task(task, path, source, output / name,
        baseline_run=frozen,
        reference_run=output / 'reference/run' if name == 'candidate' else None,
        base_config=output / 'base.yaml', timeout_seconds=3600,
        keep_workspace=False)
    print(name, json.dumps(result.to_dict()), flush=True)
    if result.metrics.get('runs_successfully') != 1:
        sys.exit(1)
    assert prefix_identity(output / name / 'run') == expected
    validate_result(output / name / 'run', frozen_input=frozen)

# Compare the native initial policy with the unmodified physical planner on the
# exact same real problem. Neither path invents or repairs a candidate result.
sys.path.insert(0, str(source / 'Place-3D'))
from src.hbt.planner import load_problem, write_result
ref = output / 'reference/run/50_hbt'
problem, context = load_problem(ref / 'problem.json')
write_result(output / 'native-no-policy', ref / 'problem.json', 'track-aware-greedy',
             problem, context, candidate_limit=None, max_cost_entries=10000000)
assert (output / 'native-no-policy/hbt.csv').read_bytes() == (ref / 'assignment/hbt.csv').read_bytes()
assert (ref / 'assignment/hbt.csv').read_bytes() != (output / 'candidate/run/50_hbt/assignment/hbt.csv').read_bytes()
negative = {}
for fault in ('upstream-chain', 'native-timing'):
    isolated = output / 'negative' / fault / 'run'
    shutil.copytree(output / 'candidate/run', isolated, copy_function=os.link)
    if fault == 'upstream-chain':
        target = isolated / '40_pdn/stage_manifest.json'
        value = json.loads(target.read_text())
        value['signature']['upstream'] = '0' * 64
        target.unlink()  # Never modify the hard-linked original inode.
        target.write_text(json.dumps(value))
    else:
        target = isolated / '55_post_placement_sta/passive/metrics.tsv'
        lines = target.read_text().splitlines()
        lines = [line if not line.startswith('wns_setup\t') else
                 'wns_setup\t' + str(float(line.split('\t')[1]) - 1) for line in lines]
        target.unlink()
        target.write_text('\n'.join(lines) + '\n')
        summary = isolated / '55_post_placement_sta/metrics.json'
        value = json.loads(summary.read_text())
        value['artifact_records']['native_metrics'].update(
            bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        summary.unlink()
        summary.write_text(json.dumps(value))
    try:
        validate_result(isolated, frozen_input=frozen)
    except ValueError as error:
        negative[fault] = str(error)
    else:
        raise AssertionError('Corrupt physical evidence accepted: ' + fault)
assert 'upstream manifest chain' in negative['upstream-chain']
assert 'STA metrics disagree' in negative['native-timing']
summary = {'status': 'verified', 'case': original['design'] + '-' + original['pdk'],
           'upstream_unchanged': expected, 'initial_matches_native': True,
           'candidate_assignment_changed': True,
           'negative_checks': negative,
           'metrics': {name: json.loads((output / name / 'evaluation_result.json').read_text())['metrics']
                       for name in ('reference', 'candidate')}}
(output / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
