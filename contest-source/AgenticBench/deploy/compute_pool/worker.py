"""Credential-free physical evaluator and exact-reference qualification."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from local_bench.benchmark_input import validate_benchmark_input
from local_bench.model import load_task
from local_bench.program_bridge import materialize_program

SOURCE = Path('/data/gcr/agentic3d-dev/open3dbench-source-grt-order-20260831-5round')


def evaluate(root, task, design, candidate, output, dry_run=False):
    if output.exists():
        raise FileExistsError('refusing to overwrite evaluation: ' + str(output))
    command = [sys.executable, '-m', 'local_bench', 'evaluate', task,
               '--candidate', str(candidate), '--open3dbench-root', str(SOURCE),
               '--base-config', str(root / 'framework/local_bench/benchmark_inputs'
                                    / (design + '_lol_ng45_mtkahypar_cts') / 'config.yaml'),
               '--benchmark-input', str(root / 'inputs' / design),
               '--reference-run', str(root / 'references' / design),
               '--output-dir', str(output), '--compact-artifacts', '--timeout', '21600']
    if dry_run:
        command.append('--dry-run')
    completed = subprocess.run(command, check=False)
    result = json.loads((output / 'evaluation_result.json').read_text())
    if completed.returncode or result['metrics'].get('runs_successfully') != 1:
        raise RuntimeError('evaluation failed; inspect ' + str(output))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--node', required=True)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('qualify')
    run = sub.add_parser('evaluate')
    run.add_argument('--task', required=True, choices=[
        'global_routing_ripup_policy', 'global_routing_maze_cost',
        'global_routing_iteration_controller', 'global_routing_net_ordering'])
    run.add_argument('--case', required=True, choices=['bp', 'ariane133'])
    run.add_argument('--candidate', required=True, type=Path)
    run.add_argument('--name', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'evaluate':
        if Path(args.name).name != args.name or args.name in ('.', '..'):
            raise ValueError('name must be a single directory name')
        candidate = args.candidate.resolve()
        candidate.relative_to(root)
        evaluate(root, args.task, args.case, candidate, root / 'results' / args.name)
        return
    validation = {}
    for repo in (SOURCE, SOURCE / 'OpenROAD-3D/openroad'):
        subprocess.run(['git', '-c', 'safe.directory=' + str(repo), '-C', str(repo),
                        'fsck', '--connectivity-only', '--no-dangling'], check=True)
    for design in ('bp', 'ariane133'):
        validation[design] = validate_benchmark_input(
            root / 'inputs' / design, task_id='global_routing_ripup_policy_source')
        print('validated_frozen_input=' + design, flush=True)
    spec = load_task(root / 'framework/local_tasks/global_routing_ripup_policy')
    program = spec.program_template.read_text()
    candidate = materialize_program(spec, program, SOURCE)
    directory = root / 'results/qualification-candidate'
    directory.mkdir(exist_ok=False)
    (directory / 'candidate.json').write_text(json.dumps(candidate, indent=2) + '\n')
    (directory / 'program.py').write_text(program)
    result = evaluate(root, 'global_routing_ripup_policy', 'bp', directory,
                      root / 'results/qualification-bp')
    metrics = result['metrics']
    expected = {'grt_wirelength_um': 5713996, 'grt_total_overflow': 0}
    observed = {key: metrics.get(key) for key in expected}
    passed = observed == expected and metrics.get('policy_calls', 0) > 0
    report = {'node': args.node, 'status': 'routing_ready' if passed else 'mismatch',
              'source_image_id': 'sha256:fe4689919f8f474c77d09bca8210b1755ae4fd862b8d3200d43d690986cd3262',
              'runtime_transfer': 'flattened never-run container filesystem via authenticated SSH or checksummed archive',
              'expected': expected, 'observed': observed,
              'frozen_inputs': validation, 'metrics': metrics,
              'scope': 'CPU GRT evaluation; not yet GPU placement or global STA qualification',
              'llm_credentials_deployed': False, 'evolution_budget_consumed': 0}
    (root / 'results/qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise RuntimeError('cross-host reference mismatch')
    print(json.dumps({'node': args.node, 'status': 'routing_ready', 'observed': observed}), flush=True)


if __name__ == '__main__':
    main()
