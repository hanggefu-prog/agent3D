"""One fresh implementation from the explicitly qualified physical baseline.

No previous model source, session, candidate outcome, or judge is imported.
Only the now-qualified runtime/interface documentation differs from the initial
5.6 draft phase; this difference is recorded, not hidden as a matched prompt.
"""
from layout import layout_path
import argparse
import json
import os
from pathlib import Path
from engineering_bench.identity import verify_evaluator
import shutil
import subprocess
import sys
import uuid

from engineering_bench.prepare_development import admission, private_command, verify_release, writable_paths
from engineering_bench.submission import clone_source, inventory, validate_scope
from engineering_bench.protocol import write_new, now, digest, sha
from engineering_bench.tasks import load_task
from engineering_bench.protocol import qualify
from engineering_bench.scoring import (CURRENT_SCORING_VERSION, qualified_contract,
    scoring_prompt, development_instruction)
from global_bench.workspace_codex import run_workspace_turn


COMMON = """
This is a fresh implementation from the original baseline, not a continuation of
another model. The current-stage evaluators and fixed inputs are ready. Submit
one implementation for all four cases. No extra full-case EDA or model calls.
You may compile and run small functional/regression tests using the real tools.

Use the container terminal. Source is at /workspace; /task contains the task and
current interfaces. Baseline-only sample contexts are under /task/examples.
Frozen inputs are read-only at /inputs/reference; qualified post-CTS reference
artifacts are read-only at /inputs/stage-reference. These are not candidate
results. Protected evaluator sources in /task/native-interface document exact
I/O expectations; they cannot be modified. The complete final evaluator remains
host-owned. Do not fabricate results, invoke full-case EDA, or tune case-specific
answers. Do not read the host filesystem or invoke external model APIs.

Read /workspace/AGENTS.md and relevant component guidance. Edit only the public
task's allowed scope. Helpers and tests are welcome, without code-length limits
or a prescribed algorithm. Allowed files are writable even when their parent
directory is read-only. Edit existing files in place with Python Path.write_text
or open(..., 'w'); git apply, patch and sed -i may fail because they replace files
or create temporary files beside them. You can apply a patch to a scratch copy,
then copy its bytes into the allowed file without renaming it.
Put caches, build and temporary files under /scratch;
do not commit, alter Git metadata, or add network dependencies. The host will
stop the container, inspect the complete diff, seal one submission and evaluate
each case exactly once, including recording a failed candidate without retry.
An empty action/unchanged net is legitimate when justified, not a substitute
for implementing the requested capability. Finish with actual tests and limits.

The host records the explicit API model and uses the task-specific turn timeout.
The independent judge uses a separate session. Historical
development.md files describe preliminary interfaces and are omitted
from this task packet; the current instruction and runtime-interface.md apply.
"""


T3 = """
# Qualified multi-HBT native interface

Implement synthesize(context, output_dir). The host calls it once under python3
with PYTHONPATH=/subject in the pinned OpenROAD image. Real odb and OpenROAD
are installed. Input ODBs are mounted at /candidate/bottom.odb and
/candidate/top.odb; use context.native paths and mappings. Write the two results
as output_dir/bottom/design.odb and output_dir/top/design.odb, and write
output_dir/branch_plan.json. A returned Python object alone is insufficient.

Read the complete example context and protected multi_readback.py and
interfaces.py. The context schema is multi-hbt-context-v1, with target_nets,
drivers, receivers, site_geometry, buffer_cells, baseline_timing, limits and
native metadata. It contains real reference delay, input slew and geometry;
these are not counterfactual post-edit measurements. Cover every fixed target.

branch_plan.json contains branch_map (logical net -> list of branch IDs) and
target_nets (a list of objects with logical_net and branches). Each branch has
branch_id, hbt_id, site_id, source_die, sink_die, and buffer_instance (null on the
original branch). The protected readback derives actual receivers/connections
from the databases, never trusts candidate assertions. Preserve one original
HBT branch per target, keeping its driver and coordinates. Each extra branch
has exactly one new noninverting isolation buffer on the source die, driven by
the original trunk; its output drives one distinct new HBT electrical link.
There are at most three total HBT branches per target. Retain all original
receivers exactly once, and all original non-target logic/clock connectivity.

New HBT IDs must be HBT_B2T_<integer> or HBT_T2B_<integer>, with globally unique
numeric suffixes allocated from context.new_hbt_id_base. Port names are the HBT
ID plus _SRC and _DST. Physical anchors use the provided native COVER masters,
sizes/layers and legal spare sites; reference master aliases can differ by die.
buffer_cells includes die: use only source-die-compatible masters and pins.
Detailed-placement max displacement uses site/row counts, provided explicitly
as native.max_buffer_displacement_sites, not microns. Fixed original cells must
keep locations, orientations, status and identities after new-cell legalization.
Preserve PG and run native placement checks. Read the protected materialization
helpers in the baseline to understand actual OpenDB/STA contracts; do not edit
them or weaken the shared-electrical-net rejection.

No-op target branches still need correct map entries and valid output ODBs.
No GRT is required: independent placement-RC fixed-original-receiver delays and
joint setup/hold STA are computed by the host. WNS/TNS cannot regress by more
than 1e-6 ns and no new hold-violating endpoint is allowed; baseline negative
slack is permitted. Electrical count/excess and stated resource limits also
apply.
"""


def development_status(turn, changes):
    """A successful model process is not evidence of an implementation."""
    if not turn['success']:
        return 'developer_failed'
    return 'implementation_ready' if changes else 'no_change'


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'experiment', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--binary', default='codex')
    p.add_argument('--scoring-version', type=int, choices=(1, 2), default=CURRENT_SCORING_VERSION,
                   help='1 (default): compatible frozen task; 2: explicitly opt in to evaluation-v2.json')
    p.add_argument('--prepare-only', action='store_true', help='Prepare a container for a host-authenticated remote client; publish its binding and leave it running')
    p.add_argument('--qualification-root', type=Path, required=True,
                   help='Directory containing <task>/<case>/qualification.json')
    p.add_argument('--context-root', type=Path,
                   help='T2/T3: complete <case>/{clock,multi}-context.json directory')
    args = p.parse_args()
    task = load_task(args.task)
    proofs = [qualify(task, case, args.qualification_root / task.task_id / case / 'qualification.json')
              for case in task.data['cases']]
    contract = qualified_contract(task, proofs, args.scoring_version)
    try:
        for q in proofs:
            verify_evaluator(q['record'])
    except ValueError:
        raise ValueError('Qualification uses another native adapter revision')
    if task.task_id != 'cross_die_cts' and args.context_root is None:
        p.error('--context-root is required for T2/T3')
    release = layout_path(args.root, 'release')
    source, image = verify_release(release)
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / 'started.json', {'started_at': now(), 'task': task.task_id,
        'model': args.model, 'fresh_session': True, 'candidate_finals': 0,
        'qualifications': [q['sha256'] for q in proofs], 'scoring_contract': contract})
    container = 'o3db-agent-engineering-' + uuid.uuid4().hex[:12]
    live = False
    result = {'task': task.task_id, 'model': args.model, 'candidate_finals': 0}
    try:
        admission(args.output)
        subject = args.output / 'source'
        revision = clone_source(source, subject)
        (subject / 'AgenticBench').mkdir(exist_ok=True)
        for relative in writable_paths(task.task_id):
            if not (subject / relative).exists():
                (subject / relative).mkdir(parents=True)
        control, scratch, results = (args.output / n for n in ('control', 'scratch', 'results'))
        for folder in (control, scratch, results):
            folder.mkdir()
        for f in task.root.iterdir():
            if f.is_file() and f.name != 'development.md':
                shutil.copy2(f, control / f.name)
        examples = control / 'examples'
        examples.mkdir()
        if task.task_id == 'clock_compensation':
            doc = (task.root / 'context-v2.md').read_text()
        elif task.task_id == 'multi_hbt_fanout':
            doc = T3
        else:
            doc = ('# Current CTS endpoint\nThe complete Python CTS replay and native '
                'skew evaluator are qualified. Use run_cts with its baseline invocation '
                'and artifact contract. The host restores post-OPT data coordinates '
                'before fixed standard legalization. The selected scoring contract '
                'below defines the timing and resource guards. Clock '
                'electrical violation count/excess cannot regress. Do not change '
                'SDC, clock domains, HBT positions or the independent evaluator. '
                'Native C++ edits are in scope but require host compilation and '
                'runtime binding before evaluation; explicitly report them.\n')
        doc += '\n' + scoring_prompt(task, contract)
        if task.task_id != 'cross_die_cts':
            filename = 'clock-context.json' if task.task_id == 'clock_compensation' else 'multi-context.json'
            for case in task.data['cases']:
                ctx = args.context_root / case / filename
                facts = json.loads((args.qualification_root / task.task_id / case / 'evidence.json').read_text())
                if facts['context_sha256'] != sha(ctx):
                    raise ValueError('Example context differs from qualification: ' + case)
                shutil.copy2(ctx, examples / (case + '.json'))
        (control / 'runtime-interface.md').write_text(doc)
        write_new(control / 'scoring-contract.json', contract)
        native = control / 'native-interface'
        native.mkdir()
        interfaces = Path(load_task.__code__.co_filename).resolve().parent / 'interfaces.py'
        shutil.copy2(interfaces, native / 'interfaces.py')
        # Read-only I/O/checker context only, never another model's implementation.
        adapter = Path(__file__).parent
        for name in ('multi_readback.py', 'target_delay.py') if task.task_id == 'multi_hbt_fanout' else ():
            shutil.copy2(adapter / name, native / name)
        before = inventory(subject)
        write_new(args.output / 'baseline-inventory.json', before)
        framework = Path(load_task.__code__.co_filename).resolve().parents[1]
        prompt = development_instruction(task, contract) + '\n\n' + doc + '\n\n' + COMMON
        (control / 'effective-prompt.txt').write_text(prompt)
        write_new(control / 'environment.json', {'source_revision': revision, 'image': image,
            'seed': task.data['seed'], 'task_digest': task.digest(), 'cpus': 8, 'memory_gb': 32,
            'prompt_sha256': sha(control / 'effective-prompt.txt'), 'old_model_artifacts_visible': False,
            'scoring_contract_sha256': sha(control / 'scoring-contract.json'),
            'scoring_contract_digest': digest(contract),
            'comparison_note': 'Same final contract; qualified interface docs replace earlier preliminary phase'})
        command = private_command(task, subject, name=container, image=image, control=control,
            scratch=scratch, results=results, benchmark=control,
            reference=layout_path(args.root, 'reference_root'),
            protected=(), cpus=8, memory_gb=32, trusted_framework=framework)
        command = command[:-3] + ['--mount', 'type=bind,src=' + str(args.experiment / 'stage-reference-v1')
            + ',dst=/inputs/stage-reference,readonly'] + command[-3:]
        write_new(args.output / 'container.json', {'command': command, 'name': container})
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        live = True
        probe = "for p in ('/workspace/AGENTS.md','/task/task.json','/workspace/.git/config','/workspace/AgenticBench/engineering_bench/metrics.py'):\n try:\n  f=open(p,'r+');f.close()\n except OSError as e:\n  assert e.errno in (13,30);continue\n raise RuntimeError('Writable protected file: '+p)\nprint('read-only mounts verified')"
        tested = subprocess.run(['docker', 'exec', container, 'python3', '-c', probe], capture_output=True, text=True, timeout=30)
        (args.output / 'mount-check.log').write_text(tested.stdout + tested.stderr)
        if tested.returncode:
            raise ValueError('Mount protection failed before model call')
        if args.prepare_only:
            write_new(args.output / 'ready.json', {'status': 'awaiting_host_model', 'container': container,
                'model': args.model, 'task': task.task_id, 'source': str(subject),
                'control': str(control), 'prompt_sha256': sha(control / 'effective-prompt.txt'),
                'baseline_inventory': str(args.output / 'baseline-inventory.json'),
                'created_at': now(), 'candidate_finals': 0})
            live = False  # Explicit ownership transfer to the host controller.
            return 0
        write_new(args.output / 'model-started.json', {'started_at': now(), 'model': args.model, 'task': task.task_id})
        turn = run_workspace_turn(binary=args.binary, control=control,
            output=args.output / 'development/turn', container=container, prompt=prompt,
            terminal_script=framework / 'global_bench/container_terminal.py', python=sys.executable,
            thread_id=None, timeout_seconds=task.data['development_timeout_seconds'], model=args.model)
        subprocess.run(['docker', 'stop', '--time', '10', container], check=True, capture_output=True, timeout=30)
        live = False
        after = inventory(subject)
        changes = validate_scope(task, before, after)
        write_new(args.output / 'draft-inventory.json', after)
        result.update(status=development_status(turn, changes),
            developer_success=turn['success'], changed_paths=changes, source_sha256=digest(after),
            thread_id=turn['thread_id'], duration_seconds=turn['duration_seconds'],
            terminal_tool_calls=turn['terminal_tool_calls'])
    except Exception as exc:
        result.update(status='blocked', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        if live:
            subprocess.run(['docker', 'stop', '--time', '10', container], capture_output=True, timeout=30)
    result['completed_at'] = now()
    write_new(args.output / 'result.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'implementation_ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())
