"""Continue the original T2 developer session against the actual v2 interface."""
from layout import layout_path
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from engineering_bench.prepare_development import private_command, verify_release, admission, writable_paths
from engineering_bench.submission import clone_source, inventory, validate_scope
from engineering_bench.protocol import write_new, now, digest
from engineering_bench.tasks import load_task, read_json
from global_bench.workspace_codex import run_workspace_turn


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'experiment', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--task', choices=('clock_compensation', 'multi_hbt_fanout'), default='clock_compensation')
    p.add_argument('--draft-root', type=Path)
    p.add_argument('--fix-hbt-prefix', action='store_true')
    p.add_argument('--binary', default='codex')
    args = p.parse_args()
    task = load_task(args.task)
    draft = args.draft_root or args.experiment / 'development-v4' / args.task
    previous = read_json(draft / 'result.json')
    if previous['status'] not in ('source_draft_ready_pending_integration', 'integration_ready'):
        raise ValueError('Expected the successful original developer draft')
    source, image = verify_release(layout_path(args.root, 'release'))
    args.output.mkdir(parents=True, exist_ok=False)
    admission(args.output)
    subject = args.output / 'source'
    clone_source(source, subject)
    for relative in previous['changed_paths']:
        path = subject / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(draft / 'source' / relative, path)
    if inventory(subject) != read_json(draft / 'draft-inventory.json'):
        raise ValueError('Integration source differs from original draft')
    for relative in writable_paths(task.task_id):
        (subject / relative).mkdir(parents=True, exist_ok=True)
    (subject / 'AgenticBench').mkdir(exist_ok=True)
    before = read_json(draft / 'baseline-inventory.json')
    write_new(args.output / 'baseline-inventory.json', before)
    control, scratch, results = (args.output / name for name in ('control', 'scratch', 'results'))
    for folder in (control, scratch, results):
        folder.mkdir()
    for path in task.root.iterdir():
        if path.is_file():
            shutil.copy2(path, control / path.name)
    write_new(control / 'environment.json', {'phase': 'interface_integration', 'image': image,
        'task': task.task_id, 'previous_thread_id': previous['thread_id'], 'candidate_finals': 0})
    framework = Path(load_task.__code__.co_filename).resolve().parents[1]
    name = 'o3db-agent-engineering-integrate-' + uuid.uuid4().hex[:12]
    command = private_command(task, subject, name=name, image=image, control=control, scratch=scratch,
        results=results, benchmark=control, reference=layout_path(args.root, 'reference_root'),
        protected=(), cpus=8, memory_gb=32, trusted_framework=framework)
    write_new(args.output / 'dispatch.json', {'started_at': now(), 'task_digest': task.digest(),
        'thread_id': previous['thread_id'], 'model': 'gpt-5.6-sol', 'command': command,
        'prior_draft_sha256': previous['source_sha256'], 'candidate_finals': 0})
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    try:
        probe = "for p in ('/workspace/AGENTS.md','/task/task.json','/workspace/.git/config'):\n try:\n  f=open(p,'r+'); f.close()\n except OSError as e:\n  assert e.errno in (13,30); continue\n raise RuntimeError('unprotected mount: '+p)\nprint('read-only mounts verified')"
        checked = subprocess.run(['docker', 'exec', name, 'python3', '-c', probe], capture_output=True, text=True, timeout=30)
        (args.output / 'mount-check.log').write_text(checked.stdout + checked.stderr)
        if checked.returncode:
            raise ValueError('Integration mount preflight failed')
        prompt = task.instruction + '\n\n' + ((task.root / 'context-v2.md').read_text() if args.task == 'clock_compensation' else '') + '''

Continue your original implementation, now copied unchanged into a fresh private
/workspace. The original draft remains preserved. The actual protected executor
supports the v2 context described in /task/context-v2.md; your draft's demand for
complete measured outcomes of all actions cannot be supplied without doing the
selection outside your algorithm. Repair that integration dependency, keeping
your own substantive selection and safety reasoning. Do not fabricate native
data or claim predicted delay is measured. A final native timing failure will
be recorded, not silently replaced by reference results.

Read the updated task and context contract. Edit only the existing task scope;
add useful tests for a predicted useful action, preserved clock-group ownership,
hold risk, unknown information, conflicts and a legitimate no-op. Use the
container terminal. Temporary outputs belong in /scratch. No full-case EDA or
model calls. No arbitrary code-length limit applies. Submit one implementation
for all cases, then the host will seal it and run each case once. This is an
integration continuation, not another optimization trial or a new task.
'''
        if args.task == 'multi_hbt_fanout':
            prompt = task.instruction + '''

Continue your original draft unchanged at /workspace in this private source
copy. The host is implementing the protected multi-HBT joint placement-RC/STA
bridge. Fix these concrete native integration issues in your own contribution:

1. The real passive_hbt.passive_links and directional assembly require HBT IDs
   ending in a globally unique underscore plus integer. Your MHBT_*_H1 IDs do
   not satisfy that interface. The protected context will supply
   new_hbt_id_base, an integer larger than every original HBT's numeric suffix.
   Allocate unique IDs deterministically using that base and sorted target IDs,
   preserving original HBT identities and the one-to-three branch limit. Inspect
   the actual assembly helpers; do not weaken or edit their parser or RC checks.
2. buffer_cells now has an explicit die field as well as the already documented
   fields. Select only buffers available on the branch's source die. A bottom
   master must not be instantiated on the top die or vice versa. Add a regression
   with both source directions and misleading area/name ordering.
3. Native detailed_placement -max_displacement uses site/row counts, not microns.
   The supplied native mapping has max_buffer_displacement_sites: [500,500].
   Use this explicit pair with unchanged original-cell fixing and legality/PG
   checks. Do not relax preservation to make placement succeed.
4. Endpoint is now current-step placement-RC fixed-original-receiver delay plus
   setup/hold STA, no GRT or downstream flow. Update obsolete descriptions but
   never manufacture a trusted qualification/result or claim measured gains.

The v1 context otherwise retains your documented units, real baseline delays,
driver/receiver geometry, legal spare sites and library models. New fields are
native integration metadata, not extra optimization hints. The host will freeze
the target set and final resource/timing guards before any candidate evaluation.
Preserve the substantive synthesis policy unless a concrete compatibility bug
requires a change. Add focused tests, using tiny native fixtures if useful. No
full-case EDA or model calls. Edit only task scope; temporary files in /scratch.
This is the final integration of the same implementation, not a new optimization
round. Finish with actual test results and remaining limitations in English.
'''
        if args.fix_hbt_prefix:
            if args.task != 'multi_hbt_fanout':
                raise ValueError('HBT namespace fix belongs only to T3')
            prompt = task.instruction + '''

Continue the SAME implementation/session. The full integrated source is already
copied into /workspace. A trusted independent one-extra-HBT qualification fixture
exposed a remaining native naming contract BEFORE any candidate final ran:

open3dbench/backend/hbt_materialize.py:_validate_boundary_verilog recognizes
HBT_*_SRC/DST ports. passive_hbt.py's physical-port lookup specifically requires
HBT_(B2T|T2B)_<integer>_(SRC|DST), and OpenROAD-3D physical-anchor cleanup also
expects HBT_ names. Therefore MHBT_B2T_<integer> is still incompatible even though
its numeric suffix passes the earlier allocator check. The required new anchor
ID is exactly HBT_B2T_<integer> or HBT_T2B_<integer>, with new_hbt_id_base already
guaranteed above all originals. New ports must derive from that exact ID.

Repair only this naming compatibility and associated tests/documentation in your
allowed contribution. Preserve your algorithm, all decisions, and other previous
integration fixes. Inspect the cited native helpers and test both directions,
unique suffix allocation and original-ID preservation. Do not change the shared
physical flow/verifier, relax a gate, run full-case EDA, or start model calls.
The independent fixture failure is an interface issue, not an optimization
result; no measured candidate outcomes are supplied. Finish with a short English
summary and actual focused test results. This is not another optimization trial.
'''
        turn = run_workspace_turn(binary=args.binary, control=control,
            output=args.output / 'turn', container=name, prompt=prompt,
            terminal_script=framework / 'global_bench/container_terminal.py', python=sys.executable,
            thread_id=previous['thread_id'], timeout_seconds=1800, model='gpt-5.6-sol')
    finally:
        subprocess.run(['docker', 'stop', '--time', '10', name], check=False, capture_output=True, timeout=30)
    after = inventory(subject)
    changes = validate_scope(task, before, after)
    write_new(args.output / 'draft-inventory.json', after)
    write_new(args.output / 'result.json', {'status': 'integration_ready' if turn['success'] else 'integration_failed',
        'changed_paths': changes, 'source_sha256': digest(after), 'thread_id': turn['thread_id'],
        'terminal_tool_calls': turn['terminal_tool_calls'], 'duration_seconds': turn['duration_seconds'],
        'completed_at': now(), 'candidate_finals': 0})


if __name__ == '__main__':
    main()
