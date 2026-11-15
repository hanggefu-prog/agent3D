"""Fresh read-only engineering reviewer; no candidate writes or QoR rankings.

Explicit review version 2 checks judge/developer family independence before
any container starts, mounts a redacted view of the sealed submission, runs at
least three fresh samples (fresh container and scratch each) and records their
median and dispersion. Version 1 reproduces the frozen single-sample review.
The judge never enters the objective headline (weight 0).
"""
from layout import layout_path
import argparse
import contextlib
import difflib
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from engineering_bench.tasks import load_task
from engineering_bench.judge import (MIN_JUDGE_SAMPLES, legality_checks, load_families, check_judge_independence,
                                     prepare_review_view, redact_text, redact_value)
from engineering_bench.model import cpu_container_command, judge, judge_panel
from engineering_bench.prepare_development import verify_release
from engineering_bench.protocol import read_json, write_new, now, sha, digest
from engineering_bench.submission import verify_seal

PROBE = ("import pathlib; p=pathlib.Path('/workspace/AGENTS.md');\ntry:\n f=p.open('r+'); f.close()\n"
         "except OSError as e:\n assert e.errno in (13,30)\nelse:\n raise RuntimeError('judge source is writable')\n"
         "print('judge source read-only')")


def submission_diff(base, subject, changed_paths, names=(), redact=False):
    patch = []
    for name in changed_paths:
        before = (base / name).read_text().splitlines(True) if (base / name).is_file() else []
        after = (subject / name).read_text().splitlines(True) if (subject / name).is_file() else []
        patch.extend(difflib.unified_diff(before, after, fromfile='baseline/' + name, tofile='submission/' + name))
    text = ''.join(patch)
    return redact_text(text, names) if redact else text


def case_legality(task, sealed, final_root, version=2):
    """Per-case legality status from the host ledger; no QoR is exposed."""
    evidence = []
    for case in task.data['cases']:
        ledger = final_root / 'ledger' / case / 'result.json'
        if not ledger.is_file():
            evidence.append({'case': case, 'status': 'not_in_this_host_evidence_bundle'})
            continue
        row = read_json(ledger)
        if row['submission_sha256'] != sealed['submission_sha256']:
            raise ValueError('Legality evidence belongs to another submission')
        native_path = final_root / case / 'final-native.json'
        checked = read_json(native_path)['checks'] if native_path.is_file() else {}
        item = {'case': case, 'status': row['status'], 'legal': row.get('legal'),
                'failure_reason': row.get('error'),
                'native_receipt_sha256': sha(native_path) if native_path.is_file() else None}
        if version == 1:
            item['checks'] = {k: v['passed'] for k, v in checked.items()}
        else:
            states = legality_checks(checked)
            item['checks'] = states['status_map']
            item['unverified_checks'] = states['unverified']
        evidence.append(item)
    return evidence


def main(argv=None):
    p = argparse.ArgumentParser()
    for k in ('root', 'experiment', 'subject', 'sealed', 'output', 'final_root'):
        p.add_argument('--' + k.replace('_', '-'), type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--evidence-bundle', type=Path)
    p.add_argument('--judge-model', '--model', dest='judge_model', default='gpt-5.6-sol')
    p.add_argument('--developer-model', help='required by review version 2 for the independence check')
    p.add_argument('--samples', type=int, default=MIN_JUDGE_SAMPLES)
    p.add_argument('--model-family-map', type=Path, help='JSON {family: [lowercase name prefixes]}')
    p.add_argument('--allow-same-family-reason', help='recorded reason to accept a same/unknown-family judge')
    p.add_argument('--redact', action='append', default=[], help='additional literal string to redact')
    p.add_argument('--review-version', type=int, choices=(1, 2), default=1,
                   help='1 (default): compatible single fresh review; 2: independent-family median panel')
    p.add_argument('--binary', default='codex')
    args = p.parse_args(argv)
    task = load_task(args.task)
    v2 = args.review_version == 2
    families = load_families(args.model_family_map)
    independence = None
    if v2:
        if not args.developer_model:
            raise ValueError('Review version 2 needs --developer-model for the independence check')
        if args.samples < MIN_JUDGE_SAMPLES:
            raise ValueError('Review version 2 needs at least three judge samples')
        # Fail closed before any container or paid request.
        independence = check_judge_independence(args.judge_model, args.developer_model, families,
                                                args.allow_same_family_reason)
    names = [n for n in [args.judge_model, args.developer_model] + args.redact if n]
    base, image = verify_release(layout_path(args.root, 'release'))
    sealed = read_json(args.sealed)
    verify_seal(args.subject, sealed)
    args.output.mkdir(parents=True, exist_ok=False)
    control, results = args.output / 'control', args.output / 'results'
    for directory in (control, results):
        directory.mkdir()
    for source in task.root.iterdir():
        if source.is_file():
            shutil.copy2(source, control / source.name)
    (control / 'submission.diff').write_text(submission_diff(base, args.subject, sealed['changed_paths'], names, v2))
    view = args.subject
    if v2:
        receipt = prepare_review_view(args.subject, sealed['changed_paths'], args.output / 'subject-view', names)
        view = Path(receipt['view'])
        # Host-side only: the literal names are not written where the judge reads.
        write_new(args.output / 'redaction-receipt.json', dict(receipt, redacted_name_count=len(set(names)),
            names_sha256=digest(sorted(set(names))), family_map_sha256=digest(families)))
    if args.evidence_bundle:
        from judge_bundle import validate_bundle
        bundle = read_json(args.evidence_bundle)
        evidence = validate_bundle(bundle, task, sealed)
        write_new(args.output / 'input-evidence-receipt.json', {'path': str(args.evidence_bundle),
            'sha256': sha(args.evidence_bundle), 'scope': 'all four cases; no previous judge or objective rankings'})
    else:
        evidence = case_legality(task, sealed, args.final_root, args.review_version)
    legality = {'scope': 'independent native stage-only checks, no QoR ranking',
        'cases': evidence, 'runtime': 'pinned image-installed native executable', 'source_diff_in_task_folder': True}
    if v2:
        legality['unverified_is_not_passed'] = True
        legality = redact_value(legality, names)
    write_new(results / 'legality.json', legality)
    aliases = args.output / 'evidence-namespaces'
    shutil.copytree(results, aliases / 'results')
    shutil.copytree(control, aliases / 'task')
    framework = Path(load_task.__code__.co_filename).resolve().parents[1]
    dispatches = []

    @contextlib.contextmanager
    def container(index):
        scratch = args.output / ('scratch' if index is None else 'scratch-%d' % index)
        scratch.mkdir()
        name = 'o3db-agent-engineering-judge-' + uuid.uuid4().hex[:12]
        cmd = cpu_container_command(name=name, image=image, subject=view, control=control, scratch=scratch,
            results=results, benchmark=control, reference=results, protected=(), cpus=2, memory_gb=32,
            subject_read_only=True)
        dispatches.append({'sample': index, 'command': cmd, 'started_at': now(), 'model': args.judge_model,
            'fresh_session': True, 'candidate_finals': 0})
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        try:
            test = subprocess.run(['docker', 'exec', name, 'python3', '-c', PROBE],
                                  capture_output=True, text=True, timeout=30)
            log = 'mount-check.log' if index is None else 'mount-check-%d.log' % index
            (args.output / log).write_text(test.stdout + test.stderr)
            if test.returncode:
                raise ValueError('Judge mount protection failed')
            yield {'container': name}
        finally:
            subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, timeout=30)

    runtime = dict(binary=args.binary, control=control, terminal_script=framework / 'global_bench/container_terminal.py',
                   python=sys.executable)
    roots = [view, control, results, aliases]
    try:
        if v2:
            result = judge_panel(task, args.subject, sealed, model=args.judge_model,
                developer_model=args.developer_model, evidence_roots=roots, output=args.output / 'review',
                samples=args.samples, families=families, allow_same_family_reason=args.allow_same_family_reason,
                sample_runtime=container, **runtime)
            summary = {'review_version': 2, 'status': result['status'], 'score': result['score'],
                       'aggregation': 'median', 'samples': result['samples'],
                       'graded_samples': result['graded_samples'], 'dispersion': result['dispersion'],
                       'independence': independence, 'weight_in_headline': 0, 'completed_at': now()}
        else:
            with container(None) as overrides:
                result = judge(task, args.subject, sealed, model=args.judge_model, evidence_roots=roots,
                               output=args.output / 'review', **dict(runtime, **overrides))
            summary = {'status': result['status'], 'score': result['score'], 'completed_at': now()}
    finally:
        write_new(args.output / 'container-dispatch.json',
                  dispatches[0] if not v2 and dispatches else {'samples': dispatches})
    write_new(args.output / 'result.json', summary)
    return summary


if __name__ == '__main__':
    main()
