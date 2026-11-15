"""Qualify the native fixed-cell executor, then run one sealed T2 plan/case.

Every required capability is recorded with its own concrete check, measured
value, threshold and source artifact hashes (capability_schema 2). The
Explicit --scoring-version 2 additionally binds the qualification to
engineering_tasks/clock_compensation/evaluation-v2.json so the aggregate
reports the calibrated (noop/expert) headline.
"""
from layout import layout_path
import argparse
import csv
import fcntl
import json
from pathlib import Path
import re
import time

from clock_executor import execute, sample_plan
from cts_final_queue import check_artifacts, evidence, electrical, export_log
from cts_reference_queue import admission
from cts_reference_stage import verify_prefix
from measure_host import launch
from engineering_bench.capabilities import Recorder
from engineering_bench.clock_equivalence import compare_local
from engineering_bench.identity import current_identity
from engineering_bench.interfaces import validate_clock_plan
from engineering_bench.native_snapshot import snapshot
from engineering_bench.legalization import load_snapshot, ports_geometry
from engineering_bench.protocol import write_new, read_json, digest, sha, now, OnceLedger
from engineering_bench.scoring import (qualification_fields, CURRENT_SCORING_VERSION,
                                      validate_development_binding)
from engineering_bench.submission import seal, verify_seal
from engineering_bench.tasks import load_task
from engineering_bench.prepare_development import verify_release
from engineering_bench.input_audit import clock_table
from engineering_bench.metrics import clock_skew, aggregate, timing_guard
from engineering_bench.evaluation import validate_native_result

DIES = ('bottom', 'top')
ELECTRICAL_TOLERANCE = 1e-6


def readback(output, *, reference, run, platform, image, additions):
    old, new = {}, {}
    for die in DIES:
        old[die], new[die] = reference / ('readback-' + die), output / ('readback-' + die)
        row = snapshot(run=run, platform=platform, image=image, stage='70_cts', die=die,
            output=new[die], clock_sdc=run / '60_opt' / die / 'design.sdc',
            odb_path=output / 'eco' / die / 'design.odb')
        if row['status'] != 'succeeded' or row['statistics']['floating_pg']:
            raise ValueError('Native readback/PG failed')
        if ports_geometry(old[die] / 'ports.tsv') != ports_geometry(new[die] / 'ports.tsv'):
            raise ValueError('Original ports/HBT geometry changed')
    with (run / '60_opt/hbt.csv').open() as stream:
        links = list(csv.DictReader(stream))
    proof = compare_local(old, new, links, links, additions)
    write_new(output / 'physical-checks.json', {'equivalence': proof, 'added_buffers': additions,
        'snapshot_hashes': {d: sha(new[d] / 'result.json') for d in new}})
    return proof


def measurement(folder):
    raw = read_json(folder / 'result.json')
    if raw['status'] != 'succeeded' or raw['measurement_mode'] != 'skew_sta':
        raise ValueError('Missing independent joint setup/hold measurement')
    check_artifacts(folder, raw)
    values = clock_table(folder / 'assembly/passive/clock_sinks_by_clock.tsv')
    expected = {g: sorted(v) for g, v in values.items()}
    metric = clock_skew(values, expected)
    if abs(metric-raw['clock_skew_ns']) > 1e-12:
        raise ValueError('Skew scalar mismatch')
    return {'values': values, 'expected': expected, 'metric': metric, 'timing': raw['timing']}


def electrical_failures(value, baseline):
    failures = []
    for key, entry in value['clock'].items():
        old = baseline['clock'][key]
        if entry['count'] > old['count'] or entry['worst_excess_ratio'] > old['worst_excess_ratio'] + ELECTRICAL_TOLERANCE:
            failures.append(key)
    return failures


def electrical_check(folder, snapshots, baseline=None):
    clocks = set()
    for die in DIES:
        _, _, cells = load_snapshot(snapshots[die])
        clocks.update('u_' + die + '/' + n for n, c in cells.items() if c['clock_cell'])
    groups = clock_table(folder / 'assembly/passive/clock_sinks_by_clock.tsv')
    value = electrical(folder / 'electrical.rpt', clocks, {p for group in groups.values() for p in group})
    if baseline:
        for key in electrical_failures(value, baseline):
            raise ValueError('Clock electrical guard failed: ' + key)
    return value


def eco_log(execution, die):
    """Trusted ECO script: detailed_placement, check_placement, fixed-cell and PG checks, then marker."""
    path = execution / 'eco' / die / 'native.log'
    text = path.read_text() if path.is_file() else ''
    marker = re.findall(r'^ENGINEERING_CLOCK_ECO_OK originals=(\d+) pg=(\d+)$', text, re.M)
    if len(marker) != 1:
        return {'path': path, 'clean': False, 'originals': None, 'pg': None}
    clean = not re.search(r'^\[ERROR\b', text, re.M)
    return {'path': path, 'clean': clean, 'originals': int(marker[0][0]), 'pg': int(marker[0][1])}


def capability_checks(recorder, task, *, context, context_path, expected_context_sha256, executions,
                      measure_dir, actual, baseline, electric, electrical_baseline, guard, tolerances):
    """One concrete check per T2 capability, for qualification or a final.

    executions are folders holding eco/<die>, readback-<die> and
    physical-checks.json (the two executor smoke tests during qualification,
    the candidate ECO in a final). baseline is the qualified reference
    measurement (the reference itself during qualification).
    """
    raw = read_json(measure_dir / 'result.json')
    sinks_table = measure_dir / 'assembly/passive/clock_sinks_by_clock.tsv'
    logs = [eco_log(e, d) for e in executions for d in DIES]
    log_paths = [x['path'] for x in logs if x['path'].is_file()]
    eco_results = [e / 'eco/result.json' for e in executions]
    readbacks = [e / ('readback-' + d) / 'result.json' for e in executions for d in DIES]
    physical = [e / 'physical-checks.json' for e in executions]
    proofs = [read_json(p)['equivalence'] for p in physical]
    context_sha = sha(context_path)
    groups_ok = context['clock_groups'] == baseline['values']
    timing_ok = context['timing'] == baseline['timing']
    recorder.add('clock_branch_context',
        check='clock context bytes equal the natively extracted context and its clock groups/timing equal the reference STA',
        measured={'context_sha256': context_sha, 'groups_match_reference_sta': groups_ok,
                  'timing_match_reference_sta': timing_ok},
        threshold={'context_sha256': expected_context_sha256, 'groups_match_reference_sta': True,
                   'timing_match_reference_sta': True},
        passed=context_sha == expected_context_sha256 and groups_ok and timing_ok, sources=[context_path])
    statuses = [read_json(p).get('status') for p in eco_results]
    markers = [x['originals'] is not None for x in logs]
    recorder.add('clock_action_executor',
        check='trusted native executor applied the validated action plan on both dies and printed its completion marker',
        measured={'executor_status': statuses, 'markers': markers},
        threshold={'executor_status': 'succeeded', 'markers': True},
        passed=all(s == 'succeeded' for s in statuses) and all(markers), sources=eco_results + log_paths)
    recorder.add('independent_joint_sta',
        check='independent joint setup/hold skew STA succeeded and its scalar equals the host reduction of the per-sink table',
        measured={'status': raw['status'], 'measurement_mode': raw['measurement_mode'],
                  'reported_skew_ns': raw['clock_skew_ns'], 'reduced_skew_ns': actual['metric']},
        threshold={'status': 'succeeded', 'measurement_mode': 'skew_sta', 'max_abs_difference_ns': 1e-12},
        passed=raw['status'] == 'succeeded' and raw['measurement_mode'] == 'skew_sta'
            and abs(raw['clock_skew_ns'] - actual['metric']) <= 1e-12,
        sources=[measure_dir / 'result.json', sinks_table])
    proofs_ok = bool(proofs) and all(p.get('passed') is True for p in proofs)
    recorder.add('physical_equivalence',
        check='read-back netlists contract to the original netlist through the declared noninverting buffers only',
        measured=[{'proof': p.get('proof'), 'passed': p.get('passed')} for p in proofs], threshold={'passed': True},
        passed=proofs_ok, sources=physical + readbacks)
    recorder.add('fixed_cell_placement_check',
        check='native executor verified every original instance kept location/orient/master (error otherwise) and the read-back proof agrees',
        measured={'checked_original_instances': [x['originals'] for x in logs],
                  'readback_fixed_cell_proof': [p.get('passed') for p in proofs]},
        threshold={'checked_original_instances': '> 0 per die', 'readback_fixed_cell_proof': True},
        passed=all(x['originals'] for x in logs) and proofs_ok, sources=log_paths + readbacks)
    recorder.add('setup_hold_guard',
        check='independent setup/hold WNS/TNS do not regress beyond the frozen context tolerances',
        measured={'timing': actual['timing'], 'failures': guard['failures']},
        threshold={'reference_timing': baseline['timing'], 'tolerances': tolerances},
        passed=guard['passed'] is True, sources=[measure_dir / 'result.json'])
    plans = [read_json(p).get('validated_plan') or {} for p in eco_results]
    limits = context['limits']
    recorder.add('clock_budget_check',
        check='added clock buffer count/area of each validated plan stay within the frozen context limits',
        measured=[{'added_buffers': p.get('added_buffers'), 'added_area_um2': p.get('added_area_um2')} for p in plans],
        threshold={'max_added_buffers': limits['max_added_buffers'], 'max_added_area_um2': limits['max_added_area_um2']},
        passed=bool(plans) and all(type(p.get('added_buffers')) is int
            and isinstance(p.get('added_area_um2'), (int, float))
            and 0 <= p['added_buffers'] <= limits['max_added_buffers']
            and 0 <= p['added_area_um2'] <= limits['max_added_area_um2'] + 1e-9 for p in plans),
        sources=eco_results)
    exports = [export_log(measure_dir, d) for d in DIES]
    # During qualification the measurement is of the reference, so the export
    # check covers the reference; the smoke ECO legality comes from its log.
    recorder.add('placement_legality',
        check='trusted ECO script ran detailed_placement and check_placement -verbose to completion with no ERROR '
              'on both dies, and the independent measurement export re-ran check_placement without ERROR',
        measured={'eco': [x['clean'] for x in logs], 'measurement_export': [clean for _, clean, _ in exports]},
        threshold=True, passed=all(x['clean'] for x in logs) and all(clean for _, clean, _ in exports),
        sources=log_paths + [path for path, _, _ in exports if path.is_file()] or eco_results)
    floating = [read_json(p).get('statistics', {}).get('floating_pg') for p in readbacks]
    known = all(type(v) is int for v in floating)
    recorder.add('pg_connectivity',
        check='native read-back reports zero floating PG pins and the executor bound every PG pin to VDD/VSS',
        measured={'floating_pg': floating, 'executor_checked_pg_pins': [x['pg'] for x in logs]},
        threshold={'floating_pg': 0},
        passed=(all(v == 0 for v in floating) and all(x['pg'] for x in logs)) if known else None,
        sources=readbacks)
    failures = electrical_failures(electric, electrical_baseline)
    recorder.add('electrical_limits',
        check='clock-net max slew/capacitance/fanout violation count and worst excess ratio do not regress',
        measured=electric['clock'],
        threshold={'reference': electrical_baseline['clock'], 'ratio_tolerance': ELECTRICAL_TOLERANCE},
        passed=not failures, sources=[measure_dir / 'electrical.rpt'])
    recorder.add('clock_inventory_check',
        check='clock analysis groups and sink inventory equal the qualified reference inventory',
        measured={'groups': len(actual['expected']), 'sinks': sum(len(v) for v in actual['expected'].values())},
        threshold={'groups': len(baseline['expected']), 'sinks': sum(len(v) for v in baseline['expected'].values()),
                   'identical_inventory': True},
        passed=bool(actual['expected']) and actual['expected'] == baseline['expected'], sources=[sinks_table])
    mismatched = []
    for e in executions:
        for d in DIES:
            row = read_json(e / 'eco' / d / 'result.json')
            for key, name in (('odb_sha256', 'design.odb'), ('log_sha256', 'native.log')):
                path = e / 'eco' / d / name
                if not path.is_file() or row.get(key) != sha(path):
                    mismatched.append(str(path.relative_to(e.parent)))
    recorder.add('artifact_consistency',
        check='native measurement artifacts and executor ODB/log hashes equal their recorded sha256',
        measured={'measurement_artifacts': len(raw['artifacts']), 'executor_mismatches': mismatched},
        threshold={'executor_mismatches': []}, passed=not mismatched,
        sources=[measure_dir / 'result.json'] + [e / 'eco' / d / 'result.json' for e in executions for d in DIES])
    return recorder.complete(task.data['required_capabilities'])


def qualify(task, case, args, base, image, platform, qpath, scoring_version=CURRENT_SCORING_VERSION):
    reference = args.experiment / 'stage-reference-v1' / case
    run = layout_path(args.root, 'reference_root') / case / 'run'
    feature = args.experiment / 'local-context-v2' / case
    context = read_json(feature / 'clock-context.json')
    if read_json(feature / 'result.json')['context_sha256'] != sha(feature / 'clock-context.json'):
        raise ValueError('Context changed after native extraction')
    measure = feature / 'reference-measurement/native/measurement'
    if sha(measure / 'result.json') != context['reference_measurement_sha256']:
        raise ValueError('Context timing provenance mismatch')
    actual = measurement(measure)
    if actual['values'] != context['clock_groups'] or actual['timing'] != context['timing']:
        raise ValueError('Clock context differs from native STA')
    dispatch = read_json(feature / 'reference-measurement/dispatch.json')
    for die in DIES:
        if dispatch['input_odb_sha256'][die] != sha(reference / ('normalized-' + die) / 'native/design.odb'):
            raise ValueError('Reference ODB changed')
    prefix = verify_prefix(run)
    replay = read_json(reference / 'replay/input-replay.json')
    if replay['seed'] != task.data['seed'] or context['seed'] != task.data['seed']:
        raise ValueError('Frozen input seed differs from task')
    if replay['prefix'] != prefix or any(sha(base / p) != h for p, h in replay['source_sha256'].items()):
        raise ValueError('Current reference source/prefix differs')
    old = {d: reference / ('readback-' + d) for d in DIES}
    electrical_ref = electrical_check(measure, old)
    # A deterministic connectivity-only qualification, not a candidate search:
    # exercise the maximum chain on one branch in each die, no QoR measurement.
    qpath.mkdir(parents=True, exist_ok=False)
    smoke = qpath / 'executor-smoke'
    actions = []
    for die in DIES:
        keys = sorted(k for k, v in context['actions'].items() if v['die'] == die and v['added_buffers'] == 8)
        if not keys:
            raise ValueError('No maximum-chain native qualification action')
        test = smoke / die
        test.mkdir(parents=True)
        additions = execute(context=context, plan={'actions': keys[:1]}, reference=reference,
            platform=platform, image=image, output=test / 'eco')
        actions.append({'action': keys[0], 'proof': readback(test, reference=reference, run=run,
            platform=platform, image=image, additions=additions)})
    facts = {'context_sha256': sha(feature / 'clock-context.json'), 'measurement_sha256': sha(measure / 'result.json'),
        'native_executor_tests': actions, 'prefix_sha256': digest(prefix), 'electrical': electrical_ref,
        'limits': context['limits'], 'timing_tolerances': context['timing_tolerances'],
        'qualification_tests_have_no_candidate_score': True, 'scoring_version': scoring_version}
    write_new(qpath / 'evidence.json', facts)
    write_new(qpath / 'reference.json', {'legal': True, 'case': case, 'endpoint': task.data['endpoint'],
        'primary_metric': actual['metric'], 'measurements': actual['values'], 'expected_inventory': actual['expected'],
        'timing': actual['timing'], 'timing_tolerances': context['timing_tolerances']})
    guard = timing_guard(actual['timing'], actual['timing'], context['timing_tolerances'])
    capabilities = capability_checks(Recorder(qpath), task, context=context,
        context_path=feature / 'clock-context.json',
        expected_context_sha256=read_json(feature / 'result.json')['context_sha256'],
        executions=[smoke / d for d in DIES], measure_dir=measure, actual=actual, baseline=actual,
        electric=electrical_ref, electrical_baseline=electrical_ref, guard=guard,
        tolerances=context['timing_tolerances'])
    q = {'kind': 'engineering-native-qualification-v1', 'task': task.task_id, 'task_digest': task.digest(),
        'case': case, 'seed': task.data['seed'], 'input_stage': task.data['input_stage'], 'endpoint': task.data['endpoint'],
        'physical_success': True, 'source_sha256': digest(replay['source_sha256']),
        'runtime_sha256': digest({'image': image, 'binary': replay['native_binary_sha256']}),
        'input_sha256': digest(dispatch['input_odb_sha256']),
        'capabilities': capabilities, 'reference': evidence(qpath / 'reference.json', qpath)}
    q.update(current_identity())
    q.update(qualification_fields(task, scoring_version))
    write_new(qpath / 'qualification.json', q)
    return q, context, actual, electrical_ref


def final_case(task, case, *, args, ledger, sealed, q, qpath, context, baseline, old_electrical,
               release, platform, image, scoring_version):
    """One candidate dispatch: plan, native ECO, read back, measure, gate, receipt."""
    folder = args.output / case
    folder.mkdir()
    feature = args.experiment / 'local-context-v2' / case
    plan = sample_plan(args.subject, feature / 'clock-context.json', image, folder / 'policy')
    verified = validate_clock_plan(context, plan)
    reference = args.experiment / 'stage-reference-v1' / case
    additions = execute(context=context, plan=plan, reference=reference, platform=platform, image=image, output=folder / 'eco')
    run = layout_path(args.root, 'reference_root') / case / 'run'
    proof = readback(folder, reference=reference, run=run, platform=platform, image=image, additions=additions)
    measure_args = argparse.Namespace(release=str(release), platform=str(platform), run=str(run),
        bottom_odb=str(folder / 'eco/bottom/design.odb'), top_odb=str(folder / 'eco/top/design.odb'),
        output=str(folder / 'independent-sta'), measurement_mode='skew_sta')
    if launch(measure_args):
        raise ValueError('Candidate native STA failed')
    mpath = folder / 'independent-sta/native/measurement'
    actual = measurement(mpath)
    clock_skew(actual['values'], baseline['expected'])
    electric = electrical_check(mpath, {d: folder / ('readback-' + d) for d in DIES}, old_electrical)
    guard = timing_guard(baseline['timing'], actual['timing'], context['timing_tolerances'])
    facts = {'actual': actual, 'plan': verified, 'physical_proof': proof, 'electrical': electric, 'setup_hold': guard,
             'scoring_version': scoring_version}
    write_new(folder / 'final-checks.json', facts)
    if not guard['passed']:
        raise ValueError('Setup/hold nonregression failed: ' + ','.join(guard['failures']))
    verify_seal(args.subject, sealed)
    recorder = Recorder(folder)
    capabilities = capability_checks(recorder, task, context=context, context_path=feature / 'clock-context.json',
        expected_context_sha256=read_json(qpath / 'evidence.json')['context_sha256'], executions=[folder],
        measure_dir=mpath, actual=actual, baseline=baseline, electric=electric,
        electrical_baseline=old_electrical, guard=guard, tolerances=context['timing_tolerances'])
    if recorder.failures():
        raise ValueError('Failed physical capability: ' + ', '.join(recorder.failures()))
    write_new(folder / 'final-measurements.json', {'values': actual['values'], 'timing': actual['timing']})
    native = {'kind': 'engineering-native-result-v1', 'task': task.task_id, 'task_digest': task.digest(),
        'case': case, 'seed': task.data['seed'], 'input_sha256': q['input_sha256'], 'evaluator_sha256': q['evaluator_sha256'],
        'endpoint': task.data['endpoint'], 'submission_sha256': sealed['submission_sha256'], 'runtime_sha256': q['runtime_sha256'],
        'qualification_sha256': sha(qpath / 'qualification.json'), 'checks': capabilities,
        'measurements': evidence(folder / 'final-measurements.json', folder)}
    if scoring_version == 2:
        native['scoring_version'] = 2
    write_new(folder / 'final-native.json', native)
    result = validate_native_result(task, case, qpath / 'qualification.json', folder / 'final-native.json',
        submission_sha256=sealed['submission_sha256'], runtime_sha256=q['runtime_sha256'])
    return ledger.finish(case, succeeded=True, legal=True, candidate_metric=result['candidate_metric'],
        artifacts=[str(folder / 'final-native.json')], details=result.get('details'))


def main(argv=None):
    p = argparse.ArgumentParser()
    for key in ('root', 'experiment', 'output', 'subject', 'baseline_inventory'):
        p.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--scoring-version', type=int, choices=(1, 2), default=CURRENT_SCORING_VERSION,
                   help='1 (default): frozen headline; 2: explicitly opt in to capability evidence and calibration')
    args = p.parse_args(argv)
    task = load_task('clock_compensation')
    contract = validate_development_binding(task, args.subject, args.scoring_version)
    if not set(args.cases) <= set(task.data['cases']) or len(set(args.cases)) != len(args.cases):
        p.error('Invalid cases')
    args.output.mkdir(parents=True, exist_ok=False)
    release = layout_path(args.root, 'release')
    base, image = verify_release(release)
    platform = layout_path(args.root, 'platform')
    sealed = seal(task, args.subject, read_json(args.baseline_inventory), args.output / 'sealed.json')
    write_new(args.output / 'queue.json', {'task': task.task_id, 'cases': args.cases, 'started_at': now(),
        'submission_sha256': sealed['submission_sha256'], 'budget': 'one native final per case; no retry',
        'scoring_version': args.scoring_version, 'scoring_contract': contract})
    ledger = OnceLedger(args.output / 'ledger', task, sealed['submission_sha256'])
    with (args.experiment / 'engineering-physical.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for case in args.cases:
            while not admission(args.output):
                time.sleep(30)
            try:
                verify_seal(args.subject, sealed)
                qpath = args.output / 'qualifications' / case
                q, context, baseline, old_electrical = qualify(task, case, args, base, image, platform, qpath,
                                                               args.scoring_version)
            except Exception as exc:
                write_new(args.output / ('qualification-failed-' + case + '.json'), {'error': repr(exc), 'time': now(), 'candidate_finals': 0})
                print(json.dumps({'case': case, 'qualification_failed': repr(exc)}), flush=True)
                continue
            ledger.claim(case, qpath / 'qualification.json')
            print(json.dumps({'case': case, 'status': 'candidate_final_dispatched', 'time': now()}), flush=True)
            try:
                row = final_case(task, case, args=args, ledger=ledger, sealed=sealed, q=q, qpath=qpath,
                    context=context, baseline=baseline, old_electrical=old_electrical, release=release,
                    platform=platform, image=image, scoring_version=args.scoring_version)
            except Exception as exc:
                row = ledger.finish(case, succeeded=False, legal=False, error=repr(exc))
            print(json.dumps(row), flush=True)
    write_new(args.output / 'result.json', aggregate(task, ledger.records()))


if __name__ == '__main__':
    main()
