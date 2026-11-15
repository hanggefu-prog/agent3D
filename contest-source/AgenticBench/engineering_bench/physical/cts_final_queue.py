"""Seal the existing T4 draft, qualify real references and dispatch each case once.

All gates are derived from independent readbacks and measurement. Candidate
code runs only in the isolated CTS subprocess, never in the host verifier.

Explicit --scoring-version 2 adds the hard gates declared in
engineering_tasks/cross_die_cts/evaluation-v2.json: setup/hold WNS/TNS
nonregression, an insertion-delay bound, tighter clock area/count ratios and a
clock-power bound when power is measured. Version 1 stays skew-only.
"""
from layout import layout_path
import argparse
import fcntl
import gzip
import json
import math
from pathlib import Path
from engineering_bench.identity import current_identity
import re
import time

from cts_reference_queue import reference_case, admission
from cts_reference_stage import verify_prefix
from engineering_bench.protocol import digest, sha, write_new, now, OnceLedger
from engineering_bench.submission import seal, verify_seal
from engineering_bench.tasks import load_task, read_json
from engineering_bench.input_audit import clock_table
from engineering_bench.legalization import load_snapshot, ports_geometry, is_fixed
from engineering_bench.native_snapshot import platform_libs
from engineering_bench.clock_equivalence import compare
from engineering_bench.metrics import clock_skew, clock_distribution, aggregate
from engineering_bench.evaluation import validate_native_result, t4_gates
from engineering_bench.prepare_development import verify_release
from engineering_bench.capabilities import Recorder
from engineering_bench.scoring import (load_profile, qualification_fields, CURRENT_SCORING_VERSION,
                                      LEGACY_CTS_POLICY, validate_development_binding)


# Fixed before candidate execution. Existing data electrical violations are
# diagnostics, not a requirement that CTS repair unrelated preexisting nets.
POLICY = dict(LEGACY_CTS_POLICY)


def evidence(path, root):
    return {'path': str(Path(path).relative_to(root)), 'sha256': sha(path)}


def check_artifacts(folder, result):
    for name, expected in result['artifacts'].items():
        path = (folder / name).resolve(strict=True)
        path.relative_to(folder.resolve())
        if sha(path) != expected:
            raise ValueError('Native artifact changed: ' + name)


def areas(platform):
    result = {}
    for die in ('bottom', 'top'):
        for path in platform_libs(platform, die):
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(path, 'rt') as stream:
                text = stream.read()
            cells = list(re.finditer(r'\bcell\s*\(\s*"?([^\s"()]+)"?\s*\)\s*\{', text))
            for n, match in enumerate(cells):
                body = text[match.end():cells[n + 1].start() if n + 1 < len(cells) else len(text)]
                area = re.search(r'\barea\s*:\s*([0-9.eE+-]+)\s*;', body)
                if area:
                    value = float(area[1])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError('Invalid Liberty area')
                    result[match[1]] = value
    return result


def electrical(path, clock_cells, sink_pins):
    """Native report values are compared as unit-free excess/limit ratios."""
    groups = {k: {'count': 0, 'worst_excess_ratio': 0.0} for k in ('max slew', 'max capacitance', 'max fanout')}
    section = None
    ignored = 0
    for line in Path(path).read_text().splitlines():
        if line.strip() in groups:
            section = line.strip()
        elif '(VIOLATED)' in line:
            fields = line.split()
            if section is None or len(fields) != 5:
                raise ValueError('Unknown electrical violation format')
            pin, limit, value, slack, _ = fields
            numbers = list(map(float, (limit, value, slack)))
            if not all(math.isfinite(v) for v in numbers) or numbers[0] <= 0:
                raise ValueError('Invalid electrical report value')
            if pin not in sink_pins and pin.rsplit('/', 1)[0] not in clock_cells:
                ignored += 1
                continue
            group = groups[section]
            group['count'] += 1
            group['worst_excess_ratio'] = max(group['worst_excess_ratio'], max(0., numbers[1] / numbers[0] - 1.))
    return {'clock': groups, 'data_violation_rows_diagnostic': ignored}


def budget_failures(actual, reference, area_ratio=None, count_ratio=None):
    """Resource/electrical nonregression; v2 passes the tighter profile ratios."""
    area_ratio = POLICY['clock_area_max_reference_ratio'] if area_ratio is None else area_ratio
    count_ratio = POLICY['clock_buffer_count_max_reference_ratio'] if count_ratio is None else count_ratio
    failures = []
    if actual['clock_area_um2'] > reference['clock_area_um2'] * area_ratio + 1e-9:
        failures.append('clock_area')
    if actual['clock_buffer_count'] > reference['clock_buffer_count'] * count_ratio:
        failures.append('clock_buffer_count')
    for kind, values in actual['electrical']['clock'].items():
        old = reference['electrical']['clock'][kind]
        if values['count'] > old['count'] or values['worst_excess_ratio'] > old['worst_excess_ratio'] + POLICY['electrical_ratio_tolerance']:
            failures.append(kind)
    return failures


def budget_check(actual, reference, area_ratio=None, count_ratio=None):
    failures = budget_failures(actual, reference, area_ratio, count_ratio)
    if failures:
        raise ValueError('Clock resource/electrical guard failed: ' + ', '.join(failures))


def clock_power(raw):
    """Clock power is gated only if the native measurement reports it."""
    value = raw.get('clock_power_mw')
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return {'status': 'measured', 'value_mw': float(value)}
    return {'status': 'not_measured', 'value_mw': None,
            'reason': 'the independent CTS measurement emits no clock power report'}


def audit_stage(folder, *, experiment, case, run, platform, image):
    row = read_json(folder / 'result.json')
    if row['status'] not in ('stage_reference_measured', 'stage_candidate_measured'):
        raise ValueError('Stage did not complete independent measurement')
    measurement = folder / 'independent-skew/native/measurement'
    raw = read_json(measurement / 'result.json')
    if raw['status'] != 'succeeded' or raw['endpoint'] != 'post_cts_placement_rc_joint_skew':
        raise ValueError('Wrong or unsuccessful independent endpoint')
    check_artifacts(measurement, raw)
    if sha(measurement / 'result.json') != row['measurement_sha256']:
        raise ValueError('Measurement receipt changed')
    original = {d: experiment / 'native-audit-v3' / case / '60_opt' / d for d in ('bottom', 'top')}
    observed = {d: folder / ('readback-' + d) for d in ('bottom', 'top')}
    masters = areas(platform)
    clocks, clock_area, clock_count = set(), 0., 0
    for die in ('bottom', 'top'):
        dispatch, _, cells = load_snapshot(observed[die])
        old_dispatch, _, before = load_snapshot(original[die])
        if dispatch['image'] != image or old_dispatch['image'] != image or dispatch['libraries'] != old_dispatch['libraries']:
            raise ValueError('Runtime/Liberty mismatch')
        odb = folder / ('normalized-' + die) / 'native/design.odb'
        if sha(odb) != dispatch['input_odb_sha256']:
            raise ValueError('Readback ODB hash mismatch')
        norm = folder / ('normalized-' + die)
        norm_result = read_json(norm / 'result.json')
        if norm_result['status'] != 'succeeded':
            raise ValueError('Legalization failed')
        check_artifacts(norm, norm_result)
        if ports_geometry(original[die] / 'ports.tsv') != ports_geometry(observed[die] / 'ports.tsv'):
            raise ValueError('Fixed port geometry changed')
        for name, value in before.items():
            if not value['clock_cell'] and is_fixed(value):
                if any(value[k] != cells[name][k] for k in ('x_dbu', 'y_dbu', 'orient', 'master')):
                    raise ValueError('Fixed cell changed')
        for name, value in cells.items():
            if value['clock_cell']:
                clocks.add('u_' + die + '/' + name)
                clock_area += masters[value['master']]
                clock_count += 1
    proof = compare(original, observed, run / '60_opt/hbt.csv')
    values = clock_table(measurement / 'assembly/passive/clock_sinks_by_clock.tsv')
    expected = {k: sorted(v) for k, v in values.items()}
    metric = clock_skew(values, expected)
    if abs(metric - raw['clock_skew_ns']) > 1e-12 or clock_count == 0 or clock_area <= 0:
        raise ValueError('Invalid independent clock measurement')
    return {'values': values, 'expected_inventory': expected, 'clock_skew_ns': metric,
            'timing': raw['timing'], 'clock_area_um2': clock_area, 'clock_buffer_count': clock_count,
            'electrical': electrical(measurement / 'electrical.rpt', clocks, {p for v in values.values() for p in v}),
            'physical_equivalence': proof, 'measurement_sha256': sha(measurement / 'result.json'),
            'readback_sha256': {d: sha(observed[d] / 'result.json') for d in observed},
            'clock_power': clock_power(raw)}


def resources(actual):
    return {'clock_area_um2': actual['clock_area_um2'], 'clock_buffer_count': actual['clock_buffer_count']}


def legalization_log(folder, die):
    """Protected legalize.tcl runs check_placement; any ERROR aborts it."""
    path = folder / ('normalized-' + die) / 'native.log'
    text = path.read_text() if path.is_file() else ''
    marker = re.findall(r'^ENGINEERING_PROTECTED_LEGALIZATION restored=(\d+) pg=(\d+)$', text, re.M)
    clean = len(marker) == 1 and not re.search(r'^\[(?:WARNING|ERROR)\b', text, re.M)
    return path, clean, (int(marker[0][1]) if len(marker) == 1 else None)


def export_log(measurement, die):
    """Independent measurement EXPORT runs check_placement and a PG check per die."""
    path = Path(measurement) / die / 'export.log'
    text = path.read_text() if path.is_file() else ''
    marker = re.findall(r'^ENGINEERING_MEASUREMENT_EXPORT pg=(\d+)$', text, re.M)
    if len(marker) != 1:
        return path, False, None
    return path, not re.search(r'^\[ERROR\b', text, re.M), int(marker[0])


def capability_checks(recorder, task, folder, actual, *, baseline, run, prefix, replay,
                      expected_binary, area_ratio, count_ratio):
    """One concrete, separately evidenced check per T4 capability.

    baseline is the qualified reference audit (the reference itself during
    qualification). Checks that audit_stage enforces by raising are recorded
    as passed with what was measured; comparisons are recorded either way.
    """
    dies = ('bottom', 'top')
    measurement = folder / 'independent-skew/native/measurement'
    raw = read_json(measurement / 'result.json')
    sinks_table = measurement / 'assembly/passive/clock_sinks_by_clock.tsv'
    replay_path = folder / 'replay/input-replay.json'
    readbacks = [folder / ('readback-' + d) / 'result.json' for d in dies]
    normalized = [folder / ('normalized-' + d) / 'result.json' for d in dies]
    recorder.add('post_opt_checkpoint_import',
        check='replayed post-opt prefix equals the independently verified 60_opt prefix and the task seed',
        measured={'prefix_sha256': digest(replay['prefix']), 'seed': replay['seed']},
        threshold={'prefix_sha256': digest(prefix), 'seed': task.data['seed']},
        passed=replay['prefix'] == prefix and replay['seed'] == task.data['seed'], sources=[replay_path])
    recorder.add('independent_joint_sta',
        check='native joint skew endpoint succeeded and its scalar equals the host reduction of the per-sink table',
        measured={'status': raw['status'], 'endpoint': raw['endpoint'], 'reported_skew_ns': raw['clock_skew_ns'],
                  'reduced_skew_ns': actual['clock_skew_ns']},
        threshold={'status': 'succeeded', 'endpoint': task.data['endpoint'], 'max_abs_difference_ns': 1e-12},
        passed=raw['status'] == 'succeeded' and raw['endpoint'] == task.data['endpoint']
            and abs(raw['clock_skew_ns'] - actual['clock_skew_ns']) <= 1e-12,
        sources=[measurement / 'result.json', sinks_table])
    proof = actual['physical_equivalence']
    recorder.add('physical_equivalence',
        check='read-back netlists preserve every original data function and the clock buffer/inverter contraction',
        measured={'proof': proof.get('proof'), 'passed': proof.get('passed')}, threshold={'passed': True},
        passed=proof.get('passed') is True, sources=readbacks + [run / '60_opt/hbt.csv'])
    statuses = {d: read_json(p).get('status') for d, p in zip(dies, normalized)}
    recorder.add('harness_owned_data_legalization',
        check='harness-owned data-cell normalization/legalization succeeded on both dies',
        measured=statuses, threshold={d: 'succeeded' for d in dies},
        passed=all(v == 'succeeded' for v in statuses.values()), sources=normalized)
    old, new = baseline['expected_inventory'], actual['expected_inventory']
    recorder.add('clock_inventory_check',
        check='clock analysis groups and sink inventory equal the qualified reference inventory',
        measured={'groups': len(new), 'sinks': sum(len(v) for v in new.values())},
        threshold={'groups': len(old), 'sinks': sum(len(v) for v in old.values()), 'identical_inventory': True},
        passed=bool(new) and new == old, sources=[sinks_table])
    failures = budget_failures(actual, baseline, area_ratio, count_ratio)
    recorder.add('clock_budget_check', check='clock buffer area and count within the declared reference ratios',
        measured=resources(actual),
        threshold={'clock_area_um2': baseline['clock_area_um2'] * area_ratio,
                   'clock_buffer_count': baseline['clock_buffer_count'] * count_ratio,
                   'area_ratio': area_ratio, 'count_ratio': count_ratio},
        passed=not {'clock_area', 'clock_buffer_count'} & set(failures), sources=readbacks)
    recorder.add('candidate_native_runtime_binding',
        check='the replayed CTS stage executed the expected native OpenROAD binary',
        measured=replay['native_binary_sha256'], threshold=expected_binary,
        passed=replay['native_binary_sha256'] == expected_binary, sources=[replay_path])
    logs = [legalization_log(folder, d) for d in dies]
    exports = [export_log(measurement, d) for d in dies]
    recorder.add('placement_legality',
        check='protected legalize.tcl (detailed_placement + check_placement) completed without ERROR/WARNING and '
              'the independent measurement export re-ran check_placement without ERROR on both dies',
        measured={'legalization': {d: clean for d, (_, clean, _) in zip(dies, logs)},
                  'measurement_export': {d: clean for d, (_, clean, _) in zip(dies, exports)}},
        threshold={'legalization': {d: True for d in dies}, 'measurement_export': {d: True for d in dies}},
        passed=all(clean for _, clean, _ in logs + exports),
        sources=[p for p, _, _ in logs + exports if p.is_file()] or normalized)
    floating = {d: read_json(p).get('statistics', {}).get('floating_pg') for d, p in zip(dies, readbacks)}
    known = all(type(v) is int for v in floating.values())
    recorder.add('pg_connectivity', check='native readback reports zero floating PG pins on both dies',
        measured={'floating_pg': floating, 'legalization_checked_pg_pins': {d: n for d, (_, _, n) in zip(dies, logs)},
                  'export_checked_pg_pins': {d: n for d, (_, _, n) in zip(dies, exports)}},
        threshold={'floating_pg': 0},
        passed=all(v == 0 for v in floating.values()) if known else None, sources=readbacks)
    electric = [k for k in failures if k not in ('clock_area', 'clock_buffer_count')]
    recorder.add('electrical_limits',
        check='clock-net max slew/capacitance/fanout violation count and worst excess ratio do not regress',
        measured=actual['electrical']['clock'],
        threshold={'reference': baseline['electrical']['clock'], 'ratio_tolerance': POLICY['electrical_ratio_tolerance']},
        passed=not electric, sources=[measurement / 'electrical.rpt'])
    recorder.add('artifact_consistency',
        check='every native measurement/legalization artifact matches its recorded sha256 (check_artifacts)',
        measured={'measurement_artifacts': len(raw['artifacts']),
                  'legalization_artifacts': sum(len(read_json(p).get('artifacts', {})) for p in normalized)},
        threshold='all recorded hashes match', passed=True, sources=[measurement / 'result.json'] + normalized)
    return recorder.complete(task.data['required_capabilities'])


def _ratios(gates):
    if gates is None:
        return POLICY['clock_area_max_reference_ratio'], POLICY['clock_buffer_count_max_reference_ratio']
    return gates['clock_area_max_reference_ratio'], gates['clock_buffer_count_max_reference_ratio']


def qualify_reference(task, case, folder, *, experiment, root, output, platform, image, source, sealed,
                      scoring_version=CURRENT_SCORING_VERSION):
    output.mkdir(parents=True, exist_ok=False)
    run = layout_path(root, 'reference_root') / case / 'run'
    prefix = verify_prefix(run)
    actual = audit_stage(folder, experiment=experiment, case=case, run=run, platform=platform, image=image)
    replay = read_json(folder / 'replay/input-replay.json')
    if replay['prefix'] != prefix or replay['seed'] != task.data['seed']:
        raise ValueError('Reference prefix/config mismatch')
    for path, expected in replay['source_sha256'].items():
        if sha(source / path) != expected:
            raise ValueError('Reference physical source changed')
    # This runner explicitly rejects native changes instead of running an old binary.
    if any(p.startswith('OpenROAD-3D/') for p in sealed['changed_paths']) and not sealed.get('candidate_runtime_verified'):
        raise ValueError('Native candidate needs separately qualified candidate image')
    budget_check(actual, actual)
    gates = load_profile(task)[0]['gates'] if scoring_version == 2 else None
    check = {'policy': POLICY, 'reference_folder': str(folder), 'actual': actual,
             'prefix_sha256': digest(prefix), 'image': image, 'native_binary_sha256': replay['native_binary_sha256'],
             'python_candidate_scope': sealed['changed_paths'], 'checks_derived_from_native_artifacts': True,
             'scoring_version': scoring_version, 'gates_v2': gates}
    write_new(output / 'evidence.json', check)
    reference = {'legal': True, 'case': case, 'endpoint': task.data['endpoint'],
                 'primary_metric': actual['clock_skew_ns'], 'measurements': actual['values'],
                 'expected_inventory': actual['expected_inventory'], 'timing': actual['timing']}
    if scoring_version == 2:
        reference.update(clock_resources=resources(actual), clock_power=actual['clock_power'],
                         skew_distribution=clock_distribution(actual['values'], actual['expected_inventory']))
    write_new(output / 'reference.json', reference)
    area_ratio, count_ratio = _ratios(gates)
    capabilities = capability_checks(Recorder(output), task, folder, actual, baseline=actual, run=run,
        prefix=prefix, replay=replay, expected_binary=replay['native_binary_sha256'],
        area_ratio=area_ratio, count_ratio=count_ratio)
    q = {'kind': 'engineering-native-qualification-v1', 'task': task.task_id,
         'task_digest': task.digest(), 'case': case, 'seed': task.data['seed'],
         'input_stage': task.data['input_stage'], 'endpoint': task.data['endpoint'], 'physical_success': True,
         'source_sha256': digest(replay['source_sha256']),
         'runtime_sha256': digest({'image': image, 'binary': replay['native_binary_sha256']}),
         'input_sha256': digest(prefix), 'capabilities': capabilities,
         'reference': evidence(output / 'reference.json', output)}
    q.update(current_identity())
    q.update(qualification_fields(task, scoring_version))
    write_new(output / 'qualification.json', q)
    return q, actual


def final_case(task, case, *, args, ledger, sealed, q, qpath, baseline, release, platform, image,
               execution_image, expected_binary, scoring_version):
    """One candidate dispatch: execute, read back, measure, gate, score, receipt."""
    folder = args.output / case
    run = layout_path(args.root, 'reference_root') / case / 'run'
    reference_case(case, root=args.root, experiment=args.experiment, output=folder,
        release_path=release, source=args.subject, image=image, platform=platform, role='candidate',
        execution_image=execution_image)
    replay = read_json(folder / 'replay/input-replay.json')
    if replay['native_binary_sha256'] != expected_binary:
        raise ValueError('CTS executed an unexpected native binary')
    actual = audit_stage(folder, experiment=args.experiment, case=case, run=run, platform=platform, image=image)
    gates = load_profile(task)[0]['gates'] if scoring_version == 2 else None
    area_ratio, count_ratio = _ratios(gates)
    budget_check(actual, baseline, area_ratio, count_ratio)
    clock_skew(actual['values'], baseline['expected_inventory'])
    verify_seal(args.subject, sealed)
    measurements = {'values': actual['values'], 'timing': actual['timing']}
    checks = {'policy': POLICY, 'actual': actual, 'baseline_measurement_sha256': baseline['measurement_sha256'],
        'execution_image': execution_image, 'executed_binary_sha256': expected_binary,
        'candidate_runtime_sha256': sha(args.candidate_runtime) if args.candidate_runtime else None,
        'scoring_version': scoring_version}
    gate_error = None
    if scoring_version == 2:
        measurements.update(clock_resources=resources(actual), clock_power=actual['clock_power'])
        try:
            checks['gates_v2'] = dict(t4_gates(read_json(qpath / 'reference.json'), measurements, gates), passed=True)
        except ValueError as exc:
            gate_error = str(exc)
            checks['gates_v2'] = {'passed': False, 'error': gate_error}
    write_new(folder / 'final-checks.json', checks)
    recorder = Recorder(folder)
    capabilities = capability_checks(recorder, task, folder, actual, baseline=baseline, run=run,
        prefix=verify_prefix(run), replay=replay, expected_binary=expected_binary,
        area_ratio=area_ratio, count_ratio=count_ratio)
    if recorder.failures():
        raise ValueError('Failed physical capability: ' + ', '.join(recorder.failures()))
    if gate_error:
        raise ValueError(gate_error)
    write_new(folder / 'final-measurements.json', measurements)
    raw = {'kind': 'engineering-native-result-v1', 'task': task.task_id, 'task_digest': task.digest(),
        'case': case, 'seed': task.data['seed'], 'input_sha256': q['input_sha256'],
        'evaluator_sha256': q['evaluator_sha256'], 'endpoint': task.data['endpoint'],
        'submission_sha256': sealed['submission_sha256'], 'runtime_sha256': q['runtime_sha256'],
        'qualification_sha256': sha(qpath / 'qualification.json'), 'checks': capabilities,
        'measurements': evidence(folder / 'final-measurements.json', folder)}
    if scoring_version == 2:
        raw['scoring_version'] = 2
    write_new(folder / 'final-native.json', raw)
    validated = validate_native_result(task, case, qpath / 'qualification.json', folder / 'final-native.json',
        submission_sha256=sealed['submission_sha256'], runtime_sha256=q['runtime_sha256'])
    return ledger.finish(case, succeeded=True, legal=True, candidate_metric=validated['candidate_metric'],
        artifacts=[str(folder / 'final-native.json')], details=validated.get('details'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'experiment', 'output', 'subject', 'baseline_inventory'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--candidate-runtime', type=Path)
    parser.add_argument('--scoring-version', type=int, choices=(1, 2), default=CURRENT_SCORING_VERSION,
                        help='1 (default): frozen skew-only scoring; 2: explicitly opt in to v2 gates and calibration')
    args = parser.parse_args(argv)
    task = load_task('cross_die_cts')
    contract = validate_development_binding(task, args.subject, args.scoring_version)
    if not args.cases or len(set(args.cases)) != len(args.cases) or not set(args.cases) <= set(task.data['cases']):
        parser.error('Unexpected or repeated cases')
    args.output.mkdir(parents=True, exist_ok=False)
    release = layout_path(args.root, 'release')
    base, image = verify_release(release)
    platform = layout_path(args.root, 'platform')
    sealed = seal(task, args.subject, read_json(args.baseline_inventory), args.output / 'sealed.json')
    execution_image = image
    qualification_seal = sealed
    receipt = None
    if args.candidate_runtime:
        from engineering_bench.candidate_runtime import verify
        receipt = read_json(args.candidate_runtime)
        execution_image = verify(args.subject, sealed, receipt, image)
        qualification_seal = {**sealed, 'candidate_runtime_verified': True}
    write_new(args.output / 'queue.json', {'task': task.task_id, 'cases': args.cases, 'started_at': now(),
        'submission_sha256': sealed['submission_sha256'], 'policy': POLICY, 'model_calls': 0,
        'scoring_version': args.scoring_version, 'scoring_contract': contract})
    ledger = OnceLedger(args.output / 'ledger', task, sealed['submission_sha256'])
    with (args.experiment / 'engineering-physical.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in args.cases:
            while not admission(args.output):
                print(json.dumps({'case': case, 'status': 'waiting_for_resources', 'time': now()}), flush=True)
                time.sleep(30)
            verify_seal(args.subject, sealed)
            qpath = args.output / 'qualifications' / case
            q, baseline = qualify_reference(task, case, args.experiment / 'stage-reference-v1' / case,
                experiment=args.experiment, root=args.root, output=qpath, platform=platform, image=image,
                source=base, sealed=qualification_seal, scoring_version=args.scoring_version)
            ledger.claim(case, qpath / 'qualification.json')
            print(json.dumps({'case': case, 'status': 'candidate_final_dispatched', 'time': now()}), flush=True)
            try:
                expected_binary = (receipt['binary_sha256'] if args.candidate_runtime
                                   else read_json(qpath / 'evidence.json')['native_binary_sha256'])
                row = final_case(task, case, args=args, ledger=ledger, sealed=sealed, q=q, qpath=qpath,
                    baseline=baseline, release=release, platform=platform, image=image,
                    execution_image=execution_image, expected_binary=expected_binary,
                    scoring_version=args.scoring_version)
            except Exception as exc:
                row = ledger.finish(case, succeeded=False, legal=False, error=type(exc).__name__ + ': ' + str(exc))
            print(json.dumps(row), flush=True)
    write_new(args.output / 'result.json', aggregate(task, ledger.records()))


if __name__ == '__main__':
    main()
