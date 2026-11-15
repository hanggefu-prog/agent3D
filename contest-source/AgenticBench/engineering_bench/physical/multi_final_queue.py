"""One sealed T3 transformation/case after real multi-link fixture qualification.

Explicit --scoring-version 2 scores every original receiver of each target
net, the remote (sink-die) receivers and the local (source-die, same net)
receivers, reports both subsets, and bounds the added isolation-buffer area
(engineering_tasks/multi_hbt_fanout/evaluation-v2.json). It needs a context
whose target nets list original_local_receivers. Version 1 keeps the frozen
remote-receiver metric.
"""
from layout import layout_path
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from cts_reference_queue import admission
from cts_reference_stage import verify_prefix
from cts_final_queue import evidence, electrical, export_log
from clock_final_queue import measurement
from local_query import query
from target_delay import parse_delays
from multi_fixture import fixture
from multi_readback import validate as branch_readback
from measure_host import launch, bind
from engineering_bench.capabilities import Recorder
from engineering_bench.identity import current_identity, verify_evaluator
from engineering_bench.legalization import load_snapshot
from engineering_bench.protocol import write_new, read_json, digest, sha, now, OnceLedger, qualify as verify_qualification
from engineering_bench.scoring import (load_profile, qualification_fields, scoring_version as record_scoring_version,
                                       CURRENT_SCORING_VERSION, validate_development_binding)
from engineering_bench.tasks import load_task
from engineering_bench.submission import seal, verify_seal
from engineering_bench.prepare_development import verify_release
from engineering_bench.native_snapshot import snapshot
from engineering_bench.metrics import target_delay, clock_skew, timing_guard, aggregate, receiver_subset
from engineering_bench.evaluation import validate_native_result

DIES = ('bottom', 'top')
ELECTRICAL_TOLERANCE = 1e-6


def snapshots(folder, *, output_odbs, reference, run, platform, image, context, plan):
    old, new = {}, {}
    for die in DIES:
        old[die], new[die] = reference / ('readback-' + die), folder / ('readback-' + die)
        result = snapshot(run=run, platform=platform, image=image, stage='70_cts', die=die,
            output=new[die], clock_sdc=run / '60_opt' / die / 'design.sdc', odb_path=output_odbs / die / 'design.odb')
        if result['status'] != 'succeeded' or result['statistics']['floating_pg']:
            raise ValueError('Multi-HBT readback/PG failed')
    return branch_readback(context, plan, old, new, run / '60_opt/hbt.csv', folder)


def sta(folder, *, output_odbs, root, source, run, platform, image, context, readback):
    args = argparse.Namespace(release=str(layout_path(root, 'release')),
        platform=str(platform), run=str(run), bottom_odb=str(output_odbs / 'bottom/design.odb'),
        top_odb=str(output_odbs / 'top/design.odb'), output=str(folder / 'independent-sta'), measurement_mode='skew_sta',
        additional_hbt_links=str(folder / 'additional-links.json'))
    if launch(args):
        raise ValueError('Independent multi-HBT placement-RC STA failed')
    query_path = query(source=source, platform=platform, run=run, measurement_root=folder / 'independent-sta',
        image=image, output=folder / 'target-delay', mode='delays', context=context, added_cells=readback['cells_by_target'])
    return measurement(folder / 'independent-sta/native/measurement'), parse_delays(query_path, context)


def inventories(context, version):
    """Evaluator-owned receiver inventories; v2 refuses contexts without local receivers."""
    targets = context['target_nets']
    remote = {n: sorted(t['original_remote_receivers']) for n, t in targets.items()}
    if version == 1:
        return {'all': remote, 'remote': remote, 'local': {n: [] for n in targets}}
    missing = sorted(n for n, t in targets.items() if 'original_local_receivers' not in t)
    if missing:
        raise ValueError('T3 scoring v2 needs a context listing original_local_receivers '
                         '(rebuild it with prepare_multi_contexts, or use --scoring-version 1): ' + ', '.join(missing))
    local = {n: sorted(t['original_local_receivers']) for n, t in targets.items()}
    if any(set(remote[n]) & set(local[n]) for n in targets):
        raise ValueError('A receiver is both local and remote')
    return {'all': {n: sorted(remote[n] + local[n]) for n in targets}, 'remote': remote, 'local': local}


def buffer_area_budget(context, ratio):
    """ratio x the smallest provided buffer area on each target's source die, summed."""
    total = 0.
    for name, target in sorted(context['target_nets'].items()):
        areas = [float(c['area_um2']) for c in context['buffer_cells'] if c['die'] == target['source_die']]
        if not areas or min(areas) <= 0:
            raise ValueError('No positive buffer area for target source die: ' + name)
        total += min(areas)
    return ratio * total


def added_buffer_area(context, folder, readback):
    """Area of the read-back isolation buffers, by native master and provided Liberty area."""
    areas = {(c['name'], c['die']): float(c['area_um2']) for c in context['buffer_cells']}
    cells = {d: load_snapshot(folder / ('readback-' + d))[2] for d in DIES}
    total, rows = 0., []
    for die, inst in readback['additional_buffers']:
        master = cells[die][inst]['master']
        if (master, die) not in areas:
            raise ValueError('Added buffer master has no provided area: ' + master)
        total += areas[(master, die)]
        rows.append({'die': die, 'instance': inst, 'master': master, 'area_um2': areas[(master, die)]})
    return total, rows


def electrical_rows(path):
    pins = {line.split()[0] for line in Path(path).read_text().splitlines() if '(VIOLATED)' in line}
    return electrical(path, set(), pins)['clock']


def check_electrical(report, reference):
    # Compare all native data/clock violations. Nonregression allows unchanged
    # baseline violations but never ignores a newly added buffer violation.
    old, new = electrical_rows(reference), electrical_rows(report)
    if any(new[k]['count'] > old[k]['count'] or new[k]['worst_excess_ratio'] > old[k]['worst_excess_ratio'] + ELECTRICAL_TOLERANCE for k in old):
        raise ValueError('Global electrical nonregression failed')
    return new


def capability_checks(recorder, task, *, context, context_path, expected_context_sha256, delays_path, execution,
                      query_path, values, inventory, actual, baseline, guard, tolerances, electrical_report,
                      electrical_reference, area=None, area_budget=None):
    """One concrete check per T3 capability.

    execution holds branch-readback.json, additional-links.json, readback-<die>
    and independent-sta (the multi-link fixture during qualification, the
    candidate in a final).
    """
    measure_dir = execution / 'independent-sta/native/measurement'
    raw = read_json(measure_dir / 'result.json')
    receipt_path = execution / 'branch-readback.json'
    links_path = execution / 'additional-links.json'
    receipt = read_json(receipt_path)
    readbacks = [execution / ('readback-' + d) / 'result.json' for d in DIES]
    targets = context['target_nets']
    context_sha = sha(context_path)
    recorder.add('fixed_target_net_context',
        check='target context bytes equal the frozen extracted context, and its baseline delays come from the hashed native query',
        measured={'context_sha256': context_sha, 'baseline_delay_sha256': sha(delays_path),
                  'pending': context.get('baseline_delay_pending_native_measurement')},
        threshold={'context_sha256': expected_context_sha256, 'baseline_delay_sha256': context['baseline_delay_sha256'],
                   'pending': False},
        passed=context_sha == expected_context_sha256 and sha(delays_path) == context['baseline_delay_sha256']
            and context.get('baseline_delay_pending_native_measurement') is False,
        sources=[context_path, delays_path])
    branches = receipt['branches']
    new_sites = [b['site_id'] for b in branches.values() if b['hbt_id'] != targets[b['logical_net']]['original_hbt_id']]
    recorder.add('legal_spare_hbt_sites',
        check='every added HBT sits on a distinct reserved legal spare site (native anchor/port geometry read back)',
        measured={'added_hbt_sites': sorted(new_sites)}, threshold={'subset_of_spare_site_ids': True, 'distinct': True},
        passed=set(new_sites) <= set(context['spare_site_ids']) and len(new_sites) == len(set(new_sites)),
        sources=[receipt_path])
    resources = receipt['resources']
    covered = sorted({b['logical_net'] for b in branches.values()})
    recorder.add('branch_materialization_api',
        check='the materialized ODBs read back into one branch set per frozen target net with declared links',
        measured={'targets_with_branches': len(covered), 'branch_count': resources['branch_count'],
                  'declared_added_links': len(read_json(links_path)['links'])},
        threshold={'targets_with_branches': len(targets), 'declared_added_links': resources['added_hbts']},
        passed=covered == sorted(targets) and len(read_json(links_path)['links']) == resources['added_hbts'],
        sources=[receipt_path, links_path])
    recorder.add('independent_branch_validator',
        check='protected validator rebuilt branches from native snapshots: single driver, noninverting isolation, receivers preserved',
        measured={'target_count': resources['target_count'],
                  'proofs': all(b.get('noninverting_buffer_proof') is True and b.get('single_driver') is True
                                for b in branches.values())},
        threshold={'target_count': len(targets), 'proofs': True},
        passed=resources['target_count'] == len(targets) and resources.get('requires_native_validation') is True
            and all(b.get('noninverting_buffer_proof') is True and b.get('single_driver') is True for b in branches.values()),
        sources=[receipt_path] + readbacks)
    recorder.add('placement_rc_joint_sta',
        check='independent passive multi-HBT placement-RC STA succeeded and its skew scalar equals the host reduction',
        measured={'status': raw['status'], 'measurement_mode': raw['measurement_mode'],
                  'reported_skew_ns': raw['clock_skew_ns'], 'reduced_skew_ns': actual['metric']},
        threshold={'status': 'succeeded', 'measurement_mode': 'skew_sta', 'max_abs_difference_ns': 1e-12},
        passed=raw['status'] == 'succeeded' and raw['measurement_mode'] == 'skew_sta'
            and abs(raw['clock_skew_ns'] - actual['metric']) <= 1e-12,
        sources=[measure_dir / 'result.json', links_path])
    counts = {n: len(values.get(n, {})) for n in sorted(targets)}
    recorder.add('fixed_sink_delay_measurement',
        check='fixed-slew native graph delay measured for every evaluator-owned receiver of every target net',
        measured={'receivers_measured': sum(counts.values())},
        threshold={'receivers_required': sum(len(v) for v in inventory.values())},
        passed=set(values) == set(inventory) and all(set(inventory[n]) <= set(values[n]) for n in inventory),
        sources=[query_path])
    proof = receipt['physical_equivalence']
    recorder.add('physical_equivalence',
        check='read-back netlists contract to the original through declared isolation buffers and HBT anchors only',
        measured={'proof': proof.get('proof'), 'passed': proof.get('passed')}, threshold={'passed': True},
        passed=proof.get('passed') is True, sources=[receipt_path] + readbacks)
    recorder.add('setup_hold_guard',
        check='independent setup/hold WNS/TNS and hold endpoints do not regress beyond frozen tolerances',
        measured={'timing': actual['timing'], 'failures': guard['failures']},
        threshold={'reference_timing': baseline['timing'], 'tolerances': tolerances},
        passed=guard['passed'] is True, sources=[measure_dir / 'result.json'])
    limit = context['limits']['max_added_hbts']
    budget_ok = resources['added_hbts'] <= limit
    measured = {'added_hbts': resources['added_hbts']}
    threshold = {'max_added_hbts': limit}
    if area_budget is not None:
        measured['added_buffer_area_um2'] = area
        threshold['added_buffer_area_budget_um2'] = area_budget
        budget_ok = budget_ok and area is not None and area <= area_budget + 1e-9
    recorder.add('hbt_resource_budget_check',
        check='added HBT count within the frozen limit' + (' and added isolation-buffer area within the declared bound'
                                                           if area_budget is not None else ''),
        measured=measured, threshold=threshold, passed=budget_ok, sources=[receipt_path] + readbacks)
    recorder.add('fixed_cell_placement_check',
        check='every original cell kept master/location/orientation/status in the native read-back (fixed-cell proof)',
        measured={'proof': proof.get('proof')}, threshold={'passed': True},
        passed=proof.get('passed') is True, sources=readbacks)
    exports = [export_log(measure_dir, d) for d in DIES]
    recorder.add('placement_legality',
        check='independent measurement export ran check_placement -verbose on each submitted die ODB without ERROR',
        measured={d: clean for d, (_, clean, _) in zip(DIES, exports)}, threshold={d: True for d in DIES},
        passed=all(clean for _, clean, _ in exports),
        sources=[p for p, _, _ in exports if p.is_file()] or [measure_dir / 'result.json'])
    floating = [read_json(p).get('statistics', {}).get('floating_pg') for p in readbacks]
    known = all(type(v) is int for v in floating)
    recorder.add('pg_connectivity',
        check='native read-back reports zero floating PG pins; the export bound every PG pin to VDD/VSS',
        measured={'floating_pg': floating, 'export_checked_pg_pins': [n for _, _, n in exports]},
        threshold={'floating_pg': 0},
        passed=(all(v == 0 for v in floating) and all(n for _, _, n in exports)) if known else None,
        sources=readbacks)
    old, new = electrical_rows(electrical_reference), electrical_rows(electrical_report)
    failures = [k for k in old if new[k]['count'] > old[k]['count']
                or new[k]['worst_excess_ratio'] > old[k]['worst_excess_ratio'] + ELECTRICAL_TOLERANCE]
    recorder.add('electrical_limits',
        check='all native max slew/capacitance/fanout violation counts and worst excess ratios do not regress',
        measured=new, threshold={'reference': old, 'ratio_tolerance': ELECTRICAL_TOLERANCE},
        passed=not failures, sources=[electrical_report, electrical_reference])
    recorder.add('clock_inventory_check',
        check='clock analysis groups and sinks equal the qualified original clock inventory',
        measured={'groups': len(actual['expected']), 'sinks': sum(len(v) for v in actual['expected'].values())},
        threshold={'groups': len(baseline['expected']), 'sinks': sum(len(v) for v in baseline['expected'].values()),
                   'identical_inventory': True},
        passed=bool(actual['expected']) and actual['expected'] == baseline['expected'],
        sources=[measure_dir / 'assembly/passive/clock_sinks_by_clock.tsv'])
    links = read_json(links_path)
    recorder.add('artifact_consistency',
        check='measurement artifacts match their hashes and the added-link file binds the exact branch read-back',
        measured={'measurement_artifacts': len(raw['artifacts']), 'links_readback_sha256': links.get('native_readback_sha256')},
        threshold={'links_readback_sha256': sha(receipt_path)},
        passed=links.get('native_readback_sha256') == sha(receipt_path),
        sources=[measure_dir / 'result.json', links_path, receipt_path])
    return recorder.complete(task.data['required_capabilities'])


def qualify(task, case, args, source, image, platform, output, scoring_version=CURRENT_SCORING_VERSION):
    context_root = args.context_root / case
    context = read_json(context_root / 'multi-context.json')
    if read_json(context_root / 'result.json')['context_sha256'] != sha(context_root / 'multi-context.json') or context['baseline_delay_pending_native_measurement']:
        raise ValueError('Unfinished or changed target context')
    delays = context_root / 'delays/native/target-delays.tsv'
    if sha(delays) != context['baseline_delay_sha256']:
        raise ValueError('Target delay evidence changed')
    values = parse_delays(delays, context)
    for n, v in values.items():
        if context['baseline_timing'][n]['receiver_delays_ns'] != v:
            raise ValueError('Published context differs from native delay')
    scopes = inventories(context, scoring_version)
    expected = scopes['all']
    values = receiver_subset(values, expected)
    metric = target_delay(values, expected)
    run = layout_path(args.root, 'reference_root') / case / 'run'
    reference = args.experiment / 'stage-reference-v1' / case
    replay = read_json(reference / 'replay/input-replay.json')
    if replay['seed'] != task.data['seed'] or context['seed'] != task.data['seed']:
        raise ValueError('Frozen input seed differs from task')
    prefix = verify_prefix(run)
    if replay['prefix'] != prefix or any(sha(source / p) != h for p, h in replay['source_sha256'].items()):
        raise ValueError('Current reference source/prefix mismatch')
    old_sta_root = args.experiment / 'local-context-v2' / case / 'reference-measurement'
    baseline = measurement(old_sta_root / 'native/measurement')
    old_dispatch = read_json(old_sta_root / 'dispatch.json')
    for die in DIES:
        if old_dispatch['input_odb_sha256'][die] != sha(reference / ('normalized-' + die) / 'native/design.odb'):
            raise ValueError('Reference ODB identity mismatch')
    output.mkdir(parents=True, exist_ok=False)
    smoke = output / 'multi-link-fixture'
    smoke.mkdir()
    plan = fixture(context, reference=reference, platform=platform, image=image, output=smoke / 'eco')
    readback = snapshots(smoke, output_odbs=smoke / 'eco', reference=reference, run=run, platform=platform,
        image=image, context=context, plan=plan)
    # Exercise the actual extended RC stitcher and graph delay query; no model
    # code or optimization outcome is involved in this diagnostic fixture.
    smoke_sta, smoke_delays = sta(smoke, output_odbs=smoke / 'eco', root=args.root, source=source,
        run=run, platform=platform, image=image, context=context, readback=readback)
    clock_skew(smoke_sta['values'], baseline['expected'])
    target_delay(receiver_subset(smoke_delays, expected), expected)
    if readback['resources']['added_hbts'] != 1:
        raise ValueError('Fixture did not exercise one additional physical HBT')
    tolerances = {k: 1e-6 for k in ('setup_wns_ns', 'setup_tns_ns', 'hold_wns_ns', 'hold_tns_ns')}
    facts = {'context_sha256': sha(context_root / 'multi-context.json'), 'fixture_readback_sha256': sha(smoke / 'branch-readback.json'),
        'fixture_sta_sha256': sha(smoke / 'independent-sta/native/measurement/result.json'),
        'baseline_sta_sha256': sha(old_sta_root / 'native/measurement/result.json'),
        'fixture_is_not_candidate_or_optimization_search': True, 'limits': context['limits'],
        'guards': 'setup/hold nonregression at 1e-6 ns; no new violating hold endpoints; all electrical excess/count nonregression',
        'scoring_version': scoring_version}
    reference_payload = {'legal': True, 'case': case, 'endpoint': task.data['endpoint'], 'primary_metric': metric,
        'measurements': values, 'expected_inventory': expected, 'timing': baseline['timing'],
        'timing_tolerances': tolerances}
    budget = None
    if scoring_version == 2:
        gate = load_profile(task)[0]['gates']['added_buffer_area']
        budget = buffer_area_budget(context, gate['max_ratio_to_min_buffer_area_per_target'])
        facts['added_buffer_area_basis'] = gate['basis']
        reference_payload.update(expected_inventory_by_scope=scopes, added_buffer_area_budget_um2=budget)
    write_new(output / 'evidence.json', facts)
    write_new(output / 'reference.json', reference_payload)
    query_path = smoke / 'target-delay/native/target-delays.tsv'
    smoke_area = added_buffer_area(context, smoke, readback)[0] if budget is not None else None
    reference_electrical = old_sta_root / 'native/measurement/electrical.rpt'
    capabilities = capability_checks(Recorder(output), task, context=context,
        context_path=context_root / 'multi-context.json',
        expected_context_sha256=read_json(context_root / 'result.json')['context_sha256'], delays_path=delays,
        execution=smoke, query_path=query_path, values=smoke_delays,
        inventory=expected, actual=smoke_sta, baseline=baseline,
        guard=timing_guard(baseline['timing'], baseline['timing'], tolerances), tolerances=tolerances,
        electrical_report=reference_electrical, electrical_reference=reference_electrical,
        area=smoke_area, area_budget=budget)
    q = {'kind': 'engineering-native-qualification-v1', 'task': task.task_id, 'task_digest': task.digest(),
        'case': case, 'seed': task.data['seed'], 'input_stage': task.data['input_stage'], 'endpoint': task.data['endpoint'],
        'physical_success': True, 'source_sha256': digest(replay['source_sha256']),
        'runtime_sha256': digest({'image': image, 'binary': replay['native_binary_sha256']}),
        'input_sha256': digest({'odb': old_dispatch['input_odb_sha256'], 'context': sha(context_root / 'multi-context.json')}),
        'capabilities': capabilities, 'reference': evidence(output / 'reference.json', output)}
    q.update(current_identity())
    q.update(qualification_fields(task, scoring_version))
    write_new(output / 'qualification.json', q)
    return q, context, baseline, reference_payload


def synthesize(subject, context_file, reference, image, output):
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / 'scratch'
    scratch.mkdir()
    name = 'o3db-engineering-multi-final-' + uuid.uuid4().hex[:12]
    cmd = ['docker', 'run', '--rm', '--name', name, '--read-only', '--network', 'none', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--cpus', '8', '--memory', '32g', '--pids-limit', '2048',
        '--tmpfs', '/tmp:rw,size=2g', '--user', str(os.getuid()) + ':' + str(os.getgid())]
    for host, target, ro in [(subject, '/subject', True), (context_file, '/context.json', True),
        (scratch, '/scratch', False),
        (output, '/out', False), *[(reference / ('normalized-' + d) / 'native/design.odb', '/candidate/' + d + '.odb', True) for d in DIES]]:
        cmd += bind(host, target, ro)
    code = 'import json; from engineering.multi_hbt import synthesize; synthesize(json.load(open("/context.json")), "/out/eco")'
    cmd += ['--env', 'PYTHONPATH=/subject', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'HOME=/tmp',
        '--env', 'OPEN3DBENCH_THREADS=8', '--entrypoint', 'python3', image, '-c', code]
    write_new(output / 'dispatch.json', {'command': cmd, 'context_sha256': sha(context_file), 'started_at': now()})
    with (output / 'native.log').open('x') as log:
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '--time', '5', name], capture_output=True, timeout=20)
            raise
    if code:
        raise ValueError('Submitted synthesis/materialization failed')
    return read_json(output / 'eco/branch_plan.json')


def final_case(task, case, *, args, ledger, sealed, q, qpath, context, baseline, reference_payload,
               source, platform, image, scoring_version):
    """One candidate dispatch: synthesize, read back, measure, gate, score, receipt."""
    folder = args.output / case
    folder.mkdir()
    run = layout_path(args.root, 'reference_root') / case / 'run'
    reference = args.experiment / 'stage-reference-v1' / case
    context_file = args.context_root / case / 'multi-context.json'
    plan = synthesize(args.subject, context_file, reference, image, folder / 'submission')
    odbs = folder / 'submission/eco'
    proof = snapshots(folder, output_odbs=odbs, reference=reference, run=run, platform=platform, image=image, context=context, plan=plan)
    actual, values = sta(folder, output_odbs=odbs, root=args.root, source=source, run=run, platform=platform, image=image, context=context, readback=proof)
    clock_skew(actual['values'], baseline['expected'])
    guard = timing_guard(baseline['timing'], actual['timing'], reference_payload['timing_tolerances'])
    reference_electrical = args.experiment / 'local-context-v2' / case / 'reference-measurement/native/measurement/electrical.rpt'
    electric = check_electrical(folder / 'independent-sta/native/measurement/electrical.rpt', reference_electrical)
    scopes = inventories(context, scoring_version)
    values = receiver_subset(values, scopes['all'])
    area, budget, buffers = None, None, []
    if scoring_version == 2:
        area, buffers = added_buffer_area(context, folder, proof)
        budget = reference_payload['added_buffer_area_budget_um2']
    write_new(folder / 'final-checks.json', {'physical': proof, 'setup_hold': guard, 'electrical': electric,
        'clock_inventory': 'matches qualified original', 'placement_rc': 'independent continuous passive multi-HBT assembly',
        'scoring_version': scoring_version, 'added_buffers': buffers, 'added_buffer_area_um2': area,
        'added_buffer_area_budget_um2': budget})
    if not guard['passed']:
        raise ValueError('Setup/hold nonregression failed: ' + ','.join(guard['failures']))
    verify_seal(args.subject, sealed)
    query_path = folder / 'target-delay/native/target-delays.tsv'
    recorder = Recorder(folder)
    capabilities = capability_checks(recorder, task, context=context, context_path=context_file,
        expected_context_sha256=read_json(qpath / 'evidence.json')['context_sha256'],
        delays_path=args.context_root / case / 'delays/native/target-delays.tsv', execution=folder,
        query_path=query_path, values=values,
        inventory=scopes['all'], actual=actual, baseline=baseline, guard=guard,
        tolerances=reference_payload['timing_tolerances'],
        electrical_report=folder / 'independent-sta/native/measurement/electrical.rpt',
        electrical_reference=reference_electrical, area=area, area_budget=budget)
    if recorder.failures():
        raise ValueError('Failed physical capability: ' + ', '.join(recorder.failures()))
    measurements = {'values': values, 'timing': actual['timing']}
    if scoring_version == 2:
        measurements['added_buffer_area_um2'] = area
    write_new(folder / 'final-measurements.json', measurements)
    raw = {'kind': 'engineering-native-result-v1', 'task': task.task_id, 'task_digest': task.digest(),
        'case': case, 'seed': task.data['seed'], 'endpoint': task.data['endpoint'], 'input_sha256': q['input_sha256'],
        'evaluator_sha256': q['evaluator_sha256'], 'runtime_sha256': q['runtime_sha256'],
        'submission_sha256': sealed['submission_sha256'], 'qualification_sha256': sha(qpath / 'qualification.json'),
        'checks': capabilities, 'measurements': evidence(folder / 'final-measurements.json', folder)}
    if scoring_version == 2:
        raw['scoring_version'] = 2
    write_new(folder / 'final-native.json', raw)
    value = validate_native_result(task, case, qpath / 'qualification.json', folder / 'final-native.json',
        submission_sha256=sealed['submission_sha256'], runtime_sha256=q['runtime_sha256'])
    return ledger.finish(case, succeeded=True, legal=True, candidate_metric=value['candidate_metric'],
        artifacts=[str(folder / 'final-native.json')], details=value.get('details'))


def main(argv=None):
    p = argparse.ArgumentParser()
    for k in ('root', 'experiment', 'output', 'subject', 'baseline_inventory', 'context_root'):
        p.add_argument('--' + k.replace('_', '-'), type=Path, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--qualification-root', type=Path)
    p.add_argument('--scoring-version', type=int, choices=(1, 2), default=CURRENT_SCORING_VERSION,
                   help='1 (default): frozen remote-receiver metric; 2: explicitly opt in to all receivers and area bound')
    args = p.parse_args(argv)
    task = load_task('multi_hbt_fanout')
    contract = validate_development_binding(task, args.subject, args.scoring_version)
    if not set(args.cases) <= set(task.data['cases']) or len(set(args.cases)) != len(args.cases):
        p.error('Invalid cases')
    args.output.mkdir(parents=True, exist_ok=False)
    source, image = verify_release(layout_path(args.root, 'release'))
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
            qpath = (args.qualification_root / case) if args.qualification_root else (args.output / 'qualifications' / case)
            try:
                verify_seal(args.subject, sealed)
                if args.qualification_root:
                    q = verify_qualification(task, case, qpath / 'qualification.json')['record']
                    if record_scoring_version(q) != args.scoring_version:
                        raise ValueError('Reused qualification has another scoring version')
                    context_file = args.context_root / case / 'multi-context.json'
                    context = read_json(context_file)
                    facts = read_json(qpath / 'evidence.json')
                    if facts['context_sha256'] != sha(context_file):
                        raise ValueError('Qualified target context changed')
                    verify_evaluator(q)
                    from engineering_bench.identity import qualification_binding
                    qualification_binding(q, source=source, image=image,
                        run=layout_path(args.root, 'reference_root') / case / 'run',
                        reference=args.experiment / 'stage-reference-v1' / case,
                        context_sha256=sha(context_file))
                    baseline = measurement(args.experiment / 'local-context-v2' / case / 'reference-measurement/native/measurement')
                    reference_payload = read_json(qpath / 'reference.json')
                    if baseline['timing'] != reference_payload['timing']:
                        raise ValueError('Reference timing changed')
                else:
                    q, context, baseline, reference_payload = qualify(task, case, args, source, image, platform, qpath,
                                                                      args.scoring_version)
            except Exception as exc:
                write_new(args.output / ('qualification-failed-' + case + '.json'), {'error': repr(exc), 'time': now(), 'candidate_finals': 0})
                print(json.dumps({'case': case, 'qualification_failed': repr(exc)}), flush=True)
                continue
            ledger.claim(case, qpath / 'qualification.json')
            print(json.dumps({'case': case, 'status': 'candidate_final_dispatched', 'time': now()}), flush=True)
            try:
                result = final_case(task, case, args=args, ledger=ledger, sealed=sealed, q=q, qpath=qpath,
                    context=context, baseline=baseline, reference_payload=reference_payload, source=source,
                    platform=platform, image=image, scoring_version=args.scoring_version)
            except Exception as exc:
                result = ledger.finish(case, succeeded=False, legal=False, error=repr(exc))
            print(json.dumps(result), flush=True)
    write_new(args.output / 'result.json', aggregate(task, ledger.records()))


if __name__ == '__main__':
    main()
