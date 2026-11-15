"""Trusted physical CLI: independent post-CTS ODB -> joint placement-RC STA.

Invoked in a pinned, offline runtime with the physical flow source read-only.
This external adapter imports the physical backend; AgenticBench communicates
with it only via a subprocess and hash-bound artifacts. It does not run CTS,
modify source inputs, publish a stage checkpoint or qualify a candidate.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from dataclasses import replace, asdict


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


EXPORT = r'''
foreach lib [split $::env(ENG_LIBS) "\n"] { if {$lib ne ""} { read_liberty $lib } }
read_db $::env(ENG_ODB)
set block [ord::get_db_block]
check_placement -verbose
set pg 0
foreach inst [$block getInsts] {
  foreach pin [$inst getITerms] {
    set type [$pin getSigType]
    if {$type ni {POWER GROUND}} { continue }
    set net [$pin getNet]
    set expected [expr {$type eq "POWER" ? "VDD" : "VSS"}]
    if {$net eq "NULL" || [$net getName] ne $expected} { error "incorrect PG at [$inst getName]" }
    incr pg
  }
}
set ports [open $::env(ENG_PORTS) r]
set result [open $::env(ENG_NETS) w]
while {[gets $ports port] >= 0} {
  if {$port eq ""} { continue }
  set term [$block findBTerm $port]
  if {$term eq "NULL" || [$term getNet] eq "NULL"} { error "missing HBT port $port" }
  puts $result "$port\t[[$term getNet] getName]"
}
close $ports
close $result
write_verilog $::env(ENG_VERILOG)
puts "ENGINEERING_MEASUREMENT_EXPORT pg=$pg"
'''

MEASUREMENT_ENDPOINTS = {
    'skew_sta': 'post_cts_placement_rc_joint_sta',
    'skew': 'post_cts_placement_rc_joint_skew',
}

ENDPOINTS = r'''
set result [open $::env(ENG_ENDPOINTS) w]
puts $result "pin\tsetup_slack_ns\thold_slack_ns"
foreach pin [sta::endpoints] {
  puts $result [join [list [get_full_name $pin] \
    [get_property $pin slack_max] [get_property $pin slack_min]] "\t"]
}
close $result
set counts [open $::env(ENG_COUNTS) w]
puts $counts "hold_violating_endpoints\t[sta::endpoint_violation_count min]"
puts $counts "setup_violating_endpoints\t[sta::endpoint_violation_count max]"
close $counts
'''

ELECTRICAL = r'''
report_check_types -max_slew -max_capacitance -max_fanout -violators -digits 9 > $::env(ENG_ELECTRICAL)
puts "ENGINEERING_MEASUREMENT_GUARDS"
'''

EXTRA = ENDPOINTS + ELECTRICAL


def measurement_supplement(mode):
    if mode not in MEASUREMENT_ENDPOINTS:
        raise ValueError('Unknown current-stage measurement mode: ' + str(mode))
    return EXTRA if mode == 'skew_sta' else ELECTRICAL


def clock_summary(path):
    """Reduce independent native arrivals; never use the candidate's skew scalar.

    This diagnostic does not by itself prove frozen original-sink coverage.
    The final framework reducer still checks that exact reference inventory.
    """
    groups = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            if (not row['clock'] or not row['pin'] or row['die'] not in ('bottom', 'top')
                    or row['active_edge'] not in ('rise', 'fall') or row['clock_domain_count'] != '1'):
                raise ValueError('Incomplete or ambiguous clock sink')
            for view in ('min', 'max'):
                group = '/'.join((row['clock'], row['active_edge'], view, 'default'))
                values = groups.setdefault(group, {})
                value = float(row['arrival_' + view + '_' + row['active_edge'] + '_ns'])
                if not math.isfinite(value) or row['pin'] in values:
                    raise ValueError('Missing or repeated native clock arrival')
                values[row['pin']] = value
    if not groups:
        raise ValueError('Empty native clock inventory')
    summaries = {k: dict(sink_count=len(v), skew_ns=max(v.values()) - min(v.values()))
                 for k, v in sorted(groups.items())}
    return dict(clock_skew_ns=max(v['skew_ns'] for v in summaries.values()), clock_groups=summaries)


def endpoint_inventory(path):
    import csv
    records, violating, absent_min, absent_max = {}, [], [], []
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            name = row['pin']
            if not name or name in records:
                raise ValueError('Empty/repeated STA endpoint')
            records[name] = row
            for field, absent in (('hold_slack_ns', absent_min), ('setup_slack_ns', absent_max)):
                value = float(row[field])
                if math.isnan(value) or value == -math.inf:
                    raise ValueError('Invalid endpoint slack: ' + name)
                # OpenSTA returns +INF when this endpoint has no applicable
                # path in the selected min/max analysis. Preserve that fact,
                # never invent a zero slack. Complete-graph checks still run.
                if value == math.inf or value >= 1e20:
                    absent.append(name)
                elif field == 'hold_slack_ns' and value < 0:
                    violating.append(name)
    if not records:
        raise ValueError('Empty native endpoint inventory')
    return dict(endpoint_count=len(records), hold_violating_endpoints=sorted(violating),
                endpoints_without_min_path=sorted(absent_min), endpoints_without_max_path=sorted(absent_max))


def measure(args):
    mode = args.measurement_mode
    supplement = measurement_supplement(mode)
    flow = Path(args.flow_source).resolve(strict=True)
    sys.path.insert(0, str(flow))
    from open3dbench.backend.opt_handoff import load_cts_handoff, HBTBoundary
    from open3dbench.backend import three_d_cts as cts
    from open3dbench.backend.three_d_opt import _run_openroad, _platform_files, _warning_summary
    from open3dbench.backend.hbt_materialize import (
        _read_boundary_net_map, _repair_boundary_verilog, _validate_boundary_verilog)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    handoff = load_cts_handoff(Path(args.checkpoint))
    extra_path = getattr(args, 'additional_hbt_links', None)
    if extra_path:
        extra_path = Path(extra_path).resolve(strict=True)
        extra = json.loads(extra_path.read_text())
        if extra.get('kind') != 'engineering-native-additional-data-hbts-v1':
            raise ValueError('Additional boundaries must come from independent native readback')
        seen = {r.hbt_id for r in handoff.boundaries}
        additions = []
        for row in extra['links']:
            if row['hbt_id'] in seen or row['is_clock'] is not False or {row['source_die'], row['sink_die']} != {'bottom', 'top'}:
                raise ValueError('Invalid additional data HBT identity')
            seen.add(row['hbt_id'])
            additions.append(HBTBoundary(**row))
        handoff = replace(handoff, boundaries=handoff.boundaries + tuple(additions))
    platform = Path(args.platform).resolve(strict=True)
    odbs = {d: Path(getattr(args, d + '_odb')).resolve(strict=True) for d in ('bottom', 'top')}
    inputs = {str(p): sha(p) for p in [handoff.checkpoint, *odbs.values(), handoff.assembly_sdc,
        handoff.checkpoint.parent / 'hbt.csv', handoff.checkpoint.parent / 'timing_model.json']}
    if extra_path:
        inputs[str(extra_path)] = sha(extra_path)
    source = {str(p.relative_to(flow)): sha(p) for p in sorted((flow / 'open3dbench').rglob('*.py'))}
    dispatch = dict(kind='engineering-independent-cts-measurement-v1', input_hashes=inputs,
        source_files=source, executable_sha256=sha(args.openroad), adapter_sha256=sha(__file__),
        threads=8, endpoint=MEASUREMENT_ENDPOINTS[mode], measurement_mode=mode,
        rc_model='placement_estimated_with_passive_hbt',
        executed_operations=['export_and_check_current_odbs', 'joint_placement_rc_sta',
                             'same_view_electrical_and_optional_endpoint_reports'],
        upstream_reexecuted=False, downstream_flow_executed=False,
        task_qualification=False, candidate_final=False)
    (output / 'dispatch.json').write_text(json.dumps(dispatch, indent=2) + '\n')
    os.environ['OPEN3DBENCH_THREADS'] = '8'
    try:
        verilogs = {}
        for die, odb in odbs.items():
            part = output / die
            part.mkdir()
            ports = part / 'boundary_ports.txt'
            ports.write_text(''.join(r.role_on(die)[1] + '\n' for r in handoff.boundaries))
            _, libs = _platform_files(platform, die)
            _run_openroad(EXPORT, dict(ENG_LIBS='\n'.join(map(str, libs)), ENG_ODB=str(odb),
                ENG_PORTS=str(ports), ENG_NETS=str(part / 'boundary_nets.tsv'),
                ENG_VERILOG=str(part / 'design.v')), part / 'export.log', args.openroad)
            rows = cts._boundary_rows(handoff)
            nets = _read_boundary_net_map(part / 'boundary_nets.tsv')
            if len(nets) != len(handoff.boundaries):
                raise ValueError('Incomplete physical boundary export')
            _repair_boundary_verilog(part / 'design.v', die, rows, nets)
            _validate_boundary_verilog(part / 'design.v', die, rows)
            verilogs[die] = part / 'design.v'
        assembly = output / 'assembly'
        metrics, skew = cts._assembly_cts_sta(handoff, odbs['bottom'], odbs['top'],
            verilogs['bottom'], verilogs['top'], platform, assembly, args.openroad)
        (assembly / 'engineering-effective-boundaries.json').write_text(json.dumps(
            {'boundaries': [asdict(r) for r in handoff.boundaries],
             'original_checkpoint_is_immutable': True,
             'legacy_boundary_json_describes_original_checkpoint_only': True}, indent=2) + '\n')
        passive = assembly / 'passive'
        # Replay the freshly generated full passive view, never candidate SDC or
        # candidate measurement code. The extra report adds all endpoints, not
        # only the top-N timing paths shown in human reports.
        script = (passive / 'validate_sta.tcl').read_text() + '\n' + supplement
        _run_openroad(script, dict(ENG_ENDPOINTS=str(output / 'endpoints.tsv'),
            ENG_COUNTS=str(output / 'endpoint_counts.tsv'),
            ENG_ELECTRICAL=str(output / 'electrical.rpt')),
            output / 'guards.log', args.openroad)
        endpoints = {}
        if mode == 'skew_sta':
            endpoints = endpoint_inventory(output / 'endpoints.tsv')
            counts = dict(line.split('\t') for line in (output / 'endpoint_counts.tsv').read_text().splitlines())
            if len(endpoints['hold_violating_endpoints']) != int(counts['hold_violating_endpoints']):
                raise ValueError('Hold endpoint coverage disagrees with native violation count')
        timing = dict(setup_wns_ns=metrics['wns_setup'], setup_tns_ns=metrics['tns_setup'],
            hold_wns_ns=metrics['wns_hold'], hold_tns_ns=metrics['tns_hold'],
            **endpoints)
        warning = _warning_summary(output)
        if warning['unexpected_count']:
            raise ValueError('Unaudited physical warnings')
        if any(sha(p) != value for p, value in inputs.items()):
            raise ValueError('Read-only physical input changed')
        clocks = clock_summary(passive / 'clock_sinks_by_clock.tsv')
        result = dict(status='succeeded', timing=timing, legacy_skew_diagnostic=skew, **clocks,
            timing_role='guard_input' if mode == 'skew_sta' else 'diagnostic_only',
            expected_metric_source='assembly/passive/clock_sinks_by_clock.tsv',
            endpoint_count=endpoints.get('endpoint_count'), warning_audit=warning,
            artifacts={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
                       if p.is_file() and p.name != 'result.json'})
    except Exception as exc:
        result = dict(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    result.update(duration_seconds=time.monotonic() - started, endpoint=MEASUREMENT_ENDPOINTS[mode],
                  measurement_mode=mode, task_qualification=False, candidate_final=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}, indent=2))
    return 0 if result['status'] == 'succeeded' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('flow_source', 'checkpoint', 'platform', 'bottom_odb', 'top_odb', 'output'):
        parser.add_argument('--' + field.replace('_', '-'), required=True)
    parser.add_argument('--openroad', default='/usr/local/bin/openroad')
    parser.add_argument('--measurement-mode', choices=tuple(MEASUREMENT_ENDPOINTS), default='skew_sta')
    parser.add_argument('--additional-hbt-links', help='Protected native-readback additional data boundaries only')
    return measure(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
