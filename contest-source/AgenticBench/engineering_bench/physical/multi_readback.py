"""Independently reconstruct isolated data branches from native ODB snapshots."""
import csv
from pathlib import Path

from engineering_bench.clock_equivalence import compare_local, read_tsv
from engineering_bench.interfaces import validate_branch_map
from engineering_bench.legalization import load_snapshot, ports_geometry
from engineering_bench.protocol import read_json, write_new, sha


def validate(context, plan, original, candidate, hbt_csv, output):
    tables = {}
    for die in ('bottom', 'top'):
        _, _, before = load_snapshot(original[die])
        _, _, after = load_snapshot(candidate[die])
        pins = read_tsv(candidate[die] / 'pins.tsv')
        ports = {r['port']: r for r in read_tsv(candidate[die] / 'ports.tsv')}
        old_ports = {r['port']: r for r in read_tsv(original[die] / 'ports.tsv')}
        old_geo, geo = ports_geometry(original[die] / 'ports.tsv'), ports_geometry(candidate[die] / 'ports.tsv')
        if not set(old_geo) <= set(geo):
            raise ValueError('Original port geometry changed')
        # Clock cells/pins remain exactly unchanged, not merely equivalent.
        old_clock = [p for p in read_tsv(original[die] / 'pins.tsv') if before[p['instance']]['clock_cell']]
        new_clock = [p for p in pins if p['instance'] in before and before[p['instance']]['clock_cell']]
        if old_clock != new_clock:
            raise ValueError('Original clock-tree connectivity changed')
        by_pin = {p['instance'] + '/' + p['pin']: p for p in pins if p['signal_type'] not in ('POWER', 'GROUND')}
        by_net = {}
        for pin in by_pin.values():
            by_net.setdefault(pin['net'], []).append(pin)
        tables[die] = {'before': before, 'after': after, 'pins': by_pin, 'nets': by_net, 'ports': ports, 'old_ports': old_ports}
    with Path(hbt_csv).open() as stream:
        links = list(csv.DictReader(stream))
    original_links = {r['hbt_id']: r for r in links}
    rows, additions, anchors, new_links, cells_by_target, allowed_ports = {}, [], [], [], {}, {'bottom': set(), 'top': set()}
    seen_targets = set()
    for item in plan['target_nets']:
        name = item['logical_net']
        if name not in context['target_nets'] or name in seen_targets:
            raise ValueError('Unexpected/repeated planned target')
        seen_targets.add(name)
        target = context['target_nets'][name]
        sd, td = target['source_die'], target['sink_die']
        source, sink = tables[sd], tables[td]
        native_driver = context['native']['driver_pins'][target['original_driver']]['iterm']
        trunk = source['pins'][native_driver]['net']
        cells_by_target[name] = []
        for branch in item['branches']:
            hbt, branch_id = branch['hbt_id'], branch['branch_id']
            if branch_id in rows or branch['source_die'] != sd or branch['sink_die'] != td:
                raise ValueError('Repeated/wrong-die branch')
            site_id = branch['site_id']
            site = context['site_geometry'][site_id]
            existing = hbt == target['original_hbt_id']
            if existing:
                old = original_links[hbt]
                sp, tp = old['source_port'], old['sink_port']
                if branch.get('buffer_instance') is not None:
                    raise ValueError('Original HBT branch cannot replace its original driver')
            else:
                if hbt in original_links or site_id not in context['spare_site_ids']:
                    raise ValueError('New HBT identity/site is not allowed')
                sp, tp = hbt + '_SRC', hbt + '_DST'
                for die, port in ((sd, sp), (td, tp)):
                    cell = tables[die]['after'][hbt]
                    if cell['master_type'] != 'COVER' or (cell['x_dbu'], cell['y_dbu']) != (site['x_dbu'], site['y_dbu']):
                        raise ValueError('HBT anchor geometry differs from reserved site')
                    box = tables[die]['ports'][port]
                    native = context['native']
                    if box['layer'] != native[die + '_hbt_layer'] or abs(int(box['xmin'])+int(box['xmax'])-2*site['x_dbu']-native['hbt_width_dbu']) > 1 or abs(int(box['ymin'])+int(box['ymax'])-2*site['y_dbu']-native['hbt_height_dbu']) > 1:
                        raise ValueError('HBT port is not centered on its native-layer anchor')
                    anchors.append((die, hbt))
                    allowed_ports[die].add(port)
                new_links.append({'hbt_id': hbt, 'net_name': branch_id, 'physical_net': branch_id,
                    'source_die': sd, 'sink_die': td, 'source_port': sp, 'sink_port': tp,
                    'direction': 'B2T' if sd == 'bottom' else 'T2B', 'is_clock': False,
                    'x_dbu': site['x_dbu'], 'y_dbu': site['y_dbu']})
            sport, tport = source['ports'][sp], sink['ports'][tp]
            if sport['direction'] != 'OUTPUT' or tport['direction'] != 'INPUT':
                raise ValueError('Incorrect HBT direction')
            snet, tnet = sport['net'], tport['net']
            outputs = [p for p in source['nets'][snet] if p['direction'] == 'OUTPUT']
            if len(outputs) != 1:
                raise ValueError('Source branch is not singly driven')
            if existing:
                if snet != trunk or outputs[0]['instance'] + '/' + outputs[0]['pin'] != native_driver:
                    raise ValueError('Original HBT driver changed')
            else:
                inst = outputs[0]['instance']
                cell = source['after'][inst]
                if inst in source['before'] or not cell['buffer'] or cell['inverter'] or cell['clock_cell']:
                    raise ValueError('Isolation is not a newly added noninverting data buffer')
                if cell['master'] not in {c['name'] for c in context['buffer_cells'] if c['die'] == sd}:
                    raise ValueError('Isolation buffer is outside provided library choices')
                inputs = [p for p in source['pins'].values() if p['instance'] == inst and p['direction'] == 'INPUT']
                if len(inputs) != 1 or inputs[0]['net'] != trunk:
                    raise ValueError('Isolation buffer carries wrong original driver')
                additions.append((sd, inst))
                cells_by_target[name].append('u_' + sd + '/' + inst)
            pins = sink['nets'][tnet]
            if any(p['direction'] != 'INPUT' for p in pins):
                raise ValueError('Remote branch has extra local driver')
            receivers = sorted('u_' + td + '/' + p['instance'] + '/' + p['pin'] for p in pins)
            if any(sum(p['net'] == net for p in table['ports'].values()) != 1 for table, net in ((source, snet), (sink, tnet))):
                raise ValueError('Multiple boundaries share an electrical net')
            rows[branch_id] = {'logical_net': name, 'original_driver': target['original_driver'],
                'noninverting_buffer_proof': True, 'single_driver': True, 'hbt_id': hbt, 'site_id': site_id,
                'original_remote_receivers': receivers, 'source_electrical_net': sd + '/' + snet,
                'sink_electrical_net': td + '/' + tnet}
    for die, table in tables.items():
        if set(table['ports']) - set(table['old_ports']) != allowed_ports[die]:
            raise ValueError('Undeclared added IO/HBT ports')
    resources = validate_branch_map(context, plan['branch_map'], rows)
    if len(set(additions)) != resources['added_hbts'] or len(additions) != len(set(additions)):
        raise ValueError('Exactly one isolated buffer is required per added HBT')
    proof = compare_local(original, candidate, links, links + new_links, additions, anchors)
    receipt = {'branches': rows, 'resources': resources, 'physical_equivalence': proof,
        'additional_buffers': additions, 'additional_hbts': anchors, 'cells_by_target': cells_by_target,
        'readbacks': {d: sha(candidate[d] / 'result.json') for d in candidate}}
    write_new(output / 'branch-readback.json', receipt)
    write_new(output / 'additional-links.json', {'kind': 'engineering-native-additional-data-hbts-v1', 'links': new_links,
        'native_readback_sha256': sha(output / 'branch-readback.json')})
    return receipt
