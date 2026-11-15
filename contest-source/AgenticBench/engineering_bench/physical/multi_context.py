"""Frozen native target selection and legal spare sites for the T3 prototype.

Select at most 16 ordinary unary-buffer-driven crossing nets by remote fanout,
then spread and stable identity; no candidate outcomes influence this set.
"""
import csv
import math
from pathlib import Path
import statistics

from local_context import geometry, library, free_sites, interpolate
from engineering_bench.clock_equivalence import read_tsv
from engineering_bench.legalization import load_snapshot
from engineering_bench.protocol import read_json, sha


def native_state(reference, features):
    tables = {}
    for die in ('bottom', 'top'):
        folder = Path(reference) / ('readback-' + die)
        _, _, cells = load_snapshot(folder)
        pins = read_tsv(folder / 'pins.tsv')
        by_net, by_cell = {}, {}
        for pin in pins:
            if pin['signal_type'] not in ('POWER', 'GROUND'):
                by_net.setdefault(pin['net'], []).append(pin)
                by_cell.setdefault(pin['instance'], []).append(pin)
        units, geo = geometry(Path(features) / (die + '-geometry.tsv'))
        tables[die] = {'cells': cells, 'pins': by_cell, 'nets': by_net, 'geo': geo, 'units': units,
            'ports': {r['port']: r for r in read_tsv(folder / 'ports.tsv')}}
    if tables['bottom']['units'] != tables['top']['units']:
        raise ValueError('Mismatched die geometry units')
    return tables


def create_context(run, reference, feature_root, platform, excluded_pins):
    run, reference, feature_root = Path(run), Path(reference), Path(feature_root)
    tables = native_state(reference, feature_root / 'features')
    pin_features = {r['pin']: r for r in read_tsv(feature_root / 'features/pins.tsv')}
    with (run / '60_opt/hbt.csv').open() as stream:
        hbts = list(csv.DictReader(stream))
    problem = read_json(run / '50_hbt/problem.json')
    grid = problem['site_grid']
    occupied = {(int(r['x_dbu']), int(r['y_dbu'])) for r in hbts}
    lib = library(platform)
    options = []
    excluded = {r['pin'] for r in read_tsv(excluded_pins)}
    for hbt in hbts:
        if hbt['is_clock'] != '0':
            continue
        source, sink = tables[hbt['source_die']], tables[hbt['sink_die']]
        source_net = source['ports'][hbt['source_port']]['net']
        sink_net = sink['ports'][hbt['sink_port']]['net']
        source_pins, sink_pins = source['nets'].get(source_net, []), sink['nets'].get(sink_net, [])
        drivers = [p for p in source_pins if p['direction'] == 'OUTPUT']
        loads = [p for p in sink_pins if p['direction'] == 'INPUT']
        if len(drivers) != 1 or not 2 <= len(loads) <= 64 or any(p['direction'] == 'INOUT' for p in source_pins + sink_pins):
            continue
        driver = drivers[0]
        cell = source['cells'][driver['instance']]
        if not cell['buffer'] or cell['clock_cell'] or cell['master'] not in lib:
            continue
        if any(sink['cells'][p['instance']]['clock_cell'] for p in loads):
            continue
        inputs = [p for p in source['pins'][driver['instance']] if p['direction'] == 'INPUT']
        if len(inputs) != 1:
            continue
        names = ['u_' + hbt['source_die'] + '/' + p['instance'] + '/' + p['pin'] for p in [driver, *inputs]]
        names += ['u_' + hbt['sink_die'] + '/' + p['instance'] + '/' + p['pin'] for p in loads]
        if set(names) & excluded:
            continue
        # Exclude external I/O aliases and non-single boundary electrical nets.
        if sum(r['net'] == source_net for r in source['ports'].values()) != 1 or sum(r['net'] == sink_net for r in sink['ports'].values()) != 1:
            continue
        points = [sink['geo']['pin'][p['instance'] + '/' + p['pin']] for p in loads]
        span = max(p['x'] for p in points)-min(p['x'] for p in points) + max(p['y'] for p in points)-min(p['y'] for p in points)
        # Same-net receivers on the source die (T3 v2 scores them too); never
        # used for target selection, so the selected set is unchanged.
        local = [p for p in source_pins if p['direction'] == 'INPUT']
        options.append((-len(loads), -span, hbt['hbt_id'], hbt, driver, inputs[0], loads, local))
    if not options:
        raise ValueError('No supported multi-receiver ordinary crossing targets')
    targets, drivers, inputs, receivers, sites, native_drivers, native_receivers, ports = {}, {}, {}, {}, {}, {}, {}, {}
    selected = sorted(options)[:16]
    for _, _, _, hbt, driver, driver_input, loads, local in selected:
        net, sd, td = hbt['net_name'], hbt['source_die'], hbt['sink_die']
        driver_name = 'u_' + sd + '/' + driver['instance'] + '/' + driver['pin']
        point = tables[sd]['geo']['pin'][driver['instance'] + '/' + driver['pin']]
        drivers[driver_name] = {'die': sd, 'x_dbu': point['x'], 'y_dbu': point['y']}
        inputs[driver_name] = ['u_' + sd + '/' + driver_input['instance'] + '/' + driver_input['pin']]
        native_drivers[driver_name] = {'die': sd, 'iterm': driver['instance'] + '/' + driver['pin']}
        sink_ids = []
        for pin in loads:
            name = 'u_' + td + '/' + pin['instance'] + '/' + pin['pin']
            point = tables[td]['geo']['pin'][pin['instance'] + '/' + pin['pin']]
            cap = float(pin_features[name]['capacitance_pf'])
            if not math.isfinite(cap) or cap < 0:
                raise ValueError('Missing native receiver capacitance')
            receivers[name] = {'die': td, 'x_dbu': point['x'], 'y_dbu': point['y'], 'capacitance_pf': cap}
            native_receivers[name] = {'die': td, 'iterm': pin['instance'] + '/' + pin['pin']}
            sink_ids.append(name)
        site_id = 'original:' + hbt['hbt_id']
        sites[site_id] = {'x_dbu': int(hbt['x_dbu']), 'y_dbu': int(hbt['y_dbu'])}
        targets[net] = {'original_driver': driver_name, 'original_remote_receivers': sorted(sink_ids),
            'original_local_receivers': sorted('u_' + sd + '/' + p['instance'] + '/' + p['pin'] for p in local),
            'original_hbt_id': hbt['hbt_id'], 'original_site_id': site_id,
            'source_die': sd, 'sink_die': td, 'signal_type': 'data', 'driver_count': 1,
            'is_clock': False, 'supported': True, 'timing_exceptions': []}
        ports[hbt['hbt_id']] = {k: hbt[k] for k in ('source_port', 'sink_port')}
    # Candidate HBT sites use the frozen lattice. Conservatively reserve a
    # three-pitch neighborhood around existing/new anchors (routing escape).
    px, py, ox, oy = (grid[k] for k in ('pitch_x', 'pitch_y', 'origin_x', 'origin_y'))
    bounds = grid['die']
    spare_ids = []
    buffer_boxes = {}
    for die in ('bottom', 'top'):
        sizes = [r for r in tables[die]['geo']['cell'].values() if r['extra'] in lib]
        if not sizes:
            raise ValueError('No native buffer geometry')
        buffer_boxes[die] = (max(r['w'] for r in sizes), max(r['h'] for r in sizes))
    def permitted(x, y):
        if not (bounds['x_min'] <= x <= bounds['x_max']-grid['hbt_width'] and bounds['y_min'] <= y <= bounds['y_max']-grid['hbt_height']):
            return False
        if any(abs(x-a) < 3*px and abs(y-b) < 3*py for a, b in occupied):
            return False
        for rect in problem['blockages']:
            if x < rect['x_max'] and x+grid['hbt_width'] > rect['x_min'] and y < rect['y_max'] and y+grid['hbt_height'] > rect['y_min']:
                return False
        return True
    anchors = sorted(receivers.values(), key=lambda r: (r['x_dbu'], r['y_dbu']))
    for point in anchors:
        if len(spare_ids) >= 64:
            break
        gx, gy = round((point['x_dbu']-ox)/px), round((point['y_dbu']-oy)/py)
        found = False
        for radius in range(10):
            if found:
                break
            for dx, dy in sorted((a,b) for a in range(-radius,radius+1) for b in range(-radius,radius+1) if max(abs(a),abs(b)) == radius):
                x, y = ox+(gx+dx)*px, oy+(gy+dy)*py
                if not permitted(x,y):
                    continue
                # Confirm nearby empty row space on both dies. The submitted
                # adapter initializes at the HBT and uses the fixed legalizer.
                positions = {}
                for die in ('bottom', 'top'):
                    w,h = buffer_boxes[die]
                    choices = free_sites(tables[die]['geo'], x,y,w,h,1)
                    if choices:
                        positions[die] = choices[0]
                if len(positions) != 2:
                    continue
                key = 'spare:' + str(x) + ':' + str(y)
                sites[key] = {'x_dbu': x, 'y_dbu': y, 'buffer_by_die': positions}
                # These optional native row locations are diagnostic. This
                # submission's adapter uses the HBT-origin initial location.
                spare_ids.append(key)
                occupied.add((x,y))
                found = True
                break
    if not spare_ids:
        raise ValueError('No reserved legal spare HBT sites')
    cells = []
    resistances = []
    slew = float(read_json(run / '60_opt/timing_model.json')['default_transition_ns'])
    for name, row in sorted(lib.items()):
        tab = row['tables']['cell_rise']
        low, high = tab['load_pf'][0], tab['load_pf'][-1]
        resistance = (interpolate(tab,slew,high)-interpolate(tab,slew,low))/((high-low)*.001*.69)
        resistances.append(resistance)
        cells.append({k: row[k] for k in ('name','die','input_pin','output_pin','input_capacitance_pf','max_capacitance_pf','area_um2','noninverting')})
        cells[-1].update(intrinsic_delay_ns=interpolate(tab,slew,low), drive_resistance_ohm=resistance, max_fanout=64)
    units = tables['bottom']['units']
    spans = [(-r[1])/units for r in selected]
    characteristic_um = max(statistics.median(spans), 1.)
    rc = read_json(run / '60_opt/timing_model.json')
    native = {'bottom_odb': '/candidate/bottom.odb', 'top_odb': '/candidate/top.odb',
        'openroad_executable': '/usr/local/bin/openroad',
        'hbt_width_dbu': grid['hbt_width'], 'hbt_height_dbu': grid['hbt_height'],
        'power_net': 'VDD', 'ground_net': 'VSS', 'hbt_ports': ports,
        'driver_pins': native_drivers, 'receiver_pins': native_receivers, 'max_buffer_displacement_sites': [500,500]}
    for die in ('bottom', 'top'):
        boundary_ports = {h[side + '_port'] for h in hbts for side in ('source', 'sink') if h[side + '_die'] == die}
        layers = {tables[die]['ports'][n]['layer'] for n in boundary_ports}
        if len(layers) != 1:
            raise ValueError('Ambiguous original HBT port layer')
        native[die + '_hbt_layer'] = next(iter(layers))
        # The flow has BOTIN/TOPIN named COVER variants. They are pinless
        # geometric anchors, not direction-bearing logic cells. Direction is
        # represented by the independently checked BTerms/passive link. A
        # stable representative is valid only after native equal-footprint and
        # pinless proofs; never choose from unrelated COVER cells.
        hbt_cells = [tables[die]['cells'][h['hbt_id']] for h in hbts]
        shapes = { (tables[die]['geo']['cover'][h['hbt_id']]['w'], tables[die]['geo']['cover'][h['hbt_id']]['h']) for h in hbts }
        if shapes != {(grid['hbt_width'], grid['hbt_height'])} or any(tables[die]['pins'].get(h['hbt_id']) for h in hbts) or any(c['master_type'] != 'COVER' for c in hbt_cells):
            raise ValueError('HBT anchor variants are not equivalent pinless footprints')
        native[die + '_hbt_master'] = sorted({r['master'] for r in hbt_cells})[0]
    return {'schema': 'multi-hbt-context-v1', 'seed': 1001,
        'units': {'coordinate':'dbu','capacitance':'pf','time':'ns','resistance':'ohm'}, 'dbu_per_micron': units,
        'target_nets': targets, 'driver_geometry': drivers, 'driver_input_pins': inputs,
        'receiver_geometry': receivers, 'site_geometry': sites, 'spare_site_ids': spare_ids,
        'limits': {'max_added_hbts': min(32,2*len(targets))}, 'buffer_cells': cells,
        'reference_input_slew_ns': slew, 'new_hbt_id_base': 1+max(int(r['hbt_id'].rsplit('_',1)[1]) for r in hbts),
        'timing_model': {'wire_delay_ns_per_dbu': .5*3.5714*.0000888758*characteristic_um*.001/units,
            'hbt_delay_ns': rc['resistance_ohm']*rc['capacitance_pf']*.001,
            'driver_resistance_ohm': statistics.median(resistances), 'min_predicted_improvement_ns': 1e-6},
        'timing_model_provenance': 'Linear predictor fitted to provided ng45 Liberty tables and metal3 RC; not measured candidate delay',
        'native': native, 'target_selection': 'up to 16 unary-buffer-driven ordinary crossings; fanout descending, spread descending, stable HBT identity',
        'exception_exclusion': {'native_pins_sha256': sha(excluded_pins), 'excluded_pin_count': len(excluded),
            'policy': 'exclude native constant-propagation pins, all latch data fanin, reset false-path fanout and asynchronous JTAG data cones'},
        'baseline_timing': {}, 'baseline_delay_pending_native_measurement': True}
