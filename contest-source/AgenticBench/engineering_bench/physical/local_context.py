"""Build current-state clock context; never precompute candidate outcomes."""
import bisect
import csv
import math
from pathlib import Path
import re

from engineering_bench.clock_equivalence import read_tsv
from engineering_bench.input_audit import clock_table
from engineering_bench.legalization import load_snapshot
from engineering_bench.protocol import digest, read_json, sha


def blocks(text, kind):
    pattern = re.compile(r'\b' + re.escape(kind) + r'\s*\(\s*([^)]*)\)\s*\{')
    for match in pattern.finditer(text):
        index, depth, quote = match.end(), 1, False
        while depth and index < len(text):
            char = text[index]
            if char == '"' and text[index - 1] != '\\':
                quote = not quote
            if not quote:
                depth += (char == '{') - (char == '}')
            index += 1
        if depth:
            raise ValueError('Unbalanced Liberty block')
        yield match[1].strip().strip('"'), text[match.end():index - 1]


def scalar(text, key):
    match = re.search(r'\b' + key + r'\s*:\s*([0-9.eE+-]+)\s*;', text)
    if not match:
        raise ValueError('Missing Liberty field ' + key)
    value = float(match[1])
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid Liberty field')
    return value


def table(text):
    arrays = []
    for key in ('index_1', 'index_2'):
        match = re.search(key + r'\s*\(\s*"([^"]+)"\s*\)', text)
        arrays.append([float(v) for v in match[1].split(',')])
    values = re.search(r'\bvalues\s*\((.*?)\)\s*;', text, re.S)[1]
    rows = [[float(v) for v in row.split(',')] for row in re.findall(r'"([^"]+)"', values)]
    if len(rows) != len(arrays[0]) or any(len(r) != len(arrays[1]) for r in rows):
        raise ValueError('Invalid NLDM table dimensions')
    return {'slew_ns': arrays[0], 'load_pf': [v * .001 for v in arrays[1]], 'values_ns': rows}


def interpolate(tab, slew, load):
    def location(axis, value):
        i = max(0, min(len(axis) - 2, bisect.bisect_right(axis, value) - 1))
        f = (value - axis[i]) / (axis[i + 1] - axis[i])
        return i, max(0., min(1., f))
    i, x = location(tab['slew_ns'], slew)
    j, y = location(tab['load_pf'], load)
    values = tab['values_ns']
    return (1-x) * ((1-y)*values[i][j] + y*values[i][j+1]) + x * ((1-y)*values[i+1][j] + y*values[i+1][j+1])


def library(platform):
    result = {}
    for die in ('bottom', 'top'):
        path = Path(platform) / ('lib_' + die) / ('NangateOpenCellLibrary_typical.' + die + '.lib')
        text = re.sub(r'/\*.*?\*/', '', path.read_text(), flags=re.S)
        if not re.search(r'time_unit\s*:\s*"1ns"', text) or not re.search(r'capacitive_load_unit\s*\(1,\s*ff\)', text, re.I):
            raise ValueError('Explicit ng45 ns/fF library required')
        for name, body in blocks(text, 'cell'):
            pins = dict(blocks(body, 'pin'))
            if not re.fullmatch(r'BUF_X[1248]_' + die, name):
                continue
            if not re.search(r'function\s*:\s*"A"', pins['Z']):
                raise ValueError('Buffer function is not noninverting')
            result[name] = {'name': name, 'die': die, 'input_pin': 'A', 'output_pin': 'Z',
                'input_capacitance_pf': scalar(pins['A'], 'capacitance') * .001,
                'max_capacitance_pf': scalar(pins['Z'], 'max_capacitance') * .001,
                'area_um2': scalar(body, 'area'), 'noninverting': True,
                'tables': {key: table(next(blocks(pins['Z'], key))[1]) for key in ('cell_rise', 'cell_fall', 'rise_transition', 'fall_transition')},
                'liberty_sha256': sha(path)}
    if len(result) != 8:
        raise ValueError('Missing expected ng45 buffer options')
    return result


def geometry(path):
    rows = read_tsv(path)
    units = next(int(r['extra']) for r in rows if r['kind'] == 'units')
    by_kind = {}
    for row in rows:
        if row['kind'] == 'units':
            continue
        for key in ('x', 'y', 'w', 'h'):
            row[key] = int(row[key])
        by_kind.setdefault(row['kind'], {})[row['name']] = row
    return units, by_kind


def free_sites(geo, x, y, width, height, count):
    """Nearest empty row slots; all original standard cells remain fixed."""
    candidates = []
    # Native COVER anchors occupy top metal, not standard-cell placement rows.
    obstacles = geo.setdefault('_obstacles', list(geo['cell'].values()) + list(geo.get('blockage', {}).values()))
    cache = geo.setdefault('_row_gaps', {})
    rows = sorted(geo['row'].values(), key=lambda r: (abs(r['y']-y), r['y'], r['x']))
    for row in rows[:64]:
        if height > row['h']:
            continue
        pitch = int(row['extra'])
        key = (row['name'], height)
        if key not in cache:
            nearby = [r for r in obstacles if r['y'] < row['y']+height and r['y']+r['h'] > row['y']]
            ranges = sorted((max(row['x'],r['x']), min(row['x']+row['w'],r['x']+r['w'])) for r in nearby if r['x'] < row['x']+row['w'] and r['x']+r['w'] > row['x'])
            cursor, gaps = row['x'], []
            for low, high in ranges:
                if low > cursor:
                    gaps.append((cursor, low))
                cursor = max(cursor, high)
            if cursor < row['x']+row['w']:
                gaps.append((cursor, row['x']+row['w']))
            cache[key] = gaps
        gaps = cache[key]
        for low, high in gaps:
            start = row['x'] + math.ceil((low-row['x'])/pitch)*pitch
            if start + width*count > high:
                continue
            start = max(start, min(row['x'] + round((x-row['x'])/pitch)*pitch, high-width*count))
            start = row['x'] + math.floor((start-row['x'])/pitch)*pitch
            if start < low:
                continue
            sites = [{'x_dbu': start+i*width, 'y_dbu': row['y']} for i in range(count)]
            candidates.append((abs(start-x)+abs(row['y']-y), start, row['y'], sites))
    return min(candidates)[3] if candidates else None


def clock_context(reference, features, platform):
    reference, features = Path(reference), Path(features)
    measurement = features.parent / 'reference-measurement/native/measurement'
    clocks = clock_table(measurement / 'assembly/passive/clock_sinks_by_clock.tsv')
    sinks = {p for group in clocks.values() for p in group}
    native_pins = {r['pin']: r for r in read_tsv(features / 'pins.tsv')}
    buffers = library(platform)
    actions, branches = {}, {}
    for die in ('bottom', 'top'):
        folder = reference / ('readback-' + die)
        _, _, cells = load_snapshot(folder)
        units, geo = geometry(features / (die + '-geometry.tsv'))
        pins = read_tsv(folder / 'pins.tsv')
        by_net, by_cell = {}, {}
        for row in pins:
            if row['signal_type'] in ('POWER', 'GROUND'):
                continue
            by_net.setdefault(row['net'], []).append(row)
            by_cell.setdefault(row['instance'], []).append(row)
        boundaries = {r['net'] for r in read_tsv(folder / 'ports.tsv') if r['net']}
        def descendants(net, seen):
            if net in seen or net in boundaries:
                return None
            seen = seen | {net}
            found = set()
            for pin in by_net.get(net, []):
                if pin['direction'] != 'INPUT':
                    continue
                full = 'u_' + die + '/' + pin['instance'] + '/' + pin['pin']
                if full in sinks:
                    found.add(full)
                elif cells[pin['instance']]['clock_cell']:
                    outputs = [p['net'] for p in by_cell[pin['instance']] if p['direction'] == 'OUTPUT' and p['net']]
                    for output in outputs:
                        children = descendants(output, seen)
                        if children is None:
                            return None
                        found |= children
                else:
                    return None
            return found
        options = []
        for name, cell in cells.items():
            if not cell['clock_cell']:
                continue
            outputs = [p for p in by_cell[name] if p['direction'] == 'OUTPUT' and p['net']]
            inputs = [p for p in by_cell[name] if p['direction'] == 'INPUT' and p['net']]
            if len(outputs) != 1 or len(inputs) != 1:
                continue
            affected = descendants(outputs[0]['net'], set())
            if not affected:
                continue
            options.append((-len(affected), name, inputs[0], affected))
        # Deterministic largest exclusive branches, not a measured candidate search.
        for _, name, pin, affected in sorted(options)[:64]:
            full = 'u_' + die + '/' + name + '/' + pin['pin']
            feature = native_pins[full]
            slew = max(float(feature['slew_rise_ns']), float(feature['slew_fall_ns']))
            load = float(feature['capacitance_pf'])
            if not all(math.isfinite(v) and 0 <= v < 1e6 for v in (slew, load)):
                continue
            branch = die + '/' + name
            branches[branch] = {'die': die, 'target_input_pin': name + '/' + pin['pin'],
                'affected_original_sinks': sorted(affected), 'input_slew_ns': slew, 'load_pf': load}
            for master in ('BUF_X1_' + die, 'BUF_X4_' + die):
                lib = buffers[master]
                examples = [r for r in geo['cell'].values() if r['extra'] == master]
                if not examples:
                    # Dimensions come from native ODB only, never guessed from area.
                    continue
                width, height = examples[0]['w'], examples[0]['h']
                for count in (1, 2, 4, 8):
                    sites = free_sites(geo, cells[name]['x_dbu'], cells[name]['y_dbu'], width, height, count)
                    if sites is None:
                        continue
                    delays = []
                    for edge in ('rise', 'fall'):
                        total, current_slew = 0., slew
                        for i in range(count):
                            stage_load = load if i == count-1 else lib['input_capacitance_pf']
                            total += interpolate(lib['tables']['cell_' + edge], current_slew, stage_load)
                            current_slew = interpolate(lib['tables'][edge + '_transition'], current_slew, stage_load)
                        delays.append(total)
                    identity = branch + '/' + master + '/' + str(count)
                    actions[identity] = {'qualified': True, 'qualification_scope': 'connectivity-and-empty-row-template-not-post-edit-STA',
                        'exclusive': True, 'branch_id': branch, 'affected_original_sinks': sorted(affected),
                        'placement_site_ids': [die + ':' + str(s['x_dbu']) + ':' + str(s['y_dbu']) for s in sites],
                        'added_buffers': count, 'added_area_um2': count*lib['area_um2'],
                        'master': master, 'die': die, 'target_input_pin': name + '/' + pin['pin'], 'sites': sites,
                        'input_slew_ns': slew, 'load_pf': load,
                        'delay_estimate_ns': {'min': min(delays), 'max': max(delays)},
                        'estimate_kind': 'Liberty-NLDM-cell-only; excludes changed upstream load and placement wire RC',
                        'post_edit_measurement_available': False}
    if not actions:
        raise ValueError('No physically supported exclusive clock branch actions; do not run an empty-context final')
    paths = []
    for row in read_tsv(features / 'paths.tsv'):
        value = float(row['slack_ns'])
        if math.isfinite(value) and abs(value) < 1e6:
            paths.append({**row, 'slack_ns': value})
    baseline = read_json(measurement / 'result.json')
    limits = {'max_added_buffers': 8, 'max_added_area_um2': 8*max(b['area_um2'] for b in buffers.values())}
    return {'schema': 'clock-compensation-context-v2', 'seed': 1001, 'actions': actions, 'branches': branches,
        'clock_groups': clocks, 'timing': baseline['timing'], 'paths': paths,
        'path_coverage': 'native endpoint worst paths by analysis and transition; not all graph paths',
        'buffer_cells': buffers, 'limits': limits,
        'timing_tolerances': {k: 1e-6 for k in ('setup_wns_ns', 'setup_tns_ns', 'hold_wns_ns', 'hold_tns_ns')},
        'reference_measurement_sha256': sha(measurement / 'result.json'),
        'final_evaluation': 'one selected plan, current-stage native RC/STA and guards, no counterfactual physical search'}
