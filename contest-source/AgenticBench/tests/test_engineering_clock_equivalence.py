import csv
from pathlib import Path

import pytest

from engineering_bench import clock_equivalence as proof


def test_inversion_union_is_order_independent():
    for relations in ([('a', 'b', 1), ('b', 'c', 1)], [('c', 'b', 1), ('b', 'a', 1)]):
        sets = proof.ParitySets()
        for a, b, phase in relations:
            sets.join(a, b, phase)
        assert sets.find('a') == sets.find('c')
        assert sets.find('a')[1] != sets.find('b')[1]
        with pytest.raises(ValueError, match='Contradictory'):
            sets.join('a', 'c', 1)


def table(path, fields, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def row(master='DFF_X1', clock=0, inverter=0):
    return dict(master=master, clock_cell=clock, buffer=int(clock and not inverter),
                inverter=inverter, cts_owned=clock, master_type='CORE',
                x_dbu=0, y_dbu=0, orient='R0', status='PLACED')


def circuit(folder, fake, *, chain=(), corrupt=False, extra_driver=False, dangling=False):
    snapshots = {}
    for die in ('bottom', 'top'):
        path = folder / die
        path.mkdir(parents=True)
        snapshots[die] = path
        cells = {'ff': row()}
        pins, ports = [], []
        net = 'clk'
        if die == 'bottom':
            ports.append(dict(port='clock', net='clk', direction='INPUT', signal_type='SIGNAL'))
            for n, inv in enumerate(chain):
                name, new = 'buf' + str(n), 'leaf' + str(n)
                cells[name] = row('INV_X1' if inv else 'BUF_X1', 1, inv)
                pins += [dict(instance=name, pin='A', net=net, direction='INPUT', signal_type='SIGNAL'),
                         dict(instance=name, pin='Y', net=new, direction='OUTPUT', signal_type='SIGNAL')]
                net = new
            if extra_driver:
                ports.append(dict(port='other_clock', net=net, direction='INPUT', signal_type='SIGNAL'))
            if dangling:
                cells['load'] = row('INV_X1', 1, 1)
                pins += [dict(instance='load', pin='A', net=net, direction='INPUT', signal_type='SIGNAL'),
                         dict(instance='load', pin='Y', net='', direction='OUTPUT', signal_type='SIGNAL')]
            ports.append(dict(port='H_SRC', net=net, direction='OUTPUT', signal_type='SIGNAL'))
        else:
            ports.append(dict(port='H_DST', net='clk', direction='INPUT', signal_type='SIGNAL'))
        ports.append(dict(port='data', net='data', direction='INPUT', signal_type='SIGNAL'))
        pins += [dict(instance='ff', pin='CK', net=net, direction='INPUT', signal_type='SIGNAL'),
                 dict(instance='ff', pin='D', net=net if corrupt else 'data', direction='INPUT', signal_type='SIGNAL')]
        table(path / 'pins.tsv', ['instance', 'pin', 'net', 'direction', 'signal_type'], pins)
        table(path / 'ports.tsv', ['port', 'net', 'direction', 'signal_type'], ports)
        fake[str(path)] = cells
    return snapshots


@pytest.fixture
def native(monkeypatch, tmp_path):
    fake = {}
    monkeypatch.setattr(proof, 'load_snapshot', lambda p: ({}, {}, fake[str(p)]))
    hbt = tmp_path / 'hbt.csv'
    hbt.write_text('hbt_id,source_die,source_port,sink_die,sink_port\nH,bottom,H_SRC,top,H_DST\n')
    before = circuit(tmp_path / 'before', fake)
    return fake, hbt, before


@pytest.mark.parametrize('chain', [(), (0,), (0, 0), (1, 1), (0, 1, 0, 1)])
def test_equivalent_physical_clock_topologies_pass(native, tmp_path, chain):
    fake, hbt, before = native
    after = circuit(tmp_path / 'after', fake, chain=chain, dangling=True)
    result = proof.compare(before, after, hbt)
    assert result['passed'] and result['original_anchor_count'] == result['candidate_anchor_count']


@pytest.mark.parametrize('change', [{'chain': (1,)}, {'corrupt': True}])
def test_phase_and_data_connectivity_changes_rejected(native, tmp_path, change):
    fake, hbt, before = native
    after = circuit(tmp_path / 'after', fake, **change)
    with pytest.raises(ValueError, match='connectivity or inversion'):
        proof.compare(before, after, hbt)


def test_parallel_electrical_drivers_rejected(native, tmp_path):
    fake, hbt, before = native
    after = circuit(tmp_path / 'after', fake, extra_driver=True)
    with pytest.raises(ValueError, match='one electrical driver'):
        proof.compare(before, after, hbt)


def test_hbt_boundary_cannot_be_reused(native):
    fake, hbt, before = native
    hbt.write_text(hbt.read_text() + 'H2,bottom,H_SRC,top,H_DST\n')
    with pytest.raises(ValueError, match='repeated HBT boundary'):
        proof.compare(before, before, hbt)


def test_local_noninverting_addition_and_fixed_cell_guard(native, tmp_path):
    fake, hbt, before = native
    after = circuit(tmp_path / 'local', fake, chain=(0,))
    with hbt.open() as stream:
        links = list(csv.DictReader(stream))
    buf = fake[str(after['bottom'])]['buf0']
    buf['clock_cell'] = buf['cts_owned'] = 0
    result = proof.compare_local(before, after, links, links, [('bottom', 'buf0')])
    assert result['passed']
    with pytest.raises(ValueError, match='Unmapped new'):
        proof.compare_local(before, after, links, links)
    fake[str(after['top'])]['ff']['x_dbu'] = 10
    with pytest.raises(ValueError, match='Original local-task cell changed'):
        proof.compare_local(before, after, links, links, [('bottom', 'buf0')])


def test_local_inverting_addition_is_not_an_allowed_buffer(native, tmp_path):
    fake, hbt, before = native
    after = circuit(tmp_path / 'local', fake, chain=(1,))
    with hbt.open() as stream:
        links = list(csv.DictReader(stream))
    with pytest.raises(ValueError, match='Unproven added buffer'):
        proof.compare_local(before, after, links, links, [('bottom', 'buf0')])
