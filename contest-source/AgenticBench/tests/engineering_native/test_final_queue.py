from copy import deepcopy

import pytest

from cts_final_queue import budget_check, electrical


def test_electrical_native_units_and_clock_filter(tmp_path):
    path = tmp_path / 'native.rpt'
    path.write_text('max slew\n\nu_bottom/b/Z 0.2 0.3 -0.1 (VIOLATED)\n'
                    'u_bottom/data/Z 0.2 1.0 -0.8 (VIOLATED)\n'
                    'max capacitance\nu_bottom/ff/CK 100 150 -50 (VIOLATED)\n')
    report = electrical(path, {'u_bottom/b'}, {'u_bottom/ff/CK'})
    assert report['data_violation_rows_diagnostic'] == 1
    assert report['clock']['max slew']['count'] == 1
    assert report['clock']['max capacitance']['worst_excess_ratio'] == .5
    assert report['clock']['max fanout']['count'] == 0


def test_electrical_malformed_report_refused(tmp_path):
    path = tmp_path / 'native.rpt'
    path.write_text('strange format (VIOLATED)')
    with pytest.raises(ValueError):
        electrical(path, set(), set())


def test_frozen_budget_and_electrical_nonregression():
    ref = {'clock_area_um2': 10., 'clock_buffer_count': 5,
           'electrical': {'clock': {'max slew': {'count': 1, 'worst_excess_ratio': .2}}}}
    budget_check(ref, ref)
    bad = deepcopy(ref)
    bad['clock_area_um2'] = 20.1
    with pytest.raises(ValueError, match='clock_area'):
        budget_check(bad, ref)
    bad = deepcopy(ref)
    bad['electrical']['clock']['max slew']['count'] = 2
    with pytest.raises(ValueError, match='max slew'):
        budget_check(bad, ref)
