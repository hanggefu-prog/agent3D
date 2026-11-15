import pytest
from cts_measure import endpoint_inventory, EXTRA, EXPORT, measurement_supplement, clock_summary


def test_unavailable_min_analysis_not_zero(tmp_path):
    p = tmp_path / 'endpoints.tsv'
    p.write_text('pin\tsetup_slack_ns\thold_slack_ns\na/D\t-1.2\tINF\nb/D\t0.1\t-0.1\n')
    rows = endpoint_inventory(p)
    assert rows['endpoint_count'] == 2
    assert rows['endpoints_without_min_path'] == ['a/D']
    assert rows['hold_violating_endpoints'] == ['b/D']


@pytest.mark.parametrize('value', ['NaN', '-INF', 'not-a-value'])
def test_invalid_timing_not_hidden(tmp_path, value):
    p = tmp_path / 'endpoints.tsv'
    p.write_text('pin\tsetup_slack_ns\thold_slack_ns\na/D\t1\t' + value + '\n')
    with pytest.raises(ValueError):
        endpoint_inventory(p)


def test_endpoint_counts_only_after_joint_sta():
    assert 'endpoint_violation_count' not in EXPORT
    assert 'endpoint_violation_count min' in EXTRA
    # endpoint_slack's group argument is exact, not an empty-string wildcard.
    assert 'endpoint_slack $pin {}' not in EXTRA
    assert 'get_property $pin slack_min' in EXTRA


def test_skew_mode_has_no_endpoint_guard_or_power_analysis():
    text = measurement_supplement('skew')
    assert 'sta::endpoints' not in text
    assert 'endpoint_violation_count' not in text
    assert 'report_check_types' in text  # Current-state electrical legality remains.
    assert 'report_power' not in EXTRA
    assert 'global_route' not in text + EXTRA
    with pytest.raises(ValueError, match='Unknown'):
        measurement_supplement('grt')


def native_clocks(tmp_path, last='1'):
    path = tmp_path / 'clocks.tsv'
    path.write_text('clock\tdie\tpin\tactive_edge\tarrival_max_rise_ns\tarrival_max_fall_ns\tarrival_min_rise_ns\tarrival_min_fall_ns\tclock_domain_count\n'
                    'clk\tbottom\ta/CK\trise\t0.1\t5.0\t0.1\t5.0\t1\n'
                    'clk\ttop\tb/CK\trise\t0.3\t5.0\t0.4\t5.0\t1\n'
                    'aux\ttop\tc/CK\tfall\t10\t20\t10\t20\t' + last + '\n')
    return path


def test_independent_skew_uses_same_domain_edge_view_only(tmp_path):
    result = clock_summary(native_clocks(tmp_path))
    assert result['clock_skew_ns'] == pytest.approx(.3)
    assert result['clock_groups']['aux/fall/max/default']['skew_ns'] == 0.
    assert result['clock_groups']['clk/rise/max/default']['sink_count'] == 2


def test_clock_skew_refuses_ambiguous_domains(tmp_path):
    with pytest.raises(ValueError, match='ambiguous'):
        clock_summary(native_clocks(tmp_path, '2'))
