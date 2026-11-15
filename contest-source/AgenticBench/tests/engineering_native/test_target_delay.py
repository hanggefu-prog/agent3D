import pytest
from target_delay import parse_delays, requests


def context():
    return {'target_nets': {'net': {'original_driver': 'u_bottom/b/Z', 'original_remote_receivers': ['u_top/f/D']}},
        'driver_input_pins': {'u_bottom/b/Z': ['u_bottom/b/A']}}


def test_fixed_driver_and_complete_receiver_contract(tmp_path):
    ctx = context()
    assert requests(ctx).strip() == 'net\tu_bottom/b/A\tu_bottom/b/Z\tu_top/f/D\tu_bottom/b'
    path = tmp_path / 'delay.tsv'
    path.write_text('target\tsink\trise_delay_ns\tfall_delay_ns\tworst_delay_ns\nnet\tu_top/f/D\t0.1\t0.2\t0.2\n')
    assert parse_delays(path, ctx) == {'net': {'u_top/f/D': .2}}
    path.write_text(path.read_text().replace('0.2\n', '0.1\n'))
    with pytest.raises(ValueError, match='Invalid native receiver delay'):
        parse_delays(path, ctx)


def test_receiver_omission_and_nonfinite_rejected(tmp_path):
    path = tmp_path / 'delay.tsv'
    header = 'target\tsink\trise_delay_ns\tfall_delay_ns\tworst_delay_ns\n'
    path.write_text(header)
    with pytest.raises(ValueError, match='Missing fixed receiver'):
        parse_delays(path, context())
    path.write_text(header + 'net\tu_top/f/D\tnan\t0.2\t0.2\n')
    with pytest.raises(ValueError, match='Invalid native receiver delay'):
        parse_delays(path, context())
