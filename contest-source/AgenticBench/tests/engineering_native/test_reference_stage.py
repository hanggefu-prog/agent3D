import json
from pathlib import Path

import pytest

from cts_measure import sha
from cts_reference_stage import cts_arguments, verify_prefix


def config():
    return {'pdk': 'ng45', 'threads': 4, 'partition': {'seed': 1001}, 'cts': {
        'algorithm': 'openroad', 'repair_hold': False, 'target_skew_ns': .1,
        'cluster_size': 0, 'cluster_diameter_um': 0., 'delay_buffer_derate': 1.}}


def test_reference_preserves_cts_parameters_and_physical_threads():
    flags, threads = cts_arguments(config(), input_root='/input', platform='/platform', output='/out')
    assert threads == 4
    assert flags[flags.index('--checkpoint') + 1] == '/input/60_opt/checkpoint.json'
    assert '--repair-hold' not in flags
    assert 'global-route' not in flags and '--from' not in flags
    assert flags[flags.index('--cluster-size') + 1] == '0'


@pytest.mark.parametrize('key,value', [('repair_hold', True), ('algorithm', 'other')])
def test_no_silent_contract_substitution(key, value):
    c = config()
    c['cts'][key] = value
    with pytest.raises(ValueError):
        cts_arguments(c, input_root='/input', platform='/platform', output='/out')


def prefix(tmp_path):
    for folder in ('00_input', '10_floorplan', '20_partition', '30_placement', '40_pdn', '50_hbt', '60_opt'):
        p = tmp_path / folder
        p.mkdir()
        artifact = p / 'data.txt'
        artifact.write_text('synthetic prefix')
        (p / 'stage_manifest.json').write_text(json.dumps({'artifacts': {
            'data': {'path': 'data.txt', 'bytes': artifact.stat().st_size, 'sha256': sha(artifact)}}}))
    return tmp_path


def test_prefix_hashes_are_read_only_and_reject_changes(tmp_path):
    root = prefix(tmp_path)
    before = {str(f): f.read_bytes() for f in root.rglob('*') if f.is_file()}
    result = verify_prefix(root)
    assert len(result) == 7
    assert before == {str(f): f.read_bytes() for f in root.rglob('*') if f.is_file()}
    (root / '60_opt/data.txt').write_text('changed')
    with pytest.raises(ValueError, match='Changed'):
        verify_prefix(root)


def test_prefix_cannot_escape_stage(tmp_path):
    root = prefix(tmp_path)
    p = root / '60_opt/stage_manifest.json'
    data = json.loads(p.read_text())
    data['artifacts']['data']['path'] = '../50_hbt/data.txt'
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Unsafe'):
        verify_prefix(root)
