from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import yaml

from contest2.halo_policy import HALOS, halo_for, validate_matrix_halos
from contest2.prepare_matrix import render
from contest2.scoring import PUBLIC_SPLITS, SEEDS


def test_both_tasks_render_the_accepted_embedding_halo_schedule(tmp_path):
    index = {}
    for case in PUBLIC_SPLITS['train']:
        index[case] = {}
        for seed in SEEDS:
            prefix = tmp_path / 'inputs' / case / str(seed)
            for stage in ('00_input', '10_floorplan'):
                (prefix / stage).mkdir(parents=True)
                (prefix / stage / 'stage_manifest.json').write_text('{}')
            index[case][str(seed)] = str(prefix)
    index_file = tmp_path / 'inputs.json'
    index_file.write_text(json.dumps(index))
    policies = []
    for baseline in ('binary_loss', 'loss_decoder'):
        path = render(baseline=baseline, split='train', input_index=index_file,
                      output=tmp_path / baseline)
        matrix = yaml.safe_load(path.read_text())
        validate_matrix_halos(SimpleNamespace(cells=matrix['cells']))
        policies.append({(cell['case'], cell['seed']): yaml.safe_load(
            Path(cell['base_config']).read_text())['placement']['macro_halo_um']
            for cell in matrix['cells']})
    assert policies[0] == {key: value for key, value in policies[1].items() if key in policies[0]}
    assert ('ariane133-ng45', '1001') not in policies[0]
    assert ('ariane133-ng45', '1002') not in policies[0]
    assert policies[0]['ariane133-ng45', '1003'] == 10
    assert [policies[1]['ariane133-ng45', str(seed)] for seed in SEEDS] == [10, 5, 10]
    assert all(policies[0]['bp_fe-ng45', str(seed)] == 5 for seed in SEEDS)


@pytest.mark.parametrize('actual', [5.0, True, None, float('nan')])
def test_old_or_invalid_probability_halo_is_rejected(tmp_path, actual):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'placement': {'macro_halo_um': actual}}))
    matrix = SimpleNamespace(cells=[{'case': 'ariane133-ng45', 'seed': '1001',
                                   'base_config': str(config)}])
    with pytest.raises(ValueError, match='Regenerate the affected reference'):
        validate_matrix_halos(matrix)


def test_public_validation_policy_is_explicit():
    assert set(HALOS) == set(PUBLIC_SPLITS['train'] + PUBLIC_SPLITS['validation'])
    assert [halo_for('swerv_wrapper-ng45', seed) for seed in SEEDS] == [5, 5, 5]
    assert [halo_for('bp_fe-asap7', seed) for seed in SEEDS] == [0, 0, 0]
    with pytest.raises(ValueError, match='Unknown public case'):
        halo_for('unknown-ng45', 1001)
