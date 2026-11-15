"""The accepted embedding-baseline halo schedule, shared by both GNN tasks."""
from pathlib import Path

import yaml

from .scoring import SEEDS

HALOS = {
    'bp_fe-ng45': (5.0, 5.0, 5.0),
    'bp-ng45': (10.0, 10.0, 10.0),
    'ariane133-ng45': (10.0, 5.0, 10.0),
    'swerv_wrapper-ng45': (5.0, 5.0, 5.0),
    'bp_fe-asap7': (0.0, 0.0, 0.0),
}


def halo_for(case, seed):
    if case not in HALOS or str(seed) not in {str(value) for value in SEEDS}:
        raise ValueError('Unknown public case or seed for the shared halo schedule')
    return HALOS[case][SEEDS.index(int(seed))]


def validate_matrix_halos(matrix):
    """Reject an old configuration before model fees or matrix acceptance."""
    for cell in matrix.cells:
        expected = halo_for(cell['case'], cell['seed'])
        config = yaml.safe_load(Path(cell['base_config']).read_text())
        actual = config.get('placement', {}).get('macro_halo_um')
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != expected:
            raise ValueError(f"Shared halo differs for {cell['case']}/{cell['seed']}: "
                             f"expected {expected}, found {actual}. Regenerate the affected "
                             "reference; do not rewrite its receipt.")
