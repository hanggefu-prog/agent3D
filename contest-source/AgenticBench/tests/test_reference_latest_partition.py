"""Current no-OPT timing must not adopt a retired cut-score authority."""
import pytest
from local_bench import reference_local_partition as local


@pytest.mark.parametrize("case", ["bp_fe", "bp", "ariane133", "swerv_wrapper"])
def test_current_timing_requires_its_own_reference_contract(tmp_path, case):
    with pytest.raises(ValueError, match="trusted versioned task"):
        local._context(local.SOURCE, "tp_gnn_post_placement_timing", case, tmp_path)
