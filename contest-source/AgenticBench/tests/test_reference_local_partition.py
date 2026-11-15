"""Retired cut-score references cannot launch or qualify a current timing task."""
from pathlib import Path
import pytest
from local_bench import reference_local_partition as local


def test_retired_partition_reference_cannot_be_recreated(tmp_path):
    with pytest.raises(FileNotFoundError, match="unknown task"):
        local._context(local.SOURCE, local.TASK, "bp_fe", tmp_path)


def test_current_timing_task_cannot_use_cut_score_reference_producer(tmp_path):
    with pytest.raises(ValueError, match="trusted versioned task"):
        local._context(local.SOURCE, "tp_gnn_post_placement_timing", "bp_fe", tmp_path)
