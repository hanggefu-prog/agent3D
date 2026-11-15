import json
from pathlib import Path

import pytest
import yaml

from local_bench.benchmark_input import load_benchmark_input_profile
from local_bench.model import load_task
from local_bench.placement_baseline import validate_tp_gnn_partition

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case", ["bp_fe", "bp", "bp_multi", "ariane133", "swerv_wrapper"])
def test_new_placement_input_is_tp_gnn_and_legacy_is_unchanged(case):
    profile = load_benchmark_input_profile(case + "_lol_ng45_tp_gnn_placement")
    config = yaml.safe_load(profile.config.read_text())
    assert config["design"] == case and config["partition"]["algorithm"] == "tp-gnn"
    assert profile.stop_stage == "partition"
    assert {"placement_preconditioner_source", "placement_dynamic_parameters", "placement_two_pass_sta"} <= set(profile.compatible_tasks)
    old = load_benchmark_input_profile(case + "_lol_ng45_mtkahypar")
    assert yaml.safe_load(old.config.read_text())["partition"]["algorithm"] == "mtkahypar"


@pytest.mark.parametrize("path", ["evaluation_profiles/placement_two_pass"])
def test_default_placement_tasks_use_tp_gnn(path):
    assert load_task(ROOT / path).benchmark_input_id.endswith("_tp_gnn_placement")


@pytest.mark.parametrize("method,decoder", [("tp-gnn", "global-2way"), ("mtkahypar", "global-2way"), ("tp-gnn", "ta3d-hierarchical")])
def test_partition_guard_reads_real_artifacts(tmp_path, method, decoder):
    part = tmp_path / "20_partition"
    part.mkdir()
    for file, data in {
        "stage_manifest.json": {"status": "success", "method": method},
        "partition_manifest.json": {"method": method},
        "tp_gnn_report.json": {"method": method, "config": {"decoder": decoder}},
    }.items():
        (part / file).write_text(json.dumps(data))
    if method == "tp-gnn" and decoder == "global-2way":
        validate_tp_gnn_partition(tmp_path)
    else:
        with pytest.raises(ValueError, match="actual TP-GNN"):
            validate_tp_gnn_partition(tmp_path)
