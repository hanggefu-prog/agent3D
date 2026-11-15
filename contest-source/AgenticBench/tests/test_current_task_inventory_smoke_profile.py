"""The CPU-only external smoke input stays small, unqualified and loadable."""
import json
from pathlib import Path

import yaml

from local_bench.benchmark_input import load_benchmark_input_profile
from local_bench.model import load_task


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROFILE = ROOT / "local_bench/benchmark_inputs/ibex_lol_ng45_cpu_smoke"


def test_cpu_smoke_profile_is_unqualified_and_cpu_only():
    raw = yaml.safe_load((PROFILE / "profile.yaml").read_text())
    assert raw["qualified"] is False
    assert raw["runtime"]["gpu"] is False
    profile = load_benchmark_input_profile("ibex_lol_ng45_cpu_smoke")
    task = load_task(ROOT / "local_tasks/placement_initialization")
    assert profile.compatible_tasks == (task.task_id,)
    assert list(profile.required_stages) == task.upstream_stages
    assert profile.partition_method == "tp-gnn-flat"
    config = yaml.safe_load(profile.config.read_text())
    assert config["design"] == "ibex" and config["pins"]["gpu"] is False
    assert config["placement"]["gpu"] is False


def test_cpu_smoke_partition_config_differs_from_default_only_by_device():
    config = yaml.safe_load((PROFILE / "config.yaml").read_text())
    # Open3DBench resolves relative partition configs under Place-3D.
    partition = (REPO / "Place-3D" / config["partition"]["config"]).resolve()
    assert partition == (PROFILE / "partition_tp_gnn_cpu.json").resolve()
    cpu = json.loads(partition.read_text())
    default = json.loads((REPO / "Place-3D/config/partition/tp_gnn_fast.json").read_text())
    assert cpu["partition"].pop("tp_gnn_device") == "cpu"
    assert default["partition"].pop("tp_gnn_device") == "cuda"
    assert cpu == default


def test_cpu_smoke_config_matches_task_base_config_outside_cpu_switches():
    task = load_task(ROOT / "local_tasks/placement_initialization")
    base = yaml.safe_load(Path(task.base_config).read_text())
    smoke = yaml.safe_load((PROFILE / "config.yaml").read_text())
    assert {k for k in set(base) | set(smoke) if base.get(k) != smoke.get(k)} == {
        "design", "pins", "partition", "placement"}
    assert {k: v for k, v in smoke["placement"].items() if k != "gpu"} == {
        k: v for k, v in base["placement"].items() if k != "gpu"}
    assert {k: v for k, v in smoke["pins"].items() if k != "gpu"} == {
        k: v for k, v in base["pins"].items() if k != "gpu"}
    assert smoke["partition"]["algorithm"] == base["partition"]["algorithm"]
    assert smoke["partition"]["seed"] == base["partition"]["seed"]
