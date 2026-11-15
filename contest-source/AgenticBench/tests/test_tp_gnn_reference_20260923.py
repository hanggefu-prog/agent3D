"""Offline task/input wiring; not native TP-GNN or reference qualification."""
import copy
import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from local_bench.benchmark_input import load_benchmark_input_profile
from local_bench.evaluation import (
    _materialize_frozen_inputs, _replay_start_stage, _validate_frozen_input,
    _write_resolved_config,
)
from local_bench.model import load_candidate, load_task, resolve_task_path
from local_bench.program_bridge import initial_program_text, materialize_program
from local_bench.verifier import verify_candidate
from local_tasks import tp_gnn_loss_patch as legacy
from local_tasks import tp_gnn_timing_context as current


ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "AgenticBench/local_tasks"
ID = "tp_gnn_post_placement_timing"
CASES = ("bp_fe", "bp", "ariane133", "swerv_wrapper")
PARTITION_CONFIG = "AgenticBench/local_bench/reference_configs/partition.json"


@pytest.fixture
def task():
    return load_task(TASKS / ID)


def initial(task):
    return initial_program_text(task, load_candidate(task.initial_candidate))


def test_current_loss_has_no_opt_post_placement_boundary(task):
    assert resolve_task_path(TASKS, ID) == task.root
    assert task.task_id == task.kind == ID
    assert task.start_stage == "partition" and task.stop_stage == "hbt"
    assert task.upstream_stages == ["input", "floorplan"]
    config = yaml.safe_load((task.root / "config.yaml").read_text())
    assert config["diff_based_evolution"] is False
    assert config["max_iterations"] == 50
    assert task.value_constraints[0]["enum"] == ["tp-gnn-flat"]




def test_prompt_preserves_effective_training_budget_policy(task):
    prompt = (task.root / "prompt.md").read_text()
    assert "optimizer steps" in prompt
    assert "20 maximum epochs" not in prompt


@pytest.mark.parametrize("factor", ["1.0", "1.1"])
def test_adapter_preserves_existing_safe_loss_patch(task, factor):
    program = initial(task).replace("return -(", "return -" + factor + " * (")
    candidate = materialize_program(task, program, ROOT)
    assert "graph_scores = deterministic_pair_dot" in candidate["source_patch"]
    assert "_agentic_loss_graph" in candidate["source_patch"]
    expected = current.materialize(program, ROOT)
    assert {k: v for k, v in candidate.items() if k != "notes"} == {k: v for k, v in expected.items() if k != "notes"}
    assert candidate["candidate_files"] == {}
    assert candidate["config_overrides"] == {"partition": {"algorithm": "tp-gnn-flat"}}
    proof = verify_candidate(task, candidate)
    assert proof["patch"]["paths"] == [legacy.TP_GNN]


@pytest.mark.parametrize("replacement", [
    "return open('/tmp/forbidden').read()",
    "return torch.load('/tmp/forbidden')",
])
def test_existing_ast_safety_is_not_weakened(task, replacement):
    program = initial(task)
    start = program.index("    return -(")
    end = program.index("# EVOLVE-BLOCK-END")
    program = program[:start] + "    " + replacement + "\n" + program[end:]
    with pytest.raises(ValueError):
        materialize_program(task, program, ROOT)


@pytest.mark.parametrize("change", ["legacy", "ta3d", "config", "threads", "source"])
def test_candidate_cannot_switch_method_recipe_or_source_scope(task, change):
    candidate = copy.deepcopy(materialize_program(task, initial(task), ROOT))
    if change in {"legacy", "ta3d"}:
        candidate["config_overrides"]["partition"]["algorithm"] = (
            "tp-gnn" if change == "legacy" else "ta3d-hierarchical")
    elif change == "config":
        candidate["config_overrides"]["partition"]["config"] = "config/partition/tp_gnn_ta3d.json"
    elif change == "threads":
        candidate["config_overrides"]["threads"] = 32
    else:
        candidate["source_patch"] = candidate["source_patch"].replace(legacy.TP_GNN, "open3dbench/runner.py")
    with pytest.raises(ValueError):
        verify_candidate(task, candidate)


def timing_fixture(task, root):
    contract = task.frozen_inputs[0]
    folder = root / contract.path
    folder.mkdir(parents=True)
    for name in contract.required_files:
        (folder / name).write_text("frozen fixture\n")
    (folder / "timing_manifest.json").write_text(json.dumps({
        "timing_status": "full_coverage_partition_last",
        "timing_weight_mode": "paper_delay_slack_hyperedge_v1"}))
    return folder


def test_frozen_timing_is_copied_hashed_not_recomputed_or_relabelled(task, tmp_path):
    baseline = tmp_path / "sealed"
    timing = timing_fixture(task, baseline)
    before = {p.name: p.read_bytes() for p in timing.iterdir()}
    output = tmp_path / "evaluation"
    output.mkdir()
    config = {"partition": {"algorithm": "tp-gnn-flat", "config": PARTITION_CONFIG}}
    record = _materialize_frozen_inputs(task, config, output / "config.yaml", output,
        baseline_run=baseline, reference_run=None)["tp_gnn_timing"]
    assert Path(record["source_path"]) == timing
    assert set(record["files"]) == set(task.frozen_inputs[0].required_files)
    adopted = Path(config["partition"]["timing_input"])
    assert {p.name: p.read_bytes() for p in adopted.iterdir()} == before
    assert {p.name: p.read_bytes() for p in timing.iterdir()} == before
    assert not (baseline / "15_timing_extract").exists()
    assert not (output / "15_timing_extract").exists()
    plan = json.dumps({"plan": [{"stage": "timing_extract", "skip": None},
                                {"stage": "partition", "skip": None}]})
    assert _replay_start_stage(task, config, plan) == "timing_extract"


@pytest.mark.parametrize("field,value", [
    ("timing_status", "partial"), ("timing_status", "placement_estimated"),
    ("timing_weight_mode", "unit_weights"), ("timing_weight_mode", None),
])
def test_incompatible_frozen_sta_contract_fails_closed(task, tmp_path, field, value):
    timing = timing_fixture(task, tmp_path)
    manifest = timing / "timing_manifest.json"
    data = json.loads(manifest.read_text())
    if value is None:
        data.pop(field)
    else:
        data[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        _validate_frozen_input(timing, task.frozen_inputs[0])


@pytest.mark.parametrize("filename", [
    "timing_features.csv", "timing_net_weights.csv", "timing_manifest.json",
    "openroad_full_partition_last.connections.tsv",
])
def test_each_frozen_sta_file_remains_mandatory(task, tmp_path, filename):
    timing = timing_fixture(task, tmp_path)
    (timing / filename).unlink()
    with pytest.raises(FileNotFoundError):
        _validate_frozen_input(timing, task.frozen_inputs[0])
