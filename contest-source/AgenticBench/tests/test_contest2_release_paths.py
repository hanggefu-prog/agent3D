from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from contest2.agent_workspace import stage_agent
from contest2.pack_references import packed_receipt
from grt_bench.common import sha, write
from local_bench.matrix import LocalEvaluationMatrix
from local_bench.model import load_task
from local_bench.program_bridge import materialize_program
from local_bench.seeds import cell_contract
from local_bench.packed_gnn_reference import admit as admit_packed, compatible_implementations
from local_bench.gnn_evolution_timing import TIMING_FILES


ROOT = Path(__file__).resolve().parents[1]


def test_agent_workspace_preserves_evolution_and_applies_developer_edits(tmp_path):
    template = tmp_path / "source/AgenticBench/agent/binary_loss"
    template.mkdir(parents=True)
    (template / "agent.py").write_text("original strategy")
    (template / "program.py").write_text("initial program")
    destination = tmp_path / "work/participant-agent/binary_loss"
    stage_agent(template, destination)
    (destination / "program.py").write_text("evolved program")
    stage_agent(template, destination)
    assert (destination / "program.py").read_text() == "evolved program"
    assert (template / "program.py").read_text() == "initial program"
    (template / "agent.py").write_text("new developer strategy")
    stage_agent(template, destination)
    assert (destination / "agent.py").read_text() == "new developer strategy"
    assert (destination / "program.py").read_text() == "evolved program"
    assert template.resolve() not in destination.resolve().parents
    from contest2.agent_container import command
    paths = {name: tmp_path / "work" / name for name in ("workspaces", "scratch", "bridge", "input")}
    for path in paths.values():
        path.mkdir()
    with patch("pathlib.Path.is_socket", return_value=True):
        argv = command(name="o3db-contest-agent-123456789abc", image="sha256:" + "a" * 64,
            agent_dir=destination, workspaces=paths["workspaces"], scratch=paths["scratch"],
            bridge_dir=paths["bridge"], framework=tmp_path / "source/AgenticBench",
            source=tmp_path / "source", training_inputs={"bp-ng45": paths["input"]})
    assert "/workspace/agent" in " ".join(argv)


def test_agent_workspace_rejects_symlink_write_target(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "agent.py").write_text("first")
    destination = stage_agent(template, tmp_path / "work/agent")
    protected = tmp_path / "protected.py"
    protected.write_text("protected")
    (destination / "agent.py").unlink()
    (destination / "agent.py").symlink_to(protected)
    (template / "agent.py").write_text("second")
    with pytest.raises(ValueError, match="symlink"):
        stage_agent(template, destination)
    assert protected.read_text() == "protected"


@pytest.mark.parametrize("task", ["tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"])
@pytest.mark.parametrize("tamper", ["reference_summary.json", "20_partition/partition_assignment.csv",
                                    "../candidate_snapshot.json"])
def test_compact_reference_retains_native_contract_and_checks_packed_files(tmp_path, task, tamper):
    spec = load_task(ROOT / "local_tasks" / task)
    source = tmp_path / "source"
    source.mkdir()
    baseline = tmp_path / "input"
    reference = tmp_path / "reference/run"
    reference.mkdir(parents=True)
    seeds = {"io": 1, "partition": 1001, "placement": 1001, "routing": -1}
    config = {"design": "bp", "type": "lol", "pdk": "ng45"}
    for stage, key in (("pins", "io"), ("partition", "partition"),
                       ("placement", "placement"), ("routing", "routing")):
        config[stage] = {"seed": seeds[key]}
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(config))
    for stage, directory in (("input", "00_input"), ("floorplan", "10_floorplan")):
        manifest = {"status": "success", "design": "bp", "integration": "lol", "pdk": "ng45"}
        if stage == "floorplan":
            metrics = baseline / directory / "io_metrics.json"
            write(metrics, {"seed": 1})
            manifest["artifacts"] = {"io_metrics": {
                "path": metrics.name, "bytes": metrics.stat().st_size, "sha256": sha(metrics)}}
        write(baseline / directory / "stage_manifest.json", manifest)
    initial = materialize_program(spec, spec.program_template.read_text(), source)
    native_contract = cell_contract(spec, initial, source, baseline, None, config_path, seeds)
    receipt = {"metrics": {"runs_successfully": 1},
               "artifacts": {"task": task, "matrix_reference": True,
                   "matrix_cell_contract": native_contract, "reference_admission": {
                   "mode": "FRESH", "task_id": task, "implementation_sha256": {
                   "physical": "a" * 64, "harness/scoped_reference.py": "b" * 64}}}}
    consumer = {"physical": "a" * 64, "harness/scoped_reference.py": "c" * 64,
                "harness/packed_gnn_reference.py": "d" * 64}
    assignment = reference / "20_partition/partition_assignment.csv"
    assignment.parent.mkdir()
    assignment.write_text("inst_name,die\na,top\n")
    summary = reference / "reference_summary.json"
    write(summary, {"schema_version": 1, "task_id": task, "wns": -1, "tns": -2,
        "partition_config": {}, "timing_sha256": {name: "a" * 64 for name in TIMING_FILES},
        "native_stage_sha256": {name: "b" * 64 for name in
            ("20_partition", "30_placement", "40_pdn", "50_hbt", "60_opt")},
        "assignment_sha256": sha(assignment), "independent_partition_metrics": {}})
    snapshot = reference.parent / "candidate_snapshot.json"
    write(snapshot, initial)
    packed = packed_receipt(receipt, summary, assignment, candidate_snapshot=snapshot,
                           producer_receipt_sha256="e" * 64, consumer_identity=consumer)
    reduced = {"metrics": receipt["metrics"], "artifacts": {
        "matrix_reference": True, "matrix_cell_contract": native_contract}}
    with pytest.raises(ValueError, match="Native reference authority is missing"):
        packed_receipt(reduced, summary, assignment, candidate_snapshot=snapshot,
                       producer_receipt_sha256="e" * 64, consumer_identity=consumer)
    assert packed["artifacts"]["matrix_cell_contract"] == native_contract
    assert "packed_reference" not in receipt["artifacts"]
    write(reference.parent / "evaluation_result.json", packed)
    matrix_path = tmp_path / "matrix.json"
    write(matrix_path, {"schema_version": 1, "task": task, "cases": ["bp-ng45"],
        "seeds": [{"id": "1001", **seeds}], "cells": [{"case": "bp-ng45", "seed": "1001",
        "base_config": str(config_path), "baseline_run": str(baseline),
        "reference_run": str(reference)}]})
    matrix = LocalEvaluationMatrix(matrix_path, spec)
    assert cell_contract(spec, initial, source, baseline, reference, config_path, seeds) == native_contract
    assert admit_packed(spec, initial, consumer, baseline, reference, packed)["mode"] == "ADOPT_PACKED"
    with pytest.raises(ValueError, match="consumer implementation differs"):
        admit_packed(spec, initial, {**consumer, "physical": "f" * 64}, baseline, reference, packed)
    matrix.assert_unchanged()
    (reference / tamper).write_text("changed")
    with pytest.raises(ValueError, match="Packed GNN reference digest differs"):
        LocalEvaluationMatrix(matrix_path, spec)


def test_packed_reference_migration_rejects_physical_or_runtime_changes():
    producer = {"physical": "a", "runtime_image": "image", "harness/scoped_reference.py": "old"}
    consumer = {**producer, "harness/scoped_reference.py": "new",
                "harness/packed_gnn_reference.py": "reader"}
    compatible_implementations(producer, consumer)
    for changed in ({**consumer, "physical": "changed"}, {**consumer, "runtime_image": "changed"}):
        with pytest.raises(ValueError, match="physical or scoring implementation differs"):
            compatible_implementations(producer, changed)
    with pytest.raises(ValueError, match="implementation fields differ"):
        compatible_implementations(producer, {**consumer, "unknown": "changed"})
