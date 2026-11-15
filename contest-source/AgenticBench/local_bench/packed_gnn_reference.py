"""Admission for official compact GNN references with preserved native authority."""
from pathlib import Path
import re

from grt_bench.common import digest
from .reference_latest import read_json, require, sha256


TASKS = frozenset({"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"})
READER_FILES = frozenset({"harness/gnn_evolution_timing.py", "harness/scoped_reference.py",
                          "harness/packed_gnn_reference.py"})
FIELDS = frozenset({"schema_version", "summary_sha256", "assignment_sha256", "candidate_sha256",
                    "producer_receipt_sha256", "consumer_implementation_sha256"})


def compatible_implementations(producer, consumer):
    require(isinstance(producer, dict) and isinstance(consumer, dict)
            and producer.keys() - READER_FILES == consumer.keys() - READER_FILES,
            "Packed reference implementation fields differ")
    require(all(producer[key] == consumer[key] for key in producer if key not in READER_FILES),
            "Packed reference physical or scoring implementation differs")


def validate_files(run, receipt):
    run = Path(run)
    packed = receipt.get("artifacts", {}).get("packed_reference")
    require(isinstance(packed, dict) and set(packed) == FIELDS and packed["schema_version"] == 1,
            "Packed GNN reference authority is incomplete")
    require(isinstance(packed["consumer_implementation_sha256"], dict)
            and bool(packed["consumer_implementation_sha256"]), "Packed reference consumer identity is missing")
    for name in ("summary_sha256", "assignment_sha256", "candidate_sha256", "producer_receipt_sha256"):
        require(isinstance(packed[name], str) and re.fullmatch("[0-9a-f]{64}", packed[name]),
                "Packed GNN reference digest is invalid")
    paths = {
        "summary_sha256": run / "reference_summary.json",
        "assignment_sha256": run / "20_partition/partition_assignment.csv",
        "candidate_sha256": run.parent / "candidate_snapshot.json",
    }
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink() and sha256(path) == packed[name],
                "Packed GNN reference digest differs: " + str(path))
    return packed, tuple(paths.values())


def admit(spec, native, identity, baseline_run, run, receipt):
    require(spec.task_id in TASKS, "Unknown packed GNN reference task")
    artifacts = receipt.get("artifacts", {})
    require(artifacts.get("task") == spec.task_id, "Reference belongs to a different task")
    require(receipt.get("metrics", {}).get("runs_successfully") == 1,
            "Reference evaluation did not succeed")
    producer = artifacts.get("reference_admission", {})
    require(isinstance(producer, dict) and producer.get("mode") == "FRESH"
            and producer.get("task_id") == spec.task_id,
            "Packed reference is not a native initial-program evaluation")
    packed, _ = validate_files(run, receipt)
    require(packed["consumer_implementation_sha256"] == identity,
            "Packed reference consumer implementation differs")
    compatible_implementations(producer.get("implementation_sha256"), identity)
    require(read_json(Path(run).parent / "candidate_snapshot.json") == native,
            "Reference initial program or trusted adapter differs")
    contract = artifacts.get("matrix_cell_contract", {})
    require(contract.get("task") == spec.task_id and contract.get("initial_candidate_sha256") == digest(native),
            "Packed reference initial matrix contract differs")
    from .evaluation import UPSTREAM_STAGE_DIRS
    require(contract.get("upstream") == {
        stage: sha256(Path(baseline_run) / UPSTREAM_STAGE_DIRS[stage] / "stage_manifest.json")
        for stage in spec.upstream_stages}, "Packed reference upstream identity differs")
    from .gnn_evolution_timing import _packed_reference
    _packed_reference(run, spec.task_id)
    return {"mode": "ADOPT_PACKED", "task_id": spec.task_id,
            "reference_receipt_sha256": sha256(Path(run).parent / "evaluation_result.json"),
            "producer_receipt_sha256": packed["producer_receipt_sha256"]}
