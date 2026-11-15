"""Publish compact, independently checked GNN reference summaries.

Run this once on the organizer's qualified native reference tree. The public
data package contains frozen inputs and summaries, not full native runs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from grt_bench.common import sha
from local_bench.gnn_evolution_timing import TIMING_FILES, _independent, _opt_snapshot
from local_bench.matrix import LocalEvaluationMatrix
from local_bench.model import load_task
from local_bench.packed_gnn_reference import compatible_implementations
from local_bench.reference_latest import require
from local_bench.scoped_reference import implementation_identity

from .controller import TASK_IDS
from .halo_policy import validate_matrix_halos
from .sampling import coordinates, validate_public_matrix
from .scoring import PUBLIC_SPLITS


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def packed_receipt(receipt: dict, summary: Path, assignment: Path, *, candidate_snapshot: Path,
                   producer_receipt_sha256: str, consumer_identity: dict) -> dict:
    """Bind compact files without changing the native matrix cell contract."""
    artifacts = receipt.get("artifacts", {})
    producer = artifacts.get("reference_admission", {})
    require(receipt.get("metrics", {}).get("runs_successfully") == 1
            and artifacts.get("matrix_reference") is True
            and artifacts.get("task") in TASK_IDS.values() and isinstance(producer, dict)
            and producer.get("mode") == "FRESH" and producer.get("task_id") == artifacts.get("task"),
            "Native reference authority is missing; do not pack a reduced receipt")
    compatible_implementations(producer.get("implementation_sha256"), consumer_identity)
    receipt = {"metrics": deepcopy(receipt["metrics"]), "artifacts": {
        key: deepcopy(artifacts[key]) for key in
        ("task", "reference_admission", "matrix_reference", "matrix_cell_contract")}}
    receipt["artifacts"]["packed_reference"] = {
        "schema_version": 1,
        "summary_sha256": sha(summary),
        "assignment_sha256": sha(assignment),
        "candidate_sha256": sha(candidate_snapshot),
        "producer_receipt_sha256": producer_receipt_sha256,
        "consumer_implementation_sha256": deepcopy(consumer_identity),
    }
    return receipt


def pack(source_data: Path, output_data: Path, *, source_root: Path = None) -> dict:
    source_data = Path(source_data).resolve(strict=True)
    output_data = Path(output_data).resolve()
    if output_data.exists() or output_data == source_data or source_data in output_data.parents:
        raise ValueError("Compact output must be a new directory outside the source data tree")
    framework = Path(__file__).resolve().parents[1]
    source_root = Path(source_root or framework.parent).resolve(strict=True)
    image = json.loads((source_data / "runtime.json").read_text())["image"]
    for baseline, task_id in TASK_IDS.items():
        spec = load_task(framework / "local_tasks" / task_id)
        for split in ("train", "validation"):
            checked = LocalEvaluationMatrix(source_data / baseline / split / "qualified_matrix.json", spec)
            validate_matrix_halos(checked)
            validate_public_matrix(checked, baseline, split)
    output_data.mkdir(parents=True)
    for name in ("inputs", "validation_inputs"):
        shutil.copytree(source_data / name, output_data / name)
    shutil.copy2(source_data / "runtime.json", output_data / "runtime.json")
    total = 0
    for baseline, task_id in TASK_IDS.items():
        spec = load_task(framework / "local_tasks" / task_id)
        consumer = implementation_identity(spec, source_root, runtime_image=image)
        for split in ("train", "validation"):
            expected_count = len(coordinates(baseline, PUBLIC_SPLITS[split]))
            source_split = source_data / baseline / split
            output_split = output_data / baseline / split
            matrix_path = source_split / "qualified_matrix.json"
            matrix = LocalEvaluationMatrix(matrix_path, spec)
            if len(matrix.cells) != expected_count:
                raise ValueError("Unexpected qualified GNN matrix size")
            raw = json.loads(matrix_path.read_text())
            for row in raw["cells"]:
                source_run = (source_split / row["reference_run"]).resolve(strict=True)
                if source_data not in source_run.parents:
                    raise ValueError("Reference escapes the qualified data tree")
                result = _opt_snapshot(source_run, task_id)
                independent = _independent(source_run)
                destination = output_split / row["reference_run"]
                assignment_source = source_run / "20_partition/partition_assignment.csv"
                assignment_target = destination / "20_partition/partition_assignment.csv"
                assignment_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(assignment_source, assignment_target)
                summary = {
                    "schema_version": 1,
                    "task_id": task_id,
                    "wns": result["wns"],
                    "tns": result["tns"],
                    "partition_config": result["partition_config"],
                    "timing_sha256": {name: sha(source_run / "20_partition/timing_input" / name)
                                      for name in TIMING_FILES},
                    "assignment_sha256": sha(assignment_target),
                    "native_stage_sha256": {stage: sha(source_run / stage / "stage_manifest.json")
                                            for stage in ("20_partition", "30_placement", "40_pdn",
                                                          "50_hbt", "60_opt")},
                    "independent_partition_metrics": independent,
                }
                summary_path = destination / "reference_summary.json"
                _write(summary_path, summary)
                receipt_source = source_run.parent / "evaluation_result.json"
                receipt = json.loads(receipt_source.read_text())
                snapshot_target = destination.parent / "candidate_snapshot.json"
                shutil.copy2(source_run.parent / "candidate_snapshot.json", snapshot_target)
                receipt = packed_receipt(receipt, summary_path, assignment_target,
                    candidate_snapshot=snapshot_target, producer_receipt_sha256=sha(receipt_source),
                    consumer_identity=consumer)
                _write(destination.parent / "evaluation_result.json", receipt)
                total += 1
            shutil.copytree(source_split / "configs", output_split / "configs")
            shutil.copy2(matrix_path, output_split / "qualified_matrix.json")
            LocalEvaluationMatrix(output_split / "qualified_matrix.json", spec)
    return {"reference_cells": total, "output_data": str(output_data)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_data", type=Path, help="qualified task2 data root")
    parser.add_argument("output_data", type=Path, help="new compact task2 data root")
    parser.add_argument("--source-root", type=Path, help="matching frozen physical source")
    args = parser.parse_args()
    print(json.dumps(pack(args.source_data, args.output_data, source_root=args.source_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
