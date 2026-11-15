#!/usr/bin/env python3
"""Select the exact materialized candidate that produced a reported best result."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from local_bench.model import load_task
from local_bench.verifier import verify_candidate


SOURCE_TASKS = {
    "dynamic": "placement_parameters",
    "source": "placement_init_preconditioner",
}


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as error:
        raise ValueError("{} escapes the selected case root: {}".format(label, resolved)) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(SOURCE_TASKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    framework = args.framework_root.expanduser().resolve()
    case_root = args.case_root.expanduser().resolve()
    comparison_path = case_root / "placement_task_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    rows = [row for row in comparison.get("rows", []) if row.get("task") == args.method]
    if len(rows) != 1:
        raise ValueError("comparison must contain exactly one {} row".format(args.method))
    row = rows[0]

    best_result = _inside(Path(str(row["best_result"])), case_root, "best_result")
    result = json.loads(best_result.read_text(encoding="utf-8"))
    metrics = result.get("metrics", {})
    for row_key, metric_key in (
        ("best_fitness", "combined_score"),
        ("best_hpwl", "placement_native_hpwl"),
        ("best_overflow_max", "placement_overflow_max"),
    ):
        expected = float(row[row_key])
        actual = float(metrics[metric_key])
        if abs(actual - expected) > max(1.0e-12, abs(expected) * 1.0e-12):
            raise ValueError(
                "comparison {} does not match best evaluation {}: {} != {}".format(
                    row_key, metric_key, expected, actual
                )
            )
    candidate_dir = _inside(
        Path(str(result.get("artifacts", {}).get("candidate", ""))),
        case_root,
        "candidate",
    )
    candidate_path = candidate_dir / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    source_spec = load_task(framework / "local_tasks" / SOURCE_TASKS[args.method])
    promotion_spec = load_task(
        framework / "evaluation_profiles" / "placement_two_pass"
    )
    source_verification = verify_candidate(source_spec, candidate)
    promotion_verification = verify_candidate(promotion_spec, candidate)

    best_info_path = case_root / args.method / "openevolve" / "best" / "best_program_info.json"
    best_program_path = case_root / args.method / "openevolve" / "best" / "best_program.py"
    best_info = json.loads(best_info_path.read_text(encoding="utf-8"))
    best_program = best_program_path.read_text(encoding="utf-8")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, output / "candidate.json")
    provenance = {
        "schema_version": 1,
        "method": args.method,
        "source_task": source_spec.task_id,
        "comparison": str(comparison_path),
        "best_result": str(best_result),
        "source_candidate": str(candidate_path),
        "program": str(best_program_path),
        "program_sha256": hashlib.sha256(best_program.encode("utf-8")).hexdigest(),
        "program_info": str(best_info_path),
        "program_id": best_info.get("id"),
        "placement_metrics": metrics,
        "source_verification": source_verification["patch"],
        "promotion_verification": promotion_verification["patch"],
    }
    (output / "promotion_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
