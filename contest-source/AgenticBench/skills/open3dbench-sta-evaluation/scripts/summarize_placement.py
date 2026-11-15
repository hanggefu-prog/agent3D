#!/usr/bin/env python3
"""Collect compact two-pass timing-promotion outputs into JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CASES = ("bp", "bp_multi", "ariane133", "swerv_wrapper")
METHODS = ("dynamic", "source")
FIELDS = (
    "case",
    "method",
    "attempt",
    "status",
    "program_id",
    "placement_native_hpwl",
    "placement_overflow_max",
    "initial_global_wns_setup_ns",
    "initial_global_tns_setup_ns",
    "pass1_global_wns_setup_ns",
    "pass1_global_tns_setup_ns",
    "pass2_global_wns_setup_ns",
    "pass2_global_tns_setup_ns",
    "final_global_wns_setup_ns",
    "final_global_tns_setup_ns",
    "hbt_count",
    "opt_passes",
    "opt_runtime_seconds",
    "sta_graph_complete",
    "sta_unconstrained_endpoints",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    prefix = (args.output_prefix or root / "summary").expanduser().resolve()

    rows = []
    for case in CASES:
        for method in METHODS:
            base = root / case / method
            provenance_path = base / "candidate" / "promotion_provenance.json"
            row = {key: None for key in FIELDS}
            row.update({"case": case, "method": method, "status": "missing"})
            if provenance_path.is_file():
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                row["program_id"] = provenance.get("program_id")
            attempts = (base, root / case / (method + "_retry_v2"))
            selected = None
            for attempt in attempts:
                result_path = attempt / "evaluation" / "evaluation_result.json"
                if not result_path.is_file():
                    continue
                result = json.loads(result_path.read_text(encoding="utf-8"))
                selected = (attempt, result)
                if result.get("metrics", {}).get("runs_successfully"):
                    break
            if selected is not None:
                attempt, result = selected
                metrics = result.get("metrics", {})
                row["attempt"] = attempt.name
                row["status"] = "success" if metrics.get("runs_successfully") else "failed"
                for field in FIELDS:
                    if field in metrics:
                        row[field] = metrics[field]
            rows.append(row)

    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
