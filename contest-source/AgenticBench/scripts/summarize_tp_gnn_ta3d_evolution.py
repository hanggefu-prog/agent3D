#!/usr/bin/env python3
"""Summarize per-iteration TA3D TP-GNN OpenEvolve results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)$")
DEFAULT_CASES = ("bp", "bp_multi", "ariane133", "swerv_wrapper")


def _checkpoint_number(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else -1


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _programs(case_root: Path) -> Iterable[Dict[str, Any]]:
    seen = set()
    checkpoints = case_root.glob("openevolve/checkpoints/checkpoint_*")
    for checkpoint in sorted(checkpoints, key=_checkpoint_number):
        for path in sorted((checkpoint / "programs").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            program_id = str(data.get("id", path.stem))
            if program_id in seen:
                continue
            seen.add(program_id)
            yield data


def summarize_case(case: str, case_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for program in _programs(case_root):
        metrics = program.get("metrics", {})
        valid = _finite(metrics.get("runs_successfully")) == 1.0
        rows.append(
            {
                "case": case,
                "iteration": int(program.get("iteration_found", -1)),
                "generation": int(program.get("generation", -1)),
                "program_id": str(program.get("id", "")),
                "parent_id": program.get("parent_id"),
                "valid": valid,
                "combined_score": _finite(metrics.get("combined_score")),
                "ta3d_signed_cut_score": (
                    _finite(metrics.get("ta3d_signed_cut_score")) if valid else None
                ),
                "ta3d_signed_cut_gain": (
                    _finite(metrics.get("ta3d_signed_cut_gain")) if valid else None
                ),
                "critical_cut_reward": (
                    _finite(metrics.get("critical_cut_reward")) if valid else None
                ),
                "noncritical_cut_penalty": (
                    _finite(metrics.get("noncritical_cut_penalty")) if valid else None
                ),
            }
        )
    rows.sort(key=lambda row: (row["iteration"], row["program_id"]))
    best = -math.inf
    for row in rows:
        best = max(best, row["combined_score"])
        row["best_so_far"] = best if math.isfinite(best) else 0.0
    best_row = max(rows, key=lambda row: row["combined_score"], default=None)
    return {
        "case": case,
        "iterations_observed": max((row["iteration"] for row in rows), default=-1),
        "program_count": len(rows),
        "valid_program_count": sum(bool(row["valid"]) for row in rows),
        "best": best_row,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    root = args.run_root.expanduser().resolve()
    summary = {
        "schema_version": 1,
        "run_root": str(root),
        "cases": [summarize_case(case, root / case) for case in args.cases],
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "case",
            "iteration",
            "generation",
            "program_id",
            "parent_id",
            "valid",
            "combined_score",
            "best_so_far",
            "ta3d_signed_cut_score",
            "ta3d_signed_cut_gain",
            "critical_cut_reward",
            "noncritical_cut_penalty",
        ]
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for case_summary in summary["cases"]:
                writer.writerows(case_summary["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
