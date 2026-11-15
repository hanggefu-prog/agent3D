#!/usr/bin/env python3
from __future__ import annotations
"""Validate the structural and area invariants of a partition artifact."""

import argparse
import csv
import json
import math
from pathlib import Path


def _legacy_json(path: Path) -> tuple[set[str], list[str], list[str], float, float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        set(map(str, data.get("all_macros", []))),
        list(map(str, data.get("upper_die_macros", []))),
        list(map(str, data.get("bottom_die_macros", []))),
        float(data.get("upper_die_area", math.nan)),
        float(data.get("bottom_die_area", math.nan)),
        float(data.get("area_balance_error", 0.1)),
    )


def _unified_csv(
    path: Path, metrics_path: Path | None, expected_path: Path | None, tolerance: float
) -> tuple[set[str], list[str], list[str], float, float, float]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"inst_name", "die"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("assignment CSV requires inst_name,die columns")
        upper: list[str] = []
        bottom: list[str] = []
        for row in reader:
            name = row["inst_name"].strip()
            die = row["die"].strip().lower()
            if not name:
                raise ValueError("assignment contains an empty instance name")
            if die == "top":
                upper.append(name)
            elif die == "bottom":
                bottom.append(name)
            else:
                raise ValueError(f"invalid die {die!r} for {name}")
    expected = set(upper) | set(bottom)
    if expected_path is not None:
        expected = {
            line.strip()
            for line in expected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    upper_area = math.nan
    bottom_area = math.nan
    if metrics_path is not None:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        upper_area = float(
            metrics.get("top_area", metrics.get("upper_die_area", math.nan))
        )
        bottom_area = float(metrics.get("bottom_area", math.nan))
        tolerance = float(metrics.get("area_balance_error", tolerance))
    return expected, upper, bottom, upper_area, bottom_area, tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--expected-instances", type=Path)
    parser.add_argument("--max-area-imbalance", type=float, default=0.1)
    args = parser.parse_args()
    try:
        if args.artifact.suffix.lower() == ".csv":
            (
                all_macros,
                upper_list,
                bottom_list,
                upper_area,
                bottom_area,
                tolerance,
            ) = _unified_csv(
                args.artifact,
                args.metrics,
                args.expected_instances,
                args.max_area_imbalance,
            )
        else:
            (
                all_macros,
                upper_list,
                bottom_list,
                upper_area,
                bottom_area,
                tolerance,
            ) = _legacy_json(args.artifact)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = {"passed": False, "checks": {}, "error": str(exc)}
        encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        return 1

    upper = set(upper_list)
    bottom = set(bottom_list)
    total_area = upper_area + bottom_area
    imbalance = abs(upper_area - bottom_area) / total_area if total_area > 0 else math.inf
    area_available = all(
        math.isfinite(value) and value > 0 for value in (upper_area, bottom_area)
    )

    checks = {
        "sets_non_empty": bool(upper and bottom),
        "no_duplicate_upper": len(upper) == len(upper_list),
        "no_duplicate_bottom": len(bottom) == len(bottom_list),
        "sets_disjoint": not bool(upper & bottom),
        "union_matches_all_macros": bool(all_macros) and upper | bottom == all_macros,
        "areas_finite_positive": area_available,
        "area_balance_within_tolerance": area_available and imbalance <= tolerance,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "upper_macro_count": len(upper),
            "bottom_macro_count": len(bottom),
            "normalized_area_imbalance": imbalance,
            "area_balance_error": tolerance,
        },
        "unexpected_overlap": sorted(upper & bottom),
        "missing_macros": sorted(all_macros - (upper | bottom)),
        "unknown_macros": sorted((upper | bottom) - all_macros),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
