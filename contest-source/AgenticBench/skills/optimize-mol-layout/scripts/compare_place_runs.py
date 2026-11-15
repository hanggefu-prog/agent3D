#!/usr/bin/env python3
"""Compare a Place-MoL baseline with candidate HPWL/density results."""

import argparse
import json
from pathlib import Path


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    density = metrics.get("density_overflow")
    if density is None:
        density = metrics.get("max_density")
    return data, float(metrics["objective_hpwl"]), float(density)


def pct(new: float, old: float) -> float:
    return (old - new) / old * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--max-hpwl-regression-pct", type=float, default=0.0)
    parser.add_argument("--max-density-regression-pct", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline_data, baseline_hpwl, baseline_density = load(args.baseline)
    rows = []
    for path in args.candidates:
        data, hpwl, density = load(path)
        hpwl_improvement = pct(hpwl, baseline_hpwl)
        density_improvement = pct(density, baseline_density)
        accepted = (
            data.get("passed", False)
            and hpwl_improvement >= -args.max_hpwl_regression_pct
            and density_improvement >= -args.max_density_regression_pct
            and (hpwl_improvement > 0 or density_improvement > 0)
        )
        rows.append({
            "path": str(path.resolve()),
            "passed_correctness": bool(data.get("passed", False)),
            "hpwl": hpwl,
            "density": density,
            "hpwl_improvement_pct": hpwl_improvement,
            "density_improvement_pct": density_improvement,
            "accepted": accepted,
            "balanced_score": hpwl_improvement + density_improvement,
        })
    accepted_rows = [row for row in rows if row["accepted"]]
    accepted_rows.sort(key=lambda row: row["balanced_score"], reverse=True)
    result = {
        "baseline": {
            "path": str(args.baseline.resolve()),
            "passed_correctness": bool(baseline_data.get("passed", False)),
            "hpwl": baseline_hpwl,
            "density": baseline_density,
        },
        "candidates": rows,
        "recommended": accepted_rows[0]["path"] if accepted_rows else None,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if accepted_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
