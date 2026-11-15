#!/usr/bin/env python3
"""Evaluate P0/P1/P2 HotSpot snapshots and attribute thermal improvement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


LEVELS = {
    "p0": ("placement", "30_placement/checkpoint.json"),
    "p1": ("cts", "70_cts/checkpoint.json"),
    "p2": ("postroute", "90_report/checkpoint.json"),
}


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: {}".format(path))
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _thermal_row(metrics: Mapping[str, Any]) -> Dict[str, float]:
    if metrics.get("status") != "success" or metrics.get("algorithm") != "hotspot-steady-grid":
        raise ValueError("thermal metrics failed their status/identity gate")
    ambient = float(metrics["ambient_k"])
    temperature = metrics.get("temperature")
    power = metrics.get("power")
    if not isinstance(temperature, Mapping) or not isinstance(power, Mapping):
        raise ValueError("thermal metrics have no temperature or power object")
    average = float(temperature["global_average_k"])
    maximum = float(temperature["global_maximum_k"])
    total_power = float(power["total_w"])
    values = (ambient, average, maximum, total_power)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("thermal metrics contain non-finite values")
    if average < ambient or maximum < average or total_power <= 0.0:
        raise ValueError("thermal metrics violate temperature/power invariants")
    return {
        "ambient_k": ambient,
        "average_k": average,
        "average_c": average - 273.15,
        "average_rise_k": average - ambient,
        "maximum_k": maximum,
        "maximum_c": maximum - 273.15,
        "maximum_rise_k": maximum - ambient,
        "total_power_w": total_power,
        "grid_size": float(metrics["grid_size"]),
    }


def compute_attribution(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return a matched-stage difference-in-differences decomposition."""

    ordered = [level for level in ("p0", "p1", "p2") if level in reference and level in candidate]
    if not ordered or set(reference) != set(candidate):
        raise ValueError("reference and candidate must contain the same non-empty levels")
    rows: Dict[str, Any] = {}
    improvements: Dict[str, Dict[str, float]] = {}
    signatures = set()
    for level in ordered:
        ref = _thermal_row(reference[level])
        cand = _thermal_row(candidate[level])
        if not math.isclose(ref["ambient_k"], cand["ambient_k"], abs_tol=1e-12):
            raise ValueError("reference and candidate ambient differ at {}".format(level))
        if ref["grid_size"] != cand["grid_size"]:
            raise ValueError("reference and candidate grids differ at {}".format(level))
        signatures.add((ref["ambient_k"], ref["grid_size"]))
        average_improvement = ref["average_rise_k"] - cand["average_rise_k"]
        maximum_improvement = ref["maximum_rise_k"] - cand["maximum_rise_k"]
        rows[level] = {"reference": ref, "candidate": cand}
        improvements[level] = {
            "average_temperature_improvement_k": average_improvement,
            "average_temperature_improvement_percent": (
                100.0 * average_improvement / ref["average_rise_k"]
                if ref["average_rise_k"] > 0.0
                else 0.0
            ),
            "maximum_temperature_improvement_k": maximum_improvement,
            "maximum_temperature_improvement_percent": (
                100.0 * maximum_improvement / ref["maximum_rise_k"]
                if ref["maximum_rise_k"] > 0.0
                else 0.0
            ),
        }
    if len(signatures) != 1:
        raise ValueError("thermal levels use different ambient/grid contracts")

    labels = {"p0": "partition_and_placement", "p1": "opt_and_cts", "p2": "routing_and_report"}
    contributions: Dict[str, Dict[str, float]] = {}
    previous_average = 0.0
    previous_maximum = 0.0
    for level in ordered:
        average = improvements[level]["average_temperature_improvement_k"]
        maximum = improvements[level]["maximum_temperature_improvement_k"]
        contributions[labels[level]] = {
            "average_temperature_k": average - previous_average,
            "maximum_temperature_k": maximum - previous_maximum,
        }
        previous_average = average
        previous_maximum = maximum

    def attach_shares(metric: str, final: float) -> None:
        absolute_total = sum(abs(row[metric]) for row in contributions.values())
        for row in contributions.values():
            row[metric.replace("_k", "_signed_share_percent")] = (
                100.0 * row[metric] / final if abs(final) > 1e-12 else 0.0
            )
            row[metric.replace("_k", "_absolute_share_percent")] = (
                100.0 * abs(row[metric]) / absolute_total if absolute_total > 1e-12 else 0.0
            )

    attach_shares("average_temperature_k", previous_average)
    attach_shares("maximum_temperature_k", previous_maximum)
    return {
        "schema_version": 1,
        "method": "matched-stage-temperature-rise-difference-in-differences",
        "positive_value_means": "candidate is cooler than reference",
        "levels": rows,
        "improvements": improvements,
        "incremental_contributions": contributions,
        "final_available_level": ordered[-1],
        "final_average_temperature_improvement_k": previous_average,
        "final_maximum_temperature_improvement_k": previous_maximum,
        "closure": {
            "average_k": sum(row["average_temperature_k"] for row in contributions.values()) - previous_average,
            "maximum_k": sum(row["maximum_temperature_k"] for row in contributions.values()) - previous_maximum,
        },
        "interpretation": (
            "P0 isolates the matched placement-level result. P1-P0 is the incremental "
            "effect after OPT and CTS. P2-P1 is the incremental routed/report effect. "
            "Signed shares may be negative or exceed 100% when a later stage cancels an "
            "earlier improvement; absolute shares describe magnitude, not direction."
        ),
    }


def _existing_or_run_snapshot(
    *,
    level: str,
    run_dir: Path,
    destination: Path,
    platform_dir: Path,
    hotspot_root: Path,
    openroad: str,
    grid_size: int,
    ambient_k: float,
    threads: int,
    resume: bool,
) -> Dict[str, Any]:
    if level == "p2":
        released = run_dir / "95_thermal/metrics.json"
        if released.is_file():
            return _read_json(released)
    metrics_path = destination / "metrics.json"
    if resume and metrics_path.is_file():
        return _read_json(metrics_path)
    if destination.exists():
        raise ValueError("snapshot output already exists: {}".format(destination))
    stage, checkpoint_rel = LEVELS[level]
    checkpoint = run_dir / checkpoint_rel
    if not checkpoint.is_file():
        raise FileNotFoundError("{} checkpoint: {}".format(stage, checkpoint))
    if level == "p2":
        from open3dbench.backend.thermal import run_thermal

        run_thermal(
            checkpoint,
            platform_dir,
            hotspot_root,
            destination,
            grid_size=grid_size,
            ambient_k=ambient_k,
        )
    else:
        from open3dbench.backend.thermal_snapshot import run_stage_snapshot

        run_stage_snapshot(
            checkpoint,
            platform_dir,
            hotspot_root,
            destination,
            openroad=openroad,
            grid_size=grid_size,
            ambient_k=ambient_k,
            threads=threads,
        )
    return _read_json(metrics_path)


def _infer_platform(root: Path, run_dir: Path) -> Path:
    for checkpoint in (
        run_dir / "30_placement/checkpoint.json",
        run_dir / "60_opt/checkpoint.json",
        run_dir / "70_cts/checkpoint.json",
        run_dir / "90_report/checkpoint.json",
    ):
        if checkpoint.is_file():
            pdk = str(_read_json(checkpoint).get("pdk", "")).lower()
            aliases = {
                "ng45": "nangate45_3D",
                "nangate45": "nangate45_3D",
                "nangate45_3d": "nangate45_3D",
                "asap7": "asap7_3D",
                "asap7_3d": "asap7_3D",
                "asap7-3d": "asap7_3D",
            }
            if pdk not in aliases:
                raise ValueError("cannot infer platform for PDK {}".format(pdk))
            return root / "OpenROAD-3D/flow/platforms" / aliases[pdk]
    raise FileNotFoundError("run has no supported physical checkpoint: {}".format(run_dir))


def run(args: argparse.Namespace) -> Path:
    root = args.open3dbench_root.expanduser().resolve()
    reference_run = args.reference_run.expanduser().resolve()
    candidate_run = args.candidate_run.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not root.is_dir() or not reference_run.is_dir() or not candidate_run.is_dir():
        raise FileNotFoundError("source root, reference run, and candidate run must exist")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    output.mkdir(parents=True, exist_ok=args.resume)
    platform = (
        args.platform_dir.expanduser().resolve()
        if args.platform_dir
        else _infer_platform(root, reference_run)
    )
    candidate_platform = _infer_platform(root, candidate_run)
    if platform.resolve() != candidate_platform.resolve():
        raise ValueError("reference and candidate use different platforms")
    levels = [part.strip().lower() for part in args.levels.split(",") if part.strip()]
    if not levels or any(level not in LEVELS for level in levels):
        raise ValueError("levels must be a non-empty subset of p0,p1,p2")
    valid_prefixes = (["p0"], ["p0", "p1"], ["p0", "p1", "p2"])
    if levels not in valid_prefixes:
        raise ValueError("levels must be the prefix p0, p0,p1, or p0,p1,p2")

    snapshots: Dict[str, Dict[str, Dict[str, Any]]] = {"reference": {}, "candidate": {}}
    for role, run_dir in (("reference", reference_run), ("candidate", candidate_run)):
        for level in levels:
            snapshots[role][level] = _existing_or_run_snapshot(
                level=level,
                run_dir=run_dir,
                destination=output / role / level,
                platform_dir=platform,
                hotspot_root=args.hotspot_root.expanduser().resolve(),
                openroad=args.openroad,
                grid_size=args.grid_size,
                ambient_k=args.ambient_k,
                threads=args.threads,
                resume=args.resume,
            )

    result = compute_attribution(snapshots["reference"], snapshots["candidate"])
    result.update(
        {
            "reference_run": str(reference_run),
            "candidate_run": str(candidate_run),
            "platform_dir": str(platform),
            "hotspot_root": str(args.hotspot_root.expanduser().resolve()),
            "levels_requested": levels,
        }
    )
    result_path = output / "thermal_attribution.json"
    _write_json(result_path, result)
    with (output / "thermal_attribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "component",
            "average_temperature_k",
            "average_temperature_signed_share_percent",
            "average_temperature_absolute_share_percent",
            "maximum_temperature_k",
            "maximum_temperature_signed_share_percent",
            "maximum_temperature_absolute_share_percent",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for component, row in result["incremental_contributions"].items():
            writer.writerow({"component": component, **row})
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open3dbench-root", required=True, type=Path)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--openroad", required=True)
    parser.add_argument("--hotspot-root", required=True, type=Path)
    parser.add_argument("--platform-dir", type=Path)
    parser.add_argument("--levels", default="p0,p1,p2")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--ambient-k", type=float, default=318.15)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = run(args)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
