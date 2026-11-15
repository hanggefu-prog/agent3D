"""Build reproducible placement-evolution convergence artifacts from history.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METRICS = (
    ("placement_native_hpwl", "DREAMPlace weighted HPWL"),
    ("placement_overflow_max", "Density overflow"),
    ("placement_max_density", "Maximum density"),
)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_convergence(history: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    incumbent: Optional[Dict[str, Any]] = None
    with history.expanduser().resolve().open(encoding="utf-8") as stream:
        for fallback_order, line in enumerate(stream, 1):
            raw = json.loads(line)
            score = _finite(raw.get("score"))
            metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
            row: Dict[str, Any] = {
                "order": int(raw.get("order", fallback_order)),
                "status": str(raw.get("status", "unknown")),
                "score": score,
            }
            for key, _label in METRICS:
                value = metrics.get(key)
                if key == "placement_native_hpwl" and value is None:
                    # Read historical runs generated before the placement-only
                    # boundary received its own metric name.
                    value = metrics.get("total_object_origin_hpwl_um")
                row[key] = _finite(value)
            if score is not None and (incumbent is None or score > incumbent["score"]):
                incumbent = dict(row)
            row["best_score"] = incumbent["score"] if incumbent else None
            for key, _label in METRICS:
                row["best_" + key] = incumbent.get(key) if incumbent else None
            rows.append(row)
    if not rows:
        raise ValueError("evolution history is empty: {}".format(history))
    return rows


def write_csv(rows: Iterable[Dict[str, Any]], output: Path) -> None:
    fields = ["order", "status", "score"]
    fields += [key for key, _label in METRICS]
    fields += ["best_score"] + ["best_" + key for key, _label in METRICS]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _polyline(points: List[tuple[float, float]], color: str, width: float, opacity: float) -> str:
    if not points:
        return ""
    coordinates = " ".join("{:.2f},{:.2f}".format(x, y) for x, y in points)
    return (
        '<polyline points="{}" fill="none" stroke="{}" stroke-width="{}" '
        'stroke-opacity="{}" stroke-linejoin="round" stroke-linecap="round"/>'.format(
            coordinates, color, width, opacity
        )
    )


def write_svg(rows: List[Dict[str, Any]], output: Path, title: str) -> None:
    width, panel_height = 1080, 240
    margin_left, margin_right, margin_top, margin_bottom = 105, 35, 42, 42
    plot_width = width - margin_left - margin_right
    total_height = 58 + panel_height * len(METRICS)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, total_height, width, total_height
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="600">{}</text>'.format(
            width / 2, title.replace("&", "&amp;").replace("<", "&lt;")
        ),
    ]
    maximum_order = max(1, max(int(row["order"]) for row in rows))
    for panel_index, (key, label) in enumerate(METRICS):
        top = 48 + panel_index * panel_height
        plot_top = top + margin_top
        plot_bottom = top + panel_height - margin_bottom
        values = [row[key] for row in rows if row[key] is not None]
        best_values = [row["best_" + key] for row in rows if row["best_" + key] is not None]
        all_values = values + best_values
        if not all_values:
            lower, upper = 0.0, 1.0
        else:
            lower, upper = min(all_values), max(all_values)
            padding = (upper - lower) * 0.08 if upper > lower else max(abs(upper) * 0.05, 1.0e-6)
            lower, upper = lower - padding, upper + padding

        def xy(order: int, value: float) -> tuple[float, float]:
            x = margin_left + (order - 1) * plot_width / max(1, maximum_order - 1)
            y = plot_bottom - (value - lower) * (plot_bottom - plot_top) / (upper - lower)
            return x, y

        parts.extend([
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#cbd5e1"/>'.format(
                margin_left, plot_bottom, width - margin_right, plot_bottom
            ),
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#cbd5e1"/>'.format(
                margin_left, plot_top, margin_left, plot_bottom
            ),
            '<text x="{}" y="{}" font-family="sans-serif" font-size="14" font-weight="600">{}</text>'.format(
                margin_left, top + 24, label
            ),
            '<text x="{}" y="{}" text-anchor="end" font-family="monospace" font-size="11">{:.6g}</text>'.format(
                margin_left - 8, plot_top + 4, upper
            ),
            '<text x="{}" y="{}" text-anchor="end" font-family="monospace" font-size="11">{:.6g}</text>'.format(
                margin_left - 8, plot_bottom + 4, lower
            ),
        ])
        raw_points = [xy(int(row["order"]), row[key]) for row in rows if row[key] is not None]
        best_points = [
            xy(int(row["order"]), row["best_" + key])
            for row in rows if row["best_" + key] is not None
        ]
        parts.append(_polyline(raw_points, "#94a3b8", 1.5, 0.75))
        parts.append(_polyline(best_points, "#0f766e", 3.0, 1.0))
        for x, y in raw_points:
            parts.append('<circle cx="{:.2f}" cy="{:.2f}" r="2.6" fill="#64748b"/>'.format(x, y))
        parts.extend([
            '<text x="{}" y="{}" font-family="sans-serif" font-size="11">1</text>'.format(
                margin_left, plot_bottom + 20
            ),
            '<text x="{}" y="{}" text-anchor="end" font-family="sans-serif" font-size="11">{}</text>'.format(
                width - margin_right, plot_bottom + 20, maximum_order
            ),
            '<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12">Evolution sample</text>'.format(
                margin_left + plot_width / 2, plot_bottom + 34
            ),
        ])
    parts.extend([
        '<line x1="760" y1="31" x2="790" y2="31" stroke="#94a3b8" stroke-width="2"/>',
        '<text x="798" y="35" font-family="sans-serif" font-size="12">evaluated</text>',
        '<line x1="875" y1="31" x2="905" y2="31" stroke="#0f766e" stroke-width="3"/>',
        '<text x="913" y="35" font-family="sans-serif" font-size="12">incumbent</text>',
        '</svg>',
    ])
    output.write_text("\n".join(part for part in parts if part) + "\n", encoding="utf-8")


def build_artifacts(run_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    run = run_dir.expanduser().resolve()
    history = run / "history.jsonl"
    output = (output_dir or run).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_convergence(history)
    csv_path = output / "convergence.csv"
    svg_path = output / "convergence.svg"
    json_path = output / "convergence.json"
    write_csv(rows, csv_path)
    summary_path = run / "summary.json"
    task = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            task = summary.get("task") if isinstance(summary, dict) else None
        except (OSError, ValueError, TypeError):
            task = None
    title = "{} — placement evolution convergence".format(task) if task else "Placement evolution convergence"
    write_svg(rows, svg_path, title)
    valid = [row for row in rows if row.get("score") is not None]
    report = {
        "schema_version": 1,
        "history": str(history),
        "samples": len(rows),
        "valid_samples": len(valid),
        "best": max(valid, key=lambda row: row["score"]) if valid else None,
        "csv": str(csv_path),
        "svg": str(svg_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="openagentic3dbench-convergence")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_artifacts(args.run_dir, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
