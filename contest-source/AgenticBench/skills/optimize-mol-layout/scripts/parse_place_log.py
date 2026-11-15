#!/usr/bin/env python3
"""Extract stable Place-MoL/DREAMPlace metrics from one or more logs."""

import argparse
import json
import math
import re
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
SUMMARY_PATTERNS = {
    "initial_2d_hpwl": rf"Initial 2D HPWL:\s*({NUMBER})",
    "bottom_refined_hpwl": rf"After Bottom-Die Refinement:\s*({NUMBER})",
    "bottom_legalized_hpwl": rf"After Bottom-Die Legalization:\s*({NUMBER})",
    "upper_legalized_hpwl": rf"After Upper-Die Legalization:\s*({NUMBER})",
    "objective_hpwl": rf"Mem-on-logic HPWL before legalization:\s*({NUMBER})",
    "legalized_hpwl": rf"Mem-on-logic HPWL after legalization:\s*({NUMBER})",
    "density_overflow": rf"Mem-on-logic density overflow:\s*({NUMBER})",
    "max_density": rf"Mem-on-logic max density:\s*({NUMBER})",
    "runtime_seconds": rf"Total Runtime:\s*({NUMBER})\s*seconds",
}


def last_number(pattern: str, text: str):
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--require-density", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in args.logs)
    metrics = {name: last_number(pattern, text) for name, pattern in SUMMARY_PATTERNS.items()}

    # Fall back to the final DREAMPlace EvalMetrics line when the pipeline
    # summary predates Agentic3D's explicit density reporting.
    eval_lines = re.findall(rf"[^\n]*wHPWL\s+({NUMBER})[^\n]*", text, flags=re.IGNORECASE)
    if metrics["objective_hpwl"] is None and eval_lines:
        metrics["objective_hpwl"] = float(eval_lines[-1])
    if metrics["density_overflow"] is None:
        metrics["density_overflow"] = last_number(rf"Overflow\s+({NUMBER})", text)
    if metrics["max_density"] is None:
        metrics["max_density"] = last_number(rf"MaxDensity\s+({NUMBER})", text)

    hpwl = metrics["objective_hpwl"]
    checks = {
        "pipeline_completed": "Pipeline Summary" in text and "Saved result to CSV" in text,
        "finite_positive_hpwl": hpwl is not None and math.isfinite(hpwl) and hpwl > 0,
        "no_traceback": "Traceback (most recent call last)" not in text,
        "no_fatal_marker": not bool(re.search(r"\b(?:FATAL|segmentation fault)\b", text, re.IGNORECASE)),
        "density_present": (
            not args.require_density
            or (metrics["density_overflow"] is not None and metrics["max_density"] is not None)
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "logs": [str(path.resolve()) for path in args.logs],
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
