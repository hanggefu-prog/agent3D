#!/usr/bin/env python3
"""Collect common OpenROAD placement metrics and artifact checks."""

import argparse
import json
import re
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
PATTERNS = {
    "hpwl": [rf"(?:HPWL|half-perimeter wirelength)[^\d+-]*({NUMBER})"],
    "wirelength": [rf"(?:total wirelength|wire length)[^\d+-]*({NUMBER})"],
    "overflow": [rf"(?:density )?overflow[^\d+-]*({NUMBER})"],
    "utilization": [rf"utilization[^\d+-]*({NUMBER})"],
    "wns": [rf"\bWNS\b[^\d+-]*({NUMBER})", rf"worst slack[^\d+-]*({NUMBER})"],
    "tns": [rf"\bTNS\b[^\d+-]*({NUMBER})", rf"total negative slack[^\d+-]*({NUMBER})"],
}


def last_match(patterns, text):
    found = []
    for pattern in patterns:
        found.extend(float(value) for value in re.findall(pattern, text, re.IGNORECASE))
    return found[-1] if found else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--require-file", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in args.logs)
    metrics = {name: last_match(patterns, text) for name, patterns in PATTERNS.items()}
    checks = {
        "logs_nonempty": bool(text.strip()),
        "required_files_exist": all(path.is_file() and path.stat().st_size > 0 for path in args.require_file),
        "no_fatal_marker": not bool(re.search(r"\b(?:FATAL|segmentation fault)\b", text, re.IGNORECASE)),
        "no_openroad_error": not bool(re.search(r"^Error:", text, re.IGNORECASE | re.MULTILINE)),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "logs": [str(path.resolve()) for path in args.logs],
        "required_files": [str(path.resolve()) for path in args.require_file],
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
