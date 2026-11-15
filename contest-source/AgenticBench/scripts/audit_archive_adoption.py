#!/usr/bin/env python3
"""Independently read a frozen archive; never execute native tools or models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_bench.reference_archive import prepare_archive_reference
from local_bench.reference_store import file_sha, protect_destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--thermal-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["bp_fe", "bp", "ariane133", "swerv_wrapper"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Audit attempt already exists; preserve it and use a new attempt")
    protect_destination(args.output, {"reference_root": str(args.reference_root),
                                      "thermal_root": str(args.thermal_root)})
    args.output.mkdir(parents=True)
    summary = {"kind": "read-only-archive-adoption-audit", "started_at": datetime.now(timezone.utc).isoformat(),
               "new_native_attempts": 0, "model_calls": 0, "candidate_finals": 0, "endpoints": []}
    for case in args.cases:
        for endpoint in ("placement", "timing", "grt", "cts_thermal"):
            started = time.monotonic()
            row = {"case": case, "endpoint": endpoint}
            try:
                value = prepare_archive_reference(args.reference_root, case, endpoint,
                                                  thermal_root=args.thermal_root)
                artifact = args.output / (case + "-" + endpoint + ".json")
                with artifact.open("x") as stream:
                    json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write("\n")
                row.update(status="qualified", evidence_count=len(value["evidence"]),
                           raw_metrics=value["baseline"]["raw_metrics"], qualification_sha256=file_sha(artifact))
            except Exception as error:
                row.update(status="rejected", error=type(error).__name__ + ": " + str(error))
            row["verification_wall_seconds"] = time.monotonic() - started
            summary["endpoints"].append(row)
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "summary.json").open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if all(row["status"] == "qualified" for row in summary["endpoints"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
