#!/usr/bin/env python3
"""Decode one saved placement program into the common candidate JSON contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from local_bench.model import load_task
from local_bench.program_bridge import materialize_program
from local_bench.verifier import verify_candidate


TASKS = {
    "dynamic": "placement_parameters",
    "source": "placement_init_preconditioner",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--open3dbench-root", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(TASKS), required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--program-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    framework = args.framework_root.expanduser().resolve()
    source = args.open3dbench_root.expanduser().resolve()
    program_path = args.program.expanduser().resolve()
    info_path = args.program_info.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    spec = load_task(framework / "local_tasks" / TASKS[args.method])

    program = program_path.read_text(encoding="utf-8")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    candidate = materialize_program(spec, program, source)
    verify_candidate(spec, candidate)

    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "method": args.method,
        "source_task": spec.task_id,
        "program": str(program_path),
        "program_sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(),
        "program_info": str(info_path),
        "program_id": info.get("id"),
        "placement_metrics": info.get("metrics", {}),
    }
    (output / "promotion_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
