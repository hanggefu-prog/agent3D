#!/usr/bin/env python3
"""Check every initial task candidate against the selected source without EDA."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK))
from local_bench.evaluation import EVALUATOR_SOURCE_PATCHES
from local_bench.model import load_candidate, load_task
from local_bench.program_bridge import initial_program_text, materialize_program
from local_bench.prompting import build_problem_description
from local_bench.verifier import verify_candidate
from local_bench.workspace import _partition_patch


def check_patch(root, text):
    if not text.strip():
        return
    result = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=str(root), input=text, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise ValueError(result.stdout.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open3dbench-root", type=Path, default=FRAMEWORK.parent)
    args = parser.parse_args(argv)
    root = args.open3dbench_root.resolve()
    rows = []
    for path in sorted((FRAMEWORK / "local_tasks").glob("*/task.yaml")):
        spec = load_task(path.parent)
        row = {"task": spec.task_id}
        try:
            initial = load_candidate(spec.initial_candidate)
            program = initial_program_text(spec, initial)
            candidate = materialize_program(spec, program, root)
            verify_candidate(spec, candidate, root)
            prefix = "OpenROAD-3D/openroad/"
            outer, nested = _partition_patch(candidate.get("source_patch", ""), prefix)
            check_patch(root, outer)
            check_patch(root / "OpenROAD-3D/openroad", nested)
            trusted = EVALUATOR_SOURCE_PATCHES.get(spec.kind)
            if trusted:
                check_patch(root, trusted.read_text())
            prompt = build_problem_description(spec, initial, root)
            row.update(status="passed", program_chars=len(program), prompt_chars=len(prompt))
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        rows.append(row)
    passed = sum(r["status"] == "passed" for r in rows)
    print(json.dumps({"total": len(rows), "passed": passed, "tasks": rows,
                      "scope": "CPU source/contract preflight only; no EDA or model calls"}, indent=2))
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
