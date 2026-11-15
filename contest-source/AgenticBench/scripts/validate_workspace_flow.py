#!/usr/bin/env python3
"""Qualify one workspace evaluation interface without spending model/final budget.

Uses the production evaluator and immutable benchmark input. Each invocation is
a new, at-most-once reference qualification, not an experiment resume or retry.
The old failed reference and all historical artifacts remain untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic3d.knowledge import write_json
from global_bench.model import load_global_task
from scripts.codesign_runtime import CASES, Runtime
from scripts.run_workspace_codesign import TASKS
from scripts.workspace_codesign_evaluator import WorkspaceEvaluator


def qualify(runtime, case, objective, bundle, output):
    output = runtime.contained(output)
    bundle = runtime.contained(bundle)
    # mkdir is the exclusive attempt claim. Never overwrite even a failed check.
    output.mkdir(parents=True, exist_ok=False)
    task = load_global_task(runtime.framework / "global_tasks" / TASKS[objective])
    started = time.time()
    contract = {
        "kind": "workspace-flow-interface-qualification-v1", "case": case,
        "objective": objective, "source_commit": runtime.source_commit,
        "image": runtime.image, "benchmark_input": str(bundle),
        "task_digest": task.digest(), "model_calls": 0, "candidate_finals": 0,
        "policy": "Production evaluator; unchanged algorithm, objective, seeds and physical settings",
        "started_at": started,
    }
    write_json(output / "qualification.json", contract)
    try:
        result = WorkspaceEvaluator(runtime, case, objective, bundle, task).evaluate(
            runtime.source, output / "reference", None)
        success = result.get("success") is True
        summary = {**contract, "state": "passed" if success else "failed",
                   "result": result, "wall_seconds": time.time() - started}
    except Exception as error:
        success = False
        summary = {**contract, "state": "failed", "error": "{}: {}".format(type(error).__name__, error),
                   "wall_seconds": time.time() - started}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if success else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--benchmark-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--objective", choices=TASKS, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=48)
    parser.add_argument("--min-gpu-free-mb", type=int, default=2048)
    args = parser.parse_args(argv)
    runtime = Runtime(args.source, args.batch_root, args.image, args.gpu,
                      cpus=args.cpus, memory_gb=args.memory_gb, share_gpu=True,
                      min_gpu_free_mb=args.min_gpu_free_mb)
    return qualify(runtime, args.case, args.objective, args.benchmark_input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
