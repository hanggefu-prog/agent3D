#!/usr/bin/env python3
"""One scheduled preparation or Codex objective job; no alternate evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from scripts.codesign_runtime import CASES, Runtime
from scripts.run_codesign_experiment import immutable_json
from scripts.run_workspace_codesign import run_workspace_experiment

TASKS = {"timing": "lol_partition_placement_timing_one_pass",
         "thermal": "lol_partition_placement_cts_thermal_one_pass"}
MODEL = "gpt-5.6-sol"
WORKSPACE_QUALIFICATION = "workspace-terminal-and-resume-v1"
REQUIRED_PREFLIGHT_CHECKS = frozenset({
    "trusted_container_isolation", "initial_turn", "resumed_turn", "same_session",
    "workspace_edit", "protected_unchanged", "input_unchanged",
    "host_canary_unchanged", "git_config_unchanged", "second_final",
})


def task_contract(framework, objective):
    task_file = framework / "global_tasks" / TASKS[objective] / "task.yaml"
    task = yaml.safe_load(task_file.read_text())
    budget = task["budget"]
    if budget["global_rounds"] != budget["expensive_evaluations"]:
        raise ValueError("This single-agent adapter requires one final per global round")
    return {"task_id": task["id"], "task_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
            "rounds": budget["expensive_evaluations"], "objective": task["objective"],
            "seed_policy": task["seed_policy"], "adapter": "codex-container-workspace-v1",
            "reference_policy": "Explicit one-pass ADOPT (default), FRESH or supported EXTEND; exact source/input qualification and no fallback"}


def check_model_preflight(path, image):
    """A plain API response is insufficient to qualify workspace execution."""
    raw = path.read_bytes()
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Workspace preflight receipt must be an object")
    if (result.get("success") is not True or type(result.get("exit_code")) is not int
            or result.get("exit_code") != 0
            or result.get("timed_out") is not False
            or not isinstance(result.get("final_message"), str)
            or result["final_message"].strip() != "OK"):
        raise ValueError("Model preflight did not succeed; no physical experiment may start")
    if result.get("model") != MODEL:
        raise ValueError("Model preflight used a different model")
    if result.get("qualification") != WORKSPACE_QUALIFICATION:
        raise ValueError("A qualified workspace terminal and resumed session are required")
    if result.get("image") != image:
        raise ValueError("Workspace preflight used a different image")
    checks = result.get("checks")
    if (not isinstance(checks, dict) or any(checks.get(name) is not True
                                          for name in REQUIRED_PREFLIGHT_CHECKS)
            or any(value is not True for value in checks.values())):
        raise ValueError("Workspace preflight did not verify every required boundary and session check")
    if result.get("eda_executed") is not False:
        raise ValueError("Workspace preflight must be a tooling-only qualification")
    return {"result": str(path), "result_sha256": hashlib.sha256(raw).hexdigest(),
            "qualification": WORKSPACE_QUALIFICATION, "image": image, "model": MODEL}


def bind_task_contracts(path, value, phase):
    """Preparation publishes once; concurrent objective jobs only read it."""
    if phase == "prepare":
        immutable_json(path, value)
        return
    if phase not in TASKS:
        raise ValueError("Unknown scheduled objective")
    if not path.is_file():
        raise ValueError("Preparation task contract is missing; objective jobs cannot create it")
    if json.loads(path.read_text()) != value:
        raise ValueError("Preparation task contract differs; objective jobs cannot rewrite it")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "timing", "thermal"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=48)
    parser.add_argument("--min-gpu-free-mb", type=int, default=2048)
    parser.add_argument("--codex-bin", required=True)
    parser.add_argument("--preflight-result", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path,
                        help="Read-only 2026-09-23 archive; required for prepare, never modified")
    parser.add_argument("--reference-policy", choices=("ADOPT", "FRESH", "EXTEND"), default="ADOPT")
    parser.add_argument("--reference-store", type=Path,
                        help="Append-only qualified reference store; default: batch-root/reference-store")
    parser.add_argument("--reference-record", type=Path,
                        help="Exact sealed ADOPT record or EXTEND parent; no automatic fallback")
    parser.add_argument("--retention", choices=("full", "best"), default="best",
                        help="Large-scale default: best; use full for complete debugging artifacts")
    args = parser.parse_args()
    preflight = check_model_preflight(args.preflight_result.resolve(), args.image)
    source, batch = args.source.resolve(), args.batch_root.resolve()
    runtime = Runtime(source, batch, args.image, args.gpu, args.cpus, args.memory_gb,
                      share_gpu=True, min_gpu_free_mb=args.min_gpu_free_mb)
    runtime.retention_mode = args.retention
    runtime.reference_baseline = "reference_20260923"
    runtime.reference_root = args.reference_root.resolve() if args.reference_root else None
    runtime.reference_policy = args.reference_policy
    runtime.reference_store = (args.reference_store.expanduser().resolve() if args.reference_store
                               else batch / "reference-store")
    runtime.reference_record = args.reference_record.expanduser().absolute() if args.reference_record else None
    contracts = {o: task_contract(runtime.framework, o) for o in TASKS}
    bind_task_contracts(batch / args.case / "task_contracts.json", {
        "source_commit": runtime.source_commit, "image": args.image,
        "case": args.case, "methods": ["codex"], "tasks": contracts,
        "model": MODEL, "model_preflight": preflight,
        "input_policy": "Pinned 2026-09-23 physical input adoption; original native provenance retained"}, args.phase)
    if args.phase == "prepare":
        runtime.prepare(args.case, batch / args.case / "input")
        print(json.dumps({"event": "input-qualified", "case": args.case}), flush=True)
        return
    rounds = contracts[args.phase]["rounds"]
    root = batch / args.case / args.phase / "codex"
    run_workspace_experiment(runtime, args.case, args.phase, batch / args.case / "input",
                             root, args.codex_bin)
    status = json.loads((root / "status.json").read_text())
    if status.get("state") != "completed":
        raise RuntimeError("Objective job needs inspection; see " + str(root / "status.json"))
    summary = json.loads((root / "summary.json").read_text())
    if (summary.get("state") != "completed" or summary.get("final_evaluations") != rounds
            or summary.get("objective") != args.phase or summary.get("case") != args.case
            or type(summary.get("reference_evaluations")) is not int
            or summary["reference_evaluations"] != (0 if args.reference_policy == "ADOPT" else 1)
            or type(summary.get("reference_adoptions")) is not int
            or summary["reference_adoptions"] != int(args.reference_policy == "ADOPT")):
        raise ValueError("Completed driver does not match the fixed task budget/endpoint")
    print(json.dumps({"event": "objective-complete", "case": args.case,
                      "objective": args.phase, "summary": str(root / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
