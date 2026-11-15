#!/usr/bin/env python3
"""Host-only, at-most-once identity/probe GRT qualification pair.

The existing bounded image builder owns compilation and runtime receipts. This
launcher holds one shared task lock for the pair, admits live resources before
each build/run, and stops on the first failure or uncertain result. No models,
search workers, archive writes, GPU allocations, or cleanup are performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK))
sys.path.insert(0, str(FRAMEWORK.parent))

from local_bench.reference_store import protect_destination
from scripts.build_routing_candidate_image import build_candidate_image
from scripts.qualify_archive_grt import (materialize, qualify_inputs, task_contract,
                                       validate_iteration_probe_effect, TASK, TASKS, PREBUILT_OPENROAD_ENV)

GIB = 1024**3
BASE_IMAGE = "sha256:f7d91b5248204b6fa11f55e7fd3bce8fe95d455d84fdca3bd582f4859ceda0db"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def guard(root):
    memory = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith("MemAvailable:"))
    result = {"available_ram_bytes": memory, "data_free_bytes": shutil.disk_usage(root / "runs").free,
              "root_free_bytes": shutil.disk_usage("/").free}
    result["passed"] = (memory >= 64 * GIB and result["data_free_bytes"] >= 64 * GIB
                        and result["root_free_bytes"] >= 10 * GIB)
    return result


@contextmanager
def slot(root, output):
    locks = root / "runs/locks"
    if any(path.is_symlink() for path in (locks, *locks.parents)):
        raise ValueError("Shared lock directory must not traverse symlinks")
    locks.mkdir(parents=True, exist_ok=True)
    path = locks / "cpu-grt-qualification.lock"
    if path.is_symlink():
        raise ValueError("Shared lock must not be a symlink")
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield str(path)


def admit(root, output, phase):
    """Waiting for headroom is not a retry of a failed native attempt."""
    while True:
        observed = guard(root)
        write(output / "admission.json", {"phase": phase, "observed": observed,
              "ram_guard_gib": 64, "data_guard_gib": 64, "root_guard_gib": 10})
        if observed["passed"]:
            return observed
        time.sleep(30)  # Detached worker polling; no physical process dispatched.


def docker_command(source, archive, output, case, policy, candidate, evaluation, built, task=TASK):
    image, receipt = built["image"], Path(built["receipt"])
    receipt.resolve(strict=True).relative_to(output)
    actual = json.loads(receipt.read_text())
    if actual.get("image_id") != image or actual.get("task") != task:
        raise ValueError("Built runtime receipt identity differs")
    scratch = output / (policy + "-tmp")
    if scratch.exists():
        raise FileExistsError("Container scratch already exists; no retry")
    scratch.mkdir()
    return ["docker", "run", "--rm", "--network=none", "--read-only",
        "--cpus", "16", "--memory", "48g", "--memory-swap", "48g",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", str(os.getuid()) + ":" + str(os.getgid()),
        "-v", str(source) + ":" + str(source) + ":ro",
        "-v", str(archive) + ":" + str(archive) + ":ro",
        "-v", str(output) + ":" + str(output) + ":rw",
        "-v", str(scratch) + ":/tmp:rw",
        "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "NVIDIA_VISIBLE_DEVICES=none", "-e", "CUDA_VISIBLE_DEVICES=",
        "-e", "OPEN3DBENCH_IMAGE_ID=" + image,
        "-e", PREBUILT_OPENROAD_ENV + "=" + str(receipt),
        "--entrypoint", "python3", image,
        str(source / "AgenticBench/scripts/qualify_archive_grt.py"), "run",
        "--source", str(source), "--archive", str(archive), "--case", case,
        "--task", task, "--policy", policy, "--candidate", str(candidate), "--output", str(evaluation)]


def run_pair(root, source, archive, case, output, task=TASK):
    contract = task_contract(task)
    root, source, archive = (Path(p).expanduser().resolve(strict=True) for p in (root, source, archive))
    if source != FRAMEWORK.parent.resolve():
        raise ValueError("Host launcher must execute from the same frozen source checkout it qualifies")
    output = protect_destination(output, {"source_root": str(source), "reference_root": str(archive)})
    output.relative_to(root / "runs")
    if output == root / "runs" or output.exists():
        raise FileExistsError("Qualification pair needs a new output directory; no retries")
    cache = protect_destination(root / "cache" / (output.name + "-grt-compiler"),
                                {"source_root": str(source), "reference_root": str(archive)})
    if cache.exists():
        raise FileExistsError("Task-specific compiler cache already exists; choose a new authorized attempt")
    state = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
                           check=True, capture_output=True, text=True)
    if state.stdout.strip():
        raise ValueError("Qualification source must be a clean frozen checkout")
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "kind": "grt-interface-qualification-pair-v1",
        "source_commit": revision, "source": str(source), "archive": str(archive), "case": case,
        "task": task, "base_image": BASE_IMAGE, "state": "started", "attempts": [],
        "model_calls": 0, "generated_model_candidates": 0, "candidate_finals": 0,
        "verifier_rejections": 0,
        "native_physical_attempts": 0, "verified_improvements": 0,
        "container_cpus": 16, "container_memory_gib": 48, "physical_threads": 4,
        "build_threads": 8, "compiler_cache": str(cache), "compiler_cache_gib": 8,
        "timeout_seconds": 21600, "gpu": None, "retry_policy": "none", "retention": "full"}
    write(output / "pair-result.json", report)
    try:
        with slot(root, output) as lock:
            report["shared_lock"] = lock
            # Avoid compiling a candidate before discovering an incompatible
            # archived input. The container still repeats all authoritative
            # checks against its read-only archive mount immediately before use.
            preflight = qualify_inputs(source, archive, case)
            write(output / "readonly-input-preflight.json", preflight)
            report["readonly_input_preflight"] = "passed"
            for policy in contract["programs"]:
                candidate = materialize(source, policy, output / (policy + "-program"), task)
                row = {"policy": policy, "state": "prepared", "build_dispatches": 0,
                       "evaluation_container_dispatches": 0, "native_physical_attempts": 0}
                report["attempts"].append(row)
                admit(root, output, policy + "-build")
                row.update(state="building", build_dispatches=1)
                write(output / "pair-result.json", report)
                built = build_candidate_image(source / "AgenticBench/local_tasks" / contract["directory"],
                    candidate, source, BASE_IMAGE, output / (policy + "-build"), 8, "retain",
                    compiler_cache_dir=cache, compiler_cache_gb=8)
                row["build"] = built
                admit(root, output, policy + "-evaluate")
                evaluation = output / (policy + "-evaluation")
                command = docker_command(source, archive, output, case, policy, candidate, evaluation, built, task)
                row.update(state="evaluating", evaluation_container_dispatches=1,
                           native_physical_attempts=None, command=command)
                write(output / "pair-result.json", report)
                with (output / (policy + "-container.log")).open("w") as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, text=True)
                row["container_exit_code"] = process.returncode
                result_path = evaluation / "qualification-result.json"
                if result_path.is_file():
                    result = json.loads(result_path.read_text())
                    if result.get("task") != task or result.get("case") != case or result.get("policy") != policy:
                        raise ValueError("Qualification result identity differs")
                    row["native_physical_attempts"] = result.get("native_physical_attempts")
                    row["qualification"] = result
                if process.returncode or not result_path.is_file() or result.get("status") != "passed":
                    raise RuntimeError(policy + " qualification failed or is uncertain; later policy is not dispatched")
                if task == "global_routing_iteration_controller_source" and policy == "probe":
                    try:
                        row["effective_probe_evidence"] = validate_iteration_probe_effect(
                            output / "identity-evaluation/run", evaluation / "run")
                    except Exception:
                        row["state"] = "effective-probe-rejected"
                        raise
                row["state"] = "passed"
                write(output / "pair-result.json", report)
            report["state"] = "passed"
    except Exception as error:
        report.update(state="failed", error=type(error).__name__ + ": " + str(error))
    finally:
        known = sum(row["native_physical_attempts"] or 0 for row in report["attempts"])
        unknown = sum(row["native_physical_attempts"] is None for row in report["attempts"])
        report.update(native_physical_attempts=None if unknown else known,
                      native_physical_attempts_known=known, native_physical_attempts_unknown=unknown,
                      build_dispatches=sum(row["build_dispatches"] for row in report["attempts"]),
                      evaluation_container_dispatches=sum(row["evaluation_container_dispatches"] for row in report["attempts"]),
                      generated_diagnostic_programs=len(report["attempts"]),
                      successful_evaluations=sum(row.get("qualification", {}).get("successful_evaluations", 0)
                                                 for row in report["attempts"]),
                      qualification_passes=sum(row.get("state") == "passed"
                                               for row in report["attempts"]),
                      qualification_rejections=sum(row.get("qualification", {}).get("status") == "failed"
                                                   or row.get("state") == "effective-probe-rejected"
                                                   for row in report["attempts"]))
        write(output / "pair-result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "archive", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("bp_fe", "bp", "ariane133", "swerv_wrapper"))
    parser.add_argument("--task", default=TASK, choices=tuple(TASKS),
                        help="Select explicitly for new tests; legacy default remains net ordering.")
    args = parser.parse_args(argv)
    report = run_pair(args.root, args.source, args.archive, args.case, args.output, args.task)
    print(json.dumps({"state": report["state"], "native_physical_attempts": report["native_physical_attempts"]}))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
