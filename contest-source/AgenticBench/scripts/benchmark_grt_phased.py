"""One-shot native qualification/timing after a previous matrix drains.

Only the immutable input/floorplan prefix is adopted from the old evaluator.
Every partition-to-GRT result is executed and verified under the new identity.
No prior job is cancelled and no old reference is relabelled.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time

from grt_bench.common import evaluator_identity, read, sha, write
from grt_bench.contract import METRICS, validate
from grt_bench.queue import Queue
from scripts.benchmark_grt_matrix import (ACTIVE, freeze_timing_candidate, identify_workers,
    launch, process_active, render, stop_idle_workers, summarize)

SCHEDULING_MODULES = {"queue.py", "service.py", "worker.py", "flow.py"}


def prepare_contract(qualified, previous, current, task):
    """Prove the old prefix anchor and preserve the entire scoring contract."""
    previous, current = Path(previous), Path(current)
    if qualified["status"] != "qualified" or evaluator_identity(previous / "grt_bench") != qualified["evaluator_sha256"]:
        raise ValueError("Previous qualification/evaluator changed")
    if validate(task) != qualified["task"] or task["evaluator_sha256"] != qualified["evaluator_sha256"]:
        raise ValueError("Previous task changed")
    old = {p.name for p in (previous / "grt_bench").glob("*.py")}
    new = {p.name for p in (current / "grt_bench").glob("*.py")}
    if old != new:
        raise ValueError("Unexpected evaluator module inventory change")
    changed = sorted(n for n in old if sha(previous / "grt_bench" / n) != sha(current / "grt_bench" / n))
    if set(changed) - SCHEDULING_MODULES:
        raise ValueError("Scoring/verification contract changed during scheduling qualification")
    helper = "scripts/workspace_cts_thermal.py"
    if sha(previous / helper) != sha(current / helper):
        raise ValueError("Thermal helper changed")
    for name, checksum in qualified["artifacts"].items():
        if sha(Path(qualified["reference"]) / name) != checksum:
            raise ValueError("Qualified prefix reference changed")
    updated = dict(task, evaluator_sha256=evaluator_identity(current / "grt_bench"))
    validate(updated)
    return updated, changed


def inspect_process(record, module, root):
    command = [p.decode() for p in (Path("/proc") / str(record["pid"]) / "cmdline").read_bytes().split(b"\0") if p]
    if command != record["command"] or module not in command or str(root) not in " ".join(command):
        raise RuntimeError("Service process identity changed")


def seal(root, output, task, candidate, rows, image):
    from grt_bench.pareto import aggregate, frontier
    outcomes, evidence = [], []
    for row in rows:
        folder = root / "runs" / row["id"]
        outcomes.append(read(folder / "result.json"))
        evidence.append({"job": row["id"], "result_sha256": sha(folder / "result.json"),
                         "physical_source": row["id"]})
    outcomes.sort(key=lambda r: (r["case"], r["seed"]["id"]))
    baseline = aggregate(task, candidate, outcomes, outcomes)
    reference = root / "references" / validate(task) / candidate
    reference.mkdir(parents=True, exist_ok=False)
    raw = {}
    for case in task["cases"]:
        group = [r for r in outcomes if r["case"] == case["id"]]
        raw[case["id"]] = {m: {"mean": statistics.mean(r["metrics"][m] for r in group),
            "stdev": statistics.stdev(r["metrics"][m] for r in group),
            "worst": max(r["metrics"][m] for r in group), "samples": [r["metrics"][m] for r in group]} for m in METRICS}
    for name, value in (("task.json", task), ("outcomes.json", outcomes), ("aggregate.json", baseline),
            ("frontier.json", frontier([baseline], task["epsilon"])), ("raw_case_statistics.json", raw), ("evidence.json", evidence)):
        write(reference / name, value)
    receipt = {"status": "qualified", "task": validate(task), "candidate": candidate,
        "evaluator_sha256": task["evaluator_sha256"], "image": image, "reference": str(reference),
        "cells": len(rows), "verified_cells": len(rows), "signoff": False, "scientific_split_frozen": False,
        "qualified_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {p.name: sha(p) for p in reference.iterdir() if p.is_file()},
        "timing": str(output / "timing.json"), "execution_mode": "gpu-frontend-cpu-backend"}
    write(output / "qualified.json", receipt)
    return receipt


def telemetry(queue, jobs, output):
    rows = [r for r in queue.rows() if r["id"] in jobs]
    live = [r for r in rows if r["state"] in {"running", "uncertain", "cancelling"}]
    sample = {"at": time.time(), "counts": dict(Counter(r["state"] for r in rows)),
        "allocations": [{"job": r["id"], "resources": json.loads(r["allocation"] or r["resources"]),
            "gpus": json.loads(r["result"])["gpus"]} for r in live]}
    try:
        sample["gpu_csv"] = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        sample["host_meminfo"] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemTotal:", "MemAvailable:"))}
    except (OSError, subprocess.SubprocessError) as error:
        sample["telemetry_error"] = str(error)
    with (output / "telemetry.jsonl").open("a") as stream:
        stream.write(json.dumps(sample) + "\n")
    return rows, sample


def run(plan_path):
    from grt_bench.__main__ import submission_order
    plan = read(plan_path)
    root, output, framework = (Path(plan[k]).resolve() for k in ("root", "output", "framework"))
    if str(framework) != str(Path(__file__).resolve().parents[1]):
        raise ValueError("Controller must execute its frozen framework")
    if (type(plan["worker_count"]) is not int or plan["worker_count"] < 6
            or plan["capacity"]["gpu_slots_per_device"] != 3 or len(plan["capacity"]["gpus"]) != 2):
        raise ValueError("This trial requires at least six workers and three slots on each of two GPUs")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    write(output / "process.json", {"pid": os.getpid(), "kind": "one-shot-phased-matrix"})
    def status(state, **extra):
        write(output / "status.json", dict(status=state, **extra), replace=True)
    jobs, services = [], []
    try:
        qualified = read(plan["qualification"])
        original_task = read(root / "control/tasks" / (qualified["task"] + ".json"))
        task, changes = prepare_contract(qualified, plan["previous_framework"], framework, original_task)
        write(output / "transition.json", {"previous_task": qualified["task"], "task": validate(task),
            "changed_modules": changes, "previous_receipt_sha256": sha(plan["qualification"]),
            "prefix_only_reuse": True, "fresh_physical_qualification_required": True})
        if plan.get("canary"):
            status("waiting_for_native_canary", canary=plan["canary"])
            while True:
                canary = read(Path(plan["canary"]) / "status.json")
                if canary["status"] == "failed":
                    raise RuntimeError("Native phase canary failed; batch was not admitted")
                if canary["status"] == "verified":
                    proof = read(Path(plan["canary"]) / "receipt.json")
                    if proof["evaluator_sha256"] != task["evaluator_sha256"]:
                        raise ValueError("Canary evaluator differs")
                    break
                time.sleep(30)
        # Do not instantiate/migrate the production DB while its old service runs.
        predecessor = Path(plan["predecessor"])
        status("waiting_for_previous_trial", predecessor=str(predecessor))
        while True:
            before = read(predecessor / "status.json")
            if before["status"] == "failed":
                raise RuntimeError("Previous trial failed; inspect it before service handover")
            if before["status"] == "measured" and (predecessor / "restored-services.json").exists():
                break
            time.sleep(30)
        queue = Queue(root / "control/queue.sqlite")
        if any(r["state"] in ACTIVE for r in queue.rows()):
            raise RuntimeError("Service handover requires an idle queue")
        status("preparing")
        candidate = freeze_timing_candidate(root, qualified["candidate"], output, plan["trial_id"])
        runtime = read(root / "control/runtimes" / (qualified["candidate"] + ".json"))
        if runtime.get("native"):
            raise ValueError("Timing requires the installed base-image candidate")
        write(root / "control/runtimes" / (candidate + ".json"), dict(runtime, candidate=candidate))
        task_path = root / "control/tasks" / (validate(task) + ".json")
        if task_path.exists():
            if read(task_path) != task: raise ValueError("Task collision")
        else:
            write(task_path, task)
        environment = read(plan["environment"])
        if environment["framework"] != str(framework) or environment["base_image"] != runtime["image"]:
            raise ValueError("Phased environment identity differs")
        if environment.get("rescore_sources") or environment.get("cache_frontend_prefix", True):
            raise ValueError("Phased timing requires an explicit immutable prefix and no replay")
        backend = environment["phase_resources"]["backend"]
        if set(backend) != set(plan["resources"]) or backend["gpu_count"] != 0 or any(backend[k] > plan["resources"][k] for k in backend):
            raise ValueError("Invalid backend resource envelope")
        prefixes = {}
        outcomes = read(Path(qualified["reference"]) / "outcomes.json")
        evidence = read(Path(qualified["reference"]) / "evidence.json")
        for case in task["cases"]:
            outcome = next(r for r in outcomes if r["case"] == case["id"] and r["seed"] == task["seeds"][0])
            source = outcome.get("physical_replay", {}).get("source_job")
            if not source:
                source = next(e["job"] for e in evidence if read(root / "runs" / e["job"] / "result.json") == outcome)
            prefix = root / "runs" / source / "output/run"
            prefixes[case["id"]] = {"source": str(prefix), "manifests": {
                s: sha(prefix / s / "stage_manifest.json") for s in ("00_input", "10_floorplan")}}
            environment["cases"][case["id"]]["frontend_prefix"] = str(prefix)
        write(output / "environment.json", environment)
        write(output / "setup.json", {"candidate": candidate, "parent": qualified["candidate"], "image": runtime["image"],
            "unchanged_source": True, "native_build_performed": False, "prefixes": prefixes})
        original = read(predecessor / "restored-services.json")
        workers = identify_workers(original, root)
        broker = next(r for r in original if "grt_bench.service" in r["command"])
        inspect_process(broker, "grt_bench.service", root)
        stop_idle_workers(workers)
        if any(r["state"] in ACTIVE for r in queue.rows()):
            raise RuntimeError("Queue changed during handover")
        os.kill(broker["pid"], signal.SIGTERM)
        for _ in range(100):
            if not process_active(broker["pid"]): break
            time.sleep(.1)
        if process_active(broker["pid"]): raise RuntimeError("Old idle broker did not stop")
        queue.register_host(plan["host"], plan["capacity"])
        process, record = launch(broker["command"], framework, output / "broker.log")
        services.append(record)
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("New broker failed to start")
            try:
                with socket.create_connection(("127.0.0.1", 8765), timeout=.2):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError("New broker did not become ready")
        write(output / "services.json", services)
        # All identities and resources are fixed before admission starts the clock.
        started = time.time()
        for cell in submission_order(task, candidate):
            jobs.append(queue.submit(cell, plan["resources"]))
        write(output / "admission.json", {"jobs": jobs, "task": validate(task), "submitted_at": started,
            "worker_count": plan["worker_count"], "gpu_concurrency": 6,
            "resources": plan["resources"], "backend_resources": backend})
        for index in range(plan["worker_count"]):
            command = ["python3", "-m", "grt_bench.worker", "--environment", str(output / "environment.json"),
                       "--host", plan["host"], "--token", plan["token"]]
            _, record = launch(command, framework, output / ("worker-{}.log".format(index)))
            services.append(record)
        write(output / "services.json", services, replace=True)
        peak, peak_gpu = 0, 0
        while True:
            rows, sample = telemetry(queue, jobs, output)
            peak = max(peak, sample["counts"].get("running", 0))
            gpu_active = sum(len(r["gpus"]) for r in sample["allocations"])
            peak_gpu = max(peak_gpu, gpu_active)
            status("running", counts=sample["counts"], expected=len(jobs), peak_running=peak,
                   gpu_active=gpu_active, peak_gpu=peak_gpu)
            if any(r["state"] in {"uncertain", "cancelling"} for r in rows):
                raise RuntimeError("Host reconciliation required; leases and queued work are retained")
            if all(r["state"] not in ACTIVE for r in rows): break
            if any(not process_active(r["pid"]) for r in services):
                raise RuntimeError("Service exited; jobs are retained for inspection")
            time.sleep(30)
        report = summarize(root, task, candidate, rows, jobs)
        report.update(worker_count=plan["worker_count"], gpu_concurrency=6, peak_gpu=peak_gpu,
                      peak_running=peak, resources_per_cell=plan["resources"],
                      backend_resources=backend, execution_mode="gpu-frontend-cpu-backend")
        write(output / "timing.json", report)
        (output / "TIMING.md").write_text(render(report))
        seal(root, output, task, candidate, rows, runtime["image"])
        status("measured", report=str(output / "timing.json"), qualified=str(output / "qualified.json"))
        # Keep the verified resource-bounded service ready for the next candidate.
        active = read(root / "control/active.json")
        write(root / "control/active.json", dict(active, provision=str(output / "qualified.json"),
            environment=str(output / "environment.json"), services=str(output / "services.json")), replace=True)
    except BaseException as error:
        status("failed", error=type(error).__name__ + ": " + str(error), jobs=jobs)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    run(parser.parse_args().plan)
