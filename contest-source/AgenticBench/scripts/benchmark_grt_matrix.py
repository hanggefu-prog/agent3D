"""One fresh, prefix-reusing matrix trial after native qualification completes.

This host-only driver uses the existing queue, resource limits and verifier.
It never edits the frozen evaluator, physical inputs or prior measurements.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from grt_bench.common import digest, evaluator_identity, finite, read, sha, write
from grt_bench.contract import matrix, validate
from grt_bench.queue import Queue

ACTIVE = {"queued", "running", "uncertain", "cancelling"}
STAGES = ("partition", "placement", "pdn", "hbt", "opt", "cts")


def summarize(root, task, candidate, rows, expected_jobs):
    """Require complete, freshly executed cells; never total historical stages."""
    root = Path(root)
    expected = {(p["case"], p["seed"]["id"]): p for p in matrix(task, candidate)}
    if {r["id"] for r in rows} != set(expected_jobs) or len(rows) != len(expected):
        raise ValueError("Timing requires the complete admitted matrix")
    if any(r["state"] != "succeeded" for r in rows):
        raise ValueError("Failed or incomplete trials have no complete-matrix timing")
    cells, seen = [], set()
    for row in rows:
        payload = json.loads(row["payload"])
        key = (payload["case"], payload["seed"]["id"])
        if key in seen or expected.get(key) != payload:
            raise ValueError("Unexpected timing matrix cell")
        seen.add(key)
        folder = root / "runs" / row["id"]
        result, plan = read(folder / "result.json"), read(folder / "control/plan.json")
        if result.get("physical_replay") or plan.get("rescore"):
            raise ValueError("Rescoring is not a partition-to-GRT timing trial")
        if result["status"] != "verified" or any(result[k] != v for k, v in payload.items()):
            raise ValueError("Unverified timing evidence")
        if result["evidence"] != {
            "plan_sha256": sha(folder / "control/plan.json"),
            "analysis_manifest_sha256": sha(folder / "output/analysis/manifest.json"),
        }:
            raise ValueError("Timing evidence changed")
        run = read(folder / "output/run/run.json")
        if run.get("frontend_prefix", {}).get("boundary") != "partition":
            raise ValueError("Timing trial must import the validated frontend prefix")
        for stage in ("input", "floorplan"):
            if not run["stages"][stage].get("source_manifest_sha256"):
                raise ValueError("Missing prefix reuse evidence")
        timing = {}
        for stage in STAGES:
            record = run["stages"][stage]
            if record["status"] != "success" or record.get("source_manifest_sha256"):
                raise ValueError("Required physical stage was not freshly executed")
            timing[stage] = finite(record["runtime_seconds"], stage, 0)
        partition_start = datetime.fromisoformat(run["stages"]["partition"]["started_at"]).timestamp()
        submitted, started, finished = (finite(row[k], k, 0) for k in ("submitted", "started", "finished"))
        if not submitted <= started <= partition_start <= finished:
            raise ValueError("Inconsistent timing timestamps")
        execution = read(folder / "output/execution.json")
        split = plan.get("execution_mode") == "gpu-frontend-cpu-backend"
        if execution.get("physical_execution_reused") or len(execution.get("commands", [])) != (3 if split else 2):
            raise ValueError("Missing fresh physical command timings")
        commands = execution["commands"]
        if split:
            if (execution.get("execution_mode") != plan["execution_mode"]
                    or execution["frontend_handoff_sha256"] != sha(folder / "output/gpu_frontend.json")
                    or "placement" not in commands[0]["command"]
                    or "pdn" not in commands[1]["command"] or "--resume" not in commands[1]["command"]):
                raise ValueError("Missing validated GPU-to-CPU handoff")
            released = read(folder / "resource-release.json")
            if released["allocation"]["gpus"] or not released["evidence"]["terminated"]:
                raise ValueError("GPU resource release was not confirmed")
        if "global-route" not in commands[-1]["command"]:
            raise ValueError("Missing GRT execution")
        timing["grt_command"] = finite(commands[-1]["wall_seconds"], "GRT", 0)
        timing["joint_scoring"] = finite(read(folder / "output/analysis/metrics.json")["runtime_seconds"], "scoring", 0)
        timing["verification"] = finite(result["accounting"]["verification"]["wall_seconds"], "verification", 0)
        cells.append({"job": row["id"], "case": key[0], "seed": key[1],
            "submitted_at": submitted, "partition_started_at": partition_start, "finished_at": finished,
            "queue_seconds": started - submitted, "worker_seconds": finished - started,
            "source_and_startup_seconds": partition_start - started,
            "partition_to_verified_seconds": finished - partition_start, "stage_seconds": timing})
    first, last = min(c["submitted_at"] for c in cells), max(c["finished_at"] for c in cells)
    duration = last - first
    if duration <= 0:
        raise ValueError("Invalid measured batch duration")
    cases = {}
    for case in task["cases"]:
        group = [c for c in cells if c["case"] == case["id"]]
        if not group:
            continue
        samples = [c["partition_to_verified_seconds"] for c in group]
        cases[case["id"]] = {"seeds": len(group), "samples_seconds": samples,
            "min_seconds": min(samples), "median_seconds": statistics.median(samples), "max_seconds": max(samples),
            "within_batch_span_seconds": max(c["finished_at"] for c in group) - min(c["partition_started_at"] for c in group)}
    return {"status": "measured", "task_digest": validate(task), "candidate": candidate,
        "cells": sorted(cells, key=lambda c: (c["case"], c["seed"])), "per_case": cases,
        "batch_submit_to_verified_seconds": duration,
        "first_partition_to_last_verified_seconds": last - min(c["partition_started_at"] for c in cells),
        "average_busy_workers": sum(c["worker_seconds"] for c in cells) / duration,
        "historical_prefix_runtime_included": False, "native_build_performed": False,
        "notes": ["Within-batch case spans include interleaving with other cases.",
                  "Summed worker durations are not a measured serial baseline or a speedup."]}


def render(report):
    lines = ["# Fresh partition-to-GRT matrix timing", "",
        "All {} cells passed independent verification.".format(len(report["cells"])), "",
        "- Submission to last verified result: {:.2f} min.".format(report["batch_submit_to_verified_seconds"] / 60),
        "- First partition start to last verified result: {:.2f} min.".format(report["first_partition_to_last_verified_seconds"] / 60),
        "- Mean concurrent busy workers: {:.2f}.".format(report["average_busy_workers"]), "",
        "Input/floorplan checkpoints and the installed image were reused. Their historical",
        "execution and compilation time are excluded. Queueing, source preparation, prefix",
        "validation/import and final verification remain in the submission-to-result time.", "",
        "| Case | Seeds | Minimum (min) | Median (min) | Maximum (min) |", "|---|---:|---:|---:|---:|"]
    for case, row in report["per_case"].items():
        lines.append("| {} | {} | {:.2f} | {:.2f} | {:.2f} |".format(case, row["seeds"],
            row["min_seconds"] / 60, row["median_seconds"] / 60, row["max_seconds"] / 60))
    return "\n".join(lines) + "\n"


def identify_workers(records, root):
    checked = []
    for row in records:
        if "grt_bench.worker" not in row["command"]:
            continue
        path = Path("/proc") / str(row["pid"]) / "cmdline"
        arguments = [p.decode() for p in path.read_bytes().split(b"\0") if p]
        if arguments != row["command"] or str(root) not in " ".join(arguments):
            raise RuntimeError("Worker PID identity changed")
        checked.append(row)
    if not checked:
        raise RuntimeError("No identified workers to hand over")
    return checked


def process_active(pid):
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(") ", 1)[1].split()[0] != "Z"


def stop_idle_workers(checked):
    for row in checked:
        os.kill(row["pid"], signal.SIGTERM)
    for _ in range(50):
        if all(not process_active(r["pid"]) for r in checked):
            return checked
        time.sleep(.1)
    raise RuntimeError("Idle worker shutdown did not complete")


def launch(command, framework, log_path):
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(command, cwd=framework,
            env=dict(os.environ, PYTHONPATH=str(framework), PYTHONDONTWRITEBYTECODE="1"),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, {"pid": process.pid, "command": command}


def freeze_timing_candidate(root, parent, output, trial_id):
    from grt_bench.candidates import freeze, materialize, verify_snapshot
    root, output = Path(root), Path(output)
    before = verify_snapshot(root / "candidates" / parent)
    # freeze intentionally rejects source trees inside the snapshot store.
    # Materialize through the same checked workspace boundary used by workers.
    source = materialize(root / "candidates" / parent, output / "candidate-source")
    policy = {"allowed_paths": ["Place-3D/src/partition"], "native": False,
              "timing_trial": trial_id}
    candidate = freeze(source, root / "candidates", parent=parent, policy=policy)
    after = verify_snapshot(root / "candidates" / candidate)
    if before["files"] != after["files"] or after["scope"]["changes"]:
        raise ValueError("Timing trial changed source code")
    return candidate


def run(plan_path):
    from grt_bench.__main__ import submission_order
    plan = read(plan_path)
    if type(plan["worker_count"]) is not int or plan["worker_count"] < 1:
        raise ValueError("A positive worker count is required")
    root, output, framework = (Path(plan[k]).resolve() for k in ("root", "output", "framework"))
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    write(output / "process.json", {"pid": os.getpid(), "kind": "one-shot-fresh-matrix-timing"})
    def status(state, **extra):
        write(output / "status.json", dict(status=state, **extra), replace=True)
    status("waiting_for_qualification")
    jobs, paused, workers = [], [], []
    queue = Queue(root / "control/queue.sqlite")
    try:
        receipt = Path(plan["qualification"])
        while not receipt.exists():
            prior = read(plan["qualification_controller_status"])
            if prior["status"] == "failed":
                raise RuntimeError("Native qualification failed; timing was not launched")
            time.sleep(30)
        qualified = read(receipt)
        if qualified["status"] != "qualified" or qualified["evaluator_sha256"] != evaluator_identity():
            raise ValueError("A matching qualified evaluator is required")
        task = read(root / "control/tasks" / (qualified["task"] + ".json"))
        if validate(task) != qualified["task"]:
            raise ValueError("Qualified task changed")
        reference = Path(qualified["reference"])
        for name, checksum in qualified["artifacts"].items():
            if sha(reference / name) != checksum:
                raise ValueError("Qualified reference changed")
        setup_start = time.monotonic()
        parent = qualified["candidate"]
        candidate = freeze_timing_candidate(root, parent, output, plan["trial_id"])
        runtime = read(root / "control/runtimes" / (parent + ".json"))
        if runtime.get("native"):
            raise ValueError("This timing driver requires the qualified base-image candidate")
        write(root / "control/runtimes" / (candidate + ".json"), dict(runtime, candidate=candidate))
        environment = read(plan["environment"])
        if environment["framework"] != str(framework) or environment["base_image"] != runtime["image"]:
            raise ValueError("Qualified environment changed")
        environment.pop("rescore_sources", None)
        environment.pop("rescore_previous_framework", None)
        bindings = {}
        outcomes = read(reference / "outcomes.json")
        for case in task["cases"]:
            outcome = next(r for r in outcomes if r["case"] == case["id"] and r["seed"] == task["seeds"][0])
            source = outcome.get("physical_replay", {}).get("source_job")
            if not source:
                raise ValueError("Missing qualified physical source")
            prefix = root / "runs" / source / "output/run"
            bindings[case["id"]] = {"source": str(prefix), "manifests": {
                stage: sha(prefix / stage / "stage_manifest.json") for stage in ("00_input", "10_floorplan")}}
            environment["cases"][case["id"]]["frontend_prefix"] = str(prefix)
        environment["cache_frontend_prefix"] = False
        write(output / "environment.json", environment)
        write(output / "setup.json", {"candidate": candidate, "parent": parent, "image": runtime["image"],
            "unchanged_source": True, "native_build_performed": False, "prefixes": bindings,
            "setup_seconds": time.monotonic() - setup_start})
        if any(r["state"] in ACTIVE for r in queue.rows()):
            raise RuntimeError("Timing handover requires an idle production queue")
        original = read(plan["worker_processes"])
        paused = identify_workers(original, root)
        stop_idle_workers(paused)
        if any(r["state"] in ACTIVE for r in queue.rows()):
            raise RuntimeError("Queue changed during timing handover")
        started, monotonic_start = time.time(), time.monotonic()
        for cell in submission_order(task, candidate):
            job = digest({"payload": cell, "resources": plan["resources"]})
            if (root / "runs" / job).exists():
                raise ValueError("Timing attempt already exists")
            jobs.append(queue.submit(cell, plan["resources"]))
        write(output / "admission.json", {"jobs": jobs, "submitted_at": started,
            "worker_count": plan["worker_count"], "resources": plan["resources"], "task": qualified["task"]})
        for index in range(plan["worker_count"]):
            command = ["python3", "-m", "grt_bench.worker", "--environment", str(output / "environment.json"),
                       "--host", plan["host"], "--token", plan["token"]]
            workers.append(launch(command, framework, output / ("worker-{}.log".format(index))))
        write(output / "workers.json", [row for _, row in workers])
        previous = None
        while True:
            rows = [r for r in queue.rows() if r["id"] in jobs]
            counts = dict(Counter(r["state"] for r in rows))
            if counts != previous:
                status("running", counts=counts, expected=len(jobs)); previous = counts
            if any(r["state"] in {"uncertain", "cancelling"} for r in rows):
                raise RuntimeError("Timing execution requires host reconciliation")
            if all(r["state"] not in ACTIVE for r in rows):
                break
            if any(process.poll() is not None for process, _ in workers):
                raise RuntimeError("Timing worker exited before the batch completed")
            time.sleep(5)
        report = summarize(root, task, candidate, rows, jobs)
        report["observed_request_seconds"] = time.monotonic() - monotonic_start
        report["wall_minus_monotonic_seconds"] = (time.time() - started) - report["observed_request_seconds"]
        report["worker_count"] = plan["worker_count"]
        report["resources_per_cell"] = plan["resources"]
        write(output / "timing.json", report)
        (output / "TIMING.md").write_text(render(report))
        status("measured", report=str(output / "timing.json"))
    except BaseException as error:
        for row in queue.rows():
            if row["id"] in jobs and row["state"] == "queued":
                queue.cancel(row["id"])
        status("failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        busy = any(r["state"] in ACTIVE for r in queue.rows() if r["id"] in jobs)
        if paused and not busy:
            for process, _ in workers:
                if process.poll() is None:
                    process.terminate(); process.wait(timeout=10)
            restored = [row for row in read(plan["worker_processes"]) if "grt_bench.worker" not in row["command"]]
            for index, row in enumerate(paused):
                if process_active(row["pid"]):
                    # A failed shutdown may leave an original worker alive.
                    # Revalidate it and do not launch a duplicate.
                    identify_workers([row], root)
                    restored.append(row)
                    continue
                _, record = launch(row["command"], framework, output / ("restored-worker-{}.log".format(index)))
                restored.append(record)
            write(output / "restored-services.json", restored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    run(parser.parse_args().plan)
