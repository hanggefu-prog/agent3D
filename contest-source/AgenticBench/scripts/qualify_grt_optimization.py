"""Host-only native baseline/candidate qualification for the frozen Pareto task.

Runs a complete paired matrix twice, preserving all prior evaluations. Service
handover is permitted only with an idle queue and matching recorded process IDs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import signal
import socket
import time

from grt_bench.__main__ import submission_order
from grt_bench.archive import Archive
from grt_bench.builder import build
from grt_bench.candidates import verify_snapshot
from grt_bench.common import digest, evaluator_identity, read, sha, write
from grt_bench.contract import validate
from grt_bench.optimization import define_task
from grt_bench.queue import Queue
from scripts.benchmark_grt_matrix import ACTIVE, identify_workers, launch, process_active, render, stop_idle_workers, summarize
from scripts.benchmark_grt_phased import inspect_process, seal, telemetry


def apply_edit(source, edit):
    path = Path(source) / edit["file"]
    before = path.read_text()
    if before.count(edit["before"]) != 1 or edit["before"] == edit["after"]:
        raise ValueError("Expected exactly one real source edit")
    after = before.replace(edit["before"], edit["after"])
    path.write_text(after)
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                      fromfile="a/" + edit["file"], tofile="b/" + edit["file"]))


def await_broker(process, port):
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("New broker exited before becoming ready")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2):
                return
        except OSError:
            time.sleep(.1)
    raise RuntimeError("New broker did not become ready")


def run(plan_path):
    plan = read(plan_path)
    root, framework, output = (Path(plan[k]).resolve() for k in ("root", "framework", "output"))
    if framework != Path(__file__).resolve().parents[1]:
        raise ValueError("Run the controller from its frozen deployment")
    if plan["worker_count"] != 12 or plan["capacity"]["gpu_slots_per_device"] != 3 or len(plan["capacity"]["gpus"]) != 2:
        raise ValueError("Qualification requires twelve workers and six GPU slots")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    write(output / "process.json", {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()})
    services = []

    def status(state, **extra):
        write(output / "status.json", dict(status=state, **extra), replace=True)

    try:
        status("preparing")
        previous = read(root / "control/active.json")
        write(output / "previous-active.json", previous)
        qualified = read(previous["provision"])
        if qualified["status"] != "qualified":
            raise ValueError("Previous baseline has not completed native qualification")
        base_task = read(root / "control/tasks" / (qualified["task"] + ".json"))
        if validate(base_task) != qualified["task"]:
            raise ValueError("Previous task changed")
        for name, checksum in qualified["artifacts"].items():
            if sha(Path(qualified["reference"]) / name) != checksum:
                raise ValueError("Previous reference changed")
        baseline = plan["baseline"]
        source = verify_snapshot(root / "candidates" / baseline)
        old_source = verify_snapshot(root / "candidates" / qualified["candidate"])
        if source["files"] != old_source["files"]:
            raise ValueError("This qualification requires the previously qualified source baseline")
        base_task["id"] = plan["task_id"]
        base_task["evaluator_sha256"] = evaluator_identity()
        prompt = (framework / "global_tasks/lol_post_grt_pareto/instruction.md").read_text()
        task = define_task(base_task, baseline, prompt, root / "candidates")
        task_id = validate(task)
        task_path = root / "control/tasks" / (task_id + ".json")
        write(task_path, task)
        archive = Archive(root, task)
        archive.admit(baseline)
        environment = read(previous["environment"])
        environment["framework"] = str(framework)
        if environment["root"] != str(root) or environment["base_image"] != qualified["image"]:
            raise ValueError("Runtime root/image changed")
        if environment.get("rescore_sources") or environment["phase_resources"]["backend"]["gpu_count"] != 0:
            raise ValueError("Requires fresh execution and CPU backend release")
        write(output / "baseline-environment.json", environment)
        runtime = build(root / "candidates" / baseline, environment["base_image"], root / "builds", root / "compiler-cache")
        runtime_path = root / "control/runtimes" / (baseline + ".json")
        if runtime_path.exists():
            old_runtime = read(runtime_path)
            if any(old_runtime.get(k) != runtime[k] for k in ("candidate", "native", "image")):
                raise ValueError("Baseline runtime differs")
        else:
            write(runtime_path, runtime)
        queue = Queue(root / "control/queue.sqlite")
        if any(r["state"] in ACTIVE for r in queue.rows()):
            raise ValueError("Deployment requires an idle queue; existing jobs were preserved")
        original_services = read(previous["services"])
        old_workers = identify_workers(original_services, root)
        broker = next(r for r in original_services if "grt_bench.service" in r["command"])
        inspect_process(broker, "grt_bench.service", root)
        stop_idle_workers(old_workers)
        if any(r["state"] in ACTIVE for r in queue.rows()):
            raise ValueError("Queue changed during service handover")
        os.kill(broker["pid"], signal.SIGTERM)
        for _ in range(100):
            if not process_active(broker["pid"]): break
            time.sleep(.1)
        if process_active(broker["pid"]):
            raise RuntimeError("Old idle broker did not stop")
        queue.register_host(plan["host"], plan["capacity"])
        process, broker_record = launch(broker["command"], framework, output / "broker.log")
        services.append(broker_record)
        await_broker(process, plan.get("port", 8765))
        write(output / "services.json", services)
        write(root / "control/optimization-active.json", {"task": task_id, "framework": str(framework),
            "output": str(output), "status": "qualifying"}, replace=True)

        def batch(candidate, label, env):
            folder = output / label; folder.mkdir()
            environment_path = folder / "environment.json"; write(environment_path, env)
            archive.admit(candidate)
            jobs = [queue.submit(cell, plan["resources"]) for cell in submission_order(task, candidate)]
            write(folder / "admission.json", {"candidate": candidate, "task": task_id, "jobs": jobs,
                "resources": plan["resources"], "backend_resources": env["phase_resources"]["backend"]})
            workers = []
            for index in range(plan["worker_count"]):
                command = ["python3", "-m", "grt_bench.worker", "--environment", str(environment_path),
                           "--host", plan["host"], "--token", plan["token"],
                           "--url", "http://127.0.0.1:" + str(plan.get("port", 8765))]
                _, record = launch(command, framework, folder / ("worker-{}.log".format(index)))
                workers.append(record)
            services[:] = [broker_record] + workers
            write(output / "services.json", services, replace=True)
            peak_gpu, peak_workers = 0, 0
            while True:
                rows, sample = telemetry(queue, jobs, folder)
                gpu = sum(len(r["gpus"]) for r in sample["allocations"])
                peak_gpu = max(peak_gpu, gpu)
                peak_workers = max(peak_workers, sample["counts"].get("running", 0))
                status(label, counts=sample["counts"], expected=len(jobs), gpu_active=gpu,
                       peak_gpu=peak_gpu, peak_running=peak_workers, candidate=candidate)
                if any(r["state"] in {"uncertain", "cancelling"} for r in rows):
                    raise RuntimeError("Ambiguous lease retained; host reconciliation required")
                if all(r["state"] not in ACTIVE for r in rows): break
                if any(not process_active(r["pid"]) for r in services):
                    raise RuntimeError("Service exited; no automatic physical retry")
                time.sleep(20)
            if any(r["state"] != "succeeded" for r in rows):
                write(folder / "failures.json", [{"job": r["id"], "payload": json.loads(r["payload"]),
                    "result": json.loads(r["result"])} for r in rows if r["state"] != "succeeded"])
                raise RuntimeError("Incomplete verified matrix; see preserved failure receipts")
            stop_idle_workers(identify_workers(workers, root))
            services[:] = [broker_record]
            write(output / "services.json", services, replace=True)
            report = summarize(root, task, candidate, rows, jobs)
            report.update(peak_gpu=peak_gpu, peak_running=peak_workers, gpu_concurrency=6, worker_count=12)
            write(folder / "timing.json", report)
            (folder / "TIMING.md").write_text(render(report))
            admitted = archive.ingest(candidate)
            write(folder / "archive-admission.json", admitted)
            write(folder / "archive.json", archive.show())
            return folder, rows, admitted

        baseline_folder, baseline_rows, _ = batch(baseline, "baseline", environment)
        baseline_receipt = seal(root, baseline_folder, task, baseline, baseline_rows, environment["base_image"])
        selection = archive.select(baseline, plan["candidate_edit"]["reason"])
        write(output / "parent-selection.json", selection)
        bad_source = output / "illegal-source"
        archive.checkout(bad_source, selection["id"])
        protected = bad_source / "open3dbench/runner.py"
        protected.write_text(protected.read_text() + "\n# Deliberate forbidden edit for verifier qualification.\n")
        try:
            archive.freeze(bad_source, selection["id"])
        except ValueError as error:
            if "Protected/out-of-scope" not in str(error): raise
            write(output / "verifier-negative.json", {"status": "rejected_as_expected", "error": str(error),
                "candidate_submitted": False, "modified_file": "open3dbench/runner.py"})
        else:
            raise AssertionError("Verifier accepted a protected evaluator edit")
        candidate_source = output / "candidate-source"
        archive.checkout(candidate_source, selection["id"])
        patch = apply_edit(candidate_source, plan["candidate_edit"])
        (output / "candidate.patch").write_text(patch)
        candidate = archive.freeze(candidate_source, selection["id"])
        write(output / "candidate-verification.json", archive.admit(candidate))
        receipt = build(root / "candidates" / candidate, environment["base_image"], root / "builds", root / "compiler-cache")
        write(root / "control/runtimes" / (candidate + ".json"), receipt)
        # The runner's prefix signature covers frontend source globally. Changed
        # placement source must prepare a fresh verified prefix, not relabel one.
        candidate_environment = json.loads(json.dumps(environment))
        candidate_environment["cache_frontend_prefix"] = True
        for case in candidate_environment["cases"].values(): case.pop("frontend_prefix", None)
        write(output / "candidate-environment.json", candidate_environment)
        candidate_folder, _, admission = batch(candidate, "candidate", candidate_environment)
        result = {"status": "qualified", "task": task_id, "baseline": baseline, "candidate": candidate,
            "baseline_cells": len(baseline_rows), "candidate_cells": admission["evaluation"]["cells"],
            "candidate_non_dominated": admission["non_dominated"], "vector": admission["evaluation"]["vector"],
            "candidate_feasible": admission["feasible"], "power_constraint": admission["evaluation"]["constraints"]["power_w"],
            "evaluator_sha256": evaluator_identity(), "image": environment["base_image"],
            "baseline_reference": baseline_receipt["reference"], "archive": str(archive.path),
            "completed_at": datetime.now(timezone.utc).isoformat(), "scientific_split_frozen": False,
            "baseline_timing": read(baseline_folder / "timing.json")["batch_submit_to_verified_seconds"],
            "candidate_timing": read(candidate_folder / "timing.json")["batch_submit_to_verified_seconds"],
            "verifier_negative": read(output / "verifier-negative.json")}
        write(output / "qualified.json", result)
        # Keep the deployed worker pool ready, with no GPU/CPU job reservation
        # while idle. All future jobs remain subject to the same broker limits.
        for index in range(plan["worker_count"]):
            command = ["python3", "-m", "grt_bench.worker", "--environment", str(output / "candidate-environment.json"),
                       "--host", plan["host"], "--token", plan["token"],
                       "--url", "http://127.0.0.1:" + str(plan.get("port", 8765))]
            _, record = launch(command, framework, output / ("ready-worker-{}.log".format(index)))
            services.append(record)
        write(output / "services.json", services, replace=True)
        write(root / "control/active.json", {"environment": str(output / "candidate-environment.json"),
            "provision": str(baseline_folder / "qualified.json"), "services": str(output / "services.json")}, replace=True)
        write(root / "control/optimization-active.json", dict(result, output=str(output)), replace=True)
        status("qualified", receipt=str(output / "qualified.json"), candidate_non_dominated=admission["non_dominated"])
    except BaseException as error:
        status("failed", error=type(error).__name__ + ": " + str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    run(parser.parse_args().plan)
