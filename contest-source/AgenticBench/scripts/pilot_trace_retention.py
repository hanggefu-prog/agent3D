#!/usr/bin/env python3
"""Small, isolated retention experiment using synthetic artifacts, never EDA.

The real frozen-source, Queue, Archive and paired-score code is exercised. Raw
ODB/SPEF/DEF/log payloads below are deliberately synthetic random bytes; neither
this script nor its output qualifies a physical design or a native verifier.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench import TIMING_ENDPOINT
from grt_bench.archive import Archive, completed_outcomes
from grt_bench.candidates import freeze
from grt_bench.common import digest, evaluator_identity, read, records, sha, verify_records, write
from grt_bench.contract import DEFAULT_SEEDS, DEFAULT_TIMING_SCORING, TIMING_METRICS, matrix, validate
from grt_bench.optimization import define_task
from grt_bench.queue import Queue

PARTITION = "Place-3D/src/partition/lol/methods/tp_gnn.py"
PROMPT = "---\ncomponents: [partition]\n---\nImprove general timing behavior.\n"
STAGES = ("00_input", "10_floorplan", "20_partition", "30_placement", "50_hbt", "60_opt", "70_cts", "80_route")


def write_payload(path: Path, size: int) -> None:
    """Allocate ordinary distinct files; do not manufacture sparse savings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        remaining = size
        while remaining:
            chunk = os.urandom(min(remaining, 65536))
            stream.write(chunk)
            remaining -= len(chunk)


def disk_usage(root: Path) -> dict:
    logical = allocated = files = 0
    inodes = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        logical += stat.st_size
        files += 1
        key = (stat.st_dev, stat.st_ino)
        if key not in inodes:
            allocated += stat.st_blocks * 512
            inodes.add(key)
    return {"logical_bytes": logical, "allocated_bytes_unique_inodes": allocated, "files": files}


class SyntheticCampaign:
    """Real control-plane receipts paired with explicitly synthetic raw files."""
    def __init__(self, root: Path, artifact_bytes: int = 4096):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_bytes = artifact_bytes
        self.source = self.root / "fixture-source"
        target = self.source / PARTITION
        target.parent.mkdir(parents=True)
        target.write_text("value = 1\n")
        self.baseline = freeze(self.source, self.root / "candidates")
        base = {"schema_version": 3, "id": "trace-pilot", "endpoint": TIMING_ENDPOINT,
                "evaluator_sha256": evaluator_identity(), "objectives": ["score"],
                "scoring": copy.deepcopy(DEFAULT_TIMING_SCORING), "metrics": list(TIMING_METRICS),
                "opt_passes": 1, "seeds": copy.deepcopy(DEFAULT_SEEDS), "epsilon": 0,
                "normalization_floors": {metric: 1.0 for metric in TIMING_METRICS},
                "cases": [{"id": name, "design": name, "pdk": "ng45", "weight": 1,
                           "split": "train", "config_sha256": "a" * 64,
                           "input_identity": "b" * 64} for name in ("bp_fe", "bp", "ariane133")]}
        self.task = define_task(base, self.baseline, PROMPT, self.root / "candidates")
        self.archive = Archive(self.root, self.task)
        self.queue = Queue(self.root / "control/queue.sqlite")
        self.queue.register_host("synthetic-pilot", {"cpu": 12, "memory_gb": 16,
                                  "storage_gb": 1, "gpus": [], "build_slots": 1})
        self.resources = {"cpu": 1, "memory_gb": 1, "storage_gb": 0.01,
                          "gpu_count": 0, "build_slots": 0}
        write(self.root / "control/tasks" / (validate(self.task) + ".json"), self.task)
        self.budget_path = self.root / "control/pilot-budget.json"
        write(self.budget_path, {"evaluation_limit": 7, "evaluations_used": 0,
                                "synthetic_only": True})
        self.serial = 1
        self.names = {}
        self.jobs = {}

    def child(self, parent: str, name: str) -> str:
        selection = self.archive.select(parent, "Synthetic retention branch: " + name)["id"]
        self.serial += 1
        workspace = self.root / ("development-" + name)
        self.archive.checkout(workspace, selection)
        (workspace / PARTITION).write_text("value = %d\n" % self.serial)
        return self.archive.freeze(workspace, selection)

    def complete(self, candidate: str, name: str, deficit: float, *, upstream_candidate=None) -> list[str]:
        jobs = []
        audit = self.archive.admit(candidate)
        for payload in matrix(self.task, candidate):
            job = self.queue.submit(payload, self.resources)
            lease = self.queue.claim("synthetic-pilot")
            if lease is None or lease["id"] != job:
                raise AssertionError("Unexpected synthetic queue admission")
            attempt = self.root / "runs" / job
            write(attempt / "control/task.json", self.task)
            write(attempt / "control/plan.json", dict(payload, output="/output"))
            for stage in STAGES:
                write_payload(attempt / "output/run" / stage / "stage.log", self.artifact_bytes)
            for suffix in ("odb", "def", "spef"):
                write_payload(attempt / "output/run/80_route" / ("design." + suffix), self.artifact_bytes)
            write_payload(attempt / "execution/frontend/container.log", self.artifact_bytes)
            incremental = None
            if upstream_candidate is not None:
                source_job = next(source for source in self.jobs[upstream_candidate]
                                  if all(read(self.root / "runs" / source / "control/plan.json").get(key) == payload[key]
                                         for key in ("case", "seed", "split")))
                source_stage = self.root / "runs" / source_job / "output/run/30_placement"
                shutil.copytree(source_stage, attempt / "output/run/30_placement", dirs_exist_ok=True)
                incremental = {"source_job": source_job, "physical_source_job": source_job,
                               "stages": {"placement": sha(source_stage / "stage_manifest.json")}}
                write(attempt / "control/incremental.json", incremental)
                plan = read(attempt / "control/plan.json")
                plan.update(incremental="/control/incremental.json", incremental_source="/parent-output")
                write(attempt / "control/plan.json", plan, replace=True)
            for stage in STAGES:
                folder = attempt / "output/run" / stage
                manifest = folder / "stage_manifest.json"
                if not manifest.exists():
                    write(manifest, {"stage": stage, "synthetic_only": True, "artifacts": records(folder)})
            analysis = attempt / "output/analysis"
            metrics = {"wns_deficit_ns": deficit, "tns_deficit_ns": deficit, "overflow": 100.0}
            write(analysis / "metrics.json", metrics)
            write(analysis / "manifest.json", {"artifacts": records(analysis)})
            write(attempt / "candidate-verification.json", audit)
            verified = {**payload, "status": "verified", "metrics": metrics,
                        "evidence": {"analysis_manifest_sha256": sha(analysis / "manifest.json"),
                                     "plan_sha256": sha(attempt / "control/plan.json")}}
            if incremental is not None:
                verified["incremental"] = incremental
            write(attempt / "verified/result.json", verified)
            result = {**verified, "accounting": {"synthetic_pilot": True},
                      "candidate_verification_sha256": sha(attempt / "candidate-verification.json")}
            write(attempt / "result.json", result)
            write(attempt / "physical-inventory.json", records(attempt / "output"))
            self.queue.finish(job, lease["token"], result, True)
            jobs.append(job)
        self.archive.ingest(candidate)
        self.names[name] = candidate
        self.jobs[candidate] = jobs
        budget = read(self.budget_path)
        budget["evaluations_used"] += 1
        write(self.budget_path, budget, replace=True)
        return jobs

    def terminal_branch(self, name: str, *, uncertain: bool = False) -> tuple[str, str]:
        parent = self.archive.show()["frontier"][0]["candidate"]
        candidate = self.child(parent, name)
        payload = matrix(self.task, candidate)[0]
        job = self.queue.submit(payload, self.resources)
        lease = self.queue.claim("synthetic-pilot")
        if lease is None or lease["id"] != job:
            raise AssertionError("Unexpected terminal branch queue order")
        attempt = self.root / "runs" / job
        write(attempt / "control/plan.json", payload)
        write_payload(attempt / "output/run/30_placement/diagnostic.log", self.artifact_bytes)
        if uncertain:
            # Deliberately expire the real lease, preserving reserved resources.
            with sqlite3.connect(self.queue.path) as db:
                db.execute("UPDATE jobs SET expires=0 WHERE id=?", (job,))
            self.queue.rows()
        else:
            self.queue.finish(job, lease["token"], {"synthetic_pilot": True, "error": "fixture failure"}, False)
        self.names[name] = candidate
        self.jobs[candidate] = [job]
        budget = read(self.budget_path)
        budget["evaluations_used"] += 1
        write(self.budget_path, budget, replace=True)
        return candidate, job

    def branches(self) -> dict[str, str]:
        self.complete(self.baseline, "baseline", 9.0)
        improved = self.child(self.baseline, "improved")
        self.complete(improved, "improved", 7.0)
        weak = self.child(improved, "weak")
        self.complete(weak, "weak", 10.0)
        tied = self.child(improved, "tie")
        self.complete(tied, "tie", 7.0)
        derived = self.child(tied, "derived")
        self.complete(derived, "derived", 6.0, upstream_candidate=tied)
        return self.names


def pins_for(fixture: SyntheticCampaign, paths=()) -> dict:
    return {"version": 1, "complete": True, "root": str(fixture.root.resolve()),
            "task_digest": validate(fixture.task), "generation": "synthetic-fixture-1",
            "paths": [{"path": str(path), "reason": "Explicit paired upstream checkpoint/log dependency"}
                      for path in paths], "candidates": [], "jobs": []}


def prune_paths(fixture: SyntheticCampaign, candidate: str) -> list[str]:
    result = []
    for job in fixture.jobs[candidate]:
        for path in (fixture.root / "runs" / job).rglob("*"):
            if path.is_file() and (path.suffix in {".odb", ".def", ".spef", ".log", ".rpt"}):
                result.append(path.relative_to(fixture.root).as_posix())
    return sorted(result)


def run_pilot(output: Path, artifact_bytes: int = 262144) -> dict:
    from global_bench.grt_trace_retention import TraceRetention
    root = output / "synthetic-pilot-campaign"
    if output.exists():
        raise ValueError("Pilot output must be new; never reuse a production or previous experiment directory")
    output.mkdir(parents=True)
    origin = output / "synthetic-origin"
    origin.mkdir()
    write(origin / "README.json", {"synthetic_only": True, "native_eda": False})
    started = time.perf_counter()
    fixture = SyntheticCampaign(root, artifact_bytes)
    fixture.branches()
    initial_before_policy = disk_usage(root)
    retention = TraceRetention.initialize(root, origin, fixture.task, acknowledge_isolated_copy=True)
    for index, name in enumerate(("baseline", "improved", "weak", "tie", "derived")):
        retention.seal(fixture.names[name], "synthetic-event-%d" % index)
    before = disk_usage(root)
    budget_before = sha(fixture.budget_path)
    queue_before = digest(fixture.queue.rows())
    archive_before = digest(fixture.archive.show())
    weak = fixture.names["weak"]
    plan_seconds = apply_seconds = 0.0
    receipts = []
    for name in ("weak", "tie", "improved", "baseline"):
        candidate = fixture.names[name]
        paths = prune_paths(fixture, candidate)
        if name == "baseline":
            # Immutable reusable input/floorplan remain dependency-pinned.
            paths = [path for path in paths if "/00_input/" not in path and "/10_floorplan/" not in path]
        plan_started = time.perf_counter()
        plan = retention.plan_prune(candidate, paths, pins=pins_for(fixture))
        plan_seconds += time.perf_counter() - plan_started
        apply_started = time.perf_counter()
        receipts.append(retention.apply(plan))
        apply_seconds += time.perf_counter() - apply_started
    history = retention.history(weak)
    after = disk_usage(root)
    if sha(fixture.budget_path) != budget_before or digest(fixture.queue.rows()) != queue_before:
        raise AssertionError("Retention changed evaluation accounting or authoritative queue")
    if digest(fixture.archive.show()) != archive_before:
        raise AssertionError("Retention changed the archive")
    fixture.terminal_branch("failed")
    fixture.terminal_branch("uncertain", uncertain=True)
    report = {"kind": "synthetic-control-plane-storage-pilot", "native_eda": False,
              "cases": [case["id"] for case in fixture.task["cases"]],
              "seed_ids": [seed["id"] for seed in fixture.task["seeds"]],
              "complete_candidate_matrices": 5, "successful_cells": 45,
              "partial_admitted_branches": 2, "budget_at_pruning": 5,
              "budget_after_failure_and_uncertain_admissions": read(fixture.budget_path),
              "verified_incremental_cells": 9,
              "artifact_bytes_per_file": artifact_bytes, "initial_before_policy": initial_before_policy,
              "peak_after_sealing": before, "after": after,
              "reclaimed_logical_bytes": before["logical_bytes"] - after["logical_bytes"],
              "reclaimed_allocated_bytes": before["allocated_bytes_unique_inodes"] - after["allocated_bytes_unique_inodes"],
              "net_saved_logical_bytes_vs_original": initial_before_policy["logical_bytes"] - after["logical_bytes"],
              "net_saved_allocated_bytes_vs_original": initial_before_policy["allocated_bytes_unique_inodes"] - after["allocated_bytes_unique_inodes"],
              "plan_seconds": plan_seconds, "apply_seconds": apply_seconds,
              "wall_seconds": time.perf_counter() - started, "trace": retention.trace(),
              "pruned_history": history, "prune_receipts": receipts,
              "budget_unchanged": True, "queue_unchanged": True, "archive_unchanged": True,
              "source_sha256": {"module": sha(Path(__file__).resolve().parents[1] / "global_bench/grt_trace_retention.py"),
                                "driver": sha(Path(__file__).resolve())},
              "limitations": ["Random byte payloads are not physical databases or real EDA logs.",
                              "This result measures local small-file retention, not native EDA runtime or server capacity."]}
    write(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-bytes", type=int, default=262144)
    args = parser.parse_args()
    if args.artifact_bytes < 4096 or args.artifact_bytes > 4 * 1024 * 1024:
        parser.error("Use between 4 KiB and 4 MiB per synthetic artifact")
    report = run_pilot(args.output.resolve(), args.artifact_bytes)
    print(json.dumps({key: report[key] for key in ("kind", "native_eda", "complete_candidate_matrices",
          "successful_cells", "reclaimed_logical_bytes", "reclaimed_allocated_bytes", "plan_seconds", "apply_seconds")}, indent=2))


if __name__ == "__main__":
    main()
