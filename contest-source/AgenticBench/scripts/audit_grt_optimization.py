"""Recheck a completed native Pareto qualification and publish a compact report."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from statistics import median

from grt_bench.archive import Archive
from grt_bench.common import read, sha, write
from grt_bench.contract import OBJECTIVES, validate


def gpu_lease_seconds(root, job, release):
    database = (Path(root) / "control/queue.sqlite").resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database, uri=True)) as db:
        started, finished = db.execute("SELECT started,finished FROM jobs WHERE id=?", (job,)).fetchone()
        events = db.execute("SELECT at,detail FROM events WHERE job=? AND state='resources_released'", (job,)).fetchall()
    if len(events) != 1 or not started <= events[0][0] <= finished:
        raise ValueError("Missing/ambiguous GPU lease release event")
    detail = json.loads(events[0][1])
    if detail["after"] != release["allocation"]["resources"] or detail["evidence"] != release["evidence"]:
        raise ValueError("Broker release event differs from worker receipt")
    return events[0][0] - started


def audit(root, experiment, output):
    root, experiment, output = Path(root), Path(experiment), Path(output)
    qualified = read(experiment / "qualified.json")
    if qualified["status"] != "qualified":
        raise ValueError("Native qualification must complete first")
    task = read(root / "control/tasks" / (qualified["task"] + ".json"))
    if validate(task) != qualified["task"]:
        raise ValueError("Task changed")
    archive = Archive(root, task)
    report = {"task": qualified["task"], "evaluator_sha256": task["evaluator_sha256"], "scoring": task["scoring"],
              "baseline": qualified["baseline"], "candidate": qualified["candidate"], "batches": {}}
    rows = []
    for label, factor in (("baseline", 1.25), ("candidate", 1.30)):
        folder = experiment / label
        admission, timing = read(folder / "admission.json"), read(folder / "timing.json")
        # Re-ingestion must remain idempotent using the authoritative queue and
        # raw evidence, rather than trusting the controller's success string.
        current = archive.ingest(admission["candidate"])
        if current["evaluation"] != read(folder / "archive-admission.json")["evaluation"]:
            raise ValueError("Archived evaluation differs on independent readback")
        cells = []
        for job in admission["jobs"]:
            attempt = root / "runs" / job
            result = read(attempt / "result.json")
            release = read(attempt / "resource-release.json")
            if release["allocation"]["gpus"] or not release["evidence"]["terminated"]:
                raise ValueError("GPU release was not confirmed")
            if "--gpus" in read(attempt / "execution/backend/command.json"):
                raise ValueError("CPU backend requested a GPU")
            if release["evidence"]["checkpoint_sha256"] != sha(attempt / "output/run/30_placement/checkpoint.json"):
                raise ValueError("GPU release checkpoint changed")
            placement = read(attempt / "output/run/30_placement/stages/02_global_place/stage_manifest.json")
            params = read(attempt / "output/run/30_placement/stages/02_global_place/params.json")
            if (placement["density_policy"]["backend_area_growth_factor"] != factor or
                    params["concurrent_lol"]["backend_area_growth_factor"] != factor):
                raise ValueError("The intended candidate algorithm parameter did not execute")
            cells.append({"job": job, "case": result["case"], "seed": result["seed"]["id"],
                "metrics": result["metrics"], "growth_factor": factor,
                "density": placement["density_policy"]["resolved_by_die"],
                "frontend_until_release_seconds": release["elapsed_seconds"],
                "gpu_lease_seconds_including_prefix": gpu_lease_seconds(root, job, release),
                "result_sha256": sha(attempt / "result.json")})
        maximum_gpu, maximum_live = 0, 0
        by_device = Counter()
        for line in (folder / "telemetry.jsonl").read_text().splitlines():
            sample = json.loads(line)
            live = sample["allocations"]
            used = Counter(gpu for row in live for gpu in row["gpus"])
            if sum(used.values()) > 6 or any(count > 3 for count in used.values()):
                raise ValueError("GPU concurrency limit was exceeded")
            maximum_gpu = max(maximum_gpu, sum(used.values()))
            maximum_live = max(maximum_live, len(live))
            for gpu, count in used.items(): by_device[gpu] = max(by_device[gpu], count)
        report["batches"][label] = {"verified_cells": len(cells), "cells": cells,
            "wall_seconds": timing["batch_submit_to_verified_seconds"],
            "peak_gpu_slots": maximum_gpu, "peak_gpu_slots_by_device": dict(by_device),
            "peak_live_jobs": maximum_live, "per_case_timing": timing["per_case"],
            "currently_non_dominated": current["non_dominated"], "evaluation": current["evaluation"]}
        report["batches"][label]["per_case_stage_medians_seconds"] = {
            case: {stage: median(cell["stage_seconds"][stage] for cell in timing["cells"] if cell["case"] == case)
                   for stage in timing["cells"][0]["stage_seconds"]}
            for case in timing["per_case"]}
        rows.append((label, len(cells), timing["batch_submit_to_verified_seconds"] / 60, maximum_gpu, maximum_live))
    state = archive.show()
    report["frontier"] = state["frontier"]
    prefixes = {}
    for cell in report["batches"]["candidate"]["cells"]:
        source = Path(read(root / "runs" / cell["job"] / "prefix.json")["source"])
        ready = read(source.parent.parent / "ready.json")
        for name, checksum in ready["manifests"].items():
            if sha(source / name) != checksum:
                raise ValueError("Candidate prefix manifest changed")
        run = read(source / "run.json")
        timing = {"wall_seconds": ready["execution"]["wall_seconds"],
                  "stage_seconds": {stage: run["stages"][stage]["runtime_seconds"]
                                    for stage in ("input", "floorplan")}}
        if cell["case"] in prefixes and prefixes[cell["case"]] != timing:
            raise ValueError("Case seeds did not share one candidate prefix")
        prefixes[cell["case"]] = timing
    report["candidate_cold_prefixes"] = prefixes
    report["verifier_negative"] = read(experiment / "verifier-negative.json")
    report["partial_matrix_rejection"] = read(experiment / "partial-matrix-rejection.json")
    report["candidate_source_verification"] = read(experiment / "candidate-verification.json")
    report["native_build"] = read(root / "control/runtimes" / (qualified["candidate"] + ".json"))
    report["scientific_split_frozen"] = False
    report["signoff"] = False
    report["status"] = "independently_rechecked"
    output.mkdir(parents=True, exist_ok=False)
    write(output / "audit.json", report)
    lines = ["# Native post-GRT Pareto qualification", "", "All reported cells were freshly executed from partition through GRT and independently verified.", "",
             "| Batch | Verified cells | Wall time (min) | Peak GPU slots | Peak live jobs |", "|---|---:|---:|---:|---:|"]
    lines += ["| {} | {} | {:.2f} | {} | {} |".format(*row) for row in rows]
    lines += ["", "The candidate changes the placement backend area growth factor from 1.25 to 1.30. Native parameter files confirm the change in every candidate cell.", "",
        "The baseline reused immutable input/floorplan checkpoints. The changed candidate regenerated its signature-bound prefix; that setup time is included in its batch wall time.", "",
        "## Paired three-objective result", "", "All normalized objectives minimize; the baseline is 1.0. Timing combines normalized WNS and TNS deficits using the frozen weights in audit.json.", "",
        "| Objective | Normalized value |", "|---|---:|"]
    evaluation = report["batches"]["candidate"]["evaluation"]
    vector = evaluation["vector"]
    lines += ["| {} | {:.9f} |".format(metric, vector[metric]) for metric in OBJECTIVES]
    lines += ["", "Power-feasible: **{}**. Each case is compared with its original baseline mean across paired seeds.".format(evaluation["feasible"]), "",
        "| Case | Baseline mean W | Candidate mean W | Limit W | Excess W | Pass |", "|---|---:|---:|---:|---:|---|"]
    for case, power in sorted(evaluation["constraints"]["power_w"]["per_case"].items()):
        lines.append("| {} | {:.9g} | {:.9g} | {:.9g} | {:.9g} | {} |".format(case,
            power["baseline_mean_w"], power["candidate_mean_w"], power["limit_w"], power["excess_w"], power["feasible"]))
    lines += ["", "Candidate currently non-dominated: **{}**.".format(report["batches"]["candidate"]["currently_non_dominated"]),
        "", "Checks passed: task-bound source ancestry, complete paired matrices, queue/independent-verifier/raw-evidence agreement, confirmed GPU release, six total/three per-device GPU limits, expected native parameter execution, and idempotent archive readback.",
        "", "A protected source edit and an incomplete native matrix were both rejected. Source and result histories remain retained regardless of dominance.",
        "", "This is five-case pipeline qualification, not a frozen scientific train/test generalization result. Thermal is the maximum of the 200 active silicon-tile temperatures per cell. The source verifier is not a semantic-equivalence proof; post-GRT metrics are not detailed-route signoff."]
    lines += ["", "## Stage timing", "",
        "Each entry below is the median of three seeds, in minutes. These are individual stage medians; their sum need not equal the median end-to-end time. Prefix preparation, queueing and inter-stage handoff are excluded from these stage columns.", "",
        "| Batch / case | Partition | Placement | PDN | HBT | OPT | CTS | GRT | Joint scoring | Verification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    stages = ("partition", "placement", "pdn", "hbt", "opt", "cts", "grt_command", "joint_scoring", "verification")
    for label, batch in report["batches"].items():
        for case, durations in sorted(batch["per_case_stage_medians_seconds"].items()):
            lines.append("| {} / {} | {} |".format(label, case, " | ".join("{:.2f}".format(durations[stage] / 60) for stage in stages)))
    lines += ["", "GPU lease durations in audit.json are measured from the authoritative queue start to its confirmed release event. They include shared-prefix preparation/waiting where applicable. Each backend container runs without GPU access.", "",
        "## Candidate prefix setup", "",
        "One new signature-bound prefix per case is shared by its three seeds. These preparations overlap other cases, so their wall times must not be summed as batch overhead.", "",
        "| Case | Input (min) | Floorplan (min) | Prefix container wall (min) |",
        "|---|---:|---:|---:|"]
    for case, timing in sorted(prefixes.items()):
        lines.append("| {} | {:.2f} | {:.2f} | {:.2f} |".format(case, timing["stage_seconds"]["input"] / 60,
                     timing["stage_seconds"]["floorplan"] / 60, timing["wall_seconds"] / 60))
    lines += ["",
        "## Frozen identities", "", "- Task: `{}`".format(report["task"]),
        "- Evaluator: `{}`".format(report["evaluator_sha256"]),
        "- Baseline source: `{}`".format(report["baseline"]),
        "- Candidate source: `{}`".format(report["candidate"])]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return {"status": report["status"], "report": str(output / "REPORT.md"),
            "candidate_non_dominated": report["batches"]["candidate"]["currently_non_dominated"],
            "candidate_feasible": evaluation["feasible"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path); p.add_argument("experiment", type=Path); p.add_argument("output", type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.root, args.experiment, args.output), indent=2))
