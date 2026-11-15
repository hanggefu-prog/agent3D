"""Run bounded native input audits before the engineering experiment.

This diagnoses the trusted interface and never publishes qualification.json,
calls a model, modifies a reference, or consumes a candidate final.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from .native_snapshot import snapshot
from .legalization import load_snapshot, restoration
from .protocol import write_new, now, read_json, sha


def release_image(release_path):
    record = read_json(release_path)
    if record.get("kind") == "engineering-source-release-v2":
        # Share the current source, installed-image and runtime-audit checks
        # with development. A v2 release is not a legacy fixed-revision record.
        from .prepare_development import verify_release
        _, image = verify_release(release_path)
        return image
    if record.get("kind") != "agenticbench-server-release-v1":
        raise ValueError("Unknown installed release contract")
    audit = record["runtime"]["equivalence"]
    if sha(audit["path"]) != audit["sha256"]:
        raise ValueError("Installed runtime equivalence audit changed")
    image = record["runtime"]["actual_image"]
    observed = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    if observed != image:
        raise ValueError("Installed image identity mismatch")
    return image


def audit_inputs(*, reference_root, platform, image, cases, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "dispatch.json", {"kind": "engineering-native-input-audit-v1",
        "started_at": now(), "cases": cases, "image": image, "model_calls": 0, "candidate_finals": 0})
    jobs = [(case, stage, die) for case in cases for stage in ("60_opt", "70_cts") for die in ("bottom", "top")]
    def execute(job):
        case, stage, die = job
        run = Path(reference_root) / case / "run"
        result = snapshot(run=run, platform=platform, image=image, stage=stage, die=die,
                          clock_sdc=run / "60_opt" / die / "design.sdc", output=output / case / stage / die)
        row = {"case": case, "stage": stage, "die": die, "status": result["status"]}
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(execute, jobs))
    comparisons = []
    for case in cases:
        for die in ("bottom", "top"):
            try:
                _, _, before = load_snapshot(output / case / "60_opt" / die)
                _, _, after = load_snapshot(output / case / "70_cts" / die)
                _, removed_clock = restoration(before, after)
                row = {"case": case, "die": die, "status": "compatible_with_protected_normalization",
                       "original_clock_cells": sum(r["clock_cell"] for r in before.values()),
                       "cts_clock_cells": sum(r["clock_cell"] for r in after.values()),
                       "removed_original_clock_cells": removed_clock}
            except (ValueError, KeyError, OSError) as exc:
                row = {"case": case, "die": die, "status": "blocked", "reason": str(exc)}
            comparisons.append(row)
    report = {"completed_at": now(), "snapshots": results, "comparisons": comparisons,
              "model_calls": 0, "candidate_finals": 0, "task_qualification": False}
    write_new(output / "result.json", report)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("reference_root", "platform", "output"):
        parser.add_argument("--" + field.replace("_", "-"), required=True)
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--image")
    runtime.add_argument("--release", help="Host's installed release.json with verified runtime audit")
    parser.add_argument("--cases", nargs="+", required=True, choices=("bp_fe", "bp", "ariane133", "swerv_wrapper"))
    args = parser.parse_args(argv)
    if len(args.cases) != len(set(args.cases)):
        parser.error("Repeated cases are not allowed")
    values = vars(args)
    release = values.pop("release")
    if release:
        values["image"] = release_image(release)
    report = audit_inputs(**values)
    return 0 if all(r["status"] == "succeeded" for r in report["snapshots"]) and all(r["status"] != "blocked" for r in report["comparisons"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
