#!/usr/bin/env python3
"""Trusted final artifact gate executed outside the candidate workspace.

This entry point intentionally does not import candidate-owned metric or
verification code. Stage execution/replay is prepared by the trusted global
bench runner; this gate checks the resulting canonical manifests before their
metrics can be admitted into Global Analyzer state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Dict

import yaml


FIXED_SEEDS = {
    "pins": 1,
    "partition": 1001,
    "placement": 1000,
    "routing": -1,
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submitted_file(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("unsafe submitted path: {}".format(relative))
    path = (root / value.as_posix()).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError("missing or unsafe submitted file: {}".format(relative))
    return path


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and finite_tree(item) for key, item in value.items())
    return False


STAGE_DIRS = {
    "input": "00_input",
    "floorplan": "10_floorplan",
    "timing_extract": "15_timing_extract",
    "partition": "20_partition",
    "placement": "30_placement",
    "pdn": "40_pdn",
    "hbt": "50_hbt",
    "opt": "60_opt",
    "cts": "70_cts",
    "global_routing": "75_global_route",
    "detailed_routing": "80_route",
    "report": "90_report",
    "thermal": "95_thermal",
}
LEGACY_STAGE_DIRS = {
    **{
        stage: directory
        for stage, directory in STAGE_DIRS.items()
        if stage not in {"global_routing", "detailed_routing"}
    },
    "routing": "80_route",
}


def validate_stage(run_root: Path, stage: str) -> None:
    directory = STAGE_DIRS.get(stage, LEGACY_STAGE_DIRS.get(stage))
    if directory is None:
        raise ValueError("unknown trusted stage {}".format(stage))
    stage_root = (run_root / directory).resolve()
    manifest_path = stage_root / "stage_manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("stage") != stage:
        raise ValueError("stage identity mismatch")
    if value.get("status") != "success" or not isinstance(value.get("signature"), dict):
        raise ValueError("stage is not a signed success")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("stage has no artifacts")
    for name, record in artifacts.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid artifact record {}".format(name))
        artifact = (stage_root / record["path"]).resolve()
        if stage_root not in artifact.parents or not artifact.is_file() or artifact.is_symlink():
            raise ValueError("stage artifact is missing or unsafe: {}".format(name))
        if artifact.stat().st_size != record.get("bytes") or sha256(artifact) != record.get("sha256"):
            raise ValueError("stage artifact hash mismatch: {}".format(name))


def validate_seed_contract(run_root: Path) -> None:
    config = yaml.safe_load((run_root / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("trusted config.yaml is not a mapping")
    observed = {}
    for section in FIXED_SEEDS:
        row = config.get(section)
        if not isinstance(row, dict) or "seed" not in row:
            raise ValueError("trusted config omits {}.seed".format(section))
        observed[section] = int(row["seed"])
    if observed != FIXED_SEEDS:
        raise ValueError(
            "benchmark seed contract mismatch: expected={} observed={}".format(
                FIXED_SEEDS, observed
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluated-run", required=True, type=Path)
    args = parser.parse_args()
    root = args.submission.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluated_run = args.evaluated_run.expanduser().resolve()
    manifest = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    records = {item["name"]: item for item in manifest.get("artifacts", [])}
    errors = []
    decoded: Dict[str, Any] = {}
    for name in ("run_manifest", "final_checkpoint", "metrics"):
        record = records.get(name)
        if not isinstance(record, dict):
            errors.append("missing artifact {}".format(name))
            continue
        try:
            path = submitted_file(root, str(record["submitted_path"]))
            if sha256(path) != record.get("sha256"):
                raise ValueError("hash mismatch")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not finite_tree(value):
                raise ValueError("artifact must be a finite JSON object")
            decoded[name] = value
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append("{}: {}".format(name, error))

    checkpoint = decoded.get("final_checkpoint", {})
    ready = any(
        checkpoint.get(marker) is True
        for marker in ("ready_for_route", "ready_for_extract", "ready_for_thermal", "complete")
    ) or (
        checkpoint.get("stage") == "thermal" and isinstance(checkpoint.get("thermal"), dict)
    )
    if not ready:
        errors.append("final checkpoint has no recognized readiness marker")
    metrics = decoded.get("metrics", {})
    if not metrics:
        errors.append("metrics artifact is empty")

    # Candidate artifacts are admitted only when they exactly match an
    # evaluator-owned run whose complete stage chain validates independently.
    try:
        trusted_run_manifest = json.loads(
            (evaluated_run / "run.json").read_text(encoding="utf-8")
        )
        if trusted_run_manifest.get("status") != "success":
            raise ValueError("trusted evaluation run is not successful")
        statuses = trusted_run_manifest.get("stages")
        if not isinstance(statuses, dict) or not statuses:
            raise ValueError("trusted evaluation run has no stage status")
        statuses = dict(statuses)
        split_routing = "detailed_routing" in statuses
        if split_routing:
            statuses.pop("routing", None)
        expected_stage_dirs = dict(STAGE_DIRS)
        # Runs released before step 15 remain admissible historical evidence.
        if "timing_extract" not in statuses:
            expected_stage_dirs.pop("timing_extract")
        if not split_routing:
            expected_stage_dirs.pop("global_routing")
            expected_stage_dirs.pop("detailed_routing")
            expected_stage_dirs["routing"] = "80_route"
        if set(statuses) != set(expected_stage_dirs):
            raise ValueError(
                "trusted evaluation stage set mismatch: missing={} unexpected={}".format(
                    sorted(set(expected_stage_dirs) - set(statuses)),
                    sorted(set(statuses) - set(expected_stage_dirs)),
                )
            )
        validate_seed_contract(evaluated_run)
        for stage, status in statuses.items():
            if isinstance(status, dict) and status.get("status") == "skipped":
                continue
            validate_stage(evaluated_run, stage)
        trusted_checkpoint = (
            evaluated_run / "95_thermal/checkpoint.json"
            if (evaluated_run / "95_thermal/checkpoint.json").is_file()
            else evaluated_run / "90_report/checkpoint.json"
        )
        trusted_metrics = evaluated_run / "90_report/metrics.json"
        trusted_files = {
            "run_manifest": evaluated_run / "run.json",
            "final_checkpoint": trusted_checkpoint,
            "metrics": trusted_metrics,
        }
        for name, trusted_path in trusted_files.items():
            record = records[name]
            submitted = submitted_file(root, str(record["submitted_path"]))
            if sha256(submitted) != sha256(trusted_path):
                raise ValueError("submitted {} differs from trusted evaluation".format(name))
        metrics = json.loads(trusted_metrics.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict) or not finite_tree(metrics):
            raise ValueError("trusted metrics are malformed")
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        errors.append("trusted evaluation: {}".format(error))

    result = {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "metrics": metrics if not errors else {},
        "provenance": {
            "task_id": manifest.get("task_id"),
            "trial_id": manifest.get("trial_id"),
            "base_revision": manifest.get("base_revision"),
        },
    }
    (output / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "reward.json").write_text(
        json.dumps({"reward": 1.0 if not errors else 0.0}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
