#!/usr/bin/env python3
"""Import the immutable 2026-09-23 physical prefix without running EDA.

The original input, floorplan and partition are copied into a new input bundle.
Only copied run-owned design-parameter paths and their manifest hashes change.
Historical producer provenance is retained; this is not a scoring evaluation or
evidence that the current source produced those physical artifacts.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_bench.benchmark_input import (
    MANIFEST_NAME, SCHEMA_VERSION, _refresh_adopted_stage_manifest, _sha256,
    _validate_stage_manifest, load_benchmark_input_profile,
)
from local_bench.placement_baseline import validate_tp_gnn_partition
from local_bench import reference_baseline
from local_bench.workspace import copy_upstream_stages


STAGES = {"input": "00_input", "floorplan": "10_floorplan", "partition": "20_partition"}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _source_identity(source: Path) -> dict:
    """Use read-only Git commands, never a Python/tool qualification process."""
    if not source.is_dir() or not (source / "open3dbench/__main__.py").is_file():
        raise ValueError("Source must be an isolated Open3DBench checkout")
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(source), *args], env=environment,
                              check=True, capture_output=True, text=True).stdout.strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
        raise ValueError("Source must be the root of its isolated Git checkout")
    if git("status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("Source checkout must be clean before reference adoption")
    return {"root": str(source), "revision": git("rev-parse", "HEAD"), "dirty": False}


def _require_disjoint(left: Path, right: Path, label: str) -> None:
    if left == right or left in right.parents or right in left.parents:
        raise ValueError(label + " must be disjoint directories")


def _validate_native_prefix(run: Path, case: str) -> tuple[dict, dict]:
    ledger_path = run / "run.json"
    if ledger_path.is_symlink():
        raise ValueError("Frozen run ledger must not be a symlink")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict) or ledger.get("design") != case:
        raise ValueError("Frozen run ledger has the wrong design")
    manifests = {}
    for stage, directory in STAGES.items():
        root = run / directory
        if root.is_symlink():
            raise ValueError("Frozen stage directory must not be a symlink: " + directory)
        # Both files are written only in the copy. Refuse inherited links that
        # could make copy/rebase modify the immutable archive itself.
        for filename in ("stage_manifest.json", "design_params.json"):
            if (root / filename).is_symlink():
                raise ValueError("Frozen writable metadata must not be a symlink: " + filename)
        manifests[stage] = _validate_stage_manifest(root / "stage_manifest.json", stage)
        record = ledger.get("stages", {}).get(stage)
        if not isinstance(record, dict) or record.get("status") != "success":
            raise ValueError("Frozen prefix lacks a successful root stage record: " + stage)
        if record.get("directory") != directory:
            raise ValueError("Frozen root stage directory differs: " + stage)
    for filename in ("partition_manifest.json", "tp_gnn_report.json"):
        path = run / "20_partition" / filename
        if path.is_symlink():
            raise ValueError("Native partition evidence must not be a symlink: " + filename)
    partition_root = run / "20_partition"
    timing_root = partition_root / "timing_input"
    timing_manifest = timing_root / "timing_manifest.json"
    if (timing_root.is_symlink() or timing_manifest.is_symlink()
            or not timing_manifest.is_file() or timing_manifest.stat().st_size == 0):
        raise ValueError("Legacy stage-20 timing manifest must be a nonempty regular, non-symlink file")
    try:
        timing_manifest.resolve(strict=True).relative_to(partition_root.resolve(strict=True))
    except ValueError as error:
        raise ValueError("Legacy stage-20 timing manifest escapes the partition stage") from error
    validate_tp_gnn_partition(run, expected_method="tp-gnn-flat", require_bound_report=True)
    return ledger, manifests


def adopt_reference_input(*, reference_root: Path, case: str, source: Path,
                          output: Path) -> dict:
    """Create a new qualified input bundle; never overwrite, resume or run EDA."""
    started = time.monotonic()
    archive = Path(reference_root).expanduser().resolve(strict=True)
    source = Path(source).expanduser().resolve(strict=True)
    raw_output = Path(output).expanduser()
    if raw_output.is_symlink() or raw_output.exists():
        raise FileExistsError("Reference input output already exists; choose a new attempt: " + str(raw_output))
    output = raw_output.resolve()
    for left, right, label in ((archive, source, "Archive and source"),
                                (archive, output, "Archive and output"),
                                (source, output, "Source and output")):
        _require_disjoint(left, right, label)
    if os.environ.get("OPEN3D_LINK_FROZEN_UPSTREAM") == "1":
        raise ValueError("Reference adoption requires independent copies, not hard-linked upstream files")
    source_identity = _source_identity(source)
    profile = load_benchmark_input_profile(reference_baseline.profile_path(case) / "profile.yaml")
    if profile.required_stages != STAGES or profile.partition_method != "tp-gnn-flat":
        raise ValueError("Reference adoption requires the explicit legacy three-stage flat profile")
    original_run = reference_baseline.validate_reference_root(archive, case)
    ledger, native_manifests = _validate_native_prefix(original_run, case)
    source_records = {stage: {
        "directory": directory,
        "manifest_sha256": _sha256(original_run / directory / "stage_manifest.json"),
        "artifact_count": len(native_manifests[stage]["artifacts"]),
    } for stage, directory in STAGES.items()}
    registry = reference_baseline.identity()
    adoption = {
        "baseline_id": reference_baseline.BASELINE_ID,
        "reference_id": registry["reference_id"],
        "kind": "frozen_reference_adoption",
        "reference_root": str(archive), "source_run": str(original_run),
        "reference_json_sha256": _sha256(archive / "reference.json"),
        "source_run_manifest_sha256": _sha256(original_run / "run.json"),
        "source_run_status": ledger.get("status"), "source_stages": source_records,
        "fixed_partition": True,
        "partition_assignment_sha256": registry["cases"][case]["partition_assignment_sha256"],
        "native_producer_preserved": True, "physical_stages_executed": [],
        "scoring_reference_qualified": False,
        "timing_features": "20_partition/timing_input (historical producer layout; no synthetic stage 15)",
    }
    # This is the only creation point. A later validation failure is retained
    # for diagnosis and cannot be retried in place by a second invocation.
    output.mkdir(parents=True, exist_ok=False)
    run = output / "run"
    copy_upstream_stages(original_run, run, list(STAGES))
    stages = {}
    for stage, directory in STAGES.items():
        manifest = run / directory / "stage_manifest.json"
        _refresh_adopted_stage_manifest(manifest, original_run)
        native = _validate_stage_manifest(manifest, stage)
        stages[stage] = {"directory": directory, "manifest_sha256": _sha256(manifest),
                         "artifact_count": len(native["artifacts"])}
    historical_config = original_run / "config.yaml"
    if historical_config.is_file():
        shutil.copyfile(historical_config, run / "config.yaml")
        adoption["source_config_sha256"] = _sha256(historical_config)
    _write_json(run / "run.json", {
        "design": case, "type": ledger.get("type"), "pdk": ledger.get("pdk"),
        "status": "prepared", "stages": {name: copy.deepcopy(ledger["stages"][name]) for name in STAGES},
        "reference_adoption": adoption,
    })
    importer_identity = {**source_identity, "script": str(Path(__file__).resolve()),
                         "script_sha256": _sha256(Path(__file__))}
    payload = {
        "schema_version": SCHEMA_VERSION, "benchmark_id": profile.profile_id,
        "description": profile.description, "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started, "run_dir": "run",
        "baseline_id": reference_baseline.BASELINE_ID, "import_kind": "frozen_reference_adoption",
        "reference_root": str(archive),
        "profile_sha256": _sha256(profile.root / "profile.yaml"), "config_sha256": _sha256(profile.config),
        "stages": stages, "reference_adoption": adoption,
        "current_importer_source": importer_identity,
        "environment": {
            "python": {"version": sys.version, "executable": sys.executable, "pyyaml": yaml.__version__},
            "python_minimum": profile.python_minimum,
            # Workspace adapters mount candidate source at this isolated alias.
            # The native producer revision remains in each copied manifest.
            "open3dbench": source_identity,
            "open3dbench_role": "current_importer_and_future_candidate_mount; not native_artifact_producer",
            "dreamplace_install": os.environ.get("PLACE3D_DREAMPLACE_INSTALL", "/opt/dreamplace"),
        },
        "command": [], "seed_run": str(original_run),
    }
    _write_json(output / MANIFEST_NAME, payload)
    # Validate the copied native report as well as every artifact and fixed
    # physical/partition fingerprint; never repair or relabel failed evidence.
    validation = reference_baseline.validate_reference_bundle(output, case, fixed_partition=True)
    (output / "prepare.log").write_text(
        "Imported the immutable 2026-09-23 input/floorplan/partition prefix.\n"
        "No physical tools or Python identity subprocess were executed.\n"
        "Original stage runtimes and producer provenance are historical, not import runtime.\n"
        "No scoring reference has been qualified by this import.\n", encoding="utf-8")
    return {"status": "ok", "manifest": str(output / MANIFEST_NAME), "run_dir": str(run),
            "benchmark_id": profile.profile_id, "reference_validation": validation}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(adopt_reference_input(reference_root=args.reference_root, case=args.case,
                                           source=args.source, output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
