"""Explicit, immutable-prefix extension of a qualified native reference.

This module does not dispatch tools. It validates the parent and copies only
independently authenticated prefix files into a new run. Historical archives
need a separate stage-source bridge and are intentionally not accepted here.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from global_bench.model import load_global_task
from global_bench.verifiers import open3dbench_codesign_verifier as verifier
from local_bench.reference_catalog import qualify_producer, task_digest
from local_bench.reference_evaluated import ENDPOINTS, expected_source_contract
from local_bench.reference_store import file_sha, load_reference, protect_destination
from scripts.workspace_task_selection import VARIANTS, load_registered_task


def qualify_parent(record_path, target_contract, *, record_sha256=None):
    """Compare the complete current producer/input contract before any suffix."""
    path = Path(record_path).expanduser().absolute()
    if record_sha256 is not None and file_sha(path) != record_sha256:
        raise ValueError("EXTEND parent record changed after selection")
    raw = json.loads(path.read_text())
    producer = raw.get("producer", {})
    endpoint = raw.get("contract", {}).get("endpoint")
    target = target_contract["endpoint"]
    if producer.get("kind") != "native-evaluation-v1":
        raise ValueError("EXTEND requires a compatible native parent; historical source composition is unsupported")
    legal = {("timing", "cts_thermal"), ("timing", "routed_thermal"), ("cts_thermal", "routed_thermal")}
    if (endpoint, target) not in legal:
        raise ValueError("EXTEND must add a missing suffix beyond a qualified earlier endpoint")
    profile, _, objective, _ = ENDPOINTS[endpoint]
    framework = Path(__file__).resolve().parents[1]
    # A physical timing producer can be bound to either registered consumer
    # budget, but a parent never supplies an arbitrary task path or definition.
    parent_digest = raw.get("contract", {}).get("task_digest")
    tasks = [load_registered_task(task_id, objective=objective, framework=framework)
             for (kind, _), task_id in VARIANTS.items() if kind == objective]
    matching = [task for task in tasks if task_digest(task.root) == parent_digest]
    if len(matching) != 1:
        raise ValueError("EXTEND parent consumer is not a registered workspace task")
    task = matching[0]
    physical = target_contract["physical"]
    contract = {**target_contract, "endpoint": endpoint, "task_digest": task_digest(task.root),
        "physical": expected_source_contract(target_contract["case"], endpoint,
            profile=profile, source_revision=physical["source_revision"], image=physical["image"])}
    record = load_reference(path, contract=contract, validate=qualify_producer)
    return {"record": str(path), "record_sha256": file_sha(path), "record_value": record,
            "target_contract": target_contract,
            "start_stage": "cts" if endpoint == "timing" else "global_routing",
            "stages": list(verifier.TIMING_STAGES) + (["cts"] if endpoint == "cts_thermal" else [])}


def copy_qualified_prefix(parent, destination):
    """Copy verified physical bytes, never re-label an old stage or run it."""
    record = parent["record_value"]
    payload = record["qualification"]
    root = Path(payload["baseline"]["source_run"]).resolve(strict=True)
    destination = protect_destination(destination, record["producer"], payload=payload,
                                      record_path=parent["record"])
    if destination.exists():
        raise FileExistsError("EXTEND run already exists; failed or uncertain attempts cannot be retried")
    evidence = {row["path"]: row for row in payload["evidence"]}
    directories = {verifier.STAGE_DIRS[stage] for stage in parent["stages"]}
    inventory = []
    copies = []
    # Complete the validation plan before creating the target directory.
    for stage in parent["stages"]:
        manifest = root / verifier.STAGE_DIRS[stage] / "stage_manifest.json"
        if str(manifest) not in evidence or file_sha(manifest) != evidence[str(manifest)]["sha256"]:
            raise ValueError("EXTEND lacks an authenticated prefix manifest: " + stage)
        inventory.append({"stage": stage, "manifest_sha256": file_sha(manifest)})
    for path_string, row in evidence.items():
        path = Path(path_string)
        if not path.is_relative_to(root):
            continue
        relative = path.relative_to(root)
        if not relative.parts or relative.parts[0] not in directories:
            continue
        if (path.is_symlink() or any(p.is_symlink() for p in path.parents)
                or path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]):
            raise ValueError("EXTEND prefix changed after qualification: " + str(path))
        copies.append((path, relative, row))
    status_path = root / "run.json"
    bound = evidence.get(str(status_path))
    if not bound or file_sha(status_path) != bound["sha256"]:
        raise ValueError("EXTEND parent run manifest changed")
    status = json.loads(status_path.read_text())
    if any(status["stages"].get(stage, {}).get("status") != "success" for stage in parent["stages"]):
        raise ValueError("EXTEND requires every prefix stage to be successful")
    destination.mkdir(parents=True, exist_ok=False)
    for source, relative, row in copies:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if target.stat().st_size != row["bytes"] or file_sha(target) != row["sha256"]:
            raise ValueError("EXTEND prefix copy failed verification: " + str(relative))
    status["stages"] = {stage: status["stages"][stage] for stage in parent["stages"]}
    status["status"] = "prepared"
    status.pop("error", None)
    (destination / "run.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return inventory
