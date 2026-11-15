"""Trusted dispatch for reference producers. No model, shell or EDA dispatch."""
from __future__ import annotations

import json
from pathlib import Path

from .reference_store import (adopt_reference, digest, file_sha, load_reference,
                              publish_reference, validate_contract)


def qualify_producer(producer):
    return _qualify_producer(producer, frozenset())


def _qualify_producer(producer, ancestors):
    if not isinstance(producer, dict):
        raise ValueError("Reference producer must be an object")
    kind = producer.get("kind")
    from scripts.workspace_latest_reference import KIND, THERMAL_KIND
    if kind in {KIND, THERMAL_KIND}:
        allowed = {"kind", "case", "benchmark_input", "run_root", "task_id"}
        if kind == THERMAL_KIND:
            allowed.add("thermal_root")
        if set(producer) != allowed:
            raise ValueError("Incomplete or unknown latest workspace producer fields")
        from scripts.workspace_latest_reference import qualify_producer as qualify_latest
        return qualify_latest(producer)
    if kind in {"native-local-placement-v1", "native-local-partition-v1"}:
        allowed = {"kind", "task_id", "case", "source_root", "evaluation_root", "benchmark_input"}
        if set(producer) != allowed:
            raise ValueError("Incomplete or unknown local native producer fields")
        if kind == "native-local-partition-v1":
            from .reference_local_partition import prepare_local_partition_reference as prepare
        else:
            from .reference_local_placement import prepare_local_placement_reference as prepare
        return prepare(Path(producer["evaluation_root"]),
            producer["task_id"], producer["case"], source_root=Path(producer["source_root"]),
            benchmark_input=Path(producer["benchmark_input"]))
    if kind == "frozen-20260923-v1":
        allowed = {"kind", "reference_root", "case", "endpoint", "thermal_root"}
        if set(producer) - allowed:
            raise ValueError("Unknown frozen archive producer fields")
        from .reference_archive import prepare_archive_reference
        return prepare_archive_reference(Path(producer["reference_root"]), producer["case"],
            producer["endpoint"], thermal_root=Path(producer["thermal_root"]) if producer.get("thermal_root") else None)
    if kind == "native-evaluation-v1":
        allowed = {"kind", "run_root", "case", "endpoint", "profile", "evaluation_root",
                   "parent_record", "parent_reference_id"}
        if set(producer) - allowed:
            raise ValueError("Unknown native evaluation producer fields")
        parent = None
        if producer.get("parent_record"):
            path = Path(producer["parent_record"]).expanduser().resolve(strict=True)
            if path in ancestors or len(ancestors) >= 8:
                raise ValueError("Cyclic or excessively deep reference lineage")
            raw = json.loads(path.read_text())
            if raw.get("reference_id") != producer.get("parent_reference_id"):
                raise ValueError("EXTEND parent ID mismatch")
            verified = load_reference(path, contract=raw["contract"],
                validate=lambda p: _qualify_producer(p, ancestors | {path}))
            parent = {**verified["qualification"], "reference_id": verified["reference_id"],
                      "record": str(path), "record_sha256": file_sha(path)}
        elif producer.get("parent_reference_id"):
            raise ValueError("EXTEND needs the qualified parent record, not just an ID")
        from .reference_evaluated import prepare_evaluated_reference
        return prepare_evaluated_reference(Path(producer["run_root"]), producer["case"],
            producer["endpoint"], profile=producer["profile"],
            evaluation_root=Path(producer["evaluation_root"]) if producer.get("evaluation_root") else None,
            parent_reference=parent)
    raise ValueError("Unsupported reference producer; no arbitrary validator or fallback is allowed")


def evaluator_digest():
    """Include independent validators and scoring semantics, not unrelated docs."""
    root = Path(__file__).resolve().parents[1]
    names = ("local_bench/reference_store.py", "local_bench/reference_catalog.py",
             "local_bench/reference_archive.py", "local_bench/reference_evaluated.py",
             "local_bench/reference_baseline.py", "local_bench/reference_20260923_identity.json",
             "local_bench/reference_placement_evidence.py", "local_bench/metrics.py",
             "local_bench/reference_local_placement.py",
             "local_bench/reference_local_partition.py",
             "local_bench/partition_assignment.py",
             "local_bench/post_placement_timing.py", "local_bench/placement_components.py",
             "local_bench/scoped_reference.py",
             "scripts/workspace_task_selection.py",
             "global_bench/verifiers/open3dbench_codesign_verifier.py")
    return digest({name: file_sha(root / name) for name in names})


def task_digest(task_dir):
    """Bind model-facing instructions as well as the formal task definition."""
    root = Path(task_dir).expanduser().resolve(strict=True)
    required = root / "task.yaml"
    if not required.is_file():
        raise ValueError("A reference consumer must be a versioned task directory")
    names = ("task.yaml", "config.yaml", "prompt.md", "instruction.md")
    files = {name: file_sha(root / name) for name in names if (root / name).is_file()}
    for path in sorted((root / "openevolve_templates").glob("*.txt")):
        files[path.relative_to(root).as_posix()] = file_sha(path)
    return digest(files)


def make_contract(payload, *, task_digest, evaluator_digest):
    """Attach a consumer to independently verified physical evidence.

    Runtime drivers must also check that their intended algorithm/config/inputs
    match this physical contract; catalog publication alone never launches one.
    """
    return validate_contract({"case": payload["case"], "endpoint": payload["endpoint"],
        "task_digest": task_digest, "evaluator_digest": evaluator_digest,
        "frozen_inputs": payload["frozen_inputs"], "physical": payload["source_contract"]})


def register_producer(store, producer, *, origin, contract):
    # Mode and origin are not freely interchangeable labels.
    if origin == "ARCHIVE" and producer.get("kind") != "frozen-20260923-v1":
        raise ValueError("ARCHIVE publication requires the frozen archive producer")
    if origin in {"FRESH", "EXTEND"} and producer.get("kind") not in {
            "native-evaluation-v1", "native-local-placement-v1", "native-local-partition-v1"}:
        raise ValueError("New native baseline publication requires actual evaluated-run evidence")
    if origin == "EXTEND" and producer.get("kind") in {"native-local-placement-v1", "native-local-partition-v1"}:
        raise ValueError("Local references have no missing physical suffix to extend")
    if (origin == "EXTEND") != bool(producer.get("parent_record")):
        raise ValueError("Only EXTEND can name a parent reference")
    return publish_reference(store, contract=contract, origin=origin, producer=producer, validate=qualify_producer)


def adopt(record, *, contract, output):
    return adopt_reference(record, contract=contract, output=output, validate=qualify_producer)
