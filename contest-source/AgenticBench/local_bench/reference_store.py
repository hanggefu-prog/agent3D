"""Append-only reference qualifications; ADOPT never dispatches an evaluator.

The store is owned by the trusted controller, outside candidate-writable mounts.
Publication and adoption both call a trusted producer validator. A receipt's
`qualified` flag or its self-consistent hash is never qualification by itself.
Physical data stay at their original location and are pinned, not rewritten.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time


SCHEMA = "agentic3d-reference-store-v1"
POLICIES = ("ADOPT", "EXTEND", "FRESH")
ORIGINS = ("ARCHIVE", "EXTEND", "FRESH")
CONTRACT_KEYS = {"case", "endpoint", "task_digest", "evaluator_digest", "frozen_inputs", "physical"}
ID_RE = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def policy(value):
    if value not in POLICIES:
        raise ValueError("reference policy must be ADOPT, EXTEND or FRESH")
    return value


def _regular(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Reference evidence must use canonical absolute non-symlink paths")
    if not path.is_file():
        raise ValueError("Missing reference evidence: " + str(path))
    return path


def validate_contract(contract):
    if not isinstance(contract, dict) or set(contract) != CONTRACT_KEYS:
        raise ValueError("Reference contract fields are incomplete or unknown")
    for name in ("case", "endpoint"):
        if not isinstance(contract[name], str) or not contract[name]:
            raise ValueError("Missing reference contract " + name)
    for name in ("task_digest", "evaluator_digest"):
        if not isinstance(contract[name], str) or not ID_RE.fullmatch(contract[name]):
            raise ValueError("Reference contract needs a full " + name)
    for name in ("frozen_inputs", "physical"):
        if not isinstance(contract[name], dict) or not contract[name]:
            raise ValueError("Reference contract needs explicit " + name)
    canonical(contract)
    return contract


def validate_evidence(evidence):
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Qualified reference needs nonempty physical evidence")
    seen = set()
    for record in evidence:
        if not isinstance(record, dict):
            raise ValueError("Invalid reference artifact record")
        path = _regular(record.get("path", ""))
        if str(path) in seen:
            raise ValueError("Duplicate reference artifact path")
        seen.add(str(path))
        size, sha = record.get("bytes"), record.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(sha, str) or not ID_RE.fullmatch(sha):
            raise ValueError("Invalid reference artifact size/hash")
        if path.stat().st_size != size or file_sha(path) != sha:
            raise ValueError("Reference artifact changed: " + str(path))


def _validate_payload(payload, contract):
    if not isinstance(payload, dict) or payload.get("qualified") is not True:
        raise ValueError("Producer did not independently qualify the reference")
    if payload.get("case") != contract["case"] or payload.get("endpoint") != contract["endpoint"]:
        raise ValueError("Reference case/endpoint differs from the task contract")
    if payload.get("source_contract") != contract["physical"]:
        raise ValueError("Reference physical contract differs; no implicit migration")
    if payload.get("frozen_inputs") != contract["frozen_inputs"]:
        raise ValueError("Reference frozen inputs differ from independently validated evidence")
    if not isinstance(payload.get("source_identity"), dict) or not payload["source_identity"]:
        raise ValueError("Reference lacks native producer identity")
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict) or not isinstance(baseline.get("raw_metrics"), dict) or not baseline["raw_metrics"]:
        raise ValueError("Reference lacks independently extracted raw metrics")
    if not isinstance(baseline.get("source_run"), str) or not Path(baseline["source_run"]).is_dir():
        raise ValueError("Reference lacks its physical source run")
    canonical(payload)  # Reject NaN/Inf, including nested diagnostics.
    validate_evidence(payload.get("evidence"))
    return payload


def _write_once(path, value):
    """No replacement, even after an uncertain or failed write."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def protect_destination(destination, producer, *, payload=None, record_path=None, _seen=()):
    """Reject writes into/above physical authorities, including EXTEND ancestors.

    This is an early path guard, not qualification. Untrusted metadata may only
    add protected paths; it can never authorize writing into a physical source.
    """
    destination = Path(destination).expanduser().absolute()
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("Reference destination cannot traverse symlinks")
    destination = destination.resolve()
    roots = []
    if isinstance(producer, dict):
        roots += [producer[k] for k in ("reference_root", "thermal_root", "run_root", "evaluation_root",
                                       "source_root", "benchmark_input")
                  if producer.get(k)]
        parent = producer.get("parent_record")
        if parent:
            path = _regular(Path(parent).expanduser())
            if str(path) in _seen or len(_seen) >= 8:
                raise ValueError("Cyclic or excessively deep reference lineage")
            raw = json.loads(path.read_text())
            protect_destination(destination, raw.get("producer"), payload=raw.get("qualification"),
                                record_path=path if record_path is not None else None,
                                _seen=(*_seen, str(path)))
    if isinstance(payload, dict):
        run = payload.get("baseline", {}).get("source_run")
        if run:
            roots.append(run)
    if record_path is not None:
        # Receipts are not store records and must not pollute the store.
        parent = Path(record_path).parent
        roots.append(parent.parent if parent.name == "records" else parent)
    for root in roots:
        root = Path(root).expanduser().resolve()
        if destination == root or root in destination.parents or destination in root.parents:
            raise ValueError("Reference destination must be disjoint from protected physical evidence: " + str(root))
    return destination


def validate_origin(origin, producer):
    if origin not in ORIGINS or not isinstance(producer, dict) or not producer:
        raise ValueError("Invalid reference publication origin/producer")
    kind = producer.get("kind")
    # Custom trusted fixture producers are possible at this low-level API, but
    # known production adapters cannot relabel their provenance on load.
    if kind in {"frozen-20260923-v1", "native-evaluation-v1", "native-local-placement-v1"}:
        if (origin == "ARCHIVE") != (kind == "frozen-20260923-v1"):
            raise ValueError("Reference origin differs from its producer")
        if (origin == "EXTEND") != bool(producer.get("parent_record")):
            raise ValueError("Only EXTEND can name a parent reference")
        if kind == "native-local-placement-v1" and origin != "FRESH":
            raise ValueError("Local placement can publish only a FRESH native baseline")
    if origin == "EXTEND" and not producer.get("parent_reference_id"):
        raise ValueError("EXTEND publication requires a qualified parent reference")


def publish_reference(store, *, contract, origin, producer, validate):
    """Seal one independently qualified producer for subsequent ADOPT calls.

    `validate` is supplied by trusted orchestration, never by a candidate. It
    must re-read native artifacts and enforce the selected producer profile.
    FRESH/EXTEND execution is deliberately outside this function.
    """
    validate_contract(contract)
    validate_origin(origin, producer)
    protect_destination(store, producer)
    payload = _validate_payload(validate(producer), contract)
    body = {"schema": SCHEMA, "contract": contract, "origin": origin,
            "producer": producer, "qualification": payload}
    reference_id = digest(body)
    store = protect_destination(store, producer, payload=payload)
    # Do not allow the candidate/physical source run to contain its authority.
    run = Path(payload["baseline"]["source_run"]).resolve()
    if store == run or run in store.parents or store in run.parents:
        raise ValueError("Reference store and physical run must be disjoint")
    # An otherwise safe store can contain a pre-existing child symlink. Check
    # the actual write directory and file, not just the store's outer path.
    records = protect_destination(store / "records", producer, payload=payload)
    records.mkdir(parents=True, exist_ok=True)
    path = protect_destination(records / (reference_id + ".json"), producer, payload=payload)
    record = {**body, "reference_id": reference_id,
              "published_at": datetime.now(timezone.utc).isoformat()}
    if path.exists():
        # Identical qualification is idempotent, not a new physical attempt.
        existing = json.loads(_regular(path).read_text())
        if {k: existing.get(k) for k in body} != body or existing.get("reference_id") != reference_id:
            raise ValueError("Existing reference record conflicts with its content identity")
        return path
    _write_once(path, record)
    path.chmod(0o444)
    return path


def load_reference(record_path, *, contract, validate):
    """Revalidate a sealed result. Never run, retry, extend or select a fallback."""
    validate_contract(contract)
    path = _regular(Path(record_path).expanduser())
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("Unsupported reference record")
    body = {key: record.get(key) for key in ("schema", "contract", "origin", "producer", "qualification")}
    if record.get("reference_id") != digest(body) or path.name != record["reference_id"] + ".json":
        raise ValueError("Reference record content identity differs")
    if record["contract"] != contract:
        raise ValueError("Reference task/physical contract mismatch")
    validate_origin(record["origin"], record["producer"])
    _validate_payload(record["qualification"], contract)
    current = _validate_payload(validate(record["producer"]), contract)
    if canonical(current) != canonical(record["qualification"]):
        raise ValueError("Producer requalification differs from the sealed reference")
    return record


def find_reference(store, *, contract, reference_id=None):
    """Discovery only; callers must use load_reference before scoring."""
    validate_contract(contract)
    if reference_id is not None and (not isinstance(reference_id, str) or not ID_RE.fullmatch(reference_id)):
        raise ValueError("Invalid reference ID")
    records = Path(store).expanduser().resolve() / "records"
    paths = [records / (reference_id + ".json")] if reference_id else sorted(records.glob("*.json"))
    matches = []
    for path in paths:
        if not path.exists():
            continue
        record = json.loads(_regular(path).read_text())
        if record.get("contract") == contract:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("ADOPT needs exactly one matching qualified reference (found {}); "
                         "select an explicit ID or qualify a reference; no automatic FRESH/EXTEND".format(len(matches)))
    return matches[0]


def adopt_reference(record_path, *, contract, validate, output):
    """Write a new adoption receipt, with honest historical/current accounting."""
    started = time.monotonic()
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Adoption attempt already exists; inspect it instead of retrying")
    raw = json.loads(_regular(Path(record_path).expanduser()).read_text())
    output = protect_destination(output, raw.get("producer"), payload=raw.get("qualification"),
                                 record_path=record_path)
    output.mkdir(parents=True, exist_ok=False)
    attempt = {"schema": SCHEMA, "operation": "ADOPT", "record": str(record_path),
               "started_at": datetime.now(timezone.utc).isoformat(), "contract": contract}
    _write_once(output / "attempt.json", attempt)
    try:
        record = load_reference(record_path, contract=contract, validate=validate)
        receipt = {**attempt, "status": "qualified", "reference_origin": "adopted",
                   "original_origin": record["origin"], "reference_id": record["reference_id"],
                   "record_sha256": file_sha(record_path), "qualification": record["qualification"],
                   "verification_wall_seconds": time.monotonic() - started,
                   "historical_native_runtime": record["qualification"]["baseline"].get("native_runtime_historical"),
                   "new_reference_native_attempts": 0, "model_calls": 0, "candidate_finals": 0}
        _write_once(output / "reference_adoption.json", receipt)
        return receipt
    except Exception as error:
        _write_once(output / "failure.json", {**attempt, "status": "failed",
                    "error": type(error).__name__ + ": " + str(error),
                    "verification_wall_seconds": time.monotonic() - started,
                    "new_reference_native_attempts": 0, "model_calls": 0, "candidate_finals": 0})
        raise
