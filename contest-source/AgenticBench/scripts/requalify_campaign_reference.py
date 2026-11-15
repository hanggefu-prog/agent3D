#!/usr/bin/env python3
"""Requalify a failed-registration timing result without any EDA/model dispatch.

The existing FRESH physical attempt retains its computation-source identity.
Only the trusted audit framework changes. The old attempt is read-only, and a
new, exclusive recovery directory records this authorized registration attempt.
This entry point has no evaluator, Docker transport, model or worker launcher.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from global_bench.model import load_global_task
from local_bench import reference_baseline as baseline
from local_bench.benchmark_input import validate_benchmark_input
from local_bench.reference_catalog import evaluator_digest, qualify_producer, register_producer, task_digest
from local_bench.reference_evaluated import expected_frozen_inputs, expected_source_contract
from local_bench.reference_store import file_sha, load_reference, protect_destination, _write_once
from scripts.workspace_framework_identity import validate_audit_framework
from scripts.workspace_task_selection import TIMING_TEN_TASK, validate_selected_task

CASES = ("bp_fe", "bp", "ariane133", "swerv_wrapper")
REGISTRATION_ERROR = "ValueError: Reference physical stage contains a symlink"


def _path(value, *, exists=True, directory=False):
    path = Path(value).expanduser().absolute()
    if ".." in path.parts:
        raise ValueError("Recovery paths must be canonical without parent traversal")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Recovery paths must not traverse symlinks")
    if exists and (not path.exists() or (directory and not path.is_dir()) or
                   (not directory and not path.is_file())):
        raise ValueError("Recovery evidence is not a regular file/directory: " + str(path))
    return path


def _read(path):
    return json.loads(_path(path).read_text())


def _assert_hashes(hashes):
    if any(file_sha(_path(path)) != expected for path, expected in hashes.items()):
        raise ValueError("Old failed attempt or frozen input changed during recovery")


def _accounting(qualified, published, publication_unknown=False):
    return {"build_dispatches": 0, "evaluator_container_dispatches": 0,
        "evaluator_container_dispatches_unknown": 0, "model_calls": 0,
        "generated_model_candidates": 0, "candidate_finals": 0,
        "candidate_budget_consumed": 0, "verifier_rejections": 0,
        "native_physical_attempts": 0, "native_physical_attempts_known": 0,
        "native_physical_attempts_unknown": 0, "successful_evaluations": 0,
        "improvements": 0, "api_latency_seconds": None,
        "registration_attempts": 1,
        "registry_publications": None if publication_unknown else int(published),
        "registry_publications_known": int(published),
        "registry_publications_unknown": int(publication_unknown),
        "successful_reference_qualifications": int(qualified),
        "historical_native_attempts_requalified": int(qualified),
        "scope": "New read-only registration operation only; historical physical work is not dispatched or counted again."}


def requalify(*, source, source_revision, audit_framework_revision, case, image,
              benchmark_input, existing_evaluation, old_summary, reference_store, output):
    if case not in CASES:
        raise ValueError("Unregistered recovery case")
    source = _path(source, directory=True)
    identity = validate_audit_framework(source, ROOT, source_revision=source_revision,
                                       framework_revision=audit_framework_revision)
    bundle = _path(benchmark_input, directory=True)
    evaluation = _path(existing_evaluation, directory=True)
    old_summary = _path(old_summary)
    old_attempt = old_summary.parent
    if evaluation != old_attempt / "reference/result":
        raise ValueError("Recovery evaluation must belong to the exact old failed attempt")
    output = _path(output, exists=False)
    store = _path(reference_store, exists=False)
    if output.exists():
        raise FileExistsError("Recovery output exists; no implicit retry, even after failure")
    if output == store or output in store.parents or store in output.parents:
        raise ValueError("Recovery output and reference store must be disjoint")
    protected = {"evaluation_root": str(old_attempt), "source_root": str(source),
                 "benchmark_input": str(bundle), "reference_root": str(ROOT.parent)}
    protect_destination(output, protected)
    protect_destination(store, protected)
    old_config_path = old_attempt / "run_config.json"
    config, previous = _read(old_config_path), _read(old_summary)
    expected = {"kind": "campaign-reference-only-v1", "case": case, "endpoint": "timing",
        "task_id": TIMING_TEN_TASK, "reference_policy": "FRESH", "source_commit": source_revision,
        "image": image, "benchmark_input": str(bundle), "search_dispatch": False,
        "candidate_finals": 0, "candidate_budget_consumed": 0, "model_calls": 0}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Old reference-only attempt identity differs from the requested recovery")
    if (previous.get("state") != "failed" or previous.get("error") != REGISTRATION_ERROR
            or previous.get("case") != case or previous.get("endpoint") != "timing"
            or previous.get("accounting", {}).get("registry_publications") != 0
            or (old_attempt / "authority.json").exists()):
        raise ValueError("Only the preserved, unpublished log-symlink registration failure can be recovered")
    task = load_global_task(ROOT / "global_tasks" / TIMING_TEN_TASK)
    validate_selected_task(task, "timing", 10)
    profile = task.objective.evaluator_profile
    validated = validate_benchmark_input(bundle, task_id=profile)
    # The input validator resolves profiles in the current trusted auditor.
    # Physical producer identity is bound separately by the pinned tree and
    # byte-identical profile contract, never by this returned absolute path.
    audit_profile = baseline.validate_audit_profile(case, identity)
    if Path(validated["config"]).resolve(strict=True) != (audit_profile / "config.yaml").resolve(strict=True):
        raise ValueError("Frozen bundle config differs from the trusted audit profile")
    baseline.validate_reference_bundle(bundle, case, audit_framework_identity=identity)
    baseline.validate_physical_prefix(Path(validated["run_dir"]), case)
    contract = {"case": case, "endpoint": "timing", "task_digest": task_digest(task.root),
        "evaluator_digest": evaluator_digest(),
        "frozen_inputs": expected_frozen_inputs(case, file_sha(bundle / "benchmark_input.json")),
        "physical": expected_source_contract(case, "timing", profile=profile,
                                            source_revision=source_revision, image=image)}
    old_contract = config.get("reference_selection", {}).get("contract", {})
    if (set(old_contract) != set(contract) or any(old_contract[key] != value for key, value in contract.items()
            if key != "evaluator_digest")):
        raise ValueError("Recovery changed the original task, frozen input or physical reference contract")
    tracked = [old_config_path, old_summary, bundle / "benchmark_input.json",
               evaluation / "evaluation-contract.json", evaluation / "feedback.json"]
    hashes = {str(_path(path)): file_sha(path) for path in tracked}
    producer = {"kind": "native-evaluation-v1", "run_root": str(evaluation / "run"),
                "evaluation_root": str(evaluation), "case": case, "endpoint": "timing", "profile": profile}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "requalification.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, clock = time.time(), time.monotonic()
        _write_once(output / "run_config.json", {"schema_version": 1,
            "kind": "campaign-existing-reference-requalification-v1", "case": case, "endpoint": "timing",
            "operation": "REQUALIFY_EXISTING_FRESH", "new_native_dispatch": False,
            "retry_policy": "none", "candidate_budget_consumed": 0,
            "producer": producer, "contract": contract, "reference_store": str(store),
            "old_failure": str(old_summary), "old_evidence_sha256": hashes,
            "old_evaluator_digest": old_contract["evaluator_digest"], "audit_framework_identity": identity})
        _write_once(output / "attempt.json", {"state": "qualifying", "started_at": started})
        result = {"state": "failed", "started_at": started, "case": case, "endpoint": "timing"}
        qualified, record_path, publication_requested = False, None, False
        try:
            payload = qualify_producer(producer)
            if (payload.get("qualified") is not True or payload.get("executed_native_stages") != []
                    or payload.get("source_contract") != contract["physical"]
                    or payload.get("frozen_inputs") != contract["frozen_inputs"]):
                raise ValueError("Independent qualification does not match the unchanged physical contract")
            _assert_hashes(hashes)
            if identity != validate_audit_framework(source, ROOT, source_revision=source_revision,
                                                    framework_revision=audit_framework_revision):
                raise ValueError("Audit/computation identity changed during recovery")
            # FRESH describes the preserved physical producer, not a new run.
            publication_requested = True
            record_path = register_producer(store, producer, origin="FRESH", contract=contract)
            record_sha = file_sha(record_path)
            record = load_reference(record_path, contract=contract, validate=qualify_producer)
            if record_sha != file_sha(record_path) or record.get("origin") != "FRESH":
                raise ValueError("Published authority changed during independent requalification")
            _assert_hashes(hashes)
            if identity != validate_audit_framework(source, ROOT, source_revision=source_revision,
                                                    framework_revision=audit_framework_revision):
                raise ValueError("Audit/computation identity changed after publication")
            authority = {"qualified": True, "independently_requalified": True,
                "record_path": str(record_path), "record": str(record_path), "record_sha256": record_sha,
                "reference_id": record["reference_id"], "contract": contract,
                "source_run": record["qualification"]["baseline"]["source_run"],
                "computation_source_revision": source_revision, "trusted_framework_revision": audit_framework_revision,
                "old_failure": str(old_summary), "new_native_attempts": 0,
                "historical_native_runtime": record["qualification"]["baseline"].get("native_runtime_historical")}
            _write_once(output / "authority.json", authority)
            result.update(state="qualified", authority=str(output / "authority.json"), **authority)
            qualified = True
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            if record_path is not None:
                result["published_record"] = str(record_path)
        result.update(finished_at=time.time(), wall_seconds=time.monotonic() - clock,
                      wall_seconds_scope="Read-only audit and registration; not native evaluation or API inference.",
                      accounting=_accounting(qualified, record_path is not None,
                                            publication_requested and record_path is None))
        _write_once(output / "summary.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "benchmark-input", "existing-evaluation", "old-summary", "reference-store", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-revision", "audit-framework-revision", "image"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    args = parser.parse_args(argv)
    value = requalify(**vars(args))
    print(json.dumps(value, sort_keys=True, indent=2), flush=True)
    return 0 if value["state"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
