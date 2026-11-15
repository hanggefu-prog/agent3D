#!/usr/bin/env python3
"""Produce one FRESH campaign authority, without any model or search dispatch.

The registered ten-round timing task owns the global reference contract. Local
tasks reuse their exact initial-program producer. Success requires independent
native-artifact requalification after publication, never just process exit.
Run with --batch-root <release-root>/runs to share the existing GPU locks.
Existing outputs are refused, including failed or uncertain attempts.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import yaml

from agentic3d.knowledge import write_json
from global_bench.model import load_global_task
from local_bench import reference_baseline as baseline
from local_bench.benchmark_input import validate_benchmark_input
from local_bench.model import load_candidate, load_task, resolve_task_path
from local_bench.program_bridge import initial_program_text, materialize_program
from local_bench.reference_catalog import qualify_producer
from local_bench.reference_store import file_sha, load_reference, protect_destination
from scripts.local_reference import LocalPlacementReference
from scripts.run_local_task_smoke import DockerEvaluationTransport, _successful
from scripts.run_placement_interface_qualification import QualificationRuntime
from scripts.workspace_codesign_evaluator import WorkspaceEvaluator, endpoint_for_task
from scripts.workspace_reference import WorkspaceReference
from scripts.workspace_task_selection import TIMING_TEN_TASK, validate_selected_task

TASKS = {
    "timing": TIMING_TEN_TASK,
    "placement_dynamic": "placement_dynamic_parameters_reference_20260923",
    "placement_source": "placement_preconditioner_source_reference_20260923",
    "partition": "tp_gnn_learning_objective_reference_20260923",
}
CASES = ("bp_fe", "bp", "ariane133", "swerv_wrapper")


def _read_optional(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def _accounting(endpoint, output, transport, qualified):
    """Unknown native starts stay unknown; Docker dispatch is not native proof."""
    value = {"model_calls": 0, "generated_model_candidates": 0, "candidate_finals": 0,
             "build_dispatches": 0, "successful_reference_qualifications": int(qualified),
             "improvements": 0, "api_latency_seconds": None}
    if transport is not None:
        rows = transport.attempts
        value["evaluator_container_dispatches"] = sum(r.get("evaluator_dispatch_attempted") is True for r in rows)
        value["evaluator_container_dispatches_unknown"] = sum(r.get("evaluator_dispatch_attempted") is None for r in rows)
        known = sum(r.get("native_flow_started") is True for r in rows)
        unknown = sum(r.get("native_flow_started") is None for r in rows)
        value["execution_attempts"] = rows
        result = _read_optional(output / "reference/result/evaluation_result.json")
        value["native_runtime_seconds"] = [r.get("wall_seconds") for r in
            result.get("artifacts", {}).get("native_flow_execution", [])]
    else:
        result = _read_optional(output / "reference/result/feedback.json")
        execution = result.get("execution", {})
        ledger = _read_optional(output / "reference/result.workspace-attempt.json")
        phases = execution.get("command_phases", ledger.get("command_phases", {}))
        value["evaluator_container_dispatches"] = sum(r.get("dispatch_attempted") is True for r in phases.values())
        value["evaluator_container_dispatches_unknown"] = sum(r.get("dispatch_attempted") is None for r in phases.values())
        flow_dispatched = any(r.get("dispatch_attempted") is True for name, r in phases.items()
                              if name in {"frontend", "backend", "thermal"})
        known, unknown = int(qualified), int(not qualified and flow_dispatched)
        value["execution"] = execution or {"command_phases": phases}
        value["timing"] = result.get("timing", {})
    verification = _read_optional(output / "reference/result/verification/verification.json")
    value["verifier_rejections"] = int(verification.get("passed") is False)
    value.update(native_physical_attempts_known=known, native_physical_attempts_unknown=unknown,
                 native_physical_attempts=None if unknown else known,
                 native_attempt_definition="Reference evaluations with native execution evidence; not individual solver subprocesses.")
    return value


def _requalify(selection):
    """Pin exact record bytes and invoke the authoritative read-only producer."""
    path = Path(selection.record_path)
    before = file_sha(path)
    record = load_reference(path, contract=selection.contract, validate=qualify_producer)
    if before != file_sha(path) or before != selection.record_sha256:
        raise ValueError("Published reference record changed during independent requalification")
    if record.get("origin") != "FRESH":
        raise ValueError("Campaign reference-only job must publish a native FRESH authority")
    return {"record": str(path), "record_sha256": before, "reference_id": record["reference_id"],
            "contract": selection.contract, "source_run": record["qualification"]["baseline"]["source_run"],
            "qualified": True, "independently_requalified": True}


def run_reference(runtime, *, endpoint, case, benchmark_input, output, reference_store):
    if endpoint not in TASKS or case not in CASES:
        raise ValueError("Unregistered campaign endpoint or case")
    if (runtime.cpus, runtime.memory_gb, runtime.min_gpu_free_mb) != (16, 48, 2048):
        raise ValueError("Keep the authorized 16 CPU / 48 GiB cap and 2 GiB GPU guard")
    # This explicit common parent prevents accidentally making a private set of
    # locks below a campaign directory while older experiments use runs/locks.
    if runtime.batch.name != "runs":
        raise ValueError("--batch-root must be the release's shared runs directory")
    raw_output = Path(output).expanduser().absolute()
    if raw_output.exists() or any(p.is_symlink() for p in (raw_output, *raw_output.parents)):
        raise FileExistsError("Reference output exists or traverses a symlink; no implicit retry")
    output, bundle = runtime.contained(raw_output), runtime.contained(benchmark_input)
    raw_store = Path(reference_store).expanduser().absolute()
    if any(p.is_symlink() for p in (raw_store, *raw_store.parents)):
        raise ValueError("Reference store cannot traverse symlinks")
    store = runtime.contained(raw_store)
    if output == runtime.batch:
        raise ValueError("Reference output cannot be the shared runs directory")
    runtime.reference_baseline = baseline.BASELINE_ID
    runtime.reference_policy, runtime.reference_record = "FRESH", None
    runtime.reference_store = store
    task_id = TASKS[endpoint]
    if endpoint == "timing":
        spec = load_global_task(runtime.framework / "global_tasks" / task_id)
        validate_selected_task(spec, "timing", 10)
        runtime.workspace_rounds, runtime.workspace_task_digest = 10, spec.digest()
        input_kind = endpoint_for_task("timing", spec)["profile"]
    else:
        spec = load_task(resolve_task_path(runtime.framework / "local_tasks", task_id))
        input_kind = spec.task_id
    validated = validate_benchmark_input(bundle, task_id=input_kind)
    config = Path(validated["config"]).resolve(strict=True)
    recipe = yaml.safe_load(config.read_text())
    if recipe.get("design") != case or config != (runtime.profile(case) / "config.yaml").resolve():
        raise ValueError("Frozen bundle/config does not match the explicitly selected case")
    aligned = baseline.validate_reference_bundle(bundle, case, fixed_partition=endpoint.startswith("placement_"))
    runtime.reference_root = Path(aligned["reference_root"])
    frozen = runtime.contained(Path(validated["run_dir"]))
    producer_paths = {"source_root": str(runtime.source), "benchmark_input": str(bundle),
                      "run_root": str(frozen), "reference_root": aligned["reference_root"],
                      "thermal_root": aligned.get("original_root")}
    protect_destination(raw_output, producer_paths)
    protect_destination(store, {**producer_paths, "evaluation_root": str(output)})
    if store == output or store in output.parents or output in store.parents:
        raise ValueError("Reference store and reference attempt must be disjoint")
    if endpoint == "timing":
        selection = WorkspaceReference(runtime, case, "timing", bundle, spec)
    else:
        spec = replace(spec, base_config=config)
        selection_class = LocalPlacementReference
        if endpoint == "partition":
            # Fail closed before output creation if the partition producer is
            # not present in the frozen campaign snapshot.
            from scripts.local_reference import LocalPartitionReference
            selection_class = LocalPartitionReference
        selection = selection_class(runtime, spec, case, bundle, output,
            selected_policy="FRESH", store=store)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "reference-only.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, started_clock = time.time(), time.monotonic()
        contract = {"schema_version": 1, "kind": "campaign-reference-only-v1", "case": case,
                    "endpoint": endpoint, "task_id": task_id, "reference_policy": "FRESH",
                    "reference_selection": selection.to_dict(), "source_commit": runtime.source_commit,
                    "image": runtime.image, "benchmark_input": str(bundle),
                    "shared_lock_root": str(runtime.batch / "locks"), "retention": "full",
                    "model_calls": 0, "candidate_finals": 0, "search_dispatch": False,
                    "retry_policy": "none", "candidate_budget_consumed": 0,
                    "resources": {"cpus": 16, "memory_gb": 48, "gpus": runtime.gpus,
                                  "min_gpu_free_mb": 2048, "min_data_free_gib": 64}}
        write_json(output / "run_config.json", contract)
        summary = {"state": "qualifying", "started_at": started, "case": case, "endpoint": endpoint}
        write_json(output / "summary.json", summary)
        transport, qualified = None, False
        try:
            evaluation = output / "reference/result"
            if endpoint == "timing":
                result = selection.obtain(WorkspaceEvaluator(runtime, case, "timing", bundle, spec), evaluation)
                score = result.get("metrics", {}).get("combined_score")
                if result.get("success") is not True or type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError("Native timing reference did not pass its authoritative evaluator")
            else:
                candidate = output / "reference/candidate"
                candidate.mkdir(parents=True)
                initial = initial_program_text(spec, load_candidate(spec.initial_candidate))
                write_json(candidate / "candidate.json", materialize_program(spec, initial, runtime.source))
                runtime.local_reference_selection = runtime.adopted_reference_run = None
                runtime.reference_purpose_receipt = selection.prepare_fresh(evaluation, candidate)
                transport = DockerEvaluationTransport(runtime)
                try:
                    result = transport(spec.root, candidate, runtime.source, evaluation,
                        baseline_run=frozen, base_config=config, timeout_seconds=21600)
                finally:
                    runtime.reference_purpose_receipt = None
                if not _successful(result):
                    raise ValueError("Native local reference did not pass its authoritative evaluator")
                selection.publish(evaluation)
                if not _successful(selection.qualified_result(origin="fresh")):
                    raise ValueError("Published local authority has no finite self-normalized score")
            authority = _requalify(selection)
            write_json(output / "authority.json", authority)
            qualified = True
            summary.update(state="qualified", authority=str(output / "authority.json"), **authority)
        except Exception as error:
            summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        summary.update(finished_at=time.time(), wall_seconds=time.monotonic() - started_clock,
                       accounting=_accounting(endpoint, output, transport, qualified))
        summary["accounting"]["registry_publications"] = int(selection.record_path is not None)
        write_json(output / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "batch-root", "benchmark-input", "output", "reference-store"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--endpoint", choices=tuple(TASKS), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args(argv)
    runtime = QualificationRuntime(args.source, args.batch_root, args.image, args.gpu,
        cpus=16, memory_gb=48, share_gpu=True, min_gpu_free_mb=2048)
    result = run_reference(runtime, endpoint=args.endpoint, case=args.case,
        benchmark_input=args.benchmark_input, output=args.output, reference_store=args.reference_store)
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)
    return 0 if result["state"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
