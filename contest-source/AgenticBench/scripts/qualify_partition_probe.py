#!/usr/bin/env python3
"""Run one loss-times-1.1 native interface probe after an existing FRESH authority.

No initial reference rerun, model calls, candidate finals or registry writes.
The fixed seed/training budget/physical recipe is unchanged. Existing outputs
are refused, including failures and uncertain attempts; there is no retry path.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from agentic3d.knowledge import write_json
from local_bench import reference_baseline as baseline
from local_bench.benchmark_input import validate_benchmark_input
from local_bench.model import load_task
from local_bench.reference_local_partition import TASK, probe_candidate, qualify_partition_probe
from local_bench.reference_store import file_sha, protect_destination
from scripts.local_reference import LocalPartitionReference
from scripts.run_local_task_smoke import DockerEvaluationTransport, _successful
from scripts.run_placement_interface_qualification import QualificationRuntime


def run_probe(runtime, *, benchmark_input, output, reference_authority):
    if (runtime.cpus, runtime.memory_gb, runtime.min_gpu_free_mb) != (16, 48, 2048) or runtime.batch.name != "runs":
        raise ValueError("Probe requires shared runs/locks, 16 CPU / 48 GiB caps and the 2 GiB GPU guard")
    raw_output = Path(output).expanduser().absolute()
    if raw_output.exists() or any(path.is_symlink() for path in (raw_output, *raw_output.parents)):
        raise FileExistsError("Probe output exists or traverses a symlink; no implicit retry")
    output, bundle = runtime.contained(raw_output), runtime.contained(benchmark_input)
    authority_path = runtime.contained(reference_authority)
    authority = json.loads(authority_path.read_text())
    record_path = Path(authority["record"]).resolve(strict=True)
    if authority.get("record_sha256") != file_sha(record_path):
        raise ValueError("Authority receipt does not pin the current reference record")
    case = authority["contract"]["case"]
    if case != "bp_fe" or authority["contract"].get("endpoint") != "partition":
        raise ValueError("This bounded interface probe is registered only for bp_fe partition")
    validated = validate_benchmark_input(bundle, task_id=TASK)
    config = Path(validated["config"]).resolve(strict=True)
    aligned = baseline.validate_reference_bundle(bundle, case, fixed_partition=False)
    runtime.reference_baseline = baseline.BASELINE_ID
    runtime.reference_root = Path(aligned["reference_root"])
    spec = replace(load_task(runtime.framework / "local_tasks" / TASK), base_config=config)
    selection = LocalPartitionReference(runtime, spec, case, bundle, output,
        selected_policy="ADOPT", record=record_path, reference_id=authority["reference_id"])
    if selection.contract != authority["contract"] or selection.record_sha256 != authority["record_sha256"]:
        raise ValueError("Probe authority differs from the independently reconstructed consumer contract")
    protect_destination(output, {"source_root": str(runtime.source), "benchmark_input": str(bundle),
        "reference_root": aligned["reference_root"], "run_root": str(selection.source_run),
        "thermal_root": aligned.get("original_root")}, record_path=record_path)
    if output == runtime.batch:
        raise ValueError("Probe cannot overwrite the shared runs root")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "probe.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        evaluation, candidate = output / "probe/result", output / "probe/candidate"
        candidate.mkdir(parents=True)
        write_json(candidate / "candidate.json", probe_candidate(spec, runtime.source))
        purpose_path = evaluation.parent / "reference-purpose.json"
        write_json(purpose_path, {"schema_version": 1, "reference_purpose": "INTERFACE_PROBE",
            "reference_run": str(selection.source_run), "source_root": str(runtime.source),
            "evaluation_root": str(evaluation), "image": runtime.image,
            "candidate_sha256": file_sha(candidate / "candidate.json"), "contract": selection.contract})
        runtime.local_reference_selection = selection
        runtime.adopted_reference_run = selection.source_run
        runtime.reference_purpose_receipt = purpose_path
        transport = DockerEvaluationTransport(runtime)
        summary = {"schema_version": 1, "state": "running", "kind": "partition-loss-interface-probe-v1",
            "case": case, "model_calls": 0, "generated_model_candidates": 0, "candidate_finals": 0,
            "registry_publications": 0, "reference_evaluations": 0, "build_dispatches": 0,
            "source_commit": runtime.source_commit, "image": runtime.image,
            "reference_selection": selection.to_dict(), "loss_multiplier": 1.1,
            "search_dispatch": False, "retry_policy": "none", "api_latency_seconds": None,
            "successful_native_evaluations": 0, "independent_qualification_rejections": 0,
            "shared_lock_root": str(runtime.batch / "locks")}
        write_json(output / "summary.json", summary)
        try:
            result = transport(spec.root, candidate, runtime.source, evaluation,
                baseline_run=Path(validated["run_dir"]), reference_run=selection.source_run,
                base_config=config, timeout_seconds=21600)
            if not _successful(result):
                raise ValueError("Partition loss probe failed its native evaluator")
            summary["successful_native_evaluations"] = 1
            try:
                evidence = qualify_partition_probe(evaluation, source_root=runtime.source, benchmark_input=bundle,
                    reference_record=record_path, contract=selection.contract)
            except Exception:
                summary["independent_qualification_rejections"] = 1
                raise
            write_json(output / "qualification.json", evidence)
            summary.update(state="qualified", qualification=str(output / "qualification.json"),
                           measured_first_epoch_loss=evidence["measured_first_epoch_loss"])
        except Exception as error:
            summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        finally:
            runtime.reference_purpose_receipt = None
        rows = transport.attempts
        known = sum(row.get("native_flow_started") is True for row in rows)
        unknown = sum(row.get("native_flow_started") is None for row in rows)
        summary.update(wall_seconds=time.monotonic() - start, execution_attempts=rows,
            evaluator_container_dispatches=sum(row.get("evaluator_dispatch_attempted") is True for row in rows),
            evaluator_container_dispatches_unknown=sum(row.get("evaluator_dispatch_attempted") is None for row in rows),
            native_physical_attempts=None if unknown else known,
            native_physical_attempts_known=known, native_physical_attempts_unknown=unknown,
            successful_interface_qualifications=int(summary["state"] == "qualified"), improvements=0)
        write_json(output / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "batch-root", "benchmark-input", "output", "reference-authority"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args(argv)
    runtime = QualificationRuntime(args.source, args.batch_root, args.image, args.gpu,
        cpus=16, memory_gb=48, share_gpu=True, min_gpu_free_mb=2048)
    summary = run_probe(runtime, benchmark_input=args.benchmark_input, output=args.output,
                        reference_authority=args.reference_authority)
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)
    return 0 if summary["state"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
