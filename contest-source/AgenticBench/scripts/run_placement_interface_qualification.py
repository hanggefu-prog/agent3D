#!/usr/bin/env python3
"""Run one native identity and one predefined interface probe, without search.

This diagnostic does not publish reference-store records, adopt historical
scores, call a model, or consume harness candidate finals. Every output is new;
failed or uncertain attempts are retained and never retried automatically.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic3d.knowledge import write_json
from local_bench import reference_baseline as baseline
from local_bench.benchmark_input import validate_benchmark_input
from local_bench.evaluation import _dreamplace_global_placement_command, _validate_dreamplace_global_output
from local_bench.metrics import placement_metrics, placement_nesterov_metrics
from local_bench.model import load_candidate, load_task, resolve_task_path
from local_bench.program_bridge import initial_program_text, materialize_program
from local_bench.reference_catalog import qualify_producer
from local_bench.reference_local_placement import DREAMPLACE, RUNTIME_MODULES, _expected_overlay, _regular, _strict_native_metrics
from local_bench.reference_placement_evidence import DYNAMIC_TASK, SOURCE_TASK, STAGE, TASKS, validate_reference_placement_evidence
from local_bench.reference_store import digest, file_sha, protect_destination
from local_bench.verifier import verify_candidate
from scripts.local_reference import LocalPlacementReference
from scripts.run_local_task_smoke import DockerEvaluationTransport, LocalSmokeRuntime, _successful

TIMEOUT_SECONDS = 21600
MIN_DATA_FREE_BYTES = 64 * 1024**3
PROBES = {
    DYNAMIC_TASK: ("return 1.0, (1.0, 1.0)", "return 1.05, (1.1, 1.0 / 1.1)",
        "Fixed shared gradient-step factor 1.05 and proposed bottom/top factors 1.1 and 1/1.1; trusted projection unchanged."),
    SOURCE_TASK: ("self.alpha = 1.0", "self.alpha = 1.1",
        "Change only initial PreconditionOp alpha from 1.0 to 1.1; initializer body and all physical settings unchanged."),
}


class QualificationRuntime(LocalSmokeRuntime):
    """Add this diagnostic's 64 GiB data guard without changing other jobs."""

    @contextmanager
    def slot(self):
        while True:
            # Parent owns shared GPU locks and checks RAM=48+16 GiB, root disk
            # 10 GiB, and the requested GPU guard. Release its lock immediately
            # if this task's stronger data-disk admission cannot be met.
            with super().slot() as gpu:
                if shutil.disk_usage(self.batch).free >= MIN_DATA_FREE_BYTES:
                    yield gpu
                    return
            time.sleep(30)


def probe_program(spec, source):
    initial = initial_program_text(spec, load_candidate(spec.initial_candidate))
    old, new, description = PROBES[spec.task_id]
    if initial.count(old) != 1:
        raise ValueError("Predefined probe anchor changed; no native execution is permitted")
    probe = initial.replace(old, new)
    for text in (initial, probe):
        verify_candidate(spec, materialize_program(spec, text, source))
    return initial, probe, description


def _audit_probe(runtime, spec, evaluation, candidate_dir, reference, contract, expected_candidate):
    """Re-read native outputs and bind the executed overlay to the exact probe."""
    run = evaluation / "run"
    workspace = evaluation / "workspace/Open3DBench-V2.0"
    overlay = evaluation / "runtime/dreamplace-install"
    result = json.loads(_regular(evaluation / "evaluation_result.json").read_text())
    artifacts = result["artifacts"]
    expected_paths = {"task": spec.task_id, "candidate": str(candidate_dir), "run_dir": str(run),
        "workspace": str(workspace), "output_dir": str(evaluation), "dreamplace_runtime": str(overlay)}
    if any(artifacts.get(key) != value for key, value in expected_paths.items()) or artifacts.get("exit_code") != 0:
        raise ValueError("Probe native task/runtime/output identity differs")
    candidate = load_candidate(candidate_dir)
    if candidate != expected_candidate:
        raise ValueError("Probe program changed after its predefined materialization")
    patched = _expected_overlay(runtime.source, candidate)
    files = {}
    for name in RUNTIME_MODULES:
        relative = str(DREAMPLACE / name)
        actual = _regular(overlay / "dreamplace" / name)
        expected = patched.get(relative, (runtime.source / relative).read_text())
        if actual.read_text() != expected:
            raise ValueError("Probe loaded an unexpected DREAMPlace module: " + name)
        files[str(actual)] = file_sha(actual)
    wrapper = workspace / "Place-3D/src/placement/concurrent_dreamplace.py"
    if file_sha(_regular(wrapper)) != contract["physical"]["implementation_sha256"][str(wrapper.relative_to(workspace))]:
        raise ValueError("Probe native command owner differs from the qualified source")
    for relative, expected in contract["frozen_inputs"]["physical_prefix_sha256"].items():
        if file_sha(_regular(run / relative)) != expected:
            raise ValueError("Probe frozen physical input changed: " + relative)
    config = yaml.safe_load(_regular(evaluation / "resolved_config.yaml").read_text())
    expected_config = json.loads(json.dumps(contract["physical"]["configuration"]))
    expected_config["placement"].update(candidate["config_overrides"]["placement"])
    expected_config["placement"]["config"] = str(evaluation / "candidate_files/placement.json")
    expected_flow = json.loads(json.dumps(candidate["candidate_files"]["placement.json"]))
    expected_flow["concurrent_dreamplace"]["initialization_seed"] = 1000
    if (config != expected_config or
            json.loads(_regular(evaluation / "candidate_files/placement.json").read_text()) != expected_flow):
        raise ValueError("Probe physical configuration differs from the fixed recipe")
    params = evaluation / "harness_inputs/design_params.json"
    if json.loads(_regular(params).read_text()) != json.loads(_regular(run / "10_floorplan/design_params.json").read_text()):
        raise ValueError("Probe changed the preserved native runtime inputs")
    command, cwd = _dreamplace_global_placement_command(spec, config, workspace, run,
        "/usr/bin/python3", base_params_override=params)
    rows = artifacts.get("native_flow_execution")
    if (not isinstance(rows, list) or len(rows) != 1 or artifacts.get("command") != command
            or rows[0].get("command") != command or rows[0].get("cwd") != str(cwd)
            or rows[0].get("process_started") is not True or rows[0].get("returncode") != 0
            or rows[0].get("status") != "completed"):
        raise ValueError("Probe lacks one successful native placement invocation")
    seconds = rows[0].get("wall_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Probe native duration is unavailable")
    for path in _validate_dreamplace_global_output(run).values():
        path = _regular(path)
        files[str(path)] = file_sha(path)
    manifest = json.loads((run / STAGE / "stage_manifest.json").read_text())
    if (manifest.get("dreamplace_root") != str(overlay / "dreamplace")
            or manifest.get("command") != ["/usr/bin/python3", str(overlay / "dreamplace/Placer.py"), str(run / STAGE / "params.json")]):
        raise ValueError("Probe native stage did not load its verified overlay")
    evidence = validate_reference_placement_evidence(run, spec.task_id, runtime_overlay=overlay)
    raw = _strict_native_metrics(run)
    scorer = placement_nesterov_metrics if spec.task_id == DYNAMIC_TASK else placement_metrics
    measured = scorer(run, reference)
    if any(result["metrics"].get(key) != value for key, value in measured.items()):
        raise ValueError("Probe evaluator metrics differ from independent native extraction")
    active = None
    if spec.task_id == DYNAMIC_TASK:
        active = not evidence["dynamic_controls"]["identity_controls_only"]
        if not active:
            raise ValueError("Nontrivial Nesterov probe did not activate nonidentity controls")
    return {"status": "passed", "scope": "native-runtime-and-interface-evidence-only",
        "raw_metrics": raw, "normalized_metrics": measured, "evidence_sha256": files,
        "native_runtime_seconds": seconds, "nonidentity_control_observed": active,
        "source_overlay_verified": True, "numerical_equivalence_qualified": False,
        "improvement_qualified": False,
        "limitation": "Preconditioner probe verifies loaded source, not instrumented call counts or a standalone initializer perturbation."
                      if spec.task_id == SOURCE_TASK else "Control activity is not proof of improvement or historical algorithm equivalence."}


def _accounting(transport, requested):
    rows = getattr(transport, "attempts", [])
    result = {"evaluator_calls_requested": len(requested), "execution_attempts": rows,
        "model_calls": 0, "generated_model_candidates": 0, "candidate_finals": 0,
        "identity_attempts_requested": requested.count("identity"),
        "nontrivial_attempts_requested": requested.count("nontrivial")}
    for name, field in (("evaluator_dispatches", "evaluator_dispatch_attempted"),
                        ("native_physical_attempts", "native_flow_started")):
        known = sum(row.get(field) is True for row in rows)
        unknown = sum(row.get(field) is None for row in rows) + max(0, len(requested) - len(rows))
        result.update({name: None if unknown else known, name + "_known": known, name + "_unknown": unknown})
    result["native_attempt_definition"] = "Evidence of native flow-process launch, not a claim that each inner tool started."
    return result


def run_qualification(runtime, *, task, benchmark_input, output, transport_factory=DockerEvaluationTransport):
    spec = load_task(resolve_task_path(runtime.framework / "local_tasks", str(task)))
    if spec.task_id not in TASKS:
        raise ValueError("Only the two versioned placement interfaces are supported")
    if runtime.cpus != 16 or runtime.memory_gb != 48 or runtime.min_gpu_free_mb != 2048:
        raise ValueError("Interface qualification keeps the 16 CPU / 48 GiB cap and 2 GiB GPU guard")
    raw_output = Path(output).expanduser().absolute()
    if raw_output.exists() or any(path.is_symlink() for path in (raw_output, *raw_output.parents)):
        raise FileExistsError("Qualification output must be a new non-symlink path; no retries in place")
    output = runtime.contained(raw_output)
    if output == runtime.batch:
        raise ValueError("The batch root cannot be a qualification output")
    bundle = runtime.contained(benchmark_input)
    validated = validate_benchmark_input(bundle, task_id=spec.task_id)
    config = Path(validated["config"]).resolve(strict=True)
    case = yaml.safe_load(config.read_text())["design"]
    aligned = baseline.validate_reference_bundle(bundle, case, fixed_partition=True)
    frozen = runtime.contained(validated["run_dir"])
    protect_destination(raw_output, {"source_root": str(runtime.source), "benchmark_input": str(bundle),
        "run_root": str(frozen), "reference_root": aligned["reference_root"],
        "thermal_root": aligned.get("original_root")})
    spec = replace(spec, base_config=config)
    selection = LocalPlacementReference(runtime, spec, case, bundle, output, selected_policy="FRESH")
    initial, probe, description = probe_program(spec, runtime.source)
    runtime.reference_baseline, runtime.reference_root = baseline.BASELINE_ID, Path(aligned["reference_root"])
    runtime.local_reference_selection = runtime.adopted_reference_run = runtime.reference_purpose_receipt = None
    output.mkdir(parents=True, exist_ok=False)
    with (output / ".qualification.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(json.dumps({"pid": os.getpid(), "task": spec.task_id, "output": str(output)}) + "\n")
        lock.flush()
        return _run_pair(runtime, spec, case, bundle, frozen, config, output,
                         selection, initial, probe, description, transport_factory)


def _run_pair(runtime, spec, case, bundle, frozen, config, output,
              selection, initial, probe, description, transport_factory):
    write_json(output / "run_config.json", {"schema_version": 1, "kind": "diagnostic-placement-interface-pair-v1",
        "task": spec.task_id, "case": case, "source_commit": runtime.source_commit, "image": runtime.image,
        "contract": selection.contract, "timeout_seconds": TIMEOUT_SECONDS, "physical_threads": 4,
        "container_cpus": runtime.cpus, "container_memory_gb": runtime.memory_gb,
        "admission": {"ram_available_gib": 64, "data_free_gib": 64, "root_free_gib": 10,
                      "gpu_free_mib": 2048, "locks": str(runtime.batch / "locks")},
        "planned_native_attempts": 2, "identity_attempts": 1, "nontrivial_attempts": 1,
        "probe_description": description, "reference_store_publication": False,
        "model_calls": 0, "candidate_finals": 0, "retry_policy": "none", "retention": "full"})
    transport = transport_factory(runtime)
    requested, passed = [], []
    summary = {"schema_version": 1, "kind": "diagnostic-placement-interface-pair-v1", "task": spec.task_id,
        "state": "running", "identity_reference_qualified": False, "interface_probe_qualified": False,
        "historical_equivalence_qualified": False, "improvement_qualified": False,
        "reference_store_records_published": 0, "error": None}
    write_json(output / "status.json", summary)
    try:
        candidate = output / "reference/candidate"
        candidate.mkdir(parents=True)
        write_json(candidate / "candidate.json", materialize_program(spec, initial, runtime.source))
        identity_output = output / "reference/result"
        runtime.reference_purpose_receipt = selection.prepare_fresh(identity_output, candidate)
        requested.append("identity")
        try:
            result = transport(spec.root, candidate, runtime.source, identity_output,
                baseline_run=frozen, base_config=config, timeout_seconds=TIMEOUT_SECONDS)
        finally:
            runtime.reference_purpose_receipt = None
        if not _successful(result):
            raise RuntimeError("Native identity evaluation failed; nontrivial probe was not dispatched")
        passed.append("identity")
        producer = {"kind": "native-local-placement-v1", "task_id": spec.task_id, "case": case,
            "source_root": str(runtime.source), "evaluation_root": str(identity_output), "benchmark_input": str(bundle)}
        qualified = qualify_producer(producer)
        pinned_qualification = digest(qualified)
        write_json(output / "reference/qualification.json", {"qualification": qualified,
            "qualification_sha256": pinned_qualification, "scope": "new-identity-interface-diagnostic-only",
            "reference_store_publication": False, "candidate_finals": 0})
        summary["identity_reference_qualified"] = True
        reference = Path(qualified["baseline"]["source_run"])
        candidate = output / "nontrivial/candidate"
        candidate.mkdir(parents=True)
        probe_candidate = materialize_program(spec, probe, runtime.source)
        write_json(candidate / "candidate.json", probe_candidate)
        if digest(qualify_producer(producer)) != pinned_qualification:
            raise ValueError("Identity authority changed before the nontrivial probe")
        probe_output = output / "nontrivial/result"
        requested.append("nontrivial")
        result = transport(spec.root, candidate, runtime.source, probe_output, baseline_run=frozen,
            reference_run=reference, base_config=config, timeout_seconds=TIMEOUT_SECONDS)
        if not _successful(result):
            raise RuntimeError("Native nontrivial probe failed; no retry is permitted")
        passed.append("nontrivial")
        if digest(qualify_producer(producer)) != pinned_qualification:
            raise ValueError("Identity authority changed during the nontrivial probe")
        audit = _audit_probe(runtime, spec, probe_output, candidate, reference, selection.contract, probe_candidate)
        write_json(output / "nontrivial/qualification.json", audit)
        summary.update(state="completed", interface_probe_qualified=True,
            nonidentity_control_observed=audit["nonidentity_control_observed"],
            identity_raw_metrics=qualified["baseline"]["raw_metrics"], probe_raw_metrics=audit["raw_metrics"],
            interpretation=audit["limitation"])
    except Exception as error:
        summary.update(state="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        runtime.local_reference_selection = runtime.adopted_reference_run = runtime.reference_purpose_receipt = None
    summary.update(_accounting(transport, requested), successful_evaluations=len(passed),
        identity_successful_evaluations=passed.count("identity"), nontrivial_successful_evaluations=passed.count("nontrivial"))
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "batch-root", "benchmark-input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpus", type=int, choices=(16,), default=16)
    parser.add_argument("--memory-gb", type=int, choices=(48,), default=48)
    parser.add_argument("--share-gpu", action="store_true")
    parser.add_argument("--min-gpu-free-mb", type=int, choices=(2048,), default=2048)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    runtime = QualificationRuntime(args.source, args.batch_root, args.image, args.gpu,
        args.cpus, args.memory_gb, args.share_gpu, args.min_gpu_free_mb)
    result = run_qualification(runtime, task=args.task, benchmark_input=args.benchmark_input, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
