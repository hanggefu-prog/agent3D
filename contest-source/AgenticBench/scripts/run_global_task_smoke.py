#!/usr/bin/env python3
"""Run five workspace rounds, or the explicit ten-round one-pass timing task.

The input bundle and qualified workspace/model preflight must already exist.
This wrapper performs no model probe or input preparation. It uses the existing
host login and process-local isolated terminal configuration. Reference ADOPT
is the default; FRESH or supported EXTEND must be requested explicitly. Reference
qualification is separate from the selected candidate budget; no fallback or retry
is implicit. The frozen input bundle must remain under --batch-root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic3d.knowledge import write_json
from global_bench.model import load_global_task
from local_bench.benchmark_input import validate_benchmark_input
from scripts.codesign_runtime import CASES, Runtime
from scripts.run_scheduled_codesign import MODEL, check_model_preflight
from scripts.run_workspace_codesign import TASKS, run_workspace_experiment
from scripts.workspace_codesign_evaluator import ENDPOINTS, endpoint_for_task
from scripts.workspace_task_selection import workspace_task_id, validate_selected_task

DEFAULT_CASE = "bp_fe"


def _event(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "batch-root", "benchmark-input", "output", "preflight-result"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--objective", choices=tuple(TASKS), required=True)
    parser.add_argument("--method", choices=("codex", "agentic3d"), default="codex",
                        help="Single workspace Codex or code-defined workspace Agentic3D roles")
    parser.add_argument("--case", choices=CASES, default=argparse.SUPPRESS,
                        help="Case override; smoke default is bp_fe because global tasks pin no case")
    parser.add_argument("--image", required=True)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--release-manifest-sha256")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=48)
    parser.add_argument("--min-gpu-free-mb", type=int, default=2048)
    parser.add_argument("--rounds", type=int, choices=(5, 10), default=5,
                        help="10 selects a registered ten-round task; latest timing and CTS thermal support it")
    parser.add_argument("--model", choices=[MODEL], default=MODEL)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--reference-policy", choices=("ADOPT", "FRESH", "EXTEND"), default="ADOPT")
    parser.add_argument("--reference-baseline", choices=("reference_20260923", "reference_20261004"),
                        default="reference_20260923")
    parser.add_argument("--framework-sha256", help="Exact AgenticBench tree digest for the latest explicit source split")
    parser.add_argument("--reference-store", type=Path,
                        help="Append-only qualified reference store; default: batch-root/reference-store")
    parser.add_argument("--reference-record", type=Path,
                        help="Exact sealed ADOPT record or EXTEND parent; no automatic fallback")
    parser.add_argument("--audit-framework-revision",
                        help="Full clean trusted-framework commit; explicit audit-only source split")
    parser.add_argument("--computation-source-revision",
                        help="Full clean physical-source commit paired with --audit-framework-revision")
    parser.add_argument("--retention", choices=("full", "best", "best-per-round"), default="full",
                        help="full: keep all; best: compact after search; best-per-round: compact at audited round boundaries")
    args = parser.parse_args(argv)

    # Fail before constructing Runtime or starting any physical/model process.
    # Reuse the existing strict workspace+resume qualification, not an API ping.
    receipt = args.preflight_result.expanduser().resolve(strict=True)
    runtime_image = None
    if args.release_manifest or args.release_manifest_sha256:
        if not (args.release_manifest and args.release_manifest_sha256) or args.reference_baseline != "reference_20261004":
            raise ValueError("Explicit release manifest/hash requires the latest family")
        from scripts.workspace_runtime_image import bind_release
        runtime_image = bind_release(args.release_manifest, args.release_manifest_sha256,
                                     source=args.source, canonical=args.image)
        if runtime_image["framework_sha256"] != args.framework_sha256:
            raise ValueError("Release framework does not match the requested framework")
    preflight = check_model_preflight(receipt, runtime_image["actual_image"] if runtime_image else args.image)
    explicit_case = hasattr(args, "case")
    case = args.case if explicit_case else DEFAULT_CASE
    task = load_global_task(ROOT / "global_tasks" / workspace_task_id(args.objective, args.rounds,
                            baseline=args.reference_baseline))
    validate_selected_task(task, args.objective, args.rounds)
    endpoint = endpoint_for_task(args.objective, task)

    split = {}
    if args.reference_baseline == "reference_20261004":
        if args.reference_policy != "ADOPT" or not args.reference_record or not args.framework_sha256:
            raise ValueError("Latest family requires explicit ADOPT record and framework SHA; no fallback")
        if args.audit_framework_revision or args.computation_source_revision:
            raise ValueError("Latest family cannot use the historical audit/source split")
        split = {"reference_baseline": args.reference_baseline, "framework_sha256": args.framework_sha256}
    elif args.framework_sha256 is not None:
        raise ValueError("Framework SHA is only valid for the latest family")
    if args.audit_framework_revision is not None or args.computation_source_revision is not None:
        if not args.audit_framework_revision or not args.computation_source_revision:
            raise ValueError("Audit framework and computation source must both be explicitly pinned")
        if args.reference_policy != "ADOPT":
            raise ValueError("The audited source split supports ADOPT only; no new reference dispatch")
        split = {"audit_framework_revision": args.audit_framework_revision,
                 "computation_source_revision": args.computation_source_revision}
    runtime = Runtime(args.source.expanduser().absolute(), args.batch_root.expanduser().resolve(),
                      args.image, args.gpu, args.cpus, args.memory_gb,
                      share_gpu=True, min_gpu_free_mb=args.min_gpu_free_mb,
                      min_data_free_gb=64, **split)
    runtime.retention_mode = args.retention
    runtime.workspace_runtime_image = runtime_image
    runtime.workspace_rounds = args.rounds
    runtime.workspace_task_digest = task.digest()
    runtime.reference_baseline = args.reference_baseline
    runtime.reference_policy = args.reference_policy
    runtime.reference_store = (args.reference_store.expanduser().resolve() if args.reference_store
                               else runtime.batch / "reference-store")
    runtime.reference_record = args.reference_record.expanduser().absolute() if args.reference_record else None
    bundle = runtime.contained(args.benchmark_input.expanduser())
    output = runtime.contained(args.output.expanduser())
    if (output == runtime.batch or output == bundle or output in bundle.parents
            or bundle in output.parents or output == runtime.source or runtime.source in output.parents):
        raise ValueError("Output must be a private run directory, separate from source and frozen input")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Existing smoke attempt requires inspection; no implicit replay: " + str(output))
    if not bundle.is_dir():
        raise ValueError("--benchmark-input must name an existing frozen bundle directory")
    input_kind = endpoint["profile"]
    validated = validate_benchmark_input(bundle, task_id=input_kind)
    if Path(validated["config"]).resolve() != (runtime.profile(case) / "config.yaml").resolve():
        raise ValueError("Frozen input profile does not match the selected case")
    runtime.contained(Path(validated["run_dir"]))
    manifest = bundle / "benchmark_input.json"
    input_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()

    # Exclusive creation makes this wrapper at-most-once even before the driver's
    # own durable reference/candidate ledgers. It never rewrites old experiment data.
    output.mkdir(parents=True, exist_ok=False)
    contract = {
        "schema_version": 1, "kind": "global-workspace-integration-smoke-v1",
        "method": args.method,
        "retention": args.retention,
        "case": case, "case_selection": "explicit-cli-override" if explicit_case else "smoke-default",
        "case_selection_reason": "Global task templates do not pin a case; bp_fe is the smoke default",
        "objective": args.objective, "task": task.to_dict(), "task_digest": task.digest(),
        "evaluation_endpoint": endpoint, "model": args.model,
        "candidate_final_budget": args.rounds, "reference_qualification_in_candidate_budget": False,
        "reference_policy": args.reference_policy,
        "reference_store": str(runtime.reference_store),
        "reference_record": str(runtime.reference_record) if runtime.reference_record else None,
        "reference_fallback": "none",
        "reference_evaluator_invocations_expected": 0 if args.reference_policy == "ADOPT" else 1,
        "source_commit": runtime.source_commit, "source": str(runtime.source), "image": args.image,
        "installed_runtime": runtime_image,
        "audit_framework_identity": getattr(runtime, "audit_framework_identity", None),
        "latest_framework_identity": getattr(runtime, "latest_framework_identity", None),
        "reference_baseline": args.reference_baseline,
        "batch_root": str(runtime.batch), "shared_lock_root": str(runtime.batch / "locks"),
        "benchmark_input": str(bundle), "input_manifest_sha256": input_hash,
        "model_preflight": preflight, "model_probe_started": False,
        "resources": {"cpus": args.cpus, "memory_gb": args.memory_gb, "gpus": runtime.gpus,
                      "share_gpu": True, "min_gpu_free_mb": args.min_gpu_free_mb,
                      "min_data_free_gb": 64},
        "authentication_policy": "Existing host login; no credential copies or global configuration changes",
        "network_policy": "Inherit caller process environment for Codex only; EDA/tool containers remain offline",
    }
    write_json(output / "smoke_contract.json", contract)
    state = {"state": "running", "case": case, "objective": args.objective,
             "started_at": time.time(), "driver_status": str(output / "status.json")}
    write_json(output / "smoke_status.json", state)
    _event("global-smoke-started", case=case, objective=args.objective, output=str(output),
           model=args.model, candidate_finals=args.rounds, reference_policy=contract["reference_policy"],
           preflight_reused=str(receipt))
    try:
        if args.method == "agentic3d":
            from scripts.run_workspace_agentic3d import run_workspace_agentic3d_experiment
            run_workspace_agentic3d_experiment(runtime, case, args.objective, bundle, output, args.codex_bin)
        else:
            run_workspace_experiment(runtime, case, args.objective, bundle, output, args.codex_bin)
        status = json.loads((output / "status.json").read_text())
        if status.get("state") != "completed":
            raise RuntimeError("Workspace driver needs inspection: " + str(output / "status.json"))
        summary = json.loads((output / "summary.json").read_text())
        if (summary.get("state") != "completed" or summary.get("case") != case
                or summary.get("objective") != args.objective
                or type(summary.get("final_evaluations")) is not int
                or summary["final_evaluations"] != args.rounds
                or type(summary.get("reference_evaluations")) is not int
                or summary["reference_evaluations"] != contract["reference_evaluator_invocations_expected"]
                or type(summary.get("reference_adoptions")) is not int
                or summary["reference_adoptions"] != int(args.reference_policy == "ADOPT")):
            raise ValueError("Completed workspace result differs from the pinned task/budget/reference contract")
        state.update(state="completed", finished_at=time.time(), summary=str(output / "summary.json"))
        write_json(output / "smoke_status.json", state)
        _event("global-smoke-completed", case=case, objective=args.objective,
               summary=str(output / "summary.json"),
               successful_final_evaluations=summary.get("successful_final_evaluations"))
        return 0
    except Exception as error:
        state.update(state="needs-attention", error=f"{type(error).__name__}: {error}", finished_at=time.time())
        write_json(output / "smoke_status.json", state)
        _event("global-smoke-needs-attention", case=case, objective=args.objective,
               error=state["error"], output=str(output))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
