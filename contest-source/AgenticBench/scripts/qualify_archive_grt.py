"""One-shot GRT hook qualification on the authenticated 2026-09-23 CTS input.

This is deliberately not a search/evaluator/cache API. It runs neither models
nor harness finals, never changes the archive, and never creates historical
stage-15/stage-75 manifests. A new native GRT publication is produced by the
existing backend, with its existing validators and frozen physical settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK))
sys.path.insert(0, str(FRAMEWORK.parent))

from local_bench.model import load_candidate, load_task
from local_bench.program_bridge import materialize_program
from local_bench.reference_archive import _Archive, prepare_archive_reference, STAGES
from local_bench.reference_store import file_sha, protect_destination
from local_bench.verifier import verify_candidate
from local_bench.workspace import (apply_candidate_patch, copy_upstream_stages,
    isolated_clone, prepare_openroad_runtime, PREBUILT_OPENROAD_ENV)

TASK = "global_routing_net_ordering_source"
TASK_DIRECTORY = "global_routing_net_ordering"
UTILITY = "OpenROAD-3D/openroad/src/grt/src/fastroute/src/utility.cpp"
TIMEOUT_SECONDS = 21600
PROGRAMS = {
    "identity": "def score_net_order(ctx):\n    return ctx.baseline_score\n",
    "reversed": "def score_net_order(ctx):\n    return -ctx.baseline_score\n",
}
TASKS = {
    TASK: {"directory": TASK_DIRECTORY, "kind": "routing_net_ordering_source",
           "source": UTILITY, "metric": "routing_net_ordering_metrics", "programs": PROGRAMS},
    "global_routing_ripup_policy_source": {
        "directory": "global_routing_ripup_policy", "kind": "routing_ripup_policy_source",
        "source": "OpenROAD-3D/openroad/src/grt/src/fastroute/src/RipUp.cpp",
        "metric": "routing_ripup_policy_metrics",
        "programs": {
            "identity": "def score_ripup_edge(ctx):\n    return ctx.baseline_decision\n",
            "probe": "def score_ripup_edge(ctx):\n    return 1.0 - ctx.baseline_decision\n"}},
    "global_routing_maze_cost_source": {
        "directory": "global_routing_maze_cost", "kind": "routing_maze_cost_source",
        "source": "OpenROAD-3D/openroad/src/grt/src/fastroute/src/maze.cpp",
        "metric": "routing_maze_cost_metrics",
        "programs": {
            "identity": "def score_maze_step(ctx):\n    return 1.0\n",
            "probe": "def score_maze_step(ctx):\n    return 1.0 + 0.1 * ctx.is_turn\n"}},
    "global_routing_iteration_controller_source": {
        "directory": "global_routing_iteration_controller", "kind": "routing_iteration_controller_source",
        "source": "OpenROAD-3D/openroad/src/grt/src/fastroute/src/FastRoute.cpp",
        "metric": "routing_iteration_controller_metrics",
        "programs": {
            "identity": "def control_reroute_iteration(ctx):\n    return (ctx.baseline_congestion_scale, ctx.baseline_search_scale, ctx.baseline_ripup_offset)\n",
            "probe": "def control_reroute_iteration(ctx):\n    return (2.0 * ctx.baseline_congestion_scale, ctx.baseline_search_scale, ctx.baseline_ripup_offset)\n"}},
}


def task_contract(task, policy=None):
    if task not in TASKS:
        raise ValueError("Unsupported qualification task: " + str(task))
    contract = TASKS[task]
    if policy is not None and policy not in contract["programs"]:
        raise ValueError("Unsupported qualification policy for " + task + ": " + str(policy))
    return contract


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fresh(path, **protected):
    destination = protect_destination(path, protected)
    if destination.exists():
        raise FileExistsError("Qualification output exists; failed or uncertain attempts cannot be retried")
    return destination


def expected_candidate(source, policy, task=TASK):
    contract = task_contract(task, policy)
    spec = load_task(Path(source) / "AgenticBench/local_tasks" / contract["directory"])
    if spec.task_id != task or spec.kind != contract["kind"]:
        raise ValueError("Unexpected qualification task")
    candidate = materialize_program(spec, contract["programs"][policy], Path(source))
    proof = verify_candidate(spec, candidate, Path(source))
    if set(proof["patch"]["paths"]) != {contract["source"]}:
        raise ValueError("Qualification requires only the selected native hook; wrapper patches are not supported")
    return candidate


def materialize(source, policy, output, task=TASK):
    source = Path(source).resolve(strict=True)
    output = _fresh(output, source_root=str(source))
    contract = task_contract(task, policy)
    candidate = expected_candidate(source, policy, task)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "candidate.json", candidate)
    (output / "program.py").write_text(contract["programs"][policy])
    _write(output / "qualification.json", {"kind": "grt-hook-qualification-program-v1",
        "policy": policy, "task": task, "candidate_sha256": _digest(candidate),
        "model_calls": 0, "candidate_finals": 0})
    return output


def _verify_evidence(rows):
    for row in rows:
        path = Path(row["path"])
        if path.is_symlink() or path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            raise ValueError("Authenticated evidence changed: " + str(path))


def copy_prefix(payload, destination):
    """Use the existing fixed-input copier; retain old manifests unchanged."""
    if os.environ.get("OPEN3D_LINK_FROZEN_UPSTREAM") == "1":
        raise ValueError("Qualification requires independent copies, not shared hardlinks")
    root = Path(payload["baseline"]["source_run"])
    _verify_evidence(payload["evidence"])
    copy_upstream_stages(root, destination, list(STAGES))
    checked = []
    # The common copier only rebases stage-local design_params.json input paths.
    # All authenticated physical artifacts/manifests remain byte identical.
    for row in payload["evidence"]:
        path = Path(row["path"])
        try:
            rel = path.relative_to(root)
        except ValueError:
            # Python 3.8 has no Path.is_relative_to. Keep the same lexical
            # root-membership test; archive evidence was authenticated above.
            continue
        if not rel.parts or rel.parts[0] not in STAGES.values():
            continue
        target = destination / rel
        if len(rel.parts) == 2 and rel.name == "design_params.json":
            original = json.loads(path.read_text())
            expected = dict(original)
            for key, name in (("def_input", "design.def"), ("verilog_input", "design.v"), ("sdc_input", "design.sdc")):
                if (target.parent / name).is_file():
                    expected[key] = str((target.parent / name).resolve())
            if json.loads(target.read_text()) != expected:
                raise ValueError("Unexpected frozen input parameter rebase")
        elif target.is_symlink() or target.stat().st_size != row["bytes"] or file_sha(target) != row["sha256"]:
            raise ValueError("Copied physical evidence differs: " + str(rel))
        checked.append(str(rel))
    if (destination / "15_timing_extract").exists() or (destination / "75_global_route").exists():
        raise ValueError("Qualification copier must not fabricate native stages")
    return checked


def command_for(source, run, executable, configuration, grt_configuration):
    """Invoke only the existing GRT backend, never runner/resume or DRT."""
    from open3dbench.pdks import get_pdk
    routing = configuration.get("routing", {})
    required = {"threads": 4, "congestion_iterations": 4, "resource_reduction": 0.0,
                "max_iterations": 5, "die_timeout_seconds": 10800, "seed": -1}
    if (configuration.get("type") != "lol" or configuration.get("pdk") != "ng45"
            or any(routing.get(k) != v for k, v in required.items())
            or grt_configuration != {"threads": 4, "congestion_iterations": 4,
                "resource_reduction": 0.0, "ndr_policy": {"bottom": "preserve", "top": "preserve"}}):
        raise ValueError("Frozen GRT configuration differs from the qualified physical contract")
    pdk = get_pdk("ng45")
    args = [sys.executable, "-m", "open3dbench.backend.three_d_route", "--mode", "global",
        "--checkpoint", str(run / "70_cts/checkpoint.json"), "--platform-dir",
        str(source / "OpenROAD-3D/flow/platforms" / (pdk.openroad_platform + "_3D")),
        "--output-dir", str(run / "75_global_route"), "--openroad", str(executable)]
    for flag, value in (("threads", 4), ("congestion-iterations", 4), ("resource-reduction", 0.0),
                        ("max-iterations", 5), ("die-timeout-seconds", 10800), ("droute-seed", -1)):
        args += ["--" + flag, str(value)]
    for die, layers, antenna in (("bottom", pdk.bottom_routing_layers, pdk.bottom_antenna_cell),
                                 ("top", pdk.top_routing_layers, pdk.top_antenna_cell)):
        args += ["--" + die + "-min-layer", layers[0], "--" + die + "-max-layer", layers[1],
                 "--" + die + "-antenna-cell", antenna, "--" + die + "-ndr-policy", "preserve"]
    return args


def qualify_platform(source, historical_source, archive):
    """Bind every external Liberty/suppression file actually read by GRT/SDC."""
    from open3dbench.backend.three_d_opt import _platform_files
    relative = Path("OpenROAD-3D/flow/platforms/nangate45_3D")
    current, original = Path(source) / relative, Path(historical_source) / relative
    checked = {}
    for die in ("bottom", "top"):
        _, libraries = _platform_files(current, die)
        _, old_libraries = _platform_files(original, die)
        names = {str(p.relative_to(current)) for p in libraries}
        if names != {str(p.relative_to(original)) for p in old_libraries}:
            raise ValueError("Frozen GRT Liberty inventory differs: " + die)
        for name in names:
            old = archive.file((original / name).relative_to(archive.root).as_posix())
            if file_sha(current / name) != file_sha(old):
                raise ValueError("Frozen GRT Liberty bytes differ: " + name)
            checked[name] = file_sha(old)
    name = "liberty_suppressions.tcl"
    if (current / name).is_file() != (original / name).is_file():
        raise ValueError("Frozen Liberty warning suppression presence differs")
    if (original / name).is_file():
        old = archive.file((original / name).relative_to(archive.root).as_posix())
        if file_sha(old) != file_sha(current / name):
            raise ValueError("Frozen Liberty warning suppression bytes differ")
        checked[name] = file_sha(old)
    return checked


def qualify_inputs(source, archive_root, case):
    """Read-only host preflight; repeated inside the installed runtime image."""
    source, archive_root = Path(source).resolve(strict=True), Path(archive_root).resolve(strict=True)
    payload = prepare_archive_reference(archive_root, case, "grt")
    archive = _Archive(archive_root)
    archive.index = archive.read("reference.json")
    package = archive.index["sdc_patch"]["path"]
    sdc_source = archive.file(package + "/open3dbench/backend/sdc_handoff.py")
    if file_sha(sdc_source) != file_sha(source / "open3dbench/backend/sdc_handoff.py"):
        raise ValueError("Current SDC projection source differs from the qualified archive override")
    old_route = archive.file(archive.index["cases"][case]["qualified_grt_checkpoint"]).parent
    projected = {die: {name: file_sha(archive.file((old_route / die / name).relative_to(archive_root).as_posix()))
                 for name in ("design.sdc", "design.sdc.ports.tsv")} for die in ("bottom", "top")}
    platform = qualify_platform(source, payload["source_identity"]["source_tree"], archive)
    wrappers = {"historical": file_sha(archive.file(package + "/open3dbench/backend/three_d_route.py")),
        "current": file_sha(source / "open3dbench/backend/three_d_route.py"),
        "note": "Wrappers differ; only native CTS input, frozen routing settings and SDC projection are reused. New publication is independently validated."}
    _verify_evidence(payload["evidence"])
    _verify_evidence(list(archive.evidence.values()))
    return {"archive_qualification": payload, "projected_sdc_sha256": projected,
            "platform_sha256": platform, "source_wrapper_comparison": wrappers,
            "additional_evidence": list(archive.evidence.values()),
            "native_physical_attempts": 0, "model_calls": 0, "candidate_finals": 0}


def validate_measurement(run, historical, policy, projected, task=TASK):
    from open3dbench.backend.three_d_route import _validate_global_route_publication
    from local_bench import metrics
    contract = task_contract(task, policy)
    route = run / "75_global_route"
    _validate_global_route_publication(route / "checkpoint.json")
    for die in ("bottom", "top"):
        for filename in ("design.sdc", "design.sdc.ports.tsv"):
            if file_sha(route / die / filename) != projected[die][filename]:
                raise ValueError("Projected frozen SDC/ports differ: " + die + "/" + filename)
    measured = getattr(metrics, contract["metric"])(run, None)
    presence, calls, changes = (("net_order_diagnostics_present", "net_order_calls", "net_order_changed_calls")
                               if task == TASK else ("policy_diagnostics_present", "policy_calls", "policy_changed"))
    if measured[presence] != 1 or not math.isfinite(measured[calls]) or measured[calls] <= 0:
        raise ValueError("Native " + task + " hook was not reached")
    if not math.isfinite(measured[changes]) or measured[changes] < 0:
        raise ValueError("Invalid native hook-change evidence")
    if policy == "identity":
        if measured[changes] != 0:
            raise ValueError("Identity hook changed native behavior")
        for current, old in (("grt_total_overflow", "grt_total_overflow"),
                             ("grt_wirelength_um", "grt_wire_length_um")):
            if not math.isclose(measured[current], historical[old], rel_tol=0, abs_tol=1e-9):
                raise ValueError("Identity GRT does not reproduce the historical raw metric: " + current)
    elif policy in {"reversed", "probe"}:
        if measured[changes] <= 0:
            raise ValueError("Qualification probe did not change native hook behavior")
    else:
        raise ValueError("Unsupported qualification policy")
    # Bootstrap proxy is not reference-normalized QoR and is not a result here.
    return {key: value for key, value in measured.items()
            if key not in {"combined_score", "overflow_score", "wirelength_score"}}


def validate_iteration_probe_effect(identity_run, probe_run):
    """Compare the first same-state controller invocation, not a scale flag.

    costheight_ is an int: a small scale change can be logged as changed while
    assignment truncation leaves the actual cost unchanged. The first call on
    each die has the same physical history in this fixed-program pair. Later
    calls may legitimately diverge and are not used as a matched-state proof.
    This check is additional to each native publication/runtime qualification.
    """
    def first_records(run):
        records = {}
        evidence = []
        for die in ("bottom", "top"):
            path = Path(run) / "75_global_route" / die / "route.log"
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing regular iteration-control log: " + die)
            evidence.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)})
            for line in path.read_text().splitlines():
                if "OPEN3D_ITERATION_CONTROL " not in line:
                    continue
                fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", line.split("OPEN3D_ITERATION_CONTROL ", 1)[1]))
                if fields.get("die") != die:
                    raise ValueError("Iteration-control diagnostic die differs")
                values = {name: float(fields[name]) for name in
                          ("calls", "cost_scale", "cost_height", "search_window", "ripup_threshold")}
                if not all(math.isfinite(value) for value in values.values()) or values["calls"] != 1:
                    raise ValueError("Missing finite first-iteration control evidence: " + die)
                if values["cost_height"] <= 0 or not values["cost_height"].is_integer():
                    raise ValueError("Invalid effective integer cost height: " + die)
                records[die] = values
                break
        return records, evidence

    identity, identity_logs = first_records(identity_run)
    probe, probe_logs = first_records(probe_run)
    if not identity or set(identity) != set(probe):
        raise ValueError("Identity/probe first-iteration reachability differs or is absent")
    comparisons = {}
    for die, before in identity.items():
        after = probe[die]
        if before["cost_scale"] != 1.0 or after["cost_scale"] != 2.0:
            raise ValueError("Unexpected fixed iteration probe scale: " + die)
        if (after["cost_height"] == before["cost_height"]
                or after["cost_height"] != 2 * before["cost_height"]):
            raise ValueError("Iteration probe did not double effective native cost_height: " + die)
        if any(before[name] != after[name] for name in ("search_window", "ripup_threshold")):
            raise ValueError("Unchanged first-iteration controls differ: " + die)
        comparisons[die] = {"iteration": 1, "identity_cost_height": before["cost_height"],
                            "probe_cost_height": after["cost_height"]}
    return {"status": "passed", "method": "same-die-first-iteration-effective-integer-cost",
            "comparisons": comparisons, "identity_logs": identity_logs, "probe_logs": probe_logs}


def qualify(source, archive_root, case, policy, candidate_dir, output, task=TASK):
    contract = task_contract(task, policy)
    source, archive_root, candidate_dir = (Path(p).resolve(strict=True)
                                          for p in (source, archive_root, candidate_dir))
    output = _fresh(output, source_root=str(source), reference_root=str(archive_root),
                    benchmark_input=str(candidate_dir))
    output.mkdir(parents=True, exist_ok=False)
    report = {"kind": "archive-grt-hook-qualification-v1", "case": case, "policy": policy,
        "task": task, "status": "started", "model_calls": 0, "candidate_finals": 0,
        "generated_model_candidates": 0, "verifier_rejections": 0,
        "qualification_programs": 1, "native_dispatch_attempts": 0, "native_physical_attempts": 0,
        "native_attempt_definition": "One started two-die GRT backend process, not each inner OpenROAD process.",
        "expected_inner_process_roles": ["bottom-SDC-buffer-proof", "top-SDC-buffer-proof",
                                         "bottom-GRT", "top-GRT"],
        "successful_evaluations": 0, "verified_improvements": 0,
        "evaluation_timeout_seconds": TIMEOUT_SECONDS,
        "not_claimed": ["harness final", "routed signoff", "end-to-end algorithm equivalence",
                        "general historical source bridge", "optimization improvement"]}
    _write(output / "qualification-result.json", report)
    try:
        candidate = load_candidate(candidate_dir)
        if candidate != expected_candidate(source, policy, task):
            raise ValueError("Candidate differs from the fixed qualification program")
        receipt_path = os.environ.get(PREBUILT_OPENROAD_ENV)
        if not receipt_path:
            raise ValueError("Qualification requires a separately built image-installed runtime receipt")
        receipt = json.loads(Path(receipt_path).read_text())
        if receipt.get("task") != task or receipt.get("candidate_sha256") != _digest(candidate):
            raise ValueError("Installed build receipt does not identify this qualification program")
        inputs = qualify_inputs(source, archive_root, case)
        report.update(inputs)
        payload, projected = inputs["archive_qualification"], inputs["projected_sdc_sha256"]
        run = output / "run"
        report["copied_authenticated_prefix"] = copy_prefix(payload, run)
        workspace = output / "workspace"
        isolated_clone(source, workspace, mutable_submodules=["OpenROAD-3D/openroad"])
        patch = output / "candidate.patch"
        patch.write_text(candidate["source_patch"])
        apply_candidate_patch(workspace, patch)
        executable = prepare_openroad_runtime(workspace, output, [contract["source"]])
        if executable is None:
            raise ValueError("No verified installed OpenROAD runtime")
        config = yaml.safe_load((Path(payload["baseline"]["source_run"]) / "config.yaml").read_text())
        command = command_for(workspace, run, executable, config, payload["source_contract"]["grt_configuration"])
        report["command"] = command
        _verify_evidence(payload["evidence"])
        _verify_evidence(inputs["additional_evidence"])
        report["native_dispatch_attempts"] = 1
        report["native_physical_attempts"] = None
        report["executed_stages"] = ["global_routing"]
        _write(output / "qualification-result.json", report)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(workspace)
        started = time.monotonic()
        try:
            with (output / "native-grt.log").open("w") as log:
                process = subprocess.run(command, cwd=workspace, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS, check=False, text=True)
            report["native_physical_attempts"] = 1
        except subprocess.TimeoutExpired:
            report["native_physical_attempts"] = 1
            raise
        except OSError:
            report["native_physical_attempts"] = 0
            raise
        finally:
            report["native_dispatch_wall_seconds"] = time.monotonic() - started
        report["native_exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError("Native GRT failed; the attempt is retained and must not be retried")
        report["measured"] = validate_measurement(run, payload["baseline"]["raw_metrics"], policy, projected, task)
        report["diagnostic_semantics"] = (
            "Hook counters are bounded logged observations, not guaranteed final invocation totals. "
            "Iteration policy_changed is per-die presence of a changed controller decision, not a total. "
            "Qualification requires observed reachability and a nonidentity hook decision, not QoR gain.")
        _verify_evidence(payload["evidence"])
        _verify_evidence(inputs["additional_evidence"])
        report.update(status="passed", successful_evaluations=1)
    except Exception as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
    report["inner_process_artifacts"] = []
    for die in ("bottom", "top"):
        for role, relative in (("SDC-buffer-proof", "sdc_evidence/" + die + "/proof.log"),
                               ("GRT", die + "/route.log")):
            path = output / "run/75_global_route" / relative
            report["inner_process_artifacts"].append({"role": die + "-" + role, "log": str(path),
                "log_present": path.is_file(), "interpretation": "Artifact presence alone is not a successful-process count."})
    _write(output / "qualification-result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("materialize", "run"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--task", default=TASK, choices=tuple(TASKS))
    parser.add_argument("--policy", required=True, choices=("identity", "reversed", "probe"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--case", choices=("bp_fe", "bp", "ariane133", "swerv_wrapper"))
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args(argv)
    if args.action == "materialize":
        print(materialize(args.source, args.policy, args.output, args.task))
        return 0
    if not args.archive or not args.case or not args.candidate:
        parser.error("run requires --archive, --case and --candidate")
    result = qualify(args.source, args.archive, args.case, args.policy, args.candidate, args.output, args.task)
    print(json.dumps({key: result[key] for key in ("status", "native_physical_attempts", "candidate_finals")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
