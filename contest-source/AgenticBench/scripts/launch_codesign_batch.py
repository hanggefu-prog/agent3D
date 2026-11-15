#!/usr/bin/env python3
"""Launch one server's shard of a matched co-design batch in tmux.

Generate separate manifests for independent servers. One case stays on one server
so its two methods always share the exact same input/reference files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic3d.codex_exec import CodexExecConfig, run_codex
from agentic3d.knowledge import write_json
from agentic3d.codesign_endpoint import DEFAULT_ROUNDS, PROFILES, TARGETS, validate_rounds
from scripts.codesign_runtime import CASES, Runtime, validate_min_gpu_free_mb
from scripts.run_codesign_experiment import immutable_json, qualify_reference

METHODS = ("codex", "agentic3d")
DEFAULT_MODEL_ALLOWLIST = ("gpt-5.6-sol",)
MODEL_ID = r"[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}"


def matrix(cases=CASES, rounds=DEFAULT_ROUNDS):
    validate_rounds(rounds)
    if not cases or len(set(cases)) != len(cases) or not set(cases) <= set(CASES):
        raise ValueError("invalid cases")
    return [{"case": case, "objective": objective, "method": method, "final_budget": rounds}
            for case in cases for objective in PROFILES for method in METHODS]


def output_root(config, runtime):
    """A versioned recovery can share the original frozen inputs/GPU locks."""
    return runtime.contained(Path(config.get("run_root", config["batch_root"])))


def selected_objectives(config):
    values = config.get("objectives", list(PROFILES))
    if not values or len(set(values)) != len(values) or not set(values) <= set(PROFILES):
        raise ValueError("invalid endpoint subset")
    return values


def selected_methods(config):
    values = config.get("methods", list(METHODS))
    if not values or len(set(values)) != len(values) or not set(values) <= set(METHODS):
        raise ValueError("invalid method subset")
    return values


def selected_rounds(config):
    return validate_rounds(config.get("rounds", DEFAULT_ROUNDS))


def selected_share_gpu(config):
    value = config.get("share_gpu", False)
    if type(value) is not bool:
        raise ValueError("share_gpu must be a JSON boolean")
    return value


def selected_min_gpu_free_mb(config):
    return validate_min_gpu_free_mb(config.get("min_gpu_free_mb", 16000))


def selected_model(config, requested=None):
    """One model per matched batch, from an explicit config allowlist."""
    allowlist = config.get("model_allowlist", list(DEFAULT_MODEL_ALLOWLIST))
    if (not isinstance(allowlist, list) or not allowlist or len(set(allowlist)) != len(allowlist)
            or not all(isinstance(m, str) and re.fullmatch(MODEL_ID, m) for m in allowlist)):
        raise ValueError("model_allowlist must list unique model identifiers")
    model = config.get("model") if requested is None else requested
    if model not in allowlist:
        raise ValueError("matched-model batch model is not in model_allowlist: " + ", ".join(allowlist))
    return model


def run_group(config, runtime, case, objective):
    rounds = selected_rounds(config)
    batch = Path(config["batch_root"])
    root = output_root(config, runtime) / case / objective
    root.mkdir(parents=True, exist_ok=True)
    status = root / "status.json"
    try:
        write_json(status, {"state": "qualifying-reference", "updated_at": time.time()})
        candidate = root / "reference-candidate"
        source = runtime.framework / "evaluation_profiles" / PROFILES[objective] / "initial_candidate/candidate.json"
        immutable_json(candidate / "candidate.json", json.loads(source.read_text()))
        bundle = batch / case / "input"
        adopted = config.get("reference_runs", {}).get(case + "/" + objective)
        if adopted:
            reference = runtime.contained(Path(adopted))
            result = qualify_reference(case, objective, bundle, reference)
            from agentic3d.knowledge import digest
            immutable_json(root / "reference_adoption.json", {
                "run": str(reference), "result_sha256": digest(result),
                "original_contract": json.loads((reference.parents[1] / "evaluation_contract.json").read_text()),
                "qualified_by_source": runtime.source_commit,
                "reason": config["reference_adoption_reason"],
            })
        else:
            runtime.evaluate(TARGETS[objective], candidate, root / "reference", "bundle", bundle,
                             None, case, root / "reference.log")
            reference = root / "reference/result/run"
            qualify_reference(case, objective, bundle, reference)
        write_json(status, {"state": "reference-qualified", "updated_at": time.time()})
        if config.get("prepare_only"):
            return
        children = []
        # Independent model calls overlap; physical evaluations queue at shared GPU locks.
        for method in selected_methods(config):
            output = root / method
            command = [sys.executable, str(ROOT / "scripts/run_codesign_experiment.py"),
                       "--mode", method, "--objective", objective, "--case", case, "--rounds", str(rounds),
                       "--output", str(output), "--batch-root", str(batch), "--source", str(runtime.source),
                       "--bundle", str(bundle), "--reference-run", str(reference),
                       "--gpus", config["gpus"], "--image", runtime.image,
                       "--codex-bin", config["codex_bin"], "--model", config["model"],
                       "--cpus", str(runtime.cpus), "--memory-gb", str(runtime.memory_gb),
                       "--min-gpu-free-mb", str(runtime.min_gpu_free_mb)]
            if runtime.share_gpu:
                command.append("--share-gpu")
            if (output / "run_config.json").exists():
                command.append("--resume")
            stream = (root / (method + ".log")).open("a")
            children.append((method, subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT), stream))
        write_json(status, {"state": "methods-started", "pids": {m: p.pid for m, p, _ in children}, "updated_at": time.time()})
        results = {}
        for method, child, stream in children:
            results[method] = child.wait()
            stream.close()
        write_json(status, {"state": "completed" if all(v == 0 for v in results.values()) else "needs-attention",
                            "exit_codes": results, "updated_at": time.time()})
    except Exception as error:
        write_json(status, {"state": "blocked", "error": str(error), "updated_at": time.time()})
        raise


def supervise(config):
    selected_rounds(config)  # Reject invalid budgets before model or physical preflight.
    runtime = Runtime(Path(config["source"]), Path(config["batch_root"]), config["image"],
                      config["gpus"], config["cpus"], config["memory_gb"],
                      share_gpu=selected_share_gpu(config),
                      min_gpu_free_mb=selected_min_gpu_free_mb(config))
    batch = output_root(config, runtime)
    batch.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (batch / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(batch / "status.json", {"state": "preflight", "updated_at": time.time()})
        if not config.get("prepare_only"):
            # Fail BEFORE expensive reference work if the model is unavailable. No credential copying,
            # provider substitution, login retry loop or bypass of provider restrictions.
            probe_dir = batch / "preflight" / str(time.time_ns())
            probe_dir.mkdir(parents=True)
            probe = run_codex("Connectivity check only. Do not use tools. Reply exactly OK.",
                CodexExecConfig(workspace=probe_dir, run_dir=probe_dir / "call",
                    codex_binary=config["codex_bin"], model=config["model"], sandbox="read-only",
                    config_overrides=("model_reasoning_effort=low",), ephemeral=True,
                    skip_git_repo_check=True, timeout_seconds=120))
            if not probe.success or probe.final_message.strip() != "OK":
                write_json(batch / "status.json", {"state": "blocked-api", "evidence": probe.metadata_path})
                raise RuntimeError("model preflight failed; no experiments started")
        write_json(batch / "status.json", {"state": "preparing-inputs", "updated_at": time.time()})
        failures = []
        # Preparation overlaps across cases; each finished input unlocks both objectives immediately.
        with ThreadPoolExecutor(max_workers=6) as objectives, ThreadPoolExecutor(max_workers=3) as inputs:
            prepared = {inputs.submit(runtime.prepare, case, runtime.batch / case / "input"): case for case in config["cases"]}
            jobs = []
            for future in as_completed(prepared):
                case = prepared[future]
                try:
                    future.result()
                except Exception as error:
                    failures.append({"case": case, "phase": "input", "error": str(error)})
                    write_json(batch / case / "status.json", failures[-1])
                    continue
                jobs += [objectives.submit(run_group, config, runtime, case, objective) for objective in selected_objectives(config)]
            for future in as_completed(jobs):
                try:
                    future.result()
                except Exception as error:
                    failures.append({"phase": "reference-or-method", "error": str(error)})
        # Child exit codes are checked through group status as well as Python exceptions.
        states = [json.loads(p.read_text()) for p in batch.glob("*/*/status.json")]
        attention = failures or any(s.get("state") != ("reference-qualified" if config.get("prepare_only") else "completed") for s in states)
        write_json(batch / "status.json", {"state": "needs-attention" if attention else "completed",
                    "failures": failures, "groups": states, "updated_at": time.time()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--launch", action="store_true", help="Otherwise validate/show the plan only")
    p.add_argument("--model", help="Batch model; must be in the config model_allowlist")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    config["share_gpu"] = selected_share_gpu(config)
    config["min_gpu_free_mb"] = selected_min_gpu_free_mb(config)
    groups = [g for g in matrix(config["cases"], selected_rounds(config))
              if g["objective"] in selected_objectives(config) and g["method"] in selected_methods(config)]
    config["model"] = selected_model(config, args.model)
    if not re.fullmatch(r"codesign-[a-z0-9-]+", config["session"]):
        raise ValueError("invalid tmux session name")
    if args.worker:
        supervise(config)
        return
    plan = {**config, "groups": groups, "candidate_final_budget": sum(g["final_budget"] for g in groups),
            "reference_budget": len(selected_objectives(config)) * len(config["cases"]), "phase": "independent",
            "knowledge": "fresh maintained seed per group; no evolved imports",
            "inner": "Agentic3D max 3 per outer round, first strict improvement; Codex none"}
    if not args.launch:
        print(json.dumps(plan, indent=2))
        return
    runtime = Runtime(Path(config["source"]), Path(config["batch_root"]), config["image"],
                      config["gpus"], config["cpus"], config["memory_gb"],
                      share_gpu=selected_share_gpu(config),
                      min_gpu_free_mb=selected_min_gpu_free_mb(config))
    # Resuming a detached supervisor is explicit; completed/cached attempts are never repeated.
    batch = output_root(config, runtime)
    immutable_json(batch / "batch_manifest.json", {**plan, "source_commit": runtime.source_commit})
    immutable_json(batch / "server_config.json", config)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(batch / "server_config.json")]
    shell = shlex.join(command) + " >> " + shlex.quote(str(batch / "supervisor.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", config["session"], shell], check=True)
    print(json.dumps({"state": "tmux-started-not-yet-qualified", "session": config["session"], "batch": str(batch)}))


if __name__ == "__main__":
    main()
