#!/usr/bin/env python3
"""Freeze this protocol and launch four independent, five-final STA experiments.

Run --prepare first; this snapshots code but spends no model/STA budget.
Then --launch CASE uses already-qualified remote paths and a new output root.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic3d.knowledge import write_json, digest
from agentic3d.sta_knowledge_runner import seed_snapshot

BATCH = ROOT / "output/sta_knowledge_v2_20260910-contract-fix"
SNAPSHOT = BATCH / "framework"
PROXY = "docker exec -i nju-easyconnect busybox nc %h %p"
POOL = "/data/gcr/agentic3d-compute-20260908"
AMAX = "/data/agentic3d/agentic3d-compute-20260908"
OLD = "/data/gcr/agentic3d-dev/runs/sta_codesign_bp_multi_20260902-v1"
TARGETS = {
    "bp": ("ta2-gcr", POOL, 2),
    "swerv_wrapper": ("ta2-gcr", POOL, 3),
    "ariane133": ("amax3-gcr", AMAX, 0),
    "bp_multi": ("ta1-gcr", "/data/gcr/agentic3d-dev", 0),
}


def command_for(case, resume=False):
    host, node, gpu = TARGETS[case]
    case_root = node + "/sta-cases/" + case
    reference = case_root + "/reference/run"
    bundle = case_root + "/benchmark-input"
    framework = node + "/framework-sta"
    helper = node + "/tools-sta/sta_knowledge_v2.sh"
    if case == "bp_multi":
        reference, bundle = OLD + "/reference-v8-exact-cutpoint-two-pass/run", OLD + "/benchmark-input"
        framework = node + "/framework-sta-codesign-20260902-v1"
        helper = node + "/sta-knowledge-v2-20260909/sta_knowledge_v2.sh"
    output = BATCH / case / "agentic3d"
    command = [sys.executable, "-u", str(SNAPSHOT / "scripts/run_sta_codesign_comparison.py"),
        "--protocol", "knowledge-v2", "--knowledge-phase", "independent",
        "--knowledge-seed", str(BATCH / "seed.json"),
        "--mode", "agentic3d", "--case", case, "--gpu", str(gpu), "--rounds", "5",
        "--model", "gpt-5.6-sol", "--output-dir", str(output),
        "--agent-workspace", str(ROOT / "tmp/sta_codesign_clean_workspace"),
        "--remote-host", host, "--remote-bundle", bundle, "--reference-run", reference,
        "--remote-comparison", node + "/sta-knowledge-v2-20260910-contract-fix/" + case,
        "--remote-helper", helper,
        "--remote-profile-config", framework + "/local_bench/benchmark_inputs/" + case + "_lol_ng45_tp_gnn_sta/config.yaml",
        "--ssh-proxy-command", PROXY]
    if resume:
        command.append("--resume")
    return command, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--launch", choices=tuple(TARGETS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if BATCH.exists():
            raise ValueError("batch already exists; inspect instead of overwriting")
        SNAPSHOT.mkdir(parents=True)
        for name in (
            "agentic3d",
            "evaluation_profiles",
            "local_bench",
            "local_tasks",
            "scripts",
            "skills",
        ):
            shutil.copytree(ROOT / name, SNAPSHOT / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        shutil.copy2(ROOT / "pyproject.toml", SNAPSHOT / "pyproject.toml")
        files = {str(p.relative_to(SNAPSHOT)): p.read_bytes().hex()
                 for p in sorted(SNAPSHOT.rglob("*")) if p.is_file()}
        seed = seed_snapshot(SNAPSHOT)
        write_json(BATCH / "seed.json", seed)
        write_json(BATCH / "batch.json", {"protocol": "knowledge-v2", "phase": "independent",
            "cases": list(TARGETS), "total_final_budget": 20, "per_case_final_budget": 5,
            "model": "gpt-5.6-sol", "reasoning_effort": "xhigh",
            "snapshot_sha256": digest(files), "seed_sha256": digest(seed),
            "knowledge_import": None})
        print("Prepared frozen batch: " + str(BATCH))
        return
    if not (BATCH / "batch.json").is_file():
        raise ValueError("run --prepare first")
    command, output = command_for(args.launch, args.resume)
    if output.exists() and not args.resume:
        raise ValueError("existing experiment; explicit resume required")
    output.parent.mkdir(parents=True, exist_ok=True)
    import time
    log = output.parent / ("launch-" + str(time.time_ns()) + ".log")
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=SNAPSHOT,
            env=dict(os.environ, LC_ALL="C", LANG="C", PYTHONUNBUFFERED="1"),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"case": args.launch, "pid": process.pid, "log": str(log), "command": command}
    write_json(output.parent / "launch.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
