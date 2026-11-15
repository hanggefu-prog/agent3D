#!/usr/bin/env python3
"""Detach one matched baseline after the driver's strict case-reference gate.

This launcher does not schedule or bypass qualification. Existing output remains
an error unless --resume is explicitly requested; the driver owns round counting.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NODES = {
    "ta2": ("ta2-gcr", "/data/gcr/agentic3d-compute-20260908"),
    "ta3": ("ta3-gcr", "/data/gcr/agentic3d-compute-20260908"),
    "amax3": ("amax3-gcr", "/data/agentic3d/agentic3d-compute-20260908"),
}


def command_for(case, node, mode, gpu, *, resume=False):
    host, node_root = NODES[node]
    remote = node_root + "/sta-cases/" + case
    output = ROOT / "output/sta_codesign_new_cases_20260909" / case / mode
    command = [sys.executable, "-u", str(ROOT / "scripts/run_sta_codesign_comparison.py"),
               "--case", case, "--mode", mode, "--gpu", str(gpu), "--rounds", "5",
               "--model", "gpt-5.6-sol", "--output-dir", str(output),
               "--remote-host", host, "--remote-bundle", remote + "/benchmark-input",
               "--reference-run", remote + "/reference/run",
               "--remote-comparison", remote + "/comparison",
               "--remote-helper", node_root + "/tools-sta/sta_eval_node.sh",
               "--remote-profile-config", node_root + "/framework-sta/local_bench/benchmark_inputs/"
               + case + "_lol_ng45_tp_gnn_sta/config.yaml",
               "--ssh-proxy-command", "docker exec -i nju-easyconnect busybox nc %h %p"]
    if resume:
        command.append("--resume")
    return command, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=("bp", "ariane133", "swerv_wrapper"))
    parser.add_argument("--node", required=True, choices=tuple(NODES))
    parser.add_argument("--mode", required=True, choices=("codex", "agentic3d"))
    parser.add_argument("--gpu", required=True, type=int, choices=range(8))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        raise ValueError("attempt must be positive")
    command, output = command_for(args.case, args.node, args.mode, args.gpu, resume=args.resume)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / (args.mode + ".launch-attempt-" + str(args.attempt) + ".log")
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUNBUFFERED="1")
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print("pid={} case={} mode={} log={}".format(process.pid, args.case, args.mode, log))


if __name__ == "__main__":
    main()
