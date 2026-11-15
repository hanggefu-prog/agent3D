#!/usr/bin/env python3
"""Detach resumed comparison drivers from the invoking terminal's process group."""

import json
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("codex", "agentic3d"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "output/sta_codesign_bp_multi_20260903-v2-exact-sta"
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUNBUFFERED="1")
    remote = "/nas1/gcr/agentic3d-dev/runs/sta_codesign_resume_20260908"
    env["OPENAGENTIC3D_STA_COMPARISON"] = remote + "/comparison-v8-exact-cutpoint-two-pass"
    env["OPENAGENTIC3D_STA_REMOTE_HELPER"] = remote + "/run_sta_codesign_remote_eval.sh"
    for mode, gpu in ((args.mode, 0 if args.mode == "codex" else 1),):
        directory = base / mode
        log = directory / "resume-20260908-nas.log"
        command = [sys.executable, "-u", str(root / "scripts/run_sta_codesign_comparison.py"),
                   "--mode", mode, "--output-dir", str(directory), "--gpu", str(gpu),
                   "--model", "gpt-5.6-sol", "--resume"]
        with log.open("x") as output:
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (directory / "resume-20260908-nas.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"mode": mode, "pid": process.pid, "log": str(log)}), flush=True)


if __name__ == "__main__":
    main()
