#!/usr/bin/env python3
"""Small real model/tool/isolation smoke; no EDA optimization is performed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_bench.workspace_codex import agent_container_command, run_workspace_turn


ISOLATION_CHECKS = frozenset({
    "workspace_readable", "workspace_writable", "scratch_writable",
    "protected_read_only", "git_read_only", "input_read_only",
    "docker_socket_absent", "host_path_absent", "network_loopback_only",
})

# This source is controlled by the qualification driver, not by the model.
# Opening a protected file for append must fail with EROFS. If it unexpectedly
# opens, close it without writing: the failed probe never mutates that input.
ISOLATION_PROBE = r'''
import errno
import json
import os
from pathlib import Path
import sys
import tempfile

def read_only(path):
    if not Path(path).is_file():
        return {"passed": False, "errno": None, "reason": "missing file"}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    except OSError as exc:
        return {"passed": exc.errno == errno.EROFS, "errno": exc.errno}
    else:
        os.close(descriptor)
        return {"passed": False, "errno": None, "reason": "write open succeeded"}

def writable(path):
    try:
        with tempfile.TemporaryFile(dir=path) as stream:
            stream.write(b"TRUSTED_PROBE")
            stream.seek(0)
            return stream.read() == b"TRUSTED_PROBE"
    except OSError:
        return False

denials = {
    "protected_read_only": read_only("/workspace/protected/verifier.txt"),
    "git_read_only": read_only("/workspace/.git/config"),
    "input_read_only": read_only("/inputs/benchmark/input.txt"),
}
checks = {
    "workspace_readable": Path("/workspace/protected/verifier.txt").read_text() == "TRUSTED_UNCHANGED\n",
    "workspace_writable": writable("/workspace"),
    "scratch_writable": writable("/scratch"),
    "docker_socket_absent": not os.path.lexists("/var/run/docker.sock"),
    "host_path_absent": not os.path.lexists(sys.argv[1]),
    "network_loopback_only": sorted(os.listdir("/sys/class/net")) == ["lo"],
}
checks.update({name: result["passed"] for name, result in denials.items()})
print(json.dumps({"checks": checks, "write_denials": denials}, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 1)
'''


def trusted_isolation_probe(container, host_canary, output):
    """Verify fixed container boundaries without using model-authored results."""
    command = ["docker", "exec", container, "/usr/bin/python3", "-c",
               ISOLATION_PROBE, str(host_canary)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    try:
        payload = json.loads(process.stdout)
    except (TypeError, ValueError):
        payload = None
    checks = payload.get("checks") if isinstance(payload, dict) else None
    success = (process.returncode == 0 and isinstance(checks, dict)
               and set(checks) == ISOLATION_CHECKS
               and all(value is True for value in checks.values()))
    receipt = {"success": success, "command": command,
               "exit_code": process.returncode, "stdout": process.stdout,
               "stderr": process.stderr, "observations": payload,
               "provenance": "trusted-driver-fixed-probe"}
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    if not success:
        raise RuntimeError("Trusted container isolation qualification failed")
    return receipt


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("subject", "control", "scratch", "results", "benchmark", "reference"):
        (root / name).mkdir()
    (root / "scratch/home").mkdir()
    subject = root / "subject"
    (subject / "protected").mkdir()
    (subject / "protected/verifier.txt").write_text("TRUSTED_UNCHANGED\n")
    (root / "benchmark/input.txt").write_text("FROZEN_INPUT\n")
    (root / "host-canary.txt").write_text("HOST_ONLY_UNCHANGED\n")
    subprocess.run(["git", "init", str(subject)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(subject), "add", "."], check=True)
    subprocess.run(["git", "-C", str(subject), "-c", "user.name=Benchmark",
                    "-c", "user.email=benchmark@localhost", "commit", "-m", "Smoke fixture"],
                   check=True, capture_output=True)
    git_config_before = _sha256(subject / ".git/config")
    name = "o3db-agent-smoke-" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    command = agent_container_command(name=name, image=args.image, subject=subject,
        control=root / "control", scratch=root / "scratch", results=root / "results",
        benchmark=root / "benchmark", reference=root / "reference", protected=("protected",),
        gpu=args.gpu, cpus=1, memory_gb=32)
    (root / "container-command.json").write_text(json.dumps(command, indent=2) + "\n")
    startup = subprocess.run(command, capture_output=True, text=True)
    (root / "container-startup.json").write_text(json.dumps({"exit_code": startup.returncode,
        "stdout": startup.stdout, "stderr": startup.stderr}, indent=2) + "\n")
    if startup.returncode:
        raise RuntimeError("Container startup failed: " + startup.stderr.strip())
    prompt = '''This is a tiny tooling qualification, not an EDA task. Use only the
configured terminal MCP tool. It executes inside an isolated container. Run one
short Python command there to check all of the following: /workspace is readable;
write /workspace/smoke.txt with exactly WORKSPACE_OK; attempts to append to
/workspace/protected/verifier.txt, /workspace/.git/config and
/inputs/benchmark/input.txt fail with permission/read-only errors; /var/run/docker.sock
does not exist; the supplied host-only path is inaccessible. Write the booleans
to /scratch/isolation.json and assert that every check passes. Do not run EDA,
install software, inspect credentials or use the network. Reply exactly OK after
the checks succeed. The host-only path is: ''' + str(root / "host-canary.txt")
    try:
        trusted_probe = trusted_isolation_probe(name, root / "host-canary.txt",
                                                root / "trusted-isolation.json")
        first = run_workspace_turn(binary=args.codex_bin, control=root / "control",
            output=root / "call-1", container=name, prompt=prompt,
            terminal_script=ROOT / "global_bench/container_terminal.py", python=sys.executable,
            timeout_seconds=180)
        if not first["success"] or not first["thread_id"] or first["final_message"].strip() != "OK":
            raise RuntimeError("Initial workspace model/tool qualification failed")
        followup = '''Continue this same tooling qualification. Using only terminal MCP,
read /workspace/smoke.txt and verify that it is exactly WORKSPACE_OK. Append exactly
_RESUMED to it. Do not run EDA. Reply exactly OK.'''
        second = run_workspace_turn(binary=args.codex_bin, control=root / "control",
            output=root / "call-2", container=name, prompt=followup,
            terminal_script=ROOT / "global_bench/container_terminal.py", python=sys.executable,
            thread_id=first["thread_id"], timeout_seconds=180)
        # Agent observations are useful diagnostics, but never qualify isolation.
        try:
            isolated = json.loads((root / "scratch/isolation.json").read_text())
        except (OSError, ValueError):
            isolated = None
        git_config_after = _sha256(subject / ".git/config")
        checks = {
            "trusted_container_isolation": trusted_probe["success"],
            "initial_turn": first["success"] and first["terminal_tool_calls"] > 0,
            "resumed_turn": second["success"] and second["terminal_tool_calls"] > 0,
            "same_session": first["thread_id"] == second["thread_id"],
            "workspace_edit": (subject / "smoke.txt").read_text() == "WORKSPACE_OK_RESUMED",
            "protected_unchanged": (subject / "protected/verifier.txt").read_text() == "TRUSTED_UNCHANGED\n",
            "input_unchanged": (root / "benchmark/input.txt").read_text() == "FROZEN_INPUT\n",
            "host_canary_unchanged": (root / "host-canary.txt").read_text() == "HOST_ONLY_UNCHANGED\n",
            "git_config_unchanged": git_config_before == git_config_after,
            "second_final": second["final_message"].strip() == "OK",
        }
        result = {"success": all(checks.values()), "checks": checks,
                  "isolation": trusted_probe["observations"], "agent_isolation_diagnostic": isolated,
                  "git_config_sha256_before": git_config_before,
                  "git_config_sha256_after": git_config_after,
                  "model": "gpt-5.6-sol", "exit_code": 0, "timed_out": False,
                  "final_message": "OK", "qualification": "workspace-terminal-and-resume-v1",
                  "image": args.image, "eda_executed": False,
                  "scope": "Container mount isolation and persistent workspace editing; not a native host-tool inventory audit"}
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["success"]:
            raise RuntimeError("Workspace qualification checks failed")
        print(json.dumps(result), flush=True)
    finally:
        # Terminate only this private smoke container, including background work.
        subprocess.run(["docker", "stop", "--time", "5", name], capture_output=True, timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--codex-bin", required=True)
    qualify(parser.parse_args())


if __name__ == "__main__":
    main()
