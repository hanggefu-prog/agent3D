"""Run participant code with physical tools but without organizer authority."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

from grt_bench.runtime import mount, stop_and_confirm


IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
NAME = re.compile(r"o3db-contest-agent-[a-z0-9]{12,32}\Z")


def command(*, name, image, agent_dir, workspaces, scratch, bridge_dir,
            framework, source, training_inputs, evaluations=None, cpus=4, memory_gb=32, gpu=None):
    if not NAME.fullmatch(name) or not IMAGE.fullmatch(image):
        raise ValueError("Agent name and image digest must be organizer-pinned")
    if type(cpus) is not int or not 1 <= cpus <= 16 or type(memory_gb) is not int or not 4 <= memory_gb <= 128:
        raise ValueError("Invalid agent resource limit")
    paths = [Path(value) for value in (agent_dir, workspaces, scratch, bridge_dir, framework)]
    if any(path.is_symlink() or not path.is_dir() for path in paths):
        raise ValueError("Agent mounts must be real directories")
    resolved = [path.resolve(strict=True) for path in paths]
    for index, writable in enumerate(resolved[:3]):
        if any(writable == other or writable in other.parents or other in writable.parents
               for other in resolved[index + 1:]):
            raise ValueError("Writable Agent mounts must not overlap protected mounts")
    if not (Path(bridge_dir) / "bridge.sock").is_socket():
        raise ValueError("Organizer bridge socket is missing")
    for name_part in ("tmp", "home", "cache"):
        child = Path(scratch) / name_part
        if child.is_symlink():
            raise ValueError("Agent scratch cannot contain aliased directories")
        child.mkdir(exist_ok=True)
        if not child.is_dir():
            raise ValueError("Agent scratch directory is invalid")
    if not isinstance(training_inputs, dict) or set(training_inputs) - {
            "bp_fe-ng45", "bp-ng45", "ariane133-ng45"}:
        raise ValueError("Only public training inputs may be mounted")
    argv = ["docker", "run", "--detach", "--rm", "--pull=never", "--name", name,
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "512",
            "--cpus", str(cpus), "--memory", str(memory_gb) + "g",
            "--memory-swap", str(memory_gb) + "g",
            "--user", "{}:{}".format(os.getuid(), os.getgid()),
            "--workdir", "/workspace/agent"]
    argv += mount(agent_dir, "/workspace/agent", False)
    argv += mount(workspaces, "/workspace/workspaces", False)
    argv += mount(scratch, "/workspace/scratch", False)
    argv += mount(scratch / "tmp", "/tmp", False)
    argv += mount(bridge_dir, "/run/contest", True)
    argv += mount(framework, "/opt/contest/AgenticBench", True)
    guide = Path(framework).parent / "README.md"
    if guide.is_file() and not guide.is_symlink():
        argv += mount(guide, "/workspace/README.md", True)
    for directory, target in ((source, "/opt/contest/source"),
                              (evaluations, "/workspace/evaluations")):
        if directory is None:
            continue
        path = Path(directory)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Project and evaluation mounts must be real directories")
        protected = path.resolve(strict=True)
        if any(protected == writable or protected in writable.parents or writable in protected.parents
               for writable in resolved[:3]):
            raise ValueError("Writable Agent mounts cannot overlap project or evaluation files")
        argv += mount(path, target, True)
    for case, directory in sorted(training_inputs.items()):
        path = Path(directory)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Training input must be a real directory")
        target = path.resolve(strict=True)
        if any(target == writable or target in writable.parents or writable in target.parents
               for writable in resolved[:3]):
            raise ValueError("Writable Agent mounts cannot overlap training inputs")
        argv += mount(path, "/inputs/" + case, True)
    if gpu is not None:
        if not isinstance(gpu, str) or not gpu.isdigit():
            raise ValueError("Invalid agent GPU device")
        argv += ["--gpus", "device=" + gpu]
    for item in ("CONTEST_BRIDGE_SOCKET=/run/contest/bridge.sock",
                 "PYTHONPATH=/opt/contest/AgenticBench:/workspace/agent:/opt/dreamplace:/opt/dreamplace/dreamplace",
                 "HOME=/workspace/scratch/home", "TMPDIR=/tmp",
                 "XDG_CACHE_HOME=/workspace/scratch/cache",
                 "PYTHONDONTWRITEBYTECODE=1", "PYTHONPYCACHEPREFIX=/tmp/python-cache"):
        argv += ["--env", item]
    return argv + ["--entrypoint", "sleep", image, "infinity"]


def run_agent(*, startup_command, name, scratch, timeout_seconds=None):
    """Start one owned container, run Agent.run, then confirm full teardown."""
    scratch = Path(scratch)
    for folder in ("tmp", "home", "cache"):
        (scratch / folder).mkdir(parents=True, exist_ok=True)
    (scratch / "container-command.json").write_text(json.dumps(startup_command, indent=2) + "\n")
    attempted = False
    try:
        attempted = True
        startup = subprocess.run(startup_command, capture_output=True, text=True)
        (scratch / "container-startup.json").write_text(json.dumps({
            "returncode": startup.returncode, "stdout": startup.stdout, "stderr": startup.stderr}) + "\n")
        if startup.returncode:
            raise RuntimeError("Agent container did not start")
        with (scratch / "agent.stdout").open("w") as stdout, (scratch / "agent.stderr").open("w") as stderr:
            process = subprocess.run(["docker", "exec", name, "python3", "-m",
                                      "contest2.agent_entry"], stdout=stdout, stderr=stderr,
                                     timeout=timeout_seconds, env={"PATH": os.environ.get("PATH", os.defpath)})
        if process.returncode:
            raise RuntimeError("Agent.run failed; see agent.stderr")
    finally:
        if attempted:
            stop_and_confirm(name, scratch)
    result = scratch / "agent-result.json"
    if not result.is_file():
        raise RuntimeError("Agent.run did not return a candidate")
    return json.loads(result.read_text())["candidate_id"]
