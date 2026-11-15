"""Persistent host-authenticated Codex with a container-only terminal surface.

Authentication stays in the existing host Codex client. The model has no host
shell and its only configured external tool executes in a disposable, offline
container. Read-only mounts, not prompt compliance, protect evaluation code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Mapping, Sequence

MODEL = "gpt-5.6-sol"


def _toml(value):
    # JSON strings and scalar arrays are also valid TOML values.
    return json.dumps(value, ensure_ascii=False)


def _bind(source: Path, destination: str, readonly=False):
    source = source.resolve(strict=True)
    if any(c in str(source) for c in (",", "\n", "\x00")):
        raise ValueError("Unsafe bind-mount source")
    return ["--mount", "type=bind,src={},dst={},{}".format(
        source, destination, "readonly" if readonly else "readonly=false")]


def agent_container_command(*, name: str, image: str, subject: Path,
                            control: Path, scratch: Path, results: Path,
                            benchmark: Path, reference: Path, protected: Sequence[str],
                            gpu: str, cpus: int, memory_gb: int,
                            subject_read_only: bool = False,
                            trusted_framework: Path | None = None) -> list[str]:
    """Task source is private; analyzer source, inputs, results and Git are RO."""
    if type(subject_read_only) is not bool:
        raise ValueError("subject_read_only must be an explicit boolean")
    if not re.fullmatch(r"o3db-agent-[a-zA-Z0-9_.-]{1,90}", name):
        raise ValueError("Invalid private agent container name")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Agent image must be pinned")
    if not gpu.isdigit() or not 1 <= cpus <= 32 or not 32 <= memory_gb <= 128:
        raise ValueError("Invalid resource allocation")
    subject = subject.resolve(strict=True)
    if not (subject / ".git").is_dir():
        raise ValueError("Agent subject needs its own standalone Git directory")
    alternates = subject / ".git/objects/info/alternates"
    if alternates.exists() and alternates.read_text().strip():
        raise ValueError("Agent Git history must not reference a host checkout")
    command = ["docker", "run", "--detach", "--rm", "--name", name,
               "--read-only", "--network", "none", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "4096",
               "--cpus", str(cpus), "--memory", f"{memory_gb}g", "--shm-size", "2g",
               "--user", f"{os.getuid()}:{os.getgid()}", "--gpus", f"device={gpu}",
               "--workdir", "/workspace"]
    command += _bind(subject, "/workspace", subject_read_only)
    command += _bind(subject / ".git", "/workspace/.git", True)
    command += _bind(control, "/task", True)
    command += _bind(scratch, "/scratch")
    command += _bind(scratch, "/tmp")
    command += _bind(results, "/results", True)
    command += _bind(benchmark, "/inputs/benchmark", True)
    command += _bind(reference, "/inputs/reference", True)
    for relative in protected:
        if trusted_framework is not None and relative == "AgenticBench":
            continue
        path = subject / relative
        if path.is_symlink():
            raise ValueError("Protected paths must not be symlinks: " + relative)
        if not path.exists():
            continue
        path.resolve().relative_to(subject)
        command += _bind(path, "/workspace/" + relative, True)
    if trusted_framework is not None:
        # A unified repository already contains AgenticBench. It can remain
        # underneath the trusted overlay when the entire source mount is RO;
        # its bytes still participate in the host's before/after scope audit.
        framework_mount = subject / "AgenticBench"
        if (not framework_mount.is_dir() or framework_mount.is_symlink()
                or (not subject_read_only and any(framework_mount.iterdir()))):
            raise ValueError("Trusted framework requires a regular directory and an empty placeholder unless source is read-only")
        command += _bind(Path(trusted_framework), "/workspace/AgenticBench", True)
    for value in ("HOME=/scratch/home", "XDG_CACHE_HOME=/scratch/cache",
                  "MPLCONFIGDIR=/scratch/matplotlib", "PYTHONDONTWRITEBYTECODE=1",
                  "PYTHONPYCACHEPREFIX=/scratch/python-cache",
                  "OPEN3DBENCH_PYTHON=/usr/bin/python3", "TMPDIR=/tmp",
                  "OPEN3D_LOCAL_DREAMPLACE_INSTALL=/opt/dreamplace",
                  "PLACE3D_DREAMPLACE_INSTALL=/opt/dreamplace",
                  "PYTHONPATH=/workspace/AgenticBench:/workspace:/opt/dreamplace:/opt/dreamplace/dreamplace"):
        command += ["--env", value]
    command += [image, "sleep", "infinity"]
    return command


def codex_overrides(*, container: str, control: Path, terminal_script: Path,
                    python: str) -> tuple[str, ...]:
    """Defense in depth: no native host shell, minimal host read permission.

    The apply_patch tool cannot write under this profile even on clients where
    removing its old feature flag no longer removes that tool.
    """
    if not control.is_absolute() or not terminal_script.is_absolute():
        raise ValueError("Use absolute host control/tool paths")
    filesystem = '{":minimal"="read",' + _toml(str(control)) + '="read"}'
    return (
        "approval_policy=\"never\"", "model_reasoning_effort=\"high\"",
        "features.shell_tool=false", "features.unified_exec=false",
        "features.apps=false", "features.plugins=false", "features.hooks=false",
        "features.browser_use=false", "features.computer_use=false",
        "features.multi_agent=false", "agents.enabled=false",
        "features.goals=false", "features.tool_suggest=false",
        "features.workspace_dependencies=false", "features.skill_search=false",
        "features.code_mode_host=true", "features.image_generation=false",
        "web_search=\"disabled\"", "project_doc_max_bytes=0",
        "default_permissions=\"agentic_container_only\"",
        "permissions.agentic_container_only.filesystem=" + filesystem,
        "permissions.agentic_container_only.network.enabled=false",
        "mcp_servers.terminal.command=" + _toml(python),
        "mcp_servers.terminal.args=" + _toml(["-B", str(terminal_script), "--container", container]),
        # Response/connection safeguards only; exec yields without killing commands.
        "mcp_servers.terminal.tool_timeout_sec=320",
        "mcp_servers.terminal.startup_timeout_sec=20",
        "mcp_servers.terminal.required=true", "mcp_servers.terminal.enabled_tools=[\"exec\"]",
        # User-authorized container-only terminal; no other tool gets this grant.
        "mcp_servers.terminal.tools.exec.approval_mode=\"approve\"",
        "mcp_servers.terminal.env_vars=[]",
    )


def build_workspace_command(*, binary: str, control: Path, final: Path,
                            overrides: Sequence[str], thread_id=None, model: str = MODEL) -> list[str]:
    if not isinstance(model, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}", model):
        raise ValueError("Invalid explicit model identifier")
    command = [binary, "exec"]
    if thread_id:
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", thread_id):
            raise ValueError("Invalid Codex session ID")
        command += ["resume"]
    command += ["--ignore-user-config", "--ignore-rules", "--json", "--model", model,
                "--skip-git-repo-check", "--output-last-message", str(final)]
    if not thread_id:
        command += ["--cd", str(control)]
    for setting in overrides:
        command += ["--config", setting]
    if thread_id:
        command += [thread_id]
    return [*command, "-"]


def run_workspace_turn(*, binary: str, control: Path, output: Path, container: str,
                       prompt: str, terminal_script: Path, python: str,
                       thread_id=None, timeout_seconds=None,
                       environment: Mapping[str, str] | None = None,
                       model: str = MODEL, extra_overrides: Sequence[str] = (),
                       output_schema: Path | None = None) -> dict:
    """One planning/editing turn; contest calls have no execution deadline."""
    output.mkdir(parents=True, exist_ok=False)
    final = output / "final.txt"
    overrides = codex_overrides(container=container, control=control,
                                terminal_script=terminal_script, python=python)
    overrides += tuple(extra_overrides)
    command = build_workspace_command(binary=binary, control=control, final=final,
                                      overrides=overrides, thread_id=thread_id, model=model)
    if output_schema is not None:
        command[-1:-1] = ["--output-schema", str(output_schema)]
    (output / "prompt.md").write_text(prompt)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with (output / "events.jsonl").open("w") as events, (output / "stderr.log").open("w") as errors:
        process = subprocess.Popen(command, cwd=control, env=environment,
                                   stdin=subprocess.PIPE, stdout=events, stderr=errors,
                                   text=True, start_new_session=True,
                                   pass_fds=((int(environment["CONTEST_TURN_LEASE_FD"]),)
                                             if environment and environment.get("CONTEST_TURN_LEASE_FD") else ()))
        try:
            process.communicate(prompt, timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            # Only our own bounded client process group; never another experiment.
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            timed_out = True
    completed = False
    observed_thread = thread_id
    tool_events = 0
    for line in (output / "events.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "thread.started":
            observed_thread = event.get("thread_id")
        completed |= event.get("type") == "turn.completed"
        item = event.get("item")
        if (isinstance(item, dict) and item.get("type") == "mcp_tool_call"
                and item.get("server") == "terminal" and item.get("tool") == "exec"):
            tool_events += event.get("type") == "item.completed"
    identity_ok = bool(observed_thread and re.fullmatch(r"[0-9a-fA-F-]{36}", observed_thread))
    identity_ok = identity_ok and (not thread_id or thread_id == observed_thread)
    result = {"model": model, "exit_code": process.returncode, "timed_out": timed_out,
              "thread_id": observed_thread, "duration_seconds": time.monotonic() - started,
              "timing_scope": "Agent-turn wall time including client, network, tool execution and waits; not pure API inference",
              "api_latency_seconds": None,
              "turn_completed": completed, "terminal_tool_calls": tool_events,
              "success": process.returncode == 0 and not timed_out and completed and identity_ok,
              "session_identity_valid": identity_ok,
              "final_message": final.read_text() if final.exists() else ""}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
