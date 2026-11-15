"""Offline contract tests: no Docker, model, credentials, or network calls."""
import json
import importlib.util
import ast
import errno
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from global_bench import workspace_codex as runtime


THREAD = "019f83ac-f095-79d1-b98d-66f5a8eab24e"
OTHER_THREAD = "019f83ac-f095-79d1-b98d-66f5a8eab24f"
IMAGE = "sha256:" + "a" * 64


def paths(tmp_path):
    values = {}
    for name in ("subject", "control", "scratch", "results", "benchmark", "reference"):
        values[name] = tmp_path / name
        values[name].mkdir()
    (values["subject"] / ".git/objects/info").mkdir(parents=True)
    (values["subject"] / "open3dbench").mkdir()
    (values["subject"] / "Place-3D/src").mkdir(parents=True)
    (values["subject"] / "Place-3D/src/api.py").write_text("# Protected contract\n")
    return values


def container_args(tmp_path):
    return dict(paths(tmp_path), name="o3db-agent-bp-fe-r1", image=IMAGE,
                protected=("open3dbench", "Place-3D/src/api.py"), gpu="1", cpus=4,
                memory_gb=32)


def values_after(command, flag):
    return [command[i + 1] for i, value in enumerate(command[:-1]) if value == flag]


def test_trusted_framework_overlay_in_readonly_unified_repository(tmp_path):
    args = container_args(tmp_path)
    bundled = args["subject"] / "AgenticBench"
    bundled.mkdir()
    (bundled / "evaluator.py").write_text("# Committed framework stays in source inventory.\n")
    trusted = tmp_path / "trusted-framework"
    trusted.mkdir()
    command = runtime.agent_container_command(**args, subject_read_only=True, trusted_framework=trusted)
    mounts = values_after(command, "--mount")
    assert f"type=bind,src={trusted},dst=/workspace/AgenticBench,readonly" in mounts
    assert f"type=bind,src={args['subject']},dst=/workspace,readonly" in mounts
    with pytest.raises(ValueError, match="empty placeholder"):
        runtime.agent_container_command(**args, trusted_framework=trusted)


def test_container_has_private_mounts_and_no_host_auth_or_docker_socket(tmp_path):
    args = container_args(tmp_path)
    command = runtime.agent_container_command(**args)
    assert command[:4] == ["docker", "run", "--detach", "--rm"]
    assert values_after(command, "--network") == ["none"]
    assert values_after(command, "--cap-drop") == ["ALL"]
    assert values_after(command, "--security-opt") == ["no-new-privileges"]
    assert "--read-only" in command
    assert values_after(command, "--gpus") == ["device=1"]
    assert values_after(command, "--cpus") == ["4"]
    assert values_after(command, "--memory") == ["32g"]
    mounts = values_after(command, "--mount")
    mounted = {row.split(",dst=", 1)[1].rsplit(",", 1)[0]: row.rsplit(",", 1)[1]
               for row in mounts}
    assert mounted == {
        "/workspace": "readonly=false", "/workspace/.git": "readonly", "/task": "readonly",
        "/scratch": "readonly=false", "/tmp": "readonly=false", "/results": "readonly",
        "/inputs/benchmark": "readonly", "/inputs/reference": "readonly",
        "/workspace/open3dbench": "readonly", "/workspace/Place-3D/src/api.py": "readonly",
    }
    assert not any("docker.sock" in row or "/.codex" in row or "auth.json" in row for row in mounts)
    assert all("TOKEN" not in row and "API_KEY" not in row for row in values_after(command, "--env"))
    assert "PYTHONPYCACHEPREFIX=/scratch/python-cache" in values_after(command, "--env")
    assert command[-3:] == [IMAGE, "sleep", "infinity"]


@pytest.mark.parametrize("override", [
    {"name": "foreign-container"}, {"name": "o3db-agent-x;touch unsafe"},
    {"image": "open3dbench:release"}, {"gpu": "all"}, {"gpu": "0,1"},
    {"cpus": 0}, {"memory_gb": 1},
])
def test_container_rejects_unpinned_or_invalid_identity_and_resources(tmp_path, override):
    args = container_args(tmp_path)
    args.update(override)
    with pytest.raises(ValueError):
        runtime.agent_container_command(**args)


def test_analyzers_mount_complete_subject_read_only(tmp_path):
    command = runtime.agent_container_command(**container_args(tmp_path), subject_read_only=True)
    mounts = values_after(command, "--mount")
    for mount in mounts:
        destination = mount.split(",dst=", 1)[1].rsplit(",", 1)[0]
        if destination == "/workspace" or destination.startswith("/workspace/"):
            assert mount.endswith(",readonly")
    assert any(",dst=/scratch,readonly=false" in mount for mount in mounts)
    assert any(",dst=/results,readonly" in mount for mount in mounts)


@pytest.mark.parametrize("invalid", [None, "true", 1])
def test_analyzer_read_only_flag_must_not_be_ambiguous(tmp_path, invalid):
    with pytest.raises(ValueError, match="explicit boolean"):
        runtime.agent_container_command(**container_args(tmp_path), subject_read_only=invalid)


def test_container_rejects_worktree_git_pointer_and_external_alternates(tmp_path):
    args = container_args(tmp_path)
    alternate = args["subject"] / ".git/objects/info/alternates"
    alternate.write_text("/host/private/objects\n")
    with pytest.raises(ValueError, match="host checkout"):
        runtime.agent_container_command(**args)
    alternate.unlink()
    # A .git file would refer outside the independently isolated checkout.
    import shutil
    shutil.rmtree(args["subject"] / ".git")
    (args["subject"] / ".git").write_text("gitdir: /host/private/worktree\n")
    with pytest.raises(ValueError, match="standalone Git"):
        runtime.agent_container_command(**args)


def test_container_rejects_protected_symlink_and_mount_source_injection(tmp_path):
    args = container_args(tmp_path)
    symlink = args["subject"] / "external"
    symlink.symlink_to(args["reference"], target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        runtime.agent_container_command(**{**args, "protected": ("external",)})
    strange = tmp_path / "unsafe,readonly"
    strange.mkdir()
    with pytest.raises(ValueError, match="bind-mount source"):
        runtime.agent_container_command(**{**args, "scratch": strange})


def test_codex_overrides_keep_host_read_only_and_mcp_arguments_structured(tmp_path):
    control = tmp_path / "control with spaces"
    tool = tmp_path / "trusted tools/terminal.py"
    overrides = runtime.codex_overrides(container="o3db-agent-test", control=control,
                                        terminal_script=tool, python="/trusted/python")
    settings = dict(setting.split("=", 1) for setting in overrides)
    assert settings["features.shell_tool"] == "false"
    assert settings["features.unified_exec"] == "false"
    for key in ("apps", "plugins", "hooks", "browser_use", "computer_use", "multi_agent",
                "goals", "tool_suggest", "workspace_dependencies", "skill_search",
                "image_generation"):
        assert settings["features." + key] == "false"
    assert settings["agents.enabled"] == "false"
    assert settings["features.code_mode_host"] == "true"
    assert settings["mcp_servers.terminal.tools.exec.approval_mode"] == '\"approve\"'
    assert "tools.view_image" not in settings  # Ignored by CLI 0.156.1.
    assert settings["web_search"] == '"disabled"'
    assert settings["default_permissions"] == '"agentic_container_only"'
    assert settings["permissions.agentic_container_only.network.enabled"] == "false"
    filesystem = settings["permissions.agentic_container_only.filesystem"]
    assert '":minimal"="read"' in filesystem
    assert json.dumps(str(control)) + '="read"' in filesystem
    assert '"write"' not in filesystem and '":root"' not in filesystem
    assert json.loads(settings["mcp_servers.terminal.command"]) == "/trusted/python"
    assert json.loads(settings["mcp_servers.terminal.args"]) == ["-B", str(tool), "--container", "o3db-agent-test"]
    assert settings["mcp_servers.terminal.required"] == "true"
    assert json.loads(settings["mcp_servers.terminal.enabled_tools"]) == ["exec"]
    assert json.loads(settings["mcp_servers.terminal.env_vars"]) == []
    assert "features.apply_patch_freeform" not in settings


@pytest.mark.parametrize("thread", [None, THREAD])
def test_commands_preserve_explicit_session_and_ignore_host_configuration(tmp_path, thread):
    overrides = ("features.shell_tool=false", 'mcp_servers.terminal.args=["/a path/tool.py"]')
    command = runtime.build_workspace_command(binary="/trusted/codex", control=tmp_path / "control",
                                               final=tmp_path / "final.txt", overrides=overrides,
                                               thread_id=thread)
    assert command[:2] == ["/trusted/codex", "exec"]
    assert "--ignore-user-config" in command and "--ignore-rules" in command
    assert "--last" not in command and "--ephemeral" not in command
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert values_after(command, "--config") == list(overrides)
    assert values_after(command, "--model") == [runtime.MODEL]
    if thread:
        assert command[2] == "resume" and command[-2:] == [thread, "-"]
        assert "--cd" not in command
    else:
        assert "resume" not in command and command[-1] == "-"
        assert values_after(command, "--cd") == [str(tmp_path / "control")]


def test_reject_invalid_session_id_before_process_start(tmp_path):
    with pytest.raises(ValueError, match="session ID"):
        runtime.build_workspace_command(binary="codex", control=tmp_path,
                                        final=tmp_path / "final", overrides=(), thread_id="--last")


def simulated_turn(tmp_path, monkeypatch, *, events, thread=None, returncode=0, timeout=False):
    control = tmp_path / "control"
    control.mkdir()
    seen = {}
    should_timeout = timeout
    class Process:
        pid = 98765
        def __init__(self, command, **kwargs):
            self.returncode = returncode
            seen.update(command=command, kwargs=kwargs)
            self.kwargs = kwargs
        def communicate(self, prompt, timeout):
            seen["prompt"] = prompt
            for event in events:
                self.kwargs["stdout"].write(json.dumps(event) + "\n")
            self.kwargs["stdout"].flush()
            if should_timeout:
                raise subprocess.TimeoutExpired(seen["command"], timeout)
        def wait(self, timeout):
            self.returncode = -15
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    killed = []
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    result = runtime.run_workspace_turn(binary="/trusted/codex", control=control,
                                       output=tmp_path / "round", container="o3db-agent-test",
                                       prompt="Plan first; use the previous trusted evaluation feedback.",
                                       terminal_script=tmp_path / "terminal.py", python="/trusted/python",
                                       thread_id=thread, environment={"PATH": "/trusted/bin"})
    return result, seen, killed


def test_turn_records_native_events_without_requiring_program_json(tmp_path, monkeypatch):
    result, seen, killed = simulated_turn(tmp_path, monkeypatch, events=[
        {"type": "thread.started", "thread_id": THREAD},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "terminal", "tool": "exec"}},
        {"type": "turn.completed"},
    ])
    assert result["success"] and result["thread_id"] == THREAD
    assert result["duration_seconds"] >= 0
    assert result["api_latency_seconds"] is None
    assert "not pure API inference" in result["timing_scope"]
    assert result["terminal_tool_calls"] == 1 and not killed
    assert seen["kwargs"]["cwd"] == tmp_path / "control"
    assert seen["kwargs"]["env"] == {"PATH": "/trusted/bin"}
    assert seen["kwargs"]["start_new_session"] is True
    assert seen["prompt"].startswith("Plan first")
    assert json.loads((tmp_path / "round/result.json").read_text()) == result


def test_timeout_only_terminates_own_client_process_group(tmp_path, monkeypatch):
    result, _, killed = simulated_turn(tmp_path, monkeypatch, events=[], timeout=True)
    assert result["timed_out"] and not result["success"]
    assert killed and all(pid == 98765 for pid, _ in killed)


def test_resume_preserves_the_same_explicit_thread(tmp_path, monkeypatch):
    result, seen, _ = simulated_turn(tmp_path, monkeypatch, thread=THREAD, events=[
        {"type": "thread.started", "thread_id": THREAD}, {"type": "turn.completed"},
    ])
    assert result["success"] and result["thread_id"] == THREAD
    assert seen["command"][-2:] == [THREAD, "-"]


@pytest.mark.parametrize("events,code", [([], 0), ([{"type": "turn.completed"}], 1)])
def test_incomplete_or_failed_turn_is_not_successful(tmp_path, monkeypatch, events, code):
    result, _, _ = simulated_turn(tmp_path, monkeypatch, events=events, returncode=code)
    assert not result["success"]


def test_resume_must_not_accept_a_different_thread(tmp_path, monkeypatch):
    result, _, _ = simulated_turn(tmp_path, monkeypatch, thread=THREAD, events=[
        {"type": "thread.started", "thread_id": OTHER_THREAD}, {"type": "turn.completed"},
    ])
    assert not result["success"]


def test_first_turn_must_publish_a_persistable_thread_id(tmp_path, monkeypatch):
    result, _, _ = simulated_turn(tmp_path, monkeypatch, events=[{"type": "turn.completed"}])
    assert not result["success"]


def qualification_module():
    path = Path(__file__).resolve().parents[1] / "scripts/qualify_workspace_codex.py"
    spec = importlib.util.spec_from_file_location("workspace_qualification_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trusted_probe_uses_fixed_source_and_exact_checks(tmp_path, monkeypatch):
    module = qualification_module()
    observations = {"checks": {name: True for name in module.ISOLATION_CHECKS},
                    "write_denials": {"git_read_only": {"passed": True, "errno": 30}}}
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(observations), stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "trusted-isolation.json"
    receipt = module.trusted_isolation_probe("o3db-agent-smoke", tmp_path / "host-only", output)
    assert receipt["success"] and receipt["provenance"] == "trusted-driver-fixed-probe"
    assert calls[0][0] == ["docker", "exec", "o3db-agent-smoke", "/usr/bin/python3", "-c",
                           module.ISOLATION_PROBE, str(tmp_path / "host-only")]
    assert calls[0][1] == {"capture_output": True, "text": True, "timeout": 30}
    assert json.loads(output.read_text()) == receipt
    assert "exc.errno == errno.EROFS" in module.ISOLATION_PROBE
    assert "os.O_WRONLY | os.O_APPEND" in module.ISOLATION_PROBE


@pytest.mark.parametrize("failure", ["missing", "extra", "false", "integer", "exit", "invalid_json"])
def test_trusted_probe_rejects_incomplete_or_unqualified_claims(tmp_path, monkeypatch, failure):
    module = qualification_module()
    checks = {name: True for name in module.ISOLATION_CHECKS}
    if failure == "missing":
        checks.pop("git_read_only")
    elif failure == "extra":
        checks["invented_check"] = True
    elif failure in ("false", "integer"):
        checks["git_read_only"] = False if failure == "false" else 1
    response = SimpleNamespace(returncode=1 if failure == "exit" else 0,
                               stdout="not JSON" if failure == "invalid_json" else json.dumps({"checks": checks}),
                               stderr="")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: response)
    output = tmp_path / "trusted-isolation.json"
    with pytest.raises(RuntimeError, match="Trusted container isolation"):
        module.trusted_isolation_probe("o3db-agent-smoke", tmp_path / "host-only", output)
    assert json.loads(output.read_text())["success"] is False


@pytest.mark.parametrize("error_number,passed", [(errno.EROFS, True), (errno.EACCES, False),
                                                (errno.ENOENT, False)])
def test_trusted_probe_requires_read_only_mount_errno(tmp_path, monkeypatch, error_number, passed):
    module = qualification_module()
    tree = ast.parse(module.ISOLATION_PROBE)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "read_only")
    namespace = {"Path": Path, "os": os, "errno": errno}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "trusted-probe", "exec"), namespace)
    protected = tmp_path / "protected.txt"
    protected.write_text("UNCHANGED")
    def denied(*args):
        raise OSError(error_number, "Simulated kernel denial")
    monkeypatch.setattr(os, "open", denied)
    result = namespace["read_only"](str(protected))
    assert result == {"passed": passed, "errno": error_number}
    assert protected.read_text() == "UNCHANGED"
