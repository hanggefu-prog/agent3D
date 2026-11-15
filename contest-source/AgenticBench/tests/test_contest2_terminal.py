import io
import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from contest2 import container_terminal as terminal


def ready():
    server = terminal.ContainerTerminal("trusted-trial-001")
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                              "params": {"protocolVersion": "2025-06-18"}})
    assert response["result"]["protocolVersion"] == "2025-06-18"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    return server


def call(server, arguments):
    return server.handle({"jsonrpc": "2.0", "id": "call-1", "method": "tools/call",
                          "params": {"name": "exec", "arguments": arguments}})


class Selector:
    def __init__(self):
        self.active = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def register(self, *args):
        self.active = True

    def unregister(self, *args):
        self.active = False

    def get_map(self):
        return {7: None} if self.active else {}

    def select(self, timeout):
        return [(SimpleNamespace(fd=7), 1)]


def mocked_execution(command="printf OK", chunks=(b"OK",), code=0, **patches):
    process = mock.Mock()
    process.wait.return_value = code
    process.poll.return_value = code
    with mock.patch.object(terminal.subprocess, "Popen", return_value=process) as popen, \
            mock.patch.object(terminal.selectors, "DefaultSelector", side_effect=Selector), \
            mock.patch.object(terminal.os, "read", side_effect=[*chunks, b""]):
        if patches:
            with mock.patch.dict(os.environ, patches, clear=True):
                response = call(ready(), {"command": command, "timeout_seconds": 30})
        else:
            response = call(ready(), {"command": command, "timeout_seconds": 30})
    return response, popen, process


@pytest.mark.parametrize("name", ["", "-x", "--privileged", "trial; touch /host", "a\nb", "/tmp/a", "a b"])
def test_container_identity_is_not_a_shell_or_option(name):
    with pytest.raises(ValueError, match="literal Docker"):
        terminal.ContainerTerminal(name)


def test_only_one_tool_and_no_host_or_container_arguments():
    result = ready().handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]
    assert [tool["name"] for tool in result["tools"]] == ["exec"]
    schema = result["tools"][0]["inputSchema"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"command", "timeout_seconds", "session_id"}


@pytest.mark.parametrize("extra", ["container", "cwd", "host_path", "env", "docker_binary"])
def test_tool_arguments_cannot_override_host_execution(extra):
    with mock.patch.object(terminal.subprocess, "Popen") as popen:
        response = call(ready(), {"command": "pwd", "timeout_seconds": 10, extra: "untrusted"})
    assert response["error"]["code"] == -32602
    popen.assert_not_called()


@pytest.mark.parametrize("seconds", [0, -1, True, 5.0, "5", None])
def test_legacy_timeout_must_be_a_positive_integer(seconds):
    with mock.patch.object(terminal.subprocess, "Popen") as popen:
        response = call(ready(), {"command": "pwd", "timeout_seconds": seconds})
    assert response["error"]["code"] == -32602
    popen.assert_not_called()


@pytest.mark.parametrize("command", ["", " \n", "x\0y", "x" * (terminal.MAX_COMMAND_BYTES + 1), 4])
def test_invalid_command_never_starts_docker(command):
    with mock.patch.object(terminal.subprocess, "Popen") as popen:
        assert "error" in call(ready(), {"command": command, "timeout_seconds": 10})
    popen.assert_not_called()


def test_shell_metacharacters_stay_one_container_local_argument():
    command = "printf '%s' \"$(cat /host/secret)\"; docker exec another id > /tmp/probe"
    response, popen, process = mocked_execution(command)
    assert popen.call_args.args[0] == [
        "docker", "exec", "trusted-trial-001", "bash", "-lc", command]
    assert popen.call_args.kwargs["shell"] is False
    assert popen.call_args.kwargs["stdin"] == terminal.subprocess.DEVNULL
    assert popen.call_args.kwargs["stderr"] == terminal.subprocess.STDOUT
    assert "cwd" not in popen.call_args.kwargs
    assert response["result"]["isError"] is False
    assert json.loads(response["result"]["content"][0]["text"])["output"] == "OK"
    process.stdout.close.assert_called_once()


def test_host_credentials_and_shell_startup_variables_are_not_forwarded():
    response, popen, _ = mocked_execution(
        OPENAI_API_KEY="secret-sentinel", CODEX_ACCESS_TOKEN="token-sentinel", BASH_ENV="/host/private",
        HTTPS_PROXY="http://user:password@proxy", DOCKER_HOST="tcp://untrusted:1234", PATH="/usr/bin:/bin")
    assert popen.call_args.kwargs["env"] == {"PATH": "/usr/bin:/bin"}
    assert "secret-sentinel" not in json.dumps(response)
    assert "token-sentinel" not in json.dumps(response)


@pytest.mark.parametrize("chunks", [
    [b"x" * 8192] * 20,
    [b"\0\n\r\t" * 2048] * 12,
    ["温度".encode() * 1000] * 20,
    [b"\xff" * 8192] * 10,
])
def test_output_is_streamed_and_serialized_text_is_at_most_48_kib(chunks):
    response, _, _ = mocked_execution(chunks=chunks)
    text = response["result"]["content"][0]["text"]
    result = json.loads(text)
    assert len(text.encode()) <= terminal.MAX_OUTPUT_BYTES
    assert result["output_bytes_seen"] == sum(map(len, chunks))
    assert result["truncated"] is True
    assert result["output_bytes_returned"] == len(result["output"].encode())


@pytest.mark.parametrize("code,status", [(3, "error"), (124, "error"), (137, "error")])
def test_nonzero_and_timeout_outcomes_are_not_success(code, status):
    response, _, _ = mocked_execution(code=code)
    assert response["result"]["isError"] is True
    result = json.loads(response["result"]["content"][0]["text"])
    assert result["status"] == status
    assert result["exit_code"] == code


def test_running_command_survives_legacy_timeout_and_can_be_polled(monkeypatch):
    import sys
    import time
    real_popen = terminal.subprocess.Popen
    processes = []
    def launch(argv, **kwargs):
        # Only this harmless test subprocess runs locally; production argv is checked separately.
        process = real_popen([sys.executable, "-c", "import time; time.sleep(1.3); print('finished')"], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(terminal.subprocess, "Popen", launch)
    server = ready()
    response = call(server, {"command": "long native command", "timeout_seconds": 1})
    result = json.loads(response["result"]["content"][0]["text"])
    assert result["status"] == "running" and not response["result"]["isError"]
    assert processes[0].poll() is None
    while result["status"] == "running":
        response = call(server, {"session_id": result["session_id"]})
        result = json.loads(response["result"]["content"][0]["text"])
    assert result["status"] == "success" and result["output"] == "finished\n"
    assert result["timeout_seconds"] is None
    assert result["duration_seconds"] > 1
    assert "error" in call(ready(), {"session_id": result["session_id"]})


def test_launch_exception_does_not_reflect_host_paths_or_command():
    with mock.patch.object(terminal.subprocess, "Popen", side_effect=OSError("private-host-sentinel")):
        response = call(ready(), {"command": "pwd", "timeout_seconds": 10})
    assert response["result"]["isError"] is True
    assert "private-host-sentinel" not in json.dumps(response)


def test_notifications_never_run_tools_or_produce_responses():
    server = ready()
    with mock.patch.object(server, "execute") as execute:
        assert server.handle({"jsonrpc": "2.0", "method": "tools/call", "params": {
            "name": "exec", "arguments": {"command": "pwd", "timeout_seconds": 5}}}) is None
        assert server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None
    execute.assert_not_called()


def test_initialize_required_ping_allowed_and_unknown_probe_answers_promptly():
    server = terminal.ContainerTerminal("trial")
    for method, code in [("tools/list", -32002), ("server/discover", -32601)]:
        assert server.handle({"jsonrpc": "2.0", "id": 1, "method": method})["error"]["code"] == code
    assert server.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"] == {}
    response = server.handle({"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {
        "protocolVersion": "future-protocol"}})
    assert response["result"]["protocolVersion"] == terminal.PROTOCOL_VERSIONS[-1]


def test_stdio_json_only_recovers_after_malformed_duplicate_and_oversized_input():
    messages = [b"not-json\n", b'{"jsonrpc":"2.0","id":1,"id":2,"method":"ping"}\n',
                b"x" * (terminal.MAX_MESSAGE_BYTES + 10) + b"\n",
                b'{"jsonrpc":"2.0","id":"ok","method":"ping"}\n']
    output = io.BytesIO()
    assert terminal.serve("trial", io.BytesIO(b"".join(messages)), output) == 0
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [item["error"]["code"] for item in responses[:3]] == [-32700] * 3
    assert responses[-1] == {"jsonrpc": "2.0", "id": "ok", "result": {}}
