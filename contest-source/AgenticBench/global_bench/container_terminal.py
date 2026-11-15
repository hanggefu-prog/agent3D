"""Minimal stdio MCP terminal for one host-selected, already-running container.

Only Docker executes on the host; the command string is one literal argv item
for the container's bash. This bridge does not isolate other Codex tools. The
caller must separately restrict host tools and configure the container's mounts,
user, network and resource limits. No containers or authentication are managed
here. The supported MCP lifecycle is the initialize-based 2024/2025 protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import subprocess
import sys
import time


MAX_OUTPUT_BYTES = 48 * 1024
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_COMMAND_BYTES = 64 * 1024
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
TOOL = {
    "name": "exec",
    "description": (
        "Run a shell command inside the single preconfigured experiment container. "
        "All command paths and shell expansion are container-local. The command "
        "cannot select another container, host working directory or host environment. "
        "Combined output is bounded; execution does not establish physical validity."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["command", "timeout_seconds"],
        "properties": {
            "command": {"type": "string", "minLength": 1},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
        },
    },
}


def _error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def _tool_result(result):
    """Bound the serialized text too, including UTF-8 and JSON escaping overhead."""
    result = dict(result)
    while True:
        result["output_bytes_returned"] = len(result["output"].encode("utf-8"))
        text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        excess = len(text.encode("utf-8")) - MAX_OUTPUT_BYTES
        if excess <= 0:
            break
        data = result["output"].encode("utf-8")
        result["output"] = data[:max(0, len(data) - excess - 32)].decode("utf-8", errors="ignore")
        result["truncated"] = True
    return {"content": [{"type": "text", "text": text}],
            "isError": result["status"] != "success"}


class ContainerTerminal:
    def __init__(self, container):
        if not isinstance(container, str) or not CONTAINER_NAME.fullmatch(container):
            raise ValueError("Container must be one literal Docker container name")
        self.container = container
        self.negotiated = False
        self.initialized = False

    def tools(self):
        return [TOOL]

    def call_tool(self, name, arguments):
        if name != "exec":
            raise ValueError("Unknown tool")
        return self.execute(arguments)

    def execute(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) != {"command", "timeout_seconds"}:
            raise ValueError("exec requires only command and timeout_seconds")
        command, seconds = arguments["command"], arguments["timeout_seconds"]
        if (not isinstance(command, str) or not command.strip() or "\0" in command
                or len(command.encode("utf-8")) > MAX_COMMAND_BYTES):
            raise ValueError("command must be nonempty UTF-8 text of at most 64 KiB, without NUL")
        if type(seconds) is not int or not 1 <= seconds <= 300:
            raise ValueError("timeout_seconds must be an integer from 1 to 300")

        # Never use shell=True or interpolate command into a host shell string.
        argv = ["docker", "exec", self.container, "timeout", "-k", "5", str(seconds),
                "bash", "-lc", command]
        start = time.monotonic()
        output, seen, code, fallback_timeout, error = bytearray(), 0, None, False, None
        process = None
        try:
            process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, shell=False, close_fds=True,
                # Do not hand the Docker client model credentials, proxies or injected
                # shell startup variables. No host variables are passed via docker -e.
                env={"PATH": os.environ.get("PATH", os.defpath)},
            )
            deadline = start + seconds + 10  # Native timeout + kill grace + client slack.
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, seconds + 10)
                    for key, _ in selector.select(min(remaining, 0.25)):
                        data = os.read(key.fd, 8192)
                        if not data:
                            selector.unregister(process.stdout)
                            continue
                        seen += len(data)
                        output.extend(data[:max(0, MAX_OUTPUT_BYTES - len(output))])
                code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            fallback_timeout = True
            error = (
                "Docker exec client exceeded its fallback deadline. The container command "
                "has a separate GNU timeout; client termination alone is not proof it stopped."
            )
        except OSError as exc:
            # Exception strings can include host paths or argv; do not reflect them.
            error = "Container execution failed ({})".format(type(exc).__name__)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()  # Only this tool's Docker client, never another process.
                    try:
                        code = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        error = "Docker exec client did not exit after termination"
                if process.stdout is not None:
                    process.stdout.close()
        if fallback_timeout or code == 124:
            status = "timeout"
        elif code == 137:
            status = "killed"
            error = error or "Exit 137: command was killed; timeout versus another SIGKILL is unknown"
        else:
            status = "success" if code == 0 and error is None else "error"
        return _tool_result({
            "status": status, "exit_code": code, "timeout_seconds": seconds,
            "client_deadline_exceeded": fallback_timeout,
            "duration_seconds": round(time.monotonic() - start, 3),
            "output": bytes(output).decode("utf-8", errors="replace"),
            "output_bytes_seen": seen, "truncated": seen > len(output), "error": error,
        })

    def handle(self, request):
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
            return _error(None, -32600, "Invalid JSON-RPC request")
        method = request.get("method")
        if not isinstance(method, str):
            return _error(None, -32600, "Request method must be a string")
        if "id" not in request:
            if method == "notifications/initialized" and self.negotiated:
                self.initialized = True
            # Notifications (including tools/call without an id) never execute tools.
            return None
        request_id = request["id"]
        if type(request_id) not in (int, str):
            return _error(None, -32600, "Request id must be a string or integer")
        params = request.get("params", {})
        if not isinstance(params, dict):
            return _error(request_id, -32602, "Request params must be an object")
        if method == "initialize":
            if self.negotiated:
                return _error(request_id, -32600, "Session already initialized")
            version = params.get("protocolVersion")
            if not isinstance(version, str):
                return _error(request_id, -32602, "protocolVersion must be a string")
            self.negotiated = True
            result = {"protocolVersion": version if version in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1],
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "agentic3d-container-terminal", "version": "1.0.0"}}
        elif method == "ping":
            result = {}
        elif method not in {"tools/list", "tools/call"}:
            return _error(request_id, -32601, "Method not found")
        elif not self.initialized:
            return _error(request_id, -32002, "Session is not initialized")
        elif method == "tools/list":
            if params.get("cursor"):
                return _error(request_id, -32602, "This tool list is not paginated")
            result = {"tools": self.tools()}
        else:
            if params.get("name") not in {tool["name"] for tool in self.tools()}:
                return _error(request_id, -32602, "Unknown tool")
            if set(params) - {"name", "arguments", "_meta"}:
                return _error(request_id, -32602, "Unsupported tools/call parameter")
            try:
                result = self.call_tool(params["name"], params.get("arguments"))
            except (ValueError, UnicodeError) as exc:
                # Validation messages are constants; never include user data.
                return _error(request_id, -32602, str(exc) if not isinstance(exc, UnicodeError)
                              else "command must be valid UTF-8 text")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member")
        result[key] = value
    return result


def serve(container, source, destination, *, terminal=None):
    """Read and write newline-delimited UTF-8 JSON; stdout is protocol-only."""
    terminal = terminal or ContainerTerminal(container)
    while True:
        line = source.readline(MAX_MESSAGE_BYTES + 1)
        if not line:
            return 0
        if len(line) > MAX_MESSAGE_BYTES:
            while not line.endswith(b"\n"):
                line = source.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    break
            response = _error(None, -32700, "Message exceeds 1 MiB")
        elif not line.strip():
            continue
        else:
            try:
                request = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
                response = terminal.handle(request)
            except (ValueError, UnicodeError):
                response = _error(None, -32700, "Invalid JSON")
            except Exception:
                response = _error(None, -32603, "Internal server error")
        if response is not None:
            destination.write((json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            destination.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True, help="Trusted fixed container name; it must already exist")
    args = parser.parse_args(argv)
    if not CONTAINER_NAME.fullmatch(args.container):
        parser.error("--container must be one literal Docker container name")
    return serve(args.container, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
