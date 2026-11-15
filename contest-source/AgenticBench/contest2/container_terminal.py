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
import uuid


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
        "required": [],
        "properties": {
            "command": {"type": "string", "minLength": 1},
            "timeout_seconds": {"type": "integer", "minimum": 1,
                                "description": "Legacy argument, ignored; commands have no time limit."},
            "session_id": {"type": "string", "description": "Poll a command returned as running."},
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
            "isError": result["status"] not in {"success", "running"}}


class ContainerTerminal:
    def __init__(self, container):
        if not isinstance(container, str) or not CONTAINER_NAME.fullmatch(container):
            raise ValueError("Container must be one literal Docker container name")
        self.container = container
        self.negotiated = False
        self.initialized = False
        self.sessions = {}

    def tools(self):
        return [TOOL]

    def call_tool(self, name, arguments):
        if name != "exec":
            raise ValueError("Unknown tool")
        return self.execute(arguments)

    def execute(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) - {"command", "timeout_seconds", "session_id"}:
            raise ValueError("exec accepts command or session_id only; legacy timeout_seconds is ignored")
        if "timeout_seconds" in arguments and (type(arguments["timeout_seconds"]) is not int or arguments["timeout_seconds"] < 1):
            raise ValueError("Legacy timeout_seconds must be a positive integer")
        session_id = arguments.get("session_id")
        if session_id is not None:
            if "command" in arguments or not isinstance(session_id, str) or session_id not in self.sessions:
                raise ValueError("Unknown command session or mixed command/session_id")
        else:
            command = arguments.get("command")
            if (not isinstance(command, str) or not command.strip() or "\0" in command
                    or len(command.encode("utf-8")) > MAX_COMMAND_BYTES):
                raise ValueError("command must be nonempty UTF-8 text of at most 64 KiB, without NUL")
            argv = ["docker", "exec", self.container, "bash", "-lc", command]
            try:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, shell=False, close_fds=True,
                    env={"PATH": os.environ.get("PATH", os.defpath)})
            except OSError:
                return _tool_result({"status": "error", "output": "", "error": "Container command could not start"})
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = {"process": process, "output": bytearray(), "seen": 0,
                                         "started": time.monotonic(), "eof": False}
        session = self.sessions[session_id]
        process = session["process"]
        # This is a response-yield interval, never an execution deadline.
        until = time.monotonic() + 1
        if not session["eof"]:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map() and time.monotonic() < until:
                    for key, _ in selector.select(0.1):
                        data = os.read(key.fd, 8192)
                        if not data:
                            selector.unregister(process.stdout)
                            session["eof"] = True
                            process.stdout.close()
                            continue
                        session["seen"] += len(data)
                        session["output"].extend(data[:max(0, MAX_OUTPUT_BYTES - len(session["output"]))])
        code = process.poll()
        status = "running" if code is None or not session["eof"] else "success" if code == 0 else "error"
        return _tool_result({"status": status, "session_id": session_id, "exit_code": code,
            "timeout_seconds": None, "client_deadline_exceeded": False,
            "duration_seconds": round(time.monotonic() - session["started"], 3),
            "output": bytes(session["output"]).decode("utf-8", errors="replace"),
            "output_bytes_seen": session["seen"], "truncated": session["seen"] > len(session["output"]),
            "error": None if status in {"success", "running"} else "Container command exited unsuccessfully"})

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
