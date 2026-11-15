"""Host-owned DeepSeek Responses gateway with pre-dispatch RMB reservations.

The participant container has no route to this listener or to the provider.
Start this process separately from the Codex campaign so the provider key never
enters the Codex process environment. Uncertain provider calls retain their
reservation for organizer reconciliation; they are never retried automatically.
"""
from __future__ import annotations

import argparse
import errno
from datetime import datetime, timezone
from decimal import Decimal
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import socket
import ssl
import urllib.error
import urllib.request
import uuid

from .budget import BudgetLedger, BudgetExceeded
from .model_pricing import MODEL_ID, MAX_OUTPUT_PER_CALL, charge_usage, peak_during, reserve_full_context

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
UPSTREAM = "https://api.deepseek.com/responses"
_TERMINAL_NAMESPACE = "mcp__terminal"
_TERMINAL_TOOL = "exec"
_FLAT_TERMINAL_TOOL = "mcp__terminal__exec"


def _restore_tool_calls(value):
    """Restore Codex's tool identity after DeepSeek's flat function response."""
    if isinstance(value, list):
        return [_restore_tool_calls(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _restore_tool_calls(item) for key, item in value.items()}
    if result.get("type") == "function_call" and result.get("name") == _FLAT_TERMINAL_TOOL:
        result["name"] = _TERMINAL_TOOL
        result["namespace"] = _TERMINAL_NAMESPACE
    return result


def restore_response_tools(body: bytes, stream: bool) -> bytes:
    """Preserve SSE framing while restoring only the pinned container tool."""
    if not stream:
        return json.dumps(_restore_tool_calls(json.loads(body)),
                          ensure_ascii=False, separators=(",", ":")).encode()
    lines = []
    for line in body.splitlines(keepends=True):
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            lines.append(line)
            continue
        newline = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
        event = json.loads(line[6:].strip())
        lines.append(b"data: " + json.dumps(_restore_tool_calls(event),
                     ensure_ascii=False, separators=(",", ":")).encode() + newline)
    return b"".join(lines)


def validate_request(body: bytes) -> dict:
    if not body or len(body) > MAX_REQUEST_BYTES:
        raise ValueError("Request exceeds the contest body limit")
    payload = json.loads(body)
    if not isinstance(payload, dict) or payload.get("model") != MODEL_ID:
        raise ValueError("Only the frozen DeepSeek Flash model is allowed")
    if type(payload.get("stream", False)) is not bool:
        raise ValueError("Invalid stream setting")
    if "input" not in payload or payload["input"] is None:
        raise ValueError("Responses input is required")
    # Flash does not provide charged server-side web tools. Client-side MCP
    # function calls remain available and execute only in the agent container.
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Server-side or unknown tools are not allowed")
    flattened = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Server-side or unknown tools are not allowed")
        if tool.get("type") == "namespace":
            children = tool.get("tools")
            if (tool.get("name") != _TERMINAL_NAMESPACE or not isinstance(children, list)
                    or len(children) != 1 or not isinstance(children[0], dict)
                    or children[0].get("type") != "function"
                    or children[0].get("name") != _TERMINAL_TOOL):
                raise ValueError("Server-side or unknown tools are not allowed")
            flattened.append({**children[0], "name": _FLAT_TERMINAL_TOOL})
        elif tool.get("type") in {"function", "custom"}:
            flattened.append(tool)
        else:
            raise ValueError("Server-side or unknown tools are not allowed")
    payload["tools"] = flattened
    if isinstance(payload["input"], list):
        for item in payload["input"]:
            if not isinstance(item, dict) or item.get("type") != "function_call" or "namespace" not in item:
                continue
            if item["namespace"] != _TERMINAL_NAMESPACE or item.get("name") != _TERMINAL_TOOL:
                raise ValueError("Unknown namespaced input tool call")
            item["name"] = _FLAT_TERMINAL_TOOL
            del item["namespace"]
    def inspect(value):
        if isinstance(value, dict):
            if value.get("type") in {"input_image", "image_url", "input_file", "file"}:
                raise ValueError("Only text input is allowed")
            for nested in value.values():
                inspect(nested)
        elif isinstance(value, list):
            for nested in value:
                inspect(nested)
    inspect(payload["input"])
    payload["max_output_tokens"] = MAX_OUTPUT_PER_CALL
    return payload


def terminal_usage(body: bytes, stream: bool) -> dict:
    if stream:
        terminal = []
        for line in body.splitlines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ValueError("Provider stream event must be an object with a string type")
            if event.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
                terminal.append(event.get("response"))
        if len(terminal) != 1 or not isinstance(terminal[0], dict):
            raise ValueError("Provider stream has no unique terminal usage event")
        response = terminal[0]
    else:
        response = json.loads(body)
    if not isinstance(response, dict) or response.get("model") != MODEL_ID:
        raise ValueError("Provider response model differs")
    usage = response.get("usage")
    charge_usage(usage)
    return usage


def provider_failure(error, *, access_token, provider_key):
    """Keep diagnostic evidence; only connection setup failures prove no POST."""
    cause = error.reason if isinstance(error, urllib.error.URLError) else error
    not_sent = isinstance(error, urllib.error.URLError) and (
        isinstance(cause, (socket.gaierror, ConnectionRefusedError, ssl.SSLCertVerificationError))
        or isinstance(cause, OSError) and cause.errno in {errno.ENETUNREACH, errno.EHOSTUNREACH})
    message = str(error)
    details = {"error_type": type(error).__name__, "cause_type": type(cause).__name__,
               "request_not_sent": not_sent}
    if isinstance(error, urllib.error.HTTPError):
        details["http_status"] = error.code
        try:
            message += " " + error.read(4096).decode("utf-8", errors="replace")
        except (OSError, HTTPException):
            pass
    for secret in (provider_key, access_token):
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", message)
    message = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [redacted]", message)
    message = re.sub(r"(https?://)[^/\s@]+@", r"\1[redacted]@", message)
    details["message"] = message[:2000]
    return details


def make_handler(ledger: BudgetLedger, *, access_token: str, provider_key: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Never log request bodies, bearer tokens, or provider credentials.
            return

        def _authorized(self) -> bool:
            return secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + access_token)

        def _reply(self, status: int, data: bytes, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._authorized() or self.path != "/health":
                self._reply(404, b'{}')
                return
            state = ledger.snapshot()
            self._reply(200, json.dumps({"task_id": state["task_id"], "model_cost_available_rmb":
                state["model_cost_available_rmb"], "dispatch_allowed": state["dispatch_allowed"]}).encode())

        def do_POST(self):
            if not self._authorized() or self.path not in {"/responses", "/v1/responses"}:
                self._reply(404, b'{}')
                return
            length = self.headers.get("Content-Length")
            if not length or not re.fullmatch(r"[0-9]+", length) or int(length) > MAX_REQUEST_BYTES:
                self._reply(413, b'{"error":"invalid request length"}')
                return
            try:
                payload = validate_request(self.rfile.read(int(length)))
                # Encoding failures are invalid input, before any cost reservation.
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._reply(400, json.dumps({"error": str(error)}).encode())
                return
            request_id = uuid.uuid4().hex
            from .runtime_recovery import lease, provider_directory
            with lease(provider_directory(ledger) / (request_id + ".lock")):
                try:
                    # A call can cross a tariff boundary while in flight. Reserve
                    # the full-context peak ceiling before any provider dispatch.
                    ledger.reserve_model_call(request_id, MODEL_ID, reserve_full_context(peak=True))
                except BudgetExceeded:
                    self._reply(402, b'{"error":"contest model budget exhausted"}')
                    return
                upstream = urllib.request.Request(UPSTREAM, data=data, headers={
                    "Authorization": "Bearer " + provider_key,
                    "Content-Type": "application/json", "Accept": "text/event-stream" if payload.get("stream") else "application/json"}, method="POST")
                try:
                    dispatched_at = datetime.now(timezone.utc)
                    with urllib.request.urlopen(upstream, timeout=1800) as response:
                        answer = response.read(MAX_RESPONSE_BYTES + 1)
                        content_type = response.headers.get("Content-Type", "application/json")
                    completed_at = datetime.now(timezone.utc)
                    if len(answer) > MAX_RESPONSE_BYTES:
                        raise ValueError("Provider response exceeded the contest limit")
                    stream = bool(payload.get("stream"))
                    usage = terminal_usage(answer, stream)
                    peak = peak_during(dispatched_at, completed_at)
                    cost = charge_usage(usage, peak=peak)
                    from grt_bench.common import write
                    write(provider_directory(ledger) / (request_id + ".json"), {
                        "request_id": request_id, "model": MODEL_ID, "usage": usage, "peak": peak})
                    answer = restore_response_tools(answer, stream)
                    if len(answer) > MAX_RESPONSE_BYTES:
                        raise ValueError("Translated provider response exceeded the contest limit")
                    ledger.record_model_call(request_id, MODEL_ID, usage["input_tokens"],
                                             usage["output_tokens"], str(cost))
                    self._reply(200, answer, content_type)
                except (OSError, ValueError, HTTPException) as error:
                    from grt_bench.common import write
                    details = provider_failure(error, access_token=access_token, provider_key=provider_key)
                    write(provider_directory(ledger) / (request_id + ".error.json"),
                          {"request_id": request_id, **details})
                    if details["request_not_sent"]:
                        ledger.record_model_call(request_id, MODEL_ID, 0, 0, "0")
                        message = "Provider connection failed before request delivery; reservation released"
                    else:
                        ledger.mark_model_uncertain(request_id, details["error_type"] + ": " + details["message"])
                        message = "Provider usage uncertain; reservation retained"
                    self._reply(502, json.dumps({"error": message, "request_id": request_id,
                                               "details": details}).encode())

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--task", choices=["binary_loss", "loss_decoder"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--evaluation-mode", choices=("formal", "smoke"), default="formal")
    args = parser.parse_args()
    access = os.environ.get("CONTEST_GATEWAY_TOKEN", "")
    provider = os.environ.get("DEEPSEEK_API_KEY", "")
    if len(access) < 32 or not provider or not 1024 <= args.port <= 65535:
        parser.error("Organizer gateway token, provider key and port are required")
    ledger = BudgetLedger(args.ledger, evaluation_limit=10, model_cost_limit_rmb=Decimal("15"),
                          task_id=args.task, allowed_model=MODEL_ID,
                          evaluation_mode=args.evaluation_mode)
    from .runtime_recovery import recover_provider
    recover_provider(ledger)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(ledger,
        access_token=access, provider_key=provider))
    server.serve_forever(poll_interval=0.25)


if __name__ == "__main__":
    main()
