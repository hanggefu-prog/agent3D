"""Small participant API. All authority stays in the organizer bridge.

This module is mounted read-only in the agent container. Its methods send
requests over the one organizer-owned Unix socket; it does not open the budget
database, run the native evaluator, or trust a participant-supplied score.
"""
from __future__ import annotations

import json
import os
import socket
import uuid


MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class AgentAPIError(RuntimeError):
    pass


class AgentContext:
    def __init__(self, socket_path=None):
        self.socket_path = socket_path or os.environ.get("CONTEST_BRIDGE_SOCKET", "")
        if not self.socket_path:
            raise AgentAPIError("The organizer agent bridge is unavailable")
        initial = self._call("initial", {})
        self.task = initial["task"]
        self.evaluation_mode = initial.get("evaluation_mode", "formal")
        self.evaluation_cells = initial.get("evaluation_cells", [])
        self.baseline_candidate = initial["baseline_candidate"]
        self.files = initial.get("files", {})
        self.model = _ModelClient(self)

    def _call(self, method, params):
        payload = json.dumps({"method": method, "params": params},
                             separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        if len(payload) > MAX_REQUEST_BYTES:
            raise AgentAPIError("Agent request exceeds the bridge limit")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(self.socket_path)
                connection.sendall(payload)
                chunks = bytearray()
                while True:
                    part = connection.recv(65536)
                    if not part:
                        break
                    chunks.extend(part)
                    if len(chunks) > MAX_MESSAGE_BYTES:
                        raise AgentAPIError("Organizer response exceeds the bridge limit")
                    if b"\n" in part:
                        break
        except OSError as error:
            raise AgentAPIError("Agent bridge communication failed; request outcome is unknown") from error
        try:
            reply = json.loads(bytes(chunks).split(b"\n", 1)[0])
        except (ValueError, IndexError) as error:
            raise AgentAPIError("Invalid organizer bridge response") from error
        if (not isinstance(reply, dict) or type(reply.get("ok")) is not bool
                or (reply["ok"] and "result" not in reply)):
            raise AgentAPIError("Invalid organizer bridge envelope")
        if not reply["ok"]:
            raise AgentAPIError(reply.get("error", "Organizer request failed"))
        return reply["result"]

    def status(self):
        return self._call("status", {})

    def submit_evaluation(self, source, request_id=None):
        """Submit the editable GNN functions for the complete fixed task matrix."""
        return self._call("submit_evaluation", {"source": source,
                                                "request_id": request_id or uuid.uuid4().hex})

    def evaluation_status(self, request_id):
        return self._call("evaluation_status", {"request_id": request_id})

    def wait_evaluation(self, request_id, poll_seconds=5):
        import time

        while True:
            result = self.evaluation_status(request_id)
            if result["status"] in {"succeeded", "failed", "rejected"}:
                return result
            time.sleep(poll_seconds)

    def codex_turn(self, prompt, request_id=None):
        return self._call("codex_turn", {"prompt": prompt,
                                         "request_id": request_id or uuid.uuid4().hex})


class _ModelClient:
    def __init__(self, context):
        self.context = context

    def responses_create(self, payload):
        return self.context._call("model_responses_create", {"payload": payload})
