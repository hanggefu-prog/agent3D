"""Bound Unix-socket RPC between an untrusted Agent and the organizer harness."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socketserver
import tempfile
import threading


MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_REPLY_BYTES = 64 * 1024 * 1024


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = False
    allow_reuse_address = False

    def __init__(self, *args):
        self._slots = threading.BoundedSemaphore(16)
        super().__init__(*args)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if not line or len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            return
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"method", "params"}:
                raise ValueError("Invalid bridge request")
            method, params = request["method"], request["params"]
            if not isinstance(method, str) or not isinstance(params, dict):
                raise ValueError("Invalid bridge parameters")
            result = self.server.controller.dispatch(method, params)
            reply = {"ok": True, "result": result}
            data = json.dumps(reply, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        except Exception as error:
            if not isinstance(error, (ValueError, KeyError, TypeError, RuntimeError, OSError)):
                self.server.handle_error(self.request, self.client_address)
            # Never send tracebacks, host paths, provider responses or secrets.
            reply = {"ok": False, "error": type(error).__name__ + ": " +
                     (str(error) if isinstance(error, (ValueError, KeyError)) else "request failed")}
            data = json.dumps(reply, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        if len(data) > MAX_REPLY_BYTES:
            data = b'{"ok":false,"error":"Organizer response exceeds bridge limit"}\n'
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass


class AgentBridge:
    def __init__(self, controller):
        self.controller = controller
        self.directory = None
        self.server = None
        self.thread = None

    def __enter__(self):
        self.directory = Path(tempfile.mkdtemp(prefix="o3db-agent-bridge-"))
        self.directory.chmod(0o700)
        path = self.directory / "bridge.sock"
        self.server = _Server(str(path), _Handler)
        self.server.controller = self.controller
        os.chmod(path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    @property
    def socket_path(self):
        if self.directory is None:
            raise RuntimeError("Bridge is not running")
        return self.directory / "bridge.sock"

    def __exit__(self, *_):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.directory is not None:
            shutil.rmtree(self.directory)
