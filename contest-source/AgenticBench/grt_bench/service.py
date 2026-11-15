"""Loopback-only queue API. Remote workers connect through an SSH tunnel.

The admin submits jobs on the coordinator. Workers cannot submit jobs, alter
budgets, register hosts, reconcile leases, or publish Pareto entries.
"""
import argparse
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .queue import Queue


def create_server(database, tokens, port):
    queue = Queue(database)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never write authentication headers into logs.

        def do_POST(self):
            host = self.headers.get("X-Worker-Host", "")
            expected = tokens.get(host)
            supplied = self.headers.get("Authorization", "")
            if not expected or not hmac.compare_digest(supplied, "Bearer " + expected):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 1024 * 1024:
                    raise ValueError("Invalid request size")
                args = json.loads(self.rfile.read(length))
                if self.path == "/claim":
                    result = queue.claim(host)
                elif self.path in {"/heartbeat", "/finish", "/release"}:
                    row = next((r for r in queue.rows() if r["id"] == args["job"]), None)
                    if row is None or row["host"] != host:
                        raise ValueError("Wrong worker host")
                    if self.path == "/release":
                        result = queue.release(args["job"], args["token"], args["resources"], args["evidence"])
                    elif self.path == "/heartbeat":
                        queue.heartbeat(args["job"], args["token"])
                        result = {"ok": True}
                    else:
                        queue.finish(args["job"], args["token"], args["result"], args["success"])
                        result = {"ok": True}
                else:
                    raise ValueError("Unsupported worker operation")
                encoded = json.dumps(result, allow_nan=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (ValueError, KeyError, TypeError) as error:
                self.send_error(409, str(error))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--database", required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    if a.tokens.is_symlink() or a.tokens.stat().st_mode & 0o077:
        raise ValueError("Worker tokens must be a private regular file")
    create_server(a.database, json.loads(a.tokens.read_text()), a.port).serve_forever()


if __name__ == "__main__":
    main()
