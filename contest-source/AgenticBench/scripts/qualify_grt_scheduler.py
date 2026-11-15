"""Qualify Docker termination and durable queue recovery in an isolated store.

This exercises execution control, not physical-design metric correctness.
"""
import argparse
from pathlib import Path
import subprocess
import threading
import time

from grt_bench.common import sha, write
from grt_bench.queue import Queue
from grt_bench.runtime import docker_command, run_container
from grt_bench.service import create_server
from grt_bench.worker import Client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output / "queue.sqlite"
    queue = Queue(database)
    queue.register_host("smoke", dict(cpu=1, memory_gb=1, storage_gb=1, gpus=[], build_slots=0))
    resources = dict(cpu=1, memory_gb=1, storage_gb=1, gpu_count=0, build_slots=0)
    server = create_server(database, {"smoke": "isolated-test-token"}, 0)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    client = Client("http://127.0.0.1:" + str(server.server_address[1]), "smoke", "isolated-test-token")
    summary = []
    try:
        for kind in ("failure", "timeout", "cancel"):
            job = queue.submit({"kind": kind}, resources)
            lease = client.call("claim", {})
            assert lease["id"] == job
            name = "grt-smoke-" + job[:24]
            code = "raise SystemExit(7)" if kind == "failure" else "import time; time.sleep(90)"
            command = docker_command(args.image, name, resources, [], [], ["-c", code])
            elapsed = time.monotonic()
            cancellation = None
            if kind == "cancel":
                cancellation = threading.Timer(1, lambda: queue.cancel(job))
                cancellation.start()
            def beat():
                client.call("heartbeat", {"job": job, "token": lease["token"]})
            try:
                run_container(command, args.output / kind, name,
                              timeout=2 if kind == "timeout" else 45, heartbeat=beat)
                raise AssertionError("Expected controlled execution failure")
            except (RuntimeError, TimeoutError) as error:
                reason = type(error).__name__
            except Exception as error:
                from urllib.error import HTTPError
                if kind != "cancel" or not isinstance(error, HTTPError) or error.code != 409:
                    raise
                reason = "CancelledLease"
            finally:
                if cancellation:
                    cancellation.join()
            inspection = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
            assert inspection.returncode != 0 and "No such" in inspection.stderr
            evidence = args.output / kind / "termination.json"
            write(evidence, {"container": name, "terminated": True, "inspection_stderr": inspection.stderr})
            if kind == "cancel":
                # Reopen the durable DB as a restarted coordinator would.
                queue = Queue(database)
                assert next(r for r in queue.rows() if r["id"] == job)["state"] == "cancelling"
                queue.reconcile(job, {"job": job, "terminated": True, "evidence_sha256": sha(evidence)})
            else:
                client.call("finish", {"job": job, "token": lease["token"], "success": False,
                                       "result": {"reason": reason, "terminated": True}})
            summary.append({"kind": kind, "reason": reason, "terminated": True,
                            "wall_seconds": time.monotonic() - elapsed})
        write(args.output / "qualification.json", {"status": "passed", "image": args.image,
              "kind": "native-docker-control-not-eda", "checks": summary})
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)


if __name__ == "__main__":
    main()
