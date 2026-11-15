"""Host-only inspection of uncertain executions; never replay an unknown job."""
import json
import os
from pathlib import Path
import subprocess
import time

from grt_bench.common import digest, read, write, sha


def process_identity(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def record_owner(path):
    write(path, {"pid": os.getpid(), "identity": process_identity(os.getpid()),
                 "execution_policy_sha256": sha(Path(__file__).with_name("execution_policy.py"))})


def _inspect_execution(queue, jobs, names, owner_path, audit_path):
    """Inspect only host-derived names. Live/unknown owners keep their allocation."""
    report = {"checked_at": time.time(), "terminated": False, "containers": [], "expected_containers": sorted(names)}
    owner_path = Path(owner_path)
    if owner_path.exists():
        owner = read(owner_path)
        if not owner.get("identity"):
            report["reason"] = "Execution-owner identity is unavailable; continuing inspection"
            write(audit_path, report, replace=Path(audit_path).exists())
            return report
        if owner["pid"] != os.getpid() and process_identity(owner["pid"]) == owner["identity"]:
            report["reason"] = "Original evaluation controller is still alive"
            write(audit_path, report, replace=Path(audit_path).exists())
            return report
        if owner["pid"] != os.getpid() and not jobs:
            report["reason"] = ("Original controller ended before publishing a native job; "
                                "retained setup/build execution needs host inspection")
            write(audit_path, report, replace=Path(audit_path).exists())
            return report
    else:
        report["reason"] = "Missing execution-owner record; automatic replay is withheld"
        write(audit_path, report, replace=Path(audit_path).exists())
        return report
    try:
        result = subprocess.run(["docker", "container", "ls", "--all", "--no-trunc",
                                 "--format", "{{json .}}"], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Docker state is unavailable")
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        for row in rows:
            if row["Names"] not in names:
                continue
            probe = subprocess.run(["docker", "container", "inspect", row["ID"],
                                    "--format", "{{json .State}}"], capture_output=True, text=True)
            if probe.returncode:
                raise RuntimeError("Container state changed during inspection")
            state = json.loads(probe.stdout)
            report["containers"].append({"name": row["Names"], "id": row["ID"], "state": state})
        report["terminated"] = all(r["state"].get("Status") in {"exited", "dead"}
            and not r["state"].get("Running") and not r["state"].get("Restarting")
            for r in report["containers"])
        report["reason"] = ("Original execution has ended; checking its retained result"
                            if report["terminated"] else "Original execution is still present; continuing observation")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        report["reason"] = type(error).__name__ + ": " + str(error)
    write(audit_path, report, replace=Path(audit_path).exists())
    if report["terminated"]:
        for row in jobs:
            if row["state"] in {"queued", "running", "uncertain"}:
                queue.cancel(row["id"])
            current = next(r for r in queue.rows() if r["id"] == row["id"])
            if current["state"] in {"uncertain", "cancelling"}:
                queue.reconcile(row["id"], {"job": row["id"], "terminated": True,
                    "evidence_sha256": digest(report), "inspection": str(audit_path)})
    return report


def inspect_execution(queue, jobs, names, owner_path, audit_path):
    try:
        return _inspect_execution(queue, jobs, names, owner_path, audit_path)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        # A failed probe or changed queue is another observation, never proof
        # of termination. The next poll rechecks without duplicating execution.
        return {"checked_at": time.time(), "terminated": False,
                "reason": "Inspection incomplete: " + type(error).__name__ + ": " + str(error)}
