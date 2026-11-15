"""Transactional host resource reservations. Keep this SQLite DB on local disk.

Expired running leases become uncertain, never automatically rerun: the old EDA
process may still be alive. Reconciliation must confirm its termination first.
"""
import json
from collections import Counter
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from .common import digest, finite
from .contract import identifier


class Queue:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS hosts(id TEXT PRIMARY KEY, capacity TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, resources TEXT NOT NULL,
                state TEXT NOT NULL, host TEXT, token TEXT, expires REAL,
                result TEXT, submitted REAL NOT NULL, started REAL, finished REAL);
              CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, job TEXT, state TEXT, detail TEXT);
            """)
            if "allocation" not in {r[1] for r in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN allocation TEXT")

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def event(db, job, state, detail=None):
        db.execute("INSERT INTO events(at,job,state,detail) VALUES(?,?,?,?)",
                   (time.time(), job, state, json.dumps(detail)))

    def register_host(self, name, capacity):
        identifier(name)
        if set(capacity) - {"gpu_slots_per_device"} != {"cpu", "memory_gb", "storage_gb", "gpus", "build_slots"}:
            raise ValueError("Explicit host resource capacities required")
        slots = capacity.get("gpu_slots_per_device", 1)
        if type(slots) is not int or slots < 1:
            raise ValueError("GPU slots per device must be a positive integer")
        for k in ("cpu", "memory_gb", "storage_gb", "build_slots"):
            finite(capacity[k], k, 0)
        if not isinstance(capacity["gpus"], list) or len(set(capacity["gpus"])) != len(capacity["gpus"]):
            raise ValueError("Invalid GPU inventory")
        for gpu in capacity["gpus"]:
            identifier(gpu)
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM jobs WHERE host=? AND state IN ('running','uncertain','cancelling')", (name,)).fetchone():
                existing = db.execute('SELECT capacity FROM hosts WHERE id=?', (name,)).fetchone()
                if existing and json.loads(existing[0]) == capacity:
                    return
                raise ValueError("Cannot change capacity while leases exist")
            db.execute("INSERT OR REPLACE INTO hosts VALUES(?,?)", (name, json.dumps(capacity)))

    def submit(self, payload, resources):
        if set(resources) != {"cpu", "memory_gb", "storage_gb", "gpu_count", "build_slots"}:
            raise ValueError("Explicit job resource request required")
        for k, v in resources.items():
            finite(v, k, 0)
        if type(resources["gpu_count"]) is not int:
            raise ValueError("GPU count must be integer")
        if resources["cpu"] <= 0 or resources["memory_gb"] <= 0:
            raise ValueError("CPU and memory must be positive")
        job = digest({"payload": payload, "resources": resources})
        with self.transaction() as db:
            inserted = db.execute("INSERT OR IGNORE INTO jobs(id,payload,resources,state,submitted) VALUES(?,?,?,'queued',?)",
                (job, json.dumps(payload, sort_keys=True), json.dumps(resources), time.time())).rowcount
            if inserted:
                self.event(db, job, "queued")
        return job

    def _expire(self, db, now):
        for row in db.execute("SELECT id FROM jobs WHERE state='running' AND expires<?", (now,)).fetchall():
            db.execute("UPDATE jobs SET state='uncertain' WHERE id=?", (row["id"],))
            self.event(db, row["id"], "uncertain", "Lease expired; resources stay reserved until reconciliation")

    def claim(self, host, ttl=120, *, kind='evaluation', job_id=None):
        finite(ttl, "lease duration", 1)
        with self.transaction() as db:
            now = time.time()
            self._expire(db, now)
            row = db.execute("SELECT capacity FROM hosts WHERE id=?", (host,)).fetchone()
            if row is None:
                raise ValueError("Unknown host")
            capacity = json.loads(row[0])
            free = dict(capacity)
            used_gpus = Counter()
            for row in db.execute("SELECT resources,allocation,result FROM jobs WHERE host=? AND state IN ('running','uncertain','cancelling')", (host,)):
                used = json.loads(row["allocation"] or row["resources"])
                for k in ("cpu", "memory_gb", "storage_gb", "build_slots"):
                    free[k] -= used[k]
                used_gpus.update(json.loads(row["result"])["gpus"])
            slots = capacity.get("gpu_slots_per_device", 1)
            available = sorted((gpu for gpu in capacity["gpus"] if used_gpus[gpu] < slots),
                               key=lambda gpu: (used_gpus[gpu], gpu))
            for row in db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY submitted,id").fetchall():
                needed = json.loads(row["resources"])
                payload = json.loads(row["payload"])
                if (payload.get('kind', 'evaluation') != kind
                        or (job_id is not None and row['id'] != job_id)):
                    continue
                if payload.get("host") not in (None, host):
                    continue
                if any(needed[k] > free[k] for k in ("cpu", "memory_gb", "storage_gb", "build_slots")) or needed["gpu_count"] > len(available):
                    continue
                token = uuid.uuid4().hex
                assignment = {"gpus": available[:needed["gpu_count"]]}
                db.execute("UPDATE jobs SET state='running',host=?,token=?,expires=?,started=?,result=?,allocation=? WHERE id=?",
                    (host, token, now + ttl, now, json.dumps(assignment), json.dumps(needed), row["id"]))
                self.event(db, row["id"], "running", {"host": host, **assignment})
                return {"id": row["id"], "token": token, "host": host,
                        "payload": payload, "resources": needed, **assignment}
        return None

    def release(self, job, token, resources, evidence):
        """Shrink a live allocation after its GPU container has terminated.

        The submitted resources remain immutable and continue to identify the
        job. Expired/cancelled leases cannot release capacity without host
        reconciliation. No resource expansion or GPU reacquisition is allowed.
        """
        with self.transaction() as db:
            self._expire(db, time.time())
            row = db.execute("SELECT * FROM jobs WHERE id=? AND token=? AND state='running'", (job, token)).fetchone()
            if row is None:
                raise ValueError("Stale/cancelled lease; cannot release resources")
            before = json.loads(row["allocation"] or row["resources"])
            if set(resources) != set(before) or type(resources.get("gpu_count")) is not int:
                raise ValueError("Invalid resource release")
            for key, value in resources.items():
                finite(value, key, 0)
                if value > before[key]:
                    raise ValueError("Resource release cannot increase allocations")
            if resources["gpu_count"] != 0 or resources["cpu"] <= 0 or resources["memory_gb"] <= 0:
                raise ValueError("Release requires a positive CPU allocation and no GPU")
            import re
            if (evidence.get("terminated") is not True
                    or evidence.get("container") != "grt-" + job[:32] + "-frontend"
                    or not re.fullmatch(r"[0-9a-f]{64}", evidence.get("checkpoint_sha256", ""))):
                raise ValueError("Verified GPU-container termination and checkpoint required")
            assignment = json.loads(row["result"])
            assignment = {**assignment, "gpus": [], "phase": "cpu-backend"}
            if before["gpu_count"]:
                assignment["previous_gpus"] = json.loads(row["result"])["gpus"]
            db.execute("UPDATE jobs SET allocation=?,result=? WHERE id=?", (json.dumps(resources), json.dumps(assignment), job))
            self.event(db, job, "resources_released", {"before": before, "after": resources, "evidence": evidence})
            return {"resources": resources, **assignment}

    def heartbeat(self, job, token, ttl=120):
        finite(ttl, "lease duration", 1)
        with self.transaction() as db:
            self._expire(db, time.time())
            changed = db.execute("UPDATE jobs SET expires=? WHERE id=? AND token=? AND state='running'",
                                 (time.time() + ttl, job, token)).rowcount
            if not changed:
                raise ValueError("Stale/cancelled lease; worker must stop")

    def finish(self, job, token, result, success):
        if type(success) is not bool:
            raise ValueError("Success must be boolean")
        with self.transaction() as db:
            self._expire(db, time.time())
            state = "succeeded" if success else "failed"
            changed = db.execute("UPDATE jobs SET state=?,result=?,finished=?,expires=NULL WHERE id=? AND token=? AND state='running'",
                (state, json.dumps(result, allow_nan=False), time.time(), job, token)).rowcount
            if not changed:
                raise ValueError("Stale result; reconcile first, never double count")
            self.event(db, job, state, result)

    def cancel(self, job):
        with self.transaction() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None:
                raise ValueError("Unknown job")
            if row[0] == "queued":
                db.execute("UPDATE jobs SET state='cancelled',finished=? WHERE id=?", (time.time(), job))
                self.event(db, job, "cancelled")
            elif row[0] in {"running", "uncertain"}:
                db.execute("UPDATE jobs SET state='cancelling' WHERE id=?", (job,))
                self.event(db, job, "cancelling")

    def reconcile(self, job, termination_receipt):
        # Caller must supply a host-owned process/container termination receipt.
        if termination_receipt.get("job") != job or termination_receipt.get("terminated") is not True or not termination_receipt.get("evidence_sha256"):
            raise ValueError("Termination evidence required")
        with self.transaction() as db:
            changed = db.execute("UPDATE jobs SET state='interrupted',finished=?,expires=NULL,result=? WHERE id=? AND state IN ('uncertain','cancelling')",
                (time.time(), json.dumps(termination_receipt), job)).rowcount
            if not changed:
                raise ValueError("Job does not need reconciliation")
            self.event(db, job, "interrupted", termination_receipt)

    def rows(self):
        with self.transaction() as db:
            self._expire(db, time.time())
            return [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY submitted,id")]
