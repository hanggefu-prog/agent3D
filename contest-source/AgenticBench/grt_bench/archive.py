"""Transactional host-owned Pareto history, parent selection and result admission.

Agents receive prompts/workspaces, never this database or the queue credentials.
Only completed worker receipts can enter the archive; aggregate JSON is not an
admission format. Dominated history and in-flight parent selections are retained.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .common import digest, read, sha, verify_records
from .contract import matrix, validate
from .optimization import edit_policy, identity, verify_candidate
from .pareto import aggregate, frontier


def completed_outcomes(root, task, candidate, expected_audit):
    """Check host queue, independent verifier receipt and immutable evidence hashes."""
    root = Path(root)
    expected = matrix(task, candidate, "train")
    database = root / "control/queue.sqlite"
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        jobs = [(dict(row), json.loads(row["payload"])) for row in db.execute("SELECT * FROM jobs")]
    finally:
        db.close()
    jobs = [(row, payload) for row, payload in jobs if payload.get("task_digest") == validate(task)
            and payload.get("candidate") == candidate and payload.get("split") == "train"]
    if len(jobs) != len(expected):
        raise ValueError("Training matrix is incomplete or contains duplicate attempts")
    remaining = {digest(p): p for p in expected}
    outcomes = []
    for row, payload in jobs:
        key = digest(payload)
        if key not in remaining or row["state"] != "succeeded":
            raise ValueError("Every unique training cell must succeed before archive admission")
        remaining.pop(key)
        if digest({"payload": payload, "resources": json.loads(row["resources"])}) != row["id"]:
            raise ValueError("Queue job identity differs")
        attempt = root / "runs" / row["id"]
        result = json.loads(row["result"])
        if result != read(attempt / "result.json"):
            raise ValueError("Worker receipt differs from the authoritative queue result")
        verified = read(attempt / "verified/result.json")
        if {k: v for k, v in result.items() if k not in {"accounting", "candidate_verification_sha256"}} != verified:
            raise ValueError("Worker result differs from independent verifier receipt")
        if any(result.get(k) != v for k, v in payload.items()) or result.get("status") != "verified":
            raise ValueError("Verifier receipt does not match the submitted cell")
        evidence = result["evidence"]
        plan_path = attempt / "control/plan.json"
        analysis = attempt / "output/analysis"
        if sha(plan_path) != evidence["plan_sha256"] or sha(analysis / "manifest.json") != evidence["analysis_manifest_sha256"]:
            raise ValueError("Verified evidence changed")
        if read(attempt / "control/task.json") != task or any(read(plan_path).get(k) != v for k, v in payload.items()):
            raise ValueError("Verified plan/task differs")
        verify_records(analysis, read(analysis / "manifest.json")["artifacts"])
        if result.get("candidate_verification_sha256") != sha(attempt / "candidate-verification.json"):
            raise ValueError("Missing/changed candidate legality receipt")
        audit = read(attempt / "candidate-verification.json")
        if audit != expected_audit:
            raise ValueError("Candidate legality receipt differs")
        outcomes.append(result)
    return sorted(outcomes, key=lambda r: (r["case"], r["seed"]["id"]))


class Archive:
    def __init__(self, root, task):
        self.root, self.task = Path(root), copy.deepcopy(task)
        self.task_id = validate(task)
        if "optimization" not in task:
            raise ValueError("Archive requires a frozen optimization task")
        self.path = self.root / "control/archives" / (self.task_id + ".sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS evaluations(candidate TEXT PRIMARY KEY, row_json TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS selections(id TEXT PRIMARY KEY, receipt TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS children(candidate TEXT PRIMARY KEY, parent TEXT NOT NULL, selection TEXT NOT NULL)")
            previous = db.execute("SELECT value FROM metadata WHERE key='task'").fetchone()
            if previous and json.loads(previous[0]) != task:
                raise ValueError("Archive task changed")
            db.execute("INSERT OR IGNORE INTO metadata VALUES('task',?)", (json.dumps(task, sort_keys=True),))

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
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
    def _rows(db):
        return [json.loads(r[0]) for r in db.execute("SELECT row_json FROM evaluations ORDER BY candidate")]

    @staticmethod
    def _frontier(rows):
        return frontier(rows, retain_equivalent=True)

    def show(self):
        with self.transaction() as db:
            rows = self._rows(db)
            return {"task_digest": self.task_id, "evaluated": rows, "frontier": self._frontier(rows),
                    "selections": [json.loads(r[0]) for r in db.execute("SELECT receipt FROM selections ORDER BY rowid")],
                    "children": [{"candidate": r[0], "parent": r[1], "selection": r[2]}
                                 for r in db.execute("SELECT * FROM children ORDER BY candidate")]}

    def select(self, candidate, reason):
        identity(candidate)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Record why this parent is promising")
        with self.transaction() as db:
            current = self._frontier(self._rows(db))
            if candidate not in {r["candidate"] for r in current}:
                raise ValueError("Parent must be on the current training frontier")
            receipt = {"id": uuid.uuid4().hex, "task_digest": self.task_id, "parent": candidate,
                       "reason": reason, "selected_at": time.time(), "frontier_digest": digest(current)}
            db.execute("INSERT INTO selections VALUES(?,?)", (receipt["id"], json.dumps(receipt)))
        return receipt

    def _selected_parent(self, selection):
        with self.transaction() as db:
            found = db.execute("SELECT receipt FROM selections WHERE id=?", (selection,)).fetchone()
            if not found:
                raise ValueError("Unknown parent selection")
            return json.loads(found[0])["parent"]

    def checkout(self, output, selection):
        from .candidates import materialize
        parent = self._selected_parent(selection)
        self.admit(parent)
        materialize(self.root / "candidates" / parent, output)
        return {"task_digest": self.task_id, "parent": parent, "selection": selection,
                "workspace": str(Path(output).resolve()), "prompt": self.task["optimization"]["prompt"],
                "allowed_files": self.task["optimization"]["allowed_files"]}

    def freeze(self, source, selection):
        from .candidates import freeze
        parent = self._selected_parent(selection)
        # A selected parent may later become dominated while its child is being edited.
        verify_candidate(self.task, self.root / "candidates", parent)
        candidate = freeze(source, self.root / "candidates", parent=parent, policy=edit_policy(self.task))
        verify_candidate(self.task, self.root / "candidates", candidate)
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO children VALUES(?,?,?)", (candidate, parent, selection))
        return candidate

    def admit(self, candidate):
        audit = verify_candidate(self.task, self.root / "candidates", candidate)
        if candidate != self.task["optimization"]["baseline"]:
            with self.transaction() as db:
                row = db.execute("SELECT parent,selection FROM children WHERE candidate=?", (candidate,)).fetchone()
                selection = db.execute("SELECT receipt FROM selections WHERE id=?", (row[1],)).fetchone() if row else None
                if not row or not selection or row[0] != audit["parent"] or json.loads(selection[0])["parent"] != row[0]:
                    raise ValueError("Candidate was not frozen from an archived parent selection")
        return audit

    def ingest(self, candidate):
        audit = self.admit(candidate)
        outcomes = completed_outcomes(self.root, self.task, candidate, audit)
        baseline = self.task["optimization"]["baseline"]
        with self.transaction() as db:
            reference = db.execute("SELECT value FROM metadata WHERE key='reference'").fetchone()
            if not reference:
                if candidate != baseline:
                    raise ValueError("Admit the complete frozen baseline matrix first")
                db.execute("INSERT INTO metadata VALUES('reference',?)", (json.dumps(outcomes),))
                refs = outcomes
            else:
                refs = json.loads(reference[0])
            row = aggregate(self.task, candidate, outcomes, refs)
            previous = db.execute("SELECT row_json FROM evaluations WHERE candidate=?", (candidate,)).fetchone()
            if previous and json.loads(previous[0]) != row:
                raise ValueError("Conflicting evidence for an archived candidate")
            before = {r["candidate"] for r in self._frontier(self._rows(db))}
            db.execute("INSERT OR IGNORE INTO evaluations VALUES(?,?)", (candidate, json.dumps(row, sort_keys=True)))
            after = {r["candidate"] for r in self._frontier(self._rows(db))}
            return {"candidate": candidate, "feasible": row["feasible"], "non_dominated": candidate in after,
                    "new_frontier_members": sorted(after - before), "now_dominated": sorted(before - after),
                    "evaluation": row}
