"""Per-capability evidence (capability_schema 2).

Version-1 producers stamped every capability {'passed': True} with one shared
evidence file, so a capability record proved nothing about that capability.
Schema-2 records describe one concrete check each, in their own evidence file:

    {"capability_schema": 2, "status": "passed" | "failed" | "unverified",
     "passed": status == "passed", "check": "...", "measured": ...,
     "threshold": ..., "sources": [{"path", "sha256"}, ...],
     "evidence": {"path": "capabilities/<name>.json", "sha256": ...}}

"unverified" means the producer had no concrete check for that capability; it
is never counted as passed (only an explicit, profile-declared allow-list may
let a qualification proceed with it, and it is still reported).
"""
from pathlib import Path

from .protocol import artifact, sha, write_new
from .tasks import read_json

STATUSES = ("passed", "failed", "unverified")
FIELDS = ("capability", "status", "check", "measured", "threshold", "sources")


def source(path, root=None):
    """Artifact reference; relative to root when inside it, else absolute."""
    path = Path(path).resolve(strict=True)
    if root is not None:
        try:
            return {"path": path.relative_to(Path(root).resolve()).as_posix(), "sha256": sha(path)}
        except ValueError:
            pass
    return {"path": str(path), "sha256": sha(path)}


class Recorder:
    """Write one evidence file per capability under <root>/capabilities."""

    def __init__(self, root):
        self.root = Path(root)
        self.records = {}

    def add(self, name, *, check, measured, threshold, passed, sources=()):
        """passed=None records an unverified capability, never a pass."""
        if name in self.records:
            raise ValueError("Repeated capability record: " + name)
        if passed is None:
            status = "unverified"
        elif passed is True or passed is False:
            status = "passed" if passed else "failed"
        else:
            raise ValueError("Capability outcome must be True, False or None")
        refs = [row if isinstance(row, dict) else source(row, self.root) for row in sources]
        if status != "unverified" and not refs:
            raise ValueError("A checked capability needs a source artifact: " + name)
        body = {"capability": name, "status": status, "check": check, "measured": measured,
                "threshold": threshold, "sources": refs}
        path = self.root / "capabilities" / (name + ".json")
        write_new(path, body)
        record = {"capability_schema": 2, "passed": status == "passed",
                  "evidence": {"path": path.relative_to(self.root).as_posix(), "sha256": sha(path)}}
        record.update(body)
        del record["capability"]
        self.records[name] = record
        return record

    def complete(self, names):
        """Every required capability, with missing ones explicitly unverified."""
        for name in names:
            if name not in self.records:
                self.add(name, check="no concrete check implemented by this producer",
                         measured=None, threshold=None, passed=None)
        extra = set(self.records) - set(names)
        if extra:
            raise ValueError("Unexpected capability records: " + ", ".join(sorted(extra)))
        return {name: self.records[name] for name in names}

    def failures(self):
        return sorted(k for k, v in self.records.items() if v["status"] == "failed")


def verify(root, records, names, *, allow_unverified=(), require_schema=2,
           missing="Unqualified capability: "):
    """Verify capability records; return a summary that separates statuses.

    require_schema=1 accepts frozen v1 records ({'passed': True} + shared
    evidence) and reports them as such; schema-2 records are always checked
    against their own evidence file.
    """
    if not isinstance(records, dict):
        raise ValueError("Invalid native capabilities")
    summary = {"passed": [], "unverified": [], "failed": [], "legacy_shared_v1": [],
               "sources_unavailable": []}
    for name in names:
        record = records.get(name)
        if not isinstance(record, dict):
            raise ValueError(missing + name)
        if record.get("capability_schema") != 2:
            if require_schema == 2:
                raise ValueError("Capability lacks per-check evidence (schema 2): " + name)
            if record.get("passed") is not True:
                raise ValueError(missing + name)
            artifact(root, record.get("evidence"))
            summary["legacy_shared_v1"].append(name)
            continue
        expected = {"capability_schema", "status", "passed", "check", "measured", "threshold",
                    "sources", "evidence"}
        if set(record) != expected or record["status"] not in STATUSES:
            raise ValueError("Invalid capability record: " + name)
        if record["passed"] is not (record["status"] == "passed"):
            raise ValueError("Capability passed flag contradicts status: " + name)
        body = read_json(artifact(root, record["evidence"]))
        if body != {"capability": name, **{k: record[k] for k in FIELDS[1:]}}:
            raise ValueError("Capability record differs from its evidence: " + name)
        if not isinstance(record["check"], str) or not record["check"].strip():
            raise ValueError("Capability needs a described check: " + name)
        if record["status"] != "unverified" and not record["sources"]:
            raise ValueError("Checked capability lacks source artifacts: " + name)
        for row in record["sources"]:
            if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
                raise ValueError("Invalid capability source: " + name)
            path = Path(row["path"])
            if not path.is_absolute():
                artifact(root, row)
            elif path.is_file():
                if sha(path) != row["sha256"]:
                    raise ValueError("Capability source changed: " + row["path"])
            else:
                summary["sources_unavailable"].append(row["path"])
        summary[record["status"]].append(name)
    if summary["failed"]:
        raise ValueError("Failed physical capability: " + ", ".join(summary["failed"]))
    blocked = [x for x in summary["unverified"] if x not in allow_unverified]
    if blocked:
        raise ValueError("Unverified capability is not a pass: " + ", ".join(blocked))
    return summary


def status_map(records):
    """Boolean view for downstream consumers; unverified/legacy-missing -> False."""
    result = {}
    for name, record in records.items():
        if isinstance(record, dict) and record.get("capability_schema") == 2:
            result[name] = record.get("status") == "passed"
        else:
            result[name] = isinstance(record, dict) and record.get("passed") is True
    return result
