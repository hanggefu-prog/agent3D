"""Fail-closed physical qualification and exactly-once case dispatch.

Qualification is issued by the host-owned native adapter, never by a candidate
or an LLM. A contract or unit-test fixture is not a physical qualification.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .tasks import read_json, relative


def now():
    return datetime.now(timezone.utc).isoformat()


def path_within(path, root):
    """Containment for resolved paths, also usable by Python 3.8 host launchers."""
    try:
        Path(path).relative_to(root)
    except ValueError:
        return False
    return True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def artifact(root, row):
    if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
        raise ValueError("Invalid evidence artifact")
    root = Path(root).resolve(strict=True)
    path = (root / relative(row["path"])).resolve(strict=True)
    path.relative_to(root)
    if not path.is_file() or sha(path) != row["sha256"]:
        raise ValueError("Changed/missing qualification evidence: " + str(path))
    return path


def qualify(task, case, qualification_path):
    if case not in task.data["cases"]:
        raise ValueError("Unexpected case")
    path = Path(qualification_path)
    value = read_json(path)
    identity = {"kind": "engineering-native-qualification-v1", "task": task.task_id,
                "task_digest": task.digest(), "case": case, "seed": task.data["seed"],
                "input_stage": task.data["input_stage"], "endpoint": task.data["endpoint"],
                "physical_success": True}
    for key, expected in identity.items():
        if type(value.get(key)) is not type(expected) or value[key] != expected:
            raise ValueError("Qualification mismatch: " + key)
    for key in ("source_sha256", "runtime_sha256", "input_sha256", "evaluator_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, ""))):
            raise ValueError("Missing native provenance: " + key)
    from .scoring import profile_for, scoring_version
    version = scoring_version(value)
    profile = profile_for(task, value)
    if version == 2 and value.get("identity_version") != 2:
        raise ValueError("A v2 qualification must use evaluator identity v2")
    from .capabilities import verify as verify_capabilities
    capabilities = verify_capabilities(
        path.parent, value.get("capabilities", {}), task.data["required_capabilities"],
        allow_unverified=profile["allow_unverified_capabilities"] if profile else (),
        require_schema=2 if version == 2 else 1)
    reference = artifact(path.parent, value.get("reference"))
    ref = read_json(reference)
    if ref.get("legal") is not True or ref.get("endpoint") != task.data["endpoint"] or ref.get("case") != case:
        raise ValueError("Reference did not pass this task's physical endpoint")
    from .metrics import number
    if number(ref.get("primary_metric"), "reference primary metric") <= 0:
        raise ValueError("Reference metric must be positive for relative scoring")
    return {"path": str(path.resolve()), "sha256": sha(path), "record": value,
            "reference_metric": ref["primary_metric"], "scoring_version": version,
            "capabilities": capabilities}


class OnceLedger:
    """One sealed program, one case attempt; failures stay consumed and visible."""

    def __init__(self, root, task, submission_sha256):
        self.root = Path(root)
        self.task = task
        if not re.fullmatch(r"[0-9a-f]{64}", submission_sha256):
            raise ValueError("Missing sealed submission identity")
        self.identity = {"schema_version": 1, "task": task.task_id,
                         "task_digest": task.digest(), "seed": task.data["seed"],
                         "submission_sha256": submission_sha256,
                         "evaluations_per_case": 1, "repetitions": 1,
                         "cases": task.data["cases"]}
        self.root.mkdir(parents=True, exist_ok=True)
        seal = self.root / "submission.json"
        try:
            write_new(seal, self.identity)
        except FileExistsError:
            if read_json(seal) != self.identity:
                raise ValueError("Cannot replace the sealed submission or task")

    def claim(self, case, qualification_path):
        # Unavailable native adapters never burn candidate slots.
        proof = qualify(self.task, case, qualification_path)
        attempt = {**self.identity, "case": case, "status": "dispatched",
                   "started_at": now(), "qualification_sha256": proof["sha256"],
                   "reference_metric": proof["reference_metric"]}
        if proof["scoring_version"] != 1:
            # Absent on v1 rows, so frozen ledgers and reports are unchanged.
            attempt["scoring_version"] = proof["scoring_version"]
            attempt["scoring_profile_sha256"] = proof["record"]["scoring_profile_sha256"]
        write_new(self.root / case / "attempt.json", attempt)
        return attempt

    def finish(self, case, *, succeeded, legal, candidate_metric=None, error=None,
               artifacts=None, details=None):
        if case not in self.task.data["cases"]:
            raise ValueError("Unexpected case")
        attempt = read_json(self.root / case / "attempt.json")
        if any(attempt.get(k) != v for k, v in self.identity.items()) or attempt.get("case") != case:
            raise ValueError("Attempt identity changed")
        if type(succeeded) is not bool or type(legal) is not bool:
            raise ValueError("Explicit completion and legality required")
        if succeeded:
            from .metrics import number
            if not legal or number(candidate_metric, "candidate_metric") < 0:
                raise ValueError("A successful final must be legal with a finite metric")
        elif candidate_metric is not None:
            raise ValueError("Failure must not fabricate a physical metric")
        result = {**attempt, "status": "succeeded" if succeeded else "failed",
                  "legal": legal, "candidate_metric": candidate_metric,
                  "completed_at": now(), "error": error, "artifacts": artifacts or []}
        if details is not None:
            if not isinstance(details, dict):
                raise ValueError("Result details must be an object")
            result["details"] = details
        write_new(self.root / case / "result.json", result)
        return result

    def records(self):
        result = []
        for case in self.task.data["cases"]:
            folder = self.root / case
            if (folder / "result.json").exists():
                result.append(read_json(folder / "result.json"))
            elif (folder / "attempt.json").exists():
                result.append(read_json(folder / "attempt.json"))
        return result
