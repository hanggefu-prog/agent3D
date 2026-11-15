"""Small versioned contracts for the engineering track (not OpenEvolve tasks)."""
from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

TASK_ROOT = Path(__file__).resolve().parents[1] / "engineering_tasks"
TASK_IDS = ("clock_compensation", "multi_hbt_fanout", "cross_die_cts")


def read_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value == "." or ".." in path.parts or value != path.as_posix() or any(ord(c) < 32 for c in value):
        raise ValueError("Unsafe relative path: " + value)
    return value


@dataclass(frozen=True)
class EngineeringTask:
    root: Path
    data: dict
    instruction: str
    judge_instruction: str

    @property
    def task_id(self):
        return self.data["id"]

    def digest(self):
        value = {"task": self.data, "instruction": self.instruction,
                 "judge_instruction": self.judge_instruction,
                 "interface_documents": {p.name: p.read_text() for p in sorted(self.root.glob("*.md"))}}
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def allows(self, path):
        path = relative(path)
        return any(fnmatch.fnmatchcase(path, pattern) for pattern in self.data["editable_paths"])


def load_task(task_id, root=None):
    root = Path(root or os.environ.get("ENGINEERING_TASK_ROOT", TASK_ROOT))
    if task_id not in TASK_IDS:
        raise ValueError("Unknown engineering task: " + str(task_id))
    folder = Path(root) / task_id
    data = read_json(folder / "task.json")
    required = {"schema_version", "id", "legacy_id", "title", "level", "instruction",
                "judge_instruction", "input_stage", "endpoint", "primary_metric", "direction",
                "seed", "cases", "submissions", "evaluations_per_case", "repetitions",
                "development_timeout_seconds", "judge_timeout_seconds", "entrypoint",
                "editable_paths", "required_capabilities", "qualification_required"}
    if set(data) != required or data["id"] != task_id or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Invalid engineering task schema")
    for key in ("submissions", "evaluations_per_case", "repetitions"):
        if type(data[key]) is not int or data[key] != 1:
            raise ValueError(key + " must be exactly one")
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("An explicit nonnegative algorithm seed is required")
    if data["direction"] != "minimize" or data["qualification_required"] is not True:
        raise ValueError("Unsupported objective or unqualified execution")
    for key in ("development_timeout_seconds", "judge_timeout_seconds"):
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError("A positive bounded timeout is required")
    for key in ("cases", "editable_paths", "required_capabilities"):
        items = data[key]
        if not isinstance(items, list) or not items or any(not isinstance(x, str) or not x for x in items):
            raise ValueError("Invalid " + key)
        if len(items) != len(set(items)):
            raise ValueError("Duplicate " + key)
    for item in data["cases"] + data["editable_paths"]:
        relative(item)
    texts = []
    for key in ("instruction", "judge_instruction"):
        path = folder / relative(data[key])
        if path.is_symlink() or not path.is_file():
            raise ValueError("Prompt must be a packaged regular file")
        text = path.read_text()
        if not text.strip():
            raise ValueError("Empty prompt")
        texts.append(text)
    return EngineeringTask(folder, data, *texts)
