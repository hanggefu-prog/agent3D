from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .model import TaskSpec
from .patch_paths import split_patch_sections


def flatten_paths(value: Mapping[str, Any], prefix: str = "") -> Set[str]:
    result: Set[str] = set()
    for key, child in value.items():
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        if isinstance(child, dict):
            nested = flatten_paths(child, path)
            result.update(nested or {path})
        else:
            result.add(path)
    return result


def _matches(path: str, allowed: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in allowed)


def validate_config_scope(spec: TaskSpec, overrides: Mapping[str, Any]) -> None:
    changed = flatten_paths(overrides)
    rejected = sorted(path for path in changed if not _matches(path, spec.allowed_config_paths))
    if rejected:
        raise ValueError(
            "candidate changes config outside this task scope: {}".format(", ".join(rejected))
        )


def patch_paths(patch_text: str) -> List[str]:
    return sorted({path for path, _ in split_patch_sections(patch_text)})


def patch_line_counts(patch_text: str) -> Tuple[int, int]:
    added = 0
    deleted = 0
    for line in patch_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return added, deleted


def validate_patch_text(spec: TaskSpec, text: str) -> Dict[str, Any]:
    if not text.strip():
        return {"present": False, "paths": [], "added_lines": 0, "deleted_lines": 0}
    if "GIT binary patch" in text or "Binary files " in text:
        raise ValueError("binary patches are not allowed")
    if any(line.startswith(("rename from ", "rename to ", "new file mode ", "deleted file mode ")) for line in text.splitlines()):
        raise ValueError("file creation, deletion, and rename are not allowed")
    paths = patch_paths(text)
    if not paths:
        raise ValueError("patch does not contain a standard file header")
    if "/dev/null" in text:
        raise ValueError("file creation and deletion are not allowed")
    if not any(line.startswith("@@ ") for line in text.splitlines()):
        raise ValueError("patch does not contain a unified diff hunk")
    if not spec.allowed_source_paths:
        raise ValueError("this task does not allow source changes")
    rejected = [path for path in paths if path not in spec.allowed_source_paths]
    if rejected:
        raise ValueError("patch changes files outside this task scope: {}".format(", ".join(rejected)))
    added, deleted = patch_line_counts(text)
    if spec.max_changed_files and len(paths) > spec.max_changed_files:
        raise ValueError("patch changes {} files; limit is {}".format(len(paths), spec.max_changed_files))
    if spec.max_added_lines and added > spec.max_added_lines:
        raise ValueError("patch adds {} lines; limit is {}".format(added, spec.max_added_lines))
    if spec.max_deleted_lines and deleted > spec.max_deleted_lines:
        raise ValueError("patch deletes {} lines; limit is {}".format(deleted, spec.max_deleted_lines))
    return {"present": True, "paths": paths, "added_lines": added, "deleted_lines": deleted}


def validate_patch_scope(spec: TaskSpec, patch_path: Path) -> Dict[str, Any]:
    if not patch_path.is_file():
        return {"present": False, "paths": [], "added_lines": 0, "deleted_lines": 0}
    return validate_patch_text(spec, patch_path.read_text(encoding="utf-8"))
