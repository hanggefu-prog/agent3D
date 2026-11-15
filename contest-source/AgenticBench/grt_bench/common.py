"""Strict JSON and immutable evidence helpers (Python 3.8+)."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # atomic create; never overwrite evidence
    finally:
        temporary.unlink(missing_ok=True)


def finite(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Missing/nonfinite numeric " + label)
    if minimum is not None and value < minimum:
        raise ValueError("Out-of-range " + label)
    return float(value)


def confined(root, name):
    root = Path(root).resolve(strict=True)
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe relative path: " + str(name))
    path = root / relative
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Symlink in evidence path: " + str(name))
    path.resolve().relative_to(root)
    return path


def records(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Evidence cannot contain symlinks: " + str(path))
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    return result


def verify_records(root, entries):
    if not entries:
        raise ValueError("Empty artifact inventory")
    for name, row in entries.items():
        path = confined(root, name)
        if not path.is_file() or row != {"bytes": path.stat().st_size, "sha256": sha(path)}:
            raise ValueError("Artifact changed: " + name)


def evaluator_identity(package=None):
    package = Path(package) if package is not None else Path(__file__).resolve().parent
    inputs = {"grt_bench/" + p.name: sha(p) for p in sorted(package.glob("*.py"))}
    helper = package.parent / "scripts/workspace_cts_thermal.py"
    inputs["scripts/workspace_cts_thermal.py"] = sha(helper)
    return digest(inputs)
