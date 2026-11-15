"""Seal task-scoped edits, including additions and native submodule changes.

This is a host-side operation after the development container has stopped.
It does not trust a developer's changed-file list, Git index, or final message.
"""
from pathlib import Path
import os
import stat
import subprocess

from .protocol import digest, sha, write_new
from .tasks import relative


def inventory(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for name in sorted(files + [d for d in dirs if (Path(parent) / d).is_symlink()]):
            path = Path(parent) / name
            if name == ".git":
                continue
            rel = relative(path.relative_to(root).as_posix())
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                result[rel] = {"kind": "symlink", "target": os.readlink(path)}
            elif stat.S_ISREG(mode):
                result[rel] = {"kind": "file", "sha256": sha(path),
                               "executable": bool(mode & 0o111)}
            else:
                raise ValueError("Unsupported source file type: " + rel)
    return result


def validate_scope(task, before, after):
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    rejected = [p for p in changed if not task.allows(p)]
    if rejected:
        raise ValueError("Changes outside task scope: " + ", ".join(rejected))
    for name in changed:
        # Existing repository symlinks may remain untouched. New code links are
        # disallowed because the submitted artifact must have self-contained bytes.
        if name in after and after[name]["kind"] != "file":
            raise ValueError("Changed task code must be a regular file: " + name)
    return changed


def seal(task, subject, baseline_inventory, output):
    output = Path(output)
    current = inventory(subject)
    changes = validate_scope(task, baseline_inventory, current)
    value = {"schema_version": 1, "task": task.task_id, "task_digest": task.digest(),
             "baseline_source_sha256": digest(baseline_inventory),
             "source_sha256": digest(current), "files": current, "changed_paths": changes}
    value["submission_sha256"] = digest(value)
    write_new(output, value)
    return value


def verify_seal(subject, sealed):
    values = dict(sealed)
    claimed = values.pop("submission_sha256")
    if digest(values) != claimed or inventory(subject) != values["files"]:
        raise ValueError("Submitted source changed after sealing")
    return claimed


def clone_source(source, target):
    """Independent recursive clones; no host-linked Git dirs/objects or reset."""
    source, target = Path(source).resolve(strict=True), Path(target).absolute()
    if target.exists():
        raise FileExistsError(target)
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                               cwd=source, text=True).strip():
        raise ValueError("Use a clean committed source for a reproducible task")
    subprocess.run(["git", "clone", "--no-hardlinks", "--no-checkout", str(source), str(target)], check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    subprocess.run(["git", "checkout", "--detach", revision], cwd=target, check=True)
    # The committed tree is authoritative. .gitmodules may retain stale entries
    # after a dependency is vendored as ordinary tracked files (e.g. ABC).
    tree = subprocess.check_output(["git", "ls-tree", "-rz", "HEAD"], cwd=source, text=True)
    for record in tree.split("\0"):
        if not record:
            continue
        metadata, name = record.split("\t", 1)
        mode, kind, expected = metadata.split()
        if mode != "160000":
            continue
        if kind != "commit":
            raise ValueError("Invalid committed gitlink")
        subpath = relative(name)
        origin, dest = source / subpath, target / subpath
        if not (origin / ".git").exists():
            raise ValueError("Source submodule is uninitialized: " + subpath)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
        if actual != expected:
            raise ValueError("Source submodule revision differs from parent: " + subpath)
        # Git creates an empty placeholder for a gitlink, never remove data.
        if dest.exists():
            dest.rmdir()
        clone_source(origin, dest)
    return revision
