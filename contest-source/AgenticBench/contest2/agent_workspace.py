"""Keep participant writes outside the read-only framework and physical source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


def _files(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Agent directory must be real")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Agent files cannot be symlinks")
        if "__pycache__" in path.relative_to(root).parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
        elif not path.is_dir():
            raise ValueError("Agent entries must be regular files or directories")
    return files


def agent_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for name, path in _files(Path(root)).items():
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _target(root: Path, name: str) -> Path:
    target = root / name
    for path in (target, *target.parents):
        if path == root.parent:
            break
        if path.is_symlink():
            raise ValueError("Staged Agent paths cannot be symlinks")
    return target


def stage_agent(template: Path, destination: Path) -> Path:
    """Copy changed developer files while retaining files evolved by the Agent."""
    template, destination = Path(template), Path(destination)
    files = _files(template)
    source = template.resolve(strict=True)
    target = destination.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Agent workspace must be separate from its template")
    if destination.is_symlink():
        raise ValueError("Agent workspace cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination.parent / (destination.name + ".template.json")
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    current = {name: hashlib.sha256(path.read_bytes()).hexdigest()
               for name, path in files.items()}
    for name, path in files.items():
        staged = _target(destination, name)
        if previous.get(name) != current[name] or not staged.exists():
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, staged)
    for name in previous.keys() - current.keys():
        staged = _target(destination, name)
        if staged.exists():
            if not staged.is_file():
                raise ValueError("Removed Agent template file became a directory")
            staged.unlink()
    pending = manifest.with_suffix(".pending")
    pending.write_text(json.dumps(current, sort_keys=True) + "\n")
    pending.replace(manifest)
    return destination
