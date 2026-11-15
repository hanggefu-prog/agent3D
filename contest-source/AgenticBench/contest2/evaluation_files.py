"""Read-only evaluation files for participant Agents, separate from control state."""
from contextlib import nullcontext
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

CONTAINER_ROOT = "/workspace/evaluations"
COMPRESS_BYTES = 256 * 1024
TEXT_SUFFIXES = {".log", ".rpt", ".csv", ".json", ".jsonl", ".yaml", ".yml",
                 ".txt", ".stdout", ".stderr", ".out", ".err"}
SKIP_DIRS = {"workspace", "runtime", "control", ".git", "__pycache__"}


def publish(root, request_id, feedback, trees):
    """Copy completed public output trees atomically; never mount host control."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", request_id):
        raise ValueError("Invalid evaluation request ID")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("Evaluation reading directory must be a real absolute path")
    final = root / request_id
    visible = CONTAINER_ROOT + "/" + request_id
    reply = {**feedback, "artifacts_dir": visible, "logs_dir": visible,
             "files_index": visible + "/files.json"}
    if final.is_dir() and not final.is_symlink():
        return reply
    stage = Path(tempfile.mkdtemp(prefix="." + request_id + "-", dir=root))
    files = {}
    try:
        for prefix, source in trees:
            prefix_path, source = Path(prefix), Path(source)
            if prefix_path.is_absolute() or ".." in prefix_path.parts:
                raise ValueError("Invalid public output prefix")
            if not source.is_dir() or source.is_symlink():
                continue
            source = source.resolve()
            for directory, folders, leaves in os.walk(source, followlinks=False):
                folders[:] = sorted(name for name in folders if name not in SKIP_DIRS
                    and not (Path(directory) / name).is_symlink())
                for leaf in sorted(leaves):
                    path = Path(directory) / leaf
                    if path.is_symlink() or not path.is_file():
                        continue
                    suffix = Path(path.stem).suffix if path.suffix == ".gz" else path.suffix
                    if suffix not in TEXT_SUFFIXES:
                        continue
                    relative = prefix_path / path.relative_to(source)
                    # Preserve both files when a native log and an existing
                    # gzip archive have the same stem; do not overwrite either.
                    compressed = (path.suffix != ".gz" and path.stat().st_size > COMPRESS_BYTES
                                  and path.name + ".gz" not in leaves)
                    target = stage / relative
                    if compressed:
                        target = target.with_name(target.name + ".gz")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    checksum = hashlib.sha256()
                    size = 0
                    with path.open("rb") as src, target.open("xb") as raw:
                        with (gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0)
                              if compressed else nullcontext(raw)) as dst:
                            while chunk := src.read(1024 * 1024):
                                checksum.update(chunk)
                                size += len(chunk)
                                dst.write(chunk)
                    files[relative.as_posix()] = {
                        "path": target.relative_to(stage).as_posix(),
                        "encoding": "gzip" if compressed or path.suffix == ".gz" else "identity",
                        "source_bytes": size, "source_sha256": checksum.hexdigest()}
        (stage / "files.json").write_text(json.dumps(files, indent=2) + "\n")
        (stage / "feedback.json").write_text(json.dumps(reply, indent=2, ensure_ascii=False) + "\n")
        try:
            stage.rename(final)
        except OSError:
            # Concurrent status queries can publish the same completed request.
            if final.is_symlink() or not (final / "feedback.json").is_file():
                raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return reply
