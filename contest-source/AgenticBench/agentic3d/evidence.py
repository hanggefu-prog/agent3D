"""Bounded, read-only evidence snapshots for the co-design agents.

The index keeps both the original run-relative citation and a hashed local text
copy. It is diagnostic evidence, never an alternative physical success gate.
"""
from __future__ import annotations

import inspect
import json
import re
import shlex
from pathlib import Path, PurePosixPath

from .knowledge import digest, write_json


def _read_run_evidence(run_name):
    """Self-contained reader sent to the evaluation host; no project imports."""
    import json
    import os
    import pathlib
    import re

    root = pathlib.Path(run_name).resolve()
    stages = (
        "00_input", "10_floorplan", "15_timing_extract", "20_partition",
        "30_placement", "40_pdn", "50_hbt", "60_opt", "70_cts",
        "75_global_route", "80_route", "80_routing", "90_report",
        "95_thermal", "95_thermal_snapshot",
    )
    private = re.compile(
        r"(^|[._-])(env|environment|auth|credentials?|secrets?|tokens?|"
        r"passwords?|api[_-]?keys?)([._-]|$)", re.I)
    error_line = re.compile(r"(?im)^.*(?:\[ERROR\b|\bFATAL\b|Traceback \(most recent|"
                            r"\b(?:RuntimeError|ValueError|AssertionError):|"
                            r"\b(?:failed|failure|segmentation fault)\b).*$")
    items, errors, warnings = [], [], []

    def safe(path, base=root):
        try:
            relative = path.relative_to(base)
            if any(part.startswith(".") or private.search(part) for part in relative.parts):
                return False
            if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
                return False
            path.resolve().relative_to(base)
            return path.is_file()
        except (OSError, ValueError):
            return False

    def priority(path, folder):
        name = path.name.lower()
        if path.parent == folder and name in ("run.json", "stage_manifest.json"):
            rank = 0
        elif re.search(r"failure|failed|error|status|warning", name):
            rank = 1
        elif path.parent == folder and name in ("metrics.json", "checkpoint.json"):
            rank = 2
        elif path.suffix == ".log":
            rank = 3 if path.parent == folder else 4
        else:
            rank = 5
        return rank, -path.stat().st_mtime_ns, str(path)

    def read(path, base=root, citation_prefix=""):
        # No following of a final symlink even if it changes after discovery.
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            prefix = stream.read(min(size, 256000))
            if b"\x00" in prefix:
                return None
            if size <= 10000:
                text = prefix.decode("utf8", "replace")
                excerpt = "complete"
            else:
                stream.seek(max(0, size - 4500))
                tail = stream.read(4500).decode("utf8", "replace")
                scanned = prefix.decode("utf8", "replace")
                match = error_line.search(scanned)
                first_error = ""
                if match:
                    start = max(0, match.start() - 300)
                    first_error = scanned[start:start + 2000]
                text = ("[HEAD]\n" + scanned[:3000]
                        + ("\n[FIRST ERROR IN FIRST 256000 BYTES]\n" + first_error if first_error else "")
                        + "\n[TAIL]\n" + tail)
                excerpt = "head/first-error/tail" if first_error else "head/tail"
        return {"path": citation_prefix + path.relative_to(base).as_posix(), "size": size,
                "truncated": size > 10000, "excerpt": excerpt, "text": text}

    groups = [(root, "", [root / "run.json"])]
    # Compact evaluation deliberately removes run/. Only the exact evaluator
    # parent may supplement it, bound by the trusted envelope's output_dir.
    # Never follow arbitrary paths in artifacts or scan sibling evaluations.
    parent = root.parent
    envelope_path = parent / "evaluation_result.json"
    compact_or_failed = False
    if root.name == "run" and safe(envelope_path, parent):
        with envelope_path.open("rb") as stream:
            raw = stream.read(1000001)
        if len(raw) > 1000000:
            raise RuntimeError("evaluation envelope exceeds evidence read limit")
        envelope = json.loads(raw)
        artifacts = envelope.get("artifacts") if isinstance(envelope, dict) else None
        metrics = envelope.get("metrics") if isinstance(envelope, dict) else None
        if (not isinstance(artifacts, dict) or not isinstance(metrics, dict)
                or not isinstance(artifacts.get("output_dir"), str)
                or pathlib.Path(artifacts["output_dir"]).resolve() != parent):
            raise RuntimeError("evaluation envelope does not identify the requested run")
        for key, expected in (("run_dir", root), ("evaluation_result", envelope_path)):
            if key in artifacts and (not isinstance(artifacts[key], str)
                                     or pathlib.Path(artifacts[key]).resolve() != expected):
                raise RuntimeError("evaluation envelope " + key + " mismatch")
        retention = artifacts.get("artifact_retention", {})
        compact_or_failed = (
            isinstance(retention, dict) and retention.get("mode") == "compact"
            and "run" in retention.get("removed_reproducible_directories", [])
        ) or (metrics.get("runs_successfully") == 0 and bool(artifacts.get("error_type")))
        retained_names = (
            "placement_metrics.json", "two_pass_opt_metrics.json", "two_pass_opt.log",
            "thermal_metrics.json", "hotspot.log", "cts_metrics.json",
            "cts_thermal_metrics.json", "cts_thermal_manifest.json", "cts_hotspot.log",
            "run.json", "placement_params.json", "placement_stage_manifest.json",
            "placement_run_tail.log",
        )
        paths = [parent / name for name in (
            "evaluation_result.json", "stage.log", "plan.log", "thermal.log", "openroad-build.log")]
        paths.extend(parent / "retained_artifacts" / name for name in retained_names)
        groups.append((parent, "evaluator/", paths))
    if not root.is_dir() and not compact_or_failed:
        raise RuntimeError("evidence run directory does not exist and has no qualified compact/failure envelope")
    for stage in stages:
        folder = root / stage
        if not folder.is_dir() or folder.is_symlink():
            continue
        paths, discovered = [], 0
        def walk_error(error):
            errors.append(stage + ": " + type(error).__name__ + " during discovery")

        for base, directories, names in os.walk(folder, followlinks=False, onerror=walk_error):
            directories[:] = sorted(name for name in directories
                                    if not name.startswith(".") and not private.search(name)
                                    and not (pathlib.Path(base) / name).is_symlink())
            for name in sorted(names):
                discovered += 1
                if discovered > 4000:
                    break
                path = pathlib.Path(base) / name
                if path.suffix.lower() in (".log", ".json", ".tsv", ".rpt", ".csv", ".txt") and safe(path):
                    paths.append(path)
            if discovered > 4000:
                warnings.append(stage + ": discovery capped at 4000 directory entries")
                break
        try:
            paths.sort(key=lambda path: priority(path, folder))
        except OSError as error:
            errors.append(stage + ": " + type(error).__name__ + " during discovery")
        if len(paths) > 12:
            warnings.append(stage + ": selected 12 of " + str(len(paths)) + " text artifacts")
        groups.append((root, "", paths[:12]))
    for base, citation_prefix, paths in groups:
        for path in paths:
            if not safe(path, base):
                continue
            try:
                item = read(path, base, citation_prefix)
                if item is not None:
                    items.append(item)
            except OSError as error:
                errors.append(citation_prefix + path.relative_to(base).as_posix() + ": " + type(error).__name__)
    if not items:
        errors.append("no readable run evidence found")
    return {"files": items, "errors": errors, "warnings": warnings}


def _redact(text: str) -> str:
    # Artifact names are filtered before reading; redact accidentally logged
    # credentials too, including quoted JSON values and Bearer authorization.
    text = re.sub(r"(?im)(authorization[\"']?\s*[:=]\s*)[^\r\n]+",
                  r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|password|(?:access[_-]?|refresh[_-]?)?token)"
                  r"[\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
                  r"\1[REDACTED]", text)
    return text


def _check_destination(path: Path) -> None:
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("evidence destination must not contain symlinks or parent traversal")


def _validate_snapshot(snapshot: object) -> None:
    """Reject malformed host output before publishing any local artifacts."""
    if not isinstance(snapshot, dict) or set(snapshot) != {"files", "errors", "warnings"}:
        raise ValueError("invalid evidence snapshot object")
    files = snapshot["files"]
    if not isinstance(files, list) or len(files) > 199:
        raise ValueError("invalid evidence snapshot file list")
    for key in ("errors", "warnings"):
        if not isinstance(snapshot[key], list) or any(not isinstance(item, str) for item in snapshot[key]):
            raise ValueError("invalid evidence snapshot " + key)
    if not files and not snapshot["errors"]:
        raise ValueError("no readable run evidence found")
    seen = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "truncated", "excerpt", "text"}:
            raise ValueError("invalid evidence artifact record")
        name = item["path"]
        if not isinstance(name, str) or not name:
            raise ValueError("invalid evidence artifact path")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name or name in seen:
            raise ValueError("unsafe or duplicate evidence artifact path")
        seen.add(name)
        if (type(item["size"]) is not int or item["size"] < 0
                or type(item["truncated"]) is not bool or not isinstance(item["excerpt"], str)
                or not isinstance(item["text"], str) or len(item["text"]) > 10000):
            raise ValueError("invalid evidence artifact metadata or text")


def collect_evidence(driver, remote_run: str, destination: Path) -> dict:
    """Copy bounded diagnostic text, preserving the existing citation index API.

    A failed/empty read is explicit in ``error``. Selection limits are only
    ``warnings``: the raw run remains authoritative, and no metrics are inferred
    from omitted artifacts. The caller decides whether missing evidence blocks
    further agent work.
    """
    script = ("import json, sys\n" + inspect.getsource(_read_run_evidence)
              + "\nprint(json.dumps(_read_run_evidence(sys.argv[1])))\n")
    try:
        _check_destination(destination)
        result = driver._remote("timeout", "25", "python3", "-c",
                                shlex.quote(script), shlex.quote(remote_run))
        if result.returncode:
            raise RuntimeError("bounded evidence read exit " + str(result.returncode))
        snapshot = json.loads(result.stdout)
        _validate_snapshot(snapshot)
        index = []
        for i, original in enumerate(snapshot["files"]):
            item = dict(original)
            text = _redact(item.pop("text"))
            path = destination / ("{:02d}.txt".format(i))
            _check_destination(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            index.append({**item, "local_file": str(path), "sha256": digest(text)})
        report = {"remote_run": remote_run, "files": index,
                  "error": "; ".join(snapshot["errors"]) or None,
                  "warnings": snapshot["warnings"]}
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        report = {"remote_run": remote_run, "files": [], "error": str(error)}
    try:
        _check_destination(destination / "index.json")
        _check_destination(destination / "index.json.tmp")
        write_json(destination / "index.json", report)
    except (ValueError, OSError) as error:
        # An unsafe destination cannot receive even a diagnostic index.
        report = {"remote_run": remote_run, "files": [], "error": str(error)}
    return report
