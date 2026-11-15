"""Bounded native error excerpts for failed cells; never scoring inputs."""
from pathlib import Path
import re


_ERROR = re.compile(r"\b(?:\w*Error|\w*Exception):|\bERROR\b|Aborted|terminate called")
_LOG_REFERENCE = re.compile(r"\bsee\s+(.+?\.log)(?=$|[\s,;])")


def _error_record(output, path):
    try:
        path.resolve().relative_to(output)
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 32768))
            lines = stream.read(32768).decode("utf-8", errors="replace").splitlines()
    except (OSError, ValueError):
        return None
    indices = [i for i, line in enumerate(lines) if _ERROR.search(line)]
    if not indices:
        return None
    index = indices[-1]
    excerpt = "\n".join(lines[max(0, index - 3):index + 2])[-1800:]
    return {"source": str(path.relative_to(output)),
            "excerpt": excerpt.replace(str(output), "$CELL")}


def _referenced_log(output, reference):
    path = Path(reference.replace("$CELL", str(output)))
    if not path.is_absolute():
        path = output / path
    try:
        path.resolve().relative_to(output)
        relative = path.relative_to(output / "run")
    except ValueError:
        return None
    retained = output / "retained_artifacts/algorithm" / relative
    return path if path.is_file() else retained


def native_failure_diagnostics(output):
    output = Path(output).resolve()
    records = []
    # Follow a wrapper's failed native command before reading older, potentially
    # nonfatal ERROR lines from a successful upstream stage.
    for name in ("backend-stage.log", "stage.log"):
        wrapper = _error_record(output, output / name)
        if wrapper is None:
            continue
        for reference in reversed(_LOG_REFERENCE.findall(wrapper["excerpt"])):
            path = _referenced_log(output, reference)
            if path is not None:
                cause = _error_record(output, path)
                # Missing native evidence must not attribute the failure to an
                # unrelated earlier stage warning.
                return ([cause] if cause else []) + [wrapper]
        records.append(wrapper)
        if len(records) == 2:
            return records
    for directory in (output / "retained_artifacts/algorithm", output / "run"):
        for path in sorted(directory.rglob("*.log"), reverse=True)[:64]:
            record = _error_record(output, path)
            if record:
                records.append(record)
                if len(records) == 2:
                    return records
    return records
