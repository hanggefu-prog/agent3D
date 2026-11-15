"""One path interpretation for candidate scope, wrapper checks and Git apply.

Candidate patches use ordinary ``a/`` and ``b/`` unified-diff headers.  Reject
aliases instead of silently cleaning them: Git and independent safety checks
must identify exactly the same file.  Hunk bodies are consumed by line counts,
so source text resembling a file header is never interpreted as another path.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import List, Tuple


_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_GIT = re.compile(r"^diff --git a/(.+) b/(.+)\r?\n?$")


def canonical_path(raw: str) -> str:
    """Require an unambiguous repository-relative POSIX path."""
    normalized = PurePosixPath(raw).as_posix()
    if (not raw or raw != raw.strip() or raw != normalized
            or raw.startswith("/") or ".." in raw.split("/")
            or "\\" in raw or any(ord(char) < 32 for char in raw)
            or raw.startswith('"') or normalized == "."):
        raise ValueError("patch contains unsafe or non-canonical path: {!r}".format(raw))
    return raw


def _header_path(line: str, marker: str, prefix: str) -> str:
    raw = line[len(marker):].split("\t", 1)[0].rstrip("\r\n")
    if raw == "/dev/null":
        raise ValueError("file creation and deletion are not allowed")
    if not raw.startswith(prefix):
        raise ValueError("patch headers must use --- a/ and +++ b/ prefixes")
    return canonical_path(raw[len(prefix):])


def split_patch_sections(patch_text: str) -> List[Tuple[str, str]]:
    """Return canonical paths and their exact, fully validated diff sections."""
    lines = patch_text.splitlines(keepends=True)
    sections: List[Tuple[str, str]] = []
    pending: List[str] = []
    current = None
    path = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("--- "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                raise ValueError("patch file header is malformed")
            if current is not None:
                sections.append((path, "".join(current)))
            before = _header_path(line, "--- ", "a/")
            path = _header_path(lines[index + 1], "+++ ", "b/")
            if before != path:
                raise ValueError("patch renames a file, which is not allowed")
            git_headers = [item for item in pending if item.startswith("diff --git ")]
            if len(git_headers) > 1:
                raise ValueError("patch contains multiple diff --git headers for one section")
            if git_headers:
                match = _GIT.fullmatch(git_headers[0])
                if not match:
                    raise ValueError("patch diff --git header is malformed")
                if canonical_path(match.group(1)) != path or canonical_path(match.group(2)) != path:
                    raise ValueError("diff --git paths do not match unified diff headers")
            current = pending + [line, lines[index + 1]]
            pending = []
            index += 2
            continue
        hunk = _HUNK.match(line)
        if hunk and current is not None:
            old_count = int(hunk.group(1)) if hunk.group(1) is not None else 1
            new_count = int(hunk.group(2)) if hunk.group(2) is not None else 1
            current.append(line)
            index += 1
            while (old_count > 0 or new_count > 0) and index < len(lines):
                body = lines[index]
                if body.startswith("\\"):
                    current.append(body)
                    index += 1
                    continue
                if body.startswith("-"):
                    old_count -= 1
                elif body.startswith("+"):
                    new_count -= 1
                elif body.startswith(" ") or body in {"\n", "\r\n"}:
                    old_count -= 1
                    new_count -= 1
                else:
                    raise ValueError("patch hunk body is malformed")
                current.append(body)
                index += 1
            if old_count != 0 or new_count != 0:
                raise ValueError("patch hunk line counts do not match its header")
            while index < len(lines) and lines[index].startswith("\\"):
                current.append(lines[index])
                index += 1
            continue
        if current is not None and not line.startswith(("diff --git ", "index ")):
            current.append(line)
        else:
            if current is not None:
                sections.append((path, "".join(current)))
                current = None
            pending.append(line)
        index += 1
    if current is not None:
        sections.append((path, "".join(current)))
    if any(item.strip() for item in pending):
        raise ValueError("patch has content without a unified diff file header")
    return sections


def strip_section_prefix(section: str, prefix: str) -> str:
    """Rewrite only actual file headers, preserving header-like hunk contents."""
    lines = section.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("diff --git "):
            match = _GIT.fullmatch(line)
            if not match:
                raise ValueError("patch diff --git header is malformed")
            lines[index] = "diff --git a/{} b/{}\n".format(
                match.group(1)[len(prefix):], match.group(2)[len(prefix):])
        if line.startswith("--- "):
            lines[index] = line.replace("--- a/" + prefix, "--- a/", 1)
            lines[index + 1] = lines[index + 1].replace("+++ b/" + prefix, "+++ b/", 1)
            break
    return "".join(lines)
