"""Deterministic source-change mediation for stage Task Agents.

LLMs may propose and implement changes, but this module alone decides whether
those changes can be checkpointed or integrated.  Every candidate lives in a
detached Git worktree; the canonical repository is modified only after a
second, full-range audit succeeds.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import require_stage


SCOPED_STAGES = ("partition", "layout", "routing")
CONTROL_ROOT = ".agentic3d"


def _git(repository: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result


def _normalize_path(value: str) -> str:
    value = value.replace("\\", "/").strip()
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe repository-relative path: {value!r}")
    normalized = path.as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized == ".":
        raise ValueError(f"unsafe repository-relative path: {value!r}")
    return normalized


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _is_control_path(path: str) -> bool:
    return _inside(path, CONTROL_ROOT)


@dataclass(frozen=True)
class SourceScope:
    allow: Tuple[str, ...]
    allow_create: bool = False
    allow_delete: bool = False
    allow_rename: bool = False
    allow_mode_change: bool = False
    allow_symlink: bool = False
    allowed_extensions: Tuple[str, ...] = ()
    max_changed_files: int = 12
    max_added_lines: int = 800
    max_deleted_lines: int = 800

    def __post_init__(self) -> None:
        if not self.allow:
            raise ValueError("source scope must allow at least one path")
        if min(self.max_changed_files, self.max_added_lines, self.max_deleted_lines) < 0:
            raise ValueError("source scope budgets must be non-negative")

    def permits(self, path: str) -> bool:
        normalized = _normalize_path(path)
        if not any(_inside(normalized, root) for root in self.allow):
            return False
        if self.allowed_extensions:
            suffix = PurePosixPath(normalized).suffix.lower()
            if suffix not in self.allowed_extensions:
                return False
        return True


@dataclass(frozen=True)
class StageScope:
    stage: str
    baseline_commit: str
    source: SourceScope
    adapter_paths: Tuple[str, ...]
    schema_version: int = 1
    digest: str = ""

    def __post_init__(self) -> None:
        require_stage(self.stage)
        if self.stage not in SCOPED_STAGES:
            raise ValueError(f"source mediation is not enabled for stage {self.stage!r}")
        if not self.baseline_commit:
            raise ValueError("scope requires a baseline commit")

    def unsigned_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value.pop("digest", None)
        value["source"]["allow"] = list(self.source.allow)
        value["source"]["allowed_extensions"] = list(self.source.allowed_extensions)
        value["adapter_paths"] = list(self.adapter_paths)
        return value

    def computed_digest(self) -> str:
        encoded = json.dumps(
            self.unsigned_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        value = self.unsigned_dict()
        value["digest"] = self.digest or self.computed_digest()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageScope":
        source_value = dict(value["source"])
        source = SourceScope(
            allow=tuple(source_value["allow"]),
            allow_create=bool(source_value.get("allow_create", False)),
            allow_delete=bool(source_value.get("allow_delete", False)),
            allow_rename=bool(source_value.get("allow_rename", False)),
            allow_mode_change=bool(source_value.get("allow_mode_change", False)),
            allow_symlink=bool(source_value.get("allow_symlink", False)),
            allowed_extensions=tuple(source_value.get("allowed_extensions", ())),
            max_changed_files=int(source_value.get("max_changed_files", 12)),
            max_added_lines=int(source_value.get("max_added_lines", 800)),
            max_deleted_lines=int(source_value.get("max_deleted_lines", 800)),
        )
        scope = cls(
            stage=str(value["stage"]),
            baseline_commit=str(value["baseline_commit"]),
            source=source,
            adapter_paths=tuple(value["adapter_paths"]),
            schema_version=int(value.get("schema_version", 1)),
            digest=str(value.get("digest", "")),
        )
        if scope.digest and scope.digest != scope.computed_digest():
            raise ValueError("scope digest does not match its contents")
        return scope

    @classmethod
    def read(cls, path: Path) -> "StageScope":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("scope file must contain a JSON object")
        return cls.from_dict(loaded)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


class ScopeCompiler:
    """Compile an LLM proposal under immutable adapter stage boundaries."""

    @staticmethod
    def compile(
        *,
        stage: str,
        baseline_commit: str,
        adapter_profile: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> StageScope:
        require_stage(stage)
        if stage not in SCOPED_STAGES:
            raise ValueError(f"source mediation is not enabled for stage {stage!r}")
        adapter_paths = tuple(
            dict.fromkeys(
                _normalize_path(str(item))
                for key in ("source_paths", "config_paths")
                for item in adapter_profile.get(key, [])
            )
        )
        if not adapter_paths:
            raise ValueError(f"adapter has no source/config paths for {stage}")
        proposed = tuple(
            dict.fromkeys(
                _normalize_path(str(item))
                for key in ("source_paths", "config_paths")
                for item in proposal.get(key, [])
            )
        )
        if not proposed:
            raise ValueError("Local Analyzer scope proposal cannot be empty")
        for path in proposed:
            if not any(_inside(path, boundary) for boundary in adapter_paths):
                raise ValueError(
                    f"Local Analyzer proposed out-of-stage path {path!r} for {stage}"
                )
        extensions = tuple(
            sorted(
                {
                    str(item).lower()
                    if str(item).startswith(".")
                    else "." + str(item).lower()
                    for item in proposal.get("allowed_extensions", [])
                }
            )
        )
        source = SourceScope(
            allow=proposed,
            allow_create=bool(proposal.get("allow_create", False)),
            allow_delete=bool(proposal.get("allow_delete", False)),
            allow_rename=bool(proposal.get("allow_rename", False)),
            allow_mode_change=bool(proposal.get("allow_mode_change", False)),
            allow_symlink=bool(proposal.get("allow_symlink", False)),
            allowed_extensions=extensions,
            max_changed_files=min(int(proposal.get("max_changed_files", 12)), 50),
            max_added_lines=min(int(proposal.get("max_added_lines", 800)), 5000),
            max_deleted_lines=min(int(proposal.get("max_deleted_lines", 800)), 5000),
        )
        return StageScope(
            stage=stage,
            baseline_commit=baseline_commit,
            source=source,
            adapter_paths=adapter_paths,
        )


@dataclass(frozen=True)
class GitChange:
    status: str
    path: str
    old_path: Optional[str] = None
    added_lines: int = 0
    deleted_lines: int = 0

    def paths(self) -> Tuple[str, ...]:
        return tuple(item for item in (self.old_path, self.path) if item)


@dataclass
class ScopeReport:
    passed: bool
    stage: str
    baseline_commit: str
    checked_head: str
    scope_digest: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[Dict[str, str]] = field(default_factory=list)
    declared_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


class WorktreeManager:
    def __init__(self, repository: Path) -> None:
        self.repository = repository.expanduser().resolve()
        _git(self.repository, "rev-parse", "--show-toplevel")

    def head(self) -> str:
        return _git(self.repository, "rev-parse", "HEAD").stdout.strip()

    def create(self, path: Path, *, baseline: Optional[str] = None) -> Path:
        path = path.expanduser().resolve()
        if path.exists():
            raise ValueError(f"worktree path already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(self.repository, "worktree", "add", "--detach", str(path), baseline or self.head())
        return path


class ScopeGuard:
    """Audit and checkpoint only changes allowed by a signed StageScope."""

    def __init__(self, worktree: Path, scope: StageScope) -> None:
        self.worktree = worktree.expanduser().resolve()
        self.scope = scope
        _git(self.worktree, "rev-parse", "--is-inside-work-tree")

    def head(self) -> str:
        return _git(self.worktree, "rev-parse", "HEAD").stdout.strip()

    def _status_changes(self) -> List[GitChange]:
        raw = subprocess.run(
            [
                "git", "-C", str(self.worktree), "status", "--porcelain=v1",
                "-z", "--untracked-files=all",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if raw.returncode != 0:
            raise RuntimeError(raw.stderr.decode("utf-8", "replace").strip())
        tokens = raw.stdout.split(b"\0")
        changes: List[GitChange] = []
        index = 0
        while index < len(tokens) and tokens[index]:
            record = tokens[index].decode("utf-8", "surrogateescape")
            index += 1
            status = record[:2]
            path = _normalize_path(record[3:])
            old_path: Optional[str] = None
            if "R" in status or "C" in status:
                if index >= len(tokens) or not tokens[index]:
                    raise RuntimeError("malformed Git rename status")
                old_path = _normalize_path(
                    tokens[index].decode("utf-8", "surrogateescape")
                )
                index += 1
            if _is_control_path(path) and (old_path is None or _is_control_path(old_path)):
                continue
            changes.append(GitChange(status=status, path=path, old_path=old_path))
        return changes

    def _line_counts(self) -> Dict[str, Tuple[int, int]]:
        result = _git(self.worktree, "diff", "--numstat", "HEAD", "--")
        counts: Dict[str, Tuple[int, int]] = {}
        for line in result.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            added = int(parts[0]) if parts[0].isdigit() else 0
            deleted = int(parts[1]) if parts[1].isdigit() else 0
            path = parts[2]
            if " => " in path:
                # The path check is authoritative; line budgets remain global.
                path = path.split(" => ")[-1].strip("{}")
            try:
                counts[_normalize_path(path)] = (added, deleted)
            except ValueError:
                continue
        return counts

    def audit(
        self,
        *,
        expected_head: str,
        declared_files: Sequence[str] = (),
    ) -> ScopeReport:
        current_head = self.head()
        violations: List[Dict[str, str]] = []
        if current_head != expected_head:
            violations.append(
                {
                    "type": "agent_changed_git_history",
                    "path": ".git",
                    "reason": f"expected HEAD {expected_head}, found {current_head}",
                }
            )
        try:
            declared = [_normalize_path(item) for item in declared_files]
        except ValueError as exc:
            declared = []
            violations.append(
                {"type": "unsafe_declared_path", "path": "", "reason": str(exc)}
            )
        for path in declared:
            if not self.scope.source.permits(path):
                violations.append(
                    {
                        "type": "declared_path_out_of_scope",
                        "path": path,
                        "reason": "plan declared a path outside the compiled scope",
                    }
                )
        line_counts = self._line_counts()
        changes = self._status_changes()
        total_added = 0
        total_deleted = 0
        for change in changes:
            added, deleted = line_counts.get(change.path, (0, 0))
            if change.status == "??":
                candidate = self.worktree / change.path
                if candidate.is_file() and not candidate.is_symlink():
                    try:
                        text = candidate.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        text = ""
                    added = len(text.splitlines())
            object.__setattr__(change, "added_lines", added)
            object.__setattr__(change, "deleted_lines", deleted)
            total_added += added
            total_deleted += deleted
            for path in change.paths():
                if not self.scope.source.permits(path):
                    violations.append(
                        {
                            "type": "path_out_of_scope",
                            "path": path,
                            "reason": "actual Git change is outside the compiled scope",
                        }
                    )
            code = change.status.replace(" ", "")
            if "?" in code and not self.scope.source.allow_create:
                violations.append(
                    {"type": "create_forbidden", "path": change.path, "reason": "new files are not allowed"}
                )
            if "A" in code and not self.scope.source.allow_create:
                violations.append(
                    {"type": "create_forbidden", "path": change.path, "reason": "new files are not allowed"}
                )
            if "D" in code and not self.scope.source.allow_delete:
                violations.append(
                    {"type": "delete_forbidden", "path": change.path, "reason": "deletions are not allowed"}
                )
            if ("R" in code or "C" in code) and not self.scope.source.allow_rename:
                violations.append(
                    {"type": "rename_forbidden", "path": change.path, "reason": "renames/copies are not allowed"}
                )
            candidate = self.worktree / change.path
            if candidate.is_symlink() and not self.scope.source.allow_symlink:
                violations.append(
                    {"type": "symlink_forbidden", "path": change.path, "reason": "symbolic links are not allowed"}
                )
        summary = _git(self.worktree, "diff", "--summary", "HEAD", "--").stdout
        if "mode change" in summary and not self.scope.source.allow_mode_change:
            violations.append(
                {"type": "mode_change_forbidden", "path": "", "reason": "file mode changes are not allowed"}
            )
        actual_paths = {path for change in changes for path in change.paths()}
        declared_set = set(declared)
        for path in sorted(actual_paths - declared_set):
            violations.append(
                {
                    "type": "undeclared_actual_change",
                    "path": path,
                    "reason": "actual change was not declared in the round plan",
                }
            )
        if len(actual_paths) > self.scope.source.max_changed_files:
            violations.append(
                {"type": "file_budget_exceeded", "path": "", "reason": f"changed {len(actual_paths)} files"}
            )
        if total_added > self.scope.source.max_added_lines:
            violations.append(
                {"type": "added_line_budget_exceeded", "path": "", "reason": f"added {total_added} lines"}
            )
        if total_deleted > self.scope.source.max_deleted_lines:
            violations.append(
                {"type": "deleted_line_budget_exceeded", "path": "", "reason": f"deleted {total_deleted} lines"}
            )
        return ScopeReport(
            passed=not violations,
            stage=self.scope.stage,
            baseline_commit=self.scope.baseline_commit,
            checked_head=current_head,
            scope_digest=self.scope.digest or self.scope.computed_digest(),
            changes=[asdict(change) for change in changes],
            violations=violations,
            declared_files=declared,
        )

    def checkpoint(self, *, message: str) -> Optional[str]:
        changes = self._status_changes()
        if not changes:
            return None
        paths = sorted({path for change in changes for path in change.paths()})
        _git(self.worktree, "add", "--all", "--", *paths)
        _git(
            self.worktree,
            "-c", "user.name=Agentic3D ScopeGuard",
            "-c", "user.email=agentic3d@localhost",
            "commit", "-m", message,
        )
        return self.head()

    def rollback(self, *, expected_head: str) -> None:
        _git(self.worktree, "reset", "--hard", expected_head)
        _git(self.worktree, "clean", "-fd", "-e", f"{CONTROL_ROOT}/", "--", ".")

    def audit_candidate(self, *, candidate_head: str) -> ScopeReport:
        if self.head() != candidate_head:
            raise ValueError("candidate_head is not the current worktree HEAD")
        raw = subprocess.run(
            [
                "git", "-C", str(self.worktree), "diff", "--name-status", "-z",
                self.scope.baseline_commit, candidate_head, "--",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if raw.returncode != 0:
            raise RuntimeError(raw.stderr.decode("utf-8", "replace").strip())
        tokens = raw.stdout.split(b"\0")
        changes: List[GitChange] = []
        index = 0
        while index < len(tokens) and tokens[index]:
            status = tokens[index].decode("ascii", "replace")
            index += 1
            old_path: Optional[str] = None
            if status.startswith(("R", "C")):
                old_path = _normalize_path(
                    tokens[index].decode("utf-8", "surrogateescape")
                )
                index += 1
            path = _normalize_path(tokens[index].decode("utf-8", "surrogateescape"))
            index += 1
            changes.append(GitChange(status=status, path=path, old_path=old_path))

        counts: Dict[str, Tuple[int, int]] = {}
        numstat = _git(
            self.worktree, "diff", "--numstat", self.scope.baseline_commit,
            candidate_head, "--",
        )
        total_added = 0
        total_deleted = 0
        for line in numstat.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            added = int(parts[0]) if parts[0].isdigit() else 0
            deleted = int(parts[1]) if parts[1].isdigit() else 0
            total_added += added
            total_deleted += deleted
            path = parts[2]
            if " => " in path:
                path = path.split(" => ")[-1].strip("{}")
            try:
                counts[_normalize_path(path)] = (added, deleted)
            except ValueError:
                pass

        violations: List[Dict[str, str]] = []
        for change in changes:
            added, deleted = counts.get(change.path, (0, 0))
            object.__setattr__(change, "added_lines", added)
            object.__setattr__(change, "deleted_lines", deleted)
            for path in change.paths():
                if not self.scope.source.permits(path):
                    violations.append(
                        {
                            "type": "path_out_of_scope",
                            "path": path,
                            "reason": "candidate range contains an out-of-scope path",
                        }
                    )
            if change.status.startswith("A") and not self.scope.source.allow_create:
                violations.append(
                    {"type": "create_forbidden", "path": change.path, "reason": "candidate creates a file"}
                )
            if change.status.startswith("D") and not self.scope.source.allow_delete:
                violations.append(
                    {"type": "delete_forbidden", "path": change.path, "reason": "candidate deletes a file"}
                )
            if change.status.startswith(("R", "C")) and not self.scope.source.allow_rename:
                violations.append(
                    {"type": "rename_forbidden", "path": change.path, "reason": "candidate renames/copies a file"}
                )
            mode = _git(
                self.worktree, "ls-tree", candidate_head, "--", change.path,
                check=False,
            ).stdout.split()
            if mode and mode[0] == "120000" and not self.scope.source.allow_symlink:
                violations.append(
                    {"type": "symlink_forbidden", "path": change.path, "reason": "candidate contains a symbolic link"}
                )
        summary = _git(
            self.worktree, "diff", "--summary", self.scope.baseline_commit,
            candidate_head, "--",
        ).stdout
        if "mode change" in summary and not self.scope.source.allow_mode_change:
            violations.append(
                {"type": "mode_change_forbidden", "path": "", "reason": "candidate changes a file mode"}
            )
        actual_paths = {path for change in changes for path in change.paths()}
        if len(actual_paths) > self.scope.source.max_changed_files:
            violations.append(
                {"type": "file_budget_exceeded", "path": "", "reason": f"candidate changes {len(actual_paths)} files"}
            )
        if total_added > self.scope.source.max_added_lines:
            violations.append(
                {"type": "added_line_budget_exceeded", "path": "", "reason": f"candidate adds {total_added} lines"}
            )
        if total_deleted > self.scope.source.max_deleted_lines:
            violations.append(
                {"type": "deleted_line_budget_exceeded", "path": "", "reason": f"candidate deletes {total_deleted} lines"}
            )
        declared = sorted(actual_paths)
        return ScopeReport(
            passed=not violations,
            stage=self.scope.stage,
            baseline_commit=self.scope.baseline_commit,
            checked_head=candidate_head,
            scope_digest=self.scope.digest or self.scope.computed_digest(),
            changes=[asdict(change) for change in changes],
            violations=violations,
            declared_files=declared,
        )


class IntegrationGate:
    """Integrate an audited candidate only when the canonical baseline matches."""

    def __init__(self, repository: Path) -> None:
        self.repository = repository.expanduser().resolve()

    def integrate(
        self,
        *,
        scope: StageScope,
        worktree: Path,
        candidate_head: str,
        message: str,
    ) -> str:
        canonical_head = _git(self.repository, "rev-parse", "HEAD").stdout.strip()
        if canonical_head != scope.baseline_commit:
            raise RuntimeError(
                "integration baseline moved; candidate must be rebased and revalidated"
            )
        unstaged = _git(self.repository, "diff", "--quiet", check=False)
        staged = _git(self.repository, "diff", "--cached", "--quiet", check=False)
        if unstaged.returncode != 0 or staged.returncode != 0:
            raise RuntimeError("canonical repository has tracked modifications")
        guard = ScopeGuard(worktree, scope)
        report = guard.audit_candidate(candidate_head=candidate_head)
        if not report.passed:
            raise RuntimeError("candidate failed final scope audit")
        patch = _git(
            worktree, "diff", "--binary", scope.baseline_commit, candidate_head, "--"
        ).stdout
        if not patch:
            raise RuntimeError("candidate has no source changes to integrate")
        check_result = subprocess.run(
            ["git", "-C", str(self.repository), "apply", "--check", "--index", "-"],
            input=patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check_result.returncode != 0:
            raise RuntimeError(f"candidate patch does not apply: {check_result.stderr.strip()}")
        apply_result = subprocess.run(
            ["git", "-C", str(self.repository), "apply", "--index", "-"],
            input=patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if apply_result.returncode != 0:
            raise RuntimeError(f"candidate integration failed: {apply_result.stderr.strip()}")
        try:
            _git(
                self.repository,
                "-c", "user.name=Agentic3D IntegrationGate",
                "-c", "user.email=agentic3d@localhost",
                "commit", "-m", message,
            )
        except Exception:
            _git(self.repository, "reset", "--hard", scope.baseline_commit)
            raise
        return _git(self.repository, "rev-parse", "HEAD").stdout.strip()
