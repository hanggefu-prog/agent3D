"""Auditable final-state submission bundles for global workspace trials."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .model import GlobalTaskSpec
from .workspace import TrialLayout, trial_storage_bytes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_relative(path: str) -> str:
    value = PurePosixPath(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError("unsafe submitted path: {}".format(path))
    return value.as_posix()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("artifact is outside the trial directory: {}".format(path))
    return resolved


def _git(repository: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("git {} failed: {}".format(" ".join(arguments), result.stderr.strip()))
    return result.stdout


def changed_workspace_files(workspace: Path, base_revision: str) -> Tuple[Tuple[str, str], ...]:
    """Return status/path pairs, including untracked files, without staging them."""

    status = _git(
        workspace,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )
    rows = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        code = line[:2].strip() or "M"
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        path = _safe_relative(raw_path.strip('"'))
        rows.append((code, path))
    # A caller may compare against a base other than HEAD. Include those paths
    # even if the worktree status itself is clean.
    baseline_names = _git(
        workspace,
        ["diff", "--name-only", "--no-renames", base_revision, "--"],
    )
    present = {path for _, path in rows}
    for raw_path in baseline_names.splitlines():
        path = _safe_relative(raw_path)
        if path not in present:
            rows.append(("M", path))
            present.add(path)
    return tuple(rows)


@dataclass(frozen=True)
class ChangeRecord:
    status: str
    path: str
    sha256: Optional[str]
    submitted_path: Optional[str]


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    source: str
    submitted_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SubmissionManifest:
    task_id: str
    task_digest: str
    trial_id: str
    base_revision: str
    patch_path: str
    patch_sha256: str
    changes: Tuple[ChangeRecord, ...]
    artifacts: Tuple[ArtifactRecord, ...]
    created_at: str

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["schema_version"] = 1
        return value

    @classmethod
    def from_path(cls, path: Path) -> "SubmissionManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("schema_version", 0)) != 1:
            raise ValueError("unsupported submission schema")
        return cls(
            task_id=str(data["task_id"]),
            task_digest=str(data["task_digest"]),
            trial_id=str(data["trial_id"]),
            base_revision=str(data["base_revision"]),
            patch_path=str(data["patch_path"]),
            patch_sha256=str(data["patch_sha256"]),
            changes=tuple(ChangeRecord(**item) for item in data["changes"]),
            artifacts=tuple(ArtifactRecord(**item) for item in data["artifacts"]),
            created_at=str(data["created_at"]),
        )


def collect_submission(
    task: GlobalTaskSpec,
    layout: TrialLayout,
    artifacts: Mapping[str, Path],
) -> SubmissionManifest:
    """Freeze source changes and declared artifacts without trusting agent metrics."""

    if layout.task_digest != task.digest():
        raise ValueError("trial was prepared for a different global task definition")
    storage_limit = int(task.environment.storage_gb * 1024 ** 3)
    if trial_storage_bytes(layout.root) > storage_limit:
        raise ValueError("trial exceeds its configured storage budget")
    missing = set(task.submission.required_artifacts) - set(artifacts)
    extra = set(artifacts) - set(task.submission.required_artifacts)
    if missing or extra:
        raise ValueError(
            "submission artifacts have missing={} extra={}".format(sorted(missing), sorted(extra))
        )
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) for name in artifacts):
        raise ValueError("artifact names must be safe identifiers")

    submission = layout.submission
    if any(submission.iterdir()):
        raise FileExistsError("submission directory is not empty: {}".format(submission))
    source_root = submission / "source_changes"
    artifact_root = submission / "artifacts"
    source_root.mkdir()
    artifact_root.mkdir()

    change_records = []
    changes = changed_workspace_files(layout.workspace, layout.base_revision)
    for status, relative in changes:
        protected = any(
            relative == prefix or relative.startswith(prefix.rstrip("/") + "/")
            for prefix in task.workspace.protected_paths
        )
        if protected:
            raise ValueError("candidate modified trusted control-plane path: {}".format(relative))
        source = layout.workspace / relative
        if not source.exists():
            change_records.append(ChangeRecord(status, relative, None, None))
            continue
        if not source.is_file() or source.is_symlink():
            raise ValueError("candidate changes must be regular files: {}".format(relative))
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        change_records.append(
            ChangeRecord(
                status=status,
                path=relative,
                sha256=_sha256(target),
                submitted_path=target.relative_to(submission).as_posix(),
            )
        )

    patch_path = submission / "patch.diff"
    patch_path.write_text(
        _git(layout.workspace, ["diff", "--binary", layout.base_revision, "--"]),
        encoding="utf-8",
    )

    artifact_records = []
    for name, candidate_path in sorted(artifacts.items()):
        source = _inside(candidate_path, layout.root)
        if not source.is_file() or source.is_symlink():
            raise ValueError("submitted artifact must be a regular file: {}".format(source))
        target = artifact_root / name / source.name
        target.parent.mkdir()
        shutil.copy2(source, target)
        artifact_records.append(
            ArtifactRecord(
                name=name,
                source=str(source),
                submitted_path=target.relative_to(submission).as_posix(),
                size=target.stat().st_size,
                sha256=_sha256(target),
            )
        )

    manifest = SubmissionManifest(
        task_id=task.task_id,
        task_digest=task.digest(),
        trial_id=layout.trial_id,
        base_revision=layout.base_revision,
        patch_path=patch_path.relative_to(submission).as_posix(),
        patch_sha256=_sha256(patch_path),
        changes=tuple(change_records),
        artifacts=tuple(artifact_records),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (submission / "submission.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if trial_storage_bytes(layout.root) > storage_limit:
        raise ValueError("submission copy exceeds the configured storage budget")
    return manifest
