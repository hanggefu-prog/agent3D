"""Parallel-safe Git worktree materialization for global benchmark trials."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .model import GlobalTaskSpec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_git(repository: Path, arguments: list) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "git {} failed: {}".format(" ".join(arguments), result.stderr.strip())
        )
    return result.stdout.strip()


def new_trial_id(task_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "{}-{}-{}".format(task_id, timestamp, uuid.uuid4().hex[:8])


def trial_storage_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def _safe_trial_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", value):
        raise ValueError("unsafe trial ID: {!r}".format(value))
    return value


@dataclass(frozen=True)
class TrialLayout:
    trial_id: str
    root: Path
    workspace: Path
    control: Path
    build: Path
    runs: Path
    logs: Path
    scratch: Path
    submission: Path
    source_repository: Path
    base_revision: str
    task_digest: str
    metadata_path: Path

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        for key, item in list(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
        return value

    @classmethod
    def from_path(cls, trial_root: Path) -> "TrialLayout":
        metadata_path = trial_root.expanduser().resolve() / "trial.json"
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(
            trial_id=str(data["trial_id"]),
            root=Path(data["root"]),
            workspace=Path(data["workspace"]),
            control=Path(data["control"]),
            build=Path(data["build"]),
            runs=Path(data["runs"]),
            logs=Path(data["logs"]),
            scratch=Path(data["scratch"]),
            submission=Path(data["submission"]),
            source_repository=Path(data["source_repository"]),
            base_revision=str(data["base_revision"]),
            task_digest=str(data["task_digest"]),
            metadata_path=metadata_path,
        )


class TrialWorkspaceManager:
    """Create disposable candidate worktrees without shared mutable outputs."""

    def __init__(self, trials_root: Path):
        self.trials_root = trials_root.expanduser().resolve()
        self.trials_root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        task: GlobalTaskSpec,
        source_repository: Path,
        *,
        revision: Optional[str] = None,
        trial_id: Optional[str] = None,
        user_command: Optional[str] = None,
    ) -> TrialLayout:
        source = source_repository.expanduser().resolve()
        if not (source / ".git").exists():
            raise ValueError("source repository is not a Git checkout: {}".format(source))
        requested_revision = revision or task.source.revision
        base_revision = _run_git(source, ["rev-parse", "--verify", requested_revision + "^{commit}"])
        identifier = _safe_trial_id(trial_id or new_trial_id(task.task_id))
        root = self.trials_root / task.task_id / identifier
        # mkdir(exist_ok=False) is the cross-process reservation for this ID.
        root.mkdir(parents=True, exist_ok=False)
        workspace = root / "workspace"
        control = root / "control"
        paths = {
            "build": root / "build",
            "runs": root / "runs",
            "logs": root / "logs",
            "scratch": root / "scratch",
            "submission": root / "submission",
        }
        for path in (control, *paths.values()):
            path.mkdir()
        (paths["scratch"] / "home").mkdir()
        (paths["scratch"] / "cache").mkdir()

        try:
            _run_git(source, ["worktree", "add", "--detach", str(workspace), base_revision])
        except Exception:
            # The reserved directory contains no user data yet.
            shutil.rmtree(root)
            raise

        instruction_copy = control / "instruction.md"
        instruction_copy.write_text(task.instruction + "\n", encoding="utf-8")
        user_command_copy = control / "user-command.md"
        user_command_copy.write_text(
            (user_command.strip() if user_command and user_command.strip() else task.instruction) + "\n",
            encoding="utf-8",
        )
        task_copy = control / "task.json"
        task_copy.write_text(
            json.dumps(task.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(instruction_copy, 0o444)
        os.chmod(user_command_copy, 0o444)
        os.chmod(task_copy, 0o444)
        metadata_path = root / "trial.json"
        layout = TrialLayout(
            trial_id=identifier,
            root=root,
            workspace=workspace,
            control=control,
            build=paths["build"],
            runs=paths["runs"],
            logs=paths["logs"],
            scratch=paths["scratch"],
            submission=paths["submission"],
            source_repository=source,
            base_revision=base_revision,
            task_digest=task.digest(),
            metadata_path=metadata_path,
        )
        value = layout.to_dict()
        value.update({"schema_version": 1, "created_at": _now(), "status": "prepared"})
        _write_json(metadata_path, value)
        return layout

    def remove(self, layout: TrialLayout) -> None:
        """Explicitly remove one disposable trial and unregister its worktree."""

        if layout.root.parent.parent != self.trials_root:
            raise ValueError("trial is outside this manager's root")
        _run_git(
            layout.source_repository,
            ["worktree", "remove", "--force", str(layout.workspace)],
        )
        # Retain evidence by default. Callers may remove the now workspace-free
        # trial directory separately under their explicit retention policy.
