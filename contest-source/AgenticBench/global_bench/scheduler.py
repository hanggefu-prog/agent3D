"""Small filesystem-backed resource leases for parallel global trials."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .model import GlobalTaskSpec
from .workspace import TrialLayout, TrialWorkspaceManager, new_trial_id


@dataclass(frozen=True)
class ResourceLease:
    pool_root: Path
    resource_id: str
    trial_id: str
    token: str
    path: Path


class ResourceUnavailable(RuntimeError):
    pass


class FileResourcePool:
    """Reserve named GPUs/licenses with atomic directory creation."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def acquire(self, resource_ids: Iterable[str], trial_id: str) -> ResourceLease:
        for resource_id in resource_ids:
            if not resource_id or "/" in resource_id or resource_id in {".", ".."}:
                raise ValueError("invalid resource ID: {!r}".format(resource_id))
            path = self.root / (resource_id + ".lease")
            try:
                path.mkdir()
            except FileExistsError:
                continue
            token = uuid.uuid4().hex
            receipt = {
                "schema_version": 1,
                "resource_id": resource_id,
                "trial_id": trial_id,
                "token": token,
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
            (path / "lease.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            return ResourceLease(self.root, resource_id, trial_id, token, path)
        raise ResourceUnavailable("no requested resource is currently available")

    def release(self, lease: ResourceLease) -> None:
        if lease.pool_root != self.root or lease.path.parent != self.root:
            raise ValueError("lease does not belong to this pool")
        receipt_path = lease.path / "lease.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("token") != lease.token or receipt.get("trial_id") != lease.trial_id:
            raise RuntimeError("resource lease ownership changed")
        receipt_path.unlink()
        lease.path.rmdir()

    def active(self) -> Tuple[str, ...]:
        return tuple(sorted(path.name[:-6] for path in self.root.glob("*.lease")))


@dataclass(frozen=True)
class ScheduledTrial:
    layout: TrialLayout
    slot: ResourceLease

    def to_dict(self):
        return {
            "layout": self.layout.to_dict(),
            "slot": {
                "resource_id": self.slot.resource_id,
                "trial_id": self.slot.trial_id,
                "lease_path": str(self.slot.path),
            },
        }


class GlobalTrialScheduler:
    """Enforce a task's concurrent-trial budget before creating a worktree."""

    def __init__(self, trials_root: Path):
        self.trials_root = trials_root.expanduser().resolve()
        self.manager = TrialWorkspaceManager(self.trials_root)

    def prepare(
        self,
        task: GlobalTaskSpec,
        source_repository: Path,
        *,
        revision: Optional[str] = None,
        trial_id: Optional[str] = None,
        user_command: Optional[str] = None,
    ) -> ScheduledTrial:
        identifier = trial_id or new_trial_id(task.task_id)
        pool = FileResourcePool(self.trials_root / ".leases" / task.task_id)
        slots = ["slot-{:03d}".format(index) for index in range(task.budget.concurrent_trials)]
        lease = pool.acquire(slots, identifier)
        try:
            layout = self.manager.create(
                task,
                source_repository,
                revision=revision,
                trial_id=identifier,
                user_command=user_command,
            )
        except Exception:
            pool.release(lease)
            raise
        receipt = {
            "schema_version": 1,
            "pool_root": str(lease.pool_root),
            "resource_id": lease.resource_id,
            "trial_id": lease.trial_id,
            "token": lease.token,
            "lease_path": str(lease.path),
        }
        (layout.root / "slot_lease.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        return ScheduledTrial(layout, lease)

    def release(self, layout: TrialLayout) -> None:
        receipt_path = layout.root / "slot_lease.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        lease = ResourceLease(
            pool_root=Path(receipt["pool_root"]),
            resource_id=str(receipt["resource_id"]),
            trial_id=str(receipt["trial_id"]),
            token=str(receipt["token"]),
            path=Path(receipt["lease_path"]),
        )
        FileResourcePool(lease.pool_root).release(lease)
        receipt["released_at"] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
