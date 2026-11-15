"""Container launch plans for existing terminal-capable coding agents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .model import GlobalTaskSpec
from .workspace import TrialLayout, trial_storage_bytes


_ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_source(value: str, environment: Mapping[str, str]) -> Path:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in environment:
            raise ValueError("input source requires unset environment variable {}".format(name))
        return environment[name]

    expanded = _ENVIRONMENT_REFERENCE.sub(replace, value)
    path = Path(expanded).expanduser().resolve()
    if not path.exists():
        raise ValueError("input source does not exist: {}".format(path))
    return path


def _verify_file_digest(path: Path, expected: Optional[str]) -> None:
    if expected is None or path.is_dir():
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("input digest mismatch for {}".format(path))


def _git_directory(workspace: Path, option: str) -> Path:
    result = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", option],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("cannot resolve {}: {}".format(option, result.stderr.strip()))
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError("resolved Git directory is unavailable: {}".format(resolved))
    return resolved


@dataclass(frozen=True)
class RuntimePlan:
    trial_id: str
    container_name: str
    image: str
    command: Tuple[str, ...]
    agent_command: Tuple[str, ...]
    environment: Mapping[str, str]
    input_mounts: Tuple[Mapping[str, str], ...]

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["schema_version"] = 1
        return value


@dataclass(frozen=True)
class RuntimeResult:
    trial_id: str
    exit_code: int
    timed_out: bool
    started_at: str
    finished_at: str
    stdout: str
    stderr: str

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["schema_version"] = 1
        return value


class DockerTrialRuntime:
    """Launch an arbitrary CLI agent without redefining its filesystem tools."""

    def __init__(self, docker_binary: str = "docker"):
        self.docker_binary = docker_binary

    def plan(
        self,
        task: GlobalTaskSpec,
        layout: TrialLayout,
        agent_command: Sequence[str],
        *,
        skills_root: Optional[Path] = None,
        host_environment: Optional[Mapping[str, str]] = None,
        assigned_gpus: Sequence[str] = (),
    ) -> RuntimePlan:
        if not agent_command:
            raise ValueError("agent_command must not be empty")
        if len(assigned_gpus) != task.environment.gpus:
            raise ValueError(
                "task requests {} GPUs but {} were assigned".format(
                    task.environment.gpus, len(assigned_gpus)
                )
            )
        environment = dict(os.environ if host_environment is None else host_environment)
        worktree_git = _git_directory(layout.workspace, "--git-dir")
        common_git = _git_directory(layout.workspace, "--git-common-dir")
        image = task.environment.image
        if task.environment.image_digest:
            image = "{}@{}".format(image.split("@")[0], task.environment.image_digest)
        container_name = re.sub(r"[^A-Za-z0-9_.-]", "-", "o3db-" + layout.trial_id)[:128]
        command = [
            self.docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(task.environment.pids_limit),
            "--cpus",
            str(task.environment.cpu),
            "--memory",
            "{}g".format(task.environment.memory_gb),
            "--network",
            {
                "none": "none",
                # Restricted egress requires an administrator-provisioned
                # network. Until one is named, fail closed with no network.
                "restricted": environment.get(
                    "OPENAGENTIC3D_RESTRICTED_NETWORK", "none"
                ),
                "host": "host",
            }[task.environment.network],
            "--workdir",
            task.workspace.container_path,
            "--mount",
            "type=bind,src={},dst={},rw".format(
                layout.workspace, task.workspace.container_path
            ),
            "--mount",
            "type=bind,src={},dst={},rw".format(layout.runs, task.workspace.runs_path),
            "--mount",
            "type=bind,src={},dst={},rw".format(
                layout.scratch, task.workspace.scratch_path
            ),
            "--mount",
            "type=bind,src={},dst=/task,readonly".format(layout.control),
            "--mount",
            "type=bind,src={},dst=/build,rw".format(layout.build),
            "--mount",
            "type=bind,src={},dst=/logs,rw".format(layout.logs),
            "--mount",
            "type=bind,src={},dst=/git/common,readonly".format(common_git),
            "--mount",
            "type=bind,src={},dst=/git/worktree,rw".format(worktree_git),
            "--env",
            "HOME={}/home".format(task.workspace.scratch_path),
            "--env",
            "XDG_CACHE_HOME={}/cache".format(task.workspace.scratch_path),
            "--env",
            "OPENAGENTIC3D_TRIAL_ID={}".format(layout.trial_id),
            "--env",
            "OPENAGENTIC3D_TASK=/task/task.json",
            "--env",
            "OPENAGENTIC3D_INSTRUCTION=/task/instruction.md",
            "--env",
            "OPENAGENTIC3D_USER_COMMAND=/task/user-command.md",
            "--env",
            "OPENAGENTIC3D_RUNS_ROOT={}".format(task.workspace.runs_path),
            "--env",
            "GIT_DIR=/git/worktree",
            "--env",
            "GIT_COMMON_DIR=/git/common",
            "--env",
            "GIT_WORK_TREE={}".format(task.workspace.container_path),
        ]
        if assigned_gpus:
            command.extend(["--gpus", "device={}".format(",".join(assigned_gpus))])
        if skills_root is not None:
            skill_path = skills_root.expanduser().resolve()
            if not skill_path.is_dir():
                raise ValueError("skills_root is not a directory: {}".format(skill_path))
            command.extend(
                ["--mount", "type=bind,src={},dst=/tools/skills,readonly".format(skill_path)]
            )
        for protected in task.workspace.protected_paths:
            host_path = (layout.workspace / protected).resolve()
            if not host_path.exists():
                # A decoupled Open3DBench checkout has no AgenticBench control
                # tree in the subject workspace, which is the preferred case.
                continue
            if layout.workspace not in host_path.parents:
                raise ValueError("protected workspace path escapes the workspace: {}".format(protected))
            destination = "{}/{}".format(
                task.workspace.container_path.rstrip("/"), protected
            )
            command.extend(
                ["--mount", "type=bind,src={},dst={},readonly".format(host_path, destination)]
            )

        mounts = []
        for item in task.inputs:
            source = _resolve_source(item.source, environment)
            _verify_file_digest(source, item.sha256)
            mode = "readonly" if item.read_only else "rw"
            command.extend(
                ["--mount", "type=bind,src={},dst={},{}".format(source, item.mount, mode)]
            )
            mounts.append({"name": item.name, "source": str(source), "mount": item.mount, "mode": mode})
        command.append(image)
        command.extend(str(item) for item in agent_command)
        agent_environment = {
            "OPENAGENTIC3D_TRIAL_ID": layout.trial_id,
            "OPENAGENTIC3D_TASK": "/task/task.json",
            "OPENAGENTIC3D_INSTRUCTION": "/task/instruction.md",
            "OPENAGENTIC3D_USER_COMMAND": "/task/user-command.md",
            "OPENAGENTIC3D_RUNS_ROOT": task.workspace.runs_path,
            "GIT_DIR": "/git/worktree",
            "GIT_COMMON_DIR": "/git/common",
            "GIT_WORK_TREE": task.workspace.container_path,
        }
        return RuntimePlan(
            trial_id=layout.trial_id,
            container_name=container_name,
            image=image,
            command=tuple(command),
            agent_command=tuple(str(item) for item in agent_command),
            environment=agent_environment,
            input_mounts=tuple(mounts),
        )

    def run(
        self,
        plan: RuntimePlan,
        layout: TrialLayout,
        timeout_seconds: int,
        *,
        storage_limit_gb: Optional[float] = None,
    ) -> RuntimeResult:
        started = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(
                list(plan.command),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )
            exit_code = int(result.returncode)
            stdout = result.stdout
            stderr = result.stderr
            timed_out = False
        except subprocess.TimeoutExpired as error:
            subprocess.run(
                [self.docker_binary, "rm", "-f", plan.container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            exit_code = 124
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            timed_out = True
        (layout.logs / "agent.stdout.log").write_text(stdout, encoding="utf-8")
        (layout.logs / "agent.stderr.log").write_text(stderr, encoding="utf-8")
        if storage_limit_gb is not None:
            storage_limit = int(storage_limit_gb * 1024 ** 3)
            used = trial_storage_bytes(layout.root)
            if used > storage_limit:
                exit_code = 125
                stderr = stderr + "\ntrial storage budget exceeded: {} > {} bytes\n".format(
                    used, storage_limit
                )
                (layout.logs / "agent.stderr.log").write_text(stderr, encoding="utf-8")
        finished = datetime.now(timezone.utc).isoformat()
        outcome = RuntimeResult(
            trial_id=layout.trial_id,
            exit_code=exit_code,
            timed_out=timed_out,
            started_at=started,
            finished_at=finished,
            stdout=str(layout.logs / "agent.stdout.log"),
            stderr=str(layout.logs / "agent.stderr.log"),
        )
        (layout.root / "runtime_result.json").write_text(
            json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return outcome
