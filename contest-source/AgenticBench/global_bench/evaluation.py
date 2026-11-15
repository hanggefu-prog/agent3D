"""Trusted preparation and execution of adaptive Open3DBench stage suffixes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from .impact import EvaluationPlan
from .model import GlobalTaskSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trusted_stage(stage_dir: Path, expected_stage: str) -> Dict[str, object]:
    """Validate an Open3DBench stage manifest without importing candidate code."""

    manifest_path = stage_dir / "stage_manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("cannot read trusted stage manifest {}: {}".format(manifest_path, error)) from error
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported trusted stage manifest: {}".format(manifest_path))
    if data.get("status") != "success" or data.get("stage") != expected_stage:
        raise ValueError("trusted stage is unsuccessful or has the wrong identity: {}".format(stage_dir))
    if not isinstance(data.get("signature"), dict):
        raise ValueError("trusted stage has no signature: {}".format(stage_dir))
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("trusted stage has no artifacts: {}".format(stage_dir))
    root = stage_dir.resolve()
    for name, record in artifacts.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid trusted artifact record {}".format(name))
        artifact = (root / record["path"]).resolve()
        if root not in artifact.parents or not artifact.is_file() or artifact.is_symlink():
            raise ValueError("trusted artifact is missing or escapes its stage: {}".format(name))
        if artifact.stat().st_size != record.get("bytes") or _sha256(artifact) != record.get("sha256"):
            raise ValueError("trusted artifact changed: {}".format(artifact))
    return data


@dataclass(frozen=True)
class StageReplay:
    accepted_run: str
    candidate_run: str
    reused_prefix: Tuple[str, ...]
    rerun_stages: Tuple[str, ...]
    command: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["schema_version"] = 1
        return value


class AdaptiveEvaluationRunner:
    """Materialize a verified prefix and invoke the maintained stage CLI."""

    def prepare(
        self,
        task: GlobalTaskSpec,
        plan: EvaluationPlan,
        *,
        accepted_run: Path,
        candidate_run: Path,
        candidate_repository: Path,
        python: str = "python3",
    ) -> StageReplay:
        accepted = accepted_run.expanduser().resolve()
        candidate = candidate_run.expanduser().resolve()
        repository = candidate_repository.expanduser().resolve()
        if not accepted.is_dir():
            raise ValueError("accepted run is unavailable: {}".format(accepted))
        if candidate.exists():
            raise FileExistsError("candidate evaluation run already exists: {}".format(candidate))
        if not (repository / "open3dbench").is_dir():
            raise ValueError("candidate repository has no open3dbench package")
        if not plan.rerun_stages:
            raise ValueError("evaluation plan has no stages to run")

        order = task.evaluation.stage_ids()
        start_index = order.index(plan.rerun_stages[0])
        prefix_ids = tuple(order[:start_index])
        candidate.mkdir(parents=True)
        for name in ("config.yaml",):
            source = accepted / name
            if not source.is_file():
                raise ValueError("accepted run is missing {}".format(name))
            shutil.copy2(source, candidate / name)
        config_path = candidate / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("accepted config.yaml is not a mapping")
        seeds = task.seed_policy
        for section, value in (
            ("pins", seeds.io_placement),
            ("partition", seeds.partition),
            ("placement", seeds.placement),
            ("routing", seeds.routing),
        ):
            row = config.setdefault(section, {})
            if not isinstance(row, dict):
                raise ValueError("accepted config section {} is not a mapping".format(section))
            row["seed"] = value
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        accepted_status_path = accepted / "run.json"
        accepted_status = json.loads(accepted_status_path.read_text(encoding="utf-8"))
        status = dict(accepted_status)
        stage_status = accepted_status.get("stages", {})
        if not isinstance(stage_status, dict):
            raise ValueError("accepted run has malformed stage status")
        status["status"] = "running"
        status["stages"] = {}
        by_id = {stage.stage_id: stage for stage in task.evaluation.stages}
        for stage_id in prefix_ids:
            stage = by_id[stage_id]
            source = accepted / stage.directory
            validate_trusted_stage(source, stage.runner_name)
            shutil.copytree(source, candidate / stage.directory, symlinks=False)
            if stage.runner_name in stage_status:
                status["stages"][stage.runner_name] = stage_status[stage.runner_name]
        (candidate / "run.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        rerun = [by_id[stage_id] for stage_id in plan.rerun_stages]
        command = (
            python,
            "-m",
            "open3dbench",
            "--repo",
            str(repository),
            "run",
            str(candidate / "config.yaml"),
            "--from",
            rerun[0].runner_name,
            "--to",
            rerun[-1].runner_name,
            "--resume",
            "--run-dir",
            str(candidate),
        )
        replay = StageReplay(
            accepted_run=str(accepted),
            candidate_run=str(candidate),
            reused_prefix=prefix_ids,
            rerun_stages=plan.rerun_stages,
            command=command,
        )
        (candidate / "global_evaluation_plan.json").write_text(
            json.dumps(
                {"impact": plan.to_dict(), "replay": replay.to_dict()},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return replay

    def run(
        self,
        replay: StageReplay,
        *,
        environment: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            list(replay.command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            timeout=timeout_seconds,
        )
