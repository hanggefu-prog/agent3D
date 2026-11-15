"""Versioned, portable upstream inputs shared by local benchmark tasks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

from .workspace import copy_upstream_stages


PROFILE_ROOT = Path(__file__).resolve().parent / "benchmark_inputs"
MANIFEST_NAME = "benchmark_input.json"
SCHEMA_VERSION = 1
TASK_COMPATIBILITY_ALIASES = {
    # Identical immutable frontend, separately qualified thermal authority.
    "tp_gnn_placement_cts_thermal_20261008": ("tp_gnn_placement_one_pass_20261004",),
    # Thermal promotion starts from the identical input/floorplan plus frozen
    # partition-last timing-feature bundle used by the STA co-design profile.
    # Keeping this relation in trusted code avoids rewriting released bundle
    # manifests merely to add a downstream evaluator.
    "tp_gnn_placement_thermal": ("tp_gnn_placement_sta",),
    "tp_gnn_placement_cts_thermal": ("tp_gnn_placement_sta",),
}
LEGACY_PROFILE_STAGE_ORDER = (
    "input",
    "floorplan",
    "partition",
    "placement",
    "pdn",
    "hbt",
    "opt",
    "cts",
    "routing",
)
PROFILE_STAGE_DIRS = {
    "input": "00_input", "floorplan": "10_floorplan",
    "timing_extract": "15_timing_extract", "partition": "20_partition",
    "placement": "30_placement", "pdn": "40_pdn", "hbt": "50_hbt",
    "opt": "60_opt", "cts": "70_cts", "global_routing": "75_global_route",
    "detailed_routing": "80_route", "report": "90_report", "thermal": "95_thermal",
}
PROFILE_STAGE_ORDER = tuple(PROFILE_STAGE_DIRS)


@dataclass(frozen=True)
class BenchmarkInputProfile:
    root: Path
    profile_id: str
    description: str
    config: Path
    stop_stage: str
    required_stages: Dict[str, str]
    compatible_tasks: tuple[str, ...]
    python_minimum: str
    partition_method: Optional[str] = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_path(value: Union[str, Path]) -> Path:
    explicit = Path(value).expanduser()
    if explicit.is_file():
        return explicit.resolve()
    return (PROFILE_ROOT / str(value) / "profile.yaml").resolve()


def load_benchmark_input_profile(value: Union[str, Path]) -> BenchmarkInputProfile:
    path = _profile_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark input profile: {}".format(path))
    stages = data.get("required_stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("required_stages must be a non-empty object: {}".format(path))
    required_stages = {str(name): str(directory) for name, directory in stages.items()}
    stage_names = list(required_stages)
    supported_orders = (PROFILE_STAGE_ORDER, LEGACY_PROFILE_STAGE_ORDER,
                        tuple(stage for stage in PROFILE_STAGE_ORDER if stage != "timing_extract"))
    if not any(stage_names == list(order[:len(stage_names)]) for order in supported_orders):
        raise ValueError(
            "required_stages must be a continuous pipeline prefix in order: {}".format(
                ", ".join(PROFILE_STAGE_ORDER)
            )
        )
    stop_stage = str(data.get("stop_stage", stage_names[-1]))
    if stop_stage != stage_names[-1]:
        raise ValueError(
            "stop_stage must equal the final frozen stage {!r}: {}".format(
                stage_names[-1], path
            )
        )
    config = (path.parent / str(data.get("config", "config.yaml"))).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    partition_contract = data.get("partition_contract")
    partition_method = None
    if (str(data["id"]).endswith("_lol_ng45_tp_gnn_flat_placement")
            and partition_contract is None):
        raise ValueError("flat placement profile requires an explicit partition contract")
    if partition_contract is not None:
        # This is an explicit new producer identity, not a normalization of
        # historical TP-GNN artifacts. Both the profile and config are hashed.
        if partition_contract != {"method": "tp-gnn-flat", "decoder": "global-2way"}:
            raise ValueError("unsupported explicit placement partition contract")
        configured = yaml.safe_load(config.read_text(encoding="utf-8"))
        if (not isinstance(configured, dict)
                or not isinstance(configured.get("partition"), dict)
                or configured.get("partition", {}).get("algorithm") != "tp-gnn-flat"
                or stop_stage != "partition"):
            raise ValueError("explicit flat placement profile must configure tp-gnn-flat through partition")
        partition_method = "tp-gnn-flat"
    return BenchmarkInputProfile(
        root=path.parent,
        profile_id=str(data["id"]),
        description=str(data.get("description", "")),
        config=config,
        stop_stage=stop_stage,
        required_stages=required_stages,
        compatible_tasks=tuple(str(item) for item in data.get("compatible_tasks", [])),
        python_minimum=str(data.get("runtime", {}).get("python_minimum", "3.9")),
        partition_method=partition_method,
    )


def _preparation_stages(profile: BenchmarkInputProfile, plan: list) -> Dict[str, str]:
    """Validate a native plan without inventing missing producer manifests."""
    if not isinstance(plan, list) or not plan or any(not isinstance(row, dict) for row in plan):
        raise ValueError("native preparation plan must be a non-empty list")
    names = [row.get("stage") for row in plan]
    supported = (PROFILE_STAGE_ORDER, LEGACY_PROFILE_STAGE_ORDER)
    if not any(names == list(order[:len(names)]) for order in supported):
        raise ValueError("native preparation plan is not a maintained ordered stage prefix")
    expected_stop = profile.stop_stage
    if expected_stop == "routing" and "detailed_routing" in names:
        expected_stop = "detailed_routing"
    if names[-1] != expected_stop:
        raise ValueError("native preparation plan ends at the wrong profile boundary")
    stages = {}
    for row in plan:
        name = row["stage"]
        directory = PROFILE_STAGE_DIRS.get(name, "80_route" if name == "routing" else None)
        if row.get("directory") != directory:
            raise ValueError("native preparation stage has the wrong directory: " + name)
        if row.get("skip") is not None:
            if not isinstance(row["skip"], str) or not row["skip"]:
                raise ValueError("skipped native stage needs an explicit reason: " + name)
            continue
        stages[name] = directory
    for name, directory in profile.required_stages.items():
        actual = "detailed_routing" if name == "routing" and "detailed_routing" in names else name
        if stages.get(actual) != directory:
            raise ValueError("native plan does not produce required profile stage: " + name)
    return stages


def _native_preparation_plan(root: Path, python: Path, profile: BenchmarkInputProfile) -> list:
    """Ask the dependency checkout which prerequisite stages actually run."""
    process = subprocess.run(
        [str(python), "-m", "open3dbench", "--repo", str(root), "plan", str(profile.config)],
        cwd=root, check=True, capture_output=True, text=True,
    )
    payload = json.loads(process.stdout)
    plan = payload.get("plan") if isinstance(payload, dict) else None
    if not isinstance(plan, list):
        raise ValueError("Open3DBench plan returned no stage list")
    stop = profile.stop_stage
    if stop == "routing" and any(row.get("stage") == "detailed_routing" for row in plan):
        stop = "detailed_routing"
    matches = [index for index, row in enumerate(plan) if row.get("stage") == stop]
    if len(matches) != 1:
        raise ValueError("Open3DBench plan does not identify the profile stop stage")
    prefix = plan[:matches[0] + 1]
    _preparation_stages(profile, prefix)
    return prefix


def _git_identity(root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    return {"root": str(root), "revision": revision, "dirty": dirty}


def _python_identity(executable: Path) -> Dict[str, str]:
    result = subprocess.run(
        [
            str(executable),
            "-c",
            "import json,sys,yaml; print(json.dumps({'version':sys.version,'executable':sys.executable,'pyyaml':yaml.__version__}))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _resolve_python(value: Optional[Path]) -> Path:
    raw = str(value) if value is not None else os.environ.get("OPEN3DBENCH_PYTHON", sys.executable)
    found = shutil.which(raw) if not Path(raw).expanduser().is_absolute() else raw
    if not found or not Path(found).expanduser().is_file():
        raise FileNotFoundError("Open3DBench Python executable: {}".format(raw))
    return Path(found).expanduser().resolve()


def _resolve_dreamplace(root: Path, value: Optional[Path]) -> Path:
    raw = value or (
        Path(os.environ["OPEN3D_LOCAL_DREAMPLACE_INSTALL"])
        if os.environ.get("OPEN3D_LOCAL_DREAMPLACE_INSTALL")
        else root / "Place-3D" / "build" / "dreamplace" / "install"
    )
    install = Path(raw).expanduser().resolve()
    if not (install / "dreamplace").is_dir():
        raise FileNotFoundError(
            "DREAMPlace install must contain dreamplace/: {}".format(install)
        )
    return install


def _validate_stage_manifest(path: Path, expected_stage: str) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError("missing stage manifest: {}".format(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "success":
        raise ValueError("stage manifest is not a successful schema-1 manifest: {}".format(path))
    if payload.get("stage") != expected_stage:
        raise ValueError(
            "stage manifest says {!r}, expected {!r}: {}".format(
                payload.get("stage"), expected_stage, path
            )
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("stage manifest has no artifacts: {}".format(path))
    stage_root = path.parent.resolve()
    for name, record in artifacts.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid artifact {!r}: {}".format(name, path))
        artifact = (stage_root / record["path"]).resolve()
        try:
            artifact.relative_to(stage_root)
        except ValueError as error:
            raise ValueError("artifact escapes stage directory: {}".format(artifact)) from error
        if not artifact.is_file():
            raise ValueError("missing stage artifact: {}".format(artifact))
        if artifact.stat().st_size != record.get("bytes"):
            raise ValueError("stage artifact size changed: {}".format(artifact))
        if _sha256(artifact) != record.get("sha256"):
            raise ValueError("stage artifact hash changed: {}".format(artifact))
    return payload


def _refresh_adopted_stage_manifest(path: Path, source_run: Path) -> None:
    """Refresh artifact records after portable run-owned paths are rebased."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("adopted stage manifest has no artifacts: {}".format(path))
    stage_root = path.parent.resolve()
    for name, record in artifacts.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid adopted artifact {!r}: {}".format(name, path))
        artifact = (stage_root / record["path"]).resolve()
        try:
            artifact.relative_to(stage_root)
        except ValueError as error:
            raise ValueError("adopted artifact escapes stage: {}".format(artifact)) from error
        if not artifact.is_file():
            raise ValueError("adopted artifact is missing: {}".format(artifact))
        record["bytes"] = artifact.stat().st_size
        record["sha256"] = _sha256(artifact)
    provenance = payload.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("adopted stage provenance is invalid: {}".format(path))
    provenance["benchmark_input_adoption"] = {
        "source_run": str(source_run),
        "paths_rebased": True,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_benchmark_input(value: Path, *, task_id: Optional[str] = None) -> Dict[str, Any]:
    value = value.expanduser().resolve()
    manifest_path = value / MANIFEST_NAME if value.is_dir() else value
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark input manifest: {}".format(manifest_path))
    profile = load_benchmark_input_profile(str(data.get("benchmark_id", "")))
    if data.get("profile_sha256") != _sha256(profile.root / "profile.yaml"):
        raise ValueError("benchmark input profile changed since preparation")
    if data.get("config_sha256") != _sha256(profile.config):
        raise ValueError("benchmark input configuration changed since preparation")
    compatible_task_ids = {task_id} if task_id else set()
    if task_id:
        compatible_task_ids.update(TASK_COMPATIBILITY_ALIASES.get(task_id, ()))
    if task_id and not compatible_task_ids.intersection(profile.compatible_tasks):
        raise ValueError(
            "benchmark input {} is not compatible with task {}".format(
                profile.profile_id, task_id
            )
        )
    run_dir = (manifest_path.parent / str(data.get("run_dir", "run"))).resolve()
    try:
        run_dir.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError("benchmark run directory escapes its input bundle") from error
    stage_records: Dict[str, Any] = {}
    # Old sealed bundles remain metadata-compatible; new preparations include
    # step 15 and split-routing producers from the dependency's actual plan.
    required_stages = profile.required_stages
    if "preparation_plan" in data:
        required_stages = _preparation_stages(profile, data["preparation_plan"])
        if set(data.get("stages", {})) != set(required_stages):
            raise ValueError("frozen stage records disagree with the native preparation plan")
    for stage, directory in required_stages.items():
        stage_manifest = run_dir / directory / "stage_manifest.json"
        payload = _validate_stage_manifest(stage_manifest, stage)
        expected = data.get("stages", {}).get(stage, {})
        if expected.get("manifest_sha256") != _sha256(stage_manifest):
            raise ValueError("frozen stage manifest changed: {}".format(stage_manifest))
        stage_records[stage] = {
            "directory": str(run_dir / directory),
            "artifact_count": len(payload["artifacts"]),
        }
    partition_method = None
    if profile.partition_method or profile.profile_id.endswith("_lol_ng45_tp_gnn_placement"):
        from .placement_baseline import validate_tp_gnn_partition
        partition_method = validate_tp_gnn_partition(
            run_dir, expected_method=profile.partition_method or "tp-gnn",
            require_bound_report=profile.partition_method is not None,
        )
    if "tp_gnn_pin_cut_learning_objective" in profile.compatible_tasks:
        raise ValueError("Retired pin-cut input validation is not supported")
    return {
        "status": "ok",
        "manifest": str(manifest_path),
        "benchmark_id": profile.profile_id,
        "config": str(profile.config),
        "run_dir": str(run_dir),
        "compatible_tasks": list(profile.compatible_tasks),
        "stages": stage_records,
        "environment": data.get("environment", {}),
        "partition_method": partition_method,
    }


def prepare_benchmark_input(
    profile_value: Union[str, Path],
    *,
    open3dbench_root: Path,
    output_dir: Path,
    python_executable: Optional[Path] = None,
    dreamplace_install: Optional[Path] = None,
    seed_run: Optional[Path] = None,
    timeout_seconds: int = 21600,
    resume: bool = False,
) -> Dict[str, Any]:
    profile = load_benchmark_input_profile(profile_value)
    if "tp_gnn_pin_cut_learning_objective" in profile.compatible_tasks:
        raise ValueError("Retired pin-cut input preparation is not supported")
    root = open3dbench_root.expanduser().resolve()
    if not (root / "open3dbench" / "__main__.py").is_file():
        raise FileNotFoundError("not an Open3DBench checkout: {}".format(root))
    python = _resolve_python(python_executable)
    python_info = _python_identity(python)
    minimum = tuple(int(item) for item in profile.python_minimum.split(".")[:2])
    version_text = python_info["version"].split()[0]
    current = tuple(int(item) for item in version_text.split(".")[:2])
    if current < minimum:
        raise RuntimeError(
            "Open3DBench Python {} is older than required {}".format(
                version_text, profile.python_minimum
            )
        )
    dreamplace = _resolve_dreamplace(root, dreamplace_install)
    output = output_dir.expanduser().resolve()
    manifest_path = output / MANIFEST_NAME
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError("benchmark input already exists; pass --resume: {}".format(output))
        return validate_benchmark_input(manifest_path)
    preparation_plan = _native_preparation_plan(root, python, profile)
    required_stages = _preparation_stages(profile, preparation_plan)
    seed = seed_run.expanduser().resolve() if seed_run is not None else None
    adopted_stages = []
    if seed is not None:
        for stage, directory in required_stages.items():
            stage_manifest = seed / directory / "stage_manifest.json"
            if not stage_manifest.is_file():
                break
            _validate_stage_manifest(stage_manifest, stage)
            adopted_stages.append(stage)
        if adopted_stages[:2] != ["input", "floorplan"]:
            raise ValueError(
                "seed run must contain successful input and floorplan stages: {}".format(
                    seed
                )
            )
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("output directory is not empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / "run"
    if seed is not None and not run_dir.exists():
        copy_upstream_stages(seed, run_dir, adopted_stages)
    if seed is not None:
        for stage in adopted_stages:
            directory = required_stages[stage]
            _refresh_adopted_stage_manifest(
                run_dir / directory / "stage_manifest.json", seed
            )
    completed_stages = []
    if resume and run_dir.is_dir():
        for stage, directory in required_stages.items():
            stage_manifest = run_dir / directory / "stage_manifest.json"
            if not stage_manifest.is_file():
                break
            _validate_stage_manifest(stage_manifest, stage)
            completed_stages.append(stage)
    else:
        completed_stages = list(adopted_stages)
    log_path = output / "prepare.log"
    command = []
    if len(completed_stages) < len(required_stages):
        command = [
            str(python),
            "-m",
            "open3dbench",
            "--repo",
            str(root),
            "run",
            str(profile.config),
        ]
        if completed_stages:
            next_stage = list(required_stages)[len(completed_stages)]
            command.extend(["--from", next_stage])
        command.extend(["--to", profile.stop_stage, "--run-dir", str(run_dir)])
        if seed is not None or (resume and run_dir.exists()):
            command.append("--resume")
    environment = os.environ.copy()
    environment["OPEN3D_LOCAL_DREAMPLACE_INSTALL"] = str(dreamplace)
    # The benchmark-input helper uses OPEN3D_LOCAL_DREAMPLACE_INSTALL to
    # locate and fingerprint a maintained runtime.  Open3DBench's placement
    # runner consumes the public PLACE3D_DREAMPLACE_INSTALL variable, so bind
    # both names to the exact same frozen runtime.
    environment["PLACE3D_DREAMPLACE_INSTALL"] = str(dreamplace)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        if command:
            process = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    "Open3DBench input preparation failed; see {}".format(log_path)
                )
        else:
            log.write(
                "All required stages were validated from the existing frozen run.\n"
            )
    stage_records: Dict[str, Any] = {}
    for stage, directory in required_stages.items():
        stage_manifest = run_dir / directory / "stage_manifest.json"
        payload = _validate_stage_manifest(stage_manifest, stage)
        stage_records[stage] = {
            "directory": directory,
            "manifest_sha256": _sha256(stage_manifest),
            "artifact_count": len(payload["artifacts"]),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": profile.profile_id,
        "description": profile.description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "run_dir": "run",
        "profile_sha256": _sha256(profile.root / "profile.yaml"),
        "config_sha256": _sha256(profile.config),
        "stages": stage_records,
        "preparation_plan": preparation_plan,
        "environment": {
            "python": python_info,
            "python_minimum": profile.python_minimum,
            "open3dbench": _git_identity(root),
            "openagentic3dbench": _git_identity(Path(__file__).resolve().parents[1]),
            "dreamplace_install": str(dreamplace),
        },
        "command": command,
        "seed_run": str(seed) if seed is not None else None,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validate_benchmark_input(manifest_path)


def resolve_benchmark_run(value: Path, *, task_id: Optional[str] = None) -> Path:
    return Path(validate_benchmark_input(value, task_id=task_id)["run_dir"])
