"""Trusted structural verification and isolated verifier launch planning."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Optional, Tuple

import yaml

from .model import GlobalTaskSpec
from .submission import SubmissionManifest


DEFAULT_VERIFIER_ROOT = Path(__file__).resolve().parent / "verifiers"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _submitted_file(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError("unsafe path in submission manifest: {}".format(relative))
    path = (root / value.as_posix()).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError("submission file is missing or unsafe: {}".format(relative))
    return path


@dataclass(frozen=True)
class TrustedVerifier:
    verifier_id: str
    image: str
    image_digest: Optional[str]
    entrypoint: Tuple[str, ...]
    timeout_seconds: int

    @property
    def image_reference(self) -> str:
        if self.image_digest:
            return "{}@{}".format(self.image.split("@")[0], self.image_digest)
        return self.image


@dataclass(frozen=True)
class VerificationReport:
    task_id: str
    trial_id: str
    verifier_id: str
    passed: bool
    checks: Mapping[str, bool]
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["schema_version"] = 1
        return value


def load_trusted_verifier(
    verifier_id: str,
    root: Path = DEFAULT_VERIFIER_ROOT,
) -> TrustedVerifier:
    path = root.expanduser().resolve() / (verifier_id + ".yaml")
    if not path.is_file():
        raise KeyError("unknown trusted verifier: {}".format(verifier_id))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema_version", "id", "image", "image_digest", "entrypoint", "timeout_seconds"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("malformed trusted verifier definition: {}".format(path))
    if int(data["schema_version"]) != 1 or str(data["id"]) != verifier_id:
        raise ValueError("trusted verifier identity mismatch")
    entrypoint = tuple(str(item) for item in data["entrypoint"])
    if not entrypoint:
        raise ValueError("trusted verifier entrypoint must not be empty")
    digest = data["image_digest"]
    verifier = TrustedVerifier(
        verifier_id=verifier_id,
        image=str(data["image"]),
        image_digest=None if digest is None else str(digest),
        entrypoint=entrypoint,
        timeout_seconds=int(data["timeout_seconds"]),
    )
    if verifier.timeout_seconds <= 0:
        raise ValueError("trusted verifier timeout must be positive")
    return verifier


def verify_submission_structure(
    task: GlobalTaskSpec,
    submission_root: Path,
    *,
    verifier_root: Path = DEFAULT_VERIFIER_ROOT,
) -> VerificationReport:
    """Verify hashes and boundaries before any expensive EDA command runs."""

    root = submission_root.expanduser().resolve()
    manifest = SubmissionManifest.from_path(root / "submission.json")
    errors = []
    checks = {
        "task_identity": manifest.task_id == task.task_id and manifest.task_digest == task.digest(),
        "patch_integrity": False,
        "change_integrity": True,
        "artifact_completeness": False,
        "artifact_integrity": True,
        "control_plane_unchanged": True,
        "trusted_verifier_available": False,
    }
    if not checks["task_identity"]:
        errors.append("task identity or digest mismatch")
    try:
        patch = _submitted_file(root, manifest.patch_path)
        checks["patch_integrity"] = _sha256(patch) == manifest.patch_sha256
    except ValueError as error:
        errors.append(str(error))
    if not checks["patch_integrity"]:
        errors.append("patch hash mismatch")

    for change in manifest.changes:
        if any(
            change.path == prefix or change.path.startswith(prefix.rstrip("/") + "/")
            for prefix in task.workspace.protected_paths
        ):
            checks["control_plane_unchanged"] = False
            errors.append("protected path changed: {}".format(change.path))
        if change.submitted_path is None:
            if change.sha256 is not None:
                checks["change_integrity"] = False
            continue
        try:
            path = _submitted_file(root, change.submitted_path)
            if _sha256(path) != change.sha256:
                checks["change_integrity"] = False
                errors.append("changed source hash mismatch: {}".format(change.path))
        except ValueError as error:
            checks["change_integrity"] = False
            errors.append(str(error))

    names = {artifact.name for artifact in manifest.artifacts}
    checks["artifact_completeness"] = names == set(task.submission.required_artifacts)
    if not checks["artifact_completeness"]:
        errors.append("required artifact set mismatch")
    for artifact in manifest.artifacts:
        try:
            path = _submitted_file(root, artifact.submitted_path)
            if path.stat().st_size != artifact.size or _sha256(path) != artifact.sha256:
                checks["artifact_integrity"] = False
                errors.append("artifact hash mismatch: {}".format(artifact.name))
        except ValueError as error:
            checks["artifact_integrity"] = False
            errors.append(str(error))

    try:
        load_trusted_verifier(task.verifier.verifier_id, verifier_root)
        checks["trusted_verifier_available"] = True
    except (KeyError, ValueError) as error:
        errors.append(str(error))
    passed = all(checks.values()) and not errors
    return VerificationReport(
        task_id=task.task_id,
        trial_id=manifest.trial_id,
        verifier_id=task.verifier.verifier_id,
        passed=passed,
        checks=checks,
        errors=tuple(dict.fromkeys(errors)),
    )


def build_verifier_container_command(
    task: GlobalTaskSpec,
    submission_root: Path,
    evaluated_run: Path,
    input_sources: Mapping[str, Path],
    output_root: Path,
    scratch_root: Path,
    *,
    verifier_root: Path = DEFAULT_VERIFIER_ROOT,
    docker_binary: str = "docker",
) -> Tuple[str, ...]:
    """Build a verifier command with no writable candidate source mount."""

    verifier = load_trusted_verifier(task.verifier.verifier_id, verifier_root)
    submission = submission_root.expanduser().resolve()
    evaluation = evaluated_run.expanduser().resolve()
    output = output_root.expanduser().resolve()
    scratch = scratch_root.expanduser().resolve()
    if not (submission / "submission.json").is_file():
        raise ValueError("verifier submission is missing submission.json")
    if not (evaluation / "run.json").is_file():
        raise ValueError("trusted evaluated run is missing run.json")
    if not output.is_dir() or not scratch.is_dir():
        raise ValueError("verifier output and scratch roots must already exist")
    command = [
        docker_binary,
        "run",
        "--rm",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--network",
        "none",
        "--mount",
        "type=bind,src={},dst=/submission,readonly".format(submission),
        "--mount",
        "type=bind,src={},dst=/evaluation-run,readonly".format(
            evaluation
        ),
        "--mount",
        "type=bind,src={},dst=/verifier-output,rw".format(output),
        "--mount",
        "type=bind,src={},dst=/verifier-scratch,rw".format(scratch),
        "--mount",
        "type=bind,src={},dst=/trusted,readonly".format(
            verifier_root.expanduser().resolve()
        ),
    ]
    expected_inputs = {item.name for item in task.inputs}
    if set(input_sources) != expected_inputs:
        raise ValueError("verifier input set does not match the global task")
    for item in task.inputs:
        source = input_sources[item.name].expanduser().resolve()
        if not source.exists():
            raise ValueError("verifier input is missing: {}".format(source))
        expected_digest = item.sha256
        if expected_digest is not None and source.is_file():
            if _sha256(source) != expected_digest:
                raise ValueError("verifier input digest mismatch: {}".format(source))
        command.extend(
            ["--mount", "type=bind,src={},dst={},readonly".format(source, item.mount)]
        )
    command.append(verifier.image_reference)
    command.extend(verifier.entrypoint)
    return tuple(command)
