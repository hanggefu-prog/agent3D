"""Pinned physical/configuration identity for the 2026-09-23 reference family.

This contract is benchmark-owned. It does not turn archived metrics into a
qualified scoring reference, and it never rewrites a released artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

BASELINE_ID = "reference_20260923"
IDENTITY_FILE = Path(__file__).with_name("reference_20260923_identity.json")
PROFILE_ROOT = Path(__file__).with_name("benchmark_inputs")
IMAGE_IDS = frozenset({
    "sha256:3b8e1793c0e780308101b480fa17f05771ad97d728209146e842bf6dea471583",
    "sha256:f7d91b5248204b6fa11f55e7fd3bce8fe95d455d84fdca3bd582f4859ceda0db",
})
PARTITION_CONFIG = "Place-3D/config/partition/tp_gnn_reference_20260923.json"
PLACEMENT_CONFIG = "Place-3D/config/placement/lol/reference_20260923.json"


def identity():
    return json.loads(IDENTITY_FILE.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_path(case):
    if case not in identity()["cases"]:
        raise ValueError("Case has no frozen 2026-09-23 reference: " + str(case))
    return PROFILE_ROOT / (case + "_lol_ng45_reference_20260923")


def validate_audit_profile(case, receipt):
    """Return the current audit profile, bound to the unchanged physical recipe.

    Benchmark-input validation resolves a versioned profile ID in the current
    framework; its returned config path is not the historical importer path.
    The importer/physical source remains separately pinned in the input manifest
    and evaluation contract. No path substitution or weaker hash check is used.
    """
    from scripts.workspace_framework_identity import validate_audit_framework
    framework = Path(__file__).resolve().parents[1]
    if not isinstance(receipt, dict) or receipt.get("trusted_framework") != str(framework):
        raise ValueError("Reference bundle audit framework identity differs")
    current = validate_audit_framework(receipt["computation_source"], framework,
        source_revision=receipt["computation_source_revision"],
        framework_revision=receipt["trusted_framework_revision"])
    if current != receipt:
        raise ValueError("Reference bundle audit compatibility changed")
    profile = profile_path(case)
    expected = framework / "local_bench/benchmark_inputs" / profile.name
    if profile != expected:
        raise ValueError("Audit reference profile is outside the pinned framework")
    physical = Path(receipt["computation_source"]) / "AgenticBench/local_bench/benchmark_inputs" / profile.name
    for name in ("profile.yaml", "config.yaml"):
        paths = (profile / name, physical / name)
        for path in paths:
            if (any(parent.is_symlink() for parent in (path, *path.parents))
                    or not path.is_file()):
                raise ValueError("Audit/physical reference profile must contain regular files")
        if paths[0].stat().st_size != paths[1].stat().st_size or sha256(paths[0]) != sha256(paths[1]):
            raise ValueError("Audit reference profile differs from the physical recipe")
    return profile


def validate_reference_image(image):
    if image not in IMAGE_IDS:
        raise ValueError("Reference-aligned evaluation requires the pinned 2026-09-23 image")
    return image


def validate_physical_prefix(run, case, *, fixed_partition=False):
    record = identity()["cases"].get(case)
    if record is None:
        raise ValueError("Case has no frozen reference")
    root = Path(run).resolve(strict=True)
    checked = {}
    for relative, expected in record["physical_prefix"].items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing regular frozen physical input: " + relative)
        path.resolve(strict=True).relative_to(root)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Physical input differs from frozen reference: " + relative)
        checked[relative] = expected["sha256"]
    if fixed_partition:
        path = root / "20_partition/partition_assignment.csv"
        path.resolve(strict=True).relative_to(root)
        if path.is_symlink() or sha256(path) != record["partition_assignment_sha256"]:
            raise ValueError("Fixed placement partition differs from frozen reference")
        checked["20_partition/partition_assignment.csv"] = record["partition_assignment_sha256"]
    return checked


def validate_reference_root(root, case):
    root = Path(root).expanduser().resolve(strict=True)
    index = root / "reference.json"
    expected = identity()
    if index.is_symlink() or sha256(index) != expected["reference_json_sha256"]:
        raise ValueError("Frozen reference index identity differs")
    content = json.loads(index.read_text())
    if content.get("reference_id") != expected["reference_id"]:
        raise ValueError("Frozen reference ID differs")
    run = root / expected["cases"][case]["run"]
    run.resolve(strict=True).relative_to(root)
    validate_physical_prefix(run, case, fixed_partition=True)
    return run


def validate_reference_bundle(bundle, case, *, fixed_partition=False, audit_framework_identity=None):
    from .benchmark_input import validate_benchmark_input
    result = validate_benchmark_input(Path(bundle))
    expected = profile_path(case) / "config.yaml"
    if audit_framework_identity is not None:
        expected = validate_audit_profile(case, audit_framework_identity) / "config.yaml"
    if Path(result["config"]).resolve() != expected.resolve():
        raise ValueError("Input bundle is not the versioned reference-aligned profile")
    checked = validate_physical_prefix(result["run_dir"], case, fixed_partition=fixed_partition)
    manifest_path = Path(bundle) / "benchmark_input.json" if Path(bundle).is_dir() else Path(bundle)
    manifest = json.loads(manifest_path.read_text())
    reference_root = manifest.get("reference_root")
    if not isinstance(reference_root, str):
        raise ValueError("Reference-aligned bundle must identify its read-only archive")
    validate_reference_root(reference_root, case)
    archive = json.loads((Path(reference_root) / "reference.json").read_text())
    return {"baseline_id": BASELINE_ID, "reference_id": identity()["reference_id"],
            "reference_root": str(Path(reference_root).resolve()),
            "original_root": archive["original_root"],
            "physical_input_sha256": checked, "fixed_partition": fixed_partition,
            "scoring_reference_qualified": False,
            "interpretation": "Frozen physical inputs verified; each endpoint still requires qualification."}


def validate_reference_config(config, case):
    """Validate the initial recipe, not the agent's legitimate frontend edits."""
    expected = yaml.safe_load((profile_path(case) / "config.yaml").read_text())
    if config != expected:
        raise ValueError("Initial configuration differs from the versioned reference recipe")
    return {"baseline_id": BASELINE_ID, "config_sha256": sha256(profile_path(case) / "config.yaml")}
