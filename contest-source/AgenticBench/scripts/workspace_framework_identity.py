"""Explicit audit-only framework/physical-source compatibility; no execution.

The caller pins both clean commits. The complete Git trees must match outside
the enumerated, reviewed audit/control changes below. This is not permission
to migrate algorithms, task contracts, scoring, physical configuration or data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


AUDIT_ONLY_PATHS = frozenset({
    "AgenticBench/local_bench/reference_evaluated.py",
    "AgenticBench/local_bench/reference_baseline.py",
    "AgenticBench/scripts/codesign_runtime.py",
    "AgenticBench/scripts/run_global_task_smoke.py",
    "AgenticBench/scripts/workspace_reference.py",
    "AgenticBench/scripts/workspace_codesign_evaluator.py",
    "AgenticBench/scripts/workspace_framework_identity.py",
    "AgenticBench/scripts/requalify_campaign_reference.py",
    "AgenticBench/tests/test_workspace_framework_identity.py",
    "AgenticBench/tests/test_reference_evaluated.py",
    "AgenticBench/tests/test_global_task_smoke.py",
    "AgenticBench/tests/test_workspace_codesign_evaluator.py",
    "AgenticBench/tests/test_requalify_campaign_reference.py",
})


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args],
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).decode()


def _identity(root, revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Audit split requires both full, explicitly pinned commit identities")
    root = Path(root).expanduser().absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("Audit source paths must not traverse symlinks")
    root = root.resolve(strict=True)
    if Path(_git(root, "rev-parse", "--show-toplevel").strip()) != root:
        raise ValueError("Audit source must be a complete checkout root")
    if _git(root, "rev-parse", "HEAD").strip() != revision:
        raise ValueError("Audit/computation checkout differs from its pinned commit")
    if _git(root, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none").strip():
        raise ValueError("Audit/computation checkout must remain completely clean")
    return root


def _tree(root):
    rows = {}
    for entry in _git(root, "ls-tree", "-rz", "--full-tree", "HEAD").split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            mode, kind, oid = metadata.split()
            rows[path] = {"mode": mode, "kind": kind, "git_object": oid}
    return rows


def validate_audit_framework(source, framework, *, source_revision, framework_revision):
    """Return an identity receipt without modifying either checkout or a run."""
    source = _identity(source, source_revision)
    framework = Path(framework).expanduser().absolute()
    if framework.name != "AgenticBench":
        raise ValueError("Trusted framework must name its AgenticBench directory")
    audit = _identity(framework.parent, framework_revision)
    if source == audit or source in audit.parents or audit in source.parents:
        raise ValueError("An explicit audit split needs disjoint source checkouts")
    before, after = _tree(source), _tree(audit)
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    if not changed or not set(changed) <= AUDIT_ONLY_PATHS:
        raise ValueError("Audit framework changes unapproved computation/task/scoring files: " +
                         ", ".join(sorted(set(changed) - AUDIT_ONLY_PATHS)))
    for path in changed:
        if path not in after or after[path]["kind"] != "blob" or after[path]["mode"] not in {"100644", "100755"}:
            raise ValueError("Audit-only changes cannot delete files or introduce symlinks/submodules")
        if path in before and before[path]["mode"] != after[path]["mode"]:
            raise ValueError("Audit-only changes cannot change tracked file modes")
    unchanged = {path: row for path, row in before.items() if path not in AUDIT_ONLY_PATHS}
    canonical = json.dumps(unchanged, sort_keys=True, separators=(",", ":")).encode()
    return {"schema_version": 1, "kind": "audit-framework-computation-source-split-v1",
        "computation_source": str(source), "computation_source_revision": source_revision,
        "trusted_framework": str(framework), "trusted_framework_revision": framework_revision,
        "clean": True, "unchanged_tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "unchanged_tree_entries": len(unchanged),
        "reviewed_audit_changes": {path: {"computation": before.get(path), "audit": after[path]} for path in changed},
        "compatibility_scope": "All tracked paths outside the exact audit/control/test allowlist are identical, including tasks, scoring, algorithms, physical profiles and dependency pins.",
        "physical_producer_revision": source_revision, "native_dispatches": 0, "model_calls": 0}


def revalidate_runtime_framework(runtime):
    latest = getattr(runtime, "latest_framework_identity", None)
    if latest is not None:
        current = validate_latest_framework(runtime.source, runtime.framework,
            expected_sha256=latest["framework_sha256"])
        if current != latest:
            raise ValueError("Latest framework/computation source changed after admission")
        return None  # Not the historical audit-only split contract.
    receipt = getattr(runtime, "audit_framework_identity", None)
    if receipt is None:
        return None
    current = validate_audit_framework(runtime.source, runtime.framework,
        source_revision=runtime.source_commit,
        framework_revision=receipt["trusted_framework_revision"])
    if current != receipt:
        raise ValueError("Audit framework/computation source identity changed after admission")
    return current


def latest_framework_digest(framework):
    """Hash the explicitly supplied control plane; never hash credentials/runs."""
    framework = Path(framework).resolve(strict=True)
    if framework.name != "AgenticBench":
        raise ValueError("Latest framework root must be AgenticBench")
    rows = []
    for path in sorted(framework.rglob("*")):
        relative = path.relative_to(framework)
        if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("Latest control plane cannot contain symlinks")
        if path.is_file():
            rows.append([str(relative), path.stat().st_mode & 0o777,
                         hashlib.sha256(path.read_bytes()).hexdigest()])
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def validate_latest_framework(source, framework, *, expected_sha256):
    """Explicit new-family split, not a claim that two source trees are identical."""
    from local_bench.reference_latest import SOURCE_REVISION, NATIVE_REVISION, STA_REVISION
    source = _identity(source, SOURCE_REVISION)
    framework = Path(framework).absolute()
    if any(path.is_symlink() for path in (framework, *framework.parents)):
        raise ValueError("Latest framework path must be non-symlink")
    framework = framework.resolve(strict=True)
    if source == framework or source in framework.parents or framework in source.parents:
        raise ValueError("Latest physical source and control plane must be separate")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Latest framework requires an explicit full SHA256 pin")
    if latest_framework_digest(framework) != expected_sha256:
        raise ValueError("Latest framework digest mismatch")
    native = source / "OpenROAD-3D/openroad"
    if _git(native, "rev-parse", "HEAD").strip() != NATIVE_REVISION:
        raise ValueError("Latest native source pin differs")
    if _git(native / "src/sta", "rev-parse", "HEAD").strip() != STA_REVISION:
        raise ValueError("Latest STA source pin differs")
    return {"kind": "latest-workspace-explicit-source-split-v1",
            "source": str(source), "source_revision": SOURCE_REVISION,
            "framework": str(framework), "framework_sha256": expected_sha256,
            "native_revision": NATIVE_REVISION, "sta_revision": STA_REVISION,
            "scope": "Pinned independent control plane; no source-equivalence claim"}
