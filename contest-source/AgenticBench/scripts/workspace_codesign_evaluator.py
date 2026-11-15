#!/usr/bin/env python3
"""Trusted evaluation of an editable source checkout, not a program candidate.

The source snapshot is derived from Git, never from an agent's change list.
Candidate frontend code runs in a separate container from the fixed backend
and verifier. Existing input manifests are copied verbatim; a read-only source
alias preserves their relative paths instead of rewriting their provenance.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Scheduler workers start in an isolated job directory, not the checkout root.
# Import the trusted sibling flow from this pinned evaluator snapshot; never
# depend on cwd/PYTHONPATH or import the candidate's editable implementation.
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import yaml

from global_bench.evaluation import validate_trusted_stage
from global_bench.submission import ArtifactRecord, ChangeRecord, SubmissionManifest
from global_bench.verifier import verify_submission_structure
from local_bench.benchmark_input import validate_benchmark_input
from local_bench.workspace import apply_candidate_patch, isolated_clone
from scripts.workspace_task_selection import TIMING_TEN_TASK, LATEST_TASKS, LATEST_PROFILE, LATEST_BASELINE, validate_selected_task
from scripts.workspace_task_selection import LATEST_PROFILES, LATEST_THERMAL_PROFILE, LATEST_THERMAL_TASKS

CANDIDATE_CONFIG = "agentic_candidate.yaml"
DREAM_PREFIX = "Place-3D/thirdparty/DREAMPlace/dreamplace/"
PREFIX = {"input": "00_input", "floorplan": "10_floorplan"}
FRONTEND = {**PREFIX, "timing_extract": "15_timing_extract",
            "partition": "20_partition", "placement": "30_placement"}
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".so", ".a"}
FLOW_COMMAND_PHASES = {"frontend", "backend", "thermal"}
# These are distinct task endpoints, not interchangeable thermal aliases. The
# legacy routed task retains native one-pass OPT and the full routed power view.
LEGACY_ENDPOINTS = {
    "timing": {
        "task_id": "lol_partition_placement_timing", "profile": "tp_gnn_placement_two_pass",
        "kind": "sta_wns_tns", "opt_passes": 2, "backend_stop": "opt",
        "checkpoint": "60_opt/checkpoint.json", "metrics": "60_opt/metrics.json",
        "verifier_mode": "timing",
    },
    "thermal": {
        "task_id": "lol_partition_placement_cts_thermal", "profile": "tp_gnn_placement_cts_thermal",
        "kind": "hotspot_average_temperature", "opt_passes": 2, "backend_stop": "cts",
        "checkpoint": "70_cts/checkpoint.json", "metrics": "95_thermal_snapshot/metrics.json",
        "verifier_mode": "cts-thermal",
    },
    "routed_thermal": {
        "task_id": "lol_partition_placement_thermal", "profile": "tp_gnn_placement_thermal",
        "kind": "hotspot_average_temperature", "opt_passes": 1, "backend_stop": "thermal",
        "checkpoint": "95_thermal/checkpoint.json", "metrics": "95_thermal/metrics.json",
        "verifier_mode": "thermal",
    },
}
# New task/profile identities prevent one-pass results from silently replacing
# the historical two-pass references. The old contracts remain explicit.
ENDPOINTS = {
    **LEGACY_ENDPOINTS,
    "timing": {**LEGACY_ENDPOINTS["timing"],
               "task_id": "lol_partition_placement_timing_one_pass",
               "profile": "tp_gnn_placement_one_pass", "opt_passes": 1},
    "thermal": {**LEGACY_ENDPOINTS["thermal"],
                "task_id": "lol_partition_placement_cts_thermal_one_pass",
                "profile": "tp_gnn_placement_cts_thermal_one_pass", "opt_passes": 1},
}


class _AttemptTiming:
    """Control-plane evidence only; never modify native run artifacts."""

    def __init__(self, output: Path, subject: Path, reference_run: Path | None):
        self.output = output
        self.ledger = output.parent / (output.name + ".workspace-attempt.json")
        self.events = output.parent / (output.name + ".workspace-attempt.events.jsonl")
        self.clock = time.monotonic()
        at = time.time()
        self.value = {"schema_version": 2, "state": "started", "subject": str(subject),
                      "reference_run": str(reference_run) if reference_run else None,
                      "started_at": at, "started_at_utc": datetime.fromtimestamp(at, timezone.utc).isoformat(),
                      "command_phases": {}}
        if self.events.exists():
            raise FileExistsError(self.events)
        with self.ledger.open("x") as stream:
            json.dump(self.value, stream)
            stream.write("\n")
        # Exclusive creation plus append-only updates prevents an uncertain
        # attempt from silently replacing its earlier evidence.
        self.events.open("x").close()
        self.event("started")

    def event(self, event: str, **values) -> None:
        self.value.update(values)
        _json(self.ledger, self.value)
        at = time.time()
        with self.events.open("a") as stream:
            stream.write(json.dumps({"event": event, "at": at,
                "at_utc": datetime.fromtimestamp(at, timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - self.clock, **values},
                sort_keys=True, allow_nan=False) + "\n")

    def command(self, event: str, phase: str, receipt: dict) -> None:
        self.event(event, command_phase=phase,
                   command_phases={**self.value["command_phases"], phase: dict(receipt)})


def endpoint_for_task(objective, task):
    if objective == "thermal" and task.task_id in LATEST_THERMAL_TASKS.values():
        rounds = next(n for n, name in LATEST_THERMAL_TASKS.items() if name == task.task_id)
        validate_selected_task(task, objective, rounds)
        return {**ENDPOINTS["thermal"], "task_id": task.task_id, "profile": LATEST_THERMAL_PROFILE}
    if objective == "timing" and task.task_id in LATEST_TASKS.values():
        rounds = next(n for n, name in LATEST_TASKS.items() if name == task.task_id)
        validate_selected_task(task, objective, rounds)
        return {**ENDPOINTS["timing"], "task_id": task.task_id, "profile": LATEST_PROFILE}
    if objective == "timing" and task.task_id == TIMING_TEN_TASK:
        validate_selected_task(task, objective, 10)
        endpoint = {**ENDPOINTS["timing"], "task_id": TIMING_TEN_TASK}
        if (task.objective.kind != endpoint["kind"]
                or task.objective.evaluator_profile != endpoint["profile"]):
            raise ValueError("Task/objective identity mismatch")
        return endpoint
    for contracts in (ENDPOINTS, LEGACY_ENDPOINTS):
        endpoint = contracts.get(objective)
        if endpoint and (task.task_id == endpoint["task_id"]
                and task.objective.kind == endpoint["kind"]
                and task.objective.evaluator_profile == endpoint["profile"]):
            return endpoint
    raise ValueError("Task/objective identity mismatch")


def reference_mounts(receipt, protected):
    """Map pinned archive paths read-only without shadowing executable/control inputs."""
    root = Path(receipt["reference_root"])
    aliases = {root, Path(receipt["original_root"])}
    reserved = [Path(name) for name in ("/usr", "/opt", "/bin", "/sbin", "/lib", "/lib64",
                                       "/etc", "/tmp", "/dev", "/proc", "/sys")]
    for alias in aliases:
        if (not alias.is_absolute() or alias == Path("/") or ".." in alias.parts
                or any(token in str(alias) for token in (":", "\n", "\r"))):
            raise ValueError("Unsafe frozen reference mount alias")
        for path in [*reserved, *(Path(path) for path in protected)]:
            if alias == path or alias in path.parents or path in alias.parents:
                raise ValueError("Frozen reference mount shadows an evaluator/control path")
    return [(root, alias) for alias in sorted(aliases)]


def input_source_alias(runtime, manifest: dict, bundle: Path, output: Path,
                       *, baseline_validation: dict | None) -> Path:
    """Accept only the old batch alias or the exact verified importer checkout.

    Call only after validating the bundle's frozen artifacts. Importer identity
    is not native producer identity; it authorizes this read-only mount target,
    never reuse of a historical score or an arbitrary external source path.
    A distinct historical batch-contained alias stays compatible. Exact source
    aliases require the importer receipt even when that source is inside batch.
    """
    identity = manifest.get("environment", {}).get("open3dbench", {})
    raw = identity.get("root") if isinstance(identity, dict) else None
    if (not isinstance(raw, str) or not raw or ":" in raw
            or any(ord(char) < 32 or ord(char) == 127 for char in raw)):
        raise ValueError("Unsafe input-source mount alias")
    alias = Path(raw)
    if (not alias.is_absolute() or ".." in alias.parts or "." in raw.split("/")
            or any(path.is_symlink() for path in (alias, *alias.parents))
            or alias != alias.resolve()):
        raise ValueError("Input-source mount alias must be canonical and non-symlink")
    if alias == runtime.source:
        importer = manifest.get("current_importer_source")
        expected = {"root": str(runtime.source), "revision": runtime.source_commit, "dirty": False}
        latest = getattr(runtime, "reference_baseline", None) == LATEST_BASELINE
        if (not isinstance(baseline_validation, dict)
                or baseline_validation.get("baseline_id") != (LATEST_BASELINE if latest else "reference_20260923")
                or manifest.get("baseline_id") != (LATEST_BASELINE if latest else "reference_20260923")
                or (not latest and manifest.get("import_kind") != "frozen_reference_adoption")
                or not isinstance(importer, dict)
                or any(record.get("root") != expected["root"]
                       or record.get("revision") != expected["revision"]
                       or record.get("dirty") is not False for record in (identity, importer))):
            raise ValueError("Input-source alias differs from the validated frozen importer identity")
    else:
        alias = runtime.contained(alias)
        # A contained mount must not replace trusted source/control directories.
        audit_root = runtime.framework.parent if getattr(runtime, "audit_framework_identity", None) else runtime.framework
        for protected in (runtime.source, audit_root, runtime.batch / "locks"):
            if alias == protected or alias in protected.parents or protected in alias.parents:
                raise ValueError("Input-source mount alias overlaps evaluator/control paths")
    if alias == runtime.batch:
        raise ValueError("The shared batch root cannot be an input-source mount alias")
    for protected in (bundle, output):
        if alias == protected or alias in protected.parents or protected in alias.parents:
            raise ValueError("Input-source mount alias overlaps input/evaluation state")
    return alias


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def _git(repo: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *arguments])


def _source_git_objects(source: Path) -> Path:
    """Resolve only the frozen checkout's own store, including Git worktrees."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    def path(*arguments):
        value = Path(subprocess.check_output(["git", "-C", str(source), "rev-parse", *arguments],
            text=True, env=environment).strip())
        return value if value.is_absolute() else source / value
    objects = path("--git-path", "objects")
    common = path("--git-common-dir")
    if (any(item.is_symlink() for item in (objects, *objects.parents))
            or objects.resolve(strict=True) != (common / "objects").resolve(strict=True)):
        raise ValueError("Frozen source Git object store must be its non-symlink common directory")
    objects = objects.resolve(strict=True)
    if not objects.is_dir():
        raise ValueError("Frozen source Git object store is missing")
    for name in ("alternates", "http-alternates"):
        alternate = objects / "info" / name
        if (alternate.is_symlink() or alternate.parent.is_symlink()
                or (alternate.exists() and alternate.read_text().strip())):
            raise ValueError("Frozen source requires a self-contained Git object store")
    return objects


def _reject_shadowed_source_links(snapshot: Path, source_alias: Path) -> None:
    """Do not mount a link back into the baseline path that this mount hides."""
    for directory, subdirs, files in os.walk(snapshot, followlinks=False):
        # Clone-owned Git administration is handled by the exact objects mount.
        subdirs[:] = [name for name in subdirs if name != ".git"]
        for name in (*subdirs, *files):
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            target = path.resolve()
            if target == source_alias or source_alias in target.parents:
                raise ValueError("Snapshot symlink points into the shadowed frozen source: " + str(path))


def source_changes(subject: Path, revision: str, protected: tuple[str, ...]) -> list[dict]:
    """Include edits, additions and deletions, with no file/count/line allowlist."""
    if _git(subject, "rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("Subject Git baseline changed")
    raw = _git(subject, "diff", "--no-ext-diff", "--name-only", "--no-renames", "-z", revision, "--")
    raw += _git(subject, "ls-files", "--others", "--exclude-standard", "-z")
    paths = sorted({os.fsdecode(item) for item in raw.split(b"\0") if item})
    changes = []
    for name in paths:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise ValueError("Unsafe source path: " + name)
        if any(name == prefix or name.startswith(prefix.rstrip("/") + "/") for prefix in protected):
            raise ValueError("Protected benchmark path changed: " + name)
        path = subject / name
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != subject and subject in parent.parents):
            raise ValueError("Changed source must not traverse a symlink: " + name)
        if path.exists() and not path.is_file():
            raise ValueError("Changed source is not a regular file: " + name)
        changes.append({"path": name, "status": "M" if path.exists() else "D",
                        "sha256": _sha(path) if path.exists() else None,
                        "executable": bool(path.stat().st_mode & stat.S_IXUSR) if path.exists() else False})
    return changes


def _no_seed_override(value: Any, prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "seed" in str(key).lower():
                raise ValueError("Seeds are benchmark-owned, not tunable: " + prefix + str(key))
            _no_seed_override(child, prefix + str(key) + ".")
    elif isinstance(value, list):
        for child in value:
            _no_seed_override(child, prefix)


def _merge(base: dict, extra: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in extra.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def resolved_config(base: dict, candidate: dict, task, cpus: int) -> dict:
    if not isinstance(candidate, dict) or set(candidate) - {"partition", "placement"}:
        raise ValueError("agentic_candidate.yaml accepts only partition and placement mappings")
    if any(not isinstance(value, dict) for value in candidate.values()):
        raise ValueError("Frontend configuration domains must be mappings")
    _no_seed_override(candidate)
    # These are the native runner's public selectors, not an algorithm-parameter
    # scope. Arbitrary method parameters belong in the editable JSON file selected
    # by `config`; the protected runner otherwise silently ignores unknown keys.
    public_keys = {
        "partition": {"algorithm", "config"},
        "placement": {"algorithm", "config", "target_density", "macro_halo_um", "gpu",
                      "execution", "solution", "export_manifest", "placed_def", "identity"},
    }
    for domain, values in candidate.items():
        unknown = set(values) - public_keys[domain]
        if unknown:
            raise ValueError("Unsupported native {} fields {}; put method parameters in the "
                             "JSON selected by {}.config".format(domain, sorted(unknown), domain))
    result = _merge(base, candidate)
    if "macro_halo_um" in candidate.get("placement", {}):
        fixed_halo = base["placement"].get("macro_halo_um", {"ng45": 10.0, "asap7": 2.0}[base["pdk"]])
        if candidate["placement"]["macro_halo_um"] != fixed_halo:
            raise ValueError("placement.macro_halo_um is a fixed physical-spacing contract")
    # Explicit input aliases are data, not tunable algorithm parameters.
    for domain in ("partition", "placement"):
        for key in ("timing_input", "assignment", "partition_dir", "input", "input_dir", "input_file", "checkpoint"):
            if key in candidate.get(domain, {}):
                raise ValueError("Frozen inputs cannot be overridden: " + domain + "." + key)
        config_path = result[domain].get("config")
        if config_path is not None:
            path = PurePosixPath(str(config_path))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Frontend config must be a source-relative path")
    for domain, seed in (("pins", task.seed_policy.io_placement),
                         ("partition", task.seed_policy.partition),
                         ("placement", task.seed_policy.placement),
                         ("routing", task.seed_policy.routing)):
        result.setdefault(domain, {})["seed"] = seed
    # Keep the physical profile's thread settings fixed. Docker's CPU quota is
    # scheduling capacity, not an unrecorded change to the algorithm contract.
    return result


def bind_frontend_config_paths(config: dict, snapshot: Path, *, latest_framework: Path | None = None) -> tuple[dict, dict]:
    """Bind algorithm JSON data to the same immutable files in both phases.

    Native Place-3D paths are relative to ``Place-3D/``, while an acting
    workspace agent also sees repository-relative paths. Resolve either
    spelling once, then use the snapshot's read-only absolute path. The
    pristine backend still calls FlowRunner.plan(), which reads TP-GNN config
    even when execution starts at PDN. It must not accidentally use the
    baseline JSON or fail because a candidate added a new JSON file.
    """
    # Public adapter facts for the supported frozen flow, not an import of
    # candidate or external implementation modules.
    PARTITION_CONFIGS = {"hmetis": "config/partition/hmetis.json", "mtkahypar": "config/partition/mtkahypar.json",
                        "tp-gnn-flat": "config/partition/tp_gnn_fast.json", "pyg-mincut": "config/partition/pyg_unsupervised.json",
                        "pyg-maxcut": "config/partition/pyg_maxcut.json"}

    result = copy.deepcopy(config)
    root = snapshot.resolve(strict=True)
    bindings = {}
    for domain in ("partition", "placement"):
        value = result[domain].get("config")
        origin = "explicit"
        if value is None and domain == "partition":
            algorithm = result[domain].get("algorithm")
            # The native loader accepts this historical public alias.
            algorithm = "tp-gnn-flat" if algorithm == "tp-gnn" else algorithm
            value = PARTITION_CONFIGS.get(algorithm)
            origin = "native-default"
        if value is None:
            continue
        relative = PurePosixPath(str(value))
        if not str(value) or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Frontend config must be a source-relative path")
        trusted_relative = "AgenticBench/local_bench/reference_configs/" + domain + ".json"
        if latest_framework is not None and str(relative) == trusted_relative:
            path = Path(latest_framework) / "local_bench/reference_configs" / (domain + ".json")
            if path.is_symlink() or not path.is_file() or not isinstance(json.loads(path.read_text()), dict):
                raise ValueError("Latest trusted algorithm recipe is missing or invalid")
            result[domain]["config"] = str(path)
            bindings[domain] = {"requested": str(value), "origin": "trusted-latest-framework",
                                "source_relative": str(relative), "resolved": str(path), "sha256": _sha(path)}
            continue
        candidates = [root / relative, root / "Place-3D" / relative]
        matches = []
        for path in candidates:
            if not path.is_file():
                continue
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                        if parent != root and root in parent.parents):
                raise ValueError("Frontend config must not traverse a symlink: " + str(value))
            path.resolve().relative_to(root)
            matches.append(path)
        if len(matches) != 1:
            raise ValueError("Frontend config must resolve to exactly one snapshot file: " + str(value))
        path = matches[0]
        # Validate data syntax without importing candidate code. The native
        # method retains ownership of its own detailed parameter schema.
        if not isinstance(json.loads(path.read_text()), dict):
            raise ValueError("Frontend config must contain a JSON object: " + str(value))
        result[domain]["config"] = str(path)
        bindings[domain] = {"requested": str(value), "origin": origin,
                            "source_relative": str(path.relative_to(root)),
                            "resolved": str(path), "sha256": _sha(path)}
    return result, bindings


def copy_frozen_prefix(baseline: Path, target: Path) -> None:
    """Preserve original manifests/parameter bytes, including source provenance."""
    target.mkdir()
    status = _read(baseline / "run.json")
    stages = {}
    for stage, directory in PREFIX.items():
        validate_trusted_stage(baseline / directory, stage)
        record = status.get("stages", {}).get(stage)
        if not isinstance(record, dict) or record.get("status") != "success":
            raise ValueError("Frozen prefix has no successful root record: " + stage)
        shutil.copytree(baseline / directory, target / directory, symlinks=False, copy_function=shutil.copyfile)
        validate_trusted_stage(target / directory, stage)
        stages[stage] = copy.deepcopy(record)
    _json(target / "run.json", {**status, "status": "running", "stages": stages})


def validate_frontend_input_boundary(run: Path) -> dict:
    """Check copied benchmark contracts independently of candidate-authored logs."""
    floorplan, placement = run / "10_floorplan", run / "30_placement"
    checked = {}
    pairs = [(floorplan / name, placement / name) for name in
             ("design.v", "design.sdc", "floorplan.json", "io_plan.csv", "preplan_manifest.json", "pdn_plan.json")]
    pairs += [(floorplan / "design.sdc", placement / "die_init" / die / "design.sdc")
              for die in ("bottom", "top")]
    pairs.append((run / "20_partition/partition_assignment.csv", placement / "partition_assignment.csv"))
    for source, candidate in pairs:
        if candidate.is_symlink() or not source.is_file() or not candidate.is_file() or _sha(source) != _sha(candidate):
            raise ValueError("Frontend changed a fixed input/clock/partition handoff: " + str(candidate))
        checked[str(candidate.relative_to(run))] = _sha(candidate)
    return checked


class WorkspaceEvaluator:
    """Use Runtime's resource reservation, but never its focused candidate API."""

    def __init__(self, runtime, case: str, objective: str, bundle: Path, task):
        if objective not in ENDPOINTS:
            raise ValueError("Unknown workspace co-design objective: " + objective)
        self.endpoint = endpoint_for_task(objective, task)
        self.latest = self.endpoint["profile"] in LATEST_PROFILES
        self.frontend = {k: v for k, v in FRONTEND.items() if not self.latest or k != "timing_extract"}
        if self.latest:
            from local_bench.reference_latest import validate_reference_image
            if getattr(runtime, "reference_baseline", None) != LATEST_BASELINE:
                raise ValueError("Latest task requires the explicit latest runtime contract")
            validate_reference_image(runtime.image)
        if self.endpoint["profile"] in {"tp_gnn_placement_one_pass", "tp_gnn_placement_cts_thermal_one_pass"}:
            runtime.reference_baseline = "reference_20260923"
        if getattr(runtime, "reference_baseline", None) == "reference_20260923":
            from local_bench.reference_baseline import validate_reference_image
            validate_reference_image(runtime.image)
        self.reference_mount = None
        self.sealed_reference = None
        self.sealed_reference_mounts = ()
        self.runtime, self.case, self.objective, self.task = runtime, case, objective, task
        self.bundle = runtime.contained(bundle)
        self.base = yaml.safe_load((runtime.profile(case) / "config.yaml").read_text())
        if self.base.get("design") != case:
            raise ValueError("Profile/case identity mismatch")
        if self.latest:
            from local_bench.reference_latest import validate_reference_config
            validate_reference_config(self.base, case)
        if getattr(runtime, "reference_baseline", None) == "reference_20260923":
            from local_bench.reference_baseline import validate_reference_config
            validate_reference_config(self.base, case)
        self.patch = (runtime.framework / "local_bench/evaluator_patches/tp_gnn_placement_sta.patch"
                      if self.endpoint["opt_passes"] == 2 else None)
        if objective == "routed_thermal":
            # Identical to the legacy profile's evaluator-owned suffix. Do not
            # inherit thermal.enabled=false from the shared STA frontend profile.
            self.base["thermal"] = {"enabled": True, "grid_size": 10, "ambient_k": 318.15}

    def initial_candidate_config(self) -> dict:
        return {domain: {key: value for key, value in self.base[domain].items() if "seed" not in key.lower()}
                for domain in ("partition", "placement")}

    def _qualified_reference(self, reference_run):
        """Recheck the pinned authority before any candidate native dispatch."""
        record_path = getattr(self.runtime, "reference_record", None)
        if record_path is None:
            return self.runtime.contained(reference_run)
        if self.latest:
            from scripts.workspace_latest_reference import qualify_producer, KIND, THERMAL_KIND
            KIND = THERMAL_KIND if self.objective == "thermal" else KIND
        else:
            from local_bench.reference_catalog import qualify_producer
            KIND = "native-evaluation-v1"
        from local_bench.reference_store import file_sha, load_reference
        if file_sha(Path(record_path)) != self.runtime.reference_record_sha256:
            raise ValueError("Selected reference record changed after run configuration was pinned")
        record = load_reference(record_path, contract=self.runtime.reference_contract,
                                validate=qualify_producer)
        if record["producer"].get("kind") != KIND:
            raise ValueError("Unsupported workspace reference source composition")
        source_run = Path(record["qualification"]["baseline"]["source_run"]).resolve(strict=True)
        if Path(reference_run).resolve(strict=True) != source_run:
            raise ValueError("Candidate reference differs from the pinned sealed record")
        self.sealed_reference = record
        mounts, seen = set(), set()
        def collect(path, selected):
            path = Path(path).resolve(strict=True)
            if path in seen or len(seen) >= 8:
                raise ValueError("Invalid sealed reference lineage")
            seen.add(path)
            mounts.add(path)
            producer = selected["producer"]
            for key in ("evaluation_root", "reference_root", "thermal_root", "run_root", "benchmark_input"):
                if producer.get(key):
                    mounts.add(Path(producer[key]).resolve(strict=True))
            if self.latest:
                # Only hash-validated evidence paths from the trusted qualifier.
                for row in selected["qualification"]["evidence"]:
                    evidence_path = Path(row["path"]).resolve(strict=True)
                    if not any(root == evidence_path or root in evidence_path.parents for root in mounts):
                        mounts.add(evidence_path)
            if producer.get("parent_record"):
                parent_path = Path(producer["parent_record"])
                parent_raw = _read(parent_path)
                parent = load_reference(parent_path, contract=parent_raw["contract"], validate=qualify_producer)
                if parent["reference_id"] != producer["parent_reference_id"]:
                    raise ValueError("Sealed reference parent identity changed")
                collect(parent_path, parent)
        collect(record_path, record)
        if file_sha(Path(record_path)) != self.runtime.reference_record_sha256:
            raise ValueError("Selected reference record changed during candidate requalification")
        self.sealed_reference_mounts = tuple(sorted(mounts))
        return source_run

    def initialize_subject(self, subject: Path) -> Path:
        path = Path(subject) / CANDIDATE_CONFIG
        if path.exists():
            raise FileExistsError(path)
        candidate = self.initial_candidate_config()
        if self.latest:
            # Empty mount point only; the separate trusted framework is mounted
            # read-only for tools, never copied into the physical Git baseline.
            (Path(subject) / "AgenticBench").mkdir(exist_ok=False)
            settings = Path(subject) / "agentic_reference_configs"
            settings.mkdir(exist_ok=False)
            for domain in ("partition", "placement"):
                source = self.runtime.framework / "local_bench/reference_configs" / (domain + ".json")
                target = settings / source.name
                shutil.copyfile(source, target)
                if _sha(source) != _sha(target):
                    raise ValueError("Initial latest settings copy differs")
                candidate[domain]["config"] = str(target.relative_to(subject))
        path.write_text(yaml.safe_dump(candidate, sort_keys=False))
        return path

    write_initial_candidate_config = initialize_subject

    def _snapshot(self, subject: Path, output: Path, changes: list[dict]) -> Path:
        snapshot = output / "source"
        isolated_clone(self.runtime.source, snapshot)
        for row in changes:
            source, target = subject / row["path"], snapshot / row["path"]
            if target.is_symlink():
                raise ValueError("Candidate cannot replace a source symlink")
            if row["sha256"] is None:
                target.unlink(missing_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o755 if row["executable"] else 0o644)
            if _sha(target) != row["sha256"]:
                raise ValueError("Source changed while being snapshotted: " + row["path"])
        # Benchmark-owned patch, separate from the candidate diff and hash list.
        if self.patch is not None:
            apply_candidate_patch(snapshot, self.patch)
        return snapshot

    def _docker(self, command: list[str], gpu: int, output: Path, snapshot: Path,
                source_alias: Path, *, writable: tuple[Path, ...], readonly: tuple[Path, ...] = (),
                dreamplace: Path | None = None, name: str) -> list[str]:
        if source_alias == self.runtime.source:
            _reject_shadowed_source_links(snapshot, source_alias)
        scratch = output / (name + "-scratch")
        scratch.mkdir()
        command_line = ["docker", "run", "--rm", "--name", name,
                        "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                        "--network", "none", "--gpus", "device=" + str(gpu),
                        "--cpus", str(self.runtime.cpus), "--memory", str(self.runtime.memory_gb) + "g",
                        "--memory-swap", str(self.runtime.memory_gb) + "g",
                        "--pids-limit", str(self.task.environment.pids_limit),
                        "--user", "{}:{}".format(os.getuid(), os.getgid()), "--shm-size", "8g"]
        targets = {}
        def mount(source, target, mode):
            source, target = Path(source), Path(target)
            if any(not path.is_absolute() or ".." in path.parts or ":" in str(path)
                   or any(ord(char) < 32 or ord(char) == 127 for char in str(path))
                   for path in (source, target)):
                raise ValueError("Unsafe evaluator mount path")
            value = (source, mode)
            if target in targets:
                if targets[target] != value:
                    raise ValueError("Conflicting evaluator mount target: " + str(target))
                return
            targets[target] = value
            command_line.extend(["-v", f"{source}:{target}:{mode}"])
        mount(self.runtime.batch, self.runtime.batch, "ro")
        if source_alias != self.runtime.source:
            mount(self.runtime.source, self.runtime.source, "ro")
        mount(snapshot, source_alias, "ro")
        # Trusted audit code may be a separately pinned, audit-only checkout.
        # Mount it independently for both supported input-source alias forms.
        audit_root = (self.runtime.framework.parent if getattr(self.runtime, "audit_framework_identity", None)
                      else self.runtime.framework)
        mount(audit_root, audit_root, "ro")
        if source_alias == self.runtime.source:
            # The importer uses this exact path as its alias. Do not also mount
            # the baseline checkout at the same target (Docker rejects it).
            # Keep trusted control-plane imports independent of the candidate.
            # isolated_clone uses --shared. Its Git alternates name this exact
            # baseline object store, which the source-alias mount can otherwise
            # hide. Expose only objects, never the baseline HEAD/index/refs, so
            # native Git provenance still describes the evaluated snapshot.
            # Resolve worktree administration through trusted Git metadata, not
            # through the input manifest. Do not expand external alternate chains.
            objects = _source_git_objects(self.runtime.source)
            mount(objects, objects, "ro")
        mount(scratch, Path("/tmp"), "rw")
        command_line += ["-w", str(source_alias)]
        for path in writable:
            mount(path, path, "rw")
        for path in readonly:
            mount(path, path, "ro")
        if self.sealed_reference is not None:
            # A reused native reference can live in an earlier batch. Mount the
            # complete qualified evidence root and sealed record read-only.
            for path in self.sealed_reference_mounts:
                for target in (output, snapshot, source_alias, *writable):
                    target = Path(target).resolve()
                    if path == target or path in target.parents or target in path.parents:
                        raise ValueError("Sealed reference mount overlaps candidate/control output")
                mount(path, path, "ro")
        if self.reference_mount is not None:
            protected = (self.runtime.batch, self.runtime.source, snapshot, source_alias,
                         output, audit_root, *writable, *readonly)
            for source, alias in reference_mounts(self.reference_mount, protected):
                mount(source, alias, "ro")
        install = str(dreamplace or Path("/opt/dreamplace"))
        physical_threads = (self.base["threads"] if getattr(self.runtime, "reference_baseline", None)
                            in {"reference_20260923", LATEST_BASELINE} else self.runtime.cpus)
        environment = {
            "HOME": "/tmp", "TMPDIR": "/tmp", "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{self.runtime.framework}:{source_alias}:{install}:{install}/dreamplace",
            "OPENROAD_EXE": "/usr/local/bin/openroad", "OPEN3DBENCH_PYTHON": "/usr/bin/python3",
            "OPEN3DBENCH_HOTSPOT_ROOT": "/opt/open3dbench-assets/hotspot",
            "PLACE3D_DREAMPLACE_INSTALL": install, "OPEN3D_LOCAL_DREAMPLACE_INSTALL": "/opt/dreamplace",
            "OPEN3D_OPT_PASSES": str(self.endpoint["opt_passes"]), "OPEN3DBENCH_IMAGE": self.runtime.image,
            "OPEN3DBENCH_IMAGE_ID": self.runtime.image,
            "OMP_NUM_THREADS": str(physical_threads), "MKL_NUM_THREADS": str(physical_threads),
        }
        for key, value in environment.items():
            command_line += ["-e", key + "=" + value]
        from scripts.workspace_runtime_image import execution_image
        return [*command_line, execution_image(self.runtime), *command]

    def _thermal_command(self, source_alias, run, config):
        if self.latest:
            return ["/usr/bin/python3", "-m", "scripts.workspace_cts_thermal",
                "--checkpoint", str(run / "70_cts/checkpoint.json"),
                "--platform-dir", str(source_alias / "OpenROAD-3D/flow/platforms/nangate45_3D"),
                "--hotspot-root", "/opt/open3dbench-assets/hotspot",
                "--output-dir", str(run / "95_thermal_snapshot")]
        from local_bench.cts_thermal import snapshot_command
        return snapshot_command(source_alias, run, config, "/usr/bin/python3",
            {"OPEN3DBENCH_HOTSPOT_ROOT": "/opt/open3dbench-assets/hotspot"},
            power_threads=(2 if getattr(self.runtime, "reference_baseline", None)
                           == "reference_20260923" else 32))

    def _call(self, phase: str, argv: list[str], output: Path, deadline: float) -> None:
        receipt_path = output / (phase + "-command-receipt.json")
        receipt = {"schema_version": 1, "phase": phase, "state": "preparing",
                   "scope": "native-flow-command" if phase in FLOW_COMMAND_PHASES else "control-command",
                   "dispatch_attempted": False, "process_started": False,
                   "process_wall_seconds": None, "cleanup_wall_seconds": None,
                   "native_execution_confirmed": None,
                   "interpretation": "Client dispatch does not prove that OpenROAD or another native solver started."}
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream)
        attempt = getattr(self, "_attempt_timing", None)
        if attempt is not None and attempt.output != output:
            attempt = None

        def record(event, **values):
            receipt.update(values)
            _json(receipt_path, receipt)
            if attempt is not None:
                attempt.command(event, phase, receipt)

        process_clock = None
        try:
            _json(output / (phase + "-command.json"), argv)
            with (output / (phase + ".log")).open("w") as stream:
                process_clock = time.monotonic()
                at = time.time()
                record("command-dispatch", state="dispatched", dispatch_attempted=True,
                       process_started=None, dispatched_at=at,
                       dispatched_at_utc=datetime.fromtimestamp(at, timezone.utc).isoformat())
                result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=max(1, deadline - time.monotonic()))
            record("command-exited", state="exited", process_started=True,
                   exit_code=result.returncode, finished_at=time.time(),
                   process_wall_seconds=time.monotonic() - process_clock)
            if result.returncode:
                raise RuntimeError(phase + " failed; inspect " + str(output / (phase + ".log")))
        except Exception as error:
            if process_clock is not None and receipt["process_wall_seconds"] is None:
                receipt["process_wall_seconds"] = time.monotonic() - process_clock
            record("command-failed", state="failed", finished_at=time.time(),
                   error=f"{type(error).__name__}: {error}")
            if isinstance(error, subprocess.TimeoutExpired):
                # Only this uniquely named evaluator container is stopped. A
                # dead Docker client must not leave a writer behind at timeout.
                name = argv[argv.index("--name") + 1]
                cleanup_clock = time.monotonic()
                try:
                    cleanup = subprocess.run(["docker", "stop", "--time", "5", name],
                                             capture_output=True, timeout=20)
                    receipt["cleanup_exit_code"] = cleanup.returncode
                except Exception as cleanup_error:
                    receipt["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                    raise
                finally:
                    record("command-timeout-cleanup", cleanup_wall_seconds=time.monotonic() - cleanup_clock)
                raise RuntimeError(phase + " exceeded the evaluation walltime") from error
            raise

    def _submission(self, output: Path, changes: list[dict], subject: Path) -> Path:
        root = output / "submission"
        root.mkdir()
        patch = root / "patch.diff"
        patch.write_bytes(_git(subject, "diff", "--no-ext-diff", "--binary", "--no-renames", self.runtime.source_commit, "--"))
        source_records = []
        for row in changes:
            relative = "source_changes/" + row["path"] if row["sha256"] else None
            if relative:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(output / "source" / row["path"], target)
            source_records.append(ChangeRecord(row["status"], row["path"], row["sha256"], relative))
        run = output / "run"
        files = {"run_manifest": run / "run.json",
                 "final_checkpoint": run / self.endpoint["checkpoint"],
                 "metrics": run / self.endpoint["metrics"]}
        artifacts = []
        for name, source in files.items():
            relative = "artifacts/" + name + ".json"
            target = root / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            artifacts.append(ArtifactRecord(name, str(source), relative, target.stat().st_size, _sha(target)))
        manifest = SubmissionManifest(self.task.task_id, self.task.digest(), output.name,
                                      self.runtime.source_commit, "patch.diff", _sha(patch),
                                      tuple(source_records), tuple(artifacts), str(time.time()))
        _json(root / "submission.json", manifest.to_dict())
        structural = verify_submission_structure(self.task, root)
        _json(output / "structural-verification.json", structural.to_dict())
        if not structural.passed:
            raise ValueError("Submission integrity rejected: " + "; ".join(structural.errors))
        return root

    def evaluate(self, subject: Path, output: Path, reference_run: Path | None = None,
                 *, parent_reference: dict | None = None) -> dict:
        self._attempt_timing = None
        subject, output = Path(subject).resolve(), self.runtime.contained(output)
        # Exclusive creation is the durable at-most-once boundary, including
        # validation failures. No cached score can silently consume a new trial.
        output.parent.mkdir(parents=True, exist_ok=True)
        attempt = _AttemptTiming(output, subject, reference_run)
        self._attempt_timing = attempt
        if output.exists():
            raise FileExistsError(output)
        output.mkdir()
        result = {"success": False, "metrics": {}, "run": str(output / "run"),
                  "verification": None, "source_snapshot": str(output / "source"),
                  "changed_files": [], "error": None}
        timing = {"preparation_wall_seconds": None, "admission_wait_seconds": None}
        phase, phase_clock = "preparation", attempt.clock
        try:
            from scripts.workspace_framework_identity import revalidate_runtime_framework
            audit_identity = revalidate_runtime_framework(self.runtime)
            if self.latest and (reference_run is None or parent_reference is not None):
                raise ValueError("Latest global evaluation requires ADOPT; no implicit reference generation")
            if audit_identity is not None and reference_run is None:
                raise ValueError("Audit-only framework split requires an adopted reference; no FRESH/EXTEND dispatch")
            extension = None
            if parent_reference is not None:
                if reference_run is not None or subject != self.runtime.source:
                    raise ValueError("EXTEND is a reference-purpose suffix, never a candidate evaluation")
                from scripts.workspace_reference_extension import qualify_parent
                extension = qualify_parent(parent_reference["record"], parent_reference["target_contract"],
                                           record_sha256=parent_reference["record_sha256"])
            reference = self._qualified_reference(reference_run) if reference_run else None
            validated = validate_benchmark_input(self.bundle)
            baseline_validation = None
            if self.latest:
                from local_bench.reference_latest import validate_reference_bundle
                baseline_validation = validate_reference_bundle(self.bundle, self.case)
            if getattr(self.runtime, "reference_baseline", None) == "reference_20260923":
                from local_bench.reference_baseline import validate_reference_bundle
                baseline_validation = validate_reference_bundle(self.bundle, self.case,
                    **({"audit_framework_identity": audit_identity} if audit_identity is not None else {}))
                self.reference_mount = baseline_validation
            manifest = _read(self.bundle / "benchmark_input.json")
            if extension is not None:
                from local_bench.reference_catalog import evaluator_digest, task_digest
                from local_bench.reference_evaluated import expected_frozen_inputs, expected_source_contract
                endpoint_name = {"timing": "timing", "thermal": "cts_thermal", "routed_thermal": "routed_thermal"}[self.objective]
                actual_contract = {"case": self.case, "endpoint": endpoint_name,
                    "task_digest": task_digest(self.task.root), "evaluator_digest": evaluator_digest(),
                    "frozen_inputs": expected_frozen_inputs(self.case, _sha(self.bundle / "benchmark_input.json")),
                    "physical": expected_source_contract(self.case, endpoint_name,
                        profile=self.endpoint["profile"], source_revision=self.runtime.source_commit, image=self.runtime.image)}
                if extension["target_contract"] != actual_contract:
                    raise ValueError("EXTEND target differs from the actual task/source/input/runtime contract")
            source_alias = input_source_alias(self.runtime, manifest, self.bundle, output,
                                              baseline_validation=baseline_validation)
            changes = source_changes(subject, self.runtime.source_commit, self.task.workspace.protected_paths)
            if extension is not None and changes:
                raise ValueError("EXTEND cannot contain algorithm changes")
            result["changed_files"] = [row["path"] for row in changes]
            # Compiled changes must not be silently ignored by an installed
            # image runtime. A separate pinned image build is required first.
            native = [row["path"] for row in changes if Path(row["path"]).suffix in NATIVE_SUFFIXES
                      or Path(row["path"]).name in {"CMakeLists.txt", "Makefile"}]
            if native:
                raise ValueError("Native source changes require a separately built pinned runtime image: " + ", ".join(native))
            candidate_path = subject / CANDIDATE_CONFIG
            candidate = yaml.safe_load(candidate_path.read_text()) if candidate_path.is_file() else {}
            config = resolved_config(self.base, candidate, self.task, self.runtime.cpus)
            snapshot = self._snapshot(subject, output, changes)
            config, bindings = bind_frontend_config_paths(config, snapshot,
                latest_framework=self.runtime.framework if self.latest else None)
            _json(output / "frontend-config-bindings.json", bindings)
            config_path = output / "evaluation-config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            reused_prefix = []
            if extension is not None:
                from scripts.workspace_reference_extension import copy_qualified_prefix
                reused_prefix = copy_qualified_prefix(extension, output / "run")
            _json(output / "evaluation-contract.json", {
                "schema_version": 1, "task": self.task.to_dict(), "task_digest": self.task.digest(),
                "case": self.case, "objective": self.objective, "source_revision": self.runtime.source_commit,
                "audit_framework_identity": audit_identity,
                "latest_framework_identity": getattr(self.runtime, "latest_framework_identity", None),
                "benchmark_input": str(self.bundle),
                "source_changes": changes, "input_manifest_sha256": _sha(self.bundle / "benchmark_input.json"),
                "image": self.runtime.image, "source_alias": str(source_alias),
                "installed_runtime": getattr(self.runtime, "workspace_runtime_image", None),
                "opt_passes": self.endpoint["opt_passes"], "evaluation_endpoint": self.endpoint,
                "resource_caps": {"cpus": self.runtime.cpus, "memory_gb": self.runtime.memory_gb},
                "profile_threads": config.get("threads", 32),
                "thermal_snapshot_threads": (4 if self.latest and self.objective == "thermal" else (2 if self.objective == "thermal"
                    and getattr(self.runtime, "reference_baseline", None) == "reference_20260923"
                    else (32 if self.objective == "thermal" else None))),
                "trusted_evaluator_patch_sha256": _sha(self.patch) if self.patch else None,
                "evaluation_config_sha256": _sha(config_path),
                "frontend_config_bindings": bindings,
                "reference_run": str(reference_run) if reference_run else None,
                "reference_purpose": "EXTEND" if extension else ("FRESH" if reference_run is None else "CANDIDATE"),
                "parent_reference_id": extension["record_value"]["reference_id"] if extension else None,
                "reused_prefix": reused_prefix,
                "reuse_policy": ("byte-identical qualified native prefix; execute missing suffix only" if extension else
                                 "byte-identical frozen input/floorplan; rerun timing features through final boundary"),
                "reference_baseline": getattr(self.runtime, "reference_baseline", None),
                "reference_baseline_validation": baseline_validation,
            })
            backend_source = output / "backend-source"
            isolated_clone(self.runtime.source, backend_source)
            if self.patch is not None:
                apply_candidate_patch(backend_source, self.patch)
            if extension is None:
                copy_frozen_prefix(Path(validated["run_dir"]), output / "run")
            run = output / "run"
            token = hashlib.sha256(str(output).encode()).hexdigest()[:18]
            timing["preparation_wall_seconds"] = time.monotonic() - phase_clock
            attempt.event("preparation-completed", timing=dict(timing))
            phase, phase_clock = "admission", time.monotonic()
            queued_at = time.time()
            attempt.event("queued", state="queued", queued_at=queued_at,
                          queued_at_utc=datetime.fromtimestamp(queued_at, timezone.utc).isoformat())
            with self.runtime.slot() as gpu:
                # Preserve the original admission boundary for the one shared
                # deadline; evidence writes must not extend the native budget.
                deadline = time.monotonic() + self.task.environment.walltime_seconds
                timing["admission_wait_seconds"] = time.monotonic() - phase_clock
                admitted_at = time.time()
                attempt.event("admitted", state="running", gpu=gpu, admitted_at=admitted_at,
                              admitted_at_utc=datetime.fromtimestamp(admitted_at, timezone.utc).isoformat(),
                              timing=dict(timing))
                phase, phase_clock = "evaluation", time.monotonic()
                # Queue waiting is not physical runtime. Once admitted, one
                # deadline covers all physical passes and never resets per die.
                dreamplace = None
                dream_changes = [row["path"] for row in changes if row["path"].startswith(DREAM_PREFIX)]
                if dream_changes:
                    runtime_dir = output / "runtime"
                    runtime_dir.mkdir()
                    command = ["/usr/bin/python3", str(Path(__file__).resolve()), "--overlay-worker",
                               "--source", str(source_alias), "--output", str(runtime_dir),
                               "--changes", json.dumps(dream_changes)]
                    command[command.index("--source") + 1] = str(snapshot)
                    self._call("runtime-overlay", self._docker(command, gpu, output, backend_source, source_alias,
                        writable=(runtime_dir,), name="o3db-ws-" + token + "-overlay"), output, deadline)
                    dreamplace = runtime_dir / "dreamplace-install"
                flow = ["/usr/bin/python3", "-m", "open3dbench", "--repo", str(source_alias),
                        "run", str(config_path), "--run-dir", str(run), "--resume"]
                if extension is None:
                    self._call("frontend", self._docker(flow + ["--from", "partition" if self.latest else "timing_extract", "--to", "placement"],
                        gpu, output, snapshot, source_alias, writable=(run,),
                        readonly=tuple(run / name for name in PREFIX.values()), dreamplace=dreamplace,
                        name="o3db-ws-" + token + "-frontend"), output, deadline)
                frontend_status = _read(run / "run.json").get("stages", {})
                completed_frontend = dict(self.frontend)
                for stage, directory in self.frontend.items():
                    if stage == "timing_extract" and frontend_status.get(stage, {}).get("status") == "skipped":
                        # Non-timing-driven partitioners do not consume this
                        # optional feature provider. Retain the native skipped
                        # record; the independent verifier decides its validity.
                        if config["partition"].get("algorithm") in {"tp-gnn", "tp-gnn-flat"}:
                            raise ValueError("TP-GNN timing-driven feature extraction cannot be disabled")
                        completed_frontend.pop(stage)
                        continue
                    validate_trusted_stage(run / directory, stage)
                # A frontend process must not manufacture future evaluator data.
                if extension:
                    from global_bench.verifiers.open3dbench_codesign_verifier import STAGE_DIRS
                    reused_directories = {STAGE_DIRS[stage] for stage in extension["stages"]}
                else:
                    reused_directories = set()
                for forbidden in ("40_pdn", "50_hbt", "60_opt", "70_cts", "75_global_route",
                                  "80_route", "90_report", "95_thermal", "95_thermal_snapshot"):
                    if (run / forbidden).exists() and forbidden not in reused_directories:
                        raise ValueError("Frontend wrote downstream evaluator directory: " + forbidden)
                _json(output / "frontend-input-audit.json", validate_frontend_input_boundary(run))
                protected_prefix = set(completed_frontend.values()) | reused_directories
                self._call("backend", self._docker(flow + ["--from", extension["start_stage"] if extension else "pdn",
                                                          "--to", self.endpoint["backend_stop"]],
                    gpu, output, backend_source, source_alias, writable=(run,),
                    readonly=tuple(run / name for name in sorted(protected_prefix)),
                    name="o3db-ws-" + token + "-backend"), output, deadline)
                if self.objective == "thermal":
                    command = self._thermal_command(source_alias, run, config)
                    readonly = tuple(run / name for name in (*completed_frontend.values(), "40_pdn", "50_hbt", "60_opt", "70_cts"))
                    self._call("thermal", self._docker(command, gpu, output, backend_source, source_alias,
                        writable=(run,), readonly=readonly,
                        name="o3db-ws-" + token + "-thermal"), output, deadline)
                submission = self._submission(output, changes, subject)
                verification_dir = output / "verification"
                verification_dir.mkdir()
                reference = reference if reference is not None else run
                verifier = self.runtime.framework / "global_bench/verifiers/open3dbench_codesign_verifier.py"
                command = ["/usr/bin/python3", str(verifier), "--mode", self.endpoint["verifier_mode"],
                           "--profile", self.endpoint["profile"],
                           "--submission", str(submission), "--output", str(verification_dir),
                           "--evaluated-run", str(run), "--reference-run", str(reference)]
                if reference_run is not None and self.sealed_reference is not None:
                    command += ["--reference-record", str(self.runtime.reference_record),
                                "--reference-record-sha256", self.runtime.reference_record_sha256,
                                "--evaluation-contract", str(output / "evaluation-contract.json")]
                # The standalone verifier imports no candidate package. It sees
                # all physical/source inputs read-only and only its receipt RW.
                argv = self._docker(command, gpu, output, backend_source, source_alias,
                                    writable=(verification_dir,), name="o3db-ws-" + token + "-verify")
                try:
                    self._call("verifier", argv, output, deadline)
                finally:
                    if (verification_dir / "verification.json").is_file():
                        result["verification"] = str(verification_dir / "verification.json")
                verified = _read(verification_dir / "verification.json")
                if not verified.get("passed"):
                    raise ValueError("Trusted verifier rejected the run")
                result.update(success=True, metrics=verified["metrics"])
            # Verify immutable sources/config/prefix again after all subprocesses.
            validate_benchmark_input(self.bundle)
            for stage, directory in PREFIX.items():
                validate_trusted_stage(run / directory, stage)
            if source_changes(subject, self.runtime.source_commit, self.task.workspace.protected_paths) != changes:
                raise ValueError("Subject changed during evaluation")
            revalidate_runtime_framework(self.runtime)
        except Exception as error:
            result.update(success=False, metrics={}, error=f"{type(error).__name__}: {error}")
            if phase in {"preparation", "admission"}:
                timing[{"preparation": "preparation_wall_seconds", "admission": "admission_wait_seconds"}[phase]] = (
                    time.monotonic() - phase_clock)
            attempt.event("evaluation-failed", failed_phase=phase, error=result["error"], timing=dict(timing))
        commands = attempt.value["command_phases"]

        def command_seconds(phases, key="process_wall_seconds"):
            values = [receipt[key] for name, receipt in commands.items()
                      if name in phases and isinstance(receipt.get(key), (int, float))]
            return sum(values) if values else None

        timing.update(native_flow_command_wall_seconds=command_seconds(FLOW_COMMAND_PHASES),
                      runtime_overlay_command_wall_seconds=command_seconds({"runtime-overlay"}),
                      verifier_command_wall_seconds=command_seconds({"verifier"}),
                      timeout_cleanup_wall_seconds=command_seconds(set(commands), "cleanup_wall_seconds"),
                      total_wall_seconds=time.monotonic() - attempt.clock)
        result["runtime_seconds"] = timing["total_wall_seconds"]
        result["runtime_seconds_scope"] = "total evaluator wall time, including source preparation and admission"
        result["timing"] = timing
        result["execution"] = {
            "command_dispatches": sum(row.get("dispatch_attempted") is True for row in commands.values()),
            "native_flow_command_dispatches": sum(row.get("dispatch_attempted") is True
                for name, row in commands.items() if name in FLOW_COMMAND_PHASES),
            "command_processes_started": sum(row.get("process_started") is True for row in commands.values()),
            "uncertain_command_process_starts": sum(row.get("dispatch_attempted") is True
                and row.get("process_started") is None for row in commands.values()),
            "native_physical_attempts": None,
            "interpretation": "Command dispatches are not native physical attempts; solver/stage evidence must be audited separately.",
            "command_phases": commands,
        }
        _json(output / "feedback.json", result)
        attempt.event("completed" if result["success"] else "failed",
                      state="completed" if result["success"] else "failed", finished_at=time.time(),
                      feedback=str(output / "feedback.json"), success=result["success"],
                      timing=timing, execution=result["execution"])
        self._attempt_timing = None
        return result


def _overlay_worker(source: Path, output: Path, paths: list[str]) -> None:
    """Install Python changes/deletions into a private copy, without importing them."""
    install = Path("/opt/dreamplace")
    target_root = output / "dreamplace-install"
    shutil.copytree(install, target_root, symlinks=True)
    for name in paths:
        if not name.startswith(DREAM_PREFIX):
            raise ValueError("Invalid DREAMPlace overlay path")
        relative = name[len(DREAM_PREFIX):]
        if ".." in PurePosixPath(relative).parts or Path(relative).is_absolute():
            raise ValueError("Unsafe runtime path")
        target, origin = target_root / "dreamplace" / relative, source / name
        if target.is_symlink():
            target.unlink()
        if not origin.exists():
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-worker", action="store_true", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--changes", required=True)
    args = parser.parse_args()
    _overlay_worker(args.source, args.output, json.loads(args.changes))
