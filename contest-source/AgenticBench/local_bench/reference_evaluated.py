"""Read-only qualification of freshly evaluated, one-pass references.

This is a trusted producer for the reference registry, not an evaluation
driver. It never launches a tool, imports the evaluated source snapshot,
accepts a cached score, or writes into a run. Reference-purpose provenance is
mandatory. EXTEND additionally needs a registry-validated parent and explicit
unchanged-prefix inventory; ordinary candidate-purpose evaluations are rejected.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from pathlib import Path

import yaml

from global_bench.model import load_global_task
from scripts.workspace_task_selection import load_registered_task
from global_bench.verifiers import open3dbench_codesign_verifier as verifier
from open3dbench.backend.contracts import validate_checkpoint
from open3dbench.config import load_config

from . import reference_baseline as baseline


ENDPOINTS = {
    "timing": ("tp_gnn_placement_one_pass", "lol_partition_placement_timing_one_pass", "timing", "opt"),
    "cts_thermal": ("tp_gnn_placement_cts_thermal_one_pass", "lol_partition_placement_cts_thermal_one_pass", "thermal", "cts"),
    "routed_thermal": ("tp_gnn_placement_thermal", "lol_partition_placement_thermal", "routed_thermal", "thermal"),
}
FRAMEWORK = Path(__file__).resolve().parents[1]
REPO = FRAMEWORK.parent

# Native one-pass OPT publishes these final-copy log aliases. This is an exact
# serialization compatibility boundary, not permission for arbitrary log links
# or symlinked physical artifacts. Manifest-listed artifacts remain regular.
OPT_FINAL_LOG_ALIASES = {
    **{f"{die}/{name}.log": (f"../iterations/01/{die}/{name}.log",
                             f"iterations/01/{die}/{name}.log")
       for die in ("bottom", "top") for name in ("repair_design", "repair_timing")},
    **{f"{die}/{kind}/{view}/proof.log":
       (f"../../../iterations/01/{die}/{kind}/{view}/proof.log",
        f"iterations/01/{die}/{kind}/{view}/proof.log")
       for die in ("bottom", "top")
       for kind in ("sdc_evidence", "timing_sdc_evidence", "final_sdc_evidence")
       for view in ("bottom", "top")},
    "assembly/passive/sta.log": ("../../iterations/01/global/passive/sta.log",
                                 "iterations/01/global/passive/sta.log"),
}


def expected_frozen_inputs(case, input_manifest_sha256):
    """Identity to match against a separately validated benchmark-input bundle."""
    if not re.fullmatch(r"[0-9a-f]{64}", str(input_manifest_sha256)):
        raise ValueError("Reference input manifest identity is missing")
    baseline.profile_path(case)
    physical = baseline.identity()["cases"][case]["physical_prefix"]
    return {"benchmark_input_manifest_sha256": input_manifest_sha256,
            "physical_prefix_sha256": {path: row["sha256"] for path, row in physical.items()}}


def expected_source_contract(case, endpoint, *, profile, source_revision, image):
    """Build the trusted physical identity before lookup or native evaluation."""
    if endpoint not in ENDPOINTS or profile != ENDPOINTS[endpoint][0]:
        raise ValueError("A matching explicit one-pass reference profile is required")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_revision)):
        raise ValueError("Reference source revision is not a commit identity")
    baseline.validate_reference_image(image)
    recipe = yaml.safe_load((baseline.profile_path(case) / "config.yaml").read_text())
    if endpoint == "routed_thermal":
        recipe["thermal"]["enabled"] = True
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"producer_contract": "workspace-reference-native-one-pass-v1",
            "profile": profile, "opt_passes": 1, "configuration": recipe,
            "physical_config_sha256": digest, "source_revision": source_revision, "image": image,
            **({"thermal_snapshot_threads": 2} if endpoint == "cts_thermal" else {}),
            "algorithm_config_sha256": {
                "partition": baseline.sha256(REPO / baseline.PARTITION_CONFIG),
                "placement": baseline.sha256(REPO / baseline.PLACEMENT_CONFIG)}}


def _regular(root: Path, path: Path) -> Path:
    """Do not let resolve() erase evidence of an internal symlink."""
    path = Path(path)
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        if part in (".", ".."):
            raise ValueError("Unsafe reference evidence path: " + str(path))
        current = current / part
        if current.is_symlink():
            raise ValueError("Reference evidence traverses a symlink: " + str(path))
    if not path.is_file():
        raise ValueError("Missing regular reference evidence: " + str(path))
    path.resolve(strict=True).relative_to(root)
    return path


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not verifier._finite_tree(value):
        raise ValueError("Reference evidence must be a finite JSON object: " + str(path))
    return value


def _record(path: Path, role: str) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": baseline.sha256(path), "role": role}


def _public_config(path: Path) -> dict:
    return {key: value for key, value in load_config(path).items() if not key.startswith("_")}


def _provenance(run: Path, evaluation: Path, case: str, endpoint: str, profile: str, parent):
    if run != evaluation / "run":
        raise ValueError("Reference run must be the evaluation root's own run directory")
    contract_path = _regular(evaluation, evaluation / "evaluation-contract.json")
    contract = _json(contract_path)
    _, _, objective, stop = ENDPOINTS[endpoint]
    if (contract.get("schema_version") != 1 or "reference_run" not in contract
            or contract["reference_run"] is not None or contract.get("source_changes") != []):
        raise ValueError("Reference-purpose evaluation provenance is required; candidates cannot become baselines")
    purpose = contract.get("reference_purpose", "FRESH")
    if purpose != ("EXTEND" if parent is not None else "FRESH"):
        raise ValueError("Reference-purpose provenance does not match FRESH/EXTEND qualification")
    if contract.get("case") != case or contract.get("objective") != objective:
        raise ValueError("Reference evaluation case/endpoint mismatch")
    if endpoint == "cts_thermal" and (type(contract.get("thermal_snapshot_threads")) is not int
                                      or contract["thermal_snapshot_threads"] != 2):
        raise ValueError("CTS thermal reference requires the explicit two-thread power snapshot contract")
    selected = contract.get("evaluation_endpoint", {})
    if (not isinstance(selected, dict) or selected.get("profile") != profile
            or selected.get("backend_stop") != stop
            or selected.get("opt_passes") != 1 or type(contract.get("opt_passes")) is not int
            or contract["opt_passes"] != 1 or contract.get("trusted_evaluator_patch_sha256") is not None):
        raise ValueError("Reference evaluation requires the explicit native one-pass endpoint")
    task = load_registered_task(selected.get("task_id"), objective=objective, framework=FRAMEWORK)
    if task.objective.evaluator_profile != profile:
        raise ValueError("Reference task/profile definition differs from trusted evaluator")
    recorded_task = copy.deepcopy(contract.get("task"))
    expected_task = json.loads(json.dumps(task.to_dict()))
    if not isinstance(recorded_task, dict):
        raise ValueError("Reference task provenance is missing")
    for value in (recorded_task, expected_task):
        value.pop("root", None)
        value.pop("instruction_path", None)
    if recorded_task != expected_task or contract.get("task_digest") != task.digest():
        raise ValueError("Reference task/profile definition differs from trusted evaluator")
    if not re.fullmatch(r"[0-9a-f]{40}", str(contract.get("source_revision", ""))):
        raise ValueError("Reference source revision is not a commit identity")
    baseline.validate_reference_image(contract.get("image"))
    if not re.fullmatch(r"[0-9a-f]{64}", str(contract.get("input_manifest_sha256", ""))):
        raise ValueError("Reference input manifest identity is missing")

    config_path = _regular(evaluation, evaluation / "evaluation-config.yaml")
    if baseline.sha256(config_path) != contract.get("evaluation_config_sha256"):
        raise ValueError("Reference evaluation configuration hash mismatch")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Reference configuration is missing")
    recipe = yaml.safe_load((baseline.profile_path(case) / "config.yaml").read_text())
    if endpoint == "routed_thermal":
        recipe["thermal"]["enabled"] = True
    rebound = copy.deepcopy(config)
    bindings = contract.get("frontend_config_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"partition", "placement"}:
        raise ValueError("Reference frontend configuration bindings are missing")
    evidence = [_record(contract_path, "reference-evaluation-contract"),
                _record(config_path, "reference-evaluation-config")]
    for domain, source_relative in (("partition", baseline.PARTITION_CONFIG),
                                    ("placement", baseline.PLACEMENT_CONFIG)):
        binding = bindings[domain]
        expected_path = evaluation / "source" / source_relative
        if (not isinstance(binding, dict) or binding.get("source_relative") != source_relative
                or binding.get("requested") != recipe[domain]["config"]
                or binding.get("origin") != "explicit" or binding.get("resolved") != str(expected_path)
                or config.get(domain, {}).get("config") != str(expected_path)):
            raise ValueError("Reference frontend binding differs from the frozen recipe")
        _regular(evaluation, expected_path)
        digest = baseline.sha256(expected_path)
        if digest != binding.get("sha256") or digest != baseline.sha256(REPO / source_relative):
            raise ValueError("Reference frontend algorithm configuration differs")
        evidence.append(_record(expected_path, "reference-" + domain + "-config"))
        rebound[domain]["config"] = recipe[domain]["config"]
    if rebound != recipe or contract.get("profile_threads") != recipe["threads"]:
        raise ValueError("Reference physical recipe differs from the selected profile")
    run_config = _regular(run, run / "config.yaml")
    if _public_config(run_config) != _public_config(config_path):
        raise ValueError("Native run configuration differs from the evaluation contract")
    evidence.append(_record(run_config, "native-run-config"))
    return contract, recipe, evidence


def _stage_evidence(run: Path, case: str, contract: dict, evidence: list, reused=()):
    status_path = _regular(run, run / "run.json")
    status = _json(status_path)
    if (status.get("design") != case or status.get("type") != "lol" or status.get("pdk") != "ng45"):
        raise ValueError("Native run identity differs from the reference contract")
    evidence.append(_record(status_path, "native-run-manifest"))
    manifests, runtime = {}, {}
    stages = status.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("Native reference run has no stage records")
    for stage, row in stages.items():
        if not isinstance(row, dict):
            raise ValueError("Native reference stage record is malformed")
        if stage == "timing_extract" and row.get("status") == "skipped":
            raise ValueError("Fresh TP-GNN-flat reference requires native timing extraction")
        directory = verifier.STAGE_DIRS.get(stage)
        if directory is None:
            raise ValueError("Fresh references require the explicit split-stage contract")
        root = run / directory
        path = _regular(run, root / "stage_manifest.json")
        manifest = _json(path)
        if (manifest.get("design") != case or manifest.get("integration") != "lol"
                or manifest.get("pdk") != "ng45"):
            raise ValueError("Stage manifest reference identity mismatch: " + stage)
        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("Stage has no authenticated artifacts: " + stage)
        for name, record in artifacts.items():
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError("Malformed native artifact record")
            artifact = _regular(root, root / record["path"])
            evidence.append(_record(artifact, "native-artifact:" + stage + ":" + name))
        verifier._validate_stage(run, stage)
        if stage not in {"input", "floorplan", *reused}:
            producer = manifest.get("provenance", {})
            if (not isinstance(producer, dict) or producer.get("repo_commit") != contract["source_revision"]
                    or producer.get("repo_dirty") is not False
                    or producer.get("container_image_id") != contract["image"]):
                raise ValueError("Fresh stage producer identity mismatch: " + stage)
        measured = manifest.get("timing", {}).get("runtime_seconds")
        if measured is not None and verifier._finite(measured, "native historical runtime") < 0:
            raise ValueError("Negative native historical runtime")
        runtime[stage] = measured  # Absent measurements stay null, never zero.
        manifests[stage] = manifest
        evidence.append(_record(path, "native-stage-manifest:" + stage))
    return manifests, runtime


def _extend_prefix(run: Path, case: str, endpoint: str, contract: dict, parent):
    """Parent must be the registry's freshly revalidated qualification envelope."""
    if parent is None:
        if contract.get("parent_reference_id") is not None or contract.get("reused_prefix"):
            raise ValueError("FRESH provenance cannot claim an unvalidated parent prefix")
        return ()
    if (not isinstance(parent, dict) or parent.get("qualified") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", str(parent.get("reference_id", "")))
            or not parent.get("record") or not parent.get("record_sha256")
            or parent.get("case") != case
            or contract.get("parent_reference_id") != parent["reference_id"]):
        raise ValueError("EXTEND requires a registry-validated matching parent reference")
    record = Path(parent["record"])
    if (not record.is_absolute() or record.is_symlink() or not record.is_file()
            or baseline.sha256(record) != parent["record_sha256"]):
        raise ValueError("EXTEND parent registry record identity changed")
    parent_endpoint = parent.get("endpoint")
    if (parent_endpoint not in ENDPOINTS or list(ENDPOINTS).index(parent_endpoint) >= list(ENDPOINTS).index(endpoint)):
        raise ValueError("EXTEND must add a suffix beyond the parent physical endpoint")
    stages = verifier.TIMING_STAGES + (("cts",) if parent_endpoint == "cts_thermal" else ())
    inventory = contract.get("reused_prefix")
    if (not isinstance(inventory, list) or any(not isinstance(row, dict) for row in inventory)
            or [row.get("stage") for row in inventory] != list(stages)):
        raise ValueError("EXTEND needs the complete contiguous unchanged-prefix inventory")
    source = Path(parent.get("baseline", {}).get("source_run", "")).resolve(strict=True)
    parent_evidence = {row["path"]: row for row in parent.get("evidence", [])}
    for item in inventory:
        stage = item["stage"]
        relative = Path(verifier.STAGE_DIRS[stage]) / "stage_manifest.json"
        original = _regular(source, source / relative)
        target = _regular(run, run / relative)
        digest = baseline.sha256(original)
        bound = parent_evidence.get(str(original), {})
        if (bound.get("sha256") != digest or item.get("manifest_sha256") != digest
                or baseline.sha256(target) != digest):
            raise ValueError("EXTEND prefix stage manifest changed: " + stage)
        artifact_rows = _json(original).get("artifacts", {}).values()
        required = {str(original)} | {str(original.parent / row["path"]) for row in artifact_rows}
        # Include native sidecar evidence that was independently read but not
        # necessarily listed by an older stage manifest.
        for value in parent_evidence:
            try:
                Path(value).relative_to(original.parent)
            except ValueError:
                continue
            required.add(value)
        for value in sorted(required):
            before = _regular(source, Path(value))
            after = _regular(run, run / before.relative_to(source))
            bound = parent_evidence.get(str(before), {})
            if (bound.get("bytes") != before.stat().st_size or bound.get("sha256") != baseline.sha256(before)
                    or after.stat().st_size != bound.get("bytes") or baseline.sha256(after) != bound.get("sha256")):
                raise ValueError("EXTEND changed authenticated prefix bytes: " + str(before.relative_to(source)))
    return stages


def _file_identity(value):
    return {"device": value.st_dev, "inode": value.st_ino, "mode": value.st_mode,
            "bytes": value.st_size, "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns}


def _observe_opt_log_alias(run: Path, path: Path):
    """Read an exact native alias without admitting its link as file evidence.

    Both the alias parents and target path must be non-symlinked. The target is
    always constructed from our fixed table, never from untrusted readlink text.
    Link identity is retained in the qualification payload so subsequent ADOPT
    requalification rejects replacement even with byte-identical link text.
    """
    stage = run / "60_opt"
    try:
        relative = path.relative_to(stage).as_posix()
        if relative not in OPT_FINAL_LOG_ALIASES:
            raise ValueError("Unrecognized native OPT log alias: " + str(path))
        link_text, target_relative = OPT_FINAL_LOG_ALIASES[relative]
        current = run
        for part in path.parent.relative_to(run).parts:
            current = current / part
            if current.is_symlink() or not current.is_dir():
                raise ValueError("Native OPT log alias traverses a symlink: " + str(path))
        before = path.lstat()
        if not stat.S_ISLNK(before.st_mode) or os.readlink(path) != link_text:
            raise ValueError("Native OPT log alias differs from the one-pass mapping: " + str(path))
        target = _regular(run, stage / target_relative)
        target_before = _file_identity(target.stat())
        if not stat.S_ISREG(target_before["mode"]):
            raise ValueError("Native OPT log alias target is not regular: " + str(target))
        record = _record(target, "native-opt-final-log-alias-target")
        if target_before != _file_identity(_regular(run, target).stat()):
            raise ValueError("Native OPT log alias target changed during qualification")
        after = path.lstat()
        if _file_identity(before) != _file_identity(after) or os.readlink(path) != link_text:
            raise ValueError("Native OPT log alias changed during qualification")
        return {"kind": "native-opt-final-copy-log-v1", "path": str(path),
                "link_text": link_text, "link_identity": _file_identity(before),
                "target_path": str(target), "target_bytes": record["bytes"],
                "target_sha256": record["sha256"]}, record
    except (OSError, RuntimeError) as error:
        raise ValueError("Invalid native OPT log alias: " + str(path)) from error


def _checkpoint_chain(run: Path, manifests: dict, evidence: list):
    previous = None
    aliases = []
    renamed = {"global_routing": "global_route", "detailed_routing": "route"}
    for stage in verifier.STAGE_DIRS:
        if stage not in manifests or stage in verifier.TIMING_STAGES[:4]:
            continue
        path = _regular(run, run / verifier.STAGE_DIRS[stage] / "checkpoint.json")
        refs = {record["path"] for record in manifests[stage]["artifacts"].values()}
        if "checkpoint.json" not in refs:
            raise ValueError("Stage manifest does not authenticate its checkpoint: " + stage)
        checkpoint = validate_checkpoint(path, require_files=True, require_spef=stage == "report")
        if checkpoint.get("stage") != renamed.get(stage, stage) or checkpoint.get("pdk") != "ng45":
            raise ValueError("Physical checkpoint stage/PDK mismatch")
        if previous is not None:
            source = checkpoint.get("source_checkpoint")
            if (not isinstance(source, str) or Path(source).is_absolute()
                    or (path.parent / source).resolve() != previous):
                raise ValueError("Reference checkpoint does not preserve the same-run prefix chain")
        previous = path
        # Include sidecars accepted by the physical contract even when an older
        # native manifest did not enumerate every auxiliary file.
        for artifact in sorted(path.parent.rglob("*")):
            if artifact.is_symlink():
                if stage != "opt":
                    raise ValueError("Reference physical stage contains a symlink")
                alias, target = _observe_opt_log_alias(run, artifact)
                aliases.append(alias)
                evidence.append(target)
                continue
            if artifact.is_file():
                evidence.append(_record(_regular(run, artifact), "native-checkpoint-evidence:" + stage))
    return aliases


def prepare_evaluated_reference(run_root, case, endpoint, *, profile,
                                evaluation_root=None, parent_reference=None):
    """Qualify an existing FRESH/EXTEND result without executing native stages.

    ``evaluation_root`` defaults to ``run_root.parent``. Only trusted workspace
    one-pass reference-purpose contracts are supported. A caller-supplied
    ``passed`` flag, cached fitness, or candidate's output is never consumed.
    ``parent_reference`` must come from the registry's current independent
    revalidation and include its record path/hash and reference ID. EXTEND also
    requires explicit reference-purpose provenance and byte-identical prefix.
    """
    if endpoint not in ENDPOINTS or profile != ENDPOINTS[endpoint][0]:
        raise ValueError("A matching explicit one-pass reference profile is required")
    run = Path(run_root).expanduser().resolve(strict=True)
    evaluation = (Path(evaluation_root).expanduser().resolve(strict=True)
                  if evaluation_root is not None else run.parent)
    contract, recipe, evidence = _provenance(run, evaluation, case, endpoint, profile, parent_reference)
    reused = _extend_prefix(run, case, endpoint, contract, parent_reference)
    physical_prefix = baseline.validate_physical_prefix(run, case)
    manifests, runtime = _stage_evidence(run, case, contract, evidence, reused)
    aliases = _checkpoint_chain(run, manifests, evidence)
    verifier._validate_thermal_reference_frontend(run)
    if endpoint == "timing":
        raw = verifier._timing_row(run, expected_passes=1)
    elif endpoint == "cts_thermal":
        raw = verifier._thermal_row(run, post_cts=True, expected_passes=1)
        power_threads = _json(run / "95_thermal_snapshot/metrics.json").get("power", {}).get("provenance", {}).get("threads")
        if type(power_threads) is not int or power_threads != 2:
            raise ValueError("CTS thermal native power evidence must record exactly two threads")
        raw["boundary_budget_model"] = _json(run / "60_opt/metrics.json").get("boundary_budget_model", "passive-cutpoint")
    else:
        raw = verifier._thermal_row(run, expected_passes=1, require_timing_stage=True)
    # These native frontend files are independently consumed by the verifier.
    for relative in ("10_floorplan/io_metrics.json", "20_partition/partition_manifest.json",
                     "20_partition/tp_gnn_report.json", "20_partition/partition_metrics.json",
                     "30_placement/metrics.json", "30_placement/stages/02_global_place/stage_manifest.json"):
        evidence.append(_record(_regular(run, run / relative), "independent-verifier-input"))
    if endpoint == "cts_thermal":
        snapshot = run / "95_thermal_snapshot"
        for path in sorted(snapshot.rglob("*")):
            if path.is_symlink():
                raise ValueError("Thermal snapshot contains a symlink")
            if path.is_file():
                evidence.append(_record(_regular(run, path), "native-cts-thermal-snapshot"))
    # Recheck after all reads to reject evidence changed during qualification.
    unique = {}
    for row in evidence:
        old = unique.setdefault(row["path"], row)
        if (old["bytes"], old["sha256"]) != (row["bytes"], row["sha256"]):
            raise ValueError("Reference evidence changed during qualification")
    for row in unique.values():
        path = _regular(evaluation, Path(row["path"]))
        if path.stat().st_size != row["bytes"] or baseline.sha256(path) != row["sha256"]:
            raise ValueError("Reference evidence changed during qualification")
    # The regular-evidence registry deliberately rejects symlinks. Bind native
    # aliases separately and recheck link identity/text plus target bytes after
    # every other qualification read; registry publication and ADOPT compare the
    # entire freshly reconstructed payload, including this alias inventory.
    for alias in aliases:
        current, _ = _observe_opt_log_alias(run, Path(alias["path"]))
        if current != alias:
            raise ValueError("Native OPT log alias changed during qualification")
    current_alias_paths = sorted(str(path) for path in (run / "60_opt").rglob("*") if path.is_symlink())
    if current_alias_paths != sorted(alias["path"] for alias in aliases):
        raise ValueError("Native OPT log alias inventory changed during qualification")
    source_contract = expected_source_contract(case, endpoint, profile=profile,
        source_revision=contract["source_revision"], image=contract["image"])
    stage_producers = {stage: {key: row.get("provenance", {}).get(key) for key in
            ("repo_commit", "repo_dirty", "implementation_sha256", "container_image_id",
             "openroad_version", "yosys_version", "python_version")}
            for stage, row in manifests.items()}
    return {"qualified": True, "producer_profile": "native-evaluation-v1",
            "case": case, "endpoint": endpoint,
            "baseline": {"raw_metrics": raw, "source_run": str(run),
                         "native_runtime_historical": runtime},
            "source_identity": {"source_revision": contract["source_revision"],
                                "image": contract["image"],
                                "evaluation_contract_sha256": baseline.sha256(evaluation / "evaluation-contract.json"),
                                "task_digest": contract["task_digest"], "stage_producers": stage_producers,
                                "parent_reference_id": contract.get("parent_reference_id"),
                                "reused_prefix": contract.get("reused_prefix", [])},
            "source_contract": source_contract,
            "frozen_inputs": {"benchmark_input_manifest_sha256": contract["input_manifest_sha256"],
                              "physical_prefix_sha256": physical_prefix},
            "evidence": [unique[key] for key in sorted(unique)],
            **({"native_log_aliases": aliases} if aliases else {}),
            "executed_native_stages": []}
