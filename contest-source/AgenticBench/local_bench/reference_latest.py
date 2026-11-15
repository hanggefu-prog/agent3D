"""Explicit 2026-10-04 reference contract. Read-only; never dispatch EDA.

The old reference_20260923 family is intentionally not widened. A current GRT
publication qualifies a GRT handoff, not a placement hook or a timing score.
"""
from __future__ import annotations

import hashlib
import json
import math
import copy
import shutil
import re
import os
from pathlib import Path, PurePosixPath

import yaml

BASELINE_ID = "reference_20261004"
SOURCE_REVISION = "4d882f0b5b9aa98ba2742510dc1b011e43b31657"
NATIVE_REVISION = "56ea5f41b1352b937cae368832f156a445097102"
IMAGE_ID = "sha256:1b5567c86d48effa03b0543089e01fec37786e4f249309b7da627b477d7639e5"
STA_REVISION = "5d8d515edfa5a22165faea5897539da1888fe68d"
BINARY_SHA256 = {"/usr/local/bin/openroad": "aa4d4a41e41945e143bf680cd87be72351b1112b766765ec66dfb2a55b174ed1",
                 "/usr/local/bin/sta": "2f4a390b7056ea39f58a4bc123e1c6f0bf22684ad510f168eab35c0a375bce6e"}
CASES = ("bp_fe", "bp", "ariane133", "swerv_wrapper")
STAGE_DIRS = {"input": "00_input", "floorplan": "10_floorplan", "partition": "20_partition",
              "placement": "30_placement", "pdn": "40_pdn", "hbt": "50_hbt",
              "opt": "60_opt", "cts": "70_cts"}
PROFILE_ROOT = Path(__file__).resolve().parent / "benchmark_inputs"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def profile_path(case):
    require(case in CASES, "Case outside the latest four-case reference family")
    return PROFILE_ROOT / (case + "_lol_ng45_" + BASELINE_ID)


def is_latest_config(path):
    path = Path(path).resolve()
    return any(path == (profile_path(case) / "config.yaml").resolve() for case in CASES)


def validate_reference_image(image):
    require(image == IMAGE_ID, "Latest reference requires its exact image digest")
    return image


def validate_reference_config(config, case):
    expected = yaml.safe_load((profile_path(case) / "config.yaml").read_text())
    require(config == expected, "Initial configuration differs from latest reference recipe")
    return {"baseline_id": BASELINE_ID, "config_sha256": sha256(profile_path(case) / "config.yaml")}


def bind_recipe_paths(config, source_root):
    """Rebind only the two exact controller-owned paths; not arbitrary traversal."""
    result = copy.deepcopy(config)
    for stage, relative in (("partition", PARTITION_CONFIG), ("placement", PLACEMENT_CONFIG)):
        if result.get(stage, {}).get("config") == relative:
            path = Path(__file__).resolve().parents[1] / "local_bench/reference_configs" / (stage + ".json")
            require(path.is_file() and not path.is_symlink(), "Trusted latest recipe is missing")
            result[stage]["config"] = str(path)
    return result


def validate_physical_prefix(run, case, *, fixed_partition=False):
    """A latest local input must have an explicit hash-bound bundle receipt."""
    run = Path(run).resolve(strict=True)
    manifest = run.parent / "benchmark_input.json"
    require(manifest.is_file(), "Latest physical prefix requires an imported input bundle")
    result = validate_reference_bundle(manifest, case, fixed_partition=fixed_partition)
    require(Path(read_json(manifest).get("run_dir", "run")).name == run.name,
            "Physical prefix differs from its bundle")
    return result["physical_input_sha256"]


def copy_grt_prefix(source, destination, stages):
    """Import validated stage bytes, preserving original ledger provenance."""
    source, destination = Path(source).resolve(strict=True), Path(destination)
    require(tuple(stages) == tuple(STAGE_DIRS), "Latest GRT requires the full eight-stage prefix")
    require(not destination.exists(), "GRT candidate output must be new")
    ledger = read_json(source / "run.json")
    require(ledger.get("stages", {}).get("cts", {}).get("status") == "success", "GRT needs successful CTS")
    inventory = {s: validate_stage(source, s)["manifest_sha256"] for s in stages}
    # Validate all aliases before copying. Never dereference an external source.
    for stage in stages:
        for path in (source / STAGE_DIRS[stage]).rglob("*"):
            if path.is_symlink():
                require(not Path(os.readlink(path)).is_absolute() and _inside(path.resolve(strict=True), source),
                        "Imported GRT prefix contains an external alias")
    destination.mkdir(parents=True)
    for stage in stages:
        shutil.copytree(source / STAGE_DIRS[stage], destination / STAGE_DIRS[stage], symlinks=True)
    imported = copy.deepcopy(ledger)
    imported["stages"] = {s: copy.deepcopy(ledger["stages"][s]) for s in stages}
    imported["status"] = "imported-prefix"
    imported["reference_import"] = {"source_run": str(source), "run_json_sha256": sha256(source / "run.json"),
                                    "stage_manifest_sha256": inventory, "executed_stages": []}
    (destination / "run.json").write_text(json.dumps(imported, sort_keys=True, indent=2) + "\n")
    for stage in stages:
        require(validate_stage(destination, stage)["manifest_sha256"] == inventory[stage], "Copied prefix changed")
    return imported["reference_import"]


def _safe_file(root, relative, *, containment=None):
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and ".." not in rel.parts and bool(rel.parts), "Unsafe artifact path")
    path = Path(root) / rel
    boundary = Path(containment or root).resolve(strict=True)
    resolved = path.resolve(strict=True)
    require(_inside(resolved, boundary) and resolved.is_file(), "Artifact escapes its immutable run")
    # Relative same-run aliases are retained as historical provenance; no
    # absolute link, broken link, directory link or external target is adopted.
    for part in (path, *path.parents):
        if part == boundary:
            break
        if part.is_symlink():
            require(not Path(os.readlink(part)).is_absolute(), "Absolute artifact alias is unsupported")
    return path


def validate_artifacts(root, artifacts, *, containment=None):
    require(isinstance(artifacts, dict) and bool(artifacts), "Empty artifact publication")
    checked = {}
    for name, record in artifacts.items():
        require(isinstance(record, dict), "Invalid artifact record")
        relative = record.get("path", name)
        require(isinstance(relative, str), "Missing artifact path")
        path = _safe_file(root, relative, containment=containment)
        size = record.get("bytes", record.get("size_bytes"))
        require(type(size) is int and path.stat().st_size == size, "Artifact size changed: " + relative)
        require(not ("bytes" in record and "size_bytes" in record) or record["bytes"] == record["size_bytes"], "Conflicting artifact sizes")
        require(sha256(path) == record.get("sha256"), "Artifact hash changed: " + relative)
        checked[relative] = record["sha256"]
    return checked


def validate_stage(run, stage):
    require(stage in STAGE_DIRS, "Unknown latest stage")
    run = Path(run).resolve(strict=True)
    path = run / STAGE_DIRS[stage] / "stage_manifest.json"
    data = read_json(path)
    require(data.get("stage") == stage and data.get("status") == "success", "Unsuccessful stage: " + stage)
    validate_artifacts(path.parent, data.get("artifacts"), containment=run)
    return {"manifest_sha256": sha256(path), "manifest": data}


def _finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Missing/invalid native metric: " + name)
    return value


def _native_grt_metrics(directory):
    """Artifact-only reader for the pinned native GRT publication grammar."""
    directory = Path(directory)
    text = (directory / "grt.log").read_text()
    require(text.count("OPEN3D_GRT_DONE die=" + directory.name) == 1 and "[ERROR" not in text,
            "GRT log lacks unique completion or contains an error")
    lengths = re.findall(r"\[INFO GRT-0018\] Total wirelength:\s*([0-9.]+)\s+um", text)
    times = re.findall(r"Global routing runtime\s*=\s*(\d+):(\d+):(\d+)", text)
    overflow = re.findall(r"^Total\s+\d+\s+\d+\s+[0-9.]+%\s+\d+\s*/\s*\d+\s*/\s*(\d+)\s*$", text, re.M)
    require(lengths and times and overflow, "Incomplete GRT native summary")
    total = int(overflow[-1])
    reports = list(directory.glob("congestion*.rpt"))
    require(reports or total == 0, "Nonzero GRT overflow requires its congestion report")
    bins = 0
    if reports:
        report = max(reports, key=lambda p: p.stat().st_mtime_ns)
        require(report.is_file() and not report.is_symlink(), "Unsafe congestion report")
        bins = sum(line.lstrip().startswith("violation type:") for line in report.read_text().splitlines())
    h, m, s = map(int, times[-1])
    return {"grt_total_overflow": total, "grt_congestion_bins": bins,
            "grt_wire_length_um": float(lengths[-1]), "grt_native_runtime_seconds": h * 3600 + m * 60 + s}


def validate_grt_reference(run, config, baseline_run=None, upstream_stages=(), *, hook_spec=None, source_root=None):
    """Validate standalone GRT, never substitute completed DRT metrics."""
    run = Path(run).resolve(strict=True)
    root = run / "80_route"
    checkpoint = read_json(root / "global_route_checkpoint.json")
    require(checkpoint.get("stage") == "global_route" and checkpoint.get("status") == "success"
            and checkpoint.get("ready_for_detailed_route") is True and checkpoint.get("ready_for_extract") is False,
            "Not a standalone GRT checkpoint")
    hook_identity = None
    if checkpoint.get("image_id") != IMAGE_ID and hook_spec is not None and source_root is not None:
        from .routing_reference import validate_initial_hook_image
        hook_identity = validate_initial_hook_image(run, hook_spec, source_root)
        require(checkpoint.get("image_id") == hook_identity["image_id"], "GRT initial hook checkpoint image differs")
    else:
        validate_reference_image(checkpoint.get("image_id"))
    require(checkpoint.get("metrics") == "grt_metrics.json", "GRT metric publication mismatch")
    source = (root / checkpoint.get("source_checkpoint", "")).resolve(strict=True)
    require(source == (run / "70_cts/checkpoint.json").resolve(strict=True)
            and sha256(source) == checkpoint.get("source_checkpoint_sha256"), "GRT CTS handoff changed")
    validate_artifacts(root, checkpoint.get("artifacts"), containment=run)
    metrics = read_json(root / "grt_metrics.json")
    require(metrics.get("stage") == "global_route" and metrics.get("status") == "success"
            and metrics.get("ready_for_detailed_route") is True and metrics.get("ready_for_extract") is False,
            "Not standalone GRT metrics")
    require(set(metrics.get("dies", {})) == {"bottom", "top"}
            and set(checkpoint.get("dies", {})) == {"bottom", "top"}, "GRT must contain both dies")
    expected = {"threads": min(config["routing"].get("threads", config["threads"]), 32),
                "congestion_iterations": config["routing"]["congestion_iterations"],
                "resource_reduction": config["routing"]["resource_reduction"]}
    require(all(metrics.get("configuration", {}).get(k) == v for k, v in expected.items()), "GRT scientific budget mismatch")
    for die in ("bottom", "top"):
        record = checkpoint["dies"][die]
        require(record == {"odb": die + "/global_route.odb", "guide": die + "/grt.guide",
                           "manifest": die + "/grt_manifest.json"}, "GRT die handoff differs")
        manifest = read_json(root / record["manifest"])
        require(manifest.get("stage") == "global_route" and manifest.get("status") == "success", "Unsuccessful GRT die")
        validate_artifacts(root / die, manifest.get("artifacts"), containment=run)
        require(manifest.get("signature") == sha256(root / die / "grt.tcl"), "GRT script signature differs")
        observed = _native_grt_metrics(root / die)
        require(manifest.get("metrics") == observed, "GRT native log metrics differ")
        row = metrics["dies"][die]
        require(all(row.get(key) == value for key, value in observed.items()), "GRT publication differs from native log")
        for key in ("grt_total_overflow", "grt_congestion_bins", "grt_wire_length_um", "runtime_seconds"):
            _finite(row.get(key), die + "." + key)
        require(row["grt_wire_length_um"] > 0, "GRT wire length must be positive")
    for key in ("grt_total_overflow", "grt_wire_length_um", "runtime_seconds"):
        require(math.isclose(_finite(metrics.get(key), key), sum(metrics["dies"][d][key] for d in ("bottom", "top")), rel_tol=1e-12), "GRT aggregate differs: " + key)
    upstream = {}
    for stage in upstream_stages:
        current = validate_stage(run, stage)["manifest_sha256"]
        require(baseline_run is not None, "GRT reference requires frozen upstream")
        require(current == validate_stage(baseline_run, stage)["manifest_sha256"], "GRT frozen upstream mismatch: " + stage)
        upstream[stage] = current
    return {"status": "passed", "baseline_id": BASELINE_ID, "endpoint": "global-route",
            "metrics": str(root / "grt_metrics.json"), "checkpoint_sha256": sha256(root / "global_route_checkpoint.json"),
            "routing_budget": expected, "upstream_manifest_sha256": upstream, "raw_metrics": metrics,
            "initial_hook_identity": hook_identity,
            "signoff": False}


def validate_reference_bundle(bundle, case, *, fixed_partition=False):
    """Accept only a new, hash-bound import receipt; never relabel an old bundle."""
    from .benchmark_input import validate_benchmark_input
    bundle = Path(bundle)
    validated = validate_benchmark_input(bundle)
    require(Path(validated["config"]).resolve() == (profile_path(case) / "config.yaml").resolve(), "Wrong latest input profile")
    manifest = read_json(bundle / "benchmark_input.json" if bundle.is_dir() else bundle)
    require(manifest.get("baseline_id") == BASELINE_ID, "Missing explicit latest input family")
    ref = manifest.get("latest_reference_receipt", {})
    path = Path(ref["path"])
    require(path.is_file() and not path.is_symlink() and sha256(path) == ref.get("sha256"), "Latest receipt changed")
    receipt = read_json(path)
    require(receipt.get("baseline_id") == BASELINE_ID and receipt.get("case") == case
            and receipt.get("source_revision") == SOURCE_REVISION, "Latest reference identity mismatch")
    validate_reference_image(receipt.get("image"))
    require(receipt.get("config_sha256") == sha256(profile_path(case) / "config.yaml"),
            "Latest reference receipt recipe differs")
    require(sha256(Path(receipt["source_run"]) / "config.yaml") == receipt.get("native_config_sha256"),
            "Latest native reference configuration changed")
    require(sha256(Path(receipt["source_run"]) / "run.json") == receipt.get("native_run_sha256"),
            "Latest native reference ledger changed")
    runtime = receipt.get("runtime_identity", {})
    require(runtime.get("canonical_image") == IMAGE_ID
            and sha256(runtime["execution_receipt"]) == runtime.get("execution_receipt_sha256"),
            "Latest runtime execution identity changed")
    if runtime.get("actual_docker_image_id") != IMAGE_ID:
        require(sha256(runtime["equivalence_proof"]) == runtime.get("equivalence_proof_sha256"),
                "Latest independent runtime equivalence proof changed")
    validate_execution_proof(runtime["execution_proof"], runtime["execution_proof_sha256"],
                             case, receipt["source_run"], runtime["actual_docker_image_id"])
    run = Path(validated["run_dir"])
    require(sha256(run / "run.json") == receipt.get("imported_run_sha256")
            and sha256(run / "config.yaml") == receipt.get("native_config_sha256"),
            "Imported prefix ledger or configuration changed")
    checked = {}
    for stage in ("input", "floorplan", "partition") if fixed_partition else ("input", "floorplan"):
        result = validate_stage(run, stage)
        require(result["manifest_sha256"] == receipt.get("stage_manifest_sha256", {}).get(stage), "Latest frozen prefix mismatch")
        checked[STAGE_DIRS[stage] + "/stage_manifest.json"] = result["manifest_sha256"]
        for relative, record in result["manifest"]["artifacts"].items():
            checked[STAGE_DIRS[stage] + "/" + record.get("path", relative)] = record["sha256"]
    return {"baseline_id": BASELINE_ID, "reference_root": receipt["source_run"],
            "original_root": receipt["source_run"], "physical_input_sha256": checked,
            "fixed_partition": fixed_partition, "scoring_reference_qualified": False,
            "receipt_sha256": sha256(path)}


def baseline_for_bundle(bundle):
    """Explicit family dispatch; no image- or path-based silent fallback."""
    from . import reference_baseline
    import sys
    path = Path(bundle)
    data = read_json(path / "benchmark_input.json" if path.is_dir() else path)
    require(data.get("baseline_id") in (None, reference_baseline.BASELINE_ID, BASELINE_ID),
            "Unknown explicit reference family")
    return sys.modules[__name__] if data.get("baseline_id") == BASELINE_ID else reference_baseline


def baseline_for_config(config):
    from . import reference_baseline
    import sys
    value = config.get("partition", {}).get("config")
    return sys.modules[__name__] if value == "AgenticBench/local_bench/reference_configs/partition.json" else reference_baseline


def validate_execution_proof(path, expected_sha256, case, source_run, actual_image):
    """Bind independently collected controller/Docker evidence, not model claims."""
    path = Path(path).resolve(strict=True)
    require(not path.is_symlink() and sha256(path) == expected_sha256, "Independent execution proof changed")
    proof = read_json(path)
    require(all(proof.get(key) == value for key, value in {
        "schema_version": 1, "kind": "latest-reference-execution-proof-v1", "status": "verified",
        "case": case, "source_run": str(Path(source_run).resolve(strict=True)),
        "source_revision": SOURCE_REVISION, "native_revision": NATIVE_REVISION,
        "sta_revision": STA_REVISION, "canonical_image": IMAGE_ID,
        "actual_docker_image_id": actual_image, "binaries_sha256": BINARY_SHA256}.items()),
        "Independent native runtime identity differs")
    container = proof.get("container", {})
    require(isinstance(container.get("id"), str) and re.fullmatch(r"[0-9a-f]{64}", container["id"])
            and container.get("state") == "exited" and type(container.get("exit_code")) is int
            and container["exit_code"] == 0 and container.get("oom_killed") is False,
            "Native container did not independently reach successful terminal state")
    bindings = proof.get("bindings", {})
    require(set(bindings) == {"controller_plan", "dispatch_receipt", "container_result", "runtime_identity"},
            "Execution proof lacks controller/dispatch/runtime evidence")
    paths = []
    for key, record in bindings.items():
        evidence = Path(record["path"])
        require(evidence.is_absolute() and evidence.is_file() and not evidence.is_symlink()
                and sha256(evidence) == record.get("sha256"), "Execution evidence changed: " + key)
        paths.append(evidence.resolve())
    require(len(set(paths)) == 4 and path not in paths, "Execution proof uses duplicate/self evidence")
    return proof


def adopt_latest_input(source_run, output, case, source_root, *, execution_proof=None, execution_proof_sha256=None,
                       runtime_proof=None, runtime_proof_sha256=None):
    """Create a new input bundle only from a complete, independently checked GRT run.

    Source artifacts remain byte-identical. This import is not local-hook
    qualification, a native dispatch, or registry publication.
    """
    from scripts.workspace_latest_reference import normalized_recipe, validate_latest_chain
    from .reference_local_placement import _source_revision
    from .benchmark_input import validate_benchmark_input
    source_run, source_root = Path(source_run).resolve(strict=True), Path(source_root).resolve(strict=True)
    output = Path(output).absolute()
    require(not output.exists(), "Latest input output must be a new directory")
    require(case in CASES and _source_revision(source_root, clean=True) == SOURCE_REVISION,
            "Latest input importer requires its clean fixed physical source")
    status = validate_latest_chain(source_run, reference_suffix=True)
    require((status.get("design"), status.get("type"), status.get("pdk")) == (case, "lol", "ng45"),
            "Reference design/integration/PDK differs")
    native = yaml.safe_load((source_run / "config.yaml").read_text())
    normalized = copy.deepcopy(native)
    expected = normalized_recipe(case)
    for domain, relative in (("partition", PARTITION_CONFIG), ("placement", PLACEMENT_CONFIG)):
        path = Path(normalized[domain]["config"])
        trusted = Path(__file__).resolve().parent / "reference_configs" / (domain + ".json")
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and sha256(path) == sha256(trusted),
                "Native reference recipe bytes differ: " + domain)
        normalized[domain]["config"] = relative
    for key in ("run_dir", "run_id"):
        normalized.pop(key, None)
        expected.pop(key, None)
    require(normalized == expected, "Native reference scientific configuration differs")
    result_path = source_run / "reference-result.json"
    result = read_json(result_path)
    require(all(result.get(k) == v for k, v in {"status": "success", "endpoint": "global-route",
        "fresh_input": True, "source_commit": SOURCE_REVISION, "image_id": IMAGE_ID,
        "drt_dispatches": 0, "signoff": False, "executed_stages": list(STAGE_DIRS) + ["global_route"]}.items()),
        "Reference lacks a successful explicit GRT-only execution receipt")
    actual_image = result.get("actual_docker_image_id")
    require(execution_proof is not None and execution_proof_sha256 is not None,
            "Latest import requires independently collected controller/Docker execution proof")
    validate_execution_proof(execution_proof, execution_proof_sha256, case, source_run, actual_image)
    runtime = {"canonical_image": IMAGE_ID, "actual_docker_image_id": actual_image,
               "execution_receipt": str(result_path), "execution_receipt_sha256": sha256(result_path),
               "execution_proof": str(Path(execution_proof).resolve()), "execution_proof_sha256": execution_proof_sha256}
    if actual_image != IMAGE_ID:
        require(runtime_proof is not None and runtime_proof_sha256 is not None, "Noncanonical Docker ID requires independent runtime equivalence proof")
        path = Path(runtime_proof).resolve(strict=True)
        require(not path.is_symlink() and sha256(path) == runtime_proof_sha256, "Runtime proof changed")
        proof = read_json(path)
        require(proof.get("status") == "verified" and proof.get("canonical_oci_digest") == IMAGE_ID
                and proof.get("docker_config_image_id") == actual_image and bool(proof.get("rootfs_layers"))
                and proof.get("binaries_sha256") == BINARY_SHA256,
                "Runtime equivalence proof does not bind both installed binaries")
        runtime.update(equivalence_proof=str(path), equivalence_proof_sha256=runtime_proof_sha256)
    grt = validate_grt_reference(source_run, native)
    stages = {}
    for stage in STAGE_DIRS:
        checked = validate_stage(source_run, stage)
        provenance = checked["manifest"].get("provenance", {})
        require(provenance.get("repo_commit") == SOURCE_REVISION and provenance.get("repo_dirty") is False
                and provenance.get("container_image_id") == IMAGE_ID, "Native reference stage identity differs: " + stage)
        stages[stage] = checked["manifest_sha256"]
    # Source aliases cannot escape the run, even when not manifest-registered.
    for stage in ("input", "floorplan", "partition"):
        for path in (source_run / STAGE_DIRS[stage]).rglob("*"):
            if path.is_symlink():
                require(not Path(os.readlink(path)).is_absolute() and _inside(path.resolve(strict=True), source_run),
                        "Imported input alias escapes its original run")
    output.mkdir(parents=True)
    run = output / "run"
    run.mkdir()
    records = {}
    for stage in ("input", "floorplan", "partition"):
        directory = STAGE_DIRS[stage]
        shutil.copytree(source_run / directory, run / directory, symlinks=True)
        checked = validate_stage(run, stage)
        require(checked["manifest_sha256"] == stages[stage], "Copied reference stage changed")
        records[stage] = {"directory": directory, "manifest_sha256": stages[stage],
                          "artifact_count": len(checked["manifest"]["artifacts"])}
    # Preserve producer metadata, but publish only the three actually imported
    # stages. Import is not a new native execution or a completed suffix.
    imported_status = copy.deepcopy(status)
    imported_status["status"] = "imported-prefix"
    imported_status["stages"] = {stage: copy.deepcopy(status["stages"][stage])
                                 for stage in ("input", "floorplan", "partition")}
    imported_status["reference_import"] = {"source_run": str(source_run),
        "source_run_sha256": sha256(source_run / "run.json"),
        "imported_stages": ["input", "floorplan", "partition"], "executed_stages": []}
    (run / "run.json").write_text(json.dumps(imported_status, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(source_run / "config.yaml", run / "config.yaml")
    receipt = {"schema_version": 1, "baseline_id": BASELINE_ID, "case": case,
        "source_run": str(source_run), "source_revision": SOURCE_REVISION, "image": IMAGE_ID,
        "native_revision": NATIVE_REVISION, "config_sha256": sha256(profile_path(case) / "config.yaml"),
        "native_config_sha256": sha256(source_run / "config.yaml"), "native_run_sha256": sha256(source_run / "run.json"),
        "imported_run_sha256": sha256(run / "run.json"),
        "stage_manifest_sha256": stages, "grt_checkpoint_sha256": grt["checkpoint_sha256"],
        "runtime_identity": runtime, "native_eda_dispatches": 0, "model_calls": 0, "registry_publications": 0,
        "scoring_reference_qualified": False}
    receipt_path = output / "latest-reference-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    identity = {"root": str(source_root), "revision": SOURCE_REVISION, "dirty": False}
    manifest = {"schema_version": 1, "baseline_id": BASELINE_ID,
        "benchmark_id": profile_path(case).name, "run_dir": "run",
        "profile_sha256": sha256(profile_path(case) / "profile.yaml"),
        "config_sha256": sha256(profile_path(case) / "config.yaml"), "stages": records,
        "environment": {"open3dbench": identity, "current_importer_source": identity},
        "current_importer_source": identity,
        "latest_reference_receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path)}}
    (output / "benchmark_input.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_benchmark_input(output)
    return validate_reference_bundle(output, case, fixed_partition=True)


PARTITION_CONFIG = "AgenticBench/local_bench/reference_configs/partition.json"
PLACEMENT_CONFIG = "AgenticBench/local_bench/reference_configs/placement.json"
