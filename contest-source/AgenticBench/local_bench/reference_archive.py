"""Read-only adoption of the *historical* 2026-09-23 measured reference.

This is a producer-specific proof reader, not a general resume bypass.  It
returns facts for a trusted registry to seal; it never runs tools, changes an
archive, fabricates stages, or certifies a new hook/runtime as equivalent.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re

from . import reference_baseline as baseline
from .placement_baseline import validate_tp_gnn_partition


PRODUCER_PROFILE = "frozen-20260923-v1"
FILES_SHA256 = "c7c3542898fc165b0db579f8234fdc5255a5190a84c30c35e79794519db172bd"
THERMAL_RECEIPT_SHA256 = {
    "bp_fe": "3f488f4eed8ab2588be73faee0d40abe0438f66db10e80adc70bbfeba7bef381",
    "bp": "7fc67ecc082141a785223d66ee45064a20319460923bf942041874263ac783e5",
    "ariane133": "faddb0929cf1a9efc3adf9dd851b89f28fd29ffc4a3a64f002fa7d5da8b46eb3",
    "swerv_wrapper": "71ec6be2b370f1316424902360712d18362abc1e55bd9207f4868201fab31bc5",
}
STAGES = {"input": "00_input", "floorplan": "10_floorplan", "partition": "20_partition",
          "placement": "30_placement", "pdn": "40_pdn", "hbt": "50_hbt",
          "opt": "60_opt", "cts": "70_cts"}
ENDPOINT_STAGE = {"placement": "placement", "timing": "opt", "cts_thermal": "cts", "grt": "cts"}


def _read(path, expected_sha256=None):
    content = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Archive changed between validation and metadata read: " + str(path))
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Missing/nonfinite native metric: " + label)
    return float(value)


def _relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Missing archive-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("Unsafe archive-relative path: " + value)
    return path


def _regular(root, relative):
    path = root / _relative(relative)
    # Reject links in every component, not only the final basename.  Historical
    # log links are handled explicitly by the authenticated inventory below.
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Unexpected symlink in archive path: " + str(path))
    if not path.is_file():
        raise ValueError("Missing regular archive file: " + str(path))
    return path


def _record(path, role):
    before = path.stat()
    digest = baseline.sha256(path)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Archive changed during verification: " + str(path))
    return {"path": str(path), "bytes": after.st_size, "sha256": digest, "role": role}


class _Archive:
    def __init__(self, root):
        raw = Path(root).expanduser()
        if raw.is_symlink():
            raise ValueError("Reference root must not be a symlink")
        self.root = raw.resolve(strict=True)
        self.evidence = {}
        path = _regular(self.root, "FILES.json")
        record = _record(path, "authenticated-archive-inventory")
        if record["sha256"] != FILES_SHA256:
            raise ValueError("Frozen archive FILES.json identity differs")
        self.evidence[str(path)] = record
        self.inventory = _read(path, record["sha256"])

    def file(self, relative, role="native-artifact", _seen=None):
        relative = str(_relative(relative))
        row = self.inventory.get(relative)
        if not isinstance(row, dict):
            raise ValueError("File is not bound by the pinned archive inventory: " + relative)
        path = self.root / relative
        if row.get("kind") == "symlink":
            seen = set() if _seen is None else set(_seen)
            if relative in seen:
                raise ValueError("Cyclic archive symlink")
            seen.add(relative)
            target = row.get("target")
            if (not isinstance(target, str) or Path(target).is_absolute()
                    or not path.is_symlink() or os.readlink(path) != target):
                raise ValueError("Archive symlink identity differs")
            try:
                dest = path.resolve(strict=True).relative_to(self.root).as_posix()
            except (ValueError, RuntimeError) as exc:
                raise ValueError("Archive symlink escapes reference") from exc
            return self.file(dest, role, seen)
        if row.get("kind") != "file":
            raise ValueError("Unsupported archive inventory entry")
        path = _regular(self.root, relative)
        record = _record(path, role)
        if record["bytes"] != row.get("bytes") or record["sha256"] != row.get("sha256"):
            raise ValueError("Frozen archive bytes/hash differ: " + relative)
        self.evidence[str(path)] = record
        return path

    def read(self, relative, role="native-metadata"):
        path = self.file(relative, role)
        return _read(path, self.evidence[str(path)]["sha256"])

    def resolve_reference(self, value, parent):
        if not isinstance(value, str) or not value:
            raise ValueError("Missing checkpoint reference")
        path = Path(value)
        if path.is_absolute():
            original = Path(self.index["original_root"])
            try:
                path = self.root / path.relative_to(original)
            except ValueError as exc:
                raise ValueError("Checkpoint uses an unknown absolute root") from exc
        else:
            path = parent / path
        try:
            relative = path.resolve(strict=True).relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError("Checkpoint reference escapes archive") from exc
        return self.file(relative, "checkpoint-dependency")


def _stage_chain(archive, run, case, last):
    relative_run = run.relative_to(archive.root).as_posix()
    ledger = archive.read(relative_run + "/run.json", "historical-run-ledger")
    if ledger.get("design") != case or ledger.get("type") != "lol" or ledger.get("pdk") != "ng45":
        raise ValueError("Historical design/integration/platform mismatch")
    stages, previous = {}, None
    for stage, directory in STAGES.items():
        prefix = relative_run + "/" + directory
        manifest = archive.read(prefix + "/stage_manifest.json", "native-stage-manifest")
        root_row = ledger.get("stages", {}).get(stage, {})
        if (root_row.get("status") != "success" or root_row.get("directory") != directory
                or manifest.get("status") != "success" or manifest.get("stage") != stage
                or manifest.get("schema_version") != 1 or not isinstance(manifest.get("signature"), dict)
                or not manifest.get("signature")):
            raise ValueError("Historical stage did not qualify: " + stage)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("Native stage lacks artifact proof: " + stage)
        for row in artifacts.values():
            if not isinstance(row, dict):
                raise ValueError("Invalid stage artifact record")
            path = archive.file(prefix + "/" + str(_relative(row.get("path"))))
            verified = archive.evidence[str(path)]
            if verified["bytes"] != row.get("bytes") or verified["sha256"] != row.get("sha256"):
                raise ValueError("Native stage artifact record differs")
        if stage in {"placement", "pdn", "hbt", "opt", "cts"}:
            checkpoint_path = archive.file(prefix + "/checkpoint.json", "native-checkpoint")
            checkpoint = _read(checkpoint_path, archive.evidence[str(checkpoint_path)]["sha256"])
            if (checkpoint.get("schema_version") != 1 or checkpoint.get("stage") != stage
                    or checkpoint.get("pdk") != "ng45" or set(checkpoint.get("dies", {})) != {"bottom", "top"}):
                raise ValueError("Historical checkpoint identity differs: " + stage)
            if previous is not None and archive.resolve_reference(
                    checkpoint.get("source_checkpoint"), checkpoint_path.parent) != previous:
                raise ValueError("Historical checkpoint predecessor differs: " + stage)
            for die in ("bottom", "top"):
                for name in ("odb", "def", "verilog", "sdc"):
                    archive.resolve_reference(checkpoint["dies"][die].get(name), checkpoint_path.parent)
            previous = checkpoint_path
        stages[stage] = {"directory": directory, "manifest_sha256": baseline.sha256(run / directory / "stage_manifest.json"),
                         "runtime_seconds": manifest.get("runtime_seconds"),
                         "runtime_provenance": "historical-not-new-execution"}
        if stage == last:
            break
    # A failed later DRT stage does not invalidate successful earlier stages.
    return ledger, stages


def _placement(archive, run):
    prefix = run.relative_to(archive.root).as_posix() + "/30_placement/stages/02_global_place/"
    manifest = archive.read(prefix + "stage_manifest.json")
    if manifest.get("returncode") != 0 or manifest.get("initialization_seed") != 1000:
        raise ValueError("Historical global placement failed or seed differs")
    params = archive.read(prefix + "params.json")
    archive.file(prefix + "placement.gp.def", "native-global-placement")
    log = archive.file(prefix + "run.log", "native-placement-measurements")
    # Unlike legacy permissive parsers, never turn missing/bad overflow into 0.
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    final = None
    for line in log.read_text(encoding="utf-8", errors="strict").splitlines():
        hpwl = re.search(r"wHPWL\s+(" + number + r")", line)
        overflow = re.search(r"(?<!Global )Overflow\s*\[([^\]]+)\]", line)
        density = re.search(r"MaxDensity\s*\[([^\]]+)\]", line)
        if hpwl and overflow and density:
            try:
                values = [float(part) for part in overflow[1].replace(",", " ").split()]
                densities = [float(part) for part in density[1].replace(",", " ").split()]
                final = (float(hpwl[1]), values, densities)
            except ValueError as exc:
                raise ValueError("Invalid final placement metric row") from exc
    if final is None or len(final[1]) != 2 or len(final[2]) != 2:
        raise ValueError("Missing native two-die placement metrics")
    for value in [final[0], *final[1], *final[2]]:
        if _finite(value, "placement") < 0:
            raise ValueError("Negative native placement metric")
    metrics = {"placement_native_hpwl": final[0], "placement_overflow_bottom": final[1][0],
               "placement_overflow_top": final[1][1], "placement_overflow_max": max(final[1]),
               "placement_overflow_mean": sum(final[1]) / 2, "placement_max_density": max(final[2])}
    return metrics, {"native_placement_parameters": params, "placement_initialization_seed": 1000}


def _timing(archive, run):
    from global_bench.verifiers.open3dbench_codesign_verifier import _validate_opt_passes
    relative = run.relative_to(archive.root).as_posix() + "/60_opt/metrics.json"
    opt = archive.read(relative, "native-post-opt-metrics")
    if opt.get("status") != "success" or opt.get("algorithm") not in {
            "openroad-coordinated-passive-cutpoint", "openroad-coordinated-passive-boundary"}:
        raise ValueError("Historical OPT producer differs")
    _validate_opt_passes(opt, 1)
    required = {"authoritative_timing_view": "continuous-passive-hbt", "graph_completeness": "passed",
                "die_internal_parasitics_in_global_sta": True, "hbt_is_passive_rc_in_authoritative_global_sta": True,
                "remote_die_load_visible_to_source_driver": True, "unclocked_or_unconstrained_endpoints": 0,
                "missing_liberty_cells": 0, "combinational_loops": 0}
    if any(opt.get("timing_validity", {}).get(key) != value for key, value in required.items()):
        raise ValueError("Historical OPT timing validity failed")
    final = opt.get("final_global", {})
    if (opt.get("final_boundary_coverage", {}).get("sta_unconstrained_endpoints") != 0
            or final.get("sta_completeness_checks_passed") != 1 or final.get("passive_hbt_rc") != 1):
        raise ValueError("Historical OPT timing coverage failed")
    wns = _finite(final.get("wns_setup"), "WNS")
    tns = _finite(final.get("tns_setup"), "TNS")
    return {"wns_setup_ns": wns, "tns_setup_ns": tns,
            "wns_deficit": max(0.0, -wns), "tns_deficit": max(0.0, -tns),
            "boundary_budget_model": opt.get("boundary_budget_model", "passive-cutpoint"),
            "opt_passes": 1}


def _grt(archive, run, case_record):
    path = archive.file(case_record["qualified_grt_checkpoint"], "qualified-grt-checkpoint")
    if baseline.sha256(path) != case_record.get("qualified_grt_checkpoint_sha256"):
        raise ValueError("Historical GRT checkpoint identity differs")
    checkpoint = _read(path, archive.evidence[str(path)]["sha256"])
    source = archive.resolve_reference(checkpoint.get("source_checkpoint"), path.parent)
    if (checkpoint.get("status") != "success" or checkpoint.get("ready_for_extract") is not False
            or checkpoint.get("ready_for_detailed_route") is not True
            or source != run / "70_cts/checkpoint.json"
            or baseline.sha256(source) != checkpoint.get("source_checkpoint_sha256")):
        raise ValueError("Historical GRT handoff differs")
    for manifest_path in (path, path.parent / "bottom/grt_manifest.json", path.parent / "top/grt_manifest.json"):
        payload = archive.read(manifest_path.relative_to(archive.root).as_posix(), "native-grt-manifest")
        if not isinstance(payload.get("artifacts"), dict) or not payload["artifacts"]:
            raise ValueError("GRT lacks artifact proof")
        for name, record in payload["artifacts"].items():
            file = archive.file((manifest_path.parent / _relative(name)).relative_to(archive.root).as_posix())
            if file.stat().st_size != record.get("size_bytes") or baseline.sha256(file) != record.get("sha256"):
                raise ValueError("GRT artifact differs")
    raw = archive.read((path.parent / "grt_metrics.json").relative_to(archive.root).as_posix(), "native-grt-metrics")
    if (raw.get("status") != "success" or raw.get("stage") != "global_route"
            or raw.get("ready_for_extract") is not False or raw != case_record.get("grt_metrics")):
        raise ValueError("GRT measurement differs from pinned index")
    if set(raw.get("dies", {})) != {"bottom", "top"}:
        raise ValueError("GRT measurement lacks both dies")
    result = {}
    for key in ("grt_total_overflow", "grt_wire_length_um"):
        result[key] = _finite(raw.get(key), key)
        measured = sum(_finite(raw["dies"][die].get(key), key) for die in ("bottom", "top"))
        if result[key] < 0 or not math.isclose(result[key], measured, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("GRT two-die aggregation differs")
    return result, raw["configuration"], raw.get("runtime_seconds")


def _historical_runtime(stages, endpoint, extra=None):
    """Describe recorded wall time, not validation time or a new execution.

    Imported frontend stages were originally run earlier than the backend.
    Therefore the sum is component wall time, not a contiguous run duration.
    A missing duration remains unknown and prevents a claimed complete total.
    """
    components = {row["directory"]: row.get("runtime_seconds") for row in stages.values()}
    components.update(extra or {})
    for key, value in components.items():
        if value is not None and _finite(value, key + " runtime") < 0:
            raise ValueError("Negative historical native runtime: " + key)
    unknown = [key for key, value in components.items() if value is None]
    known = [value for value in components.values() if value is not None]
    return {"unit": "seconds", "provenance": "historical-native-producer-records",
            "scope": "successful-physical-prefix-through-" + ENDPOINT_STAGE[endpoint],
            "components": components,
            "unknown_components": unknown,
            "sum_recorded_component_seconds": sum(known) if known else None,
            "total_component_seconds": None if unknown else sum(components.values()),
            "not_contiguous_elapsed_time": True,
            "endpoint_notes": {
                "placement": "30_placement includes global placement, legalization and die initialization; standalone global-placement API time is not inferred",
                "timing": "one OPT pass; pre-CTS STA, not routed/signoff timing",
                "grt": "global_route adds the retained GRT wrapper runtime; failed historical detailed routing is excluded",
                "cts_thermal": "cts_thermal_snapshot adds the receipt total including input-view/proof preparation, cell power and HotSpot; not routed thermal",
            }[endpoint]}


def _thermal(archive, run, case, thermal_root):
    if thermal_root is None:
        raise ValueError("CTS thermal requires the full separately retained thermal archive")
    raw_root = Path(thermal_root).expanduser()
    if raw_root.is_symlink():
        raise ValueError("Thermal root must not be a symlink")
    root = raw_root.resolve(strict=True)
    receipt_path = _regular(root, case + "/receipt.json")
    receipt_record = _record(receipt_path, "pinned-independent-thermal-receipt")
    if receipt_record["sha256"] != THERMAL_RECEIPT_SHA256[case]:
        raise ValueError("CTS thermal receipt identity differs")
    archive.evidence[str(receipt_path)] = receipt_record
    receipt = _read(receipt_path, receipt_record["sha256"])
    if (receipt.get("status") != "success" or receipt.get("role") != "diagnostic-cts-vectorless-placement-rc-thermal-snapshot"
            or receipt.get("settings") != {"ambient_k": 318.15, "grid_size": 10, "input_activity": 0.1, "threads": 2}):
        raise ValueError("Historical CTS thermal producer differs")
    for key, relative in (("original_checkpoint", "70_cts/checkpoint.json"),
                          ("original_manifest", "70_cts/stage_manifest.json"),
                          ("upstream_opt_checkpoint", "60_opt/checkpoint.json")):
        if baseline.sha256(run / relative) != receipt.get(key, {}).get("sha256"):
            raise ValueError("Thermal receipt is bound to a different physical input")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("Thermal receipt has no native output inventory")
    for relative, expected in outputs.items():
        path = _regular(root, case + "/" + str(_relative(relative)))
        record = _record(path, "native-thermal-output")
        if record["bytes"] != expected.get("bytes") or record["sha256"] != expected.get("sha256"):
            raise ValueError("Thermal output hash differs: " + relative)
        archive.evidence[str(path)] = record
    metrics = _read(root / case / "thermal/metrics.json", outputs["thermal/metrics.json"]["sha256"])
    snapshot = _read(root / case / "thermal/snapshot_manifest.json", outputs["thermal/snapshot_manifest.json"]["sha256"])
    expected_view = receipt.get("input_view_checkpoint", {}).get("path")
    power = metrics.get("power", {})
    if (snapshot.get("status") != "success" or snapshot.get("source_stage") != "cts"
            or snapshot.get("source_checkpoint") != expected_view
            or power.get("provenance", {}).get("checkpoint") != expected_view
            or power.get("activity_model") != "vectorless-primary-input-0.1-placement-estimated-rc"
            or power.get("provenance", {}).get("model") != "die-local-openroad-placement-estimated"
            or power.get("provenance", {}).get("threads") != 2
            or metrics != receipt.get("audit", {}).get("metrics")
            or metrics.get("status") != "success" or metrics.get("stage") != "thermal_snapshot"
            or metrics.get("algorithm") != "hotspot-steady-grid"
            or metrics.get("warning_count") != 0 or metrics.get("grid_size") != 10
            or metrics.get("ambient_k") != 318.15 or snapshot.get("schema_version") != 1
            or snapshot.get("grid_size") != 10 or snapshot.get("ambient_k") != 318.15
            or snapshot.get("metrics") != "metrics.json"):
        raise ValueError("Thermal native endpoint/provenance differs")
    temperatures = metrics.get("temperature", {})
    per_die = []
    for die in ("bottom", "top"):
        row = temperatures.get(die, {})
        extraction = metrics.get("power_extraction", {}).get(die, {})
        average = _finite(row.get("average_k"), "die temperature")
        if (row.get("tile_count") != 100 or average < 318.15
                or extraction.get("warning_count") != 0 or extraction.get("instance_count", 0) <= 0):
            raise ValueError("Invalid historical physical thermal tiles/power")
        per_die.append(average)
    average = _finite(temperatures.get("global_average_k"), "thermal mean")
    total = _finite(power.get("total_w"), "power")
    error = _finite(power.get("conservation_error_w"), "power conservation")
    if (not math.isclose(average, sum(per_die) / 2, abs_tol=1e-8, rel_tol=0)
            or total <= 0 or error < 0 or error > max(1e-12, total * 1e-9)):
        raise ValueError("Historical thermal mean/power conservation failed")
    cts = archive.read((run / "70_cts/metrics.json").relative_to(archive.root).as_posix())
    checkpoint = archive.read((run / "70_cts/checkpoint.json").relative_to(archive.root).as_posix())
    if (cts.get("status") != "success" or cts.get("algorithm") not in {
            "openroad-balanced", "openroad-coordinated-insertion-delay"}
            or checkpoint.get("ready_for_route") is not True):
        raise ValueError("Historical CTS native endpoint failed")
    opt = archive.read((run / "60_opt/metrics.json").relative_to(archive.root).as_posix())
    return {"global_average_temperature_k": average,
            "global_average_temperature_c": average - 273.15,
            "total_power_w": total, "stack": metrics.get("stack"),
            "cts_algorithm": cts["algorithm"],
            "boundary_budget_model": opt.get("boundary_budget_model", "passive-cutpoint")}, receipt


def prepare_archive_reference(reference_root, case, endpoint, *, thermal_root=None):
    """Validate a pinned historical endpoint, returning an unsealed fact record.

    ``thermal_root`` is the directory containing ``<case>/receipt.json`` and
    the *complete* corresponding native outputs, not the small local evidence
    subset.  No accepted-current-runtime or no-op-hook claim is made here.
    """
    if endpoint not in ENDPOINT_STAGE:
        raise ValueError("Unsupported archived endpoint (routed thermal is not qualified): " + str(endpoint))
    if case not in baseline.identity()["cases"]:
        raise ValueError("Case has no pinned historical reference")
    archive = _Archive(reference_root)
    run = baseline.validate_reference_root(archive.root, case)
    archive.index = archive.read("reference.json", "pinned-reference-index")
    index = archive.index
    case_record = index["cases"][case]
    work = archive.root / _relative(case_record.get("work"))
    if work / "run" != run:
        raise ValueError("Frozen work/run identity differs")
    requested_config = archive.file((work / "config.yaml").relative_to(archive.root).as_posix(),
                                    "historical-requested-config")
    executed_config = archive.file((run / "config.yaml").relative_to(archive.root).as_posix(),
                                   "historical-normalized-executed-config")
    # This producer's case index hashes WORK/config.yaml. RUN/config.yaml is
    # the separately inventory-pinned normalized configuration with defaults.
    # These are not interchangeable files and must not be normalized in place.
    if baseline.sha256(requested_config) != case_record.get("config_sha256"):
        raise ValueError("Frozen requested configuration differs")
    provenance = archive.read(index["source_provenance"], "historical-source-provenance")
    image = archive.read(index["image"]["full_layer_and_tool_evidence"], "historical-image-evidence")
    ledger, stages = _stage_chain(archive, run, case, ENDPOINT_STAGE[endpoint])
    prefix = run.relative_to(archive.root).as_posix()
    for name in ("partition_manifest.json", "tp_gnn_report.json", "timing_input/timing_manifest.json"):
        archive.file(prefix + "/20_partition/" + name, "native-flat-timing-partition-proof")
    for relative in ("10_floorplan/io_metrics.json", "20_partition/partition_metrics.json",
                     "30_placement/metrics.json", "30_placement/stages/02_global_place/stage_manifest.json"):
        archive.file(prefix + "/" + relative, "native-frontend-proof")
    validate_tp_gnn_partition(run, expected_method="tp-gnn-flat", require_bound_report=True)
    from global_bench.verifiers.open3dbench_codesign_verifier import _validate_seed_contract, _validate_frontend
    _validate_seed_contract(run, include_routing=False)
    _validate_frontend(run)
    metrics, placement_contract = _placement(archive, run)
    contract = {"design": case, "type": "lol", "pdk": "ng45", "endpoint": endpoint,
                "partition_algorithm": "tp-gnn-flat", "partition_decoder": "global-2way",
                "fixed_partition_sha256": baseline.identity()["cases"][case]["partition_assignment_sha256"],
                "historical_timing_layout": "20_partition/timing_input", "physical_threads": 4,
                "io_seed": 1, "partition_seed": 1001, **placement_contract}
    if endpoint in {"timing", "cts_thermal", "grt"}:
        timing = _timing(archive, run)
        contract.update(opt_passes=1, timing_view="pre-CTS-continuous-passive-hbt-placement-RC")
        if endpoint == "timing":
            metrics = timing
    extra_runtimes = {}
    if endpoint == "grt":
        metrics, contract["grt_configuration"], extra_runtimes["global_route"] = _grt(archive, run, case_record)
    elif endpoint == "cts_thermal":
        metrics, receipt = _thermal(archive, run, case, thermal_root)
        contract.update(thermal_settings=receipt["settings"], thermal_implementation=receipt["implementation"],
                        thermal_tools=receipt["tools"], thermal_model="cts-vectorless-placement-RC")
        extra_runtimes["cts_thermal_snapshot"] = receipt.get("total_wall_seconds")
    pinned_case = baseline.identity()["cases"][case]
    frozen_inputs = {"reference_index": baseline.identity()["reference_json_sha256"],
                     **{name: row["sha256"] for name, row in pinned_case["physical_prefix"].items()}}
    if endpoint == "placement":
        frozen_inputs["20_partition/partition_assignment.csv"] = pinned_case["partition_assignment_sha256"]
    elif endpoint == "grt":
        frozen_inputs["70_cts/checkpoint.json"] = baseline.sha256(run / "70_cts/checkpoint.json")
        for die in ("bottom", "top"):
            frozen_inputs["70_cts/" + die + "/design.sdc"] = baseline.sha256(run / "70_cts" / die / "design.sdc")
        # The pinned index binds the package containing the projected die SDC
        # implementation. It is not an unvalidated caller-supplied cache key.
        if "sdc_patch" in index:
            frozen_inputs["sdc_projection_package"] = index["sdc_patch"]["content_manifest_sha256"]
    return {"schema_version": 1, "producer_profile": PRODUCER_PROFILE,
            "reference_id": index["reference_id"], "case": case, "endpoint": endpoint,
            "qualified": True, "qualification_scope": "historical-producer-endpoint-only",
            "baseline": {"raw_metrics": metrics, "source_run": str(run),
                         "source_run_status": ledger.get("status"), "stage_runtimes": stages,
                         "native_runtime_historical": _historical_runtime(stages, endpoint, extra_runtimes),
                         "metrics_provenance": "previously-measured-native-results-not-reexecution"},
            "source_identity": {"reference_json_sha256": baseline.identity()["reference_json_sha256"],
                                "files_json_sha256": FILES_SHA256, "image": index["image"],
                                "image_evidence": image, "source_provenance": provenance,
                                "requested_config_sha256": baseline.sha256(requested_config),
                                "normalized_executed_config_sha256": baseline.sha256(executed_config),
                                "source_tree": str(run.parent / "source"),
                                "source_tree_byte_validation": "inventory-pinned; whole source tree not reread"},
            "source_contract": contract, "frozen_inputs": frozen_inputs,
            "evidence": list(archive.evidence.values()),
            "executed_native_stages": [], "new_model_calls": 0, "candidate_finals": 0,
            "current_runtime_equivalence_qualified": False, "new_hook_equivalence_qualified": False,
            "not_claimed": ["stage-15 execution", "stage-75 execution", "new physical evaluation",
                            "routed/signoff timing", "routed thermal", "candidate runtime compatibility"]}
