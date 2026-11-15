#!/usr/bin/env python3
"""Trusted timing/thermal gate for partition-placement co-design tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Tuple


STAGE_DIRS = {
    "input": "00_input",
    "floorplan": "10_floorplan",
    "timing_extract": "15_timing_extract",
    "partition": "20_partition",
    "placement": "30_placement",
    "pdn": "40_pdn",
    "hbt": "50_hbt",
    "opt": "60_opt",
    "cts": "70_cts",
    "global_routing": "75_global_route",
    "detailed_routing": "80_route",
    "report": "90_report",
    "thermal": "95_thermal",
}
TIMING_STAGES = tuple(STAGE_DIRS)[:8]
THERMAL_STAGES = tuple(STAGE_DIRS)
LEGACY_THERMAL_STAGE_DIRS = {
    **{
        stage: directory
        for stage, directory in STAGE_DIRS.items()
        if stage not in {"global_routing", "detailed_routing"}
    },
    "routing": "80_route",
}
LEGACY_THERMAL_STAGES = tuple(LEGACY_THERMAL_STAGE_DIRS)
FLOOR = 1.0e-12
# Explicit producer identities: preserve historical evidence without accepting
# arbitrary CTS implementations or relabeling the current native result.
CTS_ALGORITHMS = {"openroad-coordinated-insertion-delay", "openroad-balanced"}
FIXED_SEEDS = {
    "io_placement": 1,
    "partition": 1001,
    "placement": 1000,
    "routing": -1,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be finite".format(label))
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be finite".format(label)) from error
    if not math.isfinite(number):
        raise ValueError("{} must be finite".format(label))
    return number


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def _submitted_file(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("unsafe submitted path: {}".format(relative))
    path = (root / value.as_posix()).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError("missing or unsafe submitted file: {}".format(relative))
    return path


def _validate_stage(run_root: Path, stage: str) -> None:
    directory = STAGE_DIRS.get(stage, LEGACY_THERMAL_STAGE_DIRS.get(stage))
    if directory is None:
        raise ValueError("unknown trusted stage {}".format(stage))
    stage_root = (run_root / directory).resolve()
    manifest = _read(stage_root / "stage_manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != stage
        or manifest.get("status") != "success"
        or not isinstance(manifest.get("signature"), dict)
    ):
        raise ValueError("invalid trusted {} stage manifest".format(stage))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("trusted {} stage has no artifacts".format(stage))
    for name, record in artifacts.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid {} artifact record {}".format(stage, name))
        artifact = (stage_root / record["path"]).resolve()
        if stage_root not in artifact.parents or not artifact.is_file() or artifact.is_symlink():
            raise ValueError("unsafe or missing {} artifact {}".format(stage, name))
        if artifact.stat().st_size != record.get("bytes") or _sha256(artifact) != record.get("sha256"):
            raise ValueError("{} artifact hash mismatch: {}".format(stage, name))


def _validate_chain(run_root: Path, expected: Tuple[str, ...], *, require_timing_stage=False) -> None:
    status = _read(run_root / "run.json")
    stages = status.get("stages")
    if status.get("status") != "success" or not isinstance(stages, Mapping):
        raise ValueError("trusted run did not complete successfully")
    stages = dict(stages)
    if "detailed_routing" in stages:
        stages.pop("routing", None)
    resolved_expected = list(expected)
    # Step 15 did not exist in historical reference runs. New workspace
    # endpoints recompute timing/partition rather than scoring that imported
    # legacy prefix, and must retain a native stage-15 record. A legitimate
    # non-TP-GNN skip is still checked below, never inferred from absence.
    if "timing_extract" not in stages and "timing_extract" in resolved_expected:
        if require_timing_stage:
            raise ValueError("reference-aligned global evaluation requires a native timing_extract stage record")
        resolved_expected.remove("timing_extract")
    if tuple(expected) == THERMAL_STAGES and "routing" in stages:
        resolved_expected.remove("global_routing")
        resolved_expected.remove("detailed_routing")
        resolved_expected.insert(resolved_expected.index("report"), "routing")
    if set(stages) != set(resolved_expected):
        raise ValueError("trusted run has the wrong stage set")
    for stage in resolved_expected:
        row = stages[stage]
        if stage == "timing_extract" and isinstance(row, Mapping) and row.get("status") == "skipped":
            partition = _read(run_root / "20_partition/partition_manifest.json")
            stage_method = _read(run_root / "20_partition/stage_manifest.json").get("method")
            method = partition.get("method")
            if (row.get("reason") == "selected partitioner does not consume TP-GNN timing features"
                    and isinstance(method, str) and method and method not in {"tp-gnn", "tp-gnn-flat"}
                    and method == stage_method):
                continue
        if not isinstance(row, Mapping) or row.get("status") != "success":
            raise ValueError("trusted run did not complete {}".format(stage))
        _validate_stage(run_root, stage)


def _validate_seed_contract(run_root: Path, *, include_routing: bool) -> None:
    """Reject seed tuning and verify the seeds actually reported by each tool."""

    io_metrics = _read(run_root / "10_floorplan/io_metrics.json")
    partition = _read(run_root / "20_partition/partition_manifest.json")
    placement = _read(
        run_root / "30_placement/stages/02_global_place/stage_manifest.json"
    )
    observed = {
        "io_placement": int(io_metrics.get("seed", -999)),
        "partition": int(partition.get("seed", -999)),
        "placement": int(placement.get("initialization_seed", -999)),
    }
    if include_routing:
        routing = _read(run_root / "80_route/metrics.json")
        configuration = routing.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("routing metrics omit the seed configuration")
        observed["routing"] = int(configuration.get("droute_seed", -999))
    expected = {
        key: value
        for key, value in FIXED_SEEDS.items()
        if include_routing or key != "routing"
    }
    if observed != expected:
        raise ValueError(
            "benchmark seed contract mismatch: expected={} observed={}".format(
                expected, observed
            )
        )


def _validate_frontend(run_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    partition = _read(run_root / "20_partition/partition_metrics.json")
    count = int(partition.get("instance_count", 0))
    if count <= 0 or int(partition.get("top_count", -1)) + int(partition.get("bottom_count", -1)) != count:
        raise ValueError("partition assignment is incomplete")
    balance = _finite(partition.get("area_balance_error"), "area balance")
    if balance > 0.100000000001:
        raise ValueError("partition violates the 10% area-balance gate")
    placement = _read(run_root / "30_placement/metrics.json")
    dies = placement.get("dies")
    if not isinstance(dies, Mapping) or set(dies) != {"bottom", "top"}:
        raise ValueError("placement does not contain both dies")
    for die in ("bottom", "top"):
        row = dies[die]
        if not isinstance(row, Mapping) or int(row.get("missing_connection_object_count", -1)) != 0:
            raise ValueError("placement is incomplete on {}".format(die))
    return partition, placement


PROFILE_CONTRACTS = {
    "tp_gnn_placement_cts_thermal_20261008": ("cts-thermal", 1),
    "tp_gnn_placement_one_pass_20261004": ("timing", 1),
    "tp_gnn_placement_two_pass": ("timing", 2),
    "tp_gnn_placement_one_pass": ("timing", 1),
    "tp_gnn_placement_cts_thermal": ("cts-thermal", 2),
    "tp_gnn_placement_cts_thermal_one_pass": ("cts-thermal", 1),
    "tp_gnn_placement_thermal": ("thermal", 1),
}
LEGACY_PROFILES = {"timing": "tp_gnn_placement_two_pass",
                   "cts-thermal": "tp_gnn_placement_cts_thermal",
                   "thermal": "tp_gnn_placement_thermal"}


def _profile_passes(mode, profile=None):
    profile = profile or LEGACY_PROFILES[mode]
    selected_mode, passes = PROFILE_CONTRACTS[profile]
    if selected_mode != mode:
        raise ValueError("Evaluator profile does not match verifier mode")
    return passes


def _validate_opt_passes(opt, expected_passes):
    if expected_passes not in (1, 2):
        raise ValueError("Unsupported OPT pass contract")
    if any(type(opt.get(key)) is not int or opt[key] != expected_passes
           for key in ("passes_run", "iterations_run")):
        raise ValueError("timing evaluation requires exactly {} OPT passes".format(expected_passes))
    die_passes = opt.get("die_passes")
    if (not isinstance(die_passes, Mapping) or set(die_passes) != {"bottom", "top"}
            or any(type(value) is not int or value != expected_passes for value in die_passes.values())):
        raise ValueError("both dies must complete exactly {} OPT passes".format(expected_passes))
    if expected_passes == 2:
        records = opt.get("pass_records")
        if not isinstance(records, list) or len(records) != 2:
            raise ValueError("OPT two-pass audit trail is incomplete")
    else:
        # Native one-pass OPT publishes a singular record, not the evaluator
        # patch's list. Require both die repairs and the final measured STA.
        record = opt.get("pass_record")
        if ("pass_records" in opt or not isinstance(record, Mapping)
                or type(record.get("pass")) is not int or record["pass"] != 1
                or record.get("die_order") != ["bottom", "top"]
                or not isinstance(record.get("die_metrics"), Mapping)
                or set(record["die_metrics"]) != {"bottom", "top"}
                or any(not isinstance(row, Mapping) or not row for row in record["die_metrics"].values())
                or not isinstance(record.get("constraints"), Mapping)
                or set(record["constraints"]) != {"bottom", "top"}
                or not isinstance(record.get("global_after"), Mapping)):
            raise ValueError("OPT native one-pass audit trail is incomplete")
        for key in ("wns_setup", "tns_setup"):
            if _finite(record["global_after"].get(key), key) != _finite(opt.get("final_global", {}).get(key), key):
                raise ValueError("OPT one-pass record differs from final global STA")


def _timing_row(run_root: Path, *, post_cts=False, expected_passes=2,
                latest=False, reference_suffix=False) -> Dict[str, Any]:
    stages = TIMING_STAGES + (("cts",) if post_cts else ())
    if latest:
        from scripts.workspace_latest_reference import validate_latest_chain
        if expected_passes != 1 or post_cts:
            raise ValueError("Latest timing endpoint is exactly one pre-CTS OPT pass")
        validate_latest_chain(run_root, reference_suffix=reference_suffix)
    elif expected_passes == 1:
        _validate_chain(run_root, stages, require_timing_stage=True)
    else:
        _validate_chain(run_root, stages)
    _validate_seed_contract(run_root, include_routing=False)
    partition, placement = _validate_frontend(run_root)
    opt = _read(run_root / "60_opt/metrics.json")
    if opt.get("status") != "success" or opt.get("algorithm") not in {
        "openroad-coordinated-passive-cutpoint",
        "openroad-coordinated-passive-boundary",
    }:
        raise ValueError("OPT identity/status gate failed")
    _validate_opt_passes(opt, expected_passes)
    validity = opt.get("timing_validity")
    required = {
        "authoritative_timing_view": "continuous-passive-hbt",
        "graph_completeness": "passed",
        "die_internal_parasitics_in_global_sta": True,
        "hbt_is_passive_rc_in_authoritative_global_sta": True,
        "remote_die_load_visible_to_source_driver": True,
        "unclocked_or_unconstrained_endpoints": 0,
        "missing_liberty_cells": 0,
        "combinational_loops": 0,
    }
    if not isinstance(validity, Mapping) or any(validity.get(key) != value for key, value in required.items()):
        raise ValueError("continuous passive-HBT timing validity gate failed")
    final = opt.get("final_global")
    coverage = opt.get("final_boundary_coverage")
    if not isinstance(final, Mapping) or not isinstance(coverage, Mapping):
        raise ValueError("final global STA evidence is missing")
    if int(coverage.get("sta_unconstrained_endpoints", -1)) != 0:
        raise ValueError("final global STA has unconstrained endpoints")
    if _finite(final.get("sta_completeness_checks_passed"), "STA completeness") != 1.0:
        raise ValueError("final global STA completeness failed")
    if _finite(final.get("passive_hbt_rc"), "passive HBT RC marker") != 1.0:
        raise ValueError("authoritative STA omitted passive HBT RC")
    wns = _finite(final.get("wns_setup"), "WNS")
    tns = _finite(final.get("tns_setup"), "TNS")
    return {
        "opt_passes": expected_passes,
        "wns_setup_ns": wns,
        "tns_setup_ns": tns,
        "wns_deficit": max(0.0, -wns),
        "tns_deficit": max(0.0, -tns),
        "boundary_budget_model": opt.get("boundary_budget_model", "passive-cutpoint"),
        "cut_net_count": int(partition.get("cut_net_count", 0)),
        "placement_total_hpwl_um": _finite(placement.get("total_object_origin_hpwl_um"), "HPWL"),
    }


def _relative(value: float, reference: float) -> float:
    if reference > FLOOR:
        return value / reference
    return 1.0 if value <= FLOOR else 1.0 + value / FLOOR


def _score_timing(candidate: Path, reference: Path, *, profile=None) -> Dict[str, Any]:
    expected_passes = _profile_passes("timing", profile)
    row = _timing_row(candidate, expected_passes=expected_passes)
    ref = _timing_row(reference, expected_passes=expected_passes)
    return _timing_score_rows(row, ref)


def _timing_score_rows(row, ref):
    if row["boundary_budget_model"] != ref["boundary_budget_model"]:
        raise ValueError("candidate/reference boundary timing models differ")
    wns_cost = _relative(row["wns_deficit"], ref["wns_deficit"])
    tns_cost = _relative(row["tns_deficit"], ref["tns_deficit"])
    cost = 0.6 * wns_cost + 0.4 * tns_cost
    return {
        "objective_kind": "sta_wns_tns",
        "combined_score": 1.0 / max(cost, FLOOR),
        "normalized_sta_cost": cost,
        "normalized_wns_cost": wns_cost,
        "normalized_tns_cost": tns_cost,
        **row,
        "reference_wns_setup_ns": ref["wns_setup_ns"],
        "reference_tns_setup_ns": ref["tns_setup_ns"],
    }


def _thermal_row(run_root: Path, *, post_cts=False, expected_passes=2,
                 require_timing_stage=False) -> Dict[str, Any]:
    if post_cts:
        _timing_row(run_root, post_cts=True, expected_passes=expected_passes)
        cts = _read(run_root / "70_cts/metrics.json")
        checkpoint = _read(run_root / "70_cts/checkpoint.json")
        if (cts.get("status") != "success" or cts.get("algorithm") not in CTS_ALGORITHMS
                or checkpoint.get("stage") != "cts" or checkpoint.get("ready_for_route") is not True):
            raise ValueError("post-CTS thermal requires completed CTS")
        published_cts = checkpoint.get("cts")
        if (cts["algorithm"] == "openroad-balanced" or published_cts is not None):
            if (not isinstance(published_cts, Mapping)
                    or published_cts.get("algorithm") != cts["algorithm"]
                    or published_cts.get("metrics") != "metrics.json"):
                raise ValueError("post-CTS thermal CTS checkpoint identity mismatch")
    else:
        if require_timing_stage:
            _validate_chain(run_root, THERMAL_STAGES, require_timing_stage=True)
            _validate_opt_passes(_read(run_root / "60_opt/metrics.json"), expected_passes)
        else:
            _validate_chain(run_root, THERMAL_STAGES)
        _validate_seed_contract(run_root, include_routing=True)
    partition, placement = _validate_frontend(run_root)
    metrics = _read(run_root / ("95_thermal_snapshot" if post_cts else "95_thermal") / "metrics.json")
    if post_cts:
        snapshot = _read(run_root / "95_thermal_snapshot/snapshot_manifest.json")
        power = metrics.get("power", {})
        extraction = metrics.get("power_extraction", {})
        if (snapshot.get("status") != "success" or snapshot.get("source_stage") != "cts"
                or metrics.get("source_stage") != "cts"
                or power.get("activity_model") != "vectorless-primary-input-0.1-placement-estimated-rc"
                or power.get("provenance", {}).get("model") != "die-local-openroad-placement-estimated"
                or set(extraction) != {"bottom", "top"}):
            raise ValueError("post-CTS thermal power provenance mismatch")
        expected_checkpoint = (run_root / "70_cts/checkpoint.json").resolve()
        for label, value in (
            ("snapshot", snapshot.get("source_checkpoint")),
            ("power", power.get("provenance", {}).get("checkpoint")),
        ):
            if (not isinstance(value, str) or not value
                    or Path(value).resolve() != expected_checkpoint):
                raise ValueError("post-CTS thermal {} uses a different checkpoint".format(label))
        if (snapshot.get("schema_version") != 1
                or snapshot.get("metrics") != "metrics.json"
                or snapshot.get("grid_size") != 10
                or not math.isclose(_finite(snapshot.get("ambient_k"), "snapshot ambient"),
                                    318.15, rel_tol=0.0, abs_tol=1e-9)):
            raise ValueError("post-CTS thermal snapshot metadata mismatch")
        for die in ("bottom", "top"):
            if (int(extraction[die].get("instance_count", 0)) <= 0
                    or int(extraction[die].get("warning_count", -1)) != 0):
                raise ValueError("post-CTS thermal power extraction failed")
    if (
        metrics.get("stage") != ("thermal_snapshot" if post_cts else "thermal")
        or metrics.get("status") != "success"
        or metrics.get("algorithm") != "hotspot-steady-grid"
        or int(metrics.get("warning_count", -1)) != 0
        or int(metrics.get("grid_size", -1)) != 10
        or not math.isclose(_finite(metrics.get("ambient_k"), "ambient"), 318.15, abs_tol=1e-9)
    ):
        raise ValueError("fixed HotSpot identity/configuration gate failed")
    temperature = metrics.get("temperature")
    if not isinstance(temperature, Mapping):
        raise ValueError("HotSpot temperature metrics are missing")
    weighted = 0.0
    count = 0
    per_die = {}
    for die in ("bottom", "top"):
        die_row = temperature.get(die)
        if not isinstance(die_row, Mapping):
            raise ValueError("HotSpot omitted the {} silicon layer".format(die))
        average = _finite(die_row.get("average_k"), "{} average".format(die))
        tiles = int(die_row.get("tile_count", 0))
        if average < 318.15 or tiles <= 0 or (post_cts and tiles != 100):
            raise ValueError("invalid {} thermal grid".format(die))
        weighted += average * tiles
        count += tiles
        per_die[die] = average
    average = weighted / count
    if not math.isclose(average, _finite(temperature.get("global_average_k"), "global average"), abs_tol=1e-8):
        raise ValueError("published global average does not equal the physical tile mean")
    power = metrics.get("power")
    if not isinstance(power, Mapping):
        raise ValueError("HotSpot power metrics are missing")
    total_power = _finite(power.get("total_w"), "total power")
    error = _finite(power.get("conservation_error_w"), "power conservation error")
    if total_power <= 0 or error > max(FLOOR, total_power * 1e-9):
        raise ValueError("HotSpot power mapping is not positive and conserved")
    return {
        "global_average_temperature_k": average,
        "global_average_temperature_c": average - 273.15,
        "bottom_average_temperature_k": per_die["bottom"],
        "top_average_temperature_k": per_die["top"],
        "global_maximum_temperature_c": _finite(temperature.get("global_maximum_c"), "global maximum"),
        "total_power_w": total_power,
        "stack": metrics.get("stack"),
        "cut_net_count": int(partition.get("cut_net_count", 0)),
        "placement_total_hpwl_um": _finite(placement.get("total_object_origin_hpwl_um"), "HPWL"),
        **({"cts_algorithm": cts["algorithm"]} if post_cts else {}),
    }


def _validate_thermal_reference_frontend(reference: Path) -> None:
    # This standalone verifier cannot rely on the editable subject's defaults.
    # The reference is fixed; scope-free candidate algorithm edits remain free.
    partition_method = None
    for stage, methods in (("partition", {"tp-gnn", "tp-gnn-flat"}),
                           ("placement", {"concurrent-dreamplace"})):
        manifest = _read(reference / STAGE_DIRS[stage] / "stage_manifest.json")
        if (manifest.get("stage") != stage or manifest.get("status") != "success"
                or manifest.get("method") not in methods):
            raise ValueError("thermal reference requires recorded {} method {}".format(stage, sorted(methods)))
        if stage == "partition":
            partition_method = manifest["method"]
    if _read(reference / "20_partition/partition_manifest.json").get("method") != partition_method:
        raise ValueError("thermal reference partition artifact is not TP-GNN")
    report = _read(reference / "20_partition/tp_gnn_report.json")
    config = report.get("config")
    if (report.get("method") != partition_method or not isinstance(config, Mapping)
            or config.get("decoder") != "global-2way"):
        raise ValueError("thermal reference requires recorded TP-GNN global-2way decoding")
    hierarchy = report.get("hierarchy")
    if partition_method == "tp-gnn-flat":
        if (not isinstance(hierarchy, Mapping) or hierarchy.get("mode") != "flat"
                or hierarchy.get("consumed_as_feature") is not False):
            raise ValueError("thermal reference requires native TP-GNN-flat hierarchy provenance")
    elif isinstance(hierarchy, Mapping) and hierarchy.get("mode") == "flat":
        raise ValueError("thermal reference cannot relabel TP-GNN-flat as TP-GNN")


def _score_thermal(candidate: Path, reference: Path, *, profile=None) -> Dict[str, Any]:
    _validate_thermal_reference_frontend(reference)
    if profile is None:
        # Preserve the historical functional API. Explicit workspace profile
        # selection activates the native one-pass audit on both measurements.
        row = _thermal_row(candidate)
        ref = _thermal_row(reference)
    else:
        passes = _profile_passes("thermal", profile)
        row = _thermal_row(candidate, expected_passes=passes, require_timing_stage=True)
        ref = _thermal_row(reference, expected_passes=passes, require_timing_stage=True)
    return _thermal_score_rows(row, ref, passes=passes if profile is not None else None)


def _thermal_score_rows(row, ref, *, passes=None):
    if row["stack"] != ref["stack"]:
        raise ValueError("candidate/reference HotSpot stack models differ")
    average = row["global_average_temperature_k"]
    reference_average = ref["global_average_temperature_k"]
    return {
        "objective_kind": "hotspot_average_temperature",
        "combined_score": reference_average / max(average, FLOOR),
        "normalized_average_temperature_cost": average / reference_average,
        **row,
        "reference_global_average_temperature_k": reference_average,
        **({"opt_passes": passes} if passes is not None else {}),
    }


def _score_cts_thermal(candidate: Path, reference: Path, *, profile=None) -> Dict[str, Any]:
    expected_passes = _profile_passes("cts-thermal", profile)
    _validate_thermal_reference_frontend(reference)
    row = _thermal_row(candidate, post_cts=True, expected_passes=expected_passes)
    ref = _thermal_row(reference, post_cts=True, expected_passes=expected_passes)
    row["boundary_budget_model"] = _read(candidate / "60_opt/metrics.json").get("boundary_budget_model")
    ref["boundary_budget_model"] = _read(reference / "60_opt/metrics.json").get("boundary_budget_model")
    return _cts_thermal_score_rows(row, ref, expected_passes)


def _cts_thermal_score_rows(row, ref, expected_passes):
    power_model = row.get("power_model", "die-local-openroad-placement-estimated")
    if (row["stack"] != ref["stack"] or row["cts_algorithm"] != ref["cts_algorithm"] or
            row["boundary_budget_model"] != ref["boundary_budget_model"] or
            power_model != ref.get("power_model", "die-local-openroad-placement-estimated")):
        raise ValueError("candidate/reference CTS thermal models differ")
    return {**row, "objective_kind": "hotspot_average_temperature",
            "combined_score": ref["global_average_temperature_k"] / row["global_average_temperature_k"],
            "normalized_average_temperature_cost": row["global_average_temperature_k"] / ref["global_average_temperature_k"],
            "reference_global_average_temperature_k": ref["global_average_temperature_k"],
            "thermal_evaluation_boundary": ("one-opt-cts-cell-power-v1" if expected_passes == 1
                                             else "two-opt-cts-cell-power-v1"),
            "opt_passes": expected_passes,
            "cts_completed": 1, "thermal_grid_size": 10, "thermal_ambient_k": 318.15,
            "power_model": power_model, "runs_successfully": 1}


def _timing_constrained_cts_thermal_score_rows(row, ref):
    """Mean-temperature reward minus 0.1 times post-CTS cost excess above 5%."""
    checked = []
    for label, raw in (("candidate", row), ("reference", ref)):
        item = dict(raw)
        if item.get("timing_evaluation_boundary") != "post_cts_one_opt":
            raise ValueError("Thermal timing penalty requires post-CTS STA evidence after one OPT")
        for key in ("global_average_temperature_k", "wns_setup_ns", "tns_setup_ns",
                    "wns_deficit", "tns_deficit"):
            item[key] = _finite(item.get(key), "{}.{}".format(label, key))
        if item["global_average_temperature_k"] <= 0:
            raise ValueError("Thermal temperature must be positive kelvin")
        for metric in ("wns", "tns"):
            if item[metric + "_deficit"] != max(0.0, -item[metric + "_setup_ns"]):
                raise ValueError("Thermal timing deficit disagrees with raw slack")
        checked.append(item)
    candidate, reference = checked
    timing = _timing_score_rows(candidate, reference)
    cost = _finite(timing["normalized_sta_cost"], "normalized_sta_cost")
    excess = max(0.0, cost - 1.05)
    penalty_weight = 0.1
    penalty = penalty_weight * excess
    thermal = _cts_thermal_score_rows(candidate, reference, 1)
    return {**thermal,
            **{key: timing[key] for key in ("normalized_sta_cost", "normalized_wns_cost",
                "normalized_tns_cost", "reference_wns_setup_ns", "reference_tns_setup_ns")},
            "unpenalized_thermal_score": thermal["combined_score"],
            "timing_degradation_fraction": max(0.0, cost - 1.0),
            "timing_tolerance_fraction": 0.05,
            "timing_penalty_excess": excess,
            "timing_penalty_weight": penalty_weight,
            "timing_penalty": penalty,
            "timing_penalty_applied": excess > 0.0,
            "combined_score": thermal["combined_score"] - penalty}


def reference_self_metrics(endpoint, row, profile):
    """Normalize independently qualified raw data with its declared objective."""
    if endpoint == "timing":
        _profile_passes("timing", profile)
        return _timing_score_rows(row, row)
    if endpoint == "cts_thermal":
        if profile == "tp_gnn_placement_cts_thermal_20261008":
            return _timing_constrained_cts_thermal_score_rows(row, row)
        return _cts_thermal_score_rows(row, row, _profile_passes("cts-thermal", profile))
    if endpoint == "routed_thermal":
        return _thermal_score_rows(row, row, passes=_profile_passes("thermal", profile))
    raise ValueError("Unsupported reference endpoint")


def score_with_reference_record(candidate, reference, record_path, evaluation_contract_path,
                                *, mode, profile, submission):
    """Requalify sealed raw reference data; current candidate gates stay native.

    The standalone verifier derives the expected consumer/physical identity
    from its trusted task and evaluator-owned contract. Cached scores and a
    receipt's qualified flag are never sufficient to pass this boundary.
    """
    if profile in {"tp_gnn_placement_one_pass_20261004", "tp_gnn_placement_cts_thermal_20261008"}:
        from scripts.workspace_latest_reference import score_with_latest_record
        return score_with_latest_record(candidate, reference, record_path, evaluation_contract_path,
                                        mode=mode, profile=profile, submission=submission)
    from global_bench.model import load_global_task
    from local_bench.reference_catalog import evaluator_digest, qualify_producer, task_digest
    from local_bench.reference_evaluated import expected_frozen_inputs, expected_source_contract
    from local_bench.reference_store import load_reference
    from scripts.workspace_task_selection import load_registered_task

    endpoints = {"timing": ("timing", "timing"),
                 "cts-thermal": ("cts_thermal", "thermal"),
                 "thermal": ("routed_thermal", "routed_thermal")}
    endpoint, objective = endpoints[mode]
    framework = Path(__file__).resolve().parents[2]
    task = load_registered_task(submission.get("task_id"), objective=objective, framework=framework)
    task_id = task.task_id
    evaluated = _read(evaluation_contract_path)
    case = evaluated.get("case")
    if (submission.get("task_id") != task_id or submission.get("task_digest") != task.digest()
            or evaluated.get("task_digest") != task.digest()
            or evaluated.get("evaluation_endpoint", {}).get("task_id") != task_id
            or evaluated.get("evaluation_endpoint", {}).get("profile") != profile
            or task.objective.evaluator_profile != profile
            or submission.get("base_revision") != evaluated.get("source_revision")
            or _read(candidate / "run.json").get("design") != case):
        raise ValueError("Sealed reference consumer does not match the trusted candidate task/case/source")
    contract = {"case": case, "endpoint": endpoint,
                "task_digest": task_digest(task.root), "evaluator_digest": evaluator_digest(),
                "frozen_inputs": expected_frozen_inputs(case, evaluated.get("input_manifest_sha256")),
                "physical": expected_source_contract(case, endpoint, profile=profile,
                    source_revision=evaluated.get("source_revision"), image=evaluated.get("image"))}
    record = load_reference(record_path, contract=contract, validate=qualify_producer)
    if record["producer"].get("kind") != "native-evaluation-v1":
        raise ValueError("Archived workspace source composition has not been qualified")
    qualified = record["qualification"]
    if Path(qualified["baseline"]["source_run"]).resolve() != reference:
        raise ValueError("Reference run differs from the selected sealed record")
    ref = qualified["baseline"]["raw_metrics"]
    passes = _profile_passes(mode, profile)
    if mode == "timing":
        return _timing_score_rows(_timing_row(candidate, expected_passes=passes), ref)
    if mode == "cts-thermal":
        row = _thermal_row(candidate, post_cts=True, expected_passes=passes)
        threads = _read(candidate / "95_thermal_snapshot/metrics.json").get("power", {}).get("provenance", {}).get("threads")
        if type(threads) is not int or threads != contract["physical"]["thermal_snapshot_threads"]:
            raise ValueError("Candidate CTS power extraction threads differ from the sealed physical contract")
        row["boundary_budget_model"] = _read(candidate / "60_opt/metrics.json").get("boundary_budget_model", "passive-cutpoint")
        return _cts_thermal_score_rows(row, ref, passes)
    return _thermal_score_rows(_thermal_row(candidate, expected_passes=passes,
                                            require_timing_stage=True), ref, passes=passes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("timing", "thermal", "cts-thermal"))
    parser.add_argument("--profile", choices=tuple(PROFILE_CONTRACTS))
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluated-run", required=True, type=Path)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--reference-record", type=Path)
    parser.add_argument("--reference-record-sha256")
    parser.add_argument("--evaluation-contract", type=Path)
    args = parser.parse_args()
    root = args.submission.expanduser().resolve()
    output = args.output.expanduser().resolve()
    evaluated = args.evaluated_run.expanduser().resolve()
    reference = args.reference_run.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    objective: Dict[str, Any] = {}
    manifest: Dict[str, Any] = {}
    try:
        _profile_passes(args.mode, args.profile)
        manifest = _read(root / "submission.json")
        records = {item["name"]: item for item in manifest.get("artifacts", [])}
        expected = {
            "run_manifest": evaluated / "run.json",
            "final_checkpoint": evaluated / ("60_opt/checkpoint.json" if args.mode == "timing" else "95_thermal/checkpoint.json"),
            "metrics": evaluated / ("60_opt/metrics.json" if args.mode == "timing" else "95_thermal/metrics.json"),
        }
        if args.mode == "cts-thermal":
            expected.update(final_checkpoint=evaluated / "70_cts/checkpoint.json",
                            metrics=evaluated / "95_thermal_snapshot/metrics.json")
        for name, trusted in expected.items():
            record = records.get(name)
            if not isinstance(record, dict):
                raise ValueError("submission is missing {}".format(name))
            submitted = _submitted_file(root, str(record.get("submitted_path")))
            if _sha256(submitted) != record.get("sha256") or _sha256(submitted) != _sha256(trusted):
                raise ValueError("submitted {} differs from trusted evaluation".format(name))
            decoded = _read(submitted)
            if not _finite_tree(decoded):
                raise ValueError("submitted {} contains a non-finite value".format(name))
        if args.reference_record is not None:
            if args.evaluation_contract is None:
                raise ValueError("A sealed reference requires its trusted evaluation contract")
            if _sha256(args.reference_record) != args.reference_record_sha256:
                raise ValueError("Sealed reference differs from the pinned record digest")
            objective = score_with_reference_record(evaluated, reference, args.reference_record,
                args.evaluation_contract, mode=args.mode, profile=args.profile, submission=manifest)
        else:
            if args.profile in {"tp_gnn_placement_one_pass_20261004", "tp_gnn_placement_cts_thermal_20261008"}:
                raise ValueError("Latest timing requires an explicit qualified reference record")
            objective = (
                _score_timing(evaluated, reference, profile=args.profile)
                if args.mode == "timing"
                else (_score_cts_thermal(evaluated, reference, profile=args.profile) if args.mode == "cts-thermal"
                      else _score_thermal(evaluated, reference, profile=args.profile))
            )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
    result = {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "metrics": objective if not errors else {},
        "provenance": {
            "reference_record": str(args.reference_record) if args.reference_record else None,
            "reference_record_sha256": args.reference_record_sha256,
            "evaluator_profile": args.profile or LEGACY_PROFILES[args.mode],
            "task_id": manifest.get("task_id"),
            "trial_id": manifest.get("trial_id"),
            "base_revision": manifest.get("base_revision"),
        },
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    reward = float(objective.get("combined_score", 0.0)) if not errors else 0.0
    (output / "reward.json").write_text(json.dumps({"reward": reward}, indent=2) + "\n", encoding="utf-8")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
