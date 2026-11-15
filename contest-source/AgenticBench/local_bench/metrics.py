from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .tp_gnn_contract import validate_tp_gnn_method


STAGE_DIRS = {
    "partition": "20_partition",
    "placement": "30_placement",
    "opt": "60_opt",
    "routing": "80_route",
    "global_routing": "75_global_route",
}

PLACEMENT_HPWL_WEIGHT = 0.9
PLACEMENT_OVERFLOW_WEIGHT = 0.1
# Shared by the four routing-source tasks; compatibility parameter scoring is
# deliberately separate. Keep both signals, with wire length as the priority.
ROUTING_ORDER_OVERFLOW_WEIGHT = 0.1
ROUTING_ORDER_WIRELENGTH_WEIGHT = 0.9
STA_WNS_WEIGHT = 0.6
STA_TNS_WEIGHT = 0.4
RATIO_DENOMINATOR_FLOOR = 1.0e-12


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metrics file must contain an object: {}".format(path))
    return value


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _ratio_score(candidate: float, reference: Optional[float], scale: float) -> float:
    if candidate < 0:
        return 0.0
    if reference is not None and reference == 0:
        return 1.0 if candidate == 0 else 0.0
    if reference is not None and reference > 0:
        # Do not clip improvement ratios.  A cap makes distinct strong
        # candidates indistinguishable and can freeze evolutionary selection
        # after the first candidate reaches it.  The denominator floor is only
        # a finite-value guard for an exact zero metric, not a score ceiling.
        return max(0.0, reference / max(candidate, RATIO_DENOMINATOR_FLOOR))
    return 1.0 / (1.0 + candidate / scale)


def _reference(reference_run: Optional[Path], stage: str) -> Optional[Dict[str, Any]]:
    if reference_run is None:
        return None
    filename = "partition_metrics.json" if stage == "partition" else "metrics.json"
    path = (
        _routing_stage(reference_run) / filename
        if stage == "routing"
        else reference_run / STAGE_DIRS[stage] / filename
    )
    return _read(path) if path.is_file() else None


_OVERFLOW_VECTOR = re.compile(r"(?<!Global )Overflow\s*\[([^\]]+)\]", re.IGNORECASE)
_OVERFLOW_SCALAR = re.compile(r"(?<!Global )Overflow\s+([0-9.eE+-]+)", re.IGNORECASE)
_MAX_DENSITY_VECTOR = re.compile(r"MaxDensity\s*\[([^\]]+)\]", re.IGNORECASE)
_MAX_DENSITY_SCALAR = re.compile(r"MaxDensity\s+([0-9.eE+-]+)", re.IGNORECASE)
_WEIGHTED_HPWL = re.compile(r"wHPWL\s+([0-9.eE+-]+)", re.IGNORECASE)
_DYNAMIC_LR = re.compile(
    r"DynamicLR\s+step=(\d+)\s+lr=([0-9.eE+-]+)\s+hpwl=([0-9.eE+-]+)\s+overflow=([0-9.eE+-]+)"
)
_DIE_AWARE_DYNAMIC_LR = re.compile(
    r"DynamicLR\s+step=(\d+)\s+"
    r"base_lr=([0-9.eE+-]+)\s+"
    r"bottom_lr=([0-9.eE+-]+)\s+top_lr=([0-9.eE+-]+)\s+"
    r"bottom_multiplier=([0-9.eE+-]+)\s+top_multiplier=([0-9.eE+-]+)\s+"
    r"hpwl=([0-9.eE+-]+)\s+"
    r"overflow_bottom=([0-9.eE+-]+)\s+overflow_top=([0-9.eE+-]+)"
)


def _values_in_line(
    line: str, vector: re.Pattern, scalar: re.Pattern
) -> List[float]:
    vector_match = vector.search(line)
    if vector_match:
        return [
            _finite(item)
            for item in re.findall(r"[0-9.eE+-]+", vector_match.group(1))
        ]
    scalar_match = scalar.search(line)
    return [_finite(scalar_match.group(1))] if scalar_match else []


def _placement_native(run_dir: Path) -> Dict[str, float]:
    log = run_dir / STAGE_DIRS["placement"] / "stages" / "02_global_place" / "run.log"
    if not log.is_file():
        raise ValueError("placement run.log is missing")
    final = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        hpwl_match = _WEIGHTED_HPWL.search(line)
        overflow = _values_in_line(line, _OVERFLOW_VECTOR, _OVERFLOW_SCALAR)
        max_density = _values_in_line(
            line, _MAX_DENSITY_VECTOR, _MAX_DENSITY_SCALAR
        )
        # Select the final *global-placement* metric row.  The later internal
        # DREAMPlace legalization report has HPWL but no density fields and is
        # deliberately outside this optimization objective.
        if hpwl_match and overflow and max_density:
            final = (_finite(hpwl_match.group(1), float("inf")), overflow, max_density)
    if final is None:
        raise ValueError(
            "placement run.log does not contain a complete global-placement "
            "wHPWL/Overflow/MaxDensity measurement"
        )
    hpwl, overflow, max_density = final
    if hpwl < 0.0 or not math.isfinite(hpwl):
        raise ValueError("placement run.log contains an invalid final wHPWL")
    result = {
        "placement_native_hpwl": hpwl,
        "placement_overflow_max": max(overflow),
        "placement_overflow_mean": sum(overflow) / len(overflow),
        "placement_max_density": max(max_density),
    }
    if len(overflow) == 2:
        result.update(
            {
                "placement_overflow_bottom": overflow[0],
                "placement_overflow_top": overflow[1],
            }
        )
    if len(max_density) == 2:
        result.update(
            {
                "placement_max_density_bottom": max_density[0],
                "placement_max_density_top": max_density[1],
            }
        )
    return result


def _routing_stage(run_dir: Path) -> Path:
    """Prefer the first-class GRT stage, with legacy result compatibility."""

    global_route = run_dir / STAGE_DIRS["global_routing"]
    return global_route if global_route.is_dir() else run_dir / STAGE_DIRS["routing"]


def _source_grt_metrics_path(run_dir: Path) -> Path:
    """Select the explicit standalone publication, never post-DRT metrics."""
    stage = _routing_stage(run_dir)
    checkpoint = stage / "global_route_checkpoint.json"
    if checkpoint.is_file():
        value = _read(checkpoint)
        if (value.get("stage") != "global_route" or value.get("status") != "success"
                or value.get("metrics") != "grt_metrics.json"
                or value.get("ready_for_detailed_route") is not True
                or value.get("ready_for_extract") is not False):
            raise ValueError("Invalid standalone GRT checkpoint")
        raw = _read(stage / "grt_metrics.json")
        if raw.get("stage") != "global_route" or raw.get("status") != "success":
            raise ValueError("Invalid standalone GRT metrics")
        return stage / "grt_metrics.json"
    return stage / "metrics.json"


def _source_grt_log(run_dir: Path, die: str) -> Path:
    stage = _routing_stage(run_dir)
    name = "grt.log" if _source_grt_metrics_path(run_dir).name == "grt_metrics.json" else "route.log"
    return stage / die / name


def partition_metrics(run_dir: Path, reference_run: Optional[Path]) -> Dict[str, Any]:
    path = run_dir / STAGE_DIRS["partition"] / "partition_metrics.json"
    raw = _read(path)
    ref = _reference(reference_run, "partition")
    required = {"instance_count", "top_count", "bottom_count", "cut_net_count", "cut_net_ratio", "area_balance_error"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError("partition metrics are missing: {}".format(", ".join(missing)))
    instance_count = int(raw["instance_count"])
    top_count = int(raw["top_count"])
    bottom_count = int(raw["bottom_count"])
    if instance_count <= 0 or top_count + bottom_count != instance_count:
        raise ValueError("partition assignment is incomplete")
    cut_ratio = _finite(raw.get("cut_net_ratio"), 1.0)
    balance_error = _finite(raw.get("area_balance_error"), 1.0)
    cut_score = max(0.0, min(1.0, 1.0 - cut_ratio))
    balance_score = max(0.0, min(1.0, 1.0 - balance_error / 0.05))
    constraints_passed = 1.0 if balance_error <= 0.10 else 0.0
    combined = constraints_passed * (0.75 * cut_score + 0.25 * balance_score)
    result = {
        "combined_score": combined,
        "cut_score": cut_score,
        "balance_score": balance_score,
        "cut_net_count": int(raw.get("cut_net_count", 0)),
        "cut_net_ratio": cut_ratio,
        "cut_pin_count": int(raw.get("cut_pin_count", 0)),
        "area_balance_error": balance_error,
        "partition_complete": 1.0,
        "partition_constraints_passed": constraints_passed,
        "instance_count": instance_count,
        "top_count": top_count,
        "bottom_count": bottom_count,
    }
    if ref:
        result["cut_net_improvement_ratio"] = _ratio_score(
            _finite(raw.get("cut_net_count")), _finite(ref.get("cut_net_count")), 10000.0
        )
    return result


def _method_metric(raw: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    method = raw.get("method_metrics", {})
    if not isinstance(method, Mapping):
        return default
    return _finite(method.get(name), default)


def tp_gnn_learning_objective_metrics(
    run_dir: Path, reference_run: Optional[Path]
) -> Dict[str, Any]:
    """Score an evolved loss by the final legal TP-GNN assignment."""

    stage = run_dir / STAGE_DIRS["partition"]
    raw = _read(stage / "partition_metrics.json")
    ref = _reference(reference_run, "partition")
    required = {
        "instance_count",
        "top_count",
        "bottom_count",
        "macro_count",
        "top_macro_count",
        "cut_net_count",
        "cut_net_ratio",
        "cut_pin_count",
        "timing_total_net_weight",
        "timing_weighted_cut",
        "timing_weighted_cut_ratio",
        "timing_weight_mode",
        "area_balance_error",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            "TP-GNN partition metrics are missing: {}".format(", ".join(missing))
        )

    instance_count = int(raw["instance_count"])
    top_count = int(raw["top_count"])
    bottom_count = int(raw["bottom_count"])
    if instance_count <= 0 or top_count + bottom_count != instance_count:
        raise ValueError("TP-GNN partition assignment is incomplete")

    macro_count = int(raw["macro_count"])
    top_macro_count = int(raw["top_macro_count"])
    macro_count_error = (
        abs(top_macro_count / macro_count - 0.5) if macro_count else 0.0
    )
    top_macro_area = _method_metric(raw, "top_macro_area", float("nan"))
    bottom_macro_area = _method_metric(raw, "bottom_macro_area", float("nan"))
    if macro_count and (
        not math.isfinite(top_macro_area)
        or not math.isfinite(bottom_macro_area)
        or top_macro_area + bottom_macro_area <= 0.0
    ):
        raise ValueError("TP-GNN metrics do not contain valid macro areas")
    total_macro_area = top_macro_area + bottom_macro_area
    macro_area_error = (
        abs(top_macro_area / total_macro_area - 0.5)
        if total_macro_area > 0.0
        else 0.0
    )
    area_balance_error = _finite(raw["area_balance_error"], float("inf"))
    constraints_passed = 1.0 if (
        area_balance_error <= 0.060000000001
        and macro_count_error <= 0.050000000001
        and macro_area_error <= 0.050000000001
    ) else 0.0

    cut_net_count = _finite(raw["cut_net_count"], float("inf"))
    cut_pin_count = _finite(raw["cut_pin_count"], float("inf"))
    timing_total_net_weight = _finite(raw["timing_total_net_weight"], float("inf"))
    timing_weighted_cut = _finite(raw["timing_weighted_cut"], float("inf"))
    timing_weighted_cut_ratio = _finite(
        raw["timing_weighted_cut_ratio"], float("inf")
    )
    timing_weight_mode = str(raw["timing_weight_mode"])
    if timing_weight_mode != "paper_delay_slack_hyperedge_v1":
        raise ValueError(
            "TP-GNN fitness requires paper_delay_slack_hyperedge_v1 weights"
        )
    if (
        timing_total_net_weight <= 0.0
        or timing_weighted_cut < 0.0
        or timing_weighted_cut > timing_total_net_weight + 1.0e-9
        or not 0.0 <= timing_weighted_cut_ratio <= 1.0
        or abs(
            timing_weighted_cut_ratio
            - timing_weighted_cut / timing_total_net_weight
        ) > 1.0e-9
    ):
        raise ValueError("TP-GNN timing-weighted cut metrics are inconsistent")
    if ref is not None:
        reference_required = {
            "timing_total_net_weight",
            "timing_weighted_cut",
            "timing_weighted_cut_ratio",
            "timing_weight_mode",
        }
        reference_missing = sorted(reference_required - set(ref))
        if reference_missing:
            raise ValueError(
                "TP-GNN reference metrics are missing: {}".format(
                    ", ".join(reference_missing)
                )
            )
        reference_total_weight = _finite(
            ref.get("timing_total_net_weight"), float("nan")
        )
        reference_weighted_cut = _finite(
            ref.get("timing_weighted_cut"), float("nan")
        )
        if (
            not math.isfinite(reference_total_weight)
            or reference_total_weight <= 0.0
            or not math.isfinite(reference_weighted_cut)
            or reference_weighted_cut < 0.0
        ):
            raise ValueError("TP-GNN reference timing-weighted cut is invalid")
        if not math.isclose(
            timing_total_net_weight,
            reference_total_weight,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "Candidate and reference do not use the same frozen timing net weights"
            )
        if str(ref.get("timing_weight_mode")) != timing_weight_mode:
            raise ValueError(
                "Candidate and reference use different timing-weight formulas"
            )
        timing_weighted_cut_score = _ratio_score(
            timing_weighted_cut,
            reference_weighted_cut,
            timing_total_net_weight,
        )
    else:
        timing_weighted_cut_score = max(
            0.0, min(1.0, 1.0 - timing_weighted_cut_ratio)
        )
    combined = constraints_passed * timing_weighted_cut_score
    result: Dict[str, Any] = {
        "combined_score": combined,
        "timing_weighted_cut_score": timing_weighted_cut_score,
        "timing_weighted_cut": timing_weighted_cut,
        "timing_weighted_cut_ratio": timing_weighted_cut_ratio,
        "timing_total_net_weight": timing_total_net_weight,
        "timing_weight_mode": timing_weight_mode,
        "cut_net_count": int(cut_net_count),
        "cut_net_ratio": _finite(raw["cut_net_ratio"], 1.0),
        "cut_pin_count": int(cut_pin_count),
        "area_balance_error": area_balance_error,
        "macro_count_error": macro_count_error,
        "macro_area_error": macro_area_error,
        "partition_complete": 1.0,
        "partition_constraints_passed": constraints_passed,
        "instance_count": instance_count,
        "top_count": top_count,
        "bottom_count": bottom_count,
    }

    report = _read(stage / "tp_gnn_report.json")
    method = validate_tp_gnn_method(report)
    if reference_run is not None:
        validate_tp_gnn_method(_read(reference_run / STAGE_DIRS["partition"] / "tp_gnn_report.json"), expected_method=method)
    result["partition_method"] = method
    training = report.get("training", {})
    if not isinstance(training, Mapping) or training.get("status") != "trained":
        raise ValueError("TP-GNN did not complete learned embedding training")
    final_diagnostics = training.get("final_embedding_diagnostics", {})
    if not isinstance(final_diagnostics, Mapping):
        final_diagnostics = {}

    def add_diagnostic(target: str, value: Any, *, integer: bool = False) -> None:
        # Current report measurements only: no legacy coercion or zero fill.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        if not math.isfinite(value) or (integer and int(value) != value):
            return
        result[target] = int(value) if integer else float(value)

    add_diagnostic("tp_gnn_final_loss", training.get("final_loss"))
    add_diagnostic("tp_gnn_epochs_completed", training.get("epochs_completed"), integer=True)
    add_diagnostic("tp_gnn_optimizer_steps_completed", training.get("optimizer_steps_completed"), integer=True)
    add_diagnostic("tp_gnn_embedding_total_variance", final_diagnostics.get("total_dimension_variance"))
    add_diagnostic("tp_gnn_embedding_pair_cosine_median", final_diagnostics.get("sample_pair_cosine_median"))
    pair_diagnostics = final_diagnostics.get("training_pair_cosines", {})
    if (isinstance(pair_diagnostics, Mapping)
            and pair_diagnostics.get("population") == "last_completed_training_batch"
            and pair_diagnostics.get("embedding_state") == "final"):
        for label in ("positive", "negative"):
            count = pair_diagnostics.get(label + "_sample_count")
            if isinstance(count, int) and not isinstance(count, bool) and 0 < count <= 4096:
                add_diagnostic("tp_gnn_" + label + "_pair_cosine_median",
                               pair_diagnostics.get(label + "_cosine_median"))
                add_diagnostic("tp_gnn_" + label + "_pair_sample_count", count, integer=True)
    handoff = report.get("lol_handoff_contract", {})
    integrity = handoff.get("supernode_integrity", {}) if isinstance(handoff, Mapping) else {}
    if isinstance(integrity, Mapping) and integrity.get("preserved") is True:
        add_diagnostic("tp_gnn_repair_moved_supernodes", integrity.get("moved_group_count"), integer=True)
        fraction = integrity.get("moved_area_fraction")
        if isinstance(fraction, (int, float)) and 0 <= fraction <= 1:
            add_diagnostic("tp_gnn_repair_moved_area_fraction", fraction)
    assignment_stages = report.get("assignment_stages", {})
    if isinstance(assignment_stages, Mapping):
        for source, target in (
            ("raw_weighted_kmeans", "tp_gnn_raw_cut_net_count"),
            ("area_balanced_weighted_kmeans", "tp_gnn_balanced_cut_net_count"),
            ("lol_feasible_supernodes", "tp_gnn_feasible_cut_net_count"),
        ):
            row = assignment_stages.get(source, {})
            if isinstance(row, Mapping):
                add_diagnostic(target, row.get("cut_net_count"), integer=True)
                add_diagnostic(target.replace("cut_net_count", "timing_weighted_cut"), row.get("timing_weighted_cut"))
        add_diagnostic("tp_gnn_area_balance_cut_delta", assignment_stages.get("raw_to_area_balanced_cut_delta"), integer=True)
        add_diagnostic("tp_gnn_feasibility_cut_delta", assignment_stages.get("area_balanced_to_feasible_cut_delta"), integer=True)
        for start, end, label in (("raw", "balanced", "area_balance"),
                                  ("balanced", "feasible", "feasibility")):
            before = result.get("tp_gnn_" + start + "_timing_weighted_cut")
            after = result.get("tp_gnn_" + end + "_timing_weighted_cut")
            if before is not None and after is not None:
                add_diagnostic("tp_gnn_" + label + "_weighted_cut_delta", after - before)
                if before > 0:
                    add_diagnostic("tp_gnn_" + label + "_weighted_cut_relative_change", (after - before) / before)
    stage_timing = report.get("stage_timing_seconds", {})
    if isinstance(stage_timing, Mapping):
        add_diagnostic("tp_gnn_training_runtime_seconds", stage_timing.get("train_or_aggregate"))
    return result


def placement_metrics(run_dir: Path, reference_run: Optional[Path]) -> Dict[str, Any]:
    from .observed_diagnostics import placement_log_diagnostics
    native = _placement_native(run_dir)
    ref_native = _placement_native(reference_run) if reference_run else None
    hpwl = native["placement_native_hpwl"]
    ref_hpwl = ref_native["placement_native_hpwl"] if ref_native else None
    hpwl_score = _ratio_score(hpwl, ref_hpwl, 1_000_000.0)
    overflow_score = _ratio_score(
        native["placement_overflow_max"],
        ref_native["placement_overflow_max"] if ref_native else None,
        1.0,
    )
    combined = (
        PLACEMENT_HPWL_WEIGHT * hpwl_score
        + PLACEMENT_OVERFLOW_WEIGHT * overflow_score
    )
    result = {
        "combined_score": combined,
        "hpwl_score": hpwl_score,
        "overflow_score": overflow_score,
        "placement_complete": 1.0,
        "placement_evaluation_boundary": "dreamplace-global-placement",
        "placement_hpwl_metric": "dreamplace-weighted-hpwl",
        **native,
        **placement_log_diagnostics(run_dir),
    }
    native_manifest = run_dir / STAGE_DIRS["placement"] / "stages" / "02_global_place" / "stage_manifest.json"
    if native_manifest.is_file():
        manifest = _read(native_manifest)
        value = manifest.get("target_density")
        if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value):
            result["target_density"] = value
    return result


def placement_component_metrics(run_dir: Path, reference_run: Optional[Path]) -> Dict[str, Any]:
    """Native GP score with overflow reward saturated at the stopping tolerance."""
    result = placement_metrics(run_dir, reference_run)
    ref = _placement_native(reference_run) if reference_run else None
    overflow = result["placement_overflow_max"]
    if ref is None:
        result["hpwl_score"] = 1.0
    if overflow < 0 or not math.isfinite(overflow):
        raise ValueError("Invalid placement overflow")
    result["overflow_score"] = (max(ref["placement_overflow_max"], 0.01)
                                / max(overflow, 0.01)) if ref else 1.0
    result["overflow_reward_tolerance"] = 0.01
    result["combined_score"] = 0.9 * result["hpwl_score"] + 0.1 * result["overflow_score"]
    return result


def placement_dynamic_metrics(run_dir: Path, reference_run: Optional[Path]) -> Dict[str, Any]:
    result = placement_metrics(run_dir, reference_run)
    log = run_dir / STAGE_DIRS["placement"] / "stages" / "02_global_place" / "run.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    die_rows = [
        (
            int(match.group(1)),
            *(_finite(match.group(index)) for index in range(2, 10)),
        )
        for match in _DIE_AWARE_DYNAMIC_LR.finditer(text)
    ]
    if die_rows:
        base_rates = [row[1] for row in die_rows]
        bottom_rates = [row[2] for row in die_rows]
        top_rates = [row[3] for row in die_rows]
        bottom_multipliers = [row[4] for row in die_rows]
        top_multipliers = [row[5] for row in die_rows]
        if any(
            value <= 0.0
            for values in (
                base_rates,
                bottom_rates,
                top_rates,
                bottom_multipliers,
                top_multipliers,
            )
            for value in values
        ):
            raise ValueError("die-aware learning-rate trace contains a non-positive value")
        ratios = [
            max(bottom, top) / min(bottom, top)
            for bottom, top in zip(bottom_multipliers, top_multipliers)
        ]
        result.update({
            "dynamic_lr_updates": len(die_rows),
            "dynamic_lr_first_step": die_rows[0][0],
            "dynamic_lr_last_step": die_rows[-1][0],
            # Backward-compatible keys refer to the shared base schedule.
            "dynamic_lr_initial": base_rates[0],
            "dynamic_lr_final": base_rates[-1],
            "dynamic_lr_min": min(base_rates),
            "dynamic_lr_max": max(base_rates),
            "dynamic_lr_bottom_initial": bottom_rates[0],
            "dynamic_lr_bottom_final": bottom_rates[-1],
            "dynamic_lr_bottom_min": min(bottom_rates),
            "dynamic_lr_bottom_max": max(bottom_rates),
            "dynamic_lr_top_initial": top_rates[0],
            "dynamic_lr_top_final": top_rates[-1],
            "dynamic_lr_top_min": min(top_rates),
            "dynamic_lr_top_max": max(top_rates),
            "dynamic_lr_bottom_multiplier_final": bottom_multipliers[-1],
            "dynamic_lr_top_multiplier_final": top_multipliers[-1],
            "dynamic_lr_max_die_ratio": max(ratios),
        })
        return result
    rows = [
        (int(match.group(1)), _finite(match.group(2)), _finite(match.group(3)), _finite(match.group(4)))
        for match in _DYNAMIC_LR.finditer(text)
    ]
    if not rows:
        raise ValueError("placement run.log does not contain DynamicLR controller measurements")
    rates = [row[1] for row in rows]
    if any(rate <= 0.0 for rate in rates):
        raise ValueError("dynamic learning-rate trace contains a non-positive value")
    result.update({
        "dynamic_lr_updates": len(rows),
        "dynamic_lr_first_step": rows[0][0],
        "dynamic_lr_last_step": rows[-1][0],
        "dynamic_lr_initial": rates[0],
        "dynamic_lr_final": rates[-1],
        "dynamic_lr_min": min(rates),
        "dynamic_lr_max": max(rates),
    })
    return result


def placement_nesterov_metrics(run_dir: Path, reference_run: Optional[Path]) -> Dict[str, Any]:
    """Score the unchanged placement objective with native Nesterov diagnostics.

    The logged native_step is an alpha hint before backtracking, not an Adam
    learning rate or the accepted trial step. Do not publish it under legacy LR
    keys. The shared evidence parser rejects absent, malformed or inactive
    controller traces instead of falling back to a different optimizer.
    """
    from .reference_placement_evidence import parse_nesterov_step_controls

    result = placement_metrics(run_dir, reference_run)
    log = run_dir / STAGE_DIRS["placement"] / "stages" / "02_global_place" / "run.log"
    trace = parse_nesterov_step_controls(log.read_text(encoding="utf-8"))
    rows = trace["controls"]
    result.update({
        "placement_controller_protocol": "native-no-bb-nesterov-step-v1",
        "nesterov_control_updates": trace["call_count"],
        "nesterov_control_first_step": rows[0]["step"],
        "nesterov_control_last_step": rows[-1]["step"],
        "nesterov_control_segments": len(trace["segments"]),
        "nesterov_identity_controls_only": trace["identity_controls_only"],
        "nesterov_native_step_interpretation": trace["native_step_interpretation"],
    })
    for key in ("native_step", "shared", "bottom", "top"):
        values = [row[key] for row in rows]
        label = "nesterov_" + ("native_step_hint" if key == "native_step" else key + "_multiplier")
        result.update({label + "_initial": values[0], label + "_final": values[-1],
                       label + "_min": min(values), label + "_max": max(values)})
    return result


def _routing_parameter_result(run_dir: Path) -> tuple[Dict[str, Any], str]:
    """Validate the actual measurement boundary, never infer DRT from GRT."""
    stage = _routing_stage(run_dir)
    raw = _read(stage / "metrics.json")
    global_only = stage.name == "75_global_route" or raw.get("stage") in {
        "global_route", "route-grt",
    }
    if global_only:
        if (raw.get("stage") not in {"global_route", "route-grt"}
                or raw.get("status") != "success"
                or raw.get("ready_for_detailed_route") is not True
                or raw.get("ready_for_extract") is not False
                or raw.get("unexpected_warnings") != 0):
            raise ValueError("routing parameter evaluation requires a successful, warning-audited GRT boundary")
    dies = raw.get("dies", {}) if isinstance(raw.get("dies"), dict) else {}
    if set(dies) != {"bottom", "top"}:
        raise ValueError("routing metrics must contain exactly bottom and top dies")
    for die, row in dies.items():
        allowed_status = {"success"} if global_only else {None, "success"}
        if not isinstance(row, dict) or row.get("status") not in allowed_status:
            raise ValueError("routing did not complete successfully for {}".format(die))
        fields = ("grt_total_overflow", "grt_congestion_bins")
        if global_only:
            fields += ("grt_wirelength_um",)
        for key in fields:
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError("routing metrics have invalid {} for {}".format(key, die))
    return raw, "global-route-only" if global_only else "detailed-route"


def routing_metrics(run_dir: Path, reference_run: Optional[Path]) -> Dict[str, Any]:
    raw, boundary = _routing_parameter_result(run_dir)
    ref = None
    if reference_run is not None:
        ref, reference_boundary = _routing_parameter_result(reference_run)
        if reference_boundary != boundary:
            raise ValueError("routing parameter reference/candidate evaluation boundary mismatch")
        if boundary == "detailed-route" and (
                ref.get("ready_for_extract") is not True
                or _finite(ref.get("antenna_violations"), -1.0) != 0):
            raise ValueError("routing parameter reference is not a completed detailed route")
    dies = raw["dies"]
    overflow = sum(_finite(row.get("grt_total_overflow")) for row in dies.values())
    congestion_bins = sum(_finite(row.get("grt_congestion_bins")) for row in dies.values())
    ref_overflow = None
    if ref:
        ref_dies = ref.get("dies", {}) if isinstance(ref.get("dies"), dict) else {}
        ref_overflow = sum(_finite(row.get("grt_total_overflow")) for row in ref_dies.values())
    overflow_score = _ratio_score(overflow, ref_overflow, 1000.0)
    global_only = boundary == "global-route-only"
    if global_only:
        # Keep the historical 0.75/0.25 formula. GRT has no DRC evidence: its
        # DRC term is explicitly neutral (N/A), never a fabricated zero count.
        drc, antenna, drc_score, correctness = None, None, 1.0, 1.0
    else:
        drc = _finite(raw.get("drc_violations"))
        antenna = _finite(raw.get("antenna_violations"), -1.0)
        correctness = 1.0 if bool(raw.get("ready_for_extract")) and antenna == 0 else 0.0
        drc_score = 1.0 / (1.0 + drc)
    combined = correctness * (0.75 * overflow_score + 0.25 * drc_score)
    return {
        "combined_score": combined,
        "overflow_score": overflow_score,
        "drc_score": drc_score,
        "grt_total_overflow": overflow,
        "grt_congestion_bins": congestion_bins,
        "grt_wirelength_um": (
            sum(row["grt_wirelength_um"] for row in dies.values()) if global_only else None
        ),
        "drc_violations": drc,
        "antenna_violations": antenna,
        "drc_evaluation": "not-applicable-neutral-score" if global_only else "detailed-route",
        "evaluation_boundary": boundary,
        "runtime_seconds": _finite(raw.get("runtime_seconds")),
        "routing_complete": correctness,
    }


_GRT_WIRELENGTH = re.compile(
    r"^\[INFO GRT-0018\]\s+Total wirelength:\s*([0-9.eE+-]+)\s+um\s*$",
    re.MULTILINE,
)

_NET_ORDER_LINE = re.compile(r"OPEN3D_NET_ORDER\s+([^\r\n]+)")
_NET_ORDER_FIELD = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
_ROUTING_POLICY_PREFIXES = {
    "ripup": "OPEN3D_RIPUP_POLICY",
    "maze_cost": "OPEN3D_MAZE_COST",
    "iteration": "OPEN3D_ITERATION_CONTROL",
}


def _net_order_diagnostics(run_dir: Path, die: str) -> Dict[str, float]:
    """Aggregate auditable ordering changes emitted by the FastRoute hook."""

    log = _source_grt_log(run_dir, die)
    rows = []
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        for match in _NET_ORDER_LINE.finditer(text):
            fields = dict(_NET_ORDER_FIELD.findall(match.group(1)))
            if fields.get("die") != die:
                continue
            try:
                row = {
                    "phase": float(fields["phase"]),
                    "nets": float(fields["nets"]),
                    "overflow_nets": float(fields["overflow_nets"]),
                    "changed_ratio": float(fields["changed_ratio"]),
                    "overflow_changed_ratio": float(fields["overflow_changed_ratio"]),
                    "top10_overlap": float(fields["top10_overlap"]),
                    "score_min": float(fields["score_min"]),
                    "score_max": float(fields["score_max"]),
                    "baseline_hash": fields["baseline_hash"],
                    "learned_hash": fields["learned_hash"],
                }
            except (KeyError, ValueError):
                continue
            if all(
                math.isfinite(float(row[key]))
                for key in (
                    "phase", "nets", "overflow_nets", "changed_ratio",
                    "overflow_changed_ratio", "top10_overlap", "score_min",
                    "score_max",
                )
            ):
                rows.append(row)
    if not rows:
        return {
            "calls": 0.0,
            "changed_calls": 0.0,
        }
    return {
        "calls": float(len(rows)),
        "changed_calls": float(
            sum(row["baseline_hash"] != row["learned_hash"] for row in rows)
        ),
        "changed_ratio_max": max(float(row["changed_ratio"]) for row in rows),
        "overflow_changed_ratio_max": max(
            float(row["overflow_changed_ratio"]) for row in rows
        ),
        "top10_overlap_min": min(float(row["top10_overlap"]) for row in rows),
        "score_span_max": max(
            float(row["score_max"]) - float(row["score_min"]) for row in rows
        ),
        "overflow_net_fraction_max": max(
            float(row["overflow_nets"]) / max(1.0, float(row["nets"]))
            for row in rows
        ),
        "reroute_phase_last": float(rows[-1]["phase"]),
    }


def _grt_wirelength(run_dir: Path, die: str, row: Mapping[str, Any]) -> float:
    """Read FastRoute wire length, never the detailed-route wire-length report."""

    if "grt_wire_length_um" in row and (_routing_stage(run_dir) / "global_route_checkpoint.json").is_file():
        _source_grt_metrics_path(run_dir)
        value = _finite(row["grt_wire_length_um"], -1.0)
        if value < 0:
            raise ValueError("Invalid standalone GRT wire length")
        return value
    if "grt_wirelength_um" in row:
        value = _finite(row.get("grt_wirelength_um"), -1.0)
        if value >= 0.0:
            return value
    log = _source_grt_log(run_dir, die)
    if not log.is_file():
        raise ValueError("routing metrics have no GRT wire length for {}".format(die))
    matches = _GRT_WIRELENGTH.findall(
        log.read_text(encoding="utf-8", errors="replace")
    )
    if not matches:
        raise ValueError("routing log has no GRT-0018 wire length for {}".format(die))
    value = _finite(matches[-1], -1.0)
    if value < 0.0:
        raise ValueError("routing metrics have invalid GRT wire length for {}".format(die))
    return value


def _optional_runtime(result: Dict[str, Any], raw: Mapping[str, Any]) -> None:
    """Keep measured runtime only; an absent observation is not zero seconds."""
    result.pop("runtime_seconds", None)
    value = raw.get("runtime_seconds")
    if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        result["runtime_seconds"] = value


def routing_net_ordering_metrics(
    run_dir: Path, reference_run: Optional[Path]
) -> Dict[str, Any]:
    """Score only two-die GRT overflow and FastRoute wire length."""

    path = _source_grt_metrics_path(run_dir)
    raw = _read(path)
    dies = raw.get("dies", {}) if isinstance(raw.get("dies"), dict) else {}
    if set(dies) != {"bottom", "top"}:
        raise ValueError("routing metrics must contain exactly bottom and top dies")

    overflow_by_die: Dict[str, float] = {}
    wirelength_by_die: Dict[str, float] = {}
    congestion_by_die: Dict[str, float] = {}
    order_diagnostics: Dict[str, Dict[str, float]] = {}
    for die in ("bottom", "top"):
        row = dies[die]
        if not isinstance(row, dict) or row.get("status") not in {None, "success"}:
            raise ValueError("global routing did not complete successfully for {}".format(die))
        overflow = _finite(row.get("grt_total_overflow"), -1.0)
        congestion = _finite(row.get("grt_congestion_bins"), -1.0)
        if overflow < 0.0 or congestion < 0.0:
            raise ValueError("routing metrics have invalid GRT congestion for {}".format(die))
        overflow_by_die[die] = overflow
        congestion_by_die[die] = congestion
        wirelength_by_die[die] = _grt_wirelength(run_dir, die, row)
        order_diagnostics[die] = _net_order_diagnostics(run_dir, die)

    reference_overflow: Optional[float] = None
    reference_wirelength: Optional[float] = None
    if reference_run is not None:
        reference_path = _source_grt_metrics_path(reference_run)
        reference_raw = _read(reference_path)
        reference_dies = (
            reference_raw.get("dies", {})
            if isinstance(reference_raw.get("dies"), dict)
            else {}
        )
        if set(reference_dies) != {"bottom", "top"}:
            raise ValueError("reference routing metrics must contain bottom and top dies")
        reference_overflow = 0.0
        reference_wirelength = 0.0
        for die in ("bottom", "top"):
            reference_row = reference_dies[die]
            if not isinstance(reference_row, dict):
                raise ValueError("reference routing metrics are invalid for {}".format(die))
            value = _finite(reference_row.get("grt_total_overflow"), -1.0)
            if value < 0.0:
                raise ValueError("reference routing overflow is invalid for {}".format(die))
            reference_overflow += value
            reference_wirelength += _grt_wirelength(
                reference_run, die, reference_row
            )

    total_overflow = sum(overflow_by_die.values())
    total_wirelength = sum(wirelength_by_die.values())
    overflow_score = _ratio_score(total_overflow, reference_overflow, 1000.0)
    wirelength_score = _ratio_score(
        total_wirelength, reference_wirelength, 1_000_000.0
    )
    diagnostic_calls = sum(
        order_diagnostics[die]["calls"] for die in ("bottom", "top")
    )
    diagnostics_present = float(diagnostic_calls > 0.0)
    # A route can complete while the evolved hook is unreachable (for example,
    # with fewer than three congestion iterations).  Do not award such a run.
    # A lightly congested die can legitimately converge before FastRoute reaches
    # the reroute phase containing the hook, so require evidence from at least
    # one die rather than incorrectly rejecting the complete two-die route.
    grt_complete = diagnostics_present
    combined = grt_complete * (
        ROUTING_ORDER_OVERFLOW_WEIGHT * overflow_score
        + ROUTING_ORDER_WIRELENGTH_WEIGHT * wirelength_score
    )
    result = {
        "combined_score": combined,
        "overflow_score": overflow_score,
        "wirelength_score": wirelength_score,
        "overflow_weight": ROUTING_ORDER_OVERFLOW_WEIGHT,
        "wirelength_weight": ROUTING_ORDER_WIRELENGTH_WEIGHT,
        "grt_total_overflow": total_overflow,
        "grt_wirelength_um": total_wirelength,
        "grt_congestion_bins": sum(congestion_by_die.values()),
        "bottom_grt_total_overflow": overflow_by_die["bottom"],
        "top_grt_total_overflow": overflow_by_die["top"],
        "bottom_grt_wirelength_um": wirelength_by_die["bottom"],
        "top_grt_wirelength_um": wirelength_by_die["top"],
        "bottom_grt_congestion_bins": congestion_by_die["bottom"],
        "top_grt_congestion_bins": congestion_by_die["top"],
        "grt_complete": grt_complete,
        "routing_complete": grt_complete,
    }
    diagnostic_keys = (
        "calls", "changed_calls", "changed_ratio_max",
        "overflow_changed_ratio_max", "top10_overlap_min", "score_span_max",
        "overflow_net_fraction_max", "reroute_phase_last",
    )
    for die in ("bottom", "top"):
        for key in diagnostic_keys:
            if key in order_diagnostics[die]:
                result["{}_net_order_{}".format(die, key)] = order_diagnostics[die][key]
    result.update(
        {
            "net_order_diagnostics_present": diagnostics_present,
            "net_order_calls": diagnostic_calls,
            "net_order_changed_calls": sum(
                order_diagnostics[die]["changed_calls"] for die in ("bottom", "top")
            ),
        }
    )
    for key, aggregate in (("changed_ratio_max", max), ("overflow_changed_ratio_max", max),
                           ("top10_overlap_min", min)):
        values = [row[key] for row in order_diagnostics.values() if key in row]
        if values:
            result["net_order_" + key] = aggregate(values)
    _optional_runtime(result, raw)
    return result


def _routing_policy_diagnostics(
    run_dir: Path, die: str, policy: str
) -> Dict[str, float]:
    """Read current hook fields; sparse counters are observed lower bounds.

    Iteration-control 'calls' is an iteration index, possibly reset, and
    'changed' is a per-invocation flag. The other hooks log cumulative counters.
    Extrema describe logged samples only, never unobserved invocations.
    """
    prefix = _ROUTING_POLICY_PREFIXES[policy]
    pattern = re.compile(re.escape(prefix) + r"\s+([^\r\n]+)")
    log = _source_grt_log(run_dir, die)
    rows = []
    optional = {
        "ripup": ("changed_ratio", "selected", "selected_ratio", "score"),
        "maze_cost": ("changed_ratio", "multiplier_last"),
        "iteration": ("cost_scale", "search_scale", "ripup_offset",
                      "cost_height", "search_window", "ripup_threshold"),
    }[policy]
    if log.is_file():
        for match in pattern.finditer(log.read_text(encoding="utf-8", errors="replace")):
            raw = dict(_NET_ORDER_FIELD.findall(match.group(1)))
            if raw.get("die") != die:
                continue
            try:
                row = {key: float(raw[key]) for key in ("calls", "changed")}
            except (KeyError, ValueError):
                continue
            if not all(math.isfinite(value) for value in row.values()):
                continue
            # Preserve rejection of malformed activity/output evidence. Optional
            # absence is different from a supplied nonfinite diagnostic.
            evidence_field = {"ripup": "score", "maze_cost": "multiplier_last",
                              "iteration": "cost_scale"}[policy]
            try:
                evidence = [float(raw[key]) for key in ("changed_ratio", evidence_field)
                            if key in raw]
            except ValueError:
                continue
            if not all(math.isfinite(value) for value in evidence):
                continue
            for key in optional:
                try:
                    value = float(raw[key])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(value):
                    row[key] = value
            rows.append(row)
    if not rows:
        return {"rows": 0.0, "calls": 0.0, "changed": 0.0}
    result = {"rows": float(len(rows))}
    if policy == "iteration":
        result.update(calls=float(len(rows)), changed=sum(row["changed"] for row in rows),
                      iteration_index_last=rows[-1]["calls"],
                      changed_ratio_max=max(row["changed"] for row in rows))
    else:
        result.update(calls=max(row["calls"] for row in rows),
                      changed=max(row["changed"] for row in rows))
        values = [row["changed_ratio"] for row in rows if "changed_ratio" in row]
        if values:
            result["changed_ratio_max"] = max(values)
    for key in optional:
        values = [row[key] for row in rows if key in row]
        if not values or key == "changed_ratio":
            continue
        if key == "selected":
            result[key] = max(values)
            continue
        label = "multiplier" if key == "multiplier_last" else key
        result.update({label + "_min": min(values), label + "_max": max(values),
                       label + "_last": values[-1]})
    return result


def _routing_source_policy_metrics(
    run_dir: Path, reference_run: Optional[Path], *, policy: str,
    scoring: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Total-die overflow/wirelength score for active route policies.

    ``scoring`` selects the versioned contract from ``routing_scoring``:
    version 1 (default) is the smooth 0.1/0.9 blend; version 2 is an overflow
    hard gate followed by the reference/candidate wire-length ratio.
    """

    raw = _read(_source_grt_metrics_path(run_dir))
    dies = raw.get("dies", {}) if isinstance(raw.get("dies"), dict) else {}
    if set(dies) != {"bottom", "top"}:
        raise ValueError("routing metrics must contain exactly bottom and top dies")

    overflow_by_die: Dict[str, float] = {}
    wirelength_by_die: Dict[str, float] = {}
    congestion_by_die: Dict[str, float] = {}
    diagnostics: Dict[str, Dict[str, float]] = {}
    for die in ("bottom", "top"):
        row = dies[die]
        if not isinstance(row, dict) or row.get("status") not in {None, "success"}:
            raise ValueError("global routing did not complete successfully for {}".format(die))
        overflow = _finite(row.get("grt_total_overflow"), -1.0)
        congestion = _finite(row.get("grt_congestion_bins"), -1.0)
        if overflow < 0.0 or congestion < 0.0:
            raise ValueError("routing metrics have invalid GRT congestion for {}".format(die))
        overflow_by_die[die] = overflow
        congestion_by_die[die] = congestion
        wirelength_by_die[die] = _grt_wirelength(run_dir, die, row)
        diagnostics[die] = _routing_policy_diagnostics(run_dir, die, policy)

    reference_overflow: Optional[float] = None
    reference_wirelength: Optional[float] = None
    if reference_run is not None:
        reference_raw = _read(
            _source_grt_metrics_path(reference_run)
        )
        reference_dies = (
            reference_raw.get("dies", {})
            if isinstance(reference_raw.get("dies"), dict)
            else {}
        )
        if set(reference_dies) != {"bottom", "top"}:
            raise ValueError("reference routing metrics must contain bottom and top dies")
        reference_overflow = 0.0
        reference_wirelength = 0.0
        for die in ("bottom", "top"):
            reference_row = reference_dies[die]
            if not isinstance(reference_row, dict):
                raise ValueError("reference routing metrics are invalid for {}".format(die))
            overflow = _finite(reference_row.get("grt_total_overflow"), -1.0)
            if overflow < 0.0:
                raise ValueError("reference routing overflow is invalid for {}".format(die))
            reference_overflow += overflow
            reference_wirelength += _grt_wirelength(
                reference_run, die, reference_row
            )
        if reference_wirelength <= 0.0:
            raise ValueError("reference GRT wire length must be positive")

    total_overflow = sum(overflow_by_die.values())
    total_wirelength = sum(wirelength_by_die.values())
    # A die that starts with zero overflow can legitimately skip the reroute
    # loop containing these hooks. Require reachability evidence from at least
    # one die while retaining successful metrics from both dies as a hard gate.
    diagnostics_present = float(
        any(diagnostics[die]["rows"] > 0.0 for die in ("bottom", "top"))
    )
    from .routing_scoring import scoring_version
    version = scoring_version(scoring)
    if version == 2:
        result = _routing_v2_score(
            diagnostics_present, total_overflow, total_wirelength,
            reference_overflow, reference_wirelength, scoring or {},
        )
    elif version == 1:
        if reference_overflow is None or reference_wirelength is None:
            overflow_cost = 1.0 + total_overflow / 1000.0
            wirelength_cost = 1.0 + total_wirelength / 1_000_000.0
        else:
            overflow_cost = (total_overflow + 1000.0) / (
                reference_overflow + 1000.0
            )
            wirelength_cost = total_wirelength / reference_wirelength
        denominator = (
            ROUTING_ORDER_OVERFLOW_WEIGHT * overflow_cost
            + ROUTING_ORDER_WIRELENGTH_WEIGHT * wirelength_cost
        )
        combined = diagnostics_present / max(denominator, RATIO_DENOMINATOR_FLOOR)
        result = {
            "combined_score": combined,
            "overflow_cost": overflow_cost,
            "wirelength_cost": wirelength_cost,
            "overflow_weight": ROUTING_ORDER_OVERFLOW_WEIGHT,
            "wirelength_weight": ROUTING_ORDER_WIRELENGTH_WEIGHT,
            "overflow_score": 1.0 / max(overflow_cost, RATIO_DENOMINATOR_FLOOR),
            "wirelength_score": 1.0 / max(wirelength_cost, RATIO_DENOMINATOR_FLOOR),
        }
    else:
        raise ValueError("unsupported routing scoring version: {}".format(version))
    result.update({
        "grt_total_overflow": total_overflow,
        "grt_wirelength_um": total_wirelength,
        "grt_congestion_bins": sum(congestion_by_die.values()),
        "bottom_grt_total_overflow": overflow_by_die["bottom"],
        "top_grt_total_overflow": overflow_by_die["top"],
        "bottom_grt_wirelength_um": wirelength_by_die["bottom"],
        "top_grt_wirelength_um": wirelength_by_die["top"],
        "bottom_grt_congestion_bins": congestion_by_die["bottom"],
        "top_grt_congestion_bins": congestion_by_die["top"],
        "grt_complete": diagnostics_present,
        "routing_complete": diagnostics_present,
        "policy_diagnostics_present": diagnostics_present,
        "policy_diagnostic_rows": sum(
            diagnostics[die]["rows"] for die in ("bottom", "top")
        ),
        "policy_calls": sum(
            diagnostics[die]["calls"] for die in ("bottom", "top")
        ),
        "policy_changed": sum(
            diagnostics[die]["changed"] for die in ("bottom", "top")
        ),
    })
    values = [row["changed_ratio_max"] for row in diagnostics.values()
              if "changed_ratio_max" in row]
    if values:
        result["policy_changed_ratio_max"] = max(values)
    for die in ("bottom", "top"):
        for key, value in diagnostics[die].items():
            result["{}_policy_{}".format(die, key)] = value
    _optional_runtime(result, raw)
    return result


def _routing_v2_score(
    diagnostics_present: float,
    total_overflow: float,
    total_wirelength: float,
    reference_overflow: Optional[float],
    reference_wirelength: Optional[float],
    scoring: Mapping[str, Any],
) -> Dict[str, Any]:
    """Version-2 routing score: overflow hard gate, then W_ref / W."""

    from .routing_scoring import overflow_gate

    gate = scoring["overflow_gate"]
    result: Dict[str, Any] = {
        "routing_scoring_version": 2,
        "overflow_gate_max_ratio": float(gate["max_ratio"]),
        "overflow_gate_abs_slack": float(gate["abs_slack"]),
    }
    if reference_overflow is None or reference_wirelength is None:
        # Bootstrap without a reference: no gate can be evaluated, so only
        # an unqualified proxy is reported for smoke checks.
        wirelength_cost = 1.0 + total_wirelength / 1_000_000.0
        result.update({
            "combined_score": diagnostics_present
            / max(wirelength_cost, RATIO_DENOMINATOR_FLOOR),
            "wirelength_cost": wirelength_cost,
            "wirelength_score": 1.0 / max(wirelength_cost, RATIO_DENOMINATOR_FLOOR),
            "overflow_gate_passed": 0.0,
            "overflow_gate_evaluated": 0.0,
            "overflow_gate_reason": (
                "overflow gate not evaluated: no reference run; "
                "combined_score is the unqualified bootstrap proxy "
                "grt_complete / (1 + W / 1e6)"
            ),
        })
        return result
    if total_wirelength <= 0.0:
        raise ValueError("candidate GRT wire length must be positive")
    passed, limit, reason = overflow_gate(total_overflow, reference_overflow, scoring)
    wirelength_cost = total_wirelength / reference_wirelength
    wirelength_score = reference_wirelength / total_wirelength
    result.update({
        "combined_score": diagnostics_present * wirelength_score if passed else 0.0,
        "wirelength_cost": wirelength_cost,
        "wirelength_score": wirelength_score,
        "overflow_gate_passed": 1.0 if passed else 0.0,
        "overflow_gate_evaluated": 1.0,
        "overflow_gate_limit": limit,
        "overflow_gate_reason": reason,
        "reference_grt_total_overflow": reference_overflow,
        "reference_grt_wirelength_um": reference_wirelength,
    })
    return result


def routing_ripup_policy_metrics(
    run_dir: Path, reference_run: Optional[Path], *,
    scoring: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return _routing_source_policy_metrics(
        run_dir, reference_run, policy="ripup", scoring=scoring
    )


def routing_maze_cost_metrics(
    run_dir: Path, reference_run: Optional[Path], *,
    scoring: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return _routing_source_policy_metrics(
        run_dir, reference_run, policy="maze_cost", scoring=scoring
    )


def routing_iteration_controller_metrics(
    run_dir: Path, reference_run: Optional[Path], *,
    scoring: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return _routing_source_policy_metrics(
        run_dir, reference_run, policy="iteration", scoring=scoring
    )


def _validated_sta_codesign_run(run_dir: Path) -> Dict[str, Any]:
    """Load one complete TP-GNN -> placement -> two-pass OPT/STA result."""

    partition = _read(run_dir / STAGE_DIRS["partition"] / "partition_metrics.json")
    placement = _read(run_dir / STAGE_DIRS["placement"] / "metrics.json")
    opt = _read(run_dir / STAGE_DIRS["opt"] / "metrics.json")

    instance_count = int(partition.get("instance_count", 0))
    top_count = int(partition.get("top_count", -1))
    bottom_count = int(partition.get("bottom_count", -1))
    if instance_count <= 0 or top_count + bottom_count != instance_count:
        raise ValueError("STA co-design partition assignment is incomplete")
    area_balance_error = _finite(
        partition.get("area_balance_error"), float("inf")
    )
    if area_balance_error > 0.100000000001:
        raise ValueError("STA co-design partition violates the 10% area-balance gate")

    dies = placement.get("dies")
    if not isinstance(dies, Mapping) or set(dies) != {"bottom", "top"}:
        raise ValueError("STA co-design placement must contain bottom and top dies")
    for die in ("bottom", "top"):
        row = dies[die]
        if not isinstance(row, Mapping):
            raise ValueError("STA co-design placement metrics are invalid for {}".format(die))
        if int(row.get("missing_connection_object_count", -1)) != 0:
            raise ValueError("STA co-design placement has missing objects on {}".format(die))
    hpwl = _finite(placement.get("total_object_origin_hpwl_um"), -1.0)
    if hpwl < 0.0:
        raise ValueError("STA co-design placement has invalid HPWL")

    if opt.get("status") != "success":
        raise ValueError("STA co-design OPT did not complete successfully")
    if opt.get("algorithm") not in {
        "openroad-coordinated-passive-cutpoint",
        "openroad-coordinated-passive-boundary",
    }:
        raise ValueError("STA co-design OPT used an unexpected algorithm")
    boundary_budget_model = opt.get("boundary_budget_model", "passive-cutpoint")
    if boundary_budget_model not in {
        "passive-cutpoint",
        "passive-endpoint-pin-envelope",
    }:
        raise ValueError("STA co-design OPT used an unexpected boundary budget model")
    if int(opt.get("passes_run", -1)) != 2 or int(opt.get("iterations_run", -1)) != 2:
        raise ValueError("STA co-design evaluation requires exactly two OPT passes")
    die_passes = opt.get("die_passes")
    if not isinstance(die_passes, Mapping) or {
        str(key): int(value) for key, value in die_passes.items()
    } != {"bottom": 2, "top": 2}:
        raise ValueError("STA co-design evaluation requires two passes on both dies")
    pass_records = opt.get("pass_records")
    if not isinstance(pass_records, list) or len(pass_records) != 2:
        raise ValueError("STA co-design OPT is missing its two-pass audit trail")

    validity = opt.get("timing_validity")
    if not isinstance(validity, Mapping):
        raise ValueError("STA co-design OPT is missing timing-validity evidence")
    expected_validity = {
        "authoritative_timing_view": "continuous-passive-hbt",
        "graph_completeness": "passed",
        "die_internal_parasitics_in_global_sta": True,
        "hbt_is_passive_rc_in_authoritative_global_sta": True,
        "remote_die_load_visible_to_source_driver": True,
    }
    for key, expected in expected_validity.items():
        if validity.get(key) != expected:
            raise ValueError(
                "STA co-design timing-validity gate failed for {}".format(key)
            )
    if validity.get("passive_cutpoint_query_is_diagnostic_only", False):
        raise ValueError("STA co-design boundary feedback is diagnostic-only")
    if (
        boundary_budget_model == "passive-endpoint-pin-envelope"
        and validity.get("passive_cutpoint_query_exact") is not False
    ):
        raise ValueError("STA co-design endpoint-envelope provenance is inconsistent")
    if int(validity.get("unclocked_or_unconstrained_endpoints", -1)) != 0:
        raise ValueError("STA co-design has unclocked or unconstrained endpoints")
    if int(validity.get("missing_liberty_cells", -1)) != 0:
        raise ValueError("STA co-design has missing Liberty cells")
    if int(validity.get("combinational_loops", -1)) != 0:
        raise ValueError("STA co-design has combinational loops")

    coverage = opt.get("final_boundary_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("STA co-design is missing final boundary coverage")
    if int(coverage.get("sta_unconstrained_endpoints", -1)) != 0:
        raise ValueError("STA co-design final global STA has unconstrained endpoints")
    final_global = opt.get("final_global")
    initial_global = opt.get("initial_global")
    if not isinstance(final_global, Mapping) or not isinstance(initial_global, Mapping):
        raise ValueError("STA co-design is missing initial/final global STA")
    if _finite(final_global.get("sta_completeness_checks_passed"), 0.0) != 1.0:
        raise ValueError("STA co-design global STA completeness checks failed")
    if _finite(final_global.get("passive_hbt_rc"), 0.0) != 1.0:
        raise ValueError("STA co-design authoritative STA did not include passive HBT RC")

    wns = _finite(final_global.get("wns_setup"), float("nan"))
    tns = _finite(final_global.get("tns_setup"), float("nan"))
    if not math.isfinite(wns) or not math.isfinite(tns):
        raise ValueError("STA co-design final setup timing is not finite")
    return {
        "partition": partition,
        "placement": placement,
        "opt": opt,
        "instance_count": instance_count,
        "area_balance_error": area_balance_error,
        "hpwl": hpwl,
        "wns": wns,
        "tns": tns,
        "wns_deficit": max(0.0, -wns),
        "tns_deficit": max(0.0, -tns),
    }


def _relative_cost(value: float, reference: float) -> float:
    if reference > RATIO_DENOMINATOR_FLOOR:
        return value / reference
    if value <= RATIO_DENOMINATOR_FLOOR:
        return 1.0
    return 1.0 + value / RATIO_DENOMINATOR_FLOOR


def tp_gnn_placement_sta_metrics(
    run_dir: Path, reference_run: Optional[Path]
) -> Dict[str, Any]:
    """Score final two-pass global setup timing, allowing die-level trade-offs."""

    row = _validated_sta_codesign_run(run_dir)
    reference = _validated_sta_codesign_run(reference_run) if reference_run else None
    if reference is not None:
        candidate_boundary_model = row["opt"].get("boundary_budget_model")
        reference_boundary_model = reference["opt"].get("boundary_budget_model")
        if candidate_boundary_model != reference_boundary_model:
            raise ValueError(
                "STA co-design candidate/reference boundary models differ: "
                "{} != {}".format(
                    candidate_boundary_model, reference_boundary_model
                )
            )
    if reference is None:
        normalized_wns_cost = 1.0
        normalized_tns_cost = 1.0
        normalized_sta_cost = 1.0
        combined_score = 1.0
    else:
        normalized_wns_cost = _relative_cost(
            row["wns_deficit"], reference["wns_deficit"]
        )
        normalized_tns_cost = _relative_cost(
            row["tns_deficit"], reference["tns_deficit"]
        )
        normalized_sta_cost = (
            STA_WNS_WEIGHT * normalized_wns_cost
            + STA_TNS_WEIGHT * normalized_tns_cost
        )
        combined_score = 1.0 / max(
            normalized_sta_cost, RATIO_DENOMINATOR_FLOOR
        )

    partition = row["partition"]
    placement = row["placement"]
    opt = row["opt"]
    records = opt["pass_records"]
    result = {
        "combined_score": combined_score,
        "normalized_sta_cost": normalized_sta_cost,
        "normalized_wns_cost": normalized_wns_cost,
        "normalized_tns_cost": normalized_tns_cost,
        "sta_wns_weight": STA_WNS_WEIGHT,
        "sta_tns_weight": STA_TNS_WEIGHT,
        "final_global_wns_setup_ns": row["wns"],
        "final_global_tns_setup_ns": row["tns"],
        "initial_global_wns_setup_ns": _finite(
            opt["initial_global"].get("wns_setup")
        ),
        "initial_global_tns_setup_ns": _finite(
            opt["initial_global"].get("tns_setup")
        ),
        "pass1_global_wns_setup_ns": _finite(
            records[0].get("global_after", {}).get("wns_setup")
        ),
        "pass1_global_tns_setup_ns": _finite(
            records[0].get("global_after", {}).get("tns_setup")
        ),
        "pass2_global_wns_setup_ns": _finite(
            records[1].get("global_after", {}).get("wns_setup")
        ),
        "pass2_global_tns_setup_ns": _finite(
            records[1].get("global_after", {}).get("tns_setup")
        ),
        "opt_passes": 2.0,
        "sta_graph_complete": 1.0,
        "sta_unconstrained_endpoints": 0.0,
        "timing_view_pre_cts_placement_estimated": 1.0,
        "cut_net_count": int(partition.get("cut_net_count", 0)),
        "cut_pin_count": int(partition.get("cut_pin_count", 0)),
        "timing_weighted_cut": _finite(partition.get("timing_weighted_cut")),
        "area_balance_error": row["area_balance_error"],
        "placement_total_hpwl_um": row["hpwl"],
        "placement_cross_die_hpwl_um": _finite(
            placement.get("cross_die_local_object_origin_hpwl_um")
        ),
        "hbt_count": int(opt["final_boundary_coverage"].get("total", 0)),
        "opt_runtime_seconds": _finite(opt.get("runtime_seconds")),
    }
    try:
        result.update(_placement_native(run_dir))
    except (OSError, ValueError):
        # The authoritative objective is downstream STA. Native DREAMPlace
        # congestion remains optional diagnostic evidence for full-flow runs.
        pass
    return result


def _validate_thermal_frontend_methods(run_dir: Path) -> None:
    """Check recorded execution, not profile names or a rewritten config label."""
    partition = _read(run_dir / "20_partition/partition_manifest.json")
    report = _read(run_dir / "20_partition/tp_gnn_report.json")
    partition_stage = _read(run_dir / "20_partition/stage_manifest.json")
    method = validate_tp_gnn_method(report, partition.get("method"), partition_stage.get("method"))
    for stage, directory, method in (
        ("partition", "20_partition", method),
        ("placement", "30_placement", "concurrent-dreamplace"),
    ):
        manifest = _read(run_dir / directory / "stage_manifest.json")
        if (manifest.get("stage") != stage or manifest.get("status") != "success"
                or manifest.get("method") != method):
            raise ValueError("thermal frontend requires recorded {} method {}".format(stage, method))


def _validated_tp_gnn_placement_thermal_run(
    run_dir: Path, *, post_cts=False, frozen_input_run: Optional[Path] = None
) -> Dict[str, Any]:
    """Validate one complete partition/placement-to-HotSpot result."""

    root = run_dir.expanduser().resolve()
    run = _read(root / "run.json")
    if run.get("status") != "success":
        raise ValueError("thermal co-design run is not successful")
    stages = run.get("stages")
    legacy_required_stages = {
        "input",
        "floorplan",
        "partition",
        "placement",
        "pdn",
        "hbt",
        "opt",
        "cts",
        "routing",
        "report",
        "thermal",
    }
    split_required_stages = (
        legacy_required_stages - {"routing"}
    ) | {"global_routing", "detailed_routing"}
    recorded_stages = set(stages) if isinstance(stages, Mapping) else set()
    if "detailed_routing" in recorded_stages:
        recorded_stages.discard("routing")
        required_stages = split_required_stages
    else:
        required_stages = legacy_required_stages
    # Timing extraction became an explicit step after the original thermal
    # benchmark profiles were released.  Require it when the run records it,
    # while preserving validation of historical references and partition-start
    # replays whose frozen prefix already owns step 15.
    if "timing_extract" in recorded_stages:
        required_stages.add("timing_extract")
    if post_cts:
        required_stages -= {
            "routing",
            "global_routing",
            "detailed_routing",
            "report",
            "thermal",
        }
    # The runner records only stages executed in this invocation. A replay
    # starts at partition and copies input/floorplan from the frozen bundle.
    # Accept precisely that suffix, and validate the prefix separately rather
    # than allowing arbitrary missing/skipped stages or fabricating success.
    replay = post_cts and recorded_stages == required_stages - {"input", "floorplan"}
    if replay:
        from .frozen_prefix import validate_frozen_prefix
        validate_frozen_prefix(root, frozen_input_run)
    elif not isinstance(stages, Mapping) or recorded_stages != required_stages:
        raise ValueError("thermal co-design run has an incomplete stage set")
    if any(
        not isinstance(stages[stage], Mapping)
        or stages[stage].get("status") != "success"
        for stage in recorded_stages
    ):
        raise ValueError("thermal co-design run contains a failed or skipped stage")

    _validate_thermal_frontend_methods(root)
    if post_cts:
        _validated_sta_codesign_run(root)
        cts = _read(root / "70_cts/metrics.json")
        checkpoint = _read(root / "70_cts/checkpoint.json")
        if (cts.get("status") != "success"
                or cts.get("algorithm") not in {
                    "openroad-coordinated-insertion-delay", "openroad-balanced"}
                or checkpoint.get("stage") != "cts"
                or checkpoint.get("ready_for_route") is not True):
            raise ValueError("post-CTS thermal requires completed CTS")
        published_cts = checkpoint.get("cts")
        if cts["algorithm"] == "openroad-balanced" or published_cts is not None:
            if (not isinstance(published_cts, Mapping)
                    or published_cts.get("algorithm") != cts["algorithm"]
                    or published_cts.get("metrics") != "metrics.json"):
                raise ValueError("post-CTS thermal CTS checkpoint identity mismatch")
    partition = _read(root / "20_partition/partition_metrics.json")
    instance_count = int(partition.get("instance_count", 0))
    if (
        instance_count <= 0
        or int(partition.get("top_count", -1))
        + int(partition.get("bottom_count", -1))
        != instance_count
    ):
        raise ValueError("thermal co-design partition assignment is incomplete")
    area_balance_error = _finite(partition.get("area_balance_error"), float("inf"))
    if area_balance_error > 0.100000000001:
        raise ValueError("thermal co-design partition violates the 10% area-balance gate")

    placement = _read(root / "30_placement/metrics.json")
    dies = placement.get("dies")
    if not isinstance(dies, Mapping) or set(dies) != {"bottom", "top"}:
        raise ValueError("thermal co-design placement must contain both dies")
    for die in ("bottom", "top"):
        if not isinstance(dies[die], Mapping) or int(
            dies[die].get("missing_connection_object_count", -1)
        ) != 0:
            raise ValueError("thermal co-design placement is incomplete on {}".format(die))

    thermal = _read(root / ("95_thermal_snapshot" if post_cts else "95_thermal") / "metrics.json")
    if post_cts:
        power_model = thermal.get("power", {})
        if (thermal.get("source_stage") != "cts"
                or power_model.get("activity_model") != "vectorless-primary-input-0.1-placement-estimated-rc"
                or power_model.get("provenance", {}).get("model") != "die-local-openroad-placement-estimated"):
            raise ValueError("post-CTS thermal power provenance mismatch")
        snapshot = _read(root / "95_thermal_snapshot/snapshot_manifest.json")
        if snapshot.get("status") != "success" or snapshot.get("source_stage") != "cts":
            raise ValueError("post-CTS thermal snapshot incomplete")
        if Path(snapshot.get("source_checkpoint", "")).resolve() != (root / "70_cts/checkpoint.json").resolve():
            raise ValueError("post-CTS thermal snapshot uses a different checkpoint")
        source_checkpoint = power_model.get("provenance", {}).get("checkpoint")
        if (not isinstance(source_checkpoint, str) or not source_checkpoint
                or Path(source_checkpoint).resolve() != (root / "70_cts/checkpoint.json").resolve()):
            raise ValueError("post-CTS thermal power uses a different checkpoint")
        if (snapshot.get("schema_version") != 1 or snapshot.get("metrics") != "metrics.json"
                or snapshot.get("grid_size") != 10
                or not math.isclose(_finite(snapshot.get("ambient_k"), float("nan")),
                                    318.15, rel_tol=0.0, abs_tol=1e-9)):
            raise ValueError("post-CTS thermal snapshot metadata mismatch")
        extraction = thermal.get("power_extraction", {})
        if set(extraction) != {"bottom", "top"}:
            raise ValueError("post-CTS thermal missing die power extraction")
        for die in ("bottom", "top"):
            evidence = extraction[die]
            if (int(evidence.get("instance_count", 0)) <= 0
                    or int(evidence.get("warning_count", -1)) != 0):
                raise ValueError("post-CTS thermal invalid or warning-bearing power extraction on " + die)
    if (
        thermal.get("stage") != ("thermal_snapshot" if post_cts else "thermal")
        or thermal.get("status") != "success"
        or thermal.get("algorithm") != "hotspot-steady-grid"
        or int(thermal.get("warning_count", -1)) != 0
    ):
        raise ValueError("thermal co-design HotSpot identity/status gate failed")
    if int(thermal.get("grid_size", -1)) != 10:
        raise ValueError("thermal co-design requires the fixed 10x10 HotSpot grid")
    ambient_k = _finite(thermal.get("ambient_k"), float("nan"))
    if not math.isclose(ambient_k, 318.15, abs_tol=1e-9):
        raise ValueError("thermal co-design requires the fixed 318.15 K ambient")
    temperature = thermal.get("temperature")
    if not isinstance(temperature, Mapping):
        raise ValueError("thermal co-design has no temperature metrics")
    die_rows = []
    weighted_sum = 0.0
    tile_count = 0
    for die in ("bottom", "top"):
        row = temperature.get(die)
        if not isinstance(row, Mapping):
            raise ValueError("HotSpot output is missing the {} physical layer".format(die))
        average = _finite(row.get("average_k"), float("nan"))
        count = int(row.get("tile_count", 0))
        if not math.isfinite(average) or average < ambient_k or count <= 0 or (post_cts and count != 100):
            raise ValueError("HotSpot {} average temperature is invalid".format(die))
        weighted_sum += average * count
        tile_count += count
        die_rows.append((die, average))
    average_k = weighted_sum / tile_count
    published_average = _finite(temperature.get("global_average_k"), average_k)
    if not math.isclose(average_k, published_average, abs_tol=1e-8):
        raise ValueError("HotSpot global average does not match its physical die tiles")
    power = thermal.get("power")
    if not isinstance(power, Mapping):
        raise ValueError("thermal co-design has no power metrics")
    total_power = _finite(power.get("total_w"), -1.0)
    conservation_error = _finite(power.get("conservation_error_w"), float("inf"))
    if total_power <= 0 or conservation_error > max(1e-12, total_power * 1e-9):
        raise ValueError("thermal co-design power mapping is invalid")
    return {
        "partition": partition,
        "placement": placement,
        "thermal": thermal,
        "area_balance_error": area_balance_error,
        "average_k": average_k,
        "average_c": average_k - 273.15,
        "bottom_average_k": dict(die_rows)["bottom"],
        "top_average_k": dict(die_rows)["top"],
        "maximum_c": _finite(temperature.get("global_maximum_c"), float("nan")),
        "total_power_w": total_power,
        **({"cts_algorithm": cts["algorithm"]} if post_cts else {}),
    }


def tp_gnn_placement_thermal_metrics(
    run_dir: Path, reference_run: Optional[Path], *, post_cts=False,
    frozen_input_run: Optional[Path] = None
) -> Dict[str, Any]:
    """Score HotSpot average temperature after fixed full-flow evaluation."""

    row = _validated_tp_gnn_placement_thermal_run(
        run_dir, post_cts=post_cts, frozen_input_run=frozen_input_run)
    # A bootstrap run is the reference itself. Candidate source evolution may
    # alter decoding, but the fixed reference must retain the STA baseline.
    reference_root = (reference_run if reference_run is not None else run_dir).expanduser().resolve()
    report = _read(reference_root / "20_partition/tp_gnn_report.json")
    reference_method = validate_tp_gnn_method(report)
    validate_tp_gnn_method(_read(run_dir / "20_partition/tp_gnn_report.json"), expected_method=reference_method)
    config = report.get("config")
    if (not isinstance(config, Mapping)
            or config.get("decoder") != "global-2way"):
        raise ValueError("thermal reference requires recorded TP-GNN global-2way decoding")
    reference = (
        _validated_tp_gnn_placement_thermal_run(
            reference_run, post_cts=post_cts, frozen_input_run=frozen_input_run)
        if reference_run is not None
        else None
    )
    if reference is not None:
        candidate_thermal = row["thermal"]
        reference_thermal = reference["thermal"]
        for key in ("grid_size", "ambient_k"):
            if candidate_thermal.get(key) != reference_thermal.get(key):
                raise ValueError("thermal candidate/reference differ in {}".format(key))
        if candidate_thermal.get("stack") != reference_thermal.get("stack"):
            raise ValueError("thermal candidate/reference use different stack models")
        normalized_cost = row["average_k"] / reference["average_k"]
        combined_score = reference["average_k"] / max(
            row["average_k"], RATIO_DENOMINATOR_FLOOR
        )
    else:
        normalized_cost = 1.0
        combined_score = 1.0
    placement = row["placement"]
    result = {
        "combined_score": combined_score,
        "normalized_average_temperature_cost": normalized_cost,
        "global_average_temperature_k": row["average_k"],
        "global_average_temperature_c": row["average_c"],
        "bottom_average_temperature_k": row["bottom_average_k"],
        "top_average_temperature_k": row["top_average_k"],
        "global_maximum_temperature_c": row["maximum_c"],
        "total_power_w": row["total_power_w"],
        "thermal_model": row["thermal"].get("stack", {}).get("model_status"),
        "thermal_grid_size": 10.0,
        "thermal_ambient_k": 318.15,
        "cut_net_count": int(row["partition"].get("cut_net_count", 0)),
        "area_balance_error": row["area_balance_error"],
        "placement_total_hpwl_um": _finite(
            placement.get("total_object_origin_hpwl_um")
        ),
        "runs_successfully": 1.0,
    }
    if post_cts:
        if reference is not None:
            if row["cts_algorithm"] != reference["cts_algorithm"]:
                raise ValueError("post-CTS thermal candidate/reference CTS algorithms differ")
            candidate_opt = _read(run_dir / "60_opt/metrics.json")
            reference_opt = _read(reference_run / "60_opt/metrics.json")
            if candidate_opt.get("boundary_budget_model") != reference_opt.get("boundary_budget_model"):
                raise ValueError("post-CTS thermal OPT boundary models differ")
        result.update(opt_passes=2.0, cts_completed=1.0,
                      thermal_evaluation_boundary="two-opt-cts-cell-power-v1",
                      power_model="die-local-openroad-placement-estimated")
    return result


def tp_gnn_placement_cts_thermal_metrics(
    run_dir: Path, reference_run: Optional[Path], *, frozen_input_run: Optional[Path] = None
) -> Dict[str, Any]:
    return tp_gnn_placement_thermal_metrics(
        run_dir, reference_run, post_cts=True, frozen_input_run=frozen_input_run)


EXTRACTORS = {
    "partition": partition_metrics,
    "tp_gnn_learning_objective": tp_gnn_learning_objective_metrics,
    "placement_source": placement_metrics,
    "placement_initialization_source": placement_component_metrics,
    "placement_preconditioner_source": placement_component_metrics,
    "placement_parameters": placement_metrics,
    "placement_dynamic_parameters": placement_dynamic_metrics,
    "routing_parameters": routing_metrics,
    "routing_net_ordering_source": routing_net_ordering_metrics,
    "routing_ripup_policy_source": routing_ripup_policy_metrics,
    "routing_maze_cost_source": routing_maze_cost_metrics,
    "routing_iteration_controller_source": routing_iteration_controller_metrics,
    "tp_gnn_placement_sta": tp_gnn_placement_sta_metrics,
    "placement_two_pass_sta": tp_gnn_placement_sta_metrics,
    "tp_gnn_placement_thermal": tp_gnn_placement_thermal_metrics,
    "tp_gnn_placement_cts_thermal": tp_gnn_placement_cts_thermal_metrics,
}


def extract_metrics(kind: str, run_dir: Path, reference_run: Optional[Path], *,
                    frozen_input_run: Optional[Path] = None,
                    task_id: Optional[str] = None,
                    scoring: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    from .routing_scoring import parse_scoring, scoring_version
    # Validate and normalize (idempotently) any explicit score contract.
    scoring = parse_scoring(scoring, kind, source="scoring") if scoring else {}
    if kind in {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}:
        if task_id is not None and task_id != kind:
            raise ValueError("GNN evolution task identity differs from its kind")
        from .gnn_evolution_timing import timing_metrics
        return timing_metrics(run_dir, reference_run, task_id=kind)
    if kind == "hbt_assignment_policy":
        from .hbt_native import timing_metrics
        return timing_metrics(run_dir, reference_run, frozen_input=frozen_input_run)
    if kind in {"tp_gnn_bin_timing", "tp_gnn_post_placement_timing"}:
        from .post_placement_timing import timing_metrics, TASK_DECODERS
        if task_id is not None and task_id != kind:
            raise ValueError("Timing task identity does not match its decoder-bound task kind")
        return timing_metrics(run_dir, reference_run, expected_decoder=TASK_DECODERS[kind])
    if task_id == "placement_dynamic_parameters_reference_20260923":
        if kind != "placement_dynamic_parameters":
            raise ValueError("Nesterov task identity does not match its task kind")
        return placement_nesterov_metrics(run_dir, reference_run)
    if kind not in EXTRACTORS:
        raise ValueError("unknown task kind: {}".format(kind))
    if kind == "tp_gnn_placement_cts_thermal":
        return EXTRACTORS[kind](run_dir, reference_run, frozen_input_run=frozen_input_run)
    if scoring_version(scoring) != 1:
        return EXTRACTORS[kind](run_dir, reference_run, scoring=scoring)
    return EXTRACTORS[kind](run_dir, reference_run)
