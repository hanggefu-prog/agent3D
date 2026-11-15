"""Trusted objective extraction for evaluator-owned global-task runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from local_bench.metrics import (
    RATIO_DENOMINATOR_FLOOR,
    STA_TNS_WEIGHT,
    STA_WNS_WEIGHT,
    tp_gnn_placement_sta_metrics,
)

from .evaluation import validate_trusted_stage
from .model import GlobalTaskSpec


def _json_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("trusted metric file must contain an object: {}".format(path))
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be finite".format(label))
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be finite".format(label)) from error
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(label))
    return result


def _validate_run_chain(task: GlobalTaskSpec, run_dir: Path) -> None:
    root = run_dir.expanduser().resolve()
    status = _json_object(root / "run.json")
    if status.get("status") != "success":
        raise ValueError("trusted evaluation run is not successful")
    stages = status.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("trusted evaluation run has no stage status")
    stages = dict(stages)
    if "detailed_routing" in stages:
        stages.pop("routing", None)
    expected = {stage.runner_name for stage in task.evaluation.stages}
    if set(stages) != expected:
        raise ValueError("trusted evaluation run has the wrong stage set")
    for stage in task.evaluation.stages:
        record = stages[stage.runner_name]
        if isinstance(record, Mapping) and record.get("status") == "skipped":
            raise ValueError("objective stage was skipped: {}".format(stage.runner_name))
        validate_trusted_stage(root / stage.directory, stage.runner_name)


def _thermal_row(task: GlobalTaskSpec, run_dir: Path) -> Dict[str, float]:
    _validate_run_chain(task, run_dir)
    metrics = _json_object(run_dir.expanduser().resolve() / "95_thermal/metrics.json")
    if (
        metrics.get("stage") != "thermal"
        or metrics.get("status") != "success"
        or metrics.get("algorithm") != "hotspot-steady-grid"
        or int(metrics.get("warning_count", -1)) != 0
    ):
        raise ValueError("HotSpot thermal metrics failed their identity/status gates")
    parameters = task.objective.parameters
    if int(metrics.get("grid_size", -1)) != int(parameters["grid_size"]):
        raise ValueError("candidate and task use different HotSpot grid sizes")
    ambient_k = _finite(metrics.get("ambient_k"), "ambient_k")
    if not math.isclose(ambient_k, float(parameters["ambient_k"]), abs_tol=1e-9):
        raise ValueError("candidate and task use different HotSpot ambient temperatures")
    temperature = metrics.get("temperature")
    if not isinstance(temperature, Mapping):
        raise ValueError("HotSpot metrics have no temperature object")
    bottom = temperature.get("bottom")
    top = temperature.get("top")
    if not isinstance(bottom, Mapping) or not isinstance(top, Mapping):
        raise ValueError("HotSpot metrics must contain both physical die layers")
    average_k = _finite(temperature.get("global_average_k"), "global_average_k")
    average_c = _finite(temperature.get("global_average_c"), "global_average_c")
    maximum_c = _finite(temperature.get("global_maximum_c"), "global_maximum_c")
    if average_k < ambient_k or not math.isclose(average_c, average_k - 273.15, abs_tol=1e-8):
        raise ValueError("HotSpot global average temperature is inconsistent")
    power = metrics.get("power")
    if not isinstance(power, Mapping):
        raise ValueError("HotSpot metrics have no power object")
    total_power = _finite(power.get("total_w"), "total power")
    conservation_error = _finite(power.get("conservation_error_w"), "power conservation")
    if total_power <= 0 or conservation_error > max(1e-12, total_power * 1e-9):
        raise ValueError("HotSpot power mapping failed conservation")
    return {
        "global_average_temperature_k": average_k,
        "global_average_temperature_c": average_c,
        "global_maximum_temperature_c": maximum_c,
        "total_power_w": total_power,
        "bottom_average_temperature_k": _finite(
            bottom.get("average_k"), "bottom average temperature"
        ),
        "top_average_temperature_k": _finite(
            top.get("average_k"), "top average temperature"
        ),
    }


def score_evaluated_run(
    task: GlobalTaskSpec,
    candidate_run: Path,
    reference_run: Path,
) -> Dict[str, Any]:
    """Return trusted, higher-is-better feedback for a fixed global objective."""

    if task.objective.kind == "sta_wns_tns":
        parameters = task.objective.parameters
        if (
            not math.isclose(float(parameters["wns_weight"]), STA_WNS_WEIGHT)
            or not math.isclose(float(parameters["tns_weight"]), STA_TNS_WEIGHT)
            or not math.isclose(
                float(parameters["deficit_floor"]), RATIO_DENOMINATOR_FLOOR
            )
        ):
            raise ValueError("global STA objective differs from the maintained 0.6/0.4 contract")
        result = tp_gnn_placement_sta_metrics(
            candidate_run.expanduser().resolve(), reference_run.expanduser().resolve()
        )
        result["objective_kind"] = task.objective.kind
        return result
    if task.objective.kind == "hotspot_average_temperature":
        if task.objective.evaluator_profile == "tp_gnn_placement_cts_thermal_20261008":
            raise ValueError("Latest CTS thermal scoring requires the sealed reference consumer")
        if task.objective.evaluator_profile == "tp_gnn_placement_cts_thermal":
            from .verifiers.open3dbench_codesign_verifier import _score_cts_thermal
            return _score_cts_thermal(candidate_run, reference_run)
        candidate = _thermal_row(task, candidate_run)
        reference = _thermal_row(task, reference_run)
        candidate_average = candidate["global_average_temperature_k"]
        reference_average = reference["global_average_temperature_k"]
        if candidate_average <= 0 or reference_average <= 0:
            raise ValueError("absolute HotSpot average temperature must be positive")
        return {
            "objective_kind": task.objective.kind,
            "combined_score": reference_average
            / max(candidate_average, RATIO_DENOMINATOR_FLOOR),
            "normalized_average_temperature_cost": candidate_average / reference_average,
            **candidate,
            "reference_global_average_temperature_k": reference_average,
            "reference_global_average_temperature_c": reference[
                "global_average_temperature_c"
            ],
        }
    raise ValueError("task objective is not a fixed benchmark score: {}".format(task.objective.kind))
