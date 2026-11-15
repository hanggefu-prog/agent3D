from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills/open3dbench-thermal-evaluation/scripts/run_thermal_attribution.py"
)
SPEC = importlib.util.spec_from_file_location("thermal_attribution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metrics(average_rise: float, maximum_rise: float) -> dict:
    ambient = 318.15
    return {
        "status": "success",
        "algorithm": "hotspot-steady-grid",
        "ambient_k": ambient,
        "grid_size": 10,
        "power": {"total_w": 1.0},
        "temperature": {
            "global_average_k": ambient + average_rise,
            "global_maximum_k": ambient + maximum_rise,
        },
    }


def test_attribution_uses_temperature_rise_and_closes_exactly():
    reference = {
        "p0": metrics(10.0, 20.0),
        "p1": metrics(12.0, 24.0),
        "p2": metrics(15.0, 30.0),
    }
    candidate = {
        "p0": metrics(8.0, 17.0),
        "p1": metrics(11.0, 22.0),
        "p2": metrics(12.0, 25.0),
    }
    result = MODULE.compute_attribution(reference, candidate)
    contribution = result["incremental_contributions"]
    assert contribution["partition_and_placement"]["average_temperature_k"] == pytest.approx(2.0)
    assert contribution["opt_and_cts"]["average_temperature_k"] == pytest.approx(-1.0)
    assert contribution["routing_and_report"]["average_temperature_k"] == pytest.approx(2.0)
    assert contribution["partition_and_placement"][
        "average_temperature_absolute_share_percent"
    ] == pytest.approx(40.0)
    assert contribution["opt_and_cts"][
        "average_temperature_signed_share_percent"
    ] == pytest.approx(-100.0 / 3.0)
    assert result["final_average_temperature_improvement_k"] == pytest.approx(3.0)
    assert result["closure"]["average_k"] == pytest.approx(0.0)
    assert result["closure"]["maximum_k"] == pytest.approx(0.0)


def test_attribution_requires_matching_levels():
    with pytest.raises(ValueError, match="same non-empty levels"):
        MODULE.compute_attribution(
            {"p0": metrics(10.0, 20.0)},
            {"p1": metrics(10.0, 20.0)},
        )
