"""The parameter task consumes a GRT result, not fabricated DRT evidence."""
import copy
import json
from pathlib import Path

import pytest

from local_bench.metrics import extract_metrics
from local_bench.model import load_task


TASK = Path(__file__).resolve().parents[1] / "local_tasks/global_routing_parameters"


def grt_metrics():
    return {
        "stage": "global_route", "status": "success", "unexpected_warnings": 0,
        "ready_for_detailed_route": True, "ready_for_extract": False,
        "dies": {die: {"status": "success", "grt_total_overflow": 10,
                       "grt_congestion_bins": 2, "grt_wirelength_um": 100.0}
                 for die in ("bottom", "top")},
    }


def write_result(run, values, directory="75_global_route"):
    stage = run / directory
    stage.mkdir(parents=True)
    (stage / "metrics.json").write_text(json.dumps(values))
    return run


def test_task_and_initial_reference_share_grt_boundary(tmp_path):
    # Retained metric reader only: the parameter task is no longer executable.
    with pytest.raises(FileNotFoundError):
        load_task(TASK)
    reference = write_result(tmp_path / "reference", grt_metrics())
    candidate = write_result(tmp_path / "candidate", grt_metrics())
    result = extract_metrics("routing_parameters", candidate, reference)
    assert result["routing_complete"] == result["combined_score"] == 1.0
    assert result["grt_total_overflow"] == 20
    assert result["grt_wirelength_um"] == 200
    assert result["evaluation_boundary"] == "global-route-only"
    assert result["drc_violations"] is result["antenna_violations"] is None
    assert result["drc_score"] == 1.0
    assert result["drc_evaluation"] == "not-applicable-neutral-score"
    assert "drc_clean" not in result


def test_grt_keeps_existing_weights_and_does_not_score_wirelength(tmp_path):
    reference = write_result(tmp_path / "reference", grt_metrics())
    raw = grt_metrics()
    for row in raw["dies"].values():
        row["grt_total_overflow"] = 5
        row["grt_wirelength_um"] = 999999
    result = extract_metrics("routing_parameters", write_result(tmp_path / "candidate", raw), reference)
    assert result["combined_score"] == 0.75 * 2 + 0.25 * 1


@pytest.mark.parametrize("field,value", [
    ("ready_for_detailed_route", False), ("ready_for_extract", True),
    ("status", "failed"), ("stage", "route"), ("unexpected_warnings", 1),
])
def test_incomplete_or_wrong_grt_boundary_rejected(tmp_path, field, value):
    raw = grt_metrics()
    raw[field] = value
    with pytest.raises(ValueError):
        extract_metrics("routing_parameters", write_result(tmp_path, raw), None)


@pytest.mark.parametrize("field,value", [
    ("grt_wirelength_um", None), ("grt_wirelength_um", float("nan")),
    ("grt_total_overflow", -1), ("grt_total_overflow", True),
    ("grt_congestion_bins", float("inf")), ("status", "failed"),
])
def test_both_die_native_metrics_required(tmp_path, field, value):
    raw = grt_metrics()
    raw["dies"]["top"][field] = value
    with pytest.raises(ValueError):
        extract_metrics("routing_parameters", write_result(tmp_path, raw), None)


def test_missing_die_or_reference_never_becomes_zero(tmp_path):
    raw = grt_metrics()
    del raw["dies"]["top"]
    with pytest.raises(ValueError):
        extract_metrics("routing_parameters", write_result(tmp_path / "bad", raw), None)
    candidate = write_result(tmp_path / "good", grt_metrics())
    with pytest.raises(FileNotFoundError):
        extract_metrics("routing_parameters", candidate, tmp_path / "absent-reference")


def test_reference_must_be_successful_and_same_boundary(tmp_path):
    candidate = write_result(tmp_path / "candidate", grt_metrics())
    drt = copy.deepcopy(grt_metrics())
    drt.update(stage="route", ready_for_extract=True, drc_violations=0, antenna_violations=0)
    reference = write_result(tmp_path / "drt", drt, "80_route")
    with pytest.raises(ValueError, match="boundary mismatch"):
        extract_metrics("routing_parameters", candidate, reference)
    with pytest.raises(ValueError, match="boundary mismatch"):
        extract_metrics("routing_parameters", reference, candidate)
    bad = grt_metrics()
    bad["dies"]["top"]["status"] = "failed"
    with pytest.raises(ValueError):
        extract_metrics("routing_parameters", candidate, write_result(tmp_path / "bad-ref", bad))


def test_legacy_drt_still_uses_measured_antenna_and_drc(tmp_path):
    raw = grt_metrics()
    raw.update(stage="route", ready_for_extract=True, drc_violations=3, antenna_violations=0)
    candidate = write_result(tmp_path / "candidate", raw, "80_route")
    result = extract_metrics("routing_parameters", candidate, None)
    assert result["routing_complete"] == 1
    assert result["drc_score"] == 0.25
    assert result["drc_violations"] == 3
    assert result["antenna_violations"] == 0
    raw["antenna_violations"] = 1
    failed = write_result(tmp_path / "antenna-failed", raw, "80_route")
    assert extract_metrics("routing_parameters", failed, None)["combined_score"] == 0


def test_pure_grt_values_do_not_claim_drt_from_stray_fields(tmp_path):
    raw = grt_metrics()
    raw.update(drc_violations=0, antenna_violations=0, drc_clean=True)
    result = extract_metrics("routing_parameters", write_result(tmp_path, raw), None)
    assert result["drc_violations"] is result["antenna_violations"] is None
    assert "drc_clean" not in result
