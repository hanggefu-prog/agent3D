"""Scoring v2: calibration, T4 hard gates, T3 receiver scopes, capability evidence.

Synthetic contract tests; they are not native qualification receipts.
"""
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engineering_bench" / "physical"))

from engineering_bench import capabilities
from engineering_bench.capabilities import Recorder, status_map
from engineering_bench.evaluation import clock_power_gate, t4_gates
from engineering_bench.metrics import (aggregate, bounded_ratio, clock_distribution, clock_skew,
    receiver_subset, target_delay_report, wns_tns_guard)
from engineering_bench.scoring import (load_profile, profile_for, qualification_fields, scoring_version,
    validate_profile)
from engineering_bench.tasks import TASK_IDS, TASK_ROOT, load_task


# --- profiles ---------------------------------------------------------------------

@pytest.mark.parametrize("task_id", TASK_IDS)
def test_packaged_profiles_are_valid_and_outside_the_task_digest(task_id):
    task = load_task(task_id)
    profile, digest = load_profile(task)
    assert profile["judge"]["weight_in_headline"] == 0
    assert qualification_fields(task) == {}
    assert qualification_fields(task, 2) == {"scoring_version": 2, "scoring_profile_sha256": digest}
    assert qualification_fields(task, 1) == {}
    assert profile_for(task, {}) is None
    assert profile_for(task, {"scoring_version": 2, "scoring_profile_sha256": digest}) == profile
    with pytest.raises(ValueError, match="changed"):
        profile_for(task, {"scoring_version": 2, "scoring_profile_sha256": "0" * 64})


def test_profile_rejects_judge_weight_and_bad_versions():
    task = load_task("cross_die_cts")
    profile, _ = load_profile(task)
    for change in ({"judge": dict(profile["judge"], weight_in_headline=0.1)},
                   {"judge": dict(profile["judge"], min_samples=1)},
                   {"scoring_version": 1},
                   {"calibration": dict(profile["calibration"], failed_case_score=None)}):
        with pytest.raises(ValueError):
            validate_profile(task, dict(profile, **change))
    assert scoring_version({}) == 1 and scoring_version({"scoring_version": 2}) == 2
    with pytest.raises(ValueError):
        scoring_version({"scoring_version": 3})


# --- T2 calibration -----------------------------------------------------------------

def calibrated_task(tmp_path, experts):
    shutil.copytree(TASK_ROOT / "clock_compensation", tmp_path / "clock_compensation")
    path = tmp_path / "clock_compensation/expert-reference.json"
    registry = json.loads(path.read_text())
    registry["cases"] = {case: (None if qor is None else {"qor": qor, "source": "expert run",
                                                          "receipt_sha256": "a" * 64})
                         for case, qor in experts.items()}
    path.write_text(json.dumps(registry))
    task = load_task("clock_compensation", tmp_path)
    return task, load_profile(task)[1]


def rows(task, profile_sha, outcomes):
    result = []
    for case, outcome in zip(task.data["cases"], outcomes):
        row = {"case": case, "reference_metric": 2.0, "scoring_version": 2, "scoring_profile_sha256": profile_sha}
        if outcome is None:
            row.update(status="failed", legal=False)
        elif outcome == "running":
            row.update(status="dispatched")
        else:
            row.update(status="succeeded", legal=True, candidate_metric=outcome)
        result.append(row)
    return result


def test_calibration_with_expert_and_failed_case_as_zero(tmp_path):
    task, digest = calibrated_task(tmp_path, {"bp_fe": 1.0, "bp": 1.0, "ariane133": 1.0, "swerv_wrapper": 1.5})
    result = aggregate(task, rows(task, digest, [1.0, 1.5, None, 1.0]))
    # The frozen v1 fields are unchanged.
    assert result["failed_cases"] == ["ariane133"] and result["all_case_mean_improvement_percent"] is None
    assert result["successful_subset_mean_improvement_percent"] == pytest.approx((50 + 25 + 50) / 3)
    calibration = result["calibration"]
    assert calibration["bp_fe"]["normalized_score"] == pytest.approx(100.0)
    assert calibration["bp"]["normalized_score"] == pytest.approx(50.0)
    assert calibration["swerv_wrapper"]["normalized_score"] == pytest.approx(200.0)
    assert calibration["ariane133"]["status"] == "failed" and calibration["ariane133"]["headline_normalized_score"] == 0.0
    assert result["legality_rate"] == pytest.approx(0.75) and result["legal_cases"] == ["bp", "bp_fe", "swerv_wrapper"]
    assert result["legal_case_mean_noop_relative_score_percent"] == pytest.approx((50 + 25 + 50) / 3)
    assert result["headline_noop_relative_score_percent_failed_as_zero"] == pytest.approx((50 + 25 + 0 + 50) / 4)
    assert result["headline_normalized_score_failed_as_zero"] == pytest.approx((100 + 50 + 0 + 200) / 4)
    assert result["headline_normalized_status"] == "available" and result["judge_weight_in_headline"] == 0


def test_calibration_without_expert_and_with_running_case(tmp_path):
    task, digest = calibrated_task(tmp_path, {"bp_fe": None, "bp": 1.0, "ariane133": 1.0, "swerv_wrapper": 1.0})
    result = aggregate(task, rows(task, digest, [1.0, 1.0, 1.0, 1.0]))
    assert result["calibration"]["bp_fe"]["normalized_score"] is None
    assert result["headline_normalized_score_failed_as_zero"] is None
    assert result["headline_normalized_status"] == "expert_unregistered"
    assert result["headline_noop_relative_score_percent_failed_as_zero"] == pytest.approx(50.0)
    running = aggregate(task, rows(task, digest, [1.0, 1.0, 1.0, "running"]))
    assert running["headline_noop_relative_score_percent_failed_as_zero"] is None
    assert running["headline_normalized_status"] == "incomplete_cases"
    mixed = rows(task, digest, [1.0, 1.0, 1.0, 1.0])
    del mixed[0]["scoring_version"]
    with pytest.raises(ValueError, match="Mixed"):
        aggregate(task, mixed)
    stale = rows(task, "0" * 64, [1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="profile changed"):
        aggregate(task, stale)


def test_packaged_experts_are_unregistered():
    task = load_task("clock_compensation")
    digest = load_profile(task)[1]
    result = aggregate(task, rows(task, digest, [1.0, 1.0, 1.0, 1.0]))
    assert all(v["qor_expert"] is None for v in result["calibration"].values())


# --- T4 distribution and hard gates ---------------------------------------------------

INVENTORY = {"clk/rise/max/ss": ["a", "b", "c", "d"], "clk/rise/min/ff": ["a", "b"]}


def arrivals(scale=1.0, shift=0.0):
    return {"clk/rise/max/ss": {"a": 1.0 * scale, "b": 1.1 * scale, "c": 1.2 * scale, "d": 1.5 * scale + shift},
            "clk/rise/min/ff": {"a": 0.5, "b": 0.52}}


def test_clock_distribution_mean_and_p95():
    dist = clock_distribution(arrivals(), INVENTORY)
    assert dist["max_skew_ns"] == pytest.approx(clock_skew(arrivals(), INVENTORY)) == pytest.approx(0.5)
    assert dist["mean_sink_skew_ns"] == pytest.approx((0 + 0.1 + 0.2 + 0.5 + 0 + 0.02) / 6)
    assert dist["p95_sink_skew_ns"] == pytest.approx(0.5)
    assert dist["mean_group_skew_ns"] == pytest.approx((0.5 + 0.02) / 2)
    # Insertion delay uses setup (/max/) groups only when present.
    assert dist["max_insertion_delay_ns"] == pytest.approx(1.5)
    assert dist["sinks"] == 6 and dist["groups"] == 2


def test_wns_tns_and_ratio_bounds():
    ref = {"setup_wns_ns": -0.1, "setup_tns_ns": -1.0, "hold_wns_ns": 0.0, "hold_tns_ns": 0.0}
    tol = {"setup_wns_ns": 0.01, "setup_tns_ns": 0.1, "hold_wns_ns": 0.01, "hold_tns_ns": 0.1}
    assert wns_tns_guard(ref, dict(ref, setup_wns_ns=-0.105), tol)["passed"]
    failed = wns_tns_guard(ref, dict(ref, setup_tns_ns=-1.2, hold_wns_ns=-0.02), tol)
    assert failed["failures"] == ["setup_tns_ns", "hold_wns_ns"]
    assert wns_tns_guard(ref, dict(ref, setup_tns_ns=-1.2), tol, relative_tolerance=0.2)["passed"]
    with pytest.raises(ValueError):
        wns_tns_guard(ref, dict(ref, setup_wns_ns=None), tol)
    assert bounded_ratio(1.25, 1.0, 1.25, "x")["passed"] and not bounded_ratio(1.26, 1.0, 1.25, "x")["passed"]


def t4_reference():
    timing = {"setup_wns_ns": -0.1, "setup_tns_ns": -1.0, "hold_wns_ns": 0.0, "hold_tns_ns": 0.0}
    return {"expected_inventory": INVENTORY, "measurements": arrivals(), "timing": timing,
            "clock_resources": {"clock_area_um2": 100.0, "clock_buffer_count": 20},
            "clock_power": {"status": "not_measured", "value_mw": None}}


def t4_measurement(**changes):
    value = {"values": arrivals(shift=-0.3), "timing": dict(t4_reference()["timing"]),
             "clock_resources": {"clock_area_um2": 110.0, "clock_buffer_count": 22},
             "clock_power": {"status": "not_measured", "value_mw": None}}
    value.update(changes)
    return value


def test_t4_gates_pass_and_report_power_not_measured():
    gates = load_profile(load_task("cross_die_cts"))[0]["gates"]
    result = t4_gates(t4_reference(), t4_measurement(), gates)
    assert result["skew_distribution"]["max_skew_ns"] == pytest.approx(0.2)
    assert result["reference_skew_distribution"]["p95_sink_skew_ns"] == pytest.approx(0.5)
    assert result["clock_power"] == {"status": "not_measured", "value_mw": None, "passed": None,
        "reason": "the independent native measurement produces no clock power report"}


@pytest.mark.parametrize("change,failure", [
    ({"timing": {"setup_wns_ns": -0.2, "setup_tns_ns": -1.0, "hold_wns_ns": 0.0, "hold_tns_ns": 0.0}}, "timing:setup_wns_ns"),
    ({"values": arrivals(scale=1.6, shift=-0.9)}, "insertion_delay"),
    ({"clock_resources": {"clock_area_um2": 126.0, "clock_buffer_count": 20}}, "clock_area"),
    ({"clock_resources": {"clock_area_um2": 100.0, "clock_buffer_count": 26}}, "clock_buffer_count")])
def test_t4_gate_failures(change, failure):
    gates = load_profile(load_task("cross_die_cts"))[0]["gates"]
    with pytest.raises(ValueError, match="T4 v2 hard gate failed: .*" + failure):
        t4_gates(t4_reference(), t4_measurement(**change), gates)


def test_clock_power_bound_only_when_measured():
    measured = lambda mw: {"status": "measured", "value_mw": mw}
    assert clock_power_gate(measured(1.0), measured(1.05), 1.1)["passed"] is True
    assert clock_power_gate(measured(1.0), measured(1.2), 1.1)["passed"] is False
    assert clock_power_gate(None, measured(1.2), 1.1)["status"] == "not_measured"
    gates = load_profile(load_task("cross_die_cts"))[0]["gates"]
    reference = dict(t4_reference(), clock_power=measured(1.0))
    with pytest.raises(ValueError, match="clock_power"):
        t4_gates(reference, t4_measurement(clock_power=measured(2.0)), gates)


# --- T3 receivers and area ---------------------------------------------------------------

def t3_context(local=True):
    targets = {"n1": {"source_die": "bottom", "original_remote_receivers": ["u_top/r1/A", "u_top/r2/A"]},
               "n2": {"source_die": "top", "original_remote_receivers": ["u_bottom/r3/A"]}}
    if local:
        targets["n1"]["original_local_receivers"] = ["u_bottom/l1/A"]
        targets["n2"]["original_local_receivers"] = []
    cells = [{"name": "BUF_X1", "die": "bottom", "area_um2": 1.0}, {"name": "BUF_X4", "die": "bottom", "area_um2": 3.0},
             {"name": "BUF_X2", "die": "top", "area_um2": 2.0}]
    return {"target_nets": targets, "buffer_cells": cells}


def test_t3_inventories_and_area_budget():
    from multi_final_queue import buffer_area_budget, inventories
    scopes = inventories(t3_context(), 2)
    assert scopes["all"] == {"n1": ["u_bottom/l1/A", "u_top/r1/A", "u_top/r2/A"], "n2": ["u_bottom/r3/A"]}
    assert scopes["local"] == {"n1": ["u_bottom/l1/A"], "n2": []}
    v1 = inventories(t3_context(local=False), 1)
    assert v1["all"] == v1["remote"] and v1["local"] == {"n1": [], "n2": []}
    with pytest.raises(ValueError, match="original_local_receivers"):
        inventories(t3_context(local=False), 2)
    overlap = t3_context()
    overlap["target_nets"]["n1"]["original_local_receivers"] = ["u_top/r1/A"]
    with pytest.raises(ValueError, match="both local and remote"):
        inventories(overlap, 2)
    assert buffer_area_budget(t3_context(), 4.0) == pytest.approx(4.0 * (1.0 + 2.0))


def test_t3_report_covers_all_receivers_and_subsets():
    from multi_final_queue import inventories
    scopes = inventories(t3_context(), 2)
    delays = {"n1": {"u_bottom/l1/A": 0.3, "u_top/r1/A": 0.1, "u_top/r2/A": 0.2}, "n2": {"u_bottom/r3/A": 0.4}}
    report = target_delay_report(delays, scopes)
    assert report["mean_target_net_delay_ns"] == pytest.approx((0.3 + 0.4) / 2)
    assert report["remote_receivers_mean_target_net_delay_ns"] == pytest.approx((0.2 + 0.4) / 2)
    assert report["local_receivers_mean_target_net_delay_ns"] == pytest.approx(0.3)
    assert report["receiver_counts"] == {"all": 4, "remote": 3, "local": 1}
    assert receiver_subset(delays, scopes["remote"])["n1"] == {"u_top/r1/A": 0.1, "u_top/r2/A": 0.2}
    with pytest.raises(ValueError):
        target_delay_report({"n1": delays["n1"], "n2": {}}, scopes)


def test_delay_query_includes_local_receivers(tmp_path):
    from target_delay import measured_receivers, parse_delays
    context = t3_context()
    assert measured_receivers(context["target_nets"]["n1"]) == ["u_bottom/l1/A", "u_top/r1/A", "u_top/r2/A"]
    assert measured_receivers(t3_context(local=False)["target_nets"]["n1"]) == ["u_top/r1/A", "u_top/r2/A"]
    header = "target\tsink\trise_delay_ns\tfall_delay_ns\tworst_delay_ns\n"
    body = ["n1\tu_top/r1/A\t0.1\t0.05\t0.1", "n1\tu_top/r2/A\t0.2\t0.1\t0.2", "n2\tu_bottom/r3/A\t0.4\t0.3\t0.4"]
    path = tmp_path / "d.tsv"
    path.write_text(header + "\n".join(body) + "\n")
    with pytest.raises(ValueError, match="Missing"):
        parse_delays(path, context)
    assert parse_delays(path, t3_context(local=False))["n1"] == {"u_top/r1/A": 0.1, "u_top/r2/A": 0.2}
    path.write_text(header + "\n".join(body + ["n1\tu_bottom/l1/A\t0.3\t0.2\t0.3"]) + "\n")
    assert parse_delays(path, context)["n1"]["u_bottom/l1/A"] == 0.3


# --- per-capability evidence -------------------------------------------------------------

def test_capability_records_have_own_evidence_and_unverified_never_passes(tmp_path):
    source = tmp_path / "result.json"
    source.write_text("{}")
    recorder = Recorder(tmp_path)
    recorder.add("placement_legality", check="check_placement clean", measured={"errors": 0},
                 threshold={"errors": 0}, passed=True, sources=[source])
    with pytest.raises(ValueError, match="source artifact"):
        recorder.add("pg_connectivity", check="x", measured=1, threshold=0, passed=False)
    with pytest.raises(ValueError, match="Repeated"):
        recorder.add("placement_legality", check="x", measured=1, threshold=0, passed=True, sources=[source])
    records = recorder.complete(["placement_legality", "electrical_limits"])
    passed, unverified = records["placement_legality"], records["electrical_limits"]
    assert passed["status"] == "passed" and passed["sources"] == [{"path": "result.json", "sha256": passed["sources"][0]["sha256"]}]
    assert passed["evidence"]["path"] == "capabilities/placement_legality.json"
    assert unverified["status"] == "unverified" and unverified["passed"] is False
    assert status_map(records) == {"placement_legality": True, "electrical_limits": False}
    names = ["placement_legality", "electrical_limits"]
    with pytest.raises(ValueError, match="Unverified capability is not a pass"):
        capabilities.verify(tmp_path, records, names)
    summary = capabilities.verify(tmp_path, records, names, allow_unverified=["electrical_limits"])
    assert summary["passed"] == ["placement_legality"] and summary["unverified"] == ["electrical_limits"]
    # A forged pass is caught against its evidence file.
    forged = dict(records, electrical_limits=dict(unverified, status="passed", passed=True))
    with pytest.raises(ValueError, match="differs from its evidence"):
        capabilities.verify(tmp_path, forged, names)
    # A changed source artifact invalidates the record.
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError):
        capabilities.verify(tmp_path, records, names, allow_unverified=["electrical_limits"])


def test_failed_capability_and_legacy_records(tmp_path):
    source = tmp_path / "log.txt"
    source.write_text("ERROR")
    recorder = Recorder(tmp_path / "q")
    recorder.add("placement_legality", check="no ERROR lines", measured={"errors": 1}, threshold={"errors": 0},
                 passed=False, sources=[source])
    assert recorder.failures() == ["placement_legality"]
    record = recorder.records["placement_legality"]
    assert Path(record["sources"][0]["path"]).is_absolute()
    with pytest.raises(ValueError, match="Failed physical capability"):
        capabilities.verify(tmp_path / "q", recorder.records, ["placement_legality"])
    evidence = tmp_path / "q" / "evidence.json"
    evidence.write_text("{}")
    from engineering_bench.protocol import sha
    legacy = {"x": {"passed": True, "evidence": {"path": "evidence.json", "sha256": sha(evidence)}}}
    with pytest.raises(ValueError, match="schema 2"):
        capabilities.verify(tmp_path / "q", legacy, ["x"])
    assert capabilities.verify(tmp_path / "q", legacy, ["x"], require_schema=1)["legacy_shared_v1"] == ["x"]
