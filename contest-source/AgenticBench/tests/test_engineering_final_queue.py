"""End-to-end T4 final queue with a fake native executor (no docker, no OpenROAD).

Only the native boundary is faked: the CTS stage execution (reference_case)
writes the files the real stage would publish, and snapshot/Liberty/equivalence
readers return synthetic data. Sealing, the once-per-case ledger, the real
audit_stage, capability evidence, v2 gates, validate_native_result, receipts
and aggregation all run unmodified.
"""
import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engineering_bench" / "physical"))

import cts_final_queue as queue
from engineering_bench.input_audit import clock_table
from engineering_bench.metrics import clock_skew
from engineering_bench.protocol import sha
from engineering_bench.submission import inventory
from engineering_bench.tasks import load_task

TASK = load_task("cross_die_cts")
IMAGE = "sha256:" + "1" * 64
BINARY = "b" * 64
PREFIX = {"60_opt": "frozen-prefix"}
DIES = ("bottom", "top")
TIMING = {"setup_wns_ns": -0.05, "setup_tns_ns": -0.5, "hold_wns_ns": 0.01, "hold_tns_ns": 0.0}
CLOCK_FIELDS = ["clock", "die", "pin", "active_edge", "arrival_max_rise_ns", "arrival_max_fall_ns",
                "arrival_min_rise_ns", "arrival_min_fall_ns", "clock_domain_count"]
PORT_FIELDS = ["port", "direction", "signal_type", "layer", "xmin", "ymin", "xmax", "ymax", "status"]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))
    return path


def write_tsv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def cells(clock_buffers):
    """One fixed macro and one data cell per die, plus clock buffers."""
    result = {"macro": {"clock_cell": False, "master_type": "BLOCK", "status": "FIXED", "x_dbu": 0,
                        "y_dbu": 0, "orient": "R0", "master": "SRAM"},
              "ff1": {"clock_cell": False, "master_type": "CORE", "status": "PLACED", "x_dbu": 10,
                      "y_dbu": 10, "orient": "R0", "master": "DFF"}}
    for n in range(clock_buffers):
        result["ckbuf%d" % n] = {"clock_cell": True, "master_type": "CORE", "status": "PLACED",
                                 "x_dbu": 20 + n, "y_dbu": 20, "orient": "R0", "master": "CKBUF"}
    return result


def snapshot(folder, cell_rows, odb_sha=None):
    dump(folder / "snapshot.json", {"dispatch": {"image": IMAGE, "libraries": ["lib.lib"], "input_odb_sha256": odb_sha},
                                    "result": {"status": "succeeded"}, "cells": cell_rows})
    dump(folder / "result.json", {"status": "succeeded", "statistics": {"floating_pg": 0}})
    write_tsv(folder / "ports.tsv", PORT_FIELDS, [dict(zip(PORT_FIELDS, ["clk", "INPUT", "CLOCK", "M5", "0", "0", "1", "1", "FIXED"]))])


def write_stage(folder, *, base, role, arrivals, timing=TIMING, clock_buffers=4, measurement_status="succeeded",
                export_error=False, binary=BINARY):
    """What one native CTS stage publishes: replay, read-backs, legalization, measurement."""
    dump(folder / "replay/input-replay.json", {"prefix": PREFIX, "seed": TASK.data["seed"],
        "source_sha256": {"scripts/cts.tcl": sha(base / "scripts/cts.tcl")}, "native_binary_sha256": binary})
    for die in DIES:
        norm = folder / ("normalized-" + die)
        odb = norm / "native/design.odb"
        odb.parent.mkdir(parents=True, exist_ok=True)
        odb.write_bytes(("odb-%s-%s-%s" % (role, die, arrivals)).encode())
        dump(norm / "result.json", {"status": "succeeded", "artifacts": {"native/design.odb": sha(odb)}})
        (norm / "native.log").write_text("ENGINEERING_PROTECTED_LEGALIZATION restored=0 pg=12\n")
        snapshot(folder / ("readback-" + die), cells(clock_buffers), sha(odb))
    measurement = folder / "independent-skew/native/measurement"
    rows = [{"clock": "clk", "die": die, "pin": "u_%s/ff%d/CK" % (die, n), "active_edge": "rise",
             "arrival_max_rise_ns": value, "arrival_max_fall_ns": value, "arrival_min_rise_ns": value - 0.1,
             "arrival_min_fall_ns": value - 0.1, "clock_domain_count": "1"}
            for (die, n), value in zip([(d, n) for d in DIES for n in range(2)], arrivals)]
    table = measurement / "assembly/passive/clock_sinks_by_clock.tsv"
    write_tsv(table, CLOCK_FIELDS, rows)
    (measurement / "electrical.rpt").write_text("max slew\nmax capacitance\nmax fanout\n")
    for die in DIES:
        log = "[INFO DPL-0001] check_placement\n" + ("[ERROR DPL-0006] overlap\n" if export_error else "")
        (measurement / die).mkdir(parents=True, exist_ok=True)
        (measurement / die / "export.log").write_text(log + "ENGINEERING_MEASUREMENT_EXPORT pg=12\n")
    values = clock_table(table)
    raw = {"status": measurement_status, "endpoint": TASK.data["endpoint"], "timing": timing,
           "clock_skew_ns": clock_skew(values, {k: sorted(v) for k, v in values.items()}),
           "artifacts": {"electrical.rpt": sha(measurement / "electrical.rpt"),
                         "assembly/passive/clock_sinks_by_clock.tsv": sha(table)}}
    dump(measurement / "result.json", raw)
    dump(folder / "result.json", {"status": "stage_%s_measured" % role,
                                  "measurement_sha256": sha(measurement / "result.json")})


REFERENCE_ARRIVALS = (1.0, 1.2, 1.1, 1.4)       # skew 0.4 ns
SCENARIOS = {
    "success": dict(arrivals=(1.0, 1.1, 1.05, 1.2)),                         # skew 0.2 ns
    "gate_failure": dict(arrivals=(1.0, 1.1, 1.05, 1.2), timing=dict(TIMING, setup_wns_ns=-0.5)),
    "measurement_failure": dict(arrivals=(1.0, 1.1, 1.05, 1.2), measurement_status="failed"),
    "illegal_placement": dict(arrivals=(1.0, 1.1, 1.05, 1.2), export_error=True),
    "over_budget": dict(arrivals=(1.0, 1.1, 1.05, 1.2), clock_buffers=6),
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    root, experiment = tmp_path / "root", tmp_path / "experiment"
    base = tmp_path / "base"
    (base / "scripts").mkdir(parents=True)
    (base / "scripts/cts.tcl").write_text("clock_tree_synthesis\n")
    experiment.mkdir()
    subject = tmp_path / "subject"
    (subject / "open3dbench/backend").mkdir(parents=True)
    (subject / "open3dbench/backend/three_d_cts.py").write_text("def run_cts():\n    pass\n")
    dump(tmp_path / "baseline-inventory.json", inventory(subject))
    (subject / "open3dbench/backend/three_d_cts.py").write_text("def run_cts():\n    return 'improved'\n")
    for case in TASK.data["cases"]:
        run = root / "reference_root" / case / "run"
        (run / "60_opt").mkdir(parents=True)
        (run / "60_opt/hbt.csv").write_text("hbt\n")
        for die in DIES:
            snapshot(experiment / "native-audit-v3" / case / "60_opt" / die, cells(4))
        write_stage(experiment / "stage-reference-v1" / case, base=base, role="reference", arrivals=REFERENCE_ARRIVALS)
    plan = {}
    dispatched = []

    def fake_reference_case(case, *, output, role, source, **kwargs):
        assert role == "candidate" and source == subject
        dispatched.append(case)
        write_stage(output, base=base, role="candidate", **SCENARIOS[plan[case]])

    def fake_snapshot(folder):
        data = json.loads((Path(folder) / "snapshot.json").read_text())
        return data["dispatch"], data["result"], data["cells"]

    monkeypatch.setattr(queue, "layout_path", lambda root, name: root / name)
    monkeypatch.setattr(queue, "verify_release", lambda path: (base, IMAGE))
    monkeypatch.setattr(queue, "admission", lambda output: True)
    monkeypatch.setattr(queue, "verify_prefix", lambda run: PREFIX)
    monkeypatch.setattr(queue, "reference_case", fake_reference_case)
    monkeypatch.setattr(queue, "load_snapshot", fake_snapshot)
    monkeypatch.setattr(queue, "areas", lambda platform: {"CKBUF": 2.0, "DFF": 5.0, "SRAM": 100.0})
    monkeypatch.setattr(queue, "compare", lambda original, observed, hbt: {"proof": "synthetic", "passed": True})

    def run(scenarios, version=2, name="out"):
        plan.update(scenarios)
        output = tmp_path / name
        argv = ["--root", str(root), "--experiment", str(experiment), "--output", str(output),
                "--subject", str(subject), "--baseline-inventory", str(tmp_path / "baseline-inventory.json"),
                "--cases"] + list(scenarios) + ["--scoring-version", str(version)]
        queue.main(argv)
        return output

    return {"run": run, "subject": subject, "dispatched": dispatched, "tmp": tmp_path}


def read(path):
    return json.loads(Path(path).read_text())


def test_v2_queue_success_and_failure_paths(world):
    output = world["run"]({"bp_fe": "success", "bp": "gate_failure", "ariane133": "measurement_failure",
                           "swerv_wrapper": "illegal_placement"})
    assert world["dispatched"] == ["bp_fe", "bp", "ariane133", "swerv_wrapper"]
    ledger = {case: read(output / "ledger" / case / "result.json") for case in TASK.data["cases"]}
    # Success: legal, scored, with a receipt that validates.
    ok = ledger["bp_fe"]
    assert ok["status"] == "succeeded" and ok["legal"] is True and ok["scoring_version"] == 2
    assert ok["candidate_metric"] == pytest.approx(0.2) and ok["reference_metric"] == pytest.approx(0.4)
    details = ok["details"]
    assert details["skew_distribution"]["p95_sink_skew_ns"] == pytest.approx(0.2)
    assert details["clock_power"]["status"] == "not_measured" and details["clock_power"]["value_mw"] is None
    assert details["timing_gate"]["passed"] and details["insertion_delay_gate"]["passed"]
    native = read(output / "bp_fe/final-native.json")
    assert native["scoring_version"] == 2 and native["kind"] == "engineering-native-result-v1"
    assert set(native["checks"]) == set(TASK.data["required_capabilities"])
    for record in native["checks"].values():
        assert (output / "bp_fe" / record["evidence"]["path"]).is_file() and record["sources"]
    assert native["checks"]["placement_legality"]["status"] == "passed"
    assert native["checks"]["placement_legality"]["evidence"]["path"] == "capabilities/placement_legality.json"
    assert all(v["status"] == "passed" for v in native["checks"].values())
    assert read(output / "bp_fe/final-measurements.json")["clock_resources"] == {"clock_area_um2": 16.0, "clock_buffer_count": 8}
    assert ok["artifacts"] == [str(output / "bp_fe/final-native.json")]
    # Gate failure: legal placement but timing regressed beyond tolerance.
    assert ledger["bp"]["status"] == "failed" and ledger["bp"]["candidate_metric"] is None
    assert "T4 v2 hard gate failed: timing:setup_wns_ns" in ledger["bp"]["error"]
    assert read(output / "bp/final-checks.json")["gates_v2"]["passed"] is False
    assert not (output / "bp/final-native.json").exists()
    # Measurement failure: no measurement, nothing scored.
    assert "unsuccessful independent endpoint" in ledger["ariane133"]["error"]
    assert not (output / "ariane133/final-checks.json").exists()
    # Illegal placement: the measurement export re-check found an ERROR.
    assert "Failed physical capability: placement_legality" in ledger["swerv_wrapper"]["error"]
    record = read(output / "swerv_wrapper/capabilities/placement_legality.json")
    assert record["status"] == "failed" and record["measured"]["measurement_export"] == {"bottom": False, "top": False}
    # Aggregate: failures count as zero, legality reported separately.
    result = read(output / "result.json")
    assert result["failed_cases"] == ["bp", "ariane133", "swerv_wrapper"]
    assert result["legality_rate"] == pytest.approx(0.25) and result["legal_cases"] == ["bp_fe"]
    assert result["headline_noop_relative_score_percent_failed_as_zero"] == pytest.approx(50.0 / 4)
    assert result["legal_case_mean_noop_relative_score_percent"] == pytest.approx(50.0)
    assert result["headline_normalized_status"] == "expert_unregistered"
    assert result["all_case_mean_improvement_percent"] is None
    # Each qualification is v2 with identity v2 and per-capability evidence.
    q = read(output / "qualifications/bp_fe/qualification.json")
    assert q["scoring_version"] == 2 and q["identity_version"] == 2
    assert read(output / "queue.json")["model_calls"] == 0


def test_resource_budget_failure(world):
    output = world["run"]({"bp_fe": "over_budget"})
    row = read(output / "ledger/bp_fe/result.json")
    # 12 clock buffers against 8 exceeds the v2 ratio 1.25 (v1 allowed 2.0).
    assert row["status"] == "failed" and "clock_area, clock_buffer_count" in row["error"]


def test_v1_queue_keeps_frozen_semantics(world):
    output = world["run"]({"bp_fe": "gate_failure", "bp": "over_budget"}, version=1)
    ledger = {case: read(output / "ledger" / case / "result.json") for case in ("bp_fe", "bp")}
    # Timing is diagnostic and the 2.0 resource ratio applies in v1.
    assert ledger["bp_fe"]["status"] == "succeeded" and "scoring_version" not in ledger["bp_fe"]
    assert ledger["bp"]["status"] == "succeeded"
    assert "scoring_version" not in read(output / "bp_fe/final-native.json")
    result = read(output / "result.json")
    assert "calibration" not in result and result["missing_cases"] == ["ariane133", "swerv_wrapper"]


def test_out_of_scope_submission_is_rejected_before_dispatch(world):
    (world["subject"] / "evaluator_override.py").write_text("SCORE = 0\n")
    with pytest.raises(ValueError, match="outside task scope"):
        world["run"]({"bp_fe": "success"})
    assert world["dispatched"] == []
    assert not (world["tmp"] / "out/ledger").exists()


def test_submission_changed_after_sealing_is_not_scored(world, monkeypatch):
    original = queue.reference_case
    def tamper(case, **kwargs):
        original(case, **kwargs)
        (world["subject"] / "open3dbench/backend/three_d_cts.py").write_text("tampered\n")
    monkeypatch.setattr(queue, "reference_case", tamper)
    output = world["run"]({"bp_fe": "success"})
    row = read(output / "ledger/bp_fe/result.json")
    assert row["status"] == "failed" and "changed after sealing" in row["error"]


def test_each_case_runs_once(world):
    output = world["run"]({"bp_fe": "success"})
    from engineering_bench.protocol import OnceLedger
    ledger = OnceLedger(output / "ledger", TASK, read(output / "sealed.json")["submission_sha256"])
    with pytest.raises(FileExistsError):
        ledger.claim("bp_fe", output / "qualifications/bp_fe/qualification.json")
