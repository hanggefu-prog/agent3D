"""Offline scoring/consumer tests, not native thermal qualification."""
import json
from types import SimpleNamespace

import pytest
import yaml

from global_bench.model import load_global_task
from global_bench.verifiers import open3dbench_codesign_verifier as verifier
from scripts import workspace_latest_reference as authority
from scripts import workspace_cts_thermal as snapshot
from scripts.workspace_task_selection import LATEST_THERMAL_TASKS, LATEST_THERMAL_PROFILE


def task():
    return load_global_task(authority.FRAMEWORK / "global_tasks" / LATEST_THERMAL_TASKS[10])


def row(wns=-1.0, tns=-100.0, temperature=320.0):
    return {"wns_setup_ns": wns, "tns_setup_ns": tns,
            "wns_deficit": max(0.0, -wns), "tns_deficit": max(0.0, -tns),
            "global_average_temperature_k": temperature, "stack": {"version": "fixture"},
            "cts_algorithm": "fixture", "boundary_budget_model": "continuous-passive-hbt",
            "power_model": "continuous-passive-hbt-placement-estimated",
            "timing_evaluation_boundary": "post_cts_one_opt",
            "cut_net_count": 12, "placement_total_hpwl_um": 100.0}


@pytest.mark.parametrize("cost, penalty", [(0.8, 0), (1, 0), (1.049, 0), (1.05, 0),
                                          (1.050001, 1e-7), (1.10, 0.005), (1.5, 0.045)])
def test_threshold_and_continuous_penalty(cost, penalty):
    result = verifier._timing_constrained_cts_thermal_score_rows(row(-cost, -100 * cost, 319), row())
    assert result["normalized_sta_cost"] == pytest.approx(cost)
    assert result["timing_penalty_weight"] == 0.1
    assert result["timing_penalty"] == pytest.approx(penalty)
    assert "timing_penalty_factor" not in result
    assert result["combined_score"] == pytest.approx((320 / 319) - penalty)
    assert result["unpenalized_thermal_score"] == pytest.approx(320 / 319)
    assert result["timing_penalty_applied"] == (cost > 1.05)
    assert result["objective_kind"] == "hotspot_average_temperature"


def test_weighted_cost_not_individual_metric_threshold():
    result = verifier._timing_constrained_cts_thermal_score_rows(row(-1.1, -80), row())
    assert result["normalized_sta_cost"] == pytest.approx(0.98)
    assert result["combined_score"] == 1
    assert not result["timing_penalty_applied"]


def test_thermal_gain_can_be_outweighed_by_timing():
    result = verifier._timing_constrained_cts_thermal_score_rows(row(-1.1, -110, 319), row())
    assert result["unpenalized_thermal_score"] > 1 > result["combined_score"]
    assert result["reference_wns_setup_ns"] == -1
    assert result["reference_tns_setup_ns"] == -100


def test_additive_deduction_is_independent_of_temperature_reward():
    for temperature in (319, 330):
        result = verifier._timing_constrained_cts_thermal_score_rows(row(-1.1, -110, temperature), row())
        assert result["unpenalized_thermal_score"] - result["combined_score"] == pytest.approx(0.005)


def test_negative_score_is_not_clamped_or_rejected():
    from scripts.run_workspace_codesign import _score
    result = verifier._timing_constrained_cts_thermal_score_rows(row(-20, -2000), row())
    assert result["combined_score"] == pytest.approx(-0.895)
    assert _score({"success": True, "metrics": result}) == result["combined_score"]


def test_zero_deficit_rule_and_reference_self_score():
    reference = row(0, 0)
    assert verifier._timing_constrained_cts_thermal_score_rows(reference, reference)["combined_score"] == 1
    assert verifier._timing_constrained_cts_thermal_score_rows(row(1, 0), reference)["combined_score"] == 1
    assert verifier._timing_constrained_cts_thermal_score_rows(row(-1, -100), reference)["combined_score"] < 0
    metrics = verifier.reference_self_metrics("cts_thermal", row(), LATEST_THERMAL_PROFILE)
    assert metrics["combined_score"] == 1 and metrics["timing_penalty"] == 0


@pytest.mark.parametrize("key", ["wns_setup_ns", "tns_setup_ns", "wns_deficit", "tns_deficit",
                                 "global_average_temperature_k"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True])
@pytest.mark.parametrize("side", [0, 1])
def test_missing_or_nonfinite_is_not_free_timing(key, bad, side):
    rows = [row(), row()]
    rows[side][key] = bad
    with pytest.raises(ValueError):
        verifier._timing_constrained_cts_thermal_score_rows(*rows)


@pytest.mark.parametrize("edit", [{"wns_deficit": 0}, {"timing_evaluation_boundary": "pre_cts_one_opt"},
    {"global_average_temperature_k": 0}, {"boundary_budget_model": "other"}])
def test_mismatched_evidence_rejected(edit):
    with pytest.raises(ValueError):
        verifier._timing_constrained_cts_thermal_score_rows({**row(), **edit}, row())


def test_legacy_thermal_still_temperature_only():
    assert verifier._cts_thermal_score_rows(row(-2, -200), row(), 1)["combined_score"] == 1


@pytest.mark.parametrize("rounds", [5, 10])
def test_declared_task_and_authority_bind_new_score(monkeypatch, rounds):
    from local_bench.reference_store import validate_contract
    spec = load_global_task(authority.FRAMEWORK / "global_tasks" / LATEST_THERMAL_TASKS[rounds])
    assert spec.objective.direction == "maximize" and spec.objective.primary_metric == "combined_score"
    assert spec.objective.parameters["timing_tolerance_fraction"] == 0.05
    assert spec.objective.parameters["timing_penalty_weight"] == 0.1
    assert spec.objective.parameters["timing_penalty"] == "weighted_excess_cost_additive_v1"
    monkeypatch.setattr(authority.latest, "validate_reference_bundle", lambda *a: {"physical_input_sha256": "test"})
    monkeypatch.setattr(authority, "file_sha", lambda *a: "test")
    contract = authority.contract_for("bp_fe", spec, "/fixture")
    assert contract["physical"]["scoring"]["parameters"] == spec.objective.parameters
    assert contract["physical"]["scoring"]["direction"] == "maximize"
    assert validate_contract(contract) == contract


@pytest.mark.parametrize("rounds", [5, 10])
def test_thermal_contract_publishes_and_adopts_without_schema_bypass(tmp_path, monkeypatch, rounds):
    from local_bench.reference_store import publish_reference, load_reference, file_sha
    spec = load_global_task(authority.FRAMEWORK / "global_tasks" / LATEST_THERMAL_TASKS[rounds])
    monkeypatch.setattr(authority.latest, "validate_reference_bundle", lambda *a: {"physical_input_sha256": "test"})
    monkeypatch.setattr(authority, "file_sha", lambda *a: "a" * 64)
    contract = authority.contract_for("bp_fe", spec, "/fixture")
    run = tmp_path.resolve() / "native"
    run.mkdir()
    evidence = run / "metrics.json"
    evidence.write_text(json.dumps(row()))
    payload = {"qualified": True, "case": "bp_fe", "endpoint": "cts_thermal",
               "source_contract": contract["physical"], "frozen_inputs": contract["frozen_inputs"],
               "source_identity": {"fixture": True},
               "baseline": {"source_run": str(run), "raw_metrics": row()},
               "evidence": [{"path": str(evidence), "bytes": evidence.stat().st_size,
                             "sha256": file_sha(evidence)}]}
    validator = lambda producer: payload
    record = publish_reference(tmp_path.resolve() / "store", contract=contract, origin="FRESH",
                               producer={"kind": "offline-fixture", "run_root": str(run)}, validate=validator)
    assert load_reference(record, contract=contract, validate=validator)["contract"] == contract
    # Neither old top-level fields nor different penalties are silently accepted.
    with pytest.raises(ValueError, match="fields"):
        load_reference(record, contract={**contract, "scoring": contract["physical"]["scoring"]}, validate=validator)
    changed = json.loads(json.dumps(contract))
    changed["physical"]["scoring"]["parameters"]["timing_penalty_weight"] = 0.2
    with pytest.raises(ValueError, match="mismatch"):
        load_reference(record, contract=changed, validate=validator)


@pytest.mark.parametrize("name,value", [("timing_tolerance_fraction", 0.1), ("wns_weight", 0.5),
    ("tns_weight", 0.5), ("deficit_floor", 1e-6), ("timing_endpoint", "pre_cts_one_opt"),
    ("timing_penalty", "none"), ("timing_penalty", "excess_cost_divisor_v1"),
    ("timing_penalty_weight", 0.2), ("timing_penalty_weight", True), ("opt_passes", 2)])
def test_scientific_parameters_cannot_silently_drift(tmp_path, name, value):
    source = authority.FRAMEWORK / "global_tasks" / LATEST_THERMAL_TASKS[10]
    data = yaml.safe_load((source / "task.yaml").read_text())
    data["objective"]["parameters"][name] = value
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(data))
    (tmp_path / "instruction.md").write_text((source / "instruction.md").read_text())
    with pytest.raises(ValueError):
        load_global_task(tmp_path)


def cts_evidence(wns=-1.2, tns=-120):
    return {"status": "success", "algorithm": sorted(verifier.CTS_ALGORITHMS)[0],
            "assembly": {"timing": {"wns_setup": wns, "tns_setup": tns}},
            "timing_validity": {"graph_completeness": "passed", "unclocked_or_unconstrained_endpoints": 0,
                "hbt_is_passive_rc_in_authoritative_global_sta": True,
                "authoritative_timing_view": "continuous-passive-hbt",
                "global_timing_role": "post-cts-placement-estimated"}}


def write_cts_tsv(path, wns=-1.2, tns=-120):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"wns_setup {wns}\ntns_setup {tns}\npassive_hbt_rc 1\nclock_count 1\n")


def test_thermal_collector_uses_post_cts_not_opt_timing(tmp_path, monkeypatch):
    seen = []
    def timing(*args, **kwargs):
        seen.append(kwargs)
        return row(-0.5, -50)  # OPT improves, but CTS regresses: CTS must win.
    monkeypatch.setattr(verifier, "_timing_row", timing)
    monkeypatch.setattr(authority.latest, "read_json", lambda *a: cts_evidence())
    monkeypatch.setattr(snapshot, "validate_snapshot", lambda *a: {"global_average_temperature_k": 320})
    write_cts_tsv(tmp_path / "snapshot/sta/metrics.tsv")
    result = authority.thermal_row(tmp_path, tmp_path / "snapshot")
    assert seen == [{"expected_passes": 1, "latest": True, "reference_suffix": True}]
    for key in ("wns_setup_ns", "tns_setup_ns", "wns_deficit", "tns_deficit"):
        assert result[key] == row(-1.2, -120)[key]
    assert result["timing_evaluation_boundary"] == "post_cts_one_opt"
    assert verifier._timing_constrained_cts_thermal_score_rows(result | {"stack": {}},
        row() | {"stack": {}, "cts_algorithm": result["cts_algorithm"]})["timing_penalty"] == pytest.approx(0.015)


@pytest.mark.parametrize("change", ["missing_wns", "missing_tns", "nan", "duplicate", "mismatch",
    "pre_cts", "unconstrained", "no_clock", "no_passive", "missing_json"])
def test_post_cts_timing_evidence_fails_closed(tmp_path, change):
    path = tmp_path / "metrics.tsv"
    write_cts_tsv(path)
    cts = cts_evidence()
    text = path.read_text()
    if change == "missing_wns":
        text = text.replace("wns_setup -1.2\n", "")
    elif change == "missing_tns":
        text = text.replace("tns_setup -120\n", "")
    elif change == "nan":
        text = text.replace("-1.2", "nan")
    elif change == "duplicate":
        text += "wns_setup -1.2\n"
    elif change == "mismatch":
        cts["assembly"]["timing"]["wns_setup"] = -0.5
    elif change == "pre_cts":
        cts["timing_validity"]["global_timing_role"] = "pre-cts"
    elif change == "unconstrained":
        cts["timing_validity"]["unclocked_or_unconstrained_endpoints"] = 1
    elif change == "no_clock":
        text = text.replace("clock_count 1", "clock_count 0")
    elif change == "no_passive":
        text = text.replace("passive_hbt_rc 1", "passive_hbt_rc 0")
    else:
        del cts["assembly"]["timing"]["tns_setup"]
    path.write_text(text)
    with pytest.raises(ValueError):
        authority._post_cts_timing(cts, path)


def test_latest_consumer_applies_penalty(tmp_path, monkeypatch):
    spec = task()
    data = {"task_digest": spec.digest(), "source_revision": authority.latest.SOURCE_REVISION,
            "image": authority.latest.IMAGE_ID, "evaluation_endpoint": {"profile": LATEST_THERMAL_PROFILE},
            "case": "bp_fe", "benchmark_input": "/fixture"}
    (tmp_path / "contract.json").write_text(json.dumps(data))
    (tmp_path / "run.json").write_text(json.dumps({"design": "bp_fe", "type": "lol", "pdk": "ng45"}))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(authority.normalized_recipe("bp_fe")))
    monkeypatch.setattr(authority, "contract_for", lambda *a: {"test": "contract"})
    monkeypatch.setattr(authority, "load_reference", lambda *a, **k: {
        "qualification": {"baseline": {"source_run": str(tmp_path), "raw_metrics": row()}}})
    monkeypatch.setattr(authority, "thermal_row", lambda *a: row(-1.1, -110))
    result = authority.score_with_latest_record(tmp_path, tmp_path, "/record", tmp_path / "contract.json",
        mode="cts-thermal", profile=LATEST_THERMAL_PROFILE,
        submission={"task_id": spec.task_id, "task_digest": spec.digest(),
                    "base_revision": authority.latest.SOURCE_REVISION})
    assert result["combined_score"] == pytest.approx(0.995)


def test_adopted_reference_feedback_has_penalty_fields(tmp_path, monkeypatch):
    instance = authority.LatestWorkspaceReference.__new__(authority.LatestWorkspaceReference)
    instance.record_path = tmp_path / "record"
    instance.record_sha256 = "sealed"
    instance.contract = {"test": "contract"}
    instance.objective = "thermal"
    assert instance.policy == "ADOPT"
    instance.runtime = SimpleNamespace()
    monkeypatch.setattr(authority, "file_sha", lambda *a: "sealed")
    monkeypatch.setattr(authority, "adopt_reference", lambda *a, **k: {
        "qualification": {"baseline": {"source_run": str(tmp_path / "run"), "raw_metrics": row()}},
        "reference_id": "test", "verification_wall_seconds": 0.1})
    result = instance.obtain(None, tmp_path)
    assert result["metrics"]["combined_score"] == 1
    assert result["metrics"]["timing_penalty"] == 0
    assert result["metrics"]["timing_penalty_weight"] == 0.1
    assert result["new_reference_native_attempts"] == 0
