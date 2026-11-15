"""Consume trusted native receipts; candidate-supplied scores are never accepted.

This is the framework side of the subprocess/artifact boundary. The physical
adapter must run separately with protected evaluator code, independent of the
submitted implementation, and publish the raw measurements and proofs below.

scoring_version 1 receipts are validated exactly as frozen. scoring_version 2
(declared by the qualification) adds per-capability evidence and the gates in
engineering_tasks/<task>/evaluation-v2.json.
"""
from pathlib import Path

from .capabilities import verify as verify_capabilities
from .metrics import (bounded_ratio, clock_distribution, clock_skew, number, target_delay,
                      target_delay_report, timing_guard, wns_tns_guard)
from .protocol import artifact, qualify
from .tasks import read_json


def validate_native_result(task, case, qualification_path, result_path, *, submission_sha256, runtime_sha256):
    qualification = qualify(task, case, qualification_path)
    q = qualification["record"]
    version = qualification["scoring_version"]
    reference_path = artifact(Path(qualification_path).parent, q["reference"])
    reference = read_json(reference_path)
    result_path = Path(result_path)
    result = read_json(result_path)
    identities = {"kind": "engineering-native-result-v1", "task": task.task_id,
        "task_digest": task.digest(), "case": case, "seed": task.data["seed"],
        "input_sha256": q["input_sha256"], "evaluator_sha256": q["evaluator_sha256"],
        "endpoint": task.data["endpoint"], "submission_sha256": submission_sha256,
        "runtime_sha256": runtime_sha256, "qualification_sha256": qualification["sha256"]}
    if version != 1:
        identities["scoring_version"] = version
    for key, value in identities.items():
        if type(result.get(key)) is not type(value) or result[key] != value:
            raise ValueError("Native result identity mismatch: " + key)
    if version == 1 and "scoring_version" in result:
        raise ValueError("Native result identity mismatch: scoring_version")
    # Native producers list every physical gate, with bytes binding to reports.
    from .scoring import profile_for
    profile = profile_for(task, q)
    gates = result.get("checks")
    if not isinstance(gates, dict):
        raise ValueError("Missing independent native checks")
    capability_summary = verify_capabilities(
        result_path.parent, gates, task.data["required_capabilities"],
        allow_unverified=profile["allow_unverified_capabilities"] if profile else (),
        require_schema=2 if version == 2 else 1, missing="Failed/missing physical gate: ")
    measurement = read_json(artifact(result_path.parent, result.get("measurements")))
    if version == 2:
        return dict(_validate_v2(task, reference, measurement, profile),
                    capabilities=capability_summary, scoring_version=2,
                    endpoint=task.data["endpoint"], native_result_path=str(result_path.resolve()))
    # The task contract, not a candidate flag, selects acceptance semantics.
    # T4 is skew-only at CTS. Timing may be reported but is not an extra gate.
    guard = {"required": False, "passed": None, "failures": []}
    if "setup_hold_guard" in task.data["required_capabilities"]:
        guard = {"required": True, **timing_guard(
            reference["timing"], measurement["timing"], reference["timing_tolerances"])}
        if not guard["passed"]:
            raise ValueError("Independent setup/hold guard failed: " + ", ".join(guard["failures"]))
    metric = clock_skew if task.task_id != "multi_hbt_fanout" else target_delay
    # Inventory is frozen in the qualified reference, never picked by candidate.
    baseline = metric(reference["measurements"], reference["expected_inventory"])
    if abs(baseline - number(reference["primary_metric"], "reference")) > 1e-12:
        raise ValueError("Reference scalar differs from independently reduced measurements")
    value = metric(measurement["values"], reference["expected_inventory"])
    return {"legal": True, "candidate_metric": value, "reference_metric": baseline,
            "timing_guard": guard, "timing_diagnostic": measurement.get("timing"),
            "endpoint": task.data["endpoint"], "capabilities": capability_summary,
            "native_result_path": str(result_path.resolve())}


def _validate_v2(task, reference, measurement, profile):
    gates = profile["gates"]
    details = {}
    if task.task_id == "multi_hbt_fanout":
        inventories = reference["expected_inventory_by_scope"]
        baseline_report = target_delay_report(reference["measurements"], inventories)
        report = target_delay_report(measurement["values"], inventories)
        baseline, value = baseline_report["mean_target_net_delay_ns"], report["mean_target_net_delay_ns"]
        area = measurement["added_buffer_area_um2"]
        budget = reference["added_buffer_area_budget_um2"]
        if number(area, "added_buffer_area_um2") > number(budget, "added buffer area budget") + 1e-9:
            raise ValueError("Added buffer area exceeds the declared bound")
        details.update(receivers=report, reference_receivers=baseline_report,
                       added_buffer_area={"measured_um2": area, "bound_um2": budget, "passed": True})
    else:
        baseline = clock_skew(reference["measurements"], reference["expected_inventory"])
        value = clock_skew(measurement["values"], reference["expected_inventory"])
        if task.task_id == "cross_die_cts":
            details.update(t4_gates(reference, measurement, gates))
    if abs(baseline - number(reference["primary_metric"], "reference")) > 1e-12:
        raise ValueError("Reference scalar differs from independently reduced measurements")
    guard = {"required": False, "passed": None, "failures": []}
    if "setup_hold_guard" in task.data["required_capabilities"]:
        guard = {"required": True, **timing_guard(
            reference["timing"], measurement["timing"], reference["timing_tolerances"])}
        if not guard["passed"]:
            raise ValueError("Independent setup/hold guard failed: " + ", ".join(guard["failures"]))
    return {"legal": True, "candidate_metric": value, "reference_metric": baseline,
            "timing_guard": guard, "timing_diagnostic": measurement.get("timing"), "details": details}


def clock_power_gate(reference_power, candidate_power, ratio):
    """Bound clock power only when both sides were measured; never invent it."""
    def value(row):
        if isinstance(row, dict) and row.get("status") == "measured":
            return number(row.get("value_mw"), "clock power")
        return None
    before, after = value(reference_power), value(candidate_power)
    if before is None or after is None:
        return {"status": "not_measured", "value_mw": None, "passed": None,
                "reason": "the independent native measurement produces no clock power report"}
    return dict(bounded_ratio(after, before, ratio, "clock_power"), status="measured", value_mw=after)


def t4_gates(reference, measurement, gates):
    """T4 v2 hard gates; skew alone cannot be bought with timing/latency/area."""
    inventory = reference["expected_inventory"]
    before = clock_distribution(reference["measurements"], inventory)
    after = clock_distribution(measurement["values"], inventory)
    failures = []
    timing = wns_tns_guard(reference["timing"], measurement["timing"],
                           gates["timing"]["tolerances_ns"], gates["timing"]["relative_tolerance"])
    failures += ["timing:" + x for x in timing["failures"]]
    latency = bounded_ratio(after["max_insertion_delay_ns"], before["max_insertion_delay_ns"],
                            gates["insertion_delay"]["max_reference_ratio"], "insertion_delay",
                            gates["insertion_delay"]["absolute_tolerance_ns"])
    if not latency["passed"]:
        failures.append("insertion_delay")
    old, new = reference["clock_resources"], measurement["clock_resources"]
    area = bounded_ratio(new["clock_area_um2"], old["clock_area_um2"], gates["clock_area_max_reference_ratio"], "clock_area")
    count = bounded_ratio(new["clock_buffer_count"], old["clock_buffer_count"],
                          gates["clock_buffer_count_max_reference_ratio"], "clock_buffer_count")
    failures += [k for k, v in (("clock_area", area), ("clock_buffer_count", count)) if not v["passed"]]
    power = clock_power_gate(reference.get("clock_power"), measurement.get("clock_power"),
                             gates["clock_power"]["max_reference_ratio"])
    if power["passed"] is False:
        failures.append("clock_power")
    if failures:
        raise ValueError("T4 v2 hard gate failed: " + ", ".join(failures))
    return {"skew_distribution": after, "reference_skew_distribution": before,
            "timing_gate": timing, "insertion_delay_gate": latency,
            "clock_area_gate": area, "clock_buffer_count_gate": count, "clock_power": power}
