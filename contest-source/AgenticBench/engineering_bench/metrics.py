"""Objective metrics are independent of the LLM review score."""
import math


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Missing/nonfinite numeric metric: " + label)
    return float(value)


def clock_skew(groups, expected):
    """Never subtract arrivals across unrelated clock/edge/analysis groups.

    expected is evaluator-owned: group ID -> complete original sink ID list.
    Each ID encodes domain, active edge, analysis and corner in the producer.
    """
    if not expected or set(groups) != set(expected):
        raise ValueError("Clock analysis group inventory changed")
    values = []
    for group, sinks in expected.items():
        if not sinks or len(sinks) != len(set(sinks)) or set(groups[group]) != set(sinks):
            raise ValueError("Clock sink inventory changed: " + group)
        arrivals = [number(groups[group][sink], sink) for sink in sinks]
        values.append(max(arrivals) - min(arrivals))
    return max(values)


def target_delay(nets, expected):
    """Use every frozen target net and original remote receiver, unchanged or not."""
    if not expected or set(nets) != set(expected):
        raise ValueError("Target net inventory changed")
    values = []
    for name, sinks in expected.items():
        if not sinks or len(sinks) != len(set(sinks)) or set(nets[name]) != set(sinks):
            raise ValueError("Target receiver inventory changed: " + name)
        delays = [number(nets[name][sink], sink) for sink in sinks]
        if min(delays) < 0:
            raise ValueError("Negative propagation delay")
        values.append(max(delays))
    return sum(values) / len(values)


def _percentile(values, fraction):
    """Nearest-rank percentile (deterministic, no interpolation)."""
    ordered = sorted(values)
    rank = max(1, int(math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def clock_distribution(groups, expected):
    """Skew/latency distribution over the frozen clock-group inventory (v2).

    max_skew_ns equals clock_skew(). Per sink, skew is its arrival minus the
    earliest arrival of its own group (never across groups); mean/p95 are over
    all sinks of all groups. Insertion delay is the sink arrival itself, taken
    over setup ("/max/") groups when present.
    """
    max_skew = clock_skew(groups, expected)
    per_group, deviations, latency = {}, [], []
    for group, sinks in sorted(expected.items()):
        arrivals = [number(groups[group][sink], sink) for sink in sinks]
        per_group[group] = max(arrivals) - min(arrivals)
        deviations += [a - min(arrivals) for a in arrivals]
        if "/max/" in group or not any("/max/" in g for g in expected):
            latency += arrivals
    return {"max_skew_ns": max_skew,
            "mean_group_skew_ns": sum(per_group.values()) / len(per_group),
            "mean_sink_skew_ns": sum(deviations) / len(deviations),
            "p95_sink_skew_ns": _percentile(deviations, 0.95),
            "max_insertion_delay_ns": max(latency),
            "mean_insertion_delay_ns": sum(latency) / len(latency),
            "sinks": len(deviations), "groups": len(per_group)}


def wns_tns_guard(reference, candidate, tolerances, relative_tolerance=0.0):
    """Setup/hold WNS/TNS may not regress by more than abs + rel*|reference|."""
    relative_tolerance = number(relative_tolerance, "relative_tolerance")
    if relative_tolerance < 0:
        raise ValueError("Negative timing tolerance")
    failures, margins = [], {}
    for key in ("setup_wns_ns", "setup_tns_ns", "hold_wns_ns", "hold_tns_ns"):
        before, after = number(reference.get(key), key), number(candidate.get(key), key)
        tolerance = number(tolerances.get(key), key)
        if tolerance < 0:
            raise ValueError("Negative timing tolerance")
        allowed = tolerance + relative_tolerance * abs(before)
        margins[key] = {"reference": before, "candidate": after, "allowed_regression": allowed,
                        "regression": max(0.0, before - after)}
        if after < before - allowed:
            failures.append(key)
    return {"passed": not failures, "failures": failures, "margins": margins}


def bounded_ratio(actual, reference, ratio, label, absolute=0.0):
    """actual <= reference*ratio + absolute, reported with the bound."""
    actual, reference = number(actual, label), number(reference, label + " reference")
    bound = reference * number(ratio, label + " ratio") + number(absolute, label + " tolerance")
    return {"measured": actual, "reference": reference, "bound": bound,
            "passed": actual <= bound + 1e-12}


def receiver_subset(nets, expected):
    """Restrict per-net receiver delays to an evaluator-owned inventory."""
    if set(nets) != set(expected):
        raise ValueError("Target net inventory changed")
    return {name: {sink: nets[name][sink] for sink in sinks if sink in nets[name]}
            for name, sinks in expected.items()}


def target_delay_report(nets, inventories):
    """T3 v2: primary metric over all receivers plus remote/local subsets.

    inventories = {"all": ..., "remote": ..., "local": ...}; an empty local
    subset (no local receiver on any target net) is reported as None.
    """
    result = {"mean_target_net_delay_ns": target_delay(receiver_subset(nets, inventories["all"]), inventories["all"])}
    for scope in ("remote", "local"):
        expected = {n: s for n, s in inventories[scope].items() if s}
        key = scope + "_receivers_mean_target_net_delay_ns"
        result[key] = target_delay(receiver_subset({n: nets[n] for n in expected}, expected), expected) if expected else None
    result["receiver_counts"] = {scope: sum(len(s) for s in inventories[scope].values())
                                 for scope in ("all", "remote", "local")}
    return result


def timing_guard(reference, candidate, tolerances):
    """Negative baselines are allowed. Tolerances are frozen by the harness."""
    failures = []
    for key in ("setup_wns_ns", "setup_tns_ns", "hold_wns_ns", "hold_tns_ns"):
        before, after = number(reference.get(key), key), number(candidate.get(key), key)
        tolerance = number(tolerances.get(key), key)
        if tolerance < 0:
            raise ValueError("Negative timing tolerance")
        if after < before - tolerance:
            failures.append(key)
    before = reference.get("hold_violating_endpoints")
    after = candidate.get("hold_violating_endpoints")
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("Missing hold endpoint inventory")
    if any(not isinstance(x, str) for x in before + after):
        raise ValueError("Invalid hold endpoint identity")
    if set(after) - set(before):
        failures.append("new_hold_violating_endpoints")
    for key in ("endpoints_without_min_path", "endpoints_without_max_path"):
        if key in reference:
            old, new = reference[key], candidate.get(key)
            if not isinstance(old, list) or not isinstance(new, list):
                raise ValueError("Missing endpoint analysis availability: " + key)
            if set(new) != set(old):
                failures.append(key)
    if "endpoint_count" in reference and candidate.get("endpoint_count") != reference["endpoint_count"]:
        failures.append("endpoint_count")
    return {"passed": not failures, "failures": failures}


def aggregate(task, records):
    cases = task.data["cases"]
    if len({row["case"] for row in records}) != len(records):
        raise ValueError("Repeated case evaluations are not allowed")
    if any(row["case"] not in cases for row in records):
        raise ValueError("Unexpected case")
    improvements, failures, running = {}, [], []
    for row in records:
        case = row["case"]
        if row.get("status") == "dispatched":
            running.append(case)
            continue
        if row.get("status") not in {"succeeded", "failed"}:
            raise ValueError("Unknown final status")
        if row.get("status") != "succeeded" or row.get("legal") is not True:
            failures.append(case)
            continue
        ref = number(row.get("reference_metric"), "reference_metric")
        candidate = number(row.get("candidate_metric"), "candidate_metric")
        if ref <= 0 or candidate < 0:
            raise ValueError("Scored references must be positive; zero-baseline fixtures are functional tests")
        improvements[case] = 100.0 * (ref - candidate) / ref
    missing = sorted(set(cases) - {row["case"] for row in records})
    mean = sum(improvements.values()) / len(improvements) if improvements else None
    result = {"task": task.task_id, "metric": task.data["primary_metric"],
            "expected_cases": len(cases), "successful_cases": len(improvements),
            "failed_cases": failures, "running_cases": running, "missing_cases": missing,
            "per_case_improvement_percent": improvements,
            "all_case_mean_improvement_percent": mean if not failures and not missing and not running else None,
            "successful_subset_mean_improvement_percent": mean,
            "judge_score_is_separate": True}
    from .scoring import scoring_version
    versions = {scoring_version(row) for row in records}
    if len(versions) > 1:
        raise ValueError("Mixed scoring versions in one report")
    if versions == {2}:
        result.update(calibrated(task, records, running, missing))
    return result


def calibrated(task, records, running, missing):
    """v2 calibration; v1 fields above are kept unchanged.

    normalized = 100*(qor - qor_noop)/(qor_expert - qor_noop), where qor_noop
    is the qualified reference metric and qor_expert comes from the expert
    registry (None until registered). Failed cases score 0 in the headline;
    running/missing cases leave the headline undefined.
    """
    from .scoring import load_profile, expert_reference
    profile, profile_sha = load_profile(task)
    if any(row.get("scoring_profile_sha256") != profile_sha for row in records):
        raise ValueError("Evaluation profile changed after qualification")
    experts, registry_sha = expert_reference(task, profile)
    per_case, legal = {}, []
    for row in records:
        case = row["case"]
        noop = number(row.get("reference_metric"), "reference_metric")
        entry = {"status": None, "qor_noop": noop, "qor": None, "qor_expert": experts[case],
                 "raw_improvement": None, "noop_relative_score_percent": None,
                 "normalized_score": None, "headline_score_percent": None,
                 "headline_normalized_score": None}
        if row.get("status") == "dispatched":
            entry["status"] = "running"
        elif row.get("status") == "succeeded" and row.get("legal") is True:
            qor = number(row.get("candidate_metric"), "candidate_metric")
            entry.update(status="legal", qor=qor, raw_improvement=noop - qor,
                         noop_relative_score_percent=100.0 * (noop - qor) / noop)
            if experts[case] is not None and experts[case] != noop:
                entry["normalized_score"] = 100.0 * (qor - noop) / (experts[case] - noop)
            legal.append(case)
        else:
            entry["status"] = "failed"
            entry["noop_relative_score_percent"] = None
        if entry["status"] == "legal":
            entry["headline_score_percent"] = entry["noop_relative_score_percent"]
            entry["headline_normalized_score"] = entry["normalized_score"]
        elif entry["status"] == "failed":
            entry["headline_score_percent"] = 0.0
            entry["headline_normalized_score"] = 0.0 if experts[case] is not None else None
        if "details" in row:
            entry["details"] = row["details"]
        per_case[case] = entry
    complete = not running and not missing
    cases = task.data["cases"]
    headline = (sum(per_case[c]["headline_score_percent"] for c in cases) / len(cases)) if complete else None
    all_experts = all(experts[c] is not None and experts[c] != per_case[c]["qor_noop"] for c in per_case)
    normalized = (sum(per_case[c]["headline_normalized_score"] for c in cases) / len(cases)) if complete and all_experts else None
    finished = [c for c in per_case if per_case[c]["status"] in ("legal", "failed")]
    legal_scores = [per_case[c]["noop_relative_score_percent"] for c in legal]
    return {"scoring_version": 2, "scoring_profile_sha256": profile_sha,
            "expert_registry_sha256": registry_sha,
            "calibration": per_case,
            "legality_rate": (len(legal) / len(finished)) if finished else None,
            "legality_rate_denominator": len(finished),
            "legal_cases": sorted(legal),
            "legal_case_mean_noop_relative_score_percent": (sum(legal_scores) / len(legal_scores)) if legal_scores else None,
            "headline_noop_relative_score_percent_failed_as_zero": headline,
            "headline_normalized_score_failed_as_zero": normalized,
            "headline_normalized_status": "available" if normalized is not None else (
                "incomplete_cases" if not complete else "expert_unregistered"),
            "judge_weight_in_headline": 0}
