"""Opt-in scoring contract v3: rise-over-ambient temperature, log-ratio aggregation,
absolute-scale overflow, seed uncertainty and noise-aware dominance.

v2 contracts never reach this module; their aggregation is unchanged in pareto.py.
"""
import math
import statistics
from .common import finite
from .contract import METRICS, OBJECTIVES, SCORING_V3_KEYS

AMBIENT_TOLERANCE_K = 1e-6
LOG_LIMIT = 700.0  # exp() stays finite and nonzero in double precision


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Invalid scoring v3 " + label)
    return value


def validate_scoring(task):
    """Strict v3 validation; called by contract.validate after cases are validated."""
    scoring = _exact(task["scoring"], SCORING_V3_KEYS, "fields")
    if type(scoring["version"]) is not int or scoring["version"] != 3:
        raise ValueError("Unsupported scoring version")
    if scoring["temperature"] not in ("rise_over_ambient", "absolute"):
        raise ValueError("Invalid scoring v3 temperature mode")
    if scoring["aggregation"] not in ("geometric", "arithmetic"):
        raise ValueError("Invalid scoring v3 aggregation")
    overflow = scoring["overflow"]
    if not isinstance(overflow, dict) or overflow.get("mode") not in ("ratio", "absolute"):
        raise ValueError("Invalid scoring v3 overflow mode")
    if overflow["mode"] == "ratio":
        _exact(overflow, ("mode",), "overflow")
    else:
        _exact(overflow, ("mode", "scale"), "overflow")
        _exact(overflow["scale"], [c["id"] for c in task["cases"]], "overflow scale cases")
        for value in overflow["scale"].values():
            finite(value, "overflow scale", 1e-12)
    validate_dominance(scoring["dominance"])
    if task.get("epsilon", 0) != 0:
        raise ValueError("Scoring v3 replaces epsilon boxes with its dominance model (epsilon=0)")


def validate_dominance(dominance):
    if not isinstance(dominance, dict) or dominance.get("mode") not in ("exact", "noise"):
        raise ValueError("Invalid scoring v3 dominance mode")
    if dominance["mode"] == "exact":
        return _exact(dominance, ("mode",), "dominance")
    keys = set(dominance) - {"seed_uncertainty"}
    _exact({k: dominance[k] for k in keys}, ("mode", "k", "noise_floor"), "noise dominance")
    finite(dominance["k"], "dominance k", 0)
    _exact(dominance["noise_floor"], OBJECTIVES, "noise floor objectives")
    for value in dominance["noise_floor"].values():
        finite(value, "noise floor", 0)
    if type(dominance.get("seed_uncertainty", False)) is not bool:
        raise ValueError("seed_uncertainty must be boolean")
    return dominance


def cell(task, case, metric, x, r):
    """(value, log value) of one raw metric; every value is positive and neutral (1) at equality."""
    scoring, floor = task["scoring"], task["normalization_floors"][metric]
    if metric == "overflow" and scoring["overflow"]["mode"] == "absolute":
        log = (x - r) / scoring["overflow"]["scale"][case]
        if abs(log) > LOG_LIMIT:
            raise ValueError("Normalized overflow out of range (overflow scale too small)")
        return math.exp(log), log
    if metric == "temperature_peak_k" and scoring["temperature"] == "rise_over_ambient":
        ambient = task["ambient_k"]
        if min(x, r) < ambient - AMBIENT_TOLERANCE_K:
            raise ValueError("Peak temperature below ambient")
        x, r = max(0.0, x - ambient), max(0.0, r - ambient)
    value = (x + floor) / (r + floor)
    return value, math.log(value)


def compute(task, cells, rows, refs, split):
    """Score paired, already-validated rows. cells: sorted (case, seed) keys of the split.

    Geometric: case = seed mean of log values, aggregate = exp(weighted case mean).
    Arithmetic: the same with the values themselves. Timing combines WNS/TNS per cell in
    the aggregation space, so it equals the weighted combination of the two aggregates."""
    scoring = task["scoring"]
    geometric = scoring["aggregation"] == "geometric"
    to_value = math.exp if geometric else float
    total = sum(scoring["timing_weights"].values())
    weights = {m: w / total for m, w in scoring["timing_weights"].items()}
    cases = [c for c in task["cases"] if c["split"] == split]
    weight = sum(c["weight"] for c in cases)
    per_case, levels, case_objectives, stats = {}, {}, {}, {}
    for case in cases:
        keys = [k for k in cells if k[0] == case["id"]]
        pairs = {m: [cell(task, case["id"], m, rows[k]["metrics"][m], refs[k]["metrics"][m]) for k in keys]
                 for m in METRICS}
        space = {m: [p[1] if geometric else p[0] for p in pairs[m]] for m in METRICS}
        entry, levels[case["id"]] = {}, {}
        for m in METRICS:
            values = [p[0] for p in pairs[m]]
            levels[case["id"]][m] = statistics.mean(space[m])
            entry[m] = {"mean": to_value(levels[case["id"]][m]),
                        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "worst": max(values), "samples": values}
            if geometric:
                entry[m]["log_mean"] = levels[case["id"]][m]
                entry[m]["log_stdev"] = statistics.stdev(space[m]) if len(keys) > 1 else 0.0
        per_case[case["id"]] = entry
        samples = {"timing": [sum(weights[m] * space[m][i] for m in weights) for i in range(len(keys))],
                   "overflow": space["overflow"], "temperature_peak_k": space["temperature_peak_k"]}
        stats[case["id"]] = {o: (statistics.mean(s), statistics.variance(s) if len(s) > 1 else 0.0, len(s))
                             for o, s in samples.items()}
        case_objectives[case["id"]] = {o: to_value(stats[case["id"]][o][0]) for o in OBJECTIVES}
    ratios = {m: finite(to_value(sum(c["weight"] * levels[c["id"]][m] for c in cases) / weight), m + " ratio", 0)
              for m in METRICS}
    vector, uncertainty, worst = {}, {}, {}
    for o in OBJECTIVES:
        level = sum(c["weight"] * stats[c["id"]][o][0] for c in cases) / weight
        # Seed-sampling standard error: cases fixed, seeds independent within each case.
        error = math.sqrt(sum((c["weight"] / weight) ** 2 * stats[c["id"]][o][1] / stats[c["id"]][o][2]
                              for c in cases))
        vector[o] = finite(to_value(level), o, 0)
        # Geometric: delta method, d exp(L) = exp(L) dL.
        uncertainty[o] = finite(vector[o] * error if geometric else error, o + " uncertainty", 0)
        name = max(cases, key=lambda c: case_objectives[c["id"]][o])["id"]  # first maximum in case order
        worst[o] = {"case": name, "value": case_objectives[name][o]}
    return {"vector": vector, "metric_ratios": ratios, "per_case": per_case,
            "per_case_objectives": case_objectives, "uncertainty": uncertainty, "worst_case": worst}


def tolerance(dominance, a, b):
    """Per-objective tolerance for comparing two v3 rows (zero for exact dominance)."""
    if dominance["mode"] == "exact":
        return (0.0,) * len(OBJECTIVES)
    seed = dominance.get("seed_uncertainty", False)
    return tuple(dominance["k"] * math.sqrt(2 * dominance["noise_floor"][o] ** 2 +
                 ((a["uncertainty"][o] ** 2 + b["uncertainty"][o] ** 2) if seed else 0.0))
                 for o in OBJECTIVES)


def dominates(a, b, tol):
    """a dominates b: no worse than b + tol everywhere and better than b - tol somewhere.

    With any nonzero tolerance the relation is also required to strictly lower the
    objective sum. Tolerant dominance alone can be cyclic (a>b>c>a, all removed); the
    sum condition makes it acyclic, so the minimum-sum feasible row is always retained.
    Zero tolerance is exact Pareto dominance, which is already acyclic."""
    x, y = [a["vector"][o] for o in OBJECTIVES], [b["vector"][o] for o in OBJECTIVES]
    return (all(p <= q + t for p, q, t in zip(x, y, tol)) and any(p < q - t for p, q, t in zip(x, y, tol))
            and (not any(tol) or sum(x) < sum(y)))


def retain(rows, dominance, retain_equivalent):
    """Rows (feasible, unique candidates) not dominated by any other row, sorted by candidate."""
    rows = sorted(rows, key=lambda r: r["candidate"])
    kept = [r for r in rows if not any(o is not r and dominates(o, r, tolerance(dominance, o, r)) for o in rows)]
    if retain_equivalent:
        return kept
    seen, result = set(), []
    for row in kept:
        v = tuple(row["vector"][o] for o in OBJECTIVES)
        if v not in seen:
            seen.add(v)
            result.append(row)
    return result
