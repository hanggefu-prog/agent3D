"""Freeze the evaluation matrix; seeds and objectives are never agent inputs."""
import math
import re
from . import ENDPOINT, TIMING_ENDPOINT
from .common import digest, finite

METRICS = ("wns_deficit_ns", "tns_deficit_ns", "overflow", "temperature_peak_k", "power_w")
OBJECTIVES = ("timing", "overflow", "temperature_peak_k")
DEFAULT_SCORING = {"timing_weights": {"wns_deficit_ns": 0.7, "tns_deficit_ns": 0.3},
                   "power_max_ratio": 1.10}
SCORING_V2_KEYS = ("timing_weights", "power_max_ratio")
# Scoring v3 extends the peak endpoint; the timing-only schema v3 is separate.
SCORING_V3_KEYS = ("version", "timing_weights", "power_max_ratio", "temperature", "aggregation",
                   "overflow", "dominance")
TIMING_METRICS = METRICS[:3]
TIMING_OBJECTIVES = ("score",)
DEFAULT_TIMING_SCORING = {"timing_weights": dict(DEFAULT_SCORING["timing_weights"]),
    "overflow_penalty": {"min": 300.0, "max": 600.0, "lambda": 0.1}}


def timing_only(task):
    return task.get("endpoint") == TIMING_ENDPOINT


DEFAULT_SEEDS = [
    {"id": "s1", "io": 1, "partition": 1001, "placement": 1001, "routing": -1},
    {"id": "s2", "io": 1, "partition": 1002, "placement": 1002, "routing": -1},
    {"id": "s3", "io": 1, "partition": 1003, "placement": 1003, "routing": -1},
]


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value):
        raise ValueError("Invalid identifier")
    return value


def validate(task):
    timing = timing_only(task)
    if task.get("endpoint") not in {ENDPOINT, TIMING_ENDPOINT}:
        raise ValueError("Requires a supported post-GRT endpoint")
    metrics, objectives = (TIMING_METRICS, TIMING_OBJECTIVES) if timing else (METRICS, OBJECTIVES)
    if task.get("schema_version") != (3 if timing else 2) or task.get("objectives") != list(objectives):
        raise ValueError("Task schema/objectives differ from endpoint")
    scoring = task.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("Missing/unknown scoring contract fields")
    keys = ({"timing_weights", "overflow_penalty"} if timing else
            set(SCORING_V3_KEYS if "version" in scoring else SCORING_V2_KEYS))
    if set(scoring) != keys:
        raise ValueError("Missing/unknown scoring contract fields")
    if timing:
        penalty = scoring["overflow_penalty"]
        if not isinstance(penalty, dict) or set(penalty) != {"min", "max", "lambda"}:
            raise ValueError("Requires fixed overflow min/max and lambda")
        low = finite(penalty["min"], "overflow penalty minimum", 0)
        high = finite(penalty["max"], "overflow penalty maximum", 0)
        if high <= low:
            raise ValueError("Overflow penalty maximum must exceed minimum")
        finite(penalty["lambda"], "overflow penalty lambda", 0)
    weights = scoring["timing_weights"]
    if not isinstance(weights, dict) or set(weights) != {"wns_deficit_ns", "tns_deficit_ns"}:
        raise ValueError("Timing requires WNS and TNS weights")
    values = [finite(v, "timing weight", 1e-12) for v in weights.values()]
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Timing weights must sum to one")
    if not timing:
        finite(scoring["power_max_ratio"], "power maximum ratio", 1)
    identifier(task["id"])
    if not re.fullmatch(r"[0-9a-f]{64}", task.get("evaluator_sha256", "")):
        raise ValueError("Unbound evaluator implementation")
    if task.get("metrics") != list(metrics):
        raise ValueError("Raw metrics differ from endpoint")
    if task.get("opt_passes") != 1:
        raise ValueError("The post-GRT contract uses exactly one OPT pass")
    if timing and any(k in task for k in ("activity", "ambient_k", "grid_size")):
        raise ValueError("Timing-only task has no power/thermal settings")
    if not timing and (task.get("activity") != 0.1 or task.get("ambient_k") != 318.15 or task.get("grid_size") != 10):
        raise ValueError("Power/thermal contract changed")
    cases, seeds = task["cases"], task["seeds"]
    if not cases or not seeds:
        raise ValueError("Empty evaluation matrix")
    if len({identifier(c["id"]) for c in cases}) != len(cases):
        raise ValueError("Duplicate case")
    for case in cases:
        if case["split"] not in {"train", "test", "auxiliary"} or case["pdk"] not in {"ng45", "asap7"}:
            raise ValueError("Invalid case split/platform")
        identifier(case["design"])
        finite(case["weight"], "case weight", 1e-12)
        for key in ("config_sha256", "input_identity"):
            if not re.fullmatch(r"[0-9a-f]{64}", case[key]):
                raise ValueError("Unbound case " + key)
    if len({identifier(s["id"]) for s in seeds}) != len(seeds):
        raise ValueError("Duplicate seed ID")
    if len({tuple(s[k] for k in ("partition", "placement", "routing")) for s in seeds}) != len(seeds):
        raise ValueError("Duplicate seed vector")
    for seed in seeds:
        if set(seed) != {"id", "io", "partition", "placement", "routing"}:
            raise ValueError("Unsupported seed component (OPT has no exposed seed API)")
        if seed["io"] != 1 or any(type(seed[k]) is not int or seed[k] < 0 for k in ("partition", "placement")) or seed["routing"] != -1:
            raise ValueError("Invalid replication controls")
    if set(task["normalization_floors"]) != set(metrics):
        raise ValueError("Missing normalization floors")
    for name in metrics:
        finite(task["normalization_floors"][name], "floor", 1e-12)
    finite(task.get("epsilon", 0), "epsilon", 0)
    if "optimization" in task:
        from .optimization import validate_optimization, OBJECTIVE, TIMING_OBJECTIVE
        validate_optimization(task["optimization"])
        if task["optimization"]["objective"] != (TIMING_OBJECTIVE if timing else OBJECTIVE):
            raise ValueError("Optimization objective differs from endpoint")
        if task.get("epsilon", 0) != 0:
            raise ValueError("Optimization tasks retain the exact non-dominated set (epsilon=0)")
        if not any(c["split"] == "train" for c in cases):
            raise ValueError("Optimization requires a nonempty training split")
    if not timing and "version" in scoring:
        from .scoring_v3 import validate_scoring
        validate_scoring(task)
    if "frontend_inputs" in task:
        from .frozen_frontend import validate_contract
        validate_contract(task)
    if "incremental" in task:
        from .incremental import validate_contract
        validate_contract(task)
    return digest(task)


def matrix(task, candidate, split="train"):
    identity = validate(task)
    if not re.fullmatch(r"[0-9a-f]{64}", candidate):
        raise ValueError("Candidate must be a content digest")
    return [{"task_digest": identity, "candidate": candidate, "case": case["id"],
             "seed": seed, "endpoint": task["endpoint"], "split": split}
            for case in task["cases"] if case["split"] == split for seed in task["seeds"]]
