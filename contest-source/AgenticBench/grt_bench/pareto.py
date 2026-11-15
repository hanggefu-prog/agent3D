"""Paired normalization and deterministic, order-independent Pareto retention."""
import copy
import math
import re
import statistics
from .common import digest, finite
from .contract import OBJECTIVES, TIMING_OBJECTIVES, timing_only, matrix, validate


def _paired(task, candidate, outcomes, reference, split):
    task_id = validate(task)
    metrics = task["metrics"]
    expected = {(r["case"], r["seed"]["id"]) for r in matrix(task, candidate, split)}
    if not expected:
        raise ValueError("Empty split")
    def index(rows, is_reference=False):
        result = {}
        for row in rows:
            key = (row["case"], row["seed"]["id"])
            if key in result:
                raise ValueError("Duplicate case/seed outcome")
            if row["task_digest"] != task_id or row["endpoint"] != task["endpoint"] or row["split"] != split:
                raise ValueError("Incomparable outcome contract")
            if not is_reference and row["candidate"] != candidate:
                raise ValueError("Wrong candidate")
            if row.get("status") != "verified":
                raise ValueError("Every matrix cell must pass independent verification")
            seed = next((s for s in task["seeds"] if s["id"] == key[1]), None)
            if row["seed"] != seed:
                raise ValueError("Seed vector differs")
            if set(row["metrics"]) != set(metrics):
                raise ValueError("Raw metric set differs")
            for metric in metrics:
                finite(row["metrics"].get(metric), metric, 0)
            result[key] = row
        if set(result) != expected:
            raise ValueError("Incomplete/extra matrix cells")
        return result
    rows, refs = index(outcomes), index(reference, True)
    if len({r["candidate"] for r in refs.values()}) != 1:
        raise ValueError("Mixed reference candidates")
    if "optimization" in task and any(r["candidate"] != task["optimization"]["baseline"] for r in refs.values()):
        raise ValueError("Reference must be the frozen baseline, not a parent")
    return task_id, expected, rows, refs


def _power(task, keys, rows, refs):
    power = statistics.mean(rows[k]["metrics"]["power_w"] for k in keys)
    baseline_power = statistics.mean(refs[k]["metrics"]["power_w"] for k in keys)
    limit = finite(baseline_power * task["scoring"]["power_max_ratio"], "power limit", 0)
    return {"candidate_mean_w": power, "baseline_mean_w": baseline_power,
            "limit_w": limit, "excess_w": max(0.0, power - limit), "feasible": power <= limit}


def aggregate(task, candidate, outcomes, reference, split="train"):
    task_id, expected, rows, refs = _paired(task, candidate, outcomes, reference, split)
    if "version" in task["scoring"]:
        return _aggregate_v3(task, candidate, task_id, expected, rows, refs, split)
    metrics = task["metrics"]
    timing_mode = timing_only(task)
    per_case, power_cases, penalties = {}, {}, {}
    for case in task["cases"]:
        if case["split"] != split:
            continue
        samples = {m: [] for m in metrics}
        for key in sorted(expected):
            if key[0] != case["id"]:
                continue
            for metric in metrics:
                x, r = rows[key]["metrics"][metric], refs[key]["metrics"][metric]
                floor = task["normalization_floors"][metric]
                # Shifted ratios are neutral at equality and finite for zero reference overflow/slack.
                samples[metric].append((x + floor) / (r + floor))
        per_case[case["id"]] = {m: {"mean": statistics.mean(v),
            "stdev": statistics.stdev(v) if len(v) > 1 else 0.0, "worst": max(v), "samples": v}
            for m, v in samples.items()}
        keys = sorted(k for k in expected if k[0] == case["id"])
        if timing_mode:
            rule = task["scoring"]["overflow_penalty"]
            values = [min(1.0, max(0.0, (rows[k]["metrics"]["overflow"] - rule["min"]) /
                                      (rule["max"] - rule["min"]))) for k in keys]
            penalties[case["id"]] = {"samples": values, "mean": statistics.mean(values)}
        else:
            power = statistics.mean(rows[k]["metrics"]["power_w"] for k in keys)
            baseline_power = statistics.mean(refs[k]["metrics"]["power_w"] for k in keys)
            limit = finite(baseline_power * task["scoring"]["power_max_ratio"], "power limit", 0)
            power_cases[case["id"]] = {"candidate_mean_w": power, "baseline_mean_w": baseline_power,
                "limit_w": limit, "excess_w": max(0.0, power - limit), "feasible": power <= limit}
    cases = [c for c in task["cases"] if c["split"] == split]
    weight = sum(c["weight"] for c in cases)
    ratios = {m: finite(sum(c["weight"] * per_case[c["id"]][m]["mean"] for c in cases) / weight,
                        m + " ratio", 0) for m in metrics}
    timing_weights = task["scoring"]["timing_weights"]
    timing = sum(timing_weights[m] * ratios[m] for m in timing_weights) / sum(timing_weights.values())
    if timing_mode:
        penalty = sum(c["weight"] * penalties[c["id"]]["mean"] for c in cases) / weight
        vector = {"score": timing + task["scoring"]["overflow_penalty"]["lambda"] * penalty}
        feasible, constraints = True, {}
    else:
        vector = {"timing": timing, "overflow": ratios["overflow"], "temperature_peak_k": ratios["temperature_peak_k"]}
        feasible = all(c["feasible"] for c in power_cases.values())
        constraints = {"power_w": {"aggregation": "per-case-seed-mean", "reference": "baseline",
            "max_ratio": task["scoring"]["power_max_ratio"], "per_case": power_cases}}
    result = {"candidate": candidate, "task_digest": task_id, "split": split,
            "reference_digest": digest([refs[k] for k in sorted(refs)]), "outcomes_digest": digest([rows[k] for k in sorted(rows)]),
            "vector": vector, "metric_ratios": ratios, "per_case": per_case, "cells": len(expected),
            "feasible": feasible, "constraints": constraints}
    if timing_mode:
        result["score_components"] = {"timing": timing, "overflow_penalty": penalty,
            "weighted_penalty": task["scoring"]["overflow_penalty"]["lambda"] * penalty,
            "per_case_penalty": penalties}
    return result


def _aggregate_v3(task, candidate, task_id, expected, rows, refs, split):
    from .scoring_v3 import compute
    scored = compute(task, sorted(expected), rows, refs, split)
    power_cases = {c["id"]: _power(task, sorted(k for k in expected if k[0] == c["id"]), rows, refs)
                   for c in task["cases"] if c["split"] == split}
    return {"candidate": candidate, "task_digest": task_id, "split": split, "scoring_version": 3,
            "reference_digest": digest([refs[k] for k in sorted(refs)]), "outcomes_digest": digest([rows[k] for k in sorted(rows)]),
            **scored, "cells": len(expected), "feasible": all(c["feasible"] for c in power_cases.values()),
            "aggregation": task["scoring"]["aggregation"], "dominance": copy.deepcopy(task["scoring"]["dominance"]),
            "constraints": {"power_w": {"aggregation": "per-case-seed-mean", "reference": "baseline",
                "max_ratio": task["scoring"]["power_max_ratio"], "per_case": power_cases}}}


def frontier(rows, epsilon=0.0, *, retain_equivalent=False):
    """Retain feasible points only under one frozen objective set; all minimize."""
    finite(epsilon, "epsilon", 0)
    if not rows:
        return []
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Held-out results cannot drive the evolution archive")
    if len({(r["task_digest"], r["reference_digest"]) for r in rows}) != 1:
        raise ValueError("Cannot mix contracts or references in the archive")
    objectives = OBJECTIVES if set(rows[0]["vector"]) == set(OBJECTIVES) else TIMING_OBJECTIVES
    if set(rows[0]["vector"]) not in (set(OBJECTIVES), set(TIMING_OBJECTIVES)):
        raise ValueError("Unsupported objective vector")
    versions = {r.get("scoring_version") for r in rows}
    if versions not in ({None}, {3}):
        raise ValueError("Cannot mix/unknown scoring versions in the archive")
    if versions == {3}:
        from .scoring_v3 import validate_dominance
        if objectives != OBJECTIVES or epsilon or len({digest(r.get("dominance")) for r in rows}) != 1:
            raise ValueError("Scoring v3 rows share three objectives, one dominance model and no epsilon boxes")
        dominance = validate_dominance(rows[0]["dominance"])
    by_candidate = {}
    for row in rows:
        if set(row["vector"]) != set(objectives) or type(row.get("feasible")) is not bool:
            raise ValueError("Requires matching objectives and explicit feasibility")
        for m in objectives:
            finite(row["vector"].get(m), m, 0)
        if versions == {3} and dominance.get("seed_uncertainty"):
            if set(row.get("uncertainty", ())) != set(OBJECTIVES):
                raise ValueError("Seed-uncertainty dominance requires per-objective uncertainty")
            for m in OBJECTIVES:
                finite(row["uncertainty"][m], m + " uncertainty", 0)
        if row["candidate"] in by_candidate and row != by_candidate[row["candidate"]]:
            raise ValueError("Conflicting candidate evidence")
        by_candidate[row["candidate"]] = row
    if versions == {3}:
        from .scoring_v3 import retain
        return retain([r for r in by_candidate.values() if r["feasible"]], dominance, retain_equivalent)
    boxes = {}
    for row in sorted(by_candidate.values(), key=lambda r: r["candidate"]):
        if not row["feasible"]:
            continue
        v = tuple(row["vector"][m] for m in objectives)
        box = tuple(math.floor(x / epsilon) for x in v) if epsilon else v
        boxes.setdefault(box, []).append(row)
    return [row for b in sorted(boxes) if not any(
        all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))
        for a in boxes if a != b) for row in (boxes[b] if retain_equivalent else boxes[b][:1])]


def heldout_report(task, candidates, reference, selection_digest, split="test"):
    """Offline held-out vectors for a caller-supplied candidate selection.

    Pure function of verified held-out outcomes: it never reads or writes the archive.
    candidates maps candidate digest -> its verified outcomes on the held-out split."""
    if split not in ("test", "auxiliary"):
        raise ValueError("Held-out report requires a non-training split")
    if not isinstance(selection_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", selection_digest):
        raise ValueError("Held-out report requires the frozen selection digest")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Held-out report requires a nonempty frozen candidate set")
    names = sorted(candidates)
    rows = [aggregate(task, c, candidates[c], reference, split) for c in names]
    if len({r["reference_digest"] for r in rows}) != 1:
        raise ValueError("Held-out candidates must share one reference")
    # Frontier checks training-only provenance; use copies solely to compute an
    # offline diagnostic, never to admit held-out measurements into an archive.
    kept = frontier([dict(r, split="train") for r in rows], retain_equivalent=True)
    return {"task_digest": rows[0]["task_digest"], "split": split, "reference_digest": rows[0]["reference_digest"],
            "scoring_version": rows[0].get("scoring_version", 2), "selection_digest": selection_digest,
            "selection_frozen_before_evaluation": None,
            "selection_provenance": "Caller-supplied digest; offline outcomes cannot prove pre-registration",
            "candidates": names, "candidates_digest": freeze_selection(names),
            "selection_binds_candidates": selection_digest == freeze_selection(names), "archive_used": False,
            "vectors": {r["candidate"]: r["vector"] for r in rows},
            "feasible": {r["candidate"]: r["feasible"] for r in rows},
            "nondominated": [r["candidate"] for r in kept], "rows": rows}


def freeze_selection(candidates):
    """Recommended pre-registered selection digest: commit it before submitting held-out jobs."""
    names = sorted(candidates)
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[0-9a-f]{64}", str(c)) for c in names):
        raise ValueError("Selection requires unique candidate digests")
    return digest({"heldout_selection": names})
