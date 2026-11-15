"""Common timing denominators; native measurements and receipts stay unchanged."""

import json
import math
from pathlib import Path

# Fixed paired seeds for every case in both GNN baselines.
SEEDS = (1001, 1002, 1003)
PUBLIC_SPLITS = {
    "train": ("bp_fe-ng45", "bp-ng45", "ariane133-ng45"),
    "validation": ("swerv_wrapper-ng45", "bp_fe-asap7"),
}

REFERENCE_PATH = Path(__file__).with_name("manifests") / "common-timing-reference.json"
MIN_SCORE = -100.0


class PhysicalCellFailure(RuntimeError):
    """A host-confirmed, terminated native physical run, not an intake rejection."""

    def __init__(self, reason, result, evidence):
        super().__init__(reason)
        self.result = result
        self.evidence = evidence


def complete_score(metrics, matrix):
    """Admit a full scored matrix, including explicit physical-failure penalties."""
    if not isinstance(metrics, dict):
        return False
    summary = metrics.get("matrix", {})
    score = metrics.get("combined_score")
    if not isinstance(summary, dict) or isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    rows = summary.get("cells", [])
    if (not math.isfinite(score) or not isinstance(rows, list)
            or summary.get("matrix_sha256") != matrix.identity
            or summary.get("complete") is not True
            or summary.get("qualification") is not False
            or summary.get("expected_cells") != len(matrix.cells)
            or not rows or len(rows) != len(matrix.cells)):
        return False
    expected = {(c["case"], str(c["seed"])) for c in matrix.cells}
    actual = set()
    scores = {}
    for row in rows:
        if not isinstance(row, dict):
            return False
        if not isinstance(row.get("case"), str) or not isinstance(row.get("seed"), str):
            return False
        key = (row.get("case"), row.get("seed"))
        value = row.get("score")
        if (key not in expected or key in actual or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < MIN_SCORE or row.get("status") not in {"succeeded", "penalized"}):
            return False
        if row["status"] == "penalized" and (row.get("valid") is not False or value != MIN_SCORE):
            return False
        actual.add(key)
        scores.setdefault(key[0], []).append(value)
    mean = sum(sum(values) / len(values) for values in scores.values()) / len(scores)
    return actual == expected and math.isclose(score, mean, rel_tol=1e-12, abs_tol=1e-12)


def reference_table():
    return json.loads(REFERENCE_PATH.read_text())


def score_metrics(case, metrics):
    """Apply the common reference only after a verified native cell succeeds."""
    if (metrics.get("runs_successfully") != 1 or metrics.get("sta_graph_complete") != 1
            or metrics.get("timing_evaluation_boundary") != "post-opt-pre-cts"):
        raise ValueError("Common scoring requires verified post-OPT timing")
    table = reference_table()
    if case not in table["cases"]:
        raise ValueError("Missing common timing reference for case: " + case)
    reference = table["cases"][case]
    result = dict(metrics)
    result["native_paired_combined_score"] = metrics["combined_score"]
    result["scoring_reference"] = "common-five-case"
    for name in ("wns", "tns"):
        value = metrics["final_global_" + name + "_setup_ns"]
        denominator = reference["reference_" + name + "_deficit_ns"]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not math.isfinite(denominator) or denominator <= 0):
            raise ValueError("Timing and common reference must be finite with positive denominators")
        result["reference_" + name + "_deficit_ns"] = denominator
        result[name + "_deficit_improvement"] = (denominator - max(0.0, -value)) / denominator
    result["raw_score"] = 100 * (
        0.4 * result["wns_deficit_improvement"] + 0.6 * result["tns_deficit_improvement"])
    if not math.isfinite(result["raw_score"]):
        raise ValueError("Common timing score is not finite")
    result["combined_score"] = max(MIN_SCORE, result["raw_score"])
    result["valid"] = True
    return result


def validate_scoring_references(matrix, baseline):
    """Bind calibration to the admitted raw references, without rewriting them."""
    from grt_bench.common import sha
    table = reference_table()
    for cell in matrix.cells:
        expected = table["cases"][cell["case"]]["measurements"][baseline][str(cell["seed"])]
        summary = Path(cell["reference_run"]) / "reference_summary.json"
        if sha(summary) != expected["reference_summary_sha256"]:
            raise ValueError("Common scoring calibration differs from the native reference: "
                             + cell["case"] + "/" + str(cell["seed"]))
