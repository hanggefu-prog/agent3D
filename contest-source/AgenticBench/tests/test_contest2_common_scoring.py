"""Replay the 28 frozen native measurements through the contest matrix scorer."""
import statistics

import pytest

from contest2 import scoring
from grt_bench.common import read
from local_bench.matrix import LocalEvaluationMatrix
from local_bench.model import EvaluationResult
from test_contest2_sampling import matrix_fixture, native_metrics


@pytest.mark.parametrize("baseline,count", [("binary_loss", 13), ("loss_decoder", 15)])
def test_five_case_baselines_score_equally_without_rewriting_native_results(tmp_path, baseline, count):
    table = scoring.reference_table()
    cases = list(table["cases"])
    path, spec = matrix_fixture(tmp_path, baseline=baseline, cases=cases)
    matrix = LocalEvaluationMatrix(path, spec)
    native_results = []

    def replay(task, candidate, source, output, **options):
        case = output.parent.parent.name
        seed = str(options["seed_schedule"]["partition"])
        measured = table["cases"][case]["measurements"][baseline][seed]
        metrics = native_metrics(case, 0)
        metrics.update(combined_score=0,
                       final_global_wns_setup_ns=measured["wns_ns"],
                       final_global_tns_setup_ns=measured["tns_ns"])
        native_results.append(metrics)
        return EvaluationResult(metrics=metrics, artifacts={})

    result = matrix.evaluate(tmp_path, tmp_path, tmp_path / "result", transport=replay)
    assert result.metrics["runs_successfully"] == 1
    assert result.metrics["matrix"]["succeeded_cells"] == count
    assert result.metrics["combined_score"] == pytest.approx(0, abs=1e-10)
    assert all(row["combined_score"] == 0 for row in native_results)
    assert all("scoring_reference" not in row for row in native_results)
    for cell in result.metrics["matrix"]["cells"]:
        published = read(tmp_path / "result/cells" / cell["case"] / cell["seed"] / "cell.json")
        assert published["score"] == cell["metrics"]["combined_score"]
        assert published["metrics"]["native_paired_combined_score"] == 0


def test_denominators_come_from_fixed_seed_means_and_remain_positive():
    table = scoring.reference_table()
    alpha = table["probability_weight"]
    assert alpha < 0  # Five-case equality requires extrapolation, not a convex mix.
    for case, row in table["cases"].items():
        for name in ("wns", "tns"):
            means = {b: statistics.mean(max(0, -m[name + "_ns"]) for m in seeds.values())
                     for b, seeds in row["measurements"].items()}
            reference = row["reference_" + name + "_deficit_ns"]
            assert reference > 0
            assert reference == pytest.approx(alpha * means["binary_loss"] + (1-alpha) * means["loss_decoder"])
        assert len(row["measurements"]["binary_loss"]) == (1 if case == "ariane133-ng45" else 3)
        assert len(row["measurements"]["loss_decoder"]) == 3


def test_common_score_preserves_weights_and_monotonicity():
    for case in scoring.reference_table()["cases"]:
        original = native_metrics(case, 0)
        for name, expected_gain in (("wns", 4), ("tns", 6)):
            improved = dict(original)
            improved["final_global_" + name + "_setup_ns"] *= 0.9
            assert scoring.score_metrics(case, improved)["combined_score"] == pytest.approx(expected_gain)
        invalid = dict(original, final_global_wns_setup_ns=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            scoring.score_metrics(case, invalid)
        with pytest.raises(ValueError, match="verified"):
            scoring.score_metrics(case, dict(original, runs_successfully=0))


def test_matrix_pins_common_reference_and_rejects_mid_run_changes(tmp_path, monkeypatch):
    reference = tmp_path / "common.json"
    reference.write_bytes(scoring.REFERENCE_PATH.read_bytes())
    monkeypatch.setattr(scoring, "REFERENCE_PATH", reference)
    path, spec = matrix_fixture(tmp_path / "inputs")
    matrix = LocalEvaluationMatrix(path, spec)
    assert str(reference) in matrix.inventory
    matrix.assert_unchanged()
    reference.write_text("{}")
    with pytest.raises(ValueError, match="changed during search"):
        matrix.assert_unchanged()
