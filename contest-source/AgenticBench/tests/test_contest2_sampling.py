"""The Ariane seed exception is fixed before execution, not selected by score."""
from pathlib import Path
import json

import pytest
import yaml

from contest2.budget import BudgetLedger, BudgetConflict
from contest2.controller import TASK_IDS
from contest2.sampling import case_seed_ids, coordinates, validate_public_matrix
from contest2.scoring import PUBLIC_SPLITS, SEEDS, reference_table
from grt_bench.common import read, sha, write
from local_bench.matrix import LocalEvaluationMatrix
from local_bench.model import EvaluationResult, load_task


ROOT = Path(__file__).resolve().parents[1]


def matrix_fixture(tmp_path, *, baseline="binary_loss", case_seeds=None, schema=2, cases=None):
    cases = list(PUBLIC_SPLITS["train"] if cases is None else cases)
    required = case_seeds or {case: list(case_seed_ids(baseline, case)) for case in cases}
    spec = load_task(ROOT / "local_tasks" / TASK_IDS[baseline])
    value = {"schema_version": schema, "task": spec.task_id, "cases": cases,
             "seeds": [{"id": str(s), "io": 1, "partition": s, "placement": s, "routing": -1}
                       for s in SEEDS], "cells": []}
    if schema == 2:
        value["case_seeds"] = required
    for case in cases:
        design, pdk = case.rsplit("-", 1)
        upstream = tmp_path / "inputs" / case
        io = upstream / "10_floorplan/io_metrics.json"
        write(io, {"seed": 1})
        common = {"status": "success", "design": design, "integration": "lol", "pdk": pdk}
        write(upstream / "00_input/stage_manifest.json", common)
        write(upstream / "10_floorplan/stage_manifest.json", {**common, "artifacts": {
            "io_metrics": {"path": io.name, "sha256": sha(io), "bytes": io.stat().st_size}}})
        for seed in required[case]:
            schedule = {"io": 1, "partition": int(seed), "placement": int(seed), "routing": -1}
            config = tmp_path / "configs" / case / (seed + ".yaml")
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(yaml.safe_dump({"design": design, "type": "lol", "pdk": pdk,
                "pins": {"seed": 1}, "partition": {"seed": int(seed)},
                "placement": {"seed": int(seed)}, "routing": {"seed": -1}}))
            reference = tmp_path / "references" / case / seed / "run"
            reference.mkdir(parents=True)
            contract = {"task": spec.task_id, "seeds": schedule, "base_config_sha256": sha(config),
                "upstream": {stage: sha(upstream / directory / "stage_manifest.json") for stage, directory in
                             (("input", "00_input"), ("floorplan", "10_floorplan"))}}
            write(reference.parent / "evaluation_result.json", {"metrics": {"runs_successfully": 1},
                "artifacts": {"matrix_reference": True, "matrix_cell_contract": contract}})
            value["cells"].append({"case": case, "seed": seed, "base_config": str(config),
                                  "baseline_run": str(upstream), "reference_run": str(reference)})
    path = tmp_path / "matrix.json"
    write(path, value)
    return path, spec


def native_metrics(case, score):
    reference = reference_table()["cases"][case]
    return {"runs_successfully": 1, "sta_graph_complete": 1,
            "timing_evaluation_boundary": "post-opt-pre-cts", "combined_score": -999,
            "final_global_wns_setup_ns": -(1 - score / 100) * reference["reference_wns_deficit_ns"],
            "final_global_tns_setup_ns": -(1 - score / 100) * reference["reference_tns_deficit_ns"]}


def test_probability_matrix_has_seven_required_cells_and_equal_case_weights(tmp_path):
    path, spec = matrix_fixture(tmp_path)
    matrix = LocalEvaluationMatrix(path, spec)
    validate_public_matrix(matrix, "binary_loss", "train")
    seen = []
    def transport(task, candidate, source, output, **options):
        case = output.parent.parent.name
        seen.append((case, str(options["seed_schedule"]["partition"])))
        score = {"bp_fe-ng45": 0, "bp-ng45": 10, "ariane133-ng45": 40}[case]
        return EvaluationResult(metrics=native_metrics(case, score), artifacts={})
    result = matrix.evaluate(tmp_path, tmp_path, tmp_path / "result", transport=transport)
    assert set(seen) == set(coordinates("binary_loss", PUBLIC_SPLITS["train"]))
    assert result.metrics["matrix"]["expected_cells"] == 7
    assert result.metrics["combined_score"] == pytest.approx((0 + 10 + 40) / 3)
    assert "average 0 across all five" in matrix.prompt()
    assert "'ariane133-ng45': ['1003']" in matrix.prompt()


def test_failure_of_the_only_ariane_seed_invalidates_the_complete_evaluation(tmp_path):
    path, spec = matrix_fixture(tmp_path)
    matrix = LocalEvaluationMatrix(path, spec)
    def transport(task, candidate, source, output, **options):
        failed = output.parent.parent.name == "ariane133-ng45"
        return EvaluationResult(metrics={**native_metrics(output.parent.parent.name, 0),
                                         "runs_successfully": float(not failed)},
                                artifacts={"error_message": "PDN failure"} if failed else {})
    result = matrix.evaluate(tmp_path, tmp_path, tmp_path / "result", transport=transport)
    assert result.metrics["runs_successfully"] == 0
    assert result.metrics["matrix"]["succeeded_cells"] == 6
    assert "combined_score" not in result.metrics


def test_undeclared_missing_cell_is_rejected_before_dispatch(tmp_path):
    path, spec = matrix_fixture(tmp_path)
    value = read(path)
    value["cells"] = value["cells"][:-1]
    write(path, value, replace=True)
    with pytest.raises(ValueError, match="every case/seed"):
        LocalEvaluationMatrix(path, spec)


@pytest.mark.parametrize("change", ["wrong_ariane", "drop_bp_seed"])
def test_participant_cannot_choose_a_different_seven_or_six_cell_policy(tmp_path, change):
    seeds = {case: list(case_seed_ids("binary_loss", case)) for case in PUBLIC_SPLITS["train"]}
    if change == "wrong_ariane":
        seeds["ariane133-ng45"] = ["1001"]
    else:
        seeds["bp-ng45"].remove("1002")
    path, spec = matrix_fixture(tmp_path, case_seeds=seeds)
    matrix = LocalEvaluationMatrix(path, spec)
    with pytest.raises(ValueError, match="policy differs"):
        validate_public_matrix(matrix, "binary_loss", "train")


def test_embedding_retains_all_nine_cells_and_old_probability_policy_is_rejected(tmp_path):
    seeds = {case: [str(seed) for seed in SEEDS] for case in PUBLIC_SPLITS["train"]}
    path, spec = matrix_fixture(tmp_path, baseline="loss_decoder", case_seeds=seeds, schema=1)
    embedding = LocalEvaluationMatrix(path, spec)
    assert len(embedding.cells) == 9
    validate_public_matrix(embedding, "loss_decoder", "train")
    with pytest.raises(ValueError, match="policy differs"):
        validate_public_matrix(embedding, "binary_loss", "train")


def test_qualification_preserves_the_explicit_case_seed_contract(tmp_path):
    path, spec = matrix_fixture(tmp_path)
    matrix = LocalEvaluationMatrix(path, spec)
    def transport(*args, **options):
        return EvaluationResult(metrics={"runs_successfully": 1, "combined_score": 0}, artifacts={})
    matrix.evaluate(tmp_path, tmp_path, tmp_path / "qualified", transport=transport, qualify=True)
    published = read(tmp_path / "qualified/qualified_matrix.json")
    assert published["schema_version"] == 2
    assert published["case_seeds"]["ariane133-ng45"] == ["1003"]
    assert len(published["cells"]) == 7


@pytest.mark.parametrize("baseline,count", [("binary_loss", 7), ("loss_decoder", 9)])
def test_budget_outbox_is_seven_or_nine_cells_but_each_counts_once(tmp_path, baseline, count):
    ledger = BudgetLedger(tmp_path / baseline / "budget.sqlite",
                          model_cost_limit_rmb="15", task_id=baseline, allowed_model="deepseek-flash")
    ledger.accept_matrix("one", "candidate", "a" * 64, baseline=baseline)
    ledger.finish_evaluation("one", "failed", "retained old failure")
    assert len(ledger.pending_jobs("one")) == count
    assert ledger.snapshot()["evaluations_accepted"] == 1
    assert ledger.snapshot()["evaluations_remaining"] == 9
    assert ledger.snapshot()["evaluations"][0]["status"] == "failed"
    other = "binary_loss" if baseline == "loss_decoder" else "loss_decoder"
    with pytest.raises(BudgetConflict, match="Baseline differs"):
        ledger.accept_matrix("one", "candidate", "a" * 64, baseline=other)
