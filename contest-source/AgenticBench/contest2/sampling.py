"""Both GNN baselines use the complete fixed case/seed matrix."""
from .scoring import PUBLIC_SPLITS, SEEDS


SMOKE_CELL = ("bp_fe-ng45", "1003")


def case_seed_ids(baseline, case):
    if baseline not in ("binary_loss", "loss_decoder"):
        raise ValueError("Unknown GNN baseline")
    return tuple(str(value) for value in SEEDS)


def coordinates(baseline, cases):
    return [(case, seed) for case in cases for seed in case_seed_ids(baseline, case)]


def validate_public_matrix(matrix, baseline, split):
    cases = list(PUBLIC_SPLITS[split])
    expected = coordinates(baseline, cases)
    actual = [(row["case"], str(row["seed"])) for row in matrix.cells]
    if matrix.cases != cases or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Official GNN case/seed policy differs: every case requires seeds 1001, 1002 and 1003")


def smoke_matrix(matrix, baseline, destination):
    """Select the fixed smoke cell only after admitting the full public matrix."""
    from grt_bench.common import read, write
    from local_bench.matrix import LocalEvaluationMatrix

    validate_public_matrix(matrix, baseline, "train")
    cell = next(row for row in matrix.cells if (row["case"], row["seed"]) == SMOKE_CELL)
    value = {"schema_version": 1, "task": matrix.spec.task_id,
             "cases": [cell["case"]],
             "seeds": [row for row in matrix.seeds if row["id"] == cell["seed"]],
             "cells": [{key: cell[key] for key in
                        ("case", "seed", "base_config", "baseline_run", "reference_run")}]}
    if destination.exists():
        if read(destination) != value:
            raise ValueError("Existing smoke matrix differs from the fixed cell")
    else:
        write(destination, value)
    return LocalEvaluationMatrix(destination, matrix.spec)
