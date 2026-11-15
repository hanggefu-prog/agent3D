from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from local_bench.metrics import extract_metrics
from local_bench.evaluation import _write_resolved_config
from local_bench.model import load_candidate, load_task
from local_bench.search_api import (
    CandidateFeedback,
    CandidateProposal,
    FirstImprovementRunner,
    SearchProblem,
)


def _write_run(
    root: Path,
    wns: float,
    tns: float,
    *,
    passes: int = 2,
    endpoint_envelope: bool = False,
) -> None:
    partition = root / "20_partition"
    placement = root / "30_placement"
    opt = root / "60_opt"
    partition.mkdir(parents=True)
    placement.mkdir(parents=True)
    opt.mkdir(parents=True)
    (partition / "partition_metrics.json").write_text(
        json.dumps(
            {
                "instance_count": 100,
                "top_count": 50,
                "bottom_count": 50,
                "cut_net_count": 10,
                "cut_pin_count": 20,
                "timing_weighted_cut": 7.5,
                "area_balance_error": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (placement / "metrics.json").write_text(
        json.dumps(
            {
                "total_object_origin_hpwl_um": 1000.0,
                "cross_die_local_object_origin_hpwl_um": 100.0,
                "dies": {
                    "bottom": {"missing_connection_object_count": 0},
                    "top": {"missing_connection_object_count": 0},
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        {"global_after": {"wns_setup": wns - 0.1, "tns_setup": tns - 1.0}},
        {"global_after": {"wns_setup": wns, "tns_setup": tns}},
    ][:passes]
    (opt / "metrics.json").write_text(
        json.dumps(
            {
                "status": "success",
                "algorithm": (
                    "openroad-coordinated-passive-boundary"
                    if endpoint_envelope
                    else "openroad-coordinated-passive-cutpoint"
                ),
                "boundary_budget_model": (
                    "passive-endpoint-pin-envelope"
                    if endpoint_envelope
                    else "passive-cutpoint"
                ),
                "passes_run": passes,
                "iterations_run": passes,
                "die_passes": {"bottom": passes, "top": passes},
                "pass_records": records,
                "initial_global": {"wns_setup": -2.0, "tns_setup": -200.0},
                "final_global": {
                    "wns_setup": wns,
                    "tns_setup": tns,
                    "sta_completeness_checks_passed": 1.0,
                    "passive_hbt_rc": 1.0,
                },
                "final_boundary_coverage": {
                    "total": 12,
                    "sta_unconstrained_endpoints": 0,
                },
                "timing_validity": {
                    "authoritative_timing_view": "continuous-passive-hbt",
                    "graph_completeness": "passed",
                    "die_internal_parasitics_in_global_sta": True,
                    "hbt_is_passive_rc_in_authoritative_global_sta": True,
                    "remote_die_load_visible_to_source_driver": True,
                    "passive_cutpoint_query_is_diagnostic_only": False,
                    "passive_cutpoint_query_exact": not endpoint_envelope,
                    "unclocked_or_unconstrained_endpoints": 0,
                    "missing_liberty_cells": 0,
                    "combinational_loops": 0,
                },
            }
        ),
        encoding="utf-8",
    )


class StaCodesignMetricsTest(unittest.TestCase):
    def test_weighted_final_sta_cost_allows_wns_tns_tradeoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference"
            candidate = root / "candidate"
            _write_run(reference, -1.0, -100.0)
            _write_run(candidate, -0.5, -125.0)
            metrics = extract_metrics(
                "tp_gnn_placement_sta", candidate, reference
            )
            expected_cost = 0.60 * 0.5 + 0.40 * 1.25
            self.assertAlmostEqual(metrics["normalized_sta_cost"], expected_cost)
            self.assertAlmostEqual(metrics["combined_score"], 1.0 / expected_cost)
            self.assertEqual(metrics["opt_passes"], 2.0)

    def test_rejects_a_one_pass_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _write_run(run, -1.0, -100.0, passes=1)
            with self.assertRaisesRegex(ValueError, "exactly two OPT passes"):
                extract_metrics("tp_gnn_placement_sta", run, None)

    def test_accepts_audited_endpoint_envelope_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _write_run(run, -1.0, -100.0, endpoint_envelope=True)
            metrics = extract_metrics("tp_gnn_placement_sta", run, None)
            self.assertEqual(metrics["combined_score"], 1.0)

    def test_fixed_partition_placement_promotion_uses_the_same_two_pass_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _write_run(run, -0.75, -80.0)
            metrics = extract_metrics("placement_two_pass_sta", run, None)
            self.assertEqual(metrics["opt_passes"], 2.0)
            self.assertEqual(metrics["final_global_wns_setup_ns"], -0.75)
            self.assertEqual(metrics["final_global_tns_setup_ns"], -80.0)

    def test_rejects_candidate_reference_boundary_model_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference"
            candidate = root / "candidate"
            _write_run(reference, -1.0, -100.0, endpoint_envelope=True)
            _write_run(candidate, -1.0, -100.0)
            with self.assertRaisesRegex(ValueError, "boundary models differ"):
                extract_metrics("tp_gnn_placement_sta", candidate, reference)

    def test_source_only_placement_uses_frozen_density_without_candidate_file(self) -> None:
        spec = load_task(Path("evaluation_profiles/placement_proxy"))
        candidate = load_candidate(spec.initial_candidate)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "result/resolved.yaml"
            resolved = _write_resolved_config(
                spec,
                spec.initial_candidate,
                target,
                candidate,
            )
            self.assertEqual(resolved["placement"]["target_density"], 0.6)


class _SequenceBackend:
    name = "sequence"

    def __init__(self) -> None:
        self.index = 0
        self.feedback = []

    def ask(self, _context):
        self.index += 1
        return [CandidateProposal(str(self.index), "program-{}".format(self.index))]

    def tell(self, feedback):
        self.feedback.extend(feedback)

    def done(self):
        return False


class _ScoreEvaluator:
    def __init__(self, scores):
        self.scores = iter(scores)

    def evaluate_proposal(self, proposal):
        score = next(self.scores)
        return CandidateFeedback(
            proposal_id=proposal.proposal_id,
            program_sha256=hashlib.sha256(proposal.program.encode()).hexdigest(),
            status="accepted",
            score=score,
        )


class FirstImprovementTest(unittest.TestCase):
    def test_stops_on_first_strict_local_improvement(self) -> None:
        baseline = CandidateFeedback(
            proposal_id="baseline",
            program_sha256="0" * 64,
            status="accepted",
            score=1.0,
        )
        backend = _SequenceBackend()
        result = FirstImprovementRunner().run(
            backend,
            _ScoreEvaluator([0.9, 1.0, 1.01, 1.2]),
            SearchProblem("local", "local proxy", "baseline"),
            incumbent=baseline,
            max_evaluations=4,
        )
        self.assertEqual(result.stop_reason, "first_improvement")
        self.assertEqual(len(result.history), 3)
        self.assertEqual(result.incumbent.score, 1.01)
        self.assertEqual(len(backend.feedback), 3)


if __name__ == "__main__":
    unittest.main()
