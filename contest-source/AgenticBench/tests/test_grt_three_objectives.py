"""Three-objective scoring and power eligibility, using synthetic paired results."""
import copy
import math
import unittest

from grt_bench.common import digest
from grt_bench.contract import METRICS, OBJECTIVES, matrix, validate
from grt_bench.pareto import aggregate, frontier
from test_grt_matrix import task_fixture


class ThreeObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.task = task_fixture()
        self.candidate, self.baseline = "c" * 64, "b" * 64
        metrics = dict(zip(METRICS, (9, 99, 10, 320, 10)))
        self.refs = [{**p, "status": "verified", "metrics": dict(metrics)}
                     for p in matrix(self.task, self.baseline)]
        self.rows = [{**copy.deepcopy(r), "candidate": self.candidate} for r in self.refs]

    def score(self):
        return aggregate(self.task, self.candidate, self.rows, self.refs)

    def test_three_objectives_keep_five_raw_metrics(self):
        row = self.score()
        self.assertEqual(row["vector"], dict.fromkeys(OBJECTIVES, 1.0))
        self.assertEqual(set(row["metric_ratios"]), set(METRICS))
        self.assertTrue(row["feasible"])
        for raw in self.rows:
            raw["metrics"].update(wns_deficit_ns=4, tns_deficit_ns=199)
        # Paired ratios are 0.5 and 2.0, despite very different raw ns scales.
        self.assertAlmostEqual(self.score()["vector"]["timing"], .7 * .5 + .3 * 2)
        self.assertEqual(self.score(), aggregate(self.task, self.candidate,
                         list(reversed(self.rows)), list(reversed(self.refs))))

    def test_timing_uses_paired_ratios_then_seed_and_case_weights(self):
        self.task["cases"][0]["weight"] = 2
        for i, (raw, ref) in enumerate(zip(self.rows, self.refs)):
            for r in (raw, ref): r["task_digest"] = validate(self.task)
            ref["metrics"].update(wns_deficit_ns=i + 1, tns_deficit_ns=2 * i + 1)
            raw["metrics"].update(wns_deficit_ns=5, tns_deficit_ns=11)
        case_scores = []
        for ids in (range(3), range(3, 6)):
            case_scores.append(sum(.7 * 6 / (i + 2) + .3 * 12 / (2 * i + 2) for i in ids) / 3)
        self.assertAlmostEqual(self.score()["vector"]["timing"], (2 * case_scores[0] + case_scores[1]) / 3)

    def test_power_boundary_and_no_cross_case_compensation(self):
        for raw in self.rows:
            raw["metrics"]["power_w"] = 11
        self.assertTrue(self.score()["feasible"])
        for raw in self.rows:
            raw["metrics"]["power_w"] = 11.1 if raw["case"] == "bp" else 0
        row = self.score()
        self.assertFalse(row["feasible"])
        self.assertEqual(frontier([row]), [])
        self.assertAlmostEqual(row["constraints"]["power_w"]["per_case"]["bp"]["excess_w"], .1)
        self.assertTrue(row["constraints"]["power_w"]["per_case"]["bp_fe"]["feasible"])

    def test_power_uses_mean_watts_not_mean_ratios_or_worst_seed(self):
        for i, (raw, ref) in enumerate(zip(self.rows, self.refs)):
            # Mean reference = 100 W; mean candidate = 110 W. The first seed
            # is far over its own cap, so this distinguishes a case mean cap.
            ref["metrics"]["power_w"] = (1, 100, 199)[i % 3]
            raw["metrics"]["power_w"] = 110
        self.assertTrue(self.score()["feasible"])
        self.assertGreater(self.score()["metric_ratios"]["power_w"], 1.1)

    def test_zero_power_reference_has_no_floor_allowance(self):
        for raw, ref in zip(self.rows, self.refs):
            raw["metrics"]["power_w"] = ref["metrics"]["power_w"] = 0
        self.assertTrue(self.score()["feasible"])
        self.rows[0]["metrics"]["power_w"] = 1e-20
        self.assertFalse(self.score()["feasible"])
        digest(self.score())  # No infinite ratios in the power constraint report.

    def test_raw_power_is_checked_even_with_forged_feasibility(self):
        for raw in self.rows:
            raw["metrics"]["power_w"] = 12
            raw["feasible"] = True
        self.assertFalse(self.score()["feasible"])

    def test_power_is_not_a_fourth_pareto_axis(self):
        first = self.score()
        for raw in self.rows: raw["metrics"]["power_w"] = 11
        second = self.score()
        second["candidate"] = "d" * 64
        self.assertEqual(first["vector"], second["vector"])
        self.assertEqual(len(frontier([first, second], retain_equivalent=True)), 2)
        second["feasible"] = False
        second["vector"] = dict.fromkeys(OBJECTIVES, 0)
        self.assertEqual(frontier([first, second]), [first])

    def test_frontier_rejects_old_vectors_and_missing_feasibility(self):
        for mutation in (lambda r: r.update(vector=dict.fromkeys(METRICS, 1)),
                         lambda r: r.pop("feasible"), lambda r: r.update(feasible=1)):
            row = self.score(); mutation(row)
            with self.assertRaises(ValueError): frontier([row])

    def test_scoring_is_required_validated_and_part_of_task_identity(self):
        for field, value in (("schema_version", 1), ("objectives", list(METRICS)), ("scoring", {})):
            task = copy.deepcopy(self.task); task[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): validate(task)
        for weights in ({"wns_deficit_ns": 1}, {"wns_deficit_ns": .7, "tns_deficit_ns": .4},
                        {"wns_deficit_ns": -1, "tns_deficit_ns": 2},
                        {"wns_deficit_ns": True, "tns_deficit_ns": 0},
                        {"wns_deficit_ns": math.nan, "tns_deficit_ns": .3}):
            task = copy.deepcopy(self.task); task["scoring"]["timing_weights"] = weights
            with self.subTest(weights=weights), self.assertRaises(ValueError): validate(task)
        for ratio in (.99, True, math.inf):
            task = copy.deepcopy(self.task); task["scoring"]["power_max_ratio"] = ratio
            with self.subTest(ratio=ratio), self.assertRaises(ValueError): validate(task)
        for field, value in (("power_max_ratio", 1.05),
                             ("timing_weights", {"wns_deficit_ns": .5, "tns_deficit_ns": .5})):
            task = copy.deepcopy(self.task); task["scoring"][field] = value
            self.assertNotEqual(validate(task), validate(self.task))
            with self.assertRaisesRegex(ValueError, "contract"):
                aggregate(task, self.candidate, self.rows, self.refs)


if __name__ == "__main__": unittest.main()
