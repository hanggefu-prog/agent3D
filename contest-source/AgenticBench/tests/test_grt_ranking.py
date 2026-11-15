"""Ranking metrics, noise-floor estimation and their CLIs, on synthetic data."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.ranking import (area_under_curve, bootstrap_ci, campaign_summary, hv_curve, hypervolume,
                              mean_log_score, percentile)
from grt_bench.contract import METRICS, OBJECTIVES
from scripts import estimate_noise_floor, rank_campaigns
from test_grt_scoring_v3 import outcomes, v3_task


def admission(t, o, T, feasible=True, **cost):
    return {"vector": dict(zip(OBJECTIVES, (t, o, T))), "feasible": feasible, **({"cost": cost} if cost else {})}


class HypervolumeTests(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(hypervolume([(0.5, 0.5, 0.5)]), 0.125)
        # Union of two boxes 0.5*0.5*1 + 1*0.5*0.5 minus their 0.5^3 overlap... computed exactly:
        self.assertAlmostEqual(hypervolume([(0.5, 0.5, 0), (0, 0.5, 0.5)]), 0.25 + 0.25 - 0.125)
        self.assertAlmostEqual(hypervolume([(0.5, 0.5, 0.5), (0.75, 0.75, 0.75)]), 0.125)  # dominated adds 0
        self.assertAlmostEqual(hypervolume([(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]), 0.125)
        self.assertAlmostEqual(hypervolume([(0.9, 0.5, 0.5), (0.5, 0.9, 0.5)]), 2 * .1 * .5 * .5 - .1 * .1 * .5)
        self.assertEqual(hypervolume([]), 0.0)
        # Outside (or on) the reference on any axis dominates nothing inside the box.
        self.assertEqual(hypervolume([(1.0, 0.5, 0.5), (0.2, 1.3, 0.1), (2, 2, 2)]), 0.0)
        self.assertAlmostEqual(hypervolume([(0.5, 0.5, 0.5), (1.0, 0.0, 0.0)]), 0.125)
        self.assertAlmostEqual(hypervolume([{"timing": .5, "overflow": .5, "temperature_peak_k": .5}],
                                           reference=(2, 2, 2)), 1.5 ** 3)

    def test_matches_monte_carlo_free_grid_count(self):
        pts = [(.2, .6, .7), (.5, .3, .4), (.8, .1, .9), (.4, .4, .2)]
        n = 40  # cell centres on a regular grid; exact for these 0.1-aligned coordinates
        count = sum(any(all(c > p for c, p in zip(((i + .5) / n, (j + .5) / n, (k + .5) / n), q)) for q in pts)
                    for i in range(n) for j in range(n) for k in range(n))
        self.assertAlmostEqual(hypervolume(pts), count / n ** 3, places=12)
        self.assertEqual(hypervolume(pts), hypervolume(list(reversed(pts))))

    def test_rejects_malformed(self):
        for bad in ([(0.5, 0.5)], [(0.5, math.nan, 0.5)], [{"timing": .5}], [(True, .5, .5)]):
            with self.assertRaises(ValueError): hypervolume(bad)


class CurveTests(unittest.TestCase):
    def test_curve_counts_feasible_points_and_costs(self):
        campaign = [admission(1, 1, 1), admission(.5, .5, .5, feasible=False), admission(.5, .5, .5),
                    admission(.9, .9, .9)]
        self.assertEqual(hv_curve(campaign), [(0, 0), (1, 0), (2, 0), (3, .125), (4, .125)])
        timed = [admission(.5, .5, .5, wall_seconds=10, evaluations=3), admission(.25, .5, .5, wall_seconds=30)]
        self.assertEqual(hv_curve({"admissions": timed}, "wall_seconds"), [(0, 0), (10, .125), (40, .1875)])
        self.assertEqual(hv_curve(timed)[-1][0], 4)
        with self.assertRaisesRegex(ValueError, "tokens"): hv_curve(timed, "tokens")
        with self.assertRaises(ValueError): hv_curve([{"vector": [1, 1, 1]}])

    def test_final_ranking_uses_the_same_budget_cap_as_auc(self):
        late = [admission(1, 1, 1) for _ in range(9)] + [admission(.1, .1, .1)]
        early = [admission(.5, .5, .5)]
        result = campaign_summary({"late": [late], "early": [early]}, limit=1, resamples=10)
        self.assertEqual(result["ranking"]["final_hypervolume"], ["early", "late"])
        self.assertEqual(result["agents"]["late"]["final_hypervolume"]["mean"], 0)
        self.assertEqual(result["agents"]["early"]["final_hypervolume"]["mean"], .125)
        # The event at the boundary is visible; a fractional budget excludes it.
        timed = [admission(.25, .25, .25, wall_seconds=2)]
        self.assertEqual(campaign_summary({"a": [timed]}, budget="wall_seconds", limit=1.5,
                                         resamples=10)["agents"]["a"]["final_hypervolume"]["mean"], 0)
        self.assertGreater(campaign_summary({"a": [timed]}, budget="wall_seconds", limit=2,
                                           resamples=10)["agents"]["a"]["final_hypervolume"]["mean"], 0)

    def test_area_under_step_curve(self):
        curve = [(0, 0), (10, .125), (40, .1875)]
        self.assertAlmostEqual(area_under_curve(curve, 40), (30 * .125) / 40)
        self.assertAlmostEqual(area_under_curve(curve, 80), (30 * .125 + 40 * .1875) / 80)  # stopped early
        self.assertAlmostEqual(area_under_curve(curve, 20), (10 * .125) / 20)
        with self.assertRaises(ValueError): area_under_curve(curve, 0)


class BootstrapTests(unittest.TestCase):
    def test_deterministic_seeded_percentile_ci(self):
        values = [1.0, 2.0, 4.0, 8.0, 3.0]
        first = bootstrap_ci(values, resamples=2000, seed=11)
        self.assertEqual(first, bootstrap_ci(values, resamples=2000, seed=11))
        spread = [i ** 1.5 for i in range(12)]
        self.assertNotEqual(bootstrap_ci(spread, resamples=500, seed=11), bootstrap_ci(spread, resamples=500, seed=12))
        self.assertTrue(min(values) <= first[0] <= sum(values) / 5 <= first[1] <= max(values))
        self.assertEqual(bootstrap_ci([3.0], resamples=50), [3.0, 3.0])
        self.assertEqual(percentile([0, 10], .25), 2.5)
        for bad in ([], [math.inf]):
            with self.assertRaises(ValueError): bootstrap_ci(bad)

    def test_mean_log_score(self):
        r = mean_log_score([0.5, 2.0, 1.0], resamples=500, seed=3)
        self.assertAlmostEqual(r["mean"], 0.0)
        self.assertAlmostEqual(r["geometric_mean"], 1.0)
        self.assertLessEqual(r["ci_geometric"][0], 1.0)
        self.assertEqual(r, mean_log_score([0.5, 2.0, 1.0], resamples=500, seed=3))
        with self.assertRaises(ValueError): mean_log_score([1.0, 0.0])

    def test_campaign_summary(self):
        good = [[admission(1, 1, 1), admission(.5, .5, .5)], [admission(.5, .5, .5)],
                [admission(1, 1, 1), admission(1, 1, 1), admission(.6, .6, .6)]]
        poor = [[admission(.9, .9, .9)], [admission(1, 1, 1), admission(.9, .9, .9)]]
        s = campaign_summary({"poor": poor, "good": good}, resamples=200, seed=1)
        self.assertEqual(s["budget_limit"], 3)
        self.assertEqual(s["ranking"]["final_hypervolume"], ["good", "poor"])
        self.assertEqual(s["ranking"]["hv_auc"], ["good", "poor"])
        hv = s["agents"]["good"]["final_hypervolume"]
        self.assertAlmostEqual(hv["mean"], (.125 + .125 + .4 ** 3) / 3)
        self.assertGreater(hv["std"], 0)
        self.assertAlmostEqual(s["agents"]["good"]["hv_auc"]["values"][1], 2 / 3 * .125)  # zero until budget 1
        self.assertEqual(len(s["warnings"]), 1)  # poor has 2 campaigns
        self.assertEqual(s, campaign_summary({"good": good, "poor": poor}, resamples=200, seed=1))
        json.dumps(s, allow_nan=False)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dump(self, name, value):
        (self.root / name).write_text(json.dumps(value))
        return str(self.root / name)

    def test_rank_campaigns_cli(self):
        files = [self.dump("a%d.json" % i, {"admissions": [admission(1 - .1 * i, .5, .5, evaluations=2)]})
                 for i in range(3)]
        b = self.dump("b.json", [admission(.9, .9, .9)])
        scores = self.dump("scores.json", {"scores": [.9, 1.1, .8]})
        out = self.root / "out/summary.json"
        result = rank_campaigns.main(["--agent", "a"] + files + ["--agent", "b", b, "--scores", "a", scores,
                                      "--resamples", "100", "--output", str(out)])
        self.assertEqual(json.loads(out.read_text()), json.loads(json.dumps(result)))
        self.assertEqual(result["ranking"]["final_hypervolume"], ["a", "b"])
        self.assertIn("local_scores", result)
        process = subprocess.run([sys.executable, str(ROOT / "scripts/rank_campaigns.py"), "--agent", "b", b,
                                  "--resamples", "10"], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(process.stdout)["agents"]["b"]["campaigns"], 1)

    def test_noise_floor_estimation(self):
        task = v3_task()
        runs = []
        for r, wiggle in enumerate((-1, 0, 1, 0.5, -0.5)):
            runs.append(outcomes(task, "b" * 64, bp_fe={"temperature_peak_k": 328.15 + 0.1 * wiggle,
                                                        "wns_deficit_ns": 0.2 + 0.01 * wiggle}))
        result = estimate_noise_floor.estimate(task, runs, synthetic=True)
        self.assertEqual(result["repeats"], 5)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["noise_floor"]["overflow"], 0.0)
        # Geometric over 2 equal cases: temperature log ratio ~ 0.5 * 0.1 * wiggle / 11.
        self.assertAlmostEqual(result["noise_floor"]["temperature_peak_k"],
                               0.5 * 0.1 / 11 * math.sqrt(2.5 / 4), places=5)
        self.assertGreater(result["noise_floor"]["timing"], 0)
        self.assertEqual(set(result["metric_sigma"]), set(METRICS))
        # The pasted dominance block validates as a v3 contract.
        from grt_bench.contract import validate
        validate(v3_task(dominance=result["dominance"]))
        files = [self.dump("run%d.json" % i, {"outcomes": run}) for i, run in enumerate(runs[:2])]
        out = self.root / "nf.json"
        cli = estimate_noise_floor.main(["--synthetic", "--task", self.dump("task.json", task), "--output", str(out)] + files)
        self.assertEqual(len(cli["warnings"]), 1)
        self.assertEqual(json.loads(out.read_text())["noise_floor"], cli["noise_floor"])
        seeds = estimate_noise_floor.estimate(task, runs[:1], across_seeds=True, synthetic=True)
        self.assertEqual(seeds["repeats"], 3)
        with self.assertRaises(ValueError): estimate_noise_floor.estimate(task, runs[:1], synthetic=True)
        v2 = dict(task, scoring={"timing_weights": task["scoring"]["timing_weights"], "power_max_ratio": 1.1})
        with self.assertRaisesRegex(ValueError, "v3"): estimate_noise_floor.estimate(v2, runs, synthetic=True)
        with self.assertRaises(ValueError): estimate_noise_floor.estimate(task, [runs[0], runs[1][:-1]], synthetic=True)


if __name__ == "__main__": unittest.main()
