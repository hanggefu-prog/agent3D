"""Scoring contract v3 (opt-in) and the frozen v2 regression, on synthetic paired results."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.archive import Archive
from grt_bench.candidates import freeze
from grt_bench.common import digest, evaluator_identity, write
from grt_bench.contract import METRICS, OBJECTIVES, matrix, validate
from grt_bench.optimization import define_task
from grt_bench.pareto import aggregate, freeze_selection, frontier, heldout_report
from grt_bench.queue import Queue
from grt_bench.scoring_v3 import dominates, retain
from test_grt_matrix import task_fixture
import test_grt_optimization as opt

FLOORS = {"wns_deficit_ns": 0.001, "tns_deficit_ns": 0.001, "overflow": 1, "temperature_peak_k": 1, "power_w": 1e-06}


def v2_example():
    """A nontrivial frozen v2 input: unequal weights, held-out case, varied seeds."""
    task = task_fixture()
    task["normalization_floors"] = dict(FLOORS)
    task["cases"][0]["weight"] = 2
    task["cases"].append(dict(task["cases"][0], id="held", design="held", split="test", weight=1))
    baseline = "b" * 64
    def run(candidate, split, shift):
        rows = []
        for i, p in enumerate(matrix(task, candidate, split)):
            m = {"wns_deficit_ns": 0.05 * (i + 1) * shift, "tns_deficit_ns": 1.5 * (i % 3 + 1) * shift,
                 "overflow": float(i % 2) * 7 * shift, "temperature_peak_k": 320.0 + i * 0.37 * shift,
                 "power_w": 0.1 + 0.01 * i}
            rows.append({**p, "status": "verified", "metrics": m})
        return rows
    out = {}
    for split in ("train", "test"):
        refs = run(baseline, split, 1.0)
        rows = [aggregate(task, c, run(c, split, s), refs, split)
                for c, s in (("c" * 64, 0.9), ("d" * 64, 1.1), ("e" * 64, 0.9))]
        out[split] = rows
    out["frontier"] = frontier(out["train"])
    out["frontier_equiv"] = frontier(out["train"], retain_equivalent=True)
    out["frontier_eps"] = frontier(out["train"], 0.5)
    return out


def strip_stdev(value):
    if isinstance(value, dict):
        return {k: strip_stdev(v) for k, v in value.items() if k != "stdev"}
    return [strip_stdev(v) for v in value] if isinstance(value, list) else value


# Computed from the unmodified v2 implementation before scoring v3 was added. The last
# ULP of statistics.stdev differs between CPython 3.9 and 3.12 (pre-existing), so the full
# digests list both interpreter variants and everything except "stdev" must match exactly.
V2_FROZEN = {
    "train": {"466035b8fc299d154aad20d1d659b686c5da57ef16c1bc1d98a385e5136e4b0a",
              "2cd6f7682b2cece7808d8c20bba96a8611ae4382b6903f66869108c9b44a8028"},
    "test": {"7fc1e61ed80545fdba53001e87ba440c3ca9e4a72683cb06cd6d2ca4f8eb9dec",
             "5b9d4d7683628bb8b57213a46137b4ec49475395f6fab06c132464b62cd31edc"},
    "frontier": {"c925a0c0a205f5ecacab57c7a7ff125af95b96a95f41d1170bb67435b9af0c80"},
    "frontier_equiv": {"b3c946cc5805c2ddd25d1b3fdbde3a8808e8aa740f027a0597943cb99012d32f"},
    "frontier_eps": {"c925a0c0a205f5ecacab57c7a7ff125af95b96a95f41d1170bb67435b9af0c80"},
}
V2_FROZEN_WITHOUT_STDEV = "c9d8d3aea42662fded931ebc5784966baed03f20284019a46544c3dec41243f9"


class FrozenV2Tests(unittest.TestCase):
    def test_v2_outputs_are_bit_identical(self):
        out = v2_example()
        got = {k: digest(v) for k, v in out.items()}
        self.assertTrue(all(got[k] in V2_FROZEN[k] for k in V2_FROZEN) and set(got) == set(V2_FROZEN),
                        json.dumps(got, indent=1))
        self.assertEqual(digest(strip_stdev(out)), V2_FROZEN_WITHOUT_STDEV)
        self.assertNotIn("scoring_version", out["train"][0])


def v3_scoring(**changes):
    scoring = {"version": 3, "timing_weights": {"wns_deficit_ns": 0.7, "tns_deficit_ns": 0.3},
               "power_max_ratio": 1.1, "temperature": "rise_over_ambient", "aggregation": "geometric",
               "overflow": {"mode": "ratio"}, "dominance": {"mode": "exact"}}
    scoring.update(changes)
    return scoring


def v3_task(**changes):
    task = task_fixture()
    task["normalization_floors"] = dict(FLOORS)
    task["scoring"] = v3_scoring(**changes)
    return task


BASE = dict(zip(METRICS, (0.2, 2.0, 0.0, 328.15, 0.1)))  # 10 K rise over the 318.15 K ambient


def outcomes(task, candidate, split="train", **per_case):
    """Every seed of case c gets BASE updated by per_case[c] (a dict or a list per seed)."""
    rows = []
    for p in matrix(task, candidate, split):
        update = per_case.get(p["case"], {})
        if isinstance(update, list):
            update = update[[s["id"] for s in task["seeds"]].index(p["seed"]["id"])]
        rows.append({**p, "status": "verified", "metrics": dict(BASE, **update)})
    return rows


def row(candidate, timing, overflow, temperature, feasible=True, uncertainty=None, dominance=None):
    r = {"candidate": candidate, "task_digest": "t" * 64, "reference_digest": "r" * 64, "split": "train",
         "scoring_version": 3, "feasible": feasible, "dominance": dominance or {"mode": "exact"},
         "vector": dict(zip(OBJECTIVES, (timing, overflow, temperature)))}
    r["uncertainty"] = dict(zip(OBJECTIVES, uncertainty or (0.0, 0.0, 0.0)))
    return r


class ContractV3Tests(unittest.TestCase):
    def test_v3_is_opt_in_strict_and_changes_identity(self):
        v2 = task_fixture()
        self.assertNotEqual(validate(v3_task()), validate(dict(v2, normalization_floors=dict(FLOORS))))
        validate(v3_task(dominance={"mode": "noise", "k": 2, "noise_floor": dict.fromkeys(OBJECTIVES, 0.01),
                                    "seed_uncertainty": True}))
        validate(v3_task(overflow={"mode": "absolute", "scale": {"bp_fe": 100, "bp": 50}}))
        validate(v3_task(temperature="absolute", aggregation="arithmetic"))
        bad = [{"version": 2}, {"version": 3.0}, {"version": True}, {"temperature": "kelvin"},
               {"aggregation": "harmonic"}, {"overflow": {"mode": "ratio", "scale": {}}},
               {"overflow": {"mode": "absolute", "scale": {"bp_fe": 1}}},
               {"overflow": {"mode": "absolute", "scale": {"bp_fe": 1, "bp": 0}}},
               {"overflow": {"mode": "absolute", "scale": {"bp_fe": 1, "bp": 1, "x": 1}}},
               {"dominance": {"mode": "exact", "k": 1}}, {"dominance": {"mode": "noise", "k": 1}},
               {"dominance": {"mode": "noise", "k": -1, "noise_floor": dict.fromkeys(OBJECTIVES, 0)}},
               {"dominance": {"mode": "noise", "k": 1, "noise_floor": dict.fromkeys(OBJECTIVES[:2], 0)}},
               {"dominance": {"mode": "noise", "k": 1, "noise_floor": dict.fromkeys(OBJECTIVES, -0.1)}},
               {"dominance": {"mode": "noise", "k": 1, "noise_floor": dict.fromkeys(OBJECTIVES, 0),
                              "seed_uncertainty": 1}},
               {"timing_weights": {"wns_deficit_ns": 0.5, "tns_deficit_ns": 0.6}}, {"power_max_ratio": 0.9}]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(v3_task(**change))
        for extra in ({"extra": 1}, {}):
            task = v3_task()
            task["scoring"].update(extra)
            if not extra: task["scoring"].pop("dominance")
            with self.assertRaises(ValueError): validate(task)
        task = v3_task(); task["epsilon"] = 0.1
        with self.assertRaisesRegex(ValueError, "epsilon"): validate(task)
        v2["scoring"]["temperature"] = "rise_over_ambient"  # v3 fields without the version key
        with self.assertRaises(ValueError): validate(v2)


class AggregateV3Tests(unittest.TestCase):
    def setUp(self):
        self.task, self.c, self.b = v3_task(), "c" * 64, "b" * 64

    def score(self, task=None, **per_case):
        task = task or self.task
        return aggregate(task, self.c, outcomes(task, self.c, **per_case), outcomes(task, self.b))

    def test_identity_is_neutral_and_reports_v3_fields(self):
        r = self.score()
        self.assertEqual(r["vector"], dict.fromkeys(OBJECTIVES, 1.0))
        self.assertEqual(r["scoring_version"], 3)
        self.assertEqual(r["uncertainty"], dict.fromkeys(OBJECTIVES, 0.0))
        self.assertEqual(set(r["worst_case"]), set(OBJECTIVES))
        self.assertTrue(r["feasible"])
        digest(r)

    def test_temperature_uses_rise_over_ambient(self):
        # Halving the 10 K rise: (5 + 1) / (10 + 1) instead of 323.15+1 / 328.15+1.
        hot = {"temperature_peak_k": 323.15}
        self.assertAlmostEqual(self.score(bp_fe=hot, bp=hot)["vector"]["temperature_peak_k"], 6 / 11)
        absolute = self.score(v3_task(temperature="absolute"), bp_fe=hot, bp=hot)
        self.assertAlmostEqual(absolute["vector"]["temperature_peak_k"], 324.15 / 329.15)
        for value in (318.15, 318.15 - 1e-7):  # exactly ambient / within the tolerance
            self.assertAlmostEqual(self.score(bp=[{"temperature_peak_k": value}] * 3)["per_case"]["bp"]
                                   ["temperature_peak_k"]["mean"], 1 / 11)
        with self.assertRaisesRegex(ValueError, "ambient"):
            self.score(bp=[{"temperature_peak_k": 318.0}, {}, {}])

    def test_geometric_is_symmetric_and_arithmetic_is_not(self):
        better, worse = {"wns_deficit_ns": 0.1}, {"wns_deficit_ns": 0.4}  # about 0.5x and 2x
        geo = self.score(bp_fe=better, bp=worse)
        w = [(x + .001) / (.2 + .001) for x in (.1, .4)]
        self.assertAlmostEqual(geo["metric_ratios"]["wns_deficit_ns"], math.sqrt(w[0] * w[1]))
        self.assertAlmostEqual(geo["vector"]["timing"], math.sqrt(w[0] * w[1]) ** .7)
        arith = self.score(v3_task(aggregation="arithmetic"), bp_fe=better, bp=worse)
        self.assertAlmostEqual(arith["vector"]["timing"], .7 * (w[0] + w[1]) / 2 + .3)
        self.assertGreater(arith["vector"]["timing"], 1.0)
        self.assertLess(abs(math.log(geo["metric_ratios"]["wns_deficit_ns"])), 0.01)

    def test_arithmetic_absolute_ratio_v3_matches_v2_values(self):
        v2 = task_fixture(); v2["normalization_floors"] = dict(FLOORS)
        v3 = v3_task(temperature="absolute", aggregation="arithmetic")
        change = {"bp_fe": [{"wns_deficit_ns": .1 * (i + 1), "overflow": 3 * i} for i in range(3)],
                  "bp": {"tns_deficit_ns": 1.0, "temperature_peak_k": 330}}
        old = aggregate(v2, self.c, outcomes(v2, self.c, **change), outcomes(v2, self.b))
        new = self.score(v3, **change)
        for o in OBJECTIVES:
            self.assertAlmostEqual(old["vector"][o], new["vector"][o], places=12)
        self.assertNotIn("scoring_version", old)

    def test_geometric_seed_mean_case_weights_and_uncertainty(self):
        self.task["cases"][0]["weight"] = 3
        seeds = [{"overflow": v} for v in (0, 1, 3)]
        r = self.score(bp_fe=seeds)
        logs = [math.log((v + 1) / 1) for v in (0, 1, 3)]
        level = 3 / 4 * statistics_mean(logs)
        self.assertAlmostEqual(r["vector"]["overflow"], math.exp(level))
        se = 3 / 4 * stdev(logs) / math.sqrt(3)
        self.assertAlmostEqual(r["uncertainty"]["overflow"], math.exp(level) * se)
        self.assertEqual(r["worst_case"]["overflow"]["case"], "bp_fe")
        self.assertAlmostEqual(r["worst_case"]["overflow"]["value"], math.exp(statistics_mean(logs)))
        self.assertEqual(r["uncertainty"]["temperature_peak_k"], 0.0)

    def test_absolute_overflow_mode(self):
        task = v3_task(overflow={"mode": "absolute", "scale": {"bp_fe": 100, "bp": 10}})
        reference = outcomes(task, self.b, bp_fe={"overflow": 50}, bp={"overflow": 0})
        r = aggregate(task, self.c, outcomes(task, self.c, bp_fe={"overflow": 60}, bp={"overflow": 0}), reference)
        # log value = (x - r) / scale per cell: 0.1 and 0.0, case-weighted mean 0.05.
        self.assertAlmostEqual(r["vector"]["overflow"], math.exp(0.05))
        self.assertAlmostEqual(r["per_case_objectives"]["bp_fe"]["overflow"], math.exp(0.1))
        neutral = aggregate(task, self.c, outcomes(task, self.c, bp_fe={"overflow": 50}), reference)
        self.assertEqual(neutral["vector"]["overflow"], 1.0)
        tiny = v3_task(overflow={"mode": "absolute", "scale": {"bp_fe": 1e-6, "bp": 1}})
        with self.assertRaisesRegex(ValueError, "overflow"):
            aggregate(tiny, self.c, outcomes(tiny, self.c, bp_fe={"overflow": 60}), outcomes(tiny, self.b))

    def test_power_constraint_unchanged(self):
        self.assertFalse(self.score(bp={"power_w": 0.111})["feasible"])
        self.assertTrue(self.score(bp={"power_w": 0.11})["feasible"])


def statistics_mean(values):
    return sum(values) / len(values)


def stdev(values):
    m = statistics_mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


NOISE = {"mode": "noise", "k": 2.0, "noise_floor": {"timing": 0.002, "overflow": 0.01, "temperature_peak_k": 0.002}}


class NoiseDominanceTests(unittest.TestCase):
    def test_real_round_one_example_is_noise_dominated(self):
        incumbent = row("a" * 64, 1.0, 1.0, 1.0, dominance=NOISE)
        challenger = row("b" * 64, 1.007, 1.088, 0.99998, dominance=NOISE)
        # Exact dominance admits the challenger only because of a 2e-5 temperature gain.
        exact = [dict(r, dominance={"mode": "exact"}) for r in (incumbent, challenger)]
        self.assertEqual([r["candidate"] for r in frontier(exact)], ["a" * 64, "b" * 64])
        # Temperature tolerance 2*sqrt(2)*0.002 = 0.0057 >> 2e-5; overflow 8.8% >> 0.028.
        self.assertEqual([r["candidate"] for r in frontier([incumbent, challenger])], ["a" * 64])
        # A temperature gain beyond the tolerance is a genuine trade-off and is retained.
        real = row("b" * 64, 1.007, 1.088, 0.99, dominance=NOISE)
        self.assertEqual(len(frontier([incumbent, real])), 2)
        # A timing loss beyond the tolerance means the incumbent no longer dominates.
        slower = row("b" * 64, 1.0, 1.088, 0.999, dominance=NOISE)
        self.assertEqual(len(frontier([incumbent, slower])), 1)

    def test_k_zero_is_exact_and_within_noise_rows_coexist(self):
        zero = dict(NOISE, k=0)
        a, b = row("a" * 64, 1.0, 1.0, 1.0, dominance=zero), row("b" * 64, 1.007, 1.088, 0.99998, dominance=zero)
        self.assertEqual(len(frontier([a, b])), 2)
        a, b = row("a" * 64, 1.0, 1.0, 1.0, dominance=NOISE), row("b" * 64, 1.001, 0.999, 1.0, dominance=NOISE)
        self.assertEqual(len(frontier([a, b])), 2)
        same = row("c" * 64, 1.0, 1.0, 1.0, dominance=NOISE)
        self.assertEqual(len(frontier([a, same], retain_equivalent=True)), 2)
        self.assertEqual([r["candidate"] for r in frontier([same, a])], ["a" * 64])

    def test_seed_uncertainty_widens_tolerance(self):
        seed = dict(NOISE, noise_floor=dict.fromkeys(OBJECTIVES, 0.0), seed_uncertainty=True)
        a = row("a" * 64, 1.0, 1.0, 1.0, dominance=seed)
        b = row("b" * 64, 1.02, 1.0, 1.0, dominance=seed, uncertainty=(0.001, 0, 0))
        self.assertEqual(len(frontier([a, b])), 1)
        b = row("b" * 64, 1.02, 1.0, 1.0, dominance=seed, uncertainty=(0.02, 0, 0))
        self.assertEqual(len(frontier([a, b])), 2)
        b.pop("uncertainty")
        with self.assertRaisesRegex(ValueError, "uncertainty"): frontier([a, b])

    def test_cycles_cannot_empty_the_frontier(self):
        cyclic = dict(NOISE, k=1.0, noise_floor=dict.fromkeys(OBJECTIVES, 1 / math.sqrt(2)))  # tol = 1
        rows = [row(c * 64, *v, dominance=cyclic) for c, v in
                (("a", (0, .75, 1.5)), ("b", (1.5, 0, .75)), ("c", (.75, 1.5, 0)))]
        tol = (1.0,) * 3
        # Tolerant dominance without the sum condition would be a cycle a>b>c>a.
        plain = lambda x, y: all(p <= q + 1 for p, q in zip(x, y)) and any(p < q - 1 for p, q in zip(x, y))
        v = [tuple(r["vector"][o] for o in OBJECTIVES) for r in rows]
        self.assertTrue(plain(v[0], v[1]) and plain(v[1], v[2]) and plain(v[2], v[0]))
        self.assertFalse(any(dominates(x, y, tol) for x in rows for y in rows))
        self.assertEqual(len(frontier(rows)), 3)

    def test_order_independence_and_determinism(self):
        import itertools, random
        rng = random.Random(7)
        rows = [row("%064x" % i, *(1 + rng.uniform(-.02, .02) for _ in range(3)), feasible=i != 3,
                    dominance=NOISE) for i in range(7)]
        expected = frontier(rows)
        self.assertTrue(expected)
        for perm in itertools.islice(itertools.permutations(rows), 0, 5040, 97):
            self.assertEqual(frontier(list(perm)), expected)
        self.assertNotIn("%064x" % 3, [r["candidate"] for r in expected])
        self.assertEqual(retain(rows[:1], NOISE, False), rows[:1])

    def test_frontier_rejects_mixed_versions_dominance_and_epsilon(self):
        a, b = row("a" * 64, 1, 1, 1, dominance=NOISE), row("b" * 64, 1, 1, 1)
        with self.assertRaises(ValueError): frontier([a, b])
        old = dict(b); old.pop("scoring_version")
        with self.assertRaises(ValueError): frontier([a, old])
        with self.assertRaisesRegex(ValueError, "epsilon"): frontier([a], 0.1)
        with self.assertRaises(ValueError): frontier([dict(a, dominance={"mode": "noise", "k": 1})])
        with self.assertRaises(ValueError): frontier([dict(a, split="test")])

    def test_aggregate_rows_feed_frontier(self):
        task, b = v3_task(dominance=NOISE), "b" * 64
        refs = outcomes(task, b)
        rows = [aggregate(task, c * 64, outcomes(task, c * 64, bp={"temperature_peak_k": t, "overflow": o}), refs)
                for c, t, o in (("c", 328.15, 0), ("d", 328.1499, 1), ("e", 320.15, 2))]
        self.assertEqual([r["candidate"] for r in frontier(rows)], ["c" * 64, "e" * 64])
        self.assertEqual([r["candidate"] for r in frontier([dict(r, dominance={"mode": "exact"}) for r in rows])],
                         ["c" * 64, "d" * 64, "e" * 64])


class HeldoutTests(unittest.TestCase):
    def setUp(self):
        self.task = v3_task(dominance=NOISE)
        self.task["cases"].append(dict(self.task["cases"][0], id="held", design="held", split="test"))
        self.ref = outcomes(self.task, "b" * 64, "test")
        self.frozen = {c * 64: outcomes(self.task, c * 64, "test", held={"wns_deficit_ns": w})
                       for c, w in (("c", 0.1), ("d", 0.3))}

    def test_report_records_frozen_selection(self):
        sel = freeze_selection(self.frozen)
        report = heldout_report(self.task, self.frozen, self.ref, sel)
        self.assertIsNone(report["selection_frozen_before_evaluation"])
        self.assertTrue(report["selection_binds_candidates"])
        self.assertIn("cannot prove", report["selection_provenance"])
        self.assertEqual(report["selection_digest"], sel)
        self.assertFalse(report["archive_used"])
        self.assertEqual(report["candidates"], ["c" * 64, "d" * 64])
        self.assertEqual(report["nondominated"], ["c" * 64])
        self.assertLess(report["vectors"]["c" * 64]["timing"], 1)
        other = heldout_report(self.task, self.frozen, self.ref, "f" * 64)
        self.assertFalse(other["selection_binds_candidates"])
        digest(report)

    def test_report_rejects_training_split_and_unfrozen_selection(self):
        train = {"c" * 64: outcomes(self.task, "c" * 64)}
        with self.assertRaisesRegex(ValueError, "non-training"):
            heldout_report(self.task, train, outcomes(self.task, "b" * 64), "f" * 64, split="train")
        for sel in (None, "", "F" * 64, "f" * 63):
            with self.assertRaises(ValueError): heldout_report(self.task, self.frozen, self.ref, sel)
        with self.assertRaises(ValueError): heldout_report(self.task, {}, self.ref, "f" * 64)
        with self.assertRaises(ValueError):  # training outcomes are not held-out evidence
            heldout_report(self.task, train, self.ref, "f" * 64)

    def test_v2_tasks_get_exact_heldout_reports(self):
        task = task_fixture()
        task["cases"].append(dict(task["cases"][0], id="held", split="test"))
        frozen = {"c" * 64: outcomes(task, "c" * 64, "test")}
        report = heldout_report(task, frozen, outcomes(task, "b" * 64, "test"), freeze_selection(frozen))
        self.assertEqual(report["scoring_version"], 2)
        self.assertEqual(report["nondominated"], ["c" * 64])


class ArchiveV3Tests(unittest.TestCase):
    """The unchanged host archive retains the v3 noise-aware frontier."""
    receipts, child = opt.OptimizationTests.receipts, opt.OptimizationTests.child

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        for name in (opt.PARTITION, opt.PLACEMENT, opt.NATIVE, opt.PROTECTED):
            path = self.source / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("int value = 1;\n" if name == opt.NATIVE else "value = 1\n")
        self.store = self.root / "candidates"
        self.baseline = freeze(self.source, self.store)
        base = v3_task(dominance=NOISE); base["evaluator_sha256"] = evaluator_identity()
        self.task = define_task(base, self.baseline, opt.PROMPT, self.store)
        self.archive = Archive(self.root, self.task)
        write(self.root / "control/tasks" / (validate(self.task) + ".json"), self.task)
        self.queue = Queue(self.root / "control/queue.sqlite")
        self.queue.register_host("fixture", {"cpu": 8, "memory_gb": 16, "storage_gb": 40, "gpus": [], "build_slots": 1})
        self.resources = {"cpu": 1, "memory_gb": 1, "storage_gb": 1, "gpu_count": 0, "build_slots": 0}
        self.serial = 0

    def test_noise_dominated_child_does_not_enter_the_archive_frontier(self):
        base = [0.2, 2.0, 10.0, 328.15, 0.1]
        self.receipts(self.baseline, base)
        self.archive.ingest(self.baseline)
        marginal, tradeoff = self.child(), self.child()
        self.receipts(marginal, [0.2, 2.0, 11.0, 328.149, 0.1])  # +9% overflow for -0.001 K
        self.receipts(tradeoff, [0.2, 2.0, 11.0, 325.0, 0.1])
        result = self.archive.ingest(marginal)
        self.assertEqual(result["evaluation"]["scoring_version"], 3)
        self.assertFalse(result["non_dominated"])
        self.assertTrue(self.archive.ingest(tradeoff)["non_dominated"])
        frontier_ids = {r["candidate"] for r in self.archive.show()["frontier"]}
        self.assertEqual(frontier_ids, {self.baseline, tradeoff})
        with self.assertRaisesRegex(ValueError, "current training frontier"):
            self.archive.select(marginal, "Noise-dominated")


if __name__ == "__main__": unittest.main()
