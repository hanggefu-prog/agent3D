"""The producer and independent consumer must score silicon hotspots, not means."""
from contextlib import ExitStack
import copy
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench import ENDPOINT, THERMAL_STATISTIC
from grt_bench import endpoint
from grt_bench.common import read, records, sha, write
from grt_bench.contract import METRICS, validate
from grt_bench.verify import verify_analysis
from test_grt_matrix import task_fixture


def tile_text(hot_die="top"):
    lines = ["layer_{}_{}_{}_{} {}".format(layer, die, x, y,
             350 if die == hot_die and x == 9 and y == 9 else 320)
             for layer, die in ((0, "top"), (2, "bottom"))
             for x in range(10) for y in range(10)]
    # Package/TIM temperatures must never substitute for a silicon hotspot.
    return "\n".join(lines + ["layer_1_tim_0_0 999", "package 9999"]) + "\n"


def grid(die, *args):
    return {"{}_{}_{}".format(die, x, y): .001 for x in range(10) for y in range(10)}, {
        "diearea_dbu": [0, 0, 100, 100], "dbu_per_micron": 1000}


def thermal_patches(stack):
    stack.enter_context(patch("scripts.workspace_cts_thermal.validate_power",
                             return_value=({"top": {"cell": .1}, "bottom": {"cell": .1}}, {}, 0)))
    stack.enter_context(patch("open3dbench.backend.thermal._lef_sizes", return_value={}))
    stack.enter_context(patch("open3dbench.backend.thermal._grid_power", side_effect=grid))


class PeakTemperatureTests(unittest.TestCase):
    def test_contract_rejects_legacy_endpoint_and_metric(self):
        task = task_fixture()
        validate(task)
        self.assertIn("temperature_peak_k", METRICS)
        self.assertNotIn("temperature_k", METRICS)
        old = copy.deepcopy(task)
        old["endpoint"] = "post-grt-passive-hbt-v1"
        with self.assertRaisesRegex(ValueError, "supported post-GRT endpoint"):
            validate(old)
        old = copy.deepcopy(task)
        old["metrics"][3] = "temperature_k"
        with self.assertRaisesRegex(ValueError, "Raw metrics differ from endpoint"):
            validate(old)
        old = copy.deepcopy(task)
        old["normalization_floors"]["temperature_k"] = old["normalization_floors"].pop("temperature_peak_k")
        with self.assertRaisesRegex(ValueError, "floors"):
            validate(old)

    def test_producer_scores_hotspot_on_either_die_and_keeps_means_diagnostic(self):
        for hot_die in ("top", "bottom"):
            with self.subTest(hot_die=hot_die), tempfile.TemporaryDirectory() as d, ExitStack() as stack:
                root = Path(d)
                hotspot = root / "hotspot"
                (hotspot / "examples/thermal").mkdir(parents=True)
                for name in ("test.config", "test.materials"):
                    (hotspot / "examples/thermal" / name).write_text("fixture\n")
                thermal_patches(stack)
                stack.enter_context(patch("scripts.workspace_cts_thermal.normalize_def", return_value={"components": 0}))
                stack.enter_context(patch("open3dbench.backend.thermal._read_def", return_value=(None, None, [])))
                stack.enter_context(patch("open3dbench.backend.thermal._write_floorplan"))
                stack.enter_context(patch("open3dbench.backend.thermal._write_stack"))
                def simulate(*args, **kwargs):
                    (root / "thermal/temperature.steady").write_text(tile_text(hot_die))
                stack.enter_context(patch("grt_bench.endpoint.subprocess.run", side_effect=simulate))
                metrics, diagnostics = endpoint._thermal(root, root, {die: root / die for die in ("top", "bottom")}, hotspot)
                self.assertEqual(metrics["temperature_peak_k"], 350)
                self.assertAlmostEqual(metrics["power_w"], .2)
                self.assertAlmostEqual(diagnostics["temperature"][hot_die]["average_k"], 320.3)

    def fixture(self, root, hot_die="top"):
        for name in ("assembly/power.json", "assembly/instance_power.rpt", "assembly/design.spef",
                     "assembly/check_setup.rpt", "assembly/latch_proof.tsv", "thermal/hotspot.log"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        (root / "assembly/metrics.tsv").write_text("wns_setup\t-1\ntns_setup\t-2\nsta_completeness_checks_passed\t1\nhbt_adapter_cell_count\t0\n")
        (root / "thermal/temperature.steady").write_text(tile_text(hot_die))
        names = [name for die in ("top", "bottom") for name in grid(die)[0]]
        (root / "thermal/power.ptrace").write_text(" ".join(names) + "\n" + " ".join([".001"] * 200) + "\n")
        (root / "checkpoint.json").write_text("{}")
        write(root / "metrics.json", {"endpoint": ENDPOINT, "thermal_statistic": THERMAL_STATISTIC,
              "parasitics": "global-route-estimated-rc", "signoff": False,
              "metrics": {"wns_deficit_ns": 1, "tns_deficit_ns": 2, "overflow": 0,
                          "temperature_peak_k": 350, "power_w": .2}})
        self.seal(root)

    def seal(self, root):
        write(root / "manifest.json", {"endpoint": ENDPOINT, "implementation_sha256": sha(endpoint.__file__),
              "inputs": {}, "platform": str(root), "source_checkpoint": str(root / "checkpoint.json"),
              "source_checkpoint_sha256": sha(root / "checkpoint.json"),
              "artifacts": {k: v for k, v in records(root).items() if k != "manifest.json"}}, replace=True)

    def check(self, root):
        with ExitStack() as stack:
            thermal_patches(stack)
            stack.enter_context(patch("grt_bench.verify.validate_grt"))
            stack.enter_context(patch("open3dbench.backend.three_d_route._grt_metrics", return_value={"grt_total_overflow": 0}))
            return verify_analysis(root)

    def test_verifier_independently_reduces_raw_tiles(self):
        for die in ("top", "bottom"):
            with self.subTest(die=die), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root, die)
                self.assertEqual(self.check(root)["temperature_peak_k"], 350)
                recorded = read(root / "metrics.json")
                recorded["metrics"]["temperature_peak_k"] = statistics.mean([320] * 199 + [350])
                write(root / "metrics.json", recorded, replace=True)
                self.seal(root)
                with self.assertRaisesRegex(ValueError, "Objective differs.*temperature_peak_k"):
                    self.check(root)

    def test_verifier_rejects_old_semantics_even_with_new_hashes(self):
        for field in ("statistic", "metric"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root)
                recorded = read(root / "metrics.json")
                if field == "statistic": recorded["thermal_statistic"] = "mean-silicon-temperature"
                else: recorded["metrics"]["temperature_k"] = recorded["metrics"].pop("temperature_peak_k")
                write(root / "metrics.json", recorded, replace=True)
                self.seal(root)
                with self.assertRaisesRegex(ValueError, "peak-temperature metric contract"):
                    self.check(root)

    def test_verifier_rejects_missing_duplicate_and_nonfinite_tiles(self):
        for change in (lambda s: "\n".join(s.splitlines()[1:]),
                       lambda s: s + s.splitlines()[0] + "\n",
                       lambda s: s.replace("350", "nan")):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root)
                (root / "thermal/temperature.steady").write_text(change(tile_text()))
                self.seal(root)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.check(root)


if __name__ == "__main__":
    unittest.main()
