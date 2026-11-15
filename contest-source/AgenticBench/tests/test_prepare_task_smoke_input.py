from pathlib import Path
import unittest
from unittest.mock import patch

from local_bench.benchmark_input import load_benchmark_input_profile
from scripts.prepare_task_smoke_input import PreparationRuntime

ROOT = Path(__file__).resolve().parents[1]


class SmokeInputTests(unittest.TestCase):
    def test_retired_only_profiles_are_not_shipped(self):
        for task, profile in (("netlist_partition", "bp_fe_lol_ng45_mtkahypar_frontend"),
                              ("global_routing_parameters", "bp_fe_lol_ng45_mtkahypar_parameter_cts")):
            with self.assertRaises(FileNotFoundError):
                load_benchmark_input_profile(profile)
            self.assertFalse((ROOT / "local_tasks" / task / "task.yaml").exists())

    def test_explicit_sta_profile_dependency_is_preserved(self):
        # This input still backs an evaluation-only profile, not an evolution task.
        loaded = load_benchmark_input_profile("bp_lol_ng45_tp_gnn_ta3d_20260904")
        self.assertTrue(loaded.config.is_file())
        with self.assertRaises(FileNotFoundError):
            load_benchmark_input_profile("bp_lol_ng45_tp_gnn_ta3d_portable")

    def test_private_mount(self):
        runtime = object.__new__(PreparationRuntime)
        runtime.batch = Path("/runs")
        runtime.image = "sha256:" + "a" * 64
        runtime.write_root = Path("/runs/private")
        argv = ["docker", "run", "-v", "/runs:/runs", runtime.image, "true"]
        with patch("scripts.codesign_runtime.Runtime.docker", return_value=argv):
            actual = runtime.docker(["true"], 0)
        self.assertIn("/runs:/runs:ro", actual)
        self.assertIn("/runs/private:/runs/private", actual)
        self.assertNotIn("/runs:/runs", actual)


if __name__ == "__main__":
    unittest.main()
