from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import yaml
from pathlib import Path

from local_bench.benchmark_input import (
    load_benchmark_input_profile,
    resolve_benchmark_run,
    validate_benchmark_input,
)
from local_bench.model import load_task


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ID = "bp_fe_lol_ng45_mtkahypar"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BenchmarkInputTest(unittest.TestCase):
    def _fake_bundle(self, root: Path, profile_id=PROFILE_ID) -> Path:
        profile = load_benchmark_input_profile(profile_id)
        run = root / "run"
        stages = {}
        for stage, directory in profile.required_stages.items():
            stage_dir = run / directory
            stage_dir.mkdir(parents=True)
            artifact = stage_dir / "payload.txt"
            artifact.write_text(stage + "\n", encoding="utf-8")
            manifest = stage_dir / "stage_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage": stage,
                        "status": "success",
                        "signature": {"test": "fixed"},
                        "artifacts": {
                            "payload": {
                                "path": "payload.txt",
                                "bytes": artifact.stat().st_size,
                                "sha256": sha256(artifact),
                            }
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            stages[stage] = {
                "directory": directory,
                "manifest_sha256": sha256(manifest),
                "artifact_count": 1,
            }
        bundle = root / "benchmark_input.json"
        bundle.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_id": profile.profile_id,
                    "run_dir": "run",
                    "profile_sha256": sha256(profile.root / "profile.yaml"),
                    "config_sha256": sha256(profile.config),
                    "stages": stages,
                    "environment": {"test": True},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return bundle

    def test_current_placement_contracts_use_global_two_way(self):
        for task in ("placement_initialization", "placement_preconditioner", "placement_parameters_reference_20260923"):
            spec = load_task(ROOT / "local_tasks" / task)
            config = yaml.safe_load(spec.base_config.read_text())
            self.assertEqual(config["partition"]["algorithm"], "tp-gnn-flat")
            self.assertEqual(spec.upstream_stages, ["input", "floorplan", "partition"])

    def test_full_placement_cases_have_peer_profiles(self) -> None:
        expected = {
            "bp_fe_lol_ng45_mtkahypar": "bp_fe",
            "bp_lol_ng45_mtkahypar": "bp",
            "ariane133_lol_ng45_mtkahypar": "ariane133",
            "bp_multi_lol_ng45_mtkahypar": "bp_multi",
            "swerv_wrapper_lol_ng45_mtkahypar": "swerv_wrapper",
        }
        for profile_id, design in expected.items():
            profile = load_benchmark_input_profile(profile_id)
            config = yaml.safe_load(profile.config.read_text(encoding="utf-8"))
            self.assertEqual(config["design"], design)
            self.assertEqual(config["pdk"], "ng45")
            self.assertEqual(config["partition"]["algorithm"], "mtkahypar")
            compatible = {
                "placement_preconditioner_source",
                "placement_dynamic_parameters",
            }
            if design != "bp_fe":
                compatible.add("placement_two_pass_sta")
            self.assertEqual(set(profile.compatible_tasks), compatible)

    def test_current_tp_gnn_preserves_the_frozen_timing_contract(self):
        spec = load_task(ROOT / "local_tasks/tp_gnn_post_placement_timing")
        self.assertEqual(len(spec.frozen_inputs), 1)
        timing = spec.frozen_inputs[0]
        self.assertEqual(timing.source, "baseline_run")
        self.assertEqual(timing.path, "20_partition/timing_input")
        self.assertEqual(timing.inject_as, "partition.timing_input")
        self.assertEqual(timing.json_checks[0].equals, {
            "timing_status": "full_coverage_partition_last",
            "timing_weight_mode": "paper_delay_slack_hyperedge_v1"})

    def test_routing_source_has_a_dedicated_cts_profile(self) -> None:
        profile = load_benchmark_input_profile("ariane133_lol_ng45_tp_gnn_cts")
        spec = load_task(ROOT / "local_tasks" / "global_routing_ripup_policy")
        config = yaml.safe_load(profile.config.read_text(encoding="utf-8"))
        self.assertEqual(profile.stop_stage, "cts")
        self.assertEqual(list(profile.required_stages)[-1], "cts")
        self.assertEqual(len(profile.required_stages), 8)
        # FastRoute enables StNetOrder only on i % 3 == 0, and soft-NDR can
        # reset i at the configured limit.  Four reaches i=3 before each reset.
        self.assertGreaterEqual(config["routing"]["congestion_iterations"], 4)
        self.assertEqual(spec.benchmark_input_id, profile.profile_id)
        self.assertEqual(spec.base_config, profile.config)
        self.assertEqual(
            profile.compatible_tasks,
            (
                "global_routing_net_ordering_source",
                "global_routing_ripup_policy_source",
                "global_routing_maze_cost_source",
                "global_routing_iteration_controller_source",
            ),
        )

    def test_routing_source_cts_profiles_cover_all_long_run_cases(self) -> None:
        for profile_id, design in {
            "bp_lol_ng45_mtkahypar_cts": "bp",
            "bp_multi_lol_ng45_mtkahypar_cts": "bp_multi",
            "ariane133_lol_ng45_mtkahypar_cts": "ariane133",
        }.items():
            profile = load_benchmark_input_profile(profile_id)
            config = yaml.safe_load(profile.config.read_text(encoding="utf-8"))
            self.assertEqual(config["design"], design)
            self.assertEqual(profile.stop_stage, "cts")
            self.assertEqual(len(profile.required_stages), 8)
            self.assertGreaterEqual(
                config["routing"]["congestion_iterations"], 4
            )
            self.assertEqual(
                profile.compatible_tasks,
                (
                    "global_routing_net_ordering_source",
                    "global_routing_ripup_policy_source",
                    "global_routing_maze_cost_source",
                    "global_routing_iteration_controller_source",
                ),
            )

    def test_bundle_validates_and_resolves_portably(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = self._fake_bundle(Path(temp))
            report = validate_benchmark_input(
                manifest, task_id="placement_dynamic_parameters"
            )
            self.assertEqual(report["status"], "ok")
            self.assertEqual(
                Path(report["config"]), load_benchmark_input_profile(PROFILE_ID).config
            )
            self.assertEqual(resolve_benchmark_run(manifest), (Path(temp) / "run").resolve())

    def test_mutated_frozen_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._fake_bundle(root)
            (root / "run" / "20_partition" / "payload.txt").write_text(
                "changed\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "artifact (size|hash) changed"):
                validate_benchmark_input(manifest)

    def test_incompatible_task_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = self._fake_bundle(Path(temp))
            with self.assertRaisesRegex(ValueError, "not compatible"):
                validate_benchmark_input(manifest, task_id="netlist_partition")

    def test_cts_thermal_reuses_only_compatible_sta_inputs_without_rewriting(self):
        for case in ("bp_fe", "swerv_wrapper", "ariane133"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = self._fake_bundle(root, case + "_lol_ng45_tp_gnn_sta")
                original = manifest.read_bytes()
                report = validate_benchmark_input(manifest, task_id="tp_gnn_placement_cts_thermal")
                self.assertEqual(report["status"], "ok")
                self.assertEqual(manifest.read_bytes(), original)
                (root / "run/10_floorplan/payload.txt").write_text("corrupted")
                with self.assertRaisesRegex(ValueError, "artifact (size|hash) changed"):
                    validate_benchmark_input(manifest, task_id="tp_gnn_placement_cts_thermal")
        with tempfile.TemporaryDirectory() as temp:
            manifest = self._fake_bundle(Path(temp))
            with self.assertRaisesRegex(ValueError, "not compatible"):
                validate_benchmark_input(manifest, task_id="tp_gnn_placement_cts_thermal")


if __name__ == "__main__":
    unittest.main()
