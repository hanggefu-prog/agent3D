from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


FRAMEWORK = Path(__file__).resolve().parents[1]
WORKSPACE = FRAMEWORK.parent
SKILLS = FRAMEWORK / "skills"
LEGACY_PLACE_MOL = WORKSPACE / "Place-MoL"
LEGACY_METRIC_UTILS_AVAILABLE = (
    LEGACY_PLACE_MOL / "src" / "utils" / "metric_utils.py"
).is_file()
if LEGACY_METRIC_UTILS_AVAILABLE:
    sys.path.insert(0, str(LEGACY_PLACE_MOL))
    from src.utils.metric_utils import last_metric_with_hpwl, metric_scalar  # noqa: E402


class StageSkillTests(unittest.TestCase):
    def run_script(self, script: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    @unittest.skipUnless(
        LEGACY_METRIC_UTILS_AVAILABLE,
        "legacy Place-MoL integration is not present in a standalone deployment",
    )
    def test_final_dreamplace_metric_is_selected(self) -> None:
        first = SimpleNamespace(hpwl=10.0, overflow=0.2)
        final = SimpleNamespace(hpwl=8.0, overflow=0.1)
        selected = last_metric_with_hpwl([[first], [final]])
        self.assertIs(selected, final)
        self.assertEqual(metric_scalar(selected.hpwl), 8.0)

    def test_stage_contracts_are_adapter_first_and_cts_complete(self) -> None:
        partition = (
            SKILLS / "optimize-mol-partition" / "SKILL.md"
        ).read_text(encoding="utf-8")
        layout = (
            SKILLS / "optimize-mol-layout" / "SKILL.md"
        ).read_text(encoding="utf-8")
        place_opt = (
            SKILLS / "optimize-openroad-place" / "SKILL.md"
        ).read_text(encoding="utf-8")
        routing = (
            SKILLS / "optimize-openroad-routing" / "SKILL.md"
        ).read_text(encoding="utf-8")
        self.assertIn("LoL-style movable-instance assignment", partition)
        self.assertIn("unified Place-3D, legacy MoL, or LoL", layout)
        self.assertIn("does not require\nthe optional `binaries.tar.gz`", layout)
        self.assertIn("original Place-LoL benchmark tree", layout)
        self.assertIn("accepted `4_cts.odb`", place_opt)
        self.assertIn("adapter-selected CTS artifact", routing)
        self.assertIn("stages.routing.evaluation_commands", routing)

    def test_layout_log_parse_and_compare(self) -> None:
        parser = SKILLS / "optimize-mol-layout" / "scripts" / "parse_place_log.py"
        comparer = SKILLS / "optimize-mol-layout" / "scripts" / "compare_place_runs.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline_log = root / "baseline.log"
            candidate_log = root / "candidate.log"
            baseline_log.write_text(
                "Pipeline Summary\nMem-on-logic HPWL before legalization: 100.0\n"
                "Mem-on-logic density overflow: 0.20\nMem-on-logic max density: 1.20\n"
                "Saved result to CSV\n",
                encoding="utf-8",
            )
            candidate_log.write_text(
                "Pipeline Summary\nMem-on-logic HPWL before legalization: 95.0\n"
                "Mem-on-logic density overflow: 0.18\nMem-on-logic max density: 1.10\n"
                "Saved result to CSV\n",
                encoding="utf-8",
            )
            baseline_json = root / "baseline.json"
            candidate_json = root / "candidate.json"
            self.assertEqual(
                self.run_script(parser, str(baseline_log), "--require-density", "--output", str(baseline_json)).returncode,
                0,
            )
            self.assertEqual(
                self.run_script(parser, str(candidate_log), "--require-density", "--output", str(candidate_json)).returncode,
                0,
            )
            compared = self.run_script(comparer, str(baseline_json), str(candidate_json))
            self.assertEqual(compared.returncode, 0)
            self.assertIsNotNone(json.loads(compared.stdout)["recommended"])

    def test_partition_validator(self) -> None:
        validator = SKILLS / "optimize-mol-partition" / "scripts" / "validate_partition.py"
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "partition.json"
            artifact.write_text(
                json.dumps({
                    "all_macros": ["a", "b"],
                    "upper_die_macros": ["a"],
                    "bottom_die_macros": ["b"],
                    "upper_die_area": 49,
                    "bottom_die_area": 51,
                    "area_balance_error": 0.1,
                }),
                encoding="utf-8",
            )
            result = self.run_script(validator, str(artifact))
            self.assertEqual(result.returncode, 0)
            self.assertTrue(json.loads(result.stdout)["passed"])

    def test_partition_validator_supports_unified_assignment_csv(self) -> None:
        validator = SKILLS / "optimize-mol-partition" / "scripts" / "validate_partition.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assignment = root / "assignment.csv"
            assignment.write_text(
                "inst_name,die\na,top\nb,bottom\n", encoding="utf-8"
            )
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps({"top_area": 49, "bottom_area": 51}),
                encoding="utf-8",
            )
            expected = root / "expected.txt"
            expected.write_text("a\nb\n", encoding="utf-8")
            result = self.run_script(
                validator,
                str(assignment),
                "--metrics",
                str(metrics),
                "--expected-instances",
                str(expected),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["passed"])

    def test_config_variant_rejects_unknown_keys(self) -> None:
        generator = SKILLS / "optimize-mol-layout" / "scripts" / "make_place_config.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.json"
            output = root / "variant.json"
            base.write_text(json.dumps({"macro": {"weight": 1.0}}), encoding="utf-8")
            good = self.run_script(generator, str(base), str(output), "--set", "macro.weight=2.0")
            self.assertEqual(good.returncode, 0)
            self.assertEqual(json.loads(output.read_text())["macro"]["weight"], 2.0)
            density_output = root / "density.json"
            density = self.run_script(
                generator,
                str(base),
                str(density_output),
                "--set",
                "dreamplace_overrides.density_weight=0.00012",
            )
            self.assertEqual(density.returncode, 0)
            self.assertEqual(
                json.loads(density_output.read_text())["dreamplace_overrides"]["density_weight"],
                0.00012,
            )

    def test_reconstructed_benchmark_normalizes_component_masters(self) -> None:
        preparer = (
            SKILLS
            / "optimize-mol-layout"
            / "scripts"
            / "prepare_benchmark_from_eval_pack.py"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            platform = root / "platform"
            (platform / "lef").mkdir(parents=True)
            (platform / "lib").mkdir()
            (platform / "lef" / "tech.lef").write_text("VERSION 5.8 ;\n")
            (platform / "lef" / "memory.lef").write_text(
                "MACRO RAM\n  CLASS BLOCK ;\n  SIZE 10 BY 20 ;\nEND RAM\n"
            )
            (platform / "lib" / "cells.lib").write_text("library(x) {}\n")
            source = root / "input.def"
            source.write_text(
                "VERSION 5.8 ;\nCOMPONENTS 3 ;\n"
                "  - u0 INV_X1_bottom\n    + FIXED ( 0 0 ) N ;\n"
                "  - ram_top RAM_upper\n    + FIXED ( 10 20 ) N ;\n"
                "  - ram_bottom RAM_bottom\n    + FIXED ( 30 40 ) N ;\n"
                "END COMPONENTS\nNETS 0 ;\nEND NETS\nEND DESIGN\n",
                encoding="utf-8",
            )
            output = root / "benchmark"
            result = self.run_script(
                preparer,
                "--eval-def", str(source),
                "--platform", str(platform),
                "--design", "tiny",
                "--output", str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            normalized = (output / "or_3D" / "tiny" / "2_2_floorplan_io.def").read_text()
            self.assertIn("INV_X1\n", normalized)
            self.assertIn("+ PLACED", normalized)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["frozen_partition"],
                {
                    "upper_die_macro_names": ["ram_top"],
                    "bottom_die_macro_names": ["ram_bottom"],
                },
            )


if __name__ == "__main__":
    unittest.main()
