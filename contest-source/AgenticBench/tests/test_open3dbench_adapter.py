from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic3d.open3dbench import Open3DBenchAdapter  # noqa: E402
from agentic3d.ppa import GlobalPPACollector  # noqa: E402


FRAMEWORK = Path(__file__).resolve().parents[1]


class Open3DBenchAdapterTests(unittest.TestCase):
    def _lol_repo(self, root: Path) -> Path:
        (root / "Place-LoL" / "dreamplace").mkdir(parents=True)
        (root / "Place-LoL" / "dreamplace" / "Placer.py").write_text("# placer\n")
        (root / "Place-LoL" / "README.md").write_text("# Place-LoL\n")
        (root / "Place-LoL" / "test" / "3D_input_default").mkdir(parents=True)
        (root / "OpenROAD-3D" / "flow").mkdir(parents=True)
        (root / "OpenROAD-3D" / "flow" / "Makefile").write_text("all:\n")
        return root

    def test_lol_adapter_takes_priority_over_legacy_mol(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = self._lol_repo(Path(temp))
            (repo / "Place-MoL" / "src").mkdir(parents=True)
            adapter = Open3DBenchAdapter.detect(repo, FRAMEWORK / "adapters")
            context = adapter.context()
            self.assertEqual(adapter.manifest.name, "open3dbench-place-lol-v2")
            self.assertEqual(context["regime"], "lol")
            self.assertEqual(context["stages"]["layout"]["regime"], "lol")
            self.assertIn(
                "Place-LoL/dreamplace",
                context["stages"]["layout"]["source_paths"],
            )

    def _unified_repo(self, root: Path) -> Path:
        (root / "Place-3D" / "src").mkdir(parents=True)
        (root / "Place-3D" / "src" / "api.py").write_text("# api\n")
        (root / "Place-3D" / "src" / "partition").mkdir()
        (root / "Place-3D" / "src" / "placement").mkdir()
        (root / "Place-3D" / "config").mkdir()
        (root / "Place-3D" / "config" / "partition").mkdir()
        (root / "Place-3D" / "results").mkdir()
        (root / "OpenROAD-3D" / "flow").mkdir(parents=True)
        (root / "OpenROAD-3D" / "flow" / "Makefile").write_text("all:\n")
        (root / "OpenROAD-3D" / "flow" / "logs").mkdir()
        return root

    def test_detects_unified_repository_without_hard_coding_agent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = self._unified_repo(Path(temp))
            adapter = Open3DBenchAdapter.detect(repo, FRAMEWORK / "adapters")
            context = adapter.context()
            self.assertEqual(adapter.manifest.name, "open3dbench-unified-place3d-v2")
            self.assertIn(
                "Place-3D/src/partition",
                context["stages"]["partition"]["source_paths"],
            )
            self.assertIn(
                "OpenROAD-3D/flow/logs",
                context["stages"]["routing"]["log_roots"],
            )
            layout = context["stages"]["layout"]
            self.assertTrue(layout["preparation_commands"])
            self.assertIn("generated locally", " ".join(layout["required_inputs"]))
            self.assertIn("original Place-LoL", " ".join(layout["required_inputs"]))
            self.assertIn("official", " ".join(layout["optional_tools"]))

    def test_global_ppa_collector_reads_json_csv_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = self._unified_repo(Path(temp))
            results = repo / "Place-3D" / "results" / "mol" / "candidate"
            results.mkdir(parents=True)
            (results / "metrics.json").write_text(
                json.dumps({"hpwl": 123.5, "density_overflow": 0.21}),
                encoding="utf-8",
            )
            (results / "run.log").write_text(
                "Final maximum density: 1.31\nRuntime: 42.0\n",
                encoding="utf-8",
            )
            adapter = Open3DBenchAdapter.detect(repo, FRAMEWORK / "adapters")
            report = GlobalPPACollector(adapter).collect()
            metrics = report["stages"]["layout"]["latest_metrics"]
            self.assertEqual(metrics["hpwl"]["value"], 123.5)
            self.assertEqual(metrics["overflow"]["value"], 0.21)
            self.assertEqual(metrics["max_density"]["value"], 1.31)


if __name__ == "__main__":
    unittest.main()
