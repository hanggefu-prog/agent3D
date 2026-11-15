import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "summarize_tp_gnn_ta3d_evolution.py"
SPEC = importlib.util.spec_from_file_location("ta3d_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TA3DEvolutionSummaryTest(unittest.TestCase):
    def _write_program(self, root, checkpoint, program_id, iteration, score, valid):
        directory = (
            root
            / "bp"
            / "openevolve"
            / "checkpoints"
            / "checkpoint_{}".format(checkpoint)
            / "programs"
        )
        directory.mkdir(parents=True, exist_ok=True)
        metrics = {
            "combined_score": score,
            "runs_successfully": 1.0 if valid else 0.0,
        }
        if valid:
            metrics.update(
                {
                    "ta3d_signed_cut_score": score - 1.0,
                    "ta3d_signed_cut_gain": score - 1.0,
                    "critical_cut_reward": 0.2,
                    "noncritical_cut_penalty": 0.1,
                }
            )
        (directory / "{}.json".format(program_id)).write_text(
            json.dumps(
                {
                    "id": program_id,
                    "iteration_found": iteration,
                    "generation": iteration,
                    "metrics": metrics,
                }
            ),
            encoding="utf-8",
        )

    def test_deduplicates_checkpoints_and_tracks_best_so_far(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_program(root, 1, "baseline", 0, 1.0, True)
            self._write_program(root, 1, "invalid", 1, 0.0, False)
            self._write_program(root, 2, "baseline", 0, 1.0, True)
            self._write_program(root, 2, "invalid", 1, 0.0, False)
            self._write_program(root, 2, "improved", 2, 1.03, True)

            result = MODULE.summarize_case("bp", root / "bp")

        self.assertEqual(result["program_count"], 3)
        self.assertEqual(result["valid_program_count"], 2)
        self.assertEqual(result["iterations_observed"], 2)
        self.assertEqual([row["best_so_far"] for row in result["rows"]], [1.0, 1.0, 1.03])
        self.assertEqual(result["best"]["program_id"], "improved")


if __name__ == "__main__":
    unittest.main()
