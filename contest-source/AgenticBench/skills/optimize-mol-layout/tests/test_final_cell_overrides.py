#!/usr/bin/env python3
"""Unit tests for stage-scoped final-cell DREAMPlace overrides."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "Place-MoL"))

from src.utils.config_utils import args_for_final_cell_placement


class FinalCellOverrideTest(unittest.TestCase):
    def test_main_uses_scoped_args_only_for_final_cell_instance(self):
        main_path = REPO_ROOT / "Place-MoL" / "src" / "place_3d" / "main.py"
        tree = ast.parse(main_path.read_text(encoding="utf-8"))
        scoped_assignments = []
        final_instance_args = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                target_names = [
                    target.id for target in node.targets if isinstance(target, ast.Name)
                ]
                function = node.value.func
                function_name = function.id if isinstance(function, ast.Name) else None
                if function_name == "args_for_final_cell_placement":
                    scoped_assignments.extend(target_names)
                if (
                    "shrunk_problem_instance" in target_names
                    and function_name == "ProblemInstance"
                ):
                    final_instance_args.append(node.value.args[0].id)

        self.assertEqual(scoped_assignments, ["final_cell_args"])
        self.assertEqual(final_instance_args, ["final_cell_args"])

    def test_final_override_is_scoped_to_copied_args(self):
        args = SimpleNamespace(
            dreamplace_overrides={"gamma": 4.0, "density_weight": 8e-5},
            final_cell_dreamplace_overrides={"density_weight": 0.00012},
            seed=3,
        )

        final_args = args_for_final_cell_placement(args)

        self.assertIsNot(final_args, args)
        self.assertEqual(
            final_args.dreamplace_overrides,
            {"gamma": 4.0, "density_weight": 0.00012},
        )
        self.assertEqual(
            args.dreamplace_overrides,
            {"gamma": 4.0, "density_weight": 8e-5},
        )
        self.assertEqual(final_args.seed, 3)

    def test_missing_final_override_copies_base_mapping(self):
        args = SimpleNamespace(dreamplace_overrides={"stop_overflow": 0.1})

        final_args = args_for_final_cell_placement(args)

        self.assertEqual(final_args.dreamplace_overrides, {"stop_overflow": 0.1})
        self.assertIsNot(final_args.dreamplace_overrides, args.dreamplace_overrides)

    def test_non_dictionary_overrides_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "dreamplace_overrides"):
            args_for_final_cell_placement(
                SimpleNamespace(
                    dreamplace_overrides=[],
                    final_cell_dreamplace_overrides={},
                )
            )
        with self.assertRaisesRegex(TypeError, "final_cell_dreamplace_overrides"):
            args_for_final_cell_placement(
                SimpleNamespace(
                    dreamplace_overrides={},
                    final_cell_dreamplace_overrides=[],
                )
            )


if __name__ == "__main__":
    unittest.main()
