#!/usr/bin/env python3
"""Focused tests for final-cell finite checkpoints and fail-closed boundaries."""

import ast
import inspect
import io
import math
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from test_placement_diagnostics import RUN_DMP


REPO_ROOT = Path(__file__).resolve().parents[4]
NONLINEAR_PATH = (
    REPO_ROOT / "Place-MoL" / "DREAMPlace" / "dreamplace" / "NonLinearPlace.py"
)
MAIN_PATH = REPO_ROOT / "Place-MoL" / "src" / "place_3d" / "main.py"


def _load_checkpoint_primitives():
    tree = ast.parse(NONLINEAR_PATH.read_text(encoding="utf-8"))
    selected = []
    wanted = {
        "FiniteGuardError",
        "_checkpoint_key",
        "FiniteCheckpointStore",
    }
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(selected, type_ignores=[]), str(NONLINEAR_PATH), "exec"), namespace)
    return namespace


PRIMITIVES = _load_checkpoint_primitives()
FiniteGuardError = PRIMITIVES["FiniteGuardError"]
FiniteCheckpointStore = PRIMITIVES["FiniteCheckpointStore"]


class _BooleanResult:
    def __init__(self, value):
        self.value = bool(value)

    def all(self):
        return self

    def item(self):
        return self.value


class _LegalizedPosition:
    @property
    def data(self):
        return self

    def copy_(self, other):
        return self


class FiniteCheckpointStoreTest(unittest.TestCase):
    def test_checkpoint_selection_keeps_metric_position_state_atomic(self):
        store = FiniteCheckpointStore()
        first = {
            "metric": {"overflow": 0.24, "hpwl": 5_420_000.0},
            "pos": "position-2001",
            "gamma": "gamma-2001",
            "density_weight": "weight-2001",
        }
        worse = {
            "metric": {"overflow": 0.25, "hpwl": 5_000_000.0},
            "pos": "position-2100",
            "gamma": "gamma-2100",
            "density_weight": "weight-2100",
        }
        better = {
            "metric": {"overflow": 0.23, "hpwl": 5_449_000.0},
            "pos": "position-2200",
            "gamma": "gamma-2200",
            "density_weight": "weight-2200",
        }

        store.consider(first)
        store.consider(worse)
        self.assertIs(store.require(), first)
        store.consider(better)
        self.assertIs(store.require(), better)
        self.assertEqual(store.require()["pos"], "position-2200")
        self.assertEqual(store.require()["gamma"], "gamma-2200")

    def test_checkpoint_tie_breaks_on_hpwl(self):
        store = FiniteCheckpointStore()
        high_hpwl = {
            "metric": {"overflow": 0.23, "hpwl": 5_440_000.0},
            "pos": "high-hpwl",
        }
        low_hpwl = {
            "metric": {"overflow": 0.23, "hpwl": 5_430_000.0},
            "pos": "low-hpwl",
        }
        store.consider(high_hpwl)
        store.consider(low_hpwl)
        self.assertIs(store.require(), low_hpwl)

    def test_missing_checkpoint_fails_closed(self):
        with self.assertRaisesRegex(FiniteGuardError, "before any valid checkpoint"):
            FiniteCheckpointStore().require()


class GuardWiringTest(unittest.TestCase):
    def test_guard_is_keyword_only_and_position_metric_restore_is_ordered(self):
        source = NONLINEAR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        nonlinear = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NonLinearPlace"
        )
        call = next(
            node for node in nonlinear.body
            if isinstance(node, ast.FunctionDef) and node.name == "__call__"
        )
        self.assertEqual([arg.arg for arg in call.args.kwonlyargs], ["finite_guard"])
        self.assertIsInstance(call.args.kw_defaults[0], ast.Constant)
        self.assertFalse(call.args.kw_defaults[0].value)

        pre = source.index("consider_finite_checkpoint(cur_metric, objective, pos)")
        optimizer = source.index("optimizer.step()", pre)
        post = source.index('"post_optimizer"', optimizer)
        restore_pos = source.index('copy_(checkpoint["pos"])')
        reevaluate = source.index("restored_metric.evaluate", restore_pos)
        append = source.index("metrics.append(restored_metric)", reevaluate)
        self.assertLess(pre, optimizer)
        self.assertLess(optimizer, post)
        self.assertLess(restore_pos, reevaluate)
        self.assertLess(reevaluate, append)

    def test_run_dmp_enables_guard_only_for_final_cell(self):
        prototype = RUN_DMP._stage_context(
            SimpleNamespace(
                dmp_params=SimpleNamespace(global_place_stages=[]),
                benchmark="bp_fe",
                only_cells=False,
            ),
            "prototype",
        )
        final_cell = RUN_DMP._stage_context(
            SimpleNamespace(
                dmp_params=SimpleNamespace(global_place_stages=[]),
                benchmark="bp_fe",
                only_cells=True,
            ),
            "final_cell",
        )
        self.assertFalse(prototype["finite_guard_enabled"])
        self.assertTrue(final_cell["finite_guard_enabled"])
        source = inspect.getsource(RUN_DMP.ProblemInstance.place_2D)
        self.assertIn("finite_guard=finite_guard_enabled", source)

    def test_normal_iteration_cap_submits_checkpoint_without_rollback(self):
        source = NONLINEAR_PATH.read_text(encoding="utf-8")
        boundary = source.index('if finite_guard:')
        cap = source.index(
            'elif iteration >= int(global_place_params["iteration"]):',
            boundary,
        )
        submit = source.index("submit_finite_checkpoint(", cap)
        stopping = source.index('"termination_reason": "stopping_criterion"', submit)
        cap_block = source[submit:stopping]
        self.assertIn('termination_reason="best_finite_at_iteration_cap"', cap_block)
        self.assertIn("count_as_rollback=False", cap_block)
        self.assertIn("None,\n                                None,", cap_block)
        self.assertLess(cap, submit)
        self.assertLess(submit, stopping)

    def test_stage_json_rejects_nonfinite_numbers(self):
        with self.assertRaises(ValueError):
            with redirect_stdout(io.StringIO()):
                RUN_DMP._emit_stage_record("DMP_STAGE_END", {"hpwl": math.inf})
        self.assertIsNone(RUN_DMP._json_scalar(float("-inf")))
        self.assertIsNone(RUN_DMP._json_scalar(float("nan")))


class BoundaryAndLegalizationTest(unittest.TestCase):
    def _placedb(self, *, node_x=(1.0,), node_y=(2.0,)):
        return SimpleNamespace(
            node_x=list(node_x),
            node_y=list(node_y),
            node_size_x=[1.0],
            node_size_y=[1.0],
            num_movable_nodes=1,
            xl=0.0,
            yl=0.0,
            xh=10.0,
            yh=10.0,
        )

    def test_place_boundary_rejects_nonfinite_metric_and_out_of_bounds_position(self):
        finite_metric = SimpleNamespace(hpwl=10.0, overflow=0.2, max_density=1.1)
        invalid_metric = SimpleNamespace(hpwl=float("-inf"), overflow=0.0, max_density=1.0)
        with self.assertRaisesRegex(RuntimeError, "invalid final metrics"):
            RUN_DMP._require_finite_placement_result(invalid_metric, self._placedb())
        with self.assertRaisesRegex(RuntimeError, "out-of-bounds"):
            RUN_DMP._require_finite_placement_result(
                finite_metric, self._placedb(node_x=(10.0,))
            )

    def test_legalization_false_result_stops_before_placedb_apply(self):
        original_isfinite = getattr(RUN_DMP.th, "isfinite", None)
        RUN_DMP.th.isfinite = lambda value: _BooleanResult(True)
        applied = []
        instance = object.__new__(RUN_DMP.ProblemInstance)
        instance.dmp_placer = SimpleNamespace(
            pos=[_LegalizedPosition()],
            op_collections=SimpleNamespace(
                legalize_op=lambda pos: _LegalizedPosition(),
                legality_check_op=lambda pos: False,
            ),
        )
        instance.dmp_placedb = SimpleNamespace(apply=lambda *args: applied.append(args))
        instance.dmp_params = SimpleNamespace()
        try:
            with self.assertRaisesRegex(RuntimeError, "legality check failed"):
                instance.legalize()
        finally:
            if original_isfinite is None:
                del RUN_DMP.th.isfinite
            else:
                RUN_DMP.th.isfinite = original_isfinite
        self.assertEqual(applied, [])

    def test_legalization_nonfinite_position_stops_before_legality_or_apply(self):
        original_isfinite = getattr(RUN_DMP.th, "isfinite", None)
        RUN_DMP.th.isfinite = lambda value: _BooleanResult(False)
        legality_calls = []
        applied = []
        instance = object.__new__(RUN_DMP.ProblemInstance)
        instance.dmp_placer = SimpleNamespace(
            pos=[_LegalizedPosition()],
            op_collections=SimpleNamespace(
                legalize_op=lambda pos: _LegalizedPosition(),
                legality_check_op=lambda pos: legality_calls.append(pos),
            ),
        )
        instance.dmp_placedb = SimpleNamespace(apply=lambda *args: applied.append(args))
        instance.dmp_params = SimpleNamespace()
        try:
            with self.assertRaisesRegex(RuntimeError, "non-finite coordinates"):
                instance.legalize()
        finally:
            if original_isfinite is None:
                del RUN_DMP.th.isfinite
            else:
                RUN_DMP.th.isfinite = original_isfinite
        self.assertEqual(legality_calls, [])
        self.assertEqual(applied, [])

    def test_legalization_invalid_hpwl_stops_before_placedb_apply(self):
        original_isfinite = getattr(RUN_DMP.th, "isfinite", None)
        original_metrics = RUN_DMP.EvalMetrics.EvalMetrics
        RUN_DMP.th.isfinite = lambda value: _BooleanResult(True)

        class _InvalidMetrics:
            def __init__(self, iteration):
                self.hpwl = None

            def evaluate(self, placedb, ops, pos):
                self.hpwl = float("inf")

        RUN_DMP.EvalMetrics.EvalMetrics = _InvalidMetrics
        applied = []
        instance = object.__new__(RUN_DMP.ProblemInstance)
        instance.dmp_placer = SimpleNamespace(
            pos=[_LegalizedPosition()],
            op_collections=SimpleNamespace(
                legalize_op=lambda pos: _LegalizedPosition(),
                legality_check_op=lambda pos: True,
                hpwl_op=lambda pos: None,
            ),
        )
        instance.dmp_placedb = SimpleNamespace(apply=lambda *args: applied.append(args))
        instance.dmp_params = SimpleNamespace()
        try:
            with self.assertRaisesRegex(RuntimeError, "invalid HPWL"):
                instance.legalize()
        finally:
            RUN_DMP.EvalMetrics.EvalMetrics = original_metrics
            if original_isfinite is None:
                del RUN_DMP.th.isfinite
            else:
                RUN_DMP.th.isfinite = original_isfinite
        self.assertEqual(applied, [])

    def test_main_post_processing_is_after_uncaught_legalize_call(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        legalize = source.index("legalized_hpwl = legalize_problem_instance.legalize()")
        completed = source.index('print("Legalization complete")', legalize)
        suffix = source.index("DefProcessor.def_post_process", completed)
        self.assertLess(legalize, completed)
        self.assertLess(completed, suffix)


if __name__ == "__main__":
    unittest.main()
