#!/usr/bin/env python3
"""Tests for behavior-neutral, stage-local DREAMPlace diagnostics."""

import ast
import importlib.util
import inspect
import io
import json
import logging
import sys
import types
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[4]
PLACE_MOL_ROOT = REPO_ROOT / "Place-MoL"
sys.path.insert(0, str(PLACE_MOL_ROOT))


def _stub_module(name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _load_run_dmp():
    _stub_module("torch")
    pil_module = _stub_module("PIL")
    image_module = _stub_module("PIL.Image")
    pil_module.Image = image_module

    src_module = _stub_module("src")
    src_module.__path__ = []
    utils_module = _stub_module("src.utils")
    utils_module.__path__ = []
    metric_spec = importlib.util.spec_from_file_location(
        "src.utils.metric_utils", PLACE_MOL_ROOT / "src" / "utils" / "metric_utils.py"
    )
    metric_module = importlib.util.module_from_spec(metric_spec)
    sys.modules["src.utils.metric_utils"] = metric_module
    metric_spec.loader.exec_module(metric_module)
    src_module.utils = utils_module
    utils_module.metric_utils = metric_module

    dreamplace = _stub_module("dreamplace")
    params_module = _stub_module("dreamplace.Params")
    placedb_module = _stub_module("dreamplace.PlaceDB")
    nonlinear_module = _stub_module("dreamplace.NonLinearPlace")
    timer_module = _stub_module("dreamplace.Timer")
    metrics_module = _stub_module("dreamplace.EvalMetrics")
    ops_module = _stub_module("dreamplace.ops")
    place_io_package = _stub_module("dreamplace.ops.place_io")
    place_io_module = _stub_module("dreamplace.ops.place_io.place_io")

    params_module.Params = type("Params", (), {})
    placedb_module.PlaceDB = type("PlaceDB", (), {})
    nonlinear_module.NonLinearPlace = type("NonLinearPlace", (), {})
    timer_module.Timer = type("Timer", (), {})
    metrics_module.EvalMetrics = type("EvalMetrics", (), {})
    place_io_module.PlaceIOFunction = type("PlaceIOFunction", (), {})

    dreamplace.Params = params_module
    dreamplace.PlaceDB = placedb_module
    dreamplace.NonLinearPlace = nonlinear_module
    dreamplace.Timer = timer_module
    dreamplace.EvalMetrics = metrics_module
    dreamplace.ops = ops_module
    ops_module.place_io = place_io_package
    place_io_package.place_io = place_io_module

    module_name = "run_dmp_stage_diagnostics_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name, PLACE_MOL_ROOT / "src" / "run_dmp.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


RUN_DMP = _load_run_dmp()


class _FakePosition:
    @property
    def data(self):
        return self

    def clone(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return [7.0, 11.0]


class _CompletedPlacer:
    def __init__(self, metric):
        self.metric = metric
        self.pos = [_FakePosition()]

    def __call__(self, params, placedb, *, finite_guard=False):
        self.last_finite_guard_report = {
            "enabled": finite_guard,
            "termination_reason": "finite_guard_rollback" if finite_guard else None,
            "trigger_iteration": 2110 if finite_guard else None,
            "trigger_phase": "post_optimizer" if finite_guard else None,
            "checkpoint_iteration": 2109 if finite_guard else None,
            "checkpoint_metric": {
                "iteration": 2109,
                "hpwl": 5422024.5,
                "overflow": 0.249661,
                "max_density": 1.898,
                "objective": 17.0,
                "gamma": 4.0,
                "density_weight": 0.00008,
            } if finite_guard else None,
            "restored_metric": {
                "iteration": 2109,
                "hpwl": 5422024.5,
                "overflow": 0.249661,
                "max_density": 1.898,
                "objective": 17.0,
                "gamma": 4.0,
                "density_weight": 0.00008,
            } if finite_guard else None,
            "nonfinite_fields": ["optimizer[0].obj_k[0]"] if finite_guard else [],
            "trigger_position_finite": False if finite_guard else True,
            "trigger_position_in_bounds": False if finite_guard else True,
            "position_finite": True,
            "position_in_bounds": True,
            "finite_guard_rollback_count": 1 if finite_guard else 0,
            "completed_iterations": 2110,
        }
        logging.info("not captured because it is below WARNING")
        logging.warning("stage warning")
        logging.error(RUN_DMP.ROLLBACK_DIAGNOSTIC)
        return [[self.metric]]


class _ExceptionalPlacer:
    pos = [_FakePosition()]

    def __call__(self, params, placedb, *, finite_guard=False):
        self.last_finite_guard_report = {
            "enabled": finite_guard,
            "finite_guard_rollback_count": 0,
        }
        logging.warning("failure warning")
        raise ValueError("synthetic placement failure")


class _BestAtCapPlacer(_CompletedPlacer):
    def __call__(self, params, placedb, *, finite_guard=False):
        checkpoint = {
            "iteration": 1096,
            "hpwl": 5397209.0,
            "overflow": 0.24900555610656738,
            "max_density": 1.897999882698059,
            "objective": 2865059403399168.0,
            "gamma": 4.222788333892822,
            "density_weight": 148562296832.0,
        }
        self.last_finite_guard_report = {
            "enabled": finite_guard,
            "termination_reason": "best_finite_at_iteration_cap",
            "trigger_iteration": None,
            "trigger_phase": None,
            "trigger_metric": None,
            "checkpoint_iteration": 1096,
            "checkpoint_metric": checkpoint,
            "restored_metric": dict(checkpoint),
            "nonfinite_fields": [],
            "trigger_position_finite": True,
            "trigger_position_in_bounds": True,
            "position_finite": True,
            "position_in_bounds": True,
            "finite_guard_rollback_count": 0,
            "completed_iterations": 2000,
        }
        logging.warning("stage warning")
        return [[self.metric]]


class _RuntimeWarningPlacer(_CompletedPlacer):
    def __call__(self, params, placedb, *, finite_guard=False):
        warnings.warn("synthetic numeric warning", RuntimeWarning)
        return super().__call__(
            params, placedb, finite_guard=finite_guard
        )


def _problem_instance(placer):
    instance = object.__new__(RUN_DMP.ProblemInstance)
    instance.benchmark = "bp_fe"
    instance.only_cells = True
    instance.args = SimpleNamespace(use_timer_for_evaluation=False)
    instance.dmp_params = SimpleNamespace(
        def_input="/tmp/bp_fe.def",
        target_density=0.588253,
        density_weight=0.00008,
        gamma=4.0,
        stop_overflow=0.1,
        global_place_stages=[{"iteration": 2000}],
    )
    instance.dmp_placedb = SimpleNamespace(
        node_x=[1.0],
        node_y=[2.0],
        node_size_x=[1.0],
        node_size_y=[1.0],
        num_movable_nodes=1,
        xl=0.0,
        yl=0.0,
        xh=10.0,
        yh=10.0,
    )
    instance.dmp_placer = placer
    instance.results = {}
    return instance


def _stage_records(output):
    records = {}
    for line in output.splitlines():
        prefix, encoded = line.split(" ", 1)
        records[prefix] = json.loads(encoded)
    return records


class PlacementDiagnosticsTest(unittest.TestCase):
    def test_best_at_cap_is_successful_and_not_a_rollback(self):
        metric = SimpleNamespace(
            iteration=1096,
            hpwl=5397209.0,
            overflow=0.24900555610656738,
            max_density=1.897999882698059,
        )
        instance = _problem_instance(_BestAtCapPlacer(metric))
        output = io.StringIO()

        with redirect_stdout(output):
            instance.place_2D(stage_label="final_cell")

        end = _stage_records(output.getvalue())["DMP_STAGE_END"]
        self.assertEqual(end["termination_reason"], "best_finite_at_iteration_cap")
        self.assertEqual(end["checkpoint_iteration"], 1096)
        self.assertEqual(end["completed_iterations"], 2000)
        self.assertIsNone(end["trigger_iteration"])
        self.assertIsNone(end["trigger_phase"])
        self.assertEqual(end["nonfinite_fields"], [])
        self.assertEqual(end["finite_guard_rollback_count"], 0)
        self.assertEqual(end["legacy_divergence_rollback_count"], 0)
        self.assertEqual(end["rollback_count"], 0)
        self.assertFalse(end["rollback_detected"])

    def test_completed_call_emits_and_saves_stage_local_record(self):
        metric = SimpleNamespace(
            iteration=137,
            hpwl=5422024.5,
            overflow=0.249661,
            max_density=1.898,
        )
        instance = _problem_instance(_CompletedPlacer(metric))
        root_logger = logging.getLogger()
        original_handlers = tuple(root_logger.handlers)
        original_level = root_logger.level
        output = io.StringIO()

        with redirect_stdout(output):
            hpwl, tns, wns = instance.place_2D(stage_label="final_cell")

        records = _stage_records(output.getvalue())
        self.assertEqual(set(records), {"DMP_STAGE_BEGIN", "DMP_STAGE_END"})
        begin = records["DMP_STAGE_BEGIN"]
        end = records["DMP_STAGE_END"]
        self.assertEqual(begin["stage"], "final_cell")
        self.assertEqual(end["schema_version"], 2)
        self.assertEqual(end["status"], "completed")
        self.assertEqual(end["benchmark"], "bp_fe")
        self.assertTrue(end["only_cells"])
        self.assertEqual(end["effective_params"]["global_place_iterations"], [2000])
        self.assertEqual(end["last_metric"]["iteration"], 137)
        self.assertEqual(end["last_metric"]["hpwl"], 5422024.5)
        self.assertTrue(end["rollback_detected"])
        self.assertEqual(end["rollback_count"], 2)
        self.assertEqual(end["legacy_divergence_rollback_count"], 1)
        self.assertEqual(end["finite_guard_rollback_count"], 1)
        self.assertEqual(end["termination_reason"], "finite_guard_rollback")
        self.assertEqual(end["trigger_iteration"], 2110)
        self.assertEqual(end["checkpoint_iteration"], 2109)
        self.assertTrue(end["position_finite"])
        self.assertTrue(end["position_in_bounds"])
        self.assertEqual(end["python_warnings"], [])
        self.assertEqual(
            end["diagnostics"],
            [
                {"level": "WARNING", "message": "stage warning"},
                {"level": "ERROR", "message": RUN_DMP.ROLLBACK_DIAGNOSTIC},
            ],
        )
        self.assertEqual(instance.results["placement_diagnostics"], end)
        self.assertEqual(instance.results["placement_metrics"]["overflow"], 0.249661)
        self.assertEqual((hpwl, tns, wns), (5422024.5, 0, 0))
        self.assertEqual(tuple(root_logger.handlers), original_handlers)
        self.assertEqual(root_logger.level, original_level)

    def test_runtime_warning_fails_closed_and_is_json_safe(self):
        metric = SimpleNamespace(
            iteration=2999,
            hpwl=float("-inf"),
            overflow=0.0,
            max_density=0.588253,
        )
        instance = _problem_instance(_RuntimeWarningPlacer(metric))
        output = io.StringIO()

        with self.assertRaisesRegex(RuntimeError, "RuntimeWarning"):
            with redirect_stdout(output):
                instance.place_2D(stage_label="final_cell")

        raw = output.getvalue()
        self.assertNotIn("Infinity", raw)
        self.assertNotIn("NaN", raw)
        end = _stage_records(raw)["DMP_STAGE_END"]
        self.assertEqual(end["status"], "exception")
        self.assertEqual(end["termination_reason"], "exception")
        self.assertEqual(end["last_metric"]["hpwl"], None)
        self.assertEqual(end["python_warnings"][0]["category"], "RuntimeWarning")

    def test_exception_is_recorded_then_rethrown_and_handler_removed(self):
        instance = _problem_instance(_ExceptionalPlacer())
        root_logger = logging.getLogger()
        original_handlers = tuple(root_logger.handlers)
        original_level = root_logger.level
        output = io.StringIO()

        with self.assertRaisesRegex(ValueError, "synthetic placement failure"):
            with redirect_stdout(output):
                instance.place_2D(stage_label="prototype")

        end = _stage_records(output.getvalue())["DMP_STAGE_END"]
        self.assertEqual(end["status"], "exception")
        self.assertEqual(end["exception"]["type"], "ValueError")
        self.assertEqual(end["last_metric"]["hpwl"], None)
        self.assertFalse(end["rollback_detected"])
        self.assertEqual(end["diagnostics"], [{"level": "WARNING", "message": "failure warning"}])
        self.assertEqual(instance.results["placement_diagnostics"], end)
        self.assertEqual(tuple(root_logger.handlers), original_handlers)
        self.assertEqual(root_logger.level, original_level)

    def test_stage_label_is_required_and_keyword_only(self):
        signature = inspect.signature(RUN_DMP.ProblemInstance.place_2D)
        self.assertEqual(
            signature.parameters["stage_label"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        instance = _problem_instance(_ExceptionalPlacer())
        with self.assertRaisesRegex(TypeError, "stage_label"):
            instance.place_2D()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            instance.place_2D(stage_label="")

    def test_main_has_exactly_two_labeled_place_calls(self):
        main_path = PLACE_MOL_ROOT / "src" / "place_3d" / "main.py"
        tree = ast.parse(main_path.read_text(encoding="utf-8"))
        labels = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "place_2D":
                continue
            self.assertEqual(node.args, [])
            stage_keywords = [kw for kw in node.keywords if kw.arg == "stage_label"]
            self.assertEqual(len(stage_keywords), 1)
            self.assertIsInstance(stage_keywords[0].value, ast.Constant)
            labels.append(stage_keywords[0].value.value)
        self.assertEqual(sorted(labels), ["final_cell", "prototype"])


if __name__ == "__main__":
    unittest.main()
