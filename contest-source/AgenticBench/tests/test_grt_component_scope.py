"""Expanded edit scope and native-build admission; no model or EDA execution."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.candidates import freeze, inventory, materialize, verify_scope
from grt_bench.contract import validate
from grt_bench.optimization import define_task, edit_policy, verify_candidate
from test_grt_matrix import task_fixture


ALGORITHMS = {
    "partition": ("Place-3D/src/partition/lol/methods/tp_gnn.py",),
    "placement": ("Place-3D/src/placement/concurrent_dreamplace.py",),
    "hbt": ("Place-3D/src/hbt/assignment.py",),
    "timing_opt": ("OpenROAD-3D/openroad/src/rsz/src/RepairSetup.cc",),
    "cts": ("OpenROAD-3D/openroad/src/cts/src/HTreeBuilder.cpp",
            "OpenROAD-3D/openroad/src/cts/include/cts/TritonCTS.h"),
    "grt": ("OpenROAD-3D/openroad/src/grt/src/GlobalRouter.cpp",
            "OpenROAD-3D/openroad/src/grt/src/fastroute/src/FastRoute.cpp",
            "OpenROAD-3D/openroad/src/grt/include/grt/GlobalRouter.h"),
}
PROTECTED = (
    "Place-3D/src/hbt/planner.py", "Place-3D/src/hbt/def_adapter.py",
    "open3dbench/backend/hbt_materialize.py", "open3dbench/backend/three_d_cts.py",
    "open3dbench/backend/three_d_route.py", "AgenticBench/grt_bench/endpoint.py",
    "OpenROAD-3D/openroad/src/cts/src/TritonCTS.tcl",
    "OpenROAD-3D/openroad/src/grt/src/GlobalRouter.i",
    "OpenROAD-3D/openroad/src/grt/CMakeLists.txt",
    "OpenROAD-3D/openroad/src/grt/test/regression.py",
    "OpenROAD-3D/openroad/src/sta/search/Sta.cc",
)


class ComponentScopeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.store = self.root / "candidates"
        for name in (*PROTECTED, *(p for paths in ALGORITHMS.values() for p in paths)):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("value = 1\n" if name.endswith(".py") else "int value = 1;\n")
        self.baseline = freeze(self.source, self.store)

    def task(self, components):
        prompt = "---\ncomponents: [" + ", ".join(components) + "]\n---\nImprove post-GRT results.\n"
        return define_task(task_fixture(), self.baseline, prompt, self.store)

    def test_shipped_prompt_enables_six_components_without_exposing_adapters(self):
        prompt = (Path(__file__).resolve().parents[1] /
                  "global_tasks/lol_post_grt_pareto/instruction.md").read_text()
        task = define_task(task_fixture(), self.baseline, prompt, self.store)
        self.assertEqual(task["optimization"]["components"], sorted(ALGORITHMS))
        self.assertEqual(task["optimization"]["allowed_files"],
                         sorted(p for paths in ALGORITHMS.values() for p in paths))
        self.assertTrue(edit_policy(task)["native"])
        self.assertNotEqual(validate(task), validate(self.task(["partition", "placement", "timing_opt"])))

    def test_each_component_accepts_its_algorithm_and_requires_the_right_runtime(self):
        for component, paths in ALGORITHMS.items():
            with self.subTest(component=component):
                task = self.task([component])
                native = component in {"timing_opt", "cts", "grt"}
                policy = edit_policy(task)
                self.assertEqual(policy["native"], native)
                self.assertEqual(policy["allowed_paths"], sorted(paths))
                edit = materialize(self.store / self.baseline, self.root / component)
                for name in paths:
                    (edit / name).write_text("value = 2\n" if name.endswith(".py") else "int value = 2;\n")
                candidate = freeze(edit, self.store, parent=self.baseline, policy=policy)
                audit = verify_candidate(task, self.store, candidate)
                self.assertEqual(audit["changes"], sorted(paths))
                self.assertEqual(audit["native_build_required"], native)
                # Another selected component is not implicit permission to edit HBT.
                if component != "hbt":
                    (edit / ALGORITHMS["hbt"][0]).write_text("value = 3\n")
                    with self.assertRaisesRegex(ValueError, "Protected/out-of-scope"):
                        freeze(edit, self.store, parent=self.baseline, policy=policy)

    def test_hbt_requires_exact_grant_and_does_not_unlock_protected_neighbors(self):
        before = inventory(self.source)
        algorithm = ALGORITHMS["hbt"][0]
        edit = materialize(self.store / self.baseline, self.root / "hbt-exact-grant")
        (edit / algorithm).write_text("value = 2\n")
        after = inventory(edit)
        policy = edit_policy(self.task(list(ALGORITHMS)))
        self.assertEqual(verify_scope(before, after, policy)["changes"], [algorithm])
        with self.assertRaisesRegex(ValueError, "Protected/out-of-scope"):
            verify_scope(before, after, {"allowed_paths": ["Place-3D/src/hbt/"], "native": False})
        for i, name in enumerate(PROTECTED):
            edit = materialize(self.store / self.baseline, self.root / ("protected-" + str(i)))
            (edit / name).write_text("value = 2\n")
            after = inventory(edit)
            with self.subTest(path=name), self.assertRaisesRegex(ValueError, "Protected/out-of-scope"):
                verify_scope(before, after, policy)
            if name in PROTECTED[:2]:
                with self.subTest(exact_protected_grant=name), self.assertRaises(ValueError):
                    verify_scope(before, after, {"allowed_paths": [name], "native": False})

    def test_cts_and_grt_native_requirement_survives_a_python_only_child(self):
        task = self.task(list(ALGORITHMS))
        for component in ["cts", "grt"]:
            with self.subTest(component=component):
                edit = materialize(self.store / self.baseline, self.root / (component + "-parent"))
                (edit / ALGORITHMS[component][0]).write_text("int value = 3;\n")
                parent = freeze(edit, self.store, parent=self.baseline, policy=edit_policy(task))
                child_tree = materialize(self.store / parent, self.root / (component + "-child"))
                (child_tree / ALGORITHMS["hbt"][0]).write_text("value = 3\n")
                child = freeze(child_tree, self.store, parent=parent, policy=edit_policy(task))
                audit = verify_candidate(task, self.store, child)
                self.assertTrue(audit["native_build_required"])
                self.assertEqual(audit["ancestry"], [child, parent, self.baseline])

    def test_invalid_hbt_python_is_rejected(self):
        task = self.task(["hbt"])
        edit = materialize(self.store / self.baseline, self.root / "invalid")
        (edit / ALGORITHMS["hbt"][0]).write_text("def broken(\n")
        candidate = freeze(edit, self.store, parent=self.baseline, policy=edit_policy(task))
        with self.assertRaises(SyntaxError):
            verify_candidate(task, self.store, candidate)


if __name__ == "__main__":
    unittest.main()
