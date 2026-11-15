"""Execute generated C++ and compare it with trusted, hand-written Python fixtures."""

import ast
import math
import random
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from local_tasks.routing_source_common import translate_scalar_program


class Context(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


class RoutingLanguageTest(unittest.TestCase):
    def translate(self, source, arity=1, **limits):
        fields = {n.attr for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                  and n.value.id == "ctx"}
        fields |= {"die_id", "edge_overflow", "history_pressure", "is_turn",
                   "baseline_decision", "congestion_trend", "reroute_phase"}
        return translate_scalar_program(
            source, entrypoint=ast.parse(source).body[0].name,
            context_fields=fields, return_arity=arity,
            return_cpp_type="Decision" if arity > 1 else "double", **limits
        ), sorted(fields)

    def assert_cpp_matches(self, source, rows, arity=1):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("a C++ compiler is required for semantic equivalence")
        body, fields = self.translate(source, arity)
        namespace = {"math": math}
        # Only literal fixtures in this test module execute in Python.
        exec(source, namespace)
        fn = namespace[ast.parse(source).body[0].name]
        expected = []
        cases = []
        for row in rows:
            ctx = Context(**{key: row.get(key, 0.0) for key in fields})
            value = fn(ctx)
            expected.extend(value if arity > 1 else [value])
            cases.append(" ".join(repr(getattr(ctx, key)) for key in fields))
        returns = "Decision" if arity > 1 else "double"
        output = "std::cout << value;" if arity == 1 else (
            "for (double item : value.values) std::cout << item << ' ';"
        )
        cpp = (
            "#include <algorithm>\n#include <cmath>\n#include <iostream>\n"
            "#include <iomanip>\n#include <limits>\n"
            "struct Context { " + " ".join("double " + f + ";" for f in fields) + " };\n"
            "struct Decision { double values[3]; };\n"
            + returns + " policy(const Context& ctx) {\n" + body + "\n}\n"
            "int main() { Context ctx; std::cout << std::setprecision(17);\n"
            "while (std::cin >> " + " >> ".join("ctx." + f for f in fields) + ") {\n"
            "auto value = policy(ctx); " + output + " std::cout << '\\n'; } }\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / "policy")
            built = subprocess.run(
                [compiler, "-x", "c++", "-std=c++17", "-O2", "-Werror=shadow",
                 "-o", binary, "-"], input=cpp, text=True, capture_output=True, timeout=60
            )
            self.assertEqual(built.returncode, 0, built.stderr + "\n" + body)
            ran = subprocess.run([binary], input="\n".join(cases) + "\n",
                                 text=True, capture_output=True, timeout=20, check=True)
        actual = [float(v) for v in ran.stdout.split()]
        self.assertEqual(len(actual), len(expected))
        for i, (a, b) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12),
                            "row {}: C++={} Python={}".format(i, a, b))
        return body

    def test_maze_nested_branch_regression_matches_both_dies(self):
        source = '''def score_maze_step(ctx):
    if ctx.edge_overflow > 0.0:
        if ctx.die_id < 0.5:
            mult = 1.0 + 0.05 * ctx.edge_overflow
        else:
            mult = 1.0 + 0.025 * ctx.edge_overflow
    else:
        if ctx.is_turn == 0.0:
            mult = 0.95
        else:
            mult = 0.985
    if mult < 0.9:
        mult = 0.9
    return mult
'''
        rows = [dict(die_id=d, edge_overflow=o, is_turn=t)
                for d in (0., 1.) for o in (0., .1, .5, 1.) for t in (0., 1.)]
        body = self.assert_cpp_matches(source, rows)
        self.assertEqual(body.count("double mult ="), 1)

    def test_documented_policy_examples_match_cpp(self):
        root = Path(__file__).resolve().parents[1] / "local_tasks"
        for task in ("global_routing_maze_cost", "global_routing_ripup_policy"):
            with self.subTest(task=task):
                prompt = (root / task / "prompt.md").read_text()
                source = re.findall(r"```python\n(.*?)```", prompt, re.DOTALL)[-1]
                self.assert_cpp_matches(source, [
                    dict(die_id=d, edge_overflow=o, history_pressure=h, detour_ratio=t)
                    for d in (0., 1.) for o in (0., .5, 1.)
                    for h in (0., 1.) for t in (0., 1.)
                ])

    def test_controller_elif_regression(self):
        source = '''def control_reroute_iteration(ctx):
    if ctx.congestion_trend < 0.4:
        trend_factor = 1.15
    elif ctx.congestion_trend > 0.6:
        trend_factor = 0.85
    else:
        trend_factor = 1.0
    return trend_factor, 1.0, 0.0
'''
        body = self.assert_cpp_matches(source, [dict(congestion_trend=v)
                                              for v in (0., .39, .4, .5, .6, .61, 1.)], 3)
        self.assertEqual(body.count("double trend_factor ="), 1)

    def test_ripup_lookup_tables_and_field_named_locals(self):
        source = '''def score_ripup_edge(ctx):
    weights = [0.2, 0.3, 0.5]
    features = (ctx["edge_overflow"], ctx.history_pressure, ctx.baseline_decision)
    thresholds = {"bottom": 0.55, "top": 0.62}
    die_id = ctx["die_id"]
    edge_overflow = ctx.edge_overflow
    score = 0.0
    for i in range(len(weights)):
        score += weights[i] * features[i]
    if die_id < 0.5:
        threshold = thresholds["bottom"]
    else:
        threshold = thresholds["top"]
    return 1.0 if score + edge_overflow * weights[-1] >= threshold else 0.0
'''
        rng = random.Random(71)
        rows = [dict(die_id=float(i % 2), edge_overflow=rng.random(),
                     history_pressure=rng.random(), baseline_decision=float(i % 3 == 0))
                for i in range(256)]
        self.assert_cpp_matches(source, rows)

    def test_net_ordering_documented_tables_match_cpp(self):
        source = '''def score_net_order(ctx):
    weights = (0.2, 0.8)
    values = [ctx.net_overflow, ctx.baseline_score]
    offsets = {"bottom": 0.0, "top": 0.1}
    score = offsets["bottom"] if ctx.die_id < 0.5 else offsets["top"]
    for i in range(len(weights)):
        score += weights[i] * values[i]
    return score
'''
        self.assert_cpp_matches(source, [dict(die_id=d, net_overflow=o, baseline_score=b)
                                        for d in (0., 1.) for o in (0., .5, 1.)
                                        for b in (.01, .5, 1.)])

    def test_table_values_are_snapshots_and_negative_loop_indices_work(self):
        source = '''def score_maze_step(ctx):
    x = ctx.edge_overflow
    values = (x, x + 1.0, x + 2.0)
    x = 100.0
    score = 0.0
    for i in range(-3, 0):
        if i == -2:
            continue
        score += values[i]
    return score
'''
        self.assert_cpp_matches(source, [dict(edge_overflow=v) for v in (0., .25, 1.)])

    def test_nested_early_return_does_not_hide_joined_assignment(self):
        source = '''def score_ripup_edge(ctx):
    if ctx.die_id < 0.5:
        if ctx.edge_overflow > 0.5:
            return 1.0
        else:
            score = 0.2
    else:
        if ctx.is_turn > 0.5:
            score = 0.7
        else:
            score = 0.4
    return score
'''
        self.assert_cpp_matches(source, [dict(die_id=d, edge_overflow=o, is_turn=t)
                                        for d in (0., 1.) for o in (0., 1.) for t in (0., 1.)])

    def test_undefined_reads_still_rejected_after_hoisting(self):
        for body in (
            "if ctx.die_id < 0.5:\n        x = y\n        y = 1.0\n    else:\n        x = 2.0\n        y = 3.0\n    return x + y",
            "if ctx.die_id < 0.5:\n        x = 1.0\n    return x",
        ):
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "undefined local"):
                self.translate("def score_maze_step(ctx):\n    " + body + "\n")

    def test_unsafe_or_unbounded_table_operations_rejected(self):
        bodies = [
            "return ctx['unknown']", "return ctx[ctx.die_id]",
            "t = [1., 2.]\n    return t[2]",
            "t = [1., 2.]\n    return t[-3]",
            "t = [1., 2.]\n    return t[ctx.die_id]",
            "t = [1., 2.]\n    t[0] = 3.\n    return t[0]",
            "t = [1., 2.]\n    t = [3., 4.]\n    return t[0]",
            "t = [1., 2.]\n    u = t\n    return u[0]",
            "t = [1., 2.]\n    t.append(3.)\n    return t[0]",
            "t = [1., 2.]\n    s = 0.\n    for i in range(3):\n        s += t[i]\n    return s",
            "t = {'x': 1., 'x': 2.}\n    return t['x']",
            "t = [[1.]]\n    return t[0]",
            "t = []\n    return 1.",
            "t = [" + ",".join(["1."] * 65) + "]\n    return t[0]",
            "ctx = [1.]\n    return ctx[0]",
            "open3d_table_local_1_0 = 1.\n    return open3d_table_local_1_0",
        ]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.translate("def score_maze_step(ctx):\n    " + body + "\n")

    def test_maze_loop_budget_also_applies_to_table_length(self):
        source = "def score_maze_step(ctx):\n    t = [" + ",".join(["1."] * 9) + "]\n    s = 0.\n    for i in range(len(t)):\n        s += t[i]\n    return s\n"
        with self.assertRaisesRegex(ValueError, "loop has 9 iterations"):
            self.translate(source, max_static_loop_iterations=8, max_nested_loop_iterations=16)


    def test_extended_log_and_expm1_match_compiled_scalar_policy(self):
        source = "def score_maze_step(ctx):\n    value = 1.0 + ctx.edge_overflow\n    return math.expm1(ctx.edge_overflow) + math.log2(value) + math.log10(value)\n"
        self.assert_cpp_matches(source, [dict(edge_overflow=value) for value in (0., 0.01, 0.5, 1.)])


if __name__ == "__main__":
    unittest.main()
