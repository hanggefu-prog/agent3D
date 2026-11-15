from __future__ import annotations

import ast
import unittest

from local_bench.program_safety import validate_no_external_effects


class ProgramSafetyTest(unittest.TestCase):
    def test_moderately_complex_definition_is_allowed(self) -> None:
        assignments = "\n".join(
            "    value_{0} = value + {0}".format(index) for index in range(240)
        )
        tree = ast.parse("def candidate(value):\n{}\n    return value_239\n".format(assignments))
        node_count = len(list(ast.walk(tree.body[0])))
        self.assertGreater(node_count, 1800)
        self.assertLess(node_count, 3000)
        validate_no_external_effects(tree.body[0], allowed_attribute_roots=set())

    def test_complexity_error_reports_count_and_limit(self) -> None:
        assignments = "\n".join(
            "    value_{0} = value + {0}".format(index) for index in range(400)
        )
        tree = ast.parse("def candidate(value):\n{}\n    return value_399\n".format(assignments))
        with self.assertRaisesRegex(ValueError, r"has \d+ AST nodes; limit is 3000"):
            validate_no_external_effects(tree.body[0], allowed_attribute_roots=set())

    def test_restricted_try_except_is_allowed_but_bare_except_is_not(self) -> None:
        safe = ast.parse(
            "def candidate(value):\n"
            "    try:\n"
            "        return 1.0 / value\n"
            "    except (ValueError, ArithmeticError):\n"
            "        return 0.0\n"
        )
        validate_no_external_effects(safe.body[0], allowed_attribute_roots=set())
        unsafe = ast.parse(
            "def candidate(value):\n"
            "    try:\n"
            "        return 1.0 / value\n"
            "    except:\n"
            "        return 0.0\n"
        )
        with self.assertRaisesRegex(ValueError, "bare except"):
            validate_no_external_effects(unsafe.body[0], allowed_attribute_roots=set())


if __name__ == "__main__":
    unittest.main()
