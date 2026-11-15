"""Offline translation/compilation audit of exported routing prompt responses.

Input: [{"task": "bp-maze", "rows": [{"order": 1, "response": "..."}]}].
Does not execute candidate Python, route designs, or call a model provider.
"""

import argparse
import collections
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_tasks.global_routing_maze_cost import program_adapter as maze
from local_tasks.global_routing_ripup_policy import program_adapter as ripup
from local_tasks.routing_source_common import translate_scalar_program


def audit(groups):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("C++ compiler required")
    report = []
    for group in groups:
        if not group["task"].endswith(("-maze", "-ripup")):
            raise ValueError("Expected a task ending in -maze or -ripup")
        adapter = maze if group["task"].endswith("-maze") else ripup
        limits = dict(max_static_loop_iterations=8, max_nested_loop_iterations=16) if adapter is maze else {}
        accepted, errors, functions = [], [], []
        fields = " ".join("double " + key + ";" for key in sorted(adapter.CONTEXT_FIELDS))
        for row in group["rows"]:
            try:
                body = translate_scalar_program(
                    row["response"], entrypoint=adapter.ENTRYPOINT,
                    context_fields=adapter.CONTEXT_FIELDS, **limits
                )
                accepted.append(row["order"])
                functions.append("double policy_{}(const Context& ctx) {{\n{}\n}}\n".format(row["order"], body))
            except (ValueError, SyntaxError) as error:
                errors.append(dict(order=row["order"], error=str(error)))
        cpp = ("#include <algorithm>\n#include <cmath>\n#include <limits>\n"
               "struct Context { " + fields + " };\n" + "\n".join(functions))
        built = subprocess.run(
            [compiler, "-x", "c++", "-std=c++17", "-fsyntax-only", "-Werror=shadow", "-"],
            input=cpp, text=True, capture_output=True, timeout=60
        )
        report.append(dict(
            task=group["task"], candidates=len(group["rows"]),
            translated=len(accepted), rejected=len(errors), accepted_orders=accepted,
            cpp_compile_passed=built.returncode == 0, compiler_diagnostics=built.stderr,
            error_counts=dict(collections.Counter(row["error"] for row in errors)), errors=errors
        ))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.input.read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if all(row["cpp_compile_passed"] for row in result) else 1)
