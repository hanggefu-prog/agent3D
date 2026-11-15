from __future__ import annotations

import ast
import pprint
from pathlib import Path
from typing import Any, Dict, Optional

from .model import TaskSpec, validate_candidate_payload


FUNCTION_NAME = "evolve_candidate"


def render_candidate_program(
    candidate: Dict[str, Any],
    *,
    docstring: str = "Return one static OpenAgentic3DBench candidate.",
) -> str:
    value = validate_candidate_payload(candidate)
    literal = pprint.pformat(value, width=100, sort_dicts=True)
    return (
        "# EVOLVE-BLOCK-START\n"
        "def {}() -> dict:\n"
        "    {!r}\n"
        "    return {}\n"
        "# EVOLVE-BLOCK-END\n".format(
            FUNCTION_NAME,
            docstring,
            literal.replace("\n", "\n    "),
        )
    )


def parse_candidate_program(program: str) -> Dict[str, Any]:
    if not isinstance(program, str) or len(program.encode("utf-8")) > 2_100_000:
        raise ValueError("generated program must be UTF-8 text below 2.1 MB")
    try:
        tree = ast.parse(program, mode="exec")
    except SyntaxError as error:
        raise ValueError("generated candidate is not valid Python syntax: {}".format(error)) from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("generated program must contain exactly one candidate function")
    function = tree.body[0]
    if isinstance(function, ast.AsyncFunctionDef) or function.name != FUNCTION_NAME:
        raise ValueError("generated function must be named {}".format(FUNCTION_NAME))
    if function.decorator_list or function.args.args or function.args.kwonlyargs or function.args.vararg or function.args.kwarg:
        raise ValueError("candidate function cannot use decorators or arguments")
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        raise ValueError("candidate function body may contain only one literal return")
    try:
        value = ast.literal_eval(body[0].value)
    except (TypeError, ValueError) as error:
        raise ValueError("candidate return value must be a static Python literal") from error
    return validate_candidate_payload(value, source="generated candidate")


def evolution_prompt(
    spec: TaskSpec,
    initial_candidate: Dict[str, Any],
    open3dbench_root: Optional[Path] = None,
) -> str:
    # Keep the literal-program parser independent from prompt/program adapters.
    # Import lazily to avoid a cycle through prompting -> program_bridge.
    from .prompting import build_problem_description

    return build_problem_description(spec, initial_candidate, open3dbench_root)
