"""Shared trusted machinery for bounded FastRoute source-evolution tasks.

The public candidate is one side-effect-free Python function returning either
one scalar or a fixed tuple of scalars. The
shared scalar translator implements branches, early returns, and bounded loops.
These helpers parameterize the entrypoint, context
fields, and loop budget for other routing hot spots.  Materialization is
single-threaded, but a lock also makes the temporary translator configuration
safe for callers that prepare candidates concurrently.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from local_bench.program_safety import (
    read_repo_file,
    single_definition,
    unified_diff,
)
from local_tasks import routing_scalar_common as _net_order


ROUTE_BACKEND = "open3dbench/backend/three_d_route.py"
RUNNER = "open3dbench/runner.py"

_TRANSLATOR_LOCK = _net_order.TRANSLATOR_LOCK


def translate_scalar_program(
    program_text: str,
    *,
    entrypoint: str,
    context_fields: Iterable[str],
    max_static_loop_iterations: int = 64,
    max_nested_loop_iterations: int = 256,
    return_arity: int = 1,
    return_cpp_type: str = "double",
) -> str:
    """Validate one public function and return its translated C++ body."""

    _, node = single_definition(program_text, ast.FunctionDef, entrypoint)
    assert isinstance(node, ast.FunctionDef)
    with _TRANSLATOR_LOCK:
        old_fields = _net_order.CONTEXT_FIELDS
        old_static = _net_order.MAX_STATIC_LOOP_ITERATIONS
        old_nested = _net_order.MAX_NESTED_LOOP_ITERATIONS
        old_return_arity = _net_order.RETURN_ARITY
        old_return_cpp_type = _net_order.RETURN_CPP_TYPE
        try:
            _net_order.CONTEXT_FIELDS = set(context_fields)
            _net_order.MAX_STATIC_LOOP_ITERATIONS = max_static_loop_iterations
            _net_order.MAX_NESTED_LOOP_ITERATIONS = max_nested_loop_iterations
            _net_order.RETURN_ARITY = return_arity
            _net_order.RETURN_CPP_TYPE = return_cpp_type
            try:
                return _net_order._validate_and_translate(node)
            except ValueError as error:
                raise ValueError(
                    str(error).replace("routing_policy", entrypoint)
                ) from error
        finally:
            _net_order.CONTEXT_FIELDS = old_fields
            _net_order.MAX_STATIC_LOOP_ITERATIONS = old_static
            _net_order.MAX_NESTED_LOOP_ITERATIONS = old_nested
            _net_order.RETURN_ARITY = old_return_arity
            _net_order.RETURN_CPP_TYPE = old_return_cpp_type


def replace_once(source: str, needle: str, replacement: str, label: str) -> str:
    return _net_order._replace_once(source, needle, replacement, label)


def policy_source(
    body: str,
    *,
    struct_name: str,
    function_name: str,
    context_fields: Iterable[str],
    return_type: str = "double",
    extra_declarations: str = "",
) -> str:
    fields = "\n".join("  double {};".format(name) for name in sorted(context_fields))
    return (
        "namespace {\n\n"
        "struct "
        + struct_name
        + "\n{\n"
        + fields
        + "\n};\n\n"
        + (extra_declarations.rstrip() + "\n\n" if extra_declarations else "")
        + return_type
        + " "
        + function_name
        + "(const "
        + struct_name
        + "& ctx)\n{\n"
        + body
        + "\n}\n\n"
        "}  // namespace\n"
    )


def standalone_grt_supported(open3dbench_root: Path) -> bool:
    """Recognize the maintained standalone boundary without changing its source."""
    runner = read_repo_file(open3dbench_root, RUNNER)
    route = read_repo_file(open3dbench_root, ROUTE_BACKEND)
    methods = [n for n in ast.walk(ast.parse(runner))
               if isinstance(n, ast.FunctionDef) and n.name == "global_route"]
    if not methods:
        return False
    if (len(methods) != 1 or '"global_route_checkpoint.json"' not in route
            or '"grt_metrics.json"' not in route
            or "grt_only=True, output_dir=destination" not in runner):
        raise ValueError("Unsupported standalone GRT implementation contract")
    return True


def grt_only_patches(open3dbench_root: Path) -> tuple[str, str]:
    """Materialize the common frozen CTS -> two-die GRT evaluator boundary."""

    route_before = read_repo_file(open3dbench_root, ROUTE_BACKEND)
    runner_before = read_repo_file(open3dbench_root, RUNNER)
    # Current flow already owns the independent GRT command/checkpoint. Preserve
    # that boundary and expose only the existing die/HBT inputs to the hook.
    # Do not transplant the historical combined GRT/DRT evaluator into it.
    if standalone_grt_supported(open3dbench_root):
        if ('OPEN3D_ROUTING_DIE=die' in route_before
                and 'OPEN3D_HBT_PORT_FILE=str(die_dir / "hbt_ports.tsv")' in route_before):
            return route_before, runner_before
        anchor = '        grt_env = dict(env, ROUTE_GUIDE=env["ROUTE_GRT_GUIDE"])\n'
        route_after = replace_once(route_before, anchor,
            '        grt_env = dict(env, ROUTE_GUIDE=env["ROUTE_GRT_GUIDE"],\n'
            '                       OPEN3D_ROUTING_DIE=die,\n'
            '                       OPEN3D_HBT_PORT_FILE=str(die_dir / "hbt_ports.tsv"))\n',
            "standalone GRT hook context")
        return route_after, runner_before
    return (
        _net_order._patch_route_backend(route_before),
        _net_order._patch_runner(runner_before),
    )


def materialized_patch(
    *,
    source_path: str,
    source_before: str,
    source_after: str,
    route_after: str,
    runner_after: str,
    open3dbench_root: Path,
) -> str:
    route_before = read_repo_file(open3dbench_root, ROUTE_BACKEND)
    runner_before = read_repo_file(open3dbench_root, RUNNER)
    return (
        unified_diff(source_path, source_before, source_after)
        + unified_diff(ROUTE_BACKEND, route_before, route_after)
        + unified_diff(RUNNER, runner_before, runner_after)
    )
