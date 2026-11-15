"""Shared bounded C++ translator and GRT handoff; no net-ordering task hook."""
from __future__ import annotations

import ast
import copy
import math
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from local_bench.program_safety import (
    read_repo_file,
    single_definition,
    unified_diff,
    validate_no_external_effects,
)
from local_tasks.routing_language import lower_bounded_tables


ROUTE_BACKEND = "open3dbench/backend/three_d_route.py"
RUNNER = "open3dbench/runner.py"
CONTEXT_FIELDS = set()  # Each current policy supplies its own explicit fields.

SAFE_CALLS = {"abs", "float", "max", "min", "round"}
SAFE_MATH_UNARY_CALLS = {
    "acos", "asin", "atan", "ceil", "cos", "erf", "exp", "fabs", "floor",
    "log", "log1p", "log2", "log10", "expm1", "sin", "sqrt", "tanh",
}
SAFE_MATH_BINARY_CALLS = {"atan2", "copysign", "fmod", "hypot", "pow"}
MAX_STATIC_LOOP_ITERATIONS = 64
MAX_NESTED_LOOP_ITERATIONS = 256
RETURN_ARITY = 1
RETURN_CPP_TYPE = "double"
TRANSLATOR_LOCK = threading.Lock()
CPP_RESERVED = {
    "alignas", "alignof", "and", "asm", "auto", "bitand", "bitor", "bool",
    "break", "case", "catch", "char", "class", "compl", "concept", "const",
    "consteval", "constexpr", "constinit", "continue", "co_await", "co_return",
    "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
    "else", "enum", "explicit", "export", "extern", "false", "float", "for",
    "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace",
    "new", "noexcept", "not", "nullptr", "operator", "or", "private", "protected",
    "public", "register", "reinterpret_cast", "requires", "return", "short",
    "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
    "switch", "template", "this", "thread_local", "throw", "true", "try",
    "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual",
    "void", "volatile", "wchar_t", "while", "xor",
}


class _CppExpression:
    """Translate a deliberately small, side-effect-free Python subset to C++."""

    def __init__(self, defined: Iterable[str]) -> None:
        self.defined: Set[str] = set(defined)

    def expression(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                value = float(node.value)
                if not math.isfinite(value):
                    raise ValueError("routing_policy constants must be finite")
                return repr(value)
            raise ValueError("routing_policy supports only numeric and boolean constants")
        if isinstance(node, ast.Name):
            if node.id not in self.defined:
                raise ValueError("routing_policy reads undefined local {!r}".format(node.id))
            return node.id
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "ctx" and node.attr in CONTEXT_FIELDS:
                return "ctx.{}".format(node.attr)
            raise ValueError("routing_policy may access only declared ctx fields")
        if isinstance(node, ast.BinOp):
            operators: Dict[type, str] = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }
            if isinstance(node.op, ast.Mod):
                return "std::fmod({}, {})".format(
                    self.expression(node.left), self.expression(node.right)
                )
            if isinstance(node.op, ast.Pow):
                return "std::pow({}, {})".format(
                    self.expression(node.left), self.expression(node.right)
                )
            if isinstance(node.op, ast.FloorDiv):
                return "std::floor({} / {})".format(
                    self.expression(node.left), self.expression(node.right)
                )
            symbol = operators.get(type(node.op))
            if symbol is None:
                raise ValueError("routing_policy uses unsupported binary operator")
            return "({} {} {})".format(
                self.expression(node.left), symbol, self.expression(node.right)
            )
        if isinstance(node, ast.UnaryOp):
            operators = {ast.UAdd: "+", ast.USub: "-", ast.Not: "!"}
            symbol = operators.get(type(node.op))
            if symbol is None:
                raise ValueError("routing_policy uses unsupported unary operator")
            return "({}{})".format(symbol, self.expression(node.operand))
        if isinstance(node, ast.BoolOp):
            symbol = " && " if isinstance(node.op, ast.And) else " || " if isinstance(node.op, ast.Or) else None
            if symbol is None:
                raise ValueError("routing_policy uses unsupported boolean operator")
            return "(" + symbol.join(self.expression(value) for value in node.values) + ")"
        if isinstance(node, ast.Compare):
            operators = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
            }
            left = self.expression(node.left)
            comparisons = []
            for op, comparator in zip(node.ops, node.comparators):
                symbol = operators.get(type(op))
                if symbol is None:
                    raise ValueError("routing_policy uses unsupported comparison")
                right = self.expression(comparator)
                comparisons.append("({} {} {})".format(left, symbol, right))
                left = right
            return "(" + " && ".join(comparisons) + ")"
        if isinstance(node, ast.IfExp):
            return "({} ? {} : {})".format(
                self.expression(node.test),
                self.expression(node.body),
                self.expression(node.orelse),
            )
        if isinstance(node, ast.Call):
            return self._call(node)
        raise ValueError(
            "routing_policy contains unsupported expression {}".format(
                type(node).__name__
            )
        )

    def _call(self, node: ast.Call) -> str:
        if node.keywords:
            raise ValueError("routing_policy calls do not accept keyword arguments")
        args = [self.expression(arg) for arg in node.args]
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_CALLS:
            name = node.func.id
            if name in {"abs", "float", "round"}:
                if len(args) != 1:
                    raise ValueError("{} expects one argument".format(name))
                if name == "abs":
                    return "std::abs({})".format(args[0])
                if name == "round":
                    return "std::round({})".format(args[0])
                return "static_cast<double>({})".format(args[0])
            if len(args) < 2:
                raise ValueError("{} expects at least two arguments".format(name))
            cpp = "std::{}({}, {})".format(name, args[0], args[1])
            for arg in args[2:]:
                cpp = "std::{}({}, {})".format(name, cpp, arg)
            return cpp
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
        ):
            if node.func.attr in SAFE_MATH_UNARY_CALLS:
                if len(args) != 1:
                    raise ValueError("math.{} expects one argument".format(node.func.attr))
                return "std::{}({})".format(node.func.attr, args[0])
            if node.func.attr in SAFE_MATH_BINARY_CALLS:
                if len(args) != 2:
                    raise ValueError("math.{} expects two arguments".format(node.func.attr))
                return "std::{}({}, {})".format(node.func.attr, args[0], args[1])
        raise ValueError("routing_policy calls an unsupported function")


def _block_returns(statements: list[ast.stmt]) -> bool:
    for statement in statements:
        if isinstance(statement, ast.Return):
            return True
        if isinstance(statement, ast.If) and statement.orelse:
            if _block_returns(statement.body) and _block_returns(statement.orelse):
                return True
    return False


def _literal_integer(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        value = int(node.operand.value)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("routing_policy range bounds must be integer constants")


def _static_range(node: ast.AST) -> tuple[int, int, int, int]:
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "range"
        or node.keywords
        or not 1 <= len(node.args) <= 3
    ):
        raise ValueError("routing_policy loops require range() with constant bounds")
    values = [_literal_integer(arg) for arg in node.args]
    if len(values) == 1:
        start, stop, step = 0, values[0], 1
    elif len(values) == 2:
        start, stop, step = values[0], values[1], 1
    else:
        start, stop, step = values
    if step == 0:
        raise ValueError("routing_policy range step cannot be zero")
    iterations = len(range(start, stop, step))
    if iterations > MAX_STATIC_LOOP_ITERATIONS:
        raise ValueError(
            "routing_policy loop has {} iterations; limit is {}".format(
                iterations, MAX_STATIC_LOOP_ITERATIONS
            )
        )
    return start, stop, step, iterations


def _fallthrough_definitions(
    statements: list[ast.stmt], defined: Set[str]
) -> Optional[Set[str]]:
    """Return names defined on every path that continues past ``statements``.

    C++ block scope is narrower than Python function-local scope.  In
    particular, a scalar assigned in both arms of an ``if`` is available after
    the branch in Python, but declaring it independently inside the generated
    C++ arms would hide it.  This small definite-assignment analysis lets the
    translator hoist only names that are guaranteed to exist on every
    fall-through path.  ``None`` means that no path falls through.
    """

    current = set(defined)
    for statement in statements:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                current.add(statement.targets[0].id)
            continue
        if isinstance(statement, (ast.Return, ast.Break, ast.Continue)):
            return None
        if isinstance(statement, ast.If):
            body = _fallthrough_definitions(statement.body, current)
            other = (
                _fallthrough_definitions(statement.orelse, current)
                if statement.orelse
                else set(current)
            )
            if body is None and other is None:
                return None
            if body is None:
                current = set(other or ())
            elif other is None:
                current = set(body)
            else:
                current = set(body) & set(other)
            continue
        # A statically bounded loop may still execute zero times or leave via
        # break, so assignments first made inside it are conservatively local.
    return current


def _translate_statements(
    statements: list[ast.stmt],
    defined: Set[str],
    indent: int,
    *,
    declared: Optional[Set[str]] = None,
    loop_depth: int = 0,
    loop_variables: Optional[Set[str]] = None,
    nested_iterations: int = 1,
) -> list[str]:
    rows: list[str] = []
    prefix = " " * indent
    translator = _CppExpression(defined)
    declared = set(defined) if declared is None else set(declared)
    loop_variables = set(loop_variables or ())
    for index, statement in enumerate(statements):
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise ValueError("routing_policy assignments require one local scalar name")
            name = statement.targets[0].id
            if (
                name == "ctx"
                or name in SAFE_CALLS
                or name == "math"
                or name in CPP_RESERVED
            ):
                raise ValueError("routing_policy cannot assign reserved name {!r}".format(name))
            if name in loop_variables:
                raise ValueError("routing_policy cannot assign loop variable {!r}".format(name))
            value = translator.expression(statement.value)
            if name in declared:
                rows.append("{}{} = {};".format(prefix, name, value))
            else:
                rows.append("{}double {} = {};".format(prefix, name, value))
                declared.add(name)
            defined.add(name)
            translator.defined.add(name)
            continue
        if isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name):
                raise ValueError("routing_policy augmented assignments require a local scalar")
            name = statement.target.id
            if name not in defined:
                raise ValueError("routing_policy augments undefined local {!r}".format(name))
            if name in loop_variables:
                raise ValueError("routing_policy cannot assign loop variable {!r}".format(name))
            operators: Dict[type, str] = {
                ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/=",
            }
            symbol = operators.get(type(statement.op))
            if symbol is None:
                raise ValueError("routing_policy uses unsupported augmented operator")
            rows.append(
                "{}{} {} {};".format(
                    prefix, name, symbol, translator.expression(statement.value)
                )
            )
            continue
        if isinstance(statement, ast.If):
            fallthrough = _fallthrough_definitions([statement], defined)
            joined = (
                set()
                if fallthrough is None
                else set(fallthrough) - set(defined)
            )
            # A parent branch may already have hoisted storage even though
            # this path has not assigned a value yet. Never shadow that storage.
            for name in sorted(joined - declared):
                if (
                    name == "ctx"
                    or name in SAFE_CALLS
                    or name == "math"
                    or name in CPP_RESERVED
                    or name in loop_variables
                ):
                    raise ValueError(
                        "routing_policy cannot assign reserved name {!r}".format(name)
                    )
                rows.append("{}double {} = 0.0;".format(prefix, name))
            branch_declared = set(declared) | joined
            rows.append("{}if ({}) {{".format(prefix, translator.expression(statement.test)))
            rows.extend(
                _translate_statements(
                    statement.body,
                    set(defined),
                    indent + 2,
                    declared=branch_declared,
                    loop_depth=loop_depth,
                    loop_variables=loop_variables,
                    nested_iterations=nested_iterations,
                )
            )
            if statement.orelse:
                rows.append("{}}} else {{".format(prefix))
                rows.extend(
                    _translate_statements(
                        statement.orelse,
                        set(defined),
                        indent + 2,
                        declared=branch_declared,
                        loop_depth=loop_depth,
                        loop_variables=loop_variables,
                        nested_iterations=nested_iterations,
                    )
                )
            rows.append("{}}}".format(prefix))
            defined.update(joined)
            declared.update(joined)
            translator.defined.update(joined)
            continue
        if isinstance(statement, ast.For):
            if not isinstance(statement.target, ast.Name):
                raise ValueError("routing_policy loop target must be one local name")
            if statement.orelse:
                raise ValueError("routing_policy for/else is not supported")
            name = statement.target.id
            if (
                name in defined
                or name in SAFE_CALLS
                or name in {"ctx", "math", "range"}
                or name in CPP_RESERVED
            ):
                raise ValueError("routing_policy loop variable is reserved: {!r}".format(name))
            start, stop, step, iterations = _static_range(statement.iter)
            total_iterations = nested_iterations * max(1, iterations)
            if total_iterations > MAX_NESTED_LOOP_ITERATIONS:
                raise ValueError(
                    "routing_policy nested loops have up to {} iterations; limit is {}".format(
                        total_iterations, MAX_NESTED_LOOP_ITERATIONS
                    )
                )
            comparison = "<" if step > 0 else ">"
            rows.append(
                "{}for (int {} = {}; {} {} {}; {} += {}) {{".format(
                    prefix, name, start, name, comparison, stop, name, step
                )
            )
            rows.extend(
                _translate_statements(
                    statement.body,
                    set(defined) | {name},
                    indent + 2,
                    declared=set(declared) | {name},
                    loop_depth=loop_depth + 1,
                    loop_variables=loop_variables | {name},
                    nested_iterations=total_iterations,
                )
            )
            rows.append("{}}}".format(prefix))
            continue
        if isinstance(statement, ast.Break):
            if loop_depth == 0:
                raise ValueError("routing_policy break must be inside a bounded loop")
            rows.append("{}break;".format(prefix))
            continue
        if isinstance(statement, ast.Continue):
            if loop_depth == 0:
                raise ValueError("routing_policy continue must be inside a bounded loop")
            rows.append("{}continue;".format(prefix))
            continue
        if isinstance(statement, ast.Pass):
            rows.append("{};".format(prefix))
            continue
        if isinstance(statement, ast.Return):
            if statement.value is None:
                raise ValueError("routing_policy must return a numeric score")
            if RETURN_ARITY == 1:
                rows.append(
                    "{}return {};".format(
                        prefix, translator.expression(statement.value)
                    )
                )
            else:
                if (
                    not isinstance(statement.value, ast.Tuple)
                    or len(statement.value.elts) != RETURN_ARITY
                ):
                    raise ValueError(
                        "routing_policy must return exactly {} numeric values".format(
                            RETURN_ARITY
                        )
                    )
                values = ", ".join(
                    translator.expression(value) for value in statement.value.elts
                )
                rows.append(
                    "{}return {}{{{}}};".format(
                        prefix, RETURN_CPP_TYPE, values
                    )
                )
            continue
        raise ValueError(
            "routing_policy contains unsupported statement {}".format(
                type(statement).__name__
            )
        )
    return rows


def _validate_and_translate(node: ast.FunctionDef) -> str:
    node = copy.deepcopy(node)
    args = node.args
    if (
        [item.arg for item in args.args] != ["ctx"]
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
        or args.kw_defaults
        or node.decorator_list
        or node.returns is not None
    ):
        raise ValueError("routing_policy must keep the exact one-argument signature")
    forbidden = (
        ast.AsyncFor,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Lambda,
        ast.ClassDef,
        ast.FunctionDef,
        ast.NamedExpr,
    )
    for child in ast.walk(node):
        if child is not node and isinstance(child, forbidden):
            raise ValueError(
                "routing_policy contains forbidden syntax: {}".format(
                    type(child).__name__
                )
            )
    validate_no_external_effects(
        node,
        allowed_attribute_roots={"ctx", "math"},
        maximum_ast_nodes=12_000,
    )
    node = lower_bounded_tables(node, CONTEXT_FIELDS, _static_range)
    validate_no_external_effects(
        node, allowed_attribute_roots={"ctx", "math"}, maximum_ast_nodes=12_000
    )
    if not _block_returns(node.body):
        raise ValueError("routing_policy must return on every control path")
    body = _translate_statements(node.body, set(), 2)
    return "\n".join(body)


def _replace_once(source: str, needle: str, replacement: str, label: str) -> str:
    count = source.count(needle)
    if count != 1:
        raise ValueError(
            "Open3DBench anchor {} occurs {} times, expected once".format(label, count)
        )
    return source.replace(needle, replacement, 1)


def _patch_route_backend(before: str) -> str:
    if (
        'choices=("all", "global", "detailed")' in before
        and "OPEN3D_ROUTING_DIE" in before
        and "OPEN3D_HBT_PORT_FILE" in before
        and "OPEN3D_GLOBAL_ROUTE_DONE" in before
    ):
        # The maintained backend now owns the GRT-only contract and exposes
        # the runtime die/HBT context directly.  Candidate code should modify
        # only the selected FastRoute hook, never duplicate the harness.
        return before
    grt_only_tcl = r'''
if {[info exists ::env(OPEN3D_GRT_ONLY)] && $::env(OPEN3D_GRT_ONLY) eq "1"} {
  # The source-evolution benchmark ends at global routing. Preserve enough
  # physical output for auditability without publishing an extraction-ready
  # detailed-route checkpoint.
  set grt_plan [open $::env(ROUTE_HBT_PLAN) r]
  set grt_boundary_map [open $::env(ROUTE_BOUNDARY_NETS) w]
  while {[gets $grt_plan line] >= 0} {
    if {$line eq ""} { continue }
    lassign [split $line "\t"] hbt_id port_name expected_x expected_y
    set bterm [$block findBTerm $port_name]
    if {$bterm eq "NULL" || [$bterm getNet] eq "NULL"} {
      error "lost GRT HBT boundary port $port_name"
    }
    puts $grt_boundary_map "$port_name\t[[$bterm getNet] getName]"
  }
  close $grt_plan
  close $grt_boundary_map
  write_db $::env(ROUTE_OUTPUT_ODB)
  write_def $::env(ROUTE_OUTPUT_DEF)
  write_verilog $::env(ROUTE_OUTPUT_VERILOG)
  puts "OPEN3D_GRT_ONLY_DONE die=$::env(ROUTE_DIE)"
  exit
}
'''
    after = _replace_once(
        before,
        "write_guides $::env(ROUTE_GUIDE)\n\nset drt_args",
        "write_guides $::env(ROUTE_GUIDE)\n"
        + grt_only_tcl
        + "\nset drt_args",
        "GRT-only Tcl exit",
    )
    after = _replace_once(
        after,
        "import json\n",
        "import json\nimport math\n",
        "route math import",
    )
    grt_parser = '''def _grt_wire_length_um(log: Path) -> float:
    """Read FastRoute's global-route wire length, never a DRT report."""

    if not log.is_file():
        raise ThreeDRouteError("missing global-route log: {}".format(log))
    matches = re.findall(
        r"^\\[INFO GRT-0018\\]\\s+Total wirelength:\\s*([0-9.eE+-]+)\\s+um\\s*$",
        log.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    if not matches:
        raise ThreeDRouteError("OpenROAD log has no GRT wire length: {}".format(log))
    value = float(matches[-1])
    if not math.isfinite(value) or value < 0.0:
        raise ThreeDRouteError("OpenROAD log has invalid GRT wire length: {}".format(log))
    return value


'''
    after = _replace_once(
        after,
        "def _wire_length_summary(log: Path) -> Dict[str, object]:\n",
        grt_parser + "def _wire_length_summary(log: Path) -> Dict[str, object]:\n",
        "GRT wire-length parser",
    )
    after = _replace_once(
        after,
        "    started = time.perf_counter()\n"
        "    checkpoint_path = checkpoint_path.resolve()\n",
        "    started = time.perf_counter()\n"
        "    grt_only = os.environ.get(\"OPEN3D_GRT_ONLY\") == \"1\"\n"
        "    checkpoint_path = checkpoint_path.resolve()\n",
        "GRT-only mode initialization",
    )
    signature = (
        "def _run_openroad(\n"
        "    script: Path,\n"
        "    log: Path,\n"
        "    executable: str,\n"
        "    *,\n"
        "    resource_reduction: float = 0.0,\n"
        ") -> Dict[str, object]:\n"
    )
    timed_signature = signature.replace(
        "    resource_reduction: float = 0.0,\n",
        "    resource_reduction: float = 0.0,\n"
        "    timeout_seconds: int = 10800,\n",
    )
    if timed_signature in after:
        signature = timed_signature
    replacement = signature.replace(
        "    resource_reduction: float = 0.0,\n",
        "    resource_reduction: float = 0.0,\n"
        "    die: str,\n"
        "    hbt_port_file: Path,\n",
    )
    after = _replace_once(after, signature, replacement, "route OpenROAD signature")
    after = _replace_once(
        after,
        "    with log.open(\"w\", encoding=\"utf-8\") as output:\n",
        "    run_environment = dict(os.environ)\n"
        "    run_environment[\"OPEN3D_ROUTING_DIE\"] = die\n"
        "    run_environment[\"OPEN3D_HBT_PORT_FILE\"] = str(hbt_port_file)\n"
        "    with log.open(\"w\", encoding=\"utf-8\") as output:\n",
        "route OpenROAD environment",
    )
    after = _replace_once(
        after,
        "env=dict(os.environ),\n",
        "env=run_environment,\n",
        "route subprocess environment",
    )
    call = (
        "            warning_audit = _run_openroad(\n"
        "                script,\n"
        "                die_dir / \"route.log\",\n"
        "                openroad,\n"
        "                resource_reduction=resource_reduction,\n"
        "            )\n"
    )
    timed_call = call.replace(
        "                resource_reduction=resource_reduction,\n",
        "                resource_reduction=resource_reduction,\n"
        "                timeout_seconds=die_timeout_seconds,\n",
    )
    if timed_call in after:
        call = timed_call
    replacement_call = call.replace(
        "                resource_reduction=resource_reduction,\n",
        "                resource_reduction=resource_reduction,\n"
        "                die=die,\n"
        "                hbt_port_file=hbt_plans[die],\n",
    )
    after = _replace_once(after, call, replacement_call, "route OpenROAD call")
    grt_die_metrics = '''        if grt_only:
            warning_count += int(warning_audit["warning_count"])
            grt_overflow_count = _grt_overflow_count(die_dir / "route.log")
            grt_congestion_count = _grt_congestion_count(
                die_dir, grt_overflow_count
            )
            grt_wirelength = _grt_wire_length_um(die_dir / "route.log")
            for artifact in (
                "design.odb", "design.def", "design.v", "design.sdc",
                "route.guide", "global_route.odb", "boundary_nets.tsv",
            ):
                if not (die_dir / artifact).is_file():
                    raise ThreeDRouteError(
                        "missing {} GRT artifact: {}".format(die, artifact)
                    )
            dies[die] = {
                "status": "success",
                "layer_window": [min_layer, max_layer],
                "ndr_policy": ndr_policies[die],
                "hbt_boundary_ports": hbt_count,
                "grt_congestion_bins": grt_congestion_count,
                "grt_total_overflow": grt_overflow_count,
                "grt_wirelength_um": grt_wirelength,
            }
            continue
'''
    after = _replace_once(
        after,
        replacement_call + "        boundary_rows = [\n",
        replacement_call + grt_die_metrics + "        boundary_rows = [\n",
        "per-die GRT-only metrics",
    )
    publish = '''    if grt_only:
        _copy_continuation(source, output_dir)
        metrics = {
            "schema_version": 1,
            "stage": "route-grt",
            "status": "success",
            "algorithm": "openroad-die-local-net-order",
            "physical_model": "two-independent-global-routed-odbs",
            "hbt_model": "boundary-bterm-global-route-guide",
            "configuration": {
                "threads": min(threads, 32),
                "congestion_iterations": congestion_iterations,
                "resource_reduction": resource_reduction,
                "ndr_policy": ndr_policies,
                "grt_only": True,
            },
            "dies": dies,
            "warning_count": warning_count,
            "unexpected_warnings": 0,
            "ready_for_extract": False,
            "runtime_seconds": time.perf_counter() - started,
        }
        _write_json(output_dir / "metrics.json", metrics)
        source_checkpoint_ref = _relative_ref(checkpoint_path, output_dir)
        route_checkpoint = dict(checkpoint)
        route_checkpoint.update(
            {
                "stage": "route-grt",
                "source_checkpoint": source_checkpoint_ref,
                "dies": {
                    die: {
                        "odb": die + "/design.odb",
                        "def": die + "/design.def",
                        "verilog": die + "/design.v",
                        "sdc": die + "/design.sdc",
                        "guide": die + "/route.guide",
                    }
                    for die in ("bottom", "top")
                },
                "assembly": _assembly_checkpoint_refs(checkpoint["assembly"]),
                "route": {
                    "algorithm": "openroad-die-local-net-order-grt-only",
                    "metrics": "metrics.json",
                },
                "ready_for_extract": False,
            }
        )
        for key, filename in (
            ("hbt_assignment", "hbt.csv"),
            ("name_map", "name_map.csv"),
            ("partition_assignment", "partition_assignment.csv"),
            ("timing_model", "timing_model.json"),
        ):
            if (output_dir / filename).is_file():
                route_checkpoint[key] = filename
        _write_json(output_dir / "checkpoint.json", route_checkpoint)
        return output_dir / "checkpoint.json"

'''
    return _replace_once(
        after,
        "    _copy_continuation(source, output_dir)\n"
        "    total_drc = sum(int(dies[die][\"drc_violations\"]) for die in dies)\n",
        publish
        + "    _copy_continuation(source, output_dir)\n"
        + "    total_drc = sum(int(dies[die][\"drc_violations\"]) for die in dies)\n",
        "GRT-only publication",
    )


def _patch_runner(before: str) -> str:
    if '"global_routing": "75_global_route"' in before:
        return before
    after = _replace_once(
        before,
        "        artifacts = {name: root / relative for name, relative in names.items()}\n",
        "        if stage == \"routing\" and os.environ.get(\"OPEN3D_GRT_ONLY\") == \"1\":\n"
        "            names.pop(\"bottom_drc\")\n"
        "            names.pop(\"top_drc\")\n"
        "        artifacts = {name: root / relative for name, relative in names.items()}\n",
        "GRT-only stage artifacts",
    )
    routing_tail = (
        "        self._command(command, self.repo, stage / \"routing.log\")\n"
        "        validate_checkpoint(stage / \"checkpoint.json\", require_files=True)\n\n"
        "    def _run_report(self) -> None:\n"
    )
    return _replace_once(
        after,
        routing_tail,
        "        self._command(command, self.repo, stage / \"routing.log\")\n"
        "        if os.environ.get(\"OPEN3D_GRT_ONLY\") != \"1\":\n"
        "            validate_checkpoint(stage / \"checkpoint.json\", require_files=True)\n\n"
        "    def _run_report(self) -> None:\n",
        "GRT-only checkpoint validation",
    )
