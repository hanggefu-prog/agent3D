"""Static HBT policy grammar and score contract.

Candidate execution belongs to the bounded native planner; this module only
parses source. hbt_native independently verifies physical evaluation evidence.
"""
import ast
import math

from .program_safety import normalized_program

NET_FIELDS = frozenset({
    "pin_count", "bottom_pin_count", "top_pin_count", "bottom_bbox_hpwl_dbu",
    "top_bbox_hpwl_dbu", "ideal_combined_hpwl_dbu", "die_span_dbu", "dbu_per_micron",
})
SITE_FIELDS = frozenset({
    "combined_hpwl_dbu", "bottom_hpwl_dbu", "top_hpwl_dbu",
    "ideal_displacement_dbu", "x_fraction", "y_fraction",
})
INTERFACE = {
    "net_priority": {"context": NET_FIELDS},
    "site_cost": {"net_context": NET_FIELDS, "site_context": SITE_FIELDS},
}


MATH_ARITY = {
    "sqrt": 1, "log": 1, "log1p": 1, "exp": 1, "expm1": 1,
    "tanh": 1, "fabs": 1, "floor": 1, "ceil": 1,
    "hypot": 2, "pow": 2,
}


def validate_policy(text):
    source = normalized_program(text, maximum_bytes=16000)
    tree = ast.parse(source)
    if len(list(ast.walk(tree))) > 500:
        raise ValueError("HBT policy exceeds 500 AST nodes")
    if (len(tree.body) != 2 or any(not isinstance(f, ast.FunctionDef) for f in tree.body)
            or [f.name for f in tree.body] != list(INTERFACE)):
        raise ValueError("HBT policy must contain only net_priority and site_cost, in that order")
    for function in tree.body:
        contexts = INTERFACE[function.name]
        args = function.args
        if (function.decorator_list or function.returns or function.type_comment
                or [a.arg for a in args.args] != list(contexts)
                or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
                or args.defaults or args.kw_defaults or any(a.annotation for a in args.args)):
            raise ValueError("HBT policy function signature must match the exact interface")

        def expression(node, locals_):
            if isinstance(node, ast.Constant):
                if node.value is None or isinstance(node.value, bool):
                    return
                if (isinstance(node.value, (int, float)) and abs(node.value) <= 1e12
                        and math.isfinite(node.value)):
                    return
            elif isinstance(node, ast.Name) and node.id in locals_:
                return
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                key = node.slice
                if isinstance(key, ast.Index):  # Python 3.8 in maintained EDA images.
                    key = key.value
                if (node.value.id in contexts and isinstance(key, ast.Constant)
                        and isinstance(key.value, str) and key.value in contexts[node.value.id]):
                    return
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                expression(node.left, locals_)
                expression(node.right, locals_)
                return
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
                expression(node.operand, locals_)
                return
            elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                for value in node.values:
                    expression(value, locals_)
                return
            elif isinstance(node, ast.Compare) and all(isinstance(op, (
                    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Is, ast.IsNot)) for op in node.ops):
                for value in [node.left] + node.comparators:
                    expression(value, locals_)
                return
            elif isinstance(node, ast.IfExp):
                for value in (node.test, node.body, node.orelse):
                    expression(value, locals_)
                return
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in {"abs", "min", "max"} and not node.keywords):
                count = len(node.args)
                if (node.func.id == "abs" and count == 1) or (node.func.id != "abs" and 2 <= count <= 8):
                    for value in node.args:
                        expression(value, locals_)
                    return
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == "math"
                  and node.func.attr in MATH_ARITY and not node.keywords
                  and len(node.args) == MATH_ARITY[node.func.attr]):
                for value in node.args:
                    expression(value, locals_)
                return
            raise ValueError("Unsupported HBT scalar expression or context field: " + type(node).__name__)

        def statements(body, locals_, *, allow_docstring=False):
            locals_ = set(locals_)
            for index, node in enumerate(body):
                if (allow_docstring and index == 0 and isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                    continue
                if isinstance(node, ast.Return):
                    expression(node.value, locals_)
                    if index != len(body) - 1:
                        raise ValueError("Unreachable HBT policy statements")
                    return locals_, True
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name.startswith("_") or name in contexts or name in {"abs", "min", "max", "math"}:
                        raise ValueError("HBT policy cannot modify inputs or reserved names")
                    expression(node.value, locals_)
                    locals_.add(name)
                elif isinstance(node, ast.If):
                    expression(node.test, locals_)
                    left, left_returns = statements(node.body, locals_)
                    right, right_returns = statements(node.orelse, locals_)
                    if left_returns and right_returns:
                        if index != len(body) - 1:
                            raise ValueError("Unreachable HBT policy statements")
                        return locals_, True
                    locals_ = right if left_returns else left if right_returns else left & right
                else:
                    raise ValueError("Unsupported HBT policy statement: " + type(node).__name__)
            return locals_, False

        _, returns = statements(function.body, set(), allow_docstring=True)
        if not returns:
            raise ValueError("HBT policy must return on every path")
    return source


def materialize_policy(text):
    return {
        "schema_version": 1, "config_overrides": {}, "source_patch": "",
        "candidate_files": {"hbt_policy.json": {
            "interface_version": "hbt-policy-v1", "source": validate_policy(text)}},
        "notes": "HBT priority and legal-site cost policy; native execution requires task qualification.",
    }


def validate_candidate(candidate):
    # Validate the literal candidate route as strictly as the program adapter.
    if candidate["config_overrides"] or candidate["source_patch"]:
        raise ValueError("HBT policy cannot change configuration or source files")
    files = candidate["candidate_files"]
    if set(files) != {"hbt_policy.json"}:
        raise ValueError("HBT policy requires exactly hbt_policy.json")
    policy = files["hbt_policy.json"]
    if not isinstance(policy, dict) or set(policy) != {"interface_version", "source"}:
        raise ValueError("HBT policy file has unsupported fields")
    if policy["interface_version"] != "hbt-policy-v1":
        raise ValueError("Unknown HBT policy interface")
    validate_policy(policy["source"])


def score_timing(candidate, reference):
    """Pure score-contract test helper; inputs are not physical qualification."""
    from .metrics import _relative_cost, RATIO_DENOMINATOR_FLOOR
    for values in (candidate, reference):
        for name in ("wns_setup", "tns_setup"):
            value = values.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Missing finite HBT setup timing: " + name)
        if values["tns_setup"] > 0:
            raise ValueError("Setup TNS must be non-positive")
    costs = [_relative_cost(max(0., -candidate[name]), max(0., -reference[name]))
             for name in ("wns_setup", "tns_setup")]
    cost = .6 * costs[0] + .4 * costs[1]
    score = 1. / max(cost, RATIO_DENOMINATOR_FLOOR)
    if not all(math.isfinite(value) for value in costs + [cost, score]):
        raise ValueError("Non-finite normalized HBT timing cost")
    return {"combined_score": score, "normalized_wns_cost": costs[0],
            "normalized_tns_cost": costs[1], "normalized_sta_cost": cost,
            "final_global_wns_setup_ns": candidate["wns_setup"],
            "final_global_tns_setup_ns": candidate["tns_setup"]}
