"""Bounded scalar-loop grammar used by the current Nesterov controller."""

import ast

BOUNDED_SIGNAL_NAMES = {
    "overflow_by_die",
    "log_lambda_by_die",
    "log_gradient_norm_by_die",
    "die_multipliers_prev",
}

SAFE_DIRECT_CALLS = {"abs", "bool", "enumerate", "float", "int", "len", "max", "min", "range", "round", "sum", "zip"}

def _is_small_integer_constant(node: ast.AST, maximum: int = 8) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and 0 <= node.value <= maximum
    )

def _is_bounded_iteration(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in BOUNDED_SIGNAL_NAMES
    if isinstance(node, (ast.Tuple, ast.List)):
        return len(node.elts) <= 8
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    if node.func.id == "range":
        return (
            not node.keywords
            and 1 <= len(node.args) <= 3
            and all(_is_small_integer_constant(argument) for argument in node.args)
        )
    if node.func.id in {"enumerate", "zip"}:
        return (
            not node.keywords
            and bool(node.args)
            and all(_is_bounded_iteration(argument) for argument in node.args)
        )
    return False
