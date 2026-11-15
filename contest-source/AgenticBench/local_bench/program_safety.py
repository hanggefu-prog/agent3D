from __future__ import annotations

import ast
import difflib
from pathlib import Path
from typing import Iterable, Type


FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "delattr",
    "__import__",
}
FORBIDDEN_ROOTS = {
    "builtins",
    "ctypes",
    "importlib",
    "io",
    "os",
    "pathlib",
    "pickle",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
}

DEFAULT_MAXIMUM_AST_NODES = 3000
SAFE_EXCEPTION_TYPES = {
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "Exception",
    "FloatingPointError",
    "IndexError",
    "KeyError",
    "LookupError",
    "NotImplementedError",
    "OverflowError",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
}


def normalized_program(text: str, *, maximum_bytes: int = 96_000) -> str:
    if not isinstance(text, str) or len(text.encode("utf-8")) > maximum_bytes:
        raise ValueError("candidate program exceeds the task size limit")
    rows = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped in {
            "```",
            "```python",
            "# EVOLVE-BLOCK-START",
            "# EVOLVE-BLOCK-END",
        }:
            continue
        rows.append(line)
    return "\n".join(rows).strip() + "\n"


def single_definition(text: str, node_type: Type[ast.AST], name: str) -> tuple[str, ast.AST]:
    source = normalized_program(text)
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("candidate program is not valid Python: {}".format(error)) from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], node_type):
        raise ValueError("candidate program must contain exactly one {} named {}".format(node_type.__name__, name))
    node = tree.body[0]
    if getattr(node, "name", None) != name:
        raise ValueError("candidate entry point must be named {}".format(name))
    return source, node


def _exception_type_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for item in node.elts:
            names.update(_exception_type_names(item))
        return names
    return set()


def validate_no_external_effects(
    tree: ast.AST,
    *,
    allowed_attribute_roots: Iterable[str],
    maximum_ast_nodes: int = DEFAULT_MAXIMUM_AST_NODES,
) -> None:
    allowed_roots = set(allowed_attribute_roots)
    nodes = list(ast.walk(tree))
    if len(nodes) > maximum_ast_nodes:
        raise ValueError(
            "candidate definition has {} AST nodes; limit is {}".format(
                len(nodes), maximum_ast_nodes
            )
        )
    for node in nodes:
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.type is None:
                    raise ValueError("candidate cannot use a bare except handler")
                exception_types = _exception_type_names(handler.type)
                if not exception_types or not exception_types <= SAFE_EXCEPTION_TYPES:
                    raise ValueError(
                        "candidate try/except may catch only {}".format(
                            ", ".join(sorted(SAFE_EXCEPTION_TYPES))
                        )
                    )
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.Raise,
                ast.With,
            ),
        ):
            # PyTorch's no_grad context is useful for source-component evolution.
            if isinstance(node, ast.With) and all(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and isinstance(item.context_expr.func.value, ast.Name)
                and item.context_expr.func.value.id == "torch"
                and item.context_expr.func.attr == "no_grad"
                for item in node.items
            ):
                continue
            raise ValueError("candidate contains forbidden syntax: {}".format(type(node).__name__))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise ValueError("candidate calls forbidden function {}".format(node.func.id))
        if isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                if root.id in FORBIDDEN_ROOTS:
                    raise ValueError("candidate accesses forbidden module {}".format(root.id))
                if root.id not in allowed_roots and root.id not in {"self"}:
                    # Local tensors and values are allowed; this catches only known
                    # dangerous module roots rather than guessing local names.
                    continue
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_ROOTS:
            raise ValueError("candidate accesses forbidden name {}".format(node.id))


def replace_definition(source: str, node: ast.AST, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    start = int(getattr(node, "lineno")) - 1
    end = int(getattr(node, "end_lineno"))
    replacement_text = replacement.rstrip() + "\n"
    return "".join(lines[:start]) + replacement_text + "".join(lines[end:])


def unified_diff(relative_path: str, before: str, after: str) -> str:
    if before == after:
        return ""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/{}".format(relative_path),
            tofile="b/{}".format(relative_path),
        )
    )


def read_repo_file(root: Path, relative_path: str) -> str:
    path = (root / relative_path).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")
