"""Frozen v1 scalar-loss language for the two GNN evolution tasks.

Existing task languages remain independent; widening their helpers must not
widen this capsule interface or disagree with the native loader.
"""
import ast
from local_bench.program_safety import validate_no_external_effects


SAFE_TORCH_APIS = {
    "abs",
    "clamp",
    "exp",
    "isfinite",
    "log",
    "log1p",
    "mean",
    "ones_like",
    "sigmoid",
    "softmax",
    "sqrt",
    "square",
    "std",
    "sum",
    "tanh",
    "var",
    "where",
    "zeros_like",
}


SAFE_FUNCTIONAL_APIS = {
    "cosine_similarity",
    "logsigmoid",
    "mse_loss",
    "normalize",
    "relu",
    "softplus",
}


FORBIDDEN_TENSOR_ATTRIBUTES = {
    "backward",
    "bmm",
    "cpu",
    "cuda",
    "data",
    "detach",
    "grad",
    "item",
    "matmul",
    "mm",
    "numpy",
    "requires_grad_",
    "to",
}


def _validate_assignment_target(target: ast.AST) -> None:
    if isinstance(target, ast.Name):
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            _validate_assignment_target(item)
        return
    raise ValueError("TP-GNN objective may assign only local variables")


def validate_loss_objective(node: ast.FunctionDef, expected_args) -> None:
    args = node.args
    if (
        [item.arg for item in args.args] != expected_args
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
        or args.kw_defaults
        or node.decorator_list
    ):
        raise ValueError("compute_tp_gnn_loss must keep the exact declared signature")

    forbidden = (
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.MatMult,
        ast.Lambda,
        ast.ClassDef,
        ast.FunctionDef,
    )
    for child in ast.walk(node):
        if child is not node and isinstance(child, forbidden):
            raise ValueError(
                "TP-GNN objective contains forbidden syntax: {}".format(
                    type(child).__name__
                )
            )
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            if child.func.id not in {"abs", "float", "int", "len", "max", "min"}:
                raise ValueError(
                    "TP-GNN objective calls unsupported function {}".format(
                        child.func.id
                    )
                )
        if isinstance(child, ast.Attribute):
            root = child.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "torch":
                if child.attr not in SAFE_TORCH_APIS:
                    raise ValueError(
                        "TP-GNN objective uses unsupported torch API {}".format(
                            child.attr
                        )
                    )
            elif isinstance(root, ast.Name) and root.id == "F":
                if child.attr not in SAFE_FUNCTIONAL_APIS:
                    raise ValueError(
                        "TP-GNN objective uses unsupported functional API {}".format(
                            child.attr
                        )
                    )
            if child.attr in FORBIDDEN_TENSOR_ATTRIBUTES or child.attr.endswith("_"):
                raise ValueError(
                    "TP-GNN objective uses forbidden tensor attribute {}".format(
                        child.attr
                    )
                )
        if isinstance(child, ast.Assign):
            for target in child.targets:
                _validate_assignment_target(target)
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
            _validate_assignment_target(child.target)
    validate_no_external_effects(node, allowed_attribute_roots={"torch", "F"})
