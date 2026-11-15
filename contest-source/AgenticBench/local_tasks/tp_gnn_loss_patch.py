"""Shared loss AST validation and native training-hook patch primitives."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from local_bench.numerical_programs import lower_torch_compat, validate_helpers, validate_loss_loop_work
from local_tasks.placement_control_language import _is_bounded_iteration

from local_bench.program_safety import (
    read_repo_file,
    single_definition,
    unified_diff,
    validate_no_external_effects,
)


OBJECTIVE_ARGS = [
    "positive_scores",
    "negative_scores",
    "embeddings",
    "positive_sources",
    "positive_targets",
    "negative_sources",
    "negative_targets",
    "source_batch_size",
    "epoch",
    "max_epochs",
]
TP_GNN = "Place-3D/src/partition/lol/methods/tp_gnn.py"
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
    "amin", "amax", "argmax", "argmin", "argsort", "cat", "cos", "cumsum",
    "expm1", "flatten", "gather", "index_select", "logsumexp", "median",
    "nan_to_num", "norm", "numel", "pow", "prod", "quantile", "relu",
    "reshape", "sin", "sort", "stack", "transpose", "unsqueeze",
}
SAFE_FUNCTIONAL_APIS = {
    "cosine_similarity",
    "logsigmoid",
    "mse_loss",
    "normalize",
    "relu",
    "softplus",
    "l1_loss", "smooth_l1_loss", "log_softmax", "tanhshrink",
}
FORBIDDEN_TENSOR_ATTRIBUTES = {
    "backward",
    "bmm",
    "cpu",
    "cuda",
    "data",
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


def _validate_objective(node: ast.FunctionDef, expected_args=None) -> None:
    helper_names = validate_helpers(node)
    expected_args = OBJECTIVE_ARGS if expected_args is None else expected_args
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
        ast.AsyncFor,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.MatMult,
        ast.Lambda,
        ast.ClassDef,
    )
    for child in ast.walk(node):
        if isinstance(child, ast.For) and not (
            isinstance(child.iter, ast.Call) and isinstance(child.iter.func, ast.Name)
            and child.iter.func.id == "range" and _is_bounded_iteration(child.iter)
        ):
            raise ValueError("TP-GNN loops require a literal range of at most eight steps")
        if child is not node and isinstance(child, forbidden):
            raise ValueError(
                "TP-GNN objective contains forbidden syntax: {}".format(
                    type(child).__name__
                )
            )
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            if child.func.id not in {"abs", "float", "int", "len", "max", "min", "range"} | helper_names:
                raise ValueError(
                    "TP-GNN objective calls unsupported function {}".format(
                        child.func.id
                    )
                )
        if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name) and child.func.value.id == "torch"
                and any(keyword.arg == "out" for keyword in child.keywords)):
            raise ValueError("TP-GNN numerical calls cannot write an out tensor")
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
    validate_loss_loop_work(node)


def _replace_once(source: str, needle: str, replacement: str, label: str) -> str:
    count = source.count(needle)
    if count != 1:
        raise ValueError(
            "Open3DBench anchor {} occurs {} times, expected once".format(
                label, count
            )
        )
    return source.replace(needle, replacement, 1)


def _patch_tp_gnn(before: str, objective: str) -> str:
    nested_objective = textwrap.indent(lower_torch_compat(objective).rstrip(), "    ")
    after = _replace_once(
        before,
        "\n    n_nodes = len(features)\n",
        "\n" + nested_objective + "\n\n    n_nodes = len(features)\n",
        "training objective insertion",
    )
    baseline_loss = (
        "            loss = -(\n"
        "                F.logsigmoid(pos_score).sum() + F.logsigmoid(-neg_score).sum()\n"
        "            ) / len(source_batch)\n"
    )
    evolved_loss = (
        "            loss = compute_tp_gnn_loss(\n"
        "                pos_score, neg_score, emb, pos_src, pos_dst, neg_src, neg_dst,\n"
        "                len(source_batch), epoch, effective_epochs,\n"
        "            )\n"
        "            if not torch.is_tensor(loss) or loss.dim() != 0:\n"
        "                raise ValueError(\"compute_tp_gnn_loss must return one scalar torch.Tensor\")\n"
        "            if not loss.requires_grad:\n"
        "                raise ValueError(\"compute_tp_gnn_loss detached the training objective\")\n"
        "            if not bool(torch.isfinite(loss).all().item()):\n"
        "                raise ValueError(\"compute_tp_gnn_loss returned a non-finite value\")\n"
    )
    # The native capsule hook leaves the historical objective in its inactive
    # branch. A source-patched task edits that branch only.
    if "            if evolved_loss is None:\n" in before:
        baseline_loss = textwrap.indent(baseline_loss, "    ")
        evolved_loss = textwrap.indent(evolved_loss, "    ")
    after = _replace_once(
        after, baseline_loss, evolved_loss, "random-walk loss call"
    )
    after = _replace_once(
        after,
        '            "loss": "per-source sum of positive-context and negative-sample log-sigmoid terms",\n',
        '            "loss": "OpenAgentic3DBench-evolved compute_tp_gnn_loss",\n',
        "training-config loss description",
    )
    return _replace_once(
        after,
        '        "loss_semantics": "per-source sum over deduplicated contexts and M legal negatives",\n',
        '        "loss_semantics": "OpenAgentic3DBench-evolved objective over frozen positive/negative samples",\n',
        "training-report loss description",
    )
