"""Static program contracts for the two capsule-based partition research tasks.

This module parses and packages candidate text; it never executes candidates or
patches a physical checkout. Native hooks and independent collectors are
source-connected; fresh exact-contract references must still qualify each case.
"""
import ast
import math

from local_bench.program_safety import normalized_program, validate_no_external_effects
from local_tasks.tp_gnn_evolution_loss_language import validate_loss_objective as _validate_objective
from local_tasks.tp_gnn_timing_context import OBJECTIVE_ARGS as LEGACY_LOSS_ARGS


BINARY_TASK = "tp_gnn_binary_loss_timing"
DECODER_TASK = "tp_gnn_loss_decoder_timing"
TASKS = frozenset({BINARY_TASK, DECODER_TASK})
BINARY_ARGS = [
    "probabilities", "expected_cut", "net_cut_probabilities", "net_weights",
    "node_features", "node_areas", "balance_target",
    "cut_weight", "area_balance_weight", "binarization_weight", "epoch", "max_epochs",
]
DECODER_ARGS = ["embeddings", "node_areas", "seed", "max_iterations", "balance_target"]
PACKAGES = {
    BINARY_TASK: ("binary_loss_program.json", "tp-gnn-binary-loss-v1",
                  "Binary TP-GNN loss program; native execution requires same-contract task qualification."),
    DECODER_TASK: ("loss_decoder_program.json", "tp-gnn-loss-decoder-v1",
                   "Legacy TP-GNN loss and global decoder program; native execution requires same-contract task qualification."),
}
LOSS_HELPERS = frozenset({"torch", "F", "abs", "float", "int", "len", "max", "min"})
LOSS_METHODS = frozenset({
    "abs", "all", "any", "clamp", "clamp_min", "clamp_max", "exp", "log", "log1p",
    "max", "mean", "min", "numel", "pow", "prod", "reshape", "sigmoid", "sqrt",
    "square", "std", "sum", "tanh", "var", "view",
})
FUNCTIONAL_POSITIONAL_LIMITS = {
    "cosine_similarity": 4, "logsigmoid": 1, "mse_loss": 5,
    "normalize": 4, "relu": 1, "softplus": 3,
}
NP_APIS = frozenset({
    "abs", "all", "any", "argmax", "argmin", "argsort", "arange", "array_equal",
    "clip", "concatenate", "cumsum", "exp", "isfinite", "lexsort", "log", "log1p", "maximum",
    "mean", "minimum", "ones_like", "reshape", "sort", "sqrt", "square", "stack",
    "sum", "take", "tanh", "where", "zeros_like", "float64", "int64",
})
ARRAY_METHODS = frozenset({
    "all", "any", "argmax", "argmin", "argsort", "astype", "clip", "copy", "cumsum", "max",
    "mean", "min", "reshape", "sum",
})
DECODER_HELPERS = frozenset({
    "np", "abs", "float", "int", "len", "max", "min", "range", "reference_global_two_way",
})
# Omit every positional/keyword form of NumPy's mutable output argument.
NP_POSITIONAL_LIMITS = {
    "abs": 1, "all": 2, "any": 2, "argmax": 2, "argmin": 2, "argsort": 4,
    "arange": 4, "array_equal": 3, "clip": 3, "concatenate": 2, "cumsum": 3, "exp": 1,
    "isfinite": 1, "lexsort": 2, "log": 1, "log1p": 1, "maximum": 2,
    "mean": 3, "minimum": 2, "ones_like": 2, "reshape": 3, "sort": 4,
    "sqrt": 1, "square": 1, "stack": 2, "sum": 3, "take": 3, "tanh": 1,
    "where": 3, "zeros_like": 2,
}
ARRAY_POSITIONAL_LIMITS = {
    "all": 1, "any": 1, "argmax": 1, "argmin": 1, "argsort": 3,
    "astype": 5, "clip": 2, "copy": 1, "cumsum": 2, "max": 1, "mean": 2,
    "min": 1, "reshape": 2, "sum": 2,
}


def _signature(function, arguments):
    args = function.args
    if (function.decorator_list or function.returns or function.type_comment
            or [arg.arg for arg in args.args] != list(arguments)
            or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
            or args.defaults or args.kw_defaults or any(arg.annotation for arg in args.args)):
        raise ValueError(function.name + " must keep the exact declared signature")


def _bindings(function, arguments, helpers):
    stores = {node.id for node in ast.walk(function)
              if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    if stores & (set(arguments) | helpers) or any(name.startswith("_") for name in stores):
        raise ValueError("Candidate cannot rebind inputs or trusted helpers")
    allowed = set(arguments) | stores | helpers
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in allowed:
            raise ValueError("Candidate may read only declared inputs, local variables and trusted helpers")
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            targets = []
        for target in targets:
            if any(not isinstance(value, (ast.Name, ast.Tuple, ast.List, ast.Store))
                   for value in ast.walk(target)):
                raise ValueError("Candidate may assign only local variables, never array elements")
        if isinstance(node, (ast.AugAssign, ast.Delete, ast.NamedExpr, ast.Try)):
            raise ValueError("Candidate forbids mutation, assignment expressions and exception handling")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr.endswith("_")):
            raise ValueError("Candidate cannot access private or mutating attributes")
        if isinstance(node, ast.Call) and (
                any(isinstance(arg, ast.Starred) for arg in node.args)
                or any(keyword.arg is None or keyword.arg in {"out", "inplace"}
                       for keyword in node.keywords)):
            raise ValueError("Candidate forbids output mutation and argument unpacking")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if not math.isfinite(node.value) or abs(node.value) > 1e12:
                raise ValueError("Candidate numeric literals must be finite and bounded")
    if not any(isinstance(node, ast.Return) for node in ast.walk(function)):
        raise ValueError("Candidate function must return a value")


def _loss(function, arguments):
    _signature(function, arguments)
    _validate_objective(function, arguments)
    _bindings(function, arguments, LOSS_HELPERS)
    for node in ast.walk(function):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "F"
                and len(node.args) > FUNCTIONAL_POSITIONAL_LIMITS[node.func.attr]):
            raise ValueError("Functional positional mutation/output arguments are forbidden")
        if isinstance(node, ast.Attribute):
            # Module APIs retain the existing timing-context-v2 allowlist.
            if isinstance(node.value, ast.Name) and node.value.id in {"torch", "F"}:
                continue
            if node.attr not in LOSS_METHODS | {"shape", "ndim"}:
                raise ValueError("Unsupported loss tensor operation: " + node.attr)


def _decoder(function):
    _signature(function, DECODER_ARGS)
    _bindings(function, DECODER_ARGS, DECODER_HELPERS)
    validate_no_external_effects(function, allowed_attribute_roots={"np"})
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    if len(loops) > 1:
        raise ValueError("Decoder permits at most one bounded iteration loop")
    reference_calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == "reference_global_two_way"]
    if len(reference_calls) > 1 or (reference_calls and loops):
        raise ValueError("Decoder may delegate once to the reference or run its own bounded loop")
    for call in reference_calls:
        if (call.keywords or len(call.args) != len(DECODER_ARGS)
                or any(not isinstance(arg, ast.Name) or arg.id != name
                       for arg, name in zip(call.args, DECODER_ARGS))):
            raise ValueError("Reference decoder helper requires the exact unchanged inputs")
    for node in ast.walk(function):
        if node is not function and isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Lambda)):
            raise ValueError("Decoder cannot define helpers or classes")
        if isinstance(node, (ast.While, ast.AsyncFor, ast.ListComp, ast.SetComp,
                             ast.DictComp, ast.GeneratorExp, ast.MatMult)):
            raise ValueError("Decoder forbids unbounded loops, comprehensions and matrix products")
        if isinstance(node, ast.For):
            iterator = node.iter
            if (not isinstance(node.target, ast.Name) or not isinstance(iterator, ast.Call)
                    or not isinstance(iterator.func, ast.Name) or iterator.func.id != "range"
                    or iterator.keywords or len(iterator.args) != 1
                    or not isinstance(iterator.args[0], ast.Name)
                    or iterator.args[0].id != "max_iterations"):
                raise ValueError("Decoder loop must be for a local variable in range(max_iterations)")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in DECODER_HELPERS - {"np"}:
                    raise ValueError("Unsupported decoder helper: " + node.func.id)
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "np":
                    if node.func.attr not in NP_APIS - {"float64", "int64"}:
                        raise ValueError("Unsupported NumPy operation: " + node.func.attr)
                    if len(node.args) > NP_POSITIONAL_LIMITS[node.func.attr]:
                        raise ValueError("NumPy positional output arguments are forbidden")
                elif node.func.attr not in ARRAY_METHODS:
                    raise ValueError("Unsupported decoder array operation: " + node.func.attr)
                elif len(node.args) > ARRAY_POSITIONAL_LIMITS[node.func.attr]:
                    raise ValueError("Array positional output arguments are forbidden")
            else:
                raise ValueError("Decoder cannot call computed functions")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "np":
                if node.attr not in NP_APIS:
                    raise ValueError("Unsupported NumPy API: " + node.attr)
            elif node.attr not in ARRAY_METHODS | {"shape", "ndim", "size"}:
                raise ValueError("Unsupported decoder array attribute: " + node.attr)


def validate_program(text, task_id):
    source = normalized_program(text, maximum_bytes=18000)
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("Candidate program is not valid Python") from error
    if len(list(ast.walk(tree))) > 3000:
        raise ValueError("Partition program exceeds 3000 AST nodes")
    expected = (["compute_binary_partition_loss"] if task_id == BINARY_TASK
                else ["compute_tp_gnn_loss", "decode_tp_gnn_embeddings"] if task_id == DECODER_TASK else [])
    if (not expected or len(tree.body) != len(expected)
            or any(not isinstance(node, ast.FunctionDef) for node in tree.body)
            or [node.name for node in tree.body] != expected):
        raise ValueError("Partition program must contain only the exact declared functions, in order")
    _loss(tree.body[0], BINARY_ARGS if task_id == BINARY_TASK else LEGACY_LOSS_ARGS)
    if task_id == DECODER_TASK:
        _decoder(tree.body[1])
    return source


def _materialize(text, task_id):
    filename, interface, notes = PACKAGES[task_id]
    return {"schema_version": 1, "config_overrides": {}, "source_patch": "",
            "candidate_files": {filename: {"interface_version": interface,
                                           "source": validate_program(text, task_id)}},
            "notes": notes}


def materialize_binary_loss(program_text, open3dbench_root=None):
    return _materialize(program_text, BINARY_TASK)


def materialize_loss_decoder(program_text, open3dbench_root=None):
    return _materialize(program_text, DECODER_TASK)


def validate_candidate(candidate, task_id):
    if task_id not in TASKS:
        raise ValueError("Unknown partition task")
    if candidate["config_overrides"] or candidate["source_patch"]:
        raise ValueError("Partition task programs cannot change configuration or source files")
    filename, interface, _ = PACKAGES[task_id]
    if set(candidate["candidate_files"]) != {filename}:
        raise ValueError("Partition task requires exactly " + filename)
    program = candidate["candidate_files"][filename]
    if (not isinstance(program, dict) or set(program) != {"interface_version", "source"}
            or program["interface_version"] != interface):
        raise ValueError("Unsupported partition program interface or fields")
    validate_program(program["source"], task_id)
