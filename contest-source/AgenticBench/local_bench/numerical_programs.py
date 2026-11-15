"""Local numerical helpers and source-local compatibility for maintained Torch."""
import ast
import inspect


def validate_helpers(owner, *, methods=False):
    """Allow immediate named helpers, but no recursion, decorators or aliases."""
    definitions = [item for item in owner.body if isinstance(item, ast.FunctionDef)]
    if len(definitions) > 16 or len({item.name for item in definitions}) != len(definitions):
        raise ValueError("Use at most 16 uniquely named local helpers")
    names = {item.name for item in definitions}
    entries = {"__init__", "__call__", "set_overflow"} if methods else set()
    if not methods and any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for statement in owner.body if not isinstance(statement, ast.FunctionDef)
        for item in ast.walk(statement)
    ):
        raise ValueError("Define numerical helpers directly inside the entrypoint")
    for function in definitions:
        args = function.args
        if function.name not in entries:
            if (function.name == owner.name or function.name.startswith("__") or function.name.startswith("_agentic3d")
                    or function.name in {"torch", "np", "F", "math", "logging"}
                    or function.decorator_list or function.returns
                    or args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs
                    or any(a.annotation for a in args.args)
                    or any(not isinstance(v, ast.Constant) for v in args.defaults)):
                raise ValueError("Numerical helpers need plain named arguments and constant defaults")
            if methods and (not args.args or args.args[0].arg != "self"):
                raise ValueError("Preconditioner helper methods must take self first")
        if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for statement in function.body for item in ast.walk(statement)):
            raise ValueError("Helpers cannot define further functions or classes")
    functions = definitions if methods else [owner] + definitions
    graph = {function.name: set() for function in functions}
    for function in functions:
        nodes = [item for statement in function.body if not isinstance(statement, ast.FunctionDef)
                 for item in ast.walk(statement)]
        call_targets = {id(item.func) for item in nodes if isinstance(item, ast.Call)}
        for item in nodes:
            name = None
            if not methods and isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load):
                name = item.id
            if methods and isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id == "self":
                name = item.attr
            if name in graph:
                if id(item) not in call_targets:
                    raise ValueError("Call numerical helpers directly; do not alias or pass them")
                graph[function.name].add(name)
    def visit(name, active):
        if name in active:
            raise ValueError("Recursive numerical helpers are not allowed")
        for target in graph[name]:
            visit(target, active | {name})
    for name in graph:
        visit(name, set())
    return names


def _agentic3d_nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):
    """Torch 1.7 equivalent for real floating and integer tensors, without global mutation."""
    if input.is_complex():
        raise ValueError("The local compatibility API supports real tensors only")
    if input.is_floating_point():
        limits = torch.finfo(input.dtype)
        result = torch.where(torch.isnan(input), torch.full_like(input, nan), input)
        result = torch.where(input == float("inf"),
                             torch.full_like(input, limits.max if posinf is None else posinf), result)
        result = torch.where(input == -float("inf"),
                             torch.full_like(input, limits.min if neginf is None else neginf), result)
    else:
        result = input.clone()
    if out is not None:
        if torch.is_grad_enabled() and input.requires_grad:
            raise RuntimeError("nan_to_num with out does not support automatic differentiation")
        if out.shape != input.shape or out.dtype != input.dtype or out.device != input.device:
            raise ValueError("nan_to_num out must preserve input shape, dtype and device")
        out.copy_(result)
        return out
    return result


def lower_torch_compat(source):
    """Lower only an explicit torch.nan_to_num reference; leave other source byte-identical."""
    tree = ast.parse(source)
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
             and isinstance(node.value, ast.Name) and node.value.id == "torch"
             and node.attr == "nan_to_num"]
    if not nodes:
        return source
    if any(isinstance(node, ast.Name) and node.id == "_agentic3d_nan_to_num" for node in ast.walk(tree)):
        raise ValueError("Candidate shadows the reserved numerical compatibility helper")
    lines = source.encode("utf-8").splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    data = source.encode("utf-8")
    for node in sorted(nodes, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        data = data[:start] + b"_agentic3d_nan_to_num" + data[end:]
    return inspect.getsource(_agentic3d_nan_to_num) + "\n\n" + data.decode("utf-8")


def validate_local_reads(function, allowed, *, protected=()):
    """Keep a nested loss isolated from its enclosing training implementation."""
    nodes = [node for statement in function.body if not isinstance(statement, ast.FunctionDef)
             for node in ast.walk(statement)]
    helpers = [statement for statement in function.body if isinstance(statement, ast.FunctionDef)]
    bound = {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    bound |= {arg.arg for arg in function.args.args} | {helper.name for helper in helpers}
    if bound & set(protected):
        raise ValueError("Loss may not shadow trusted helpers")
    scope = set(allowed) | bound
    if any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in scope for node in nodes):
        raise ValueError("Loss may read only its declared inputs and local variables")
    for helper in helpers:
        validate_local_reads(helper, scope, protected=protected)


def validate_loss_loop_work(function):
    """Bound literal-loop expansion, including calls into already acyclic helpers."""
    helpers = {item.name: item for item in function.body if isinstance(item, ast.FunctionDef)}
    budget = [12000]

    def visit(node, multiplier=1):
        budget[0] -= multiplier
        if budget[0] < 0:
            raise ValueError("TP-GNN helper/loop expansion exceeds 12000 AST operations")
        if isinstance(node, ast.FunctionDef):
            return
        if isinstance(node, ast.For):
            try:
                count = len(range(*(arg.value for arg in node.iter.args)))
            except (ValueError, TypeError):
                raise ValueError("TP-GNN loop requires a nonzero literal range step")
            nested = multiplier * count
            if nested > 64:
                raise ValueError("TP-GNN nested loops exceed 64 iterations")
            for statement in node.body:
                visit(statement, nested)
            for statement in node.orelse:
                visit(statement, multiplier)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in helpers:
            for statement in helpers[node.func.id].body:
                visit(statement, multiplier)
        for child in ast.iter_child_nodes(node):
            visit(child, multiplier)

    for statement in function.body:
        visit(statement)
