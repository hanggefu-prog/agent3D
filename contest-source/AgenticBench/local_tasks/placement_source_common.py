"""Trusted source validation and patching shared by the two placement components."""
from __future__ import annotations

import ast
from pathlib import Path
from local_bench.numerical_programs import lower_torch_compat, validate_helpers

from local_bench.program_safety import (
    normalized_program,
    read_repo_file,
    replace_definition,
    unified_diff,
    validate_no_external_effects,
)


BASIC_PLACE = "Place-3D/thirdparty/DREAMPlace/dreamplace/BasicPlace.py"
PLACE_OBJ = "Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"
CONCURRENT = "Place-3D/src/placement/concurrent_dreamplace.py"
EXPECTED_METHODS = {
    "__init__": ["self", "placedb", "data_collections", "op_collections", "die_contexts"],
    "set_overflow": ["self", "overflow"],
    "__call__": ["self", "grad", "density_weight", "update_mask", "fix_nodes_mask"],
}
EXPECTED_DEFAULTS = {"__init__": 1, "set_overflow": 0, "__call__": 2}
# NumPy and Torch are computation libraries, so maintaining a positive list of
# every legal operation is both brittle and hostile to meaningful evolution.
# Permit their ordinary numerical APIs and reject only namespaces/functions
# that can perform external I/O, load code, spawn workers, or mutate global
# runtime state.  The generic AST guard still forbids imports, files, sockets,
# subprocesses, dynamic code execution, and access to dangerous Python modules.
FORBIDDEN_NUMPY_APIS = {
    "DataSource", "RandomState", "SeedSequence", "ctypeslib", "default_rng",
    "distutils", "f2py", "fromfile", "genfromtxt", "load", "loadtxt", "memmap",
    "npyio", "open_memmap", "recfromcsv", "recfromtxt", "save", "savetxt",
    "savez", "savez_compressed", "seed", "set_state", "setbufsize", "seterr",
    "seterrcall", "set_printoptions", "set_string_function", "show_config",
    "testing", "tofile",
}
FORBIDDEN_TORCH_APIS = {
    "compile", "distributed", "export", "hub", "jit", "library", "load",
    "load_library", "manual_seed", "multiprocessing", "onnx", "ops", "package",
    "monitor", "profiler", "save", "seed", "set_default_device", "set_default_dtype",
    "set_default_tensor_type", "set_deterministic_debug_mode", "set_flush_denormal",
    "set_num_interop_threads", "set_num_threads", "use_deterministic_algorithms",
    "utils",
}
SAFE_DIRECT_CALLS = {
    "abs", "all", "any", "bool", "complex", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "pow", "range", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
}








def _module_attribute_chain(node: ast.Attribute, module: str) -> tuple[str, ...]:
    parts = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name) or cursor.id != module:
        return ()
    return tuple(reversed(parts))


def _validate_module_api(
    node: ast.Attribute,
    *,
    module: str,
    forbidden: set[str],
    owner: str,
) -> None:
    chain = _module_attribute_chain(node, module)
    if not chain:
        return
    blocked = next(
        (
            part
            for part in chain
            if part in forbidden or part.startswith("_")
        ),
        None,
    )
    if blocked is not None:
        raise ValueError(
            "{} uses forbidden {} external-effect API {}".format(
                owner, module, ".".join(chain)
            )
        )


def _parse_joint_program(program_text: str) -> tuple[str, str, ast.FunctionDef, ast.ClassDef]:
    source = normalized_program(program_text)
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("candidate program is not valid Python: {}".format(error)) from error
    if (
        len(tree.body) != 2
        or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].name != "initialize_position"
        or not isinstance(tree.body[1], ast.ClassDef)
        or tree.body[1].name != "PreconditionOp"
    ):
        raise ValueError(
            "candidate program must contain exactly initialize_position followed by PreconditionOp"
        )
    init_node = tree.body[0]
    preconditioner_node = tree.body[1]
    init_source = ast.get_source_segment(source, init_node)
    preconditioner_source = ast.get_source_segment(source, preconditioner_node)
    if not init_source or not preconditioner_source:
        raise ValueError("cannot recover candidate source definitions")
    return init_source.rstrip() + "\n", preconditioner_source.rstrip() + "\n", init_node, preconditioner_node


def _validate_initializer(node: ast.FunctionDef) -> None:
    helper_names = validate_helpers(node)
    args = node.args
    if (
        [item.arg for item in args.args] != ["params", "placedb", "die_contexts"]
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or len(args.defaults) != 1
        or not isinstance(args.defaults[0], ast.Constant)
        or args.defaults[0].value is not None
        or args.kw_defaults
        or node.decorator_list
    ):
        raise ValueError(
            "initialize_position must keep the exact (params, placedb, die_contexts=None) signature"
        )
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            _validate_module_api(
                child,
                module="np",
                forbidden=FORBIDDEN_NUMPY_APIS,
                owner="initialize_position",
            )
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            if child.func.id not in SAFE_DIRECT_CALLS | helper_names:
                raise ValueError("initialize_position calls unsupported function {}".format(child.func.id))
        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    if target.value.id in {"params", "placedb"}:
                        raise ValueError("initialize_position cannot mutate params or placedb")
    validate_no_external_effects(node, allowed_attribute_roots={"np", "logging"})


def _validate_class(node: ast.ClassDef) -> None:
    if node.bases or node.keywords or node.decorator_list:
        raise ValueError("PreconditionOp cannot add bases, keywords, or decorators")
    methods = {child.name: child for child in node.body if isinstance(child, ast.FunctionDef)}
    if not set(EXPECTED_METHODS) <= set(methods):
        raise ValueError("PreconditionOp must retain {}".format(", ".join(EXPECTED_METHODS)))
    validate_helpers(node, methods=True)
    for child in node.body:
        is_docstring = (
            isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Constant)
            and isinstance(child.value.value, str)
        )
        if not isinstance(child, ast.FunctionDef) and not is_docstring:
            raise ValueError("PreconditionOp may contain only its docstring and declared methods")
    for name, expected in EXPECTED_METHODS.items():
        function = methods[name]
        if (
            function.decorator_list
            or function.args.posonlyargs
            or function.args.kwonlyargs
            or function.args.vararg
            or function.args.kwarg
            or function.args.kw_defaults
        ):
            raise ValueError("{} cannot use decorators or variadic arguments".format(name))
        actual = [arg.arg for arg in function.args.args]
        if actual != expected:
            raise ValueError("{} must keep arguments {}".format(name, expected))
        if len(function.args.defaults) != EXPECTED_DEFAULTS[name]:
            raise ValueError("{} must preserve its positional defaults".format(name))
        if name == "__call__" and not all(
            isinstance(default, ast.Constant) and default.value is None
            for default in function.args.defaults
        ):
            raise ValueError("PreconditionOp.__call__ defaults must remain None")
        if name == "__init__" and not (
            len(function.args.defaults) == 1
            and isinstance(function.args.defaults[0], ast.Constant)
            and function.args.defaults[0].value is None
        ):
            raise ValueError("PreconditionOp.__init__ die_contexts default must remain None")
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute):
            continue
        _validate_module_api(
            child,
            module="torch",
            forbidden=FORBIDDEN_TORCH_APIS,
            owner="PreconditionOp",
        )
        root = child.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id == "logging" and child.attr not in {"debug", "info", "warning"}:
            raise ValueError("PreconditionOp may use only standard logging methods")
    validate_no_external_effects(node, allowed_attribute_roots={"torch", "logging"})


def materialize(program_text: str, open3dbench_root: Path, *, component: str) -> dict:
    if component not in {"initializer", "preconditioner"}:
        raise ValueError("unknown placement source component")
    initializer, preconditioner, init_node, preconditioner_node = _parse_joint_program(program_text)
    _validate_initializer(init_node)
    _validate_class(preconditioner_node)

    basic_before = read_repo_file(open3dbench_root, BASIC_PLACE)
    basic_tree = ast.parse(basic_before)
    basic_classes = [
        item for item in basic_tree.body
        if isinstance(item, ast.ClassDef) and item.name == "BasicPlace"
    ]
    if len(basic_classes) != 1:
        raise ValueError("Open3DBench must contain exactly one BasicPlace")
    init_methods = [
        item for item in basic_classes[0].body
        if isinstance(item, ast.FunctionDef) and item.name == "_initialize_position"
    ]
    if len(init_methods) != 1:
        raise ValueError("Open3DBench BasicPlace must contain exactly one _initialize_position")
    context_builder = (
        "def _agentic3d_build_die_contexts(placedb):\n"
        "    if len(placedb.regions) != 2:\n"
        "        return None\n"
        "    names = getattr(placedb, \"concurrent_lol_region_names\", None)\n"
        "    if names != [\"bottom\", \"top\"]:\n"
        "        return None\n"
        "    contexts = []\n"
        "    movable_map = placedb.node2fence_region_map[:placedb.num_movable_nodes]\n"
        "    targets = getattr(placedb, \"target_density_fence_region\", None)\n"
        "    for region_id, name in enumerate(names):\n"
        "        boxes = np.asarray(placedb.regions[region_id], dtype=placedb.dtype)\n"
        "        widths = boxes[:, 2] - boxes[:, 0]\n"
        "        heights = boxes[:, 3] - boxes[:, 1]\n"
        "        region_area = max(float(np.sum(widths * heights)), 1.0e-30)\n"
        "        node_mask = movable_map == region_id\n"
        "        movable_area = float(np.sum(\n"
        "            placedb.node_size_x[:placedb.num_movable_nodes][node_mask]\n"
        "            * placedb.node_size_y[:placedb.num_movable_nodes][node_mask]\n"
        "        ))\n"
        "        target_density = (\n"
        "            float(targets[region_id])\n"
        "            if targets is not None and region_id < len(targets)\n"
        "            else None\n"
        "        )\n"
        "        total_width = max(float(np.max(boxes[:, 2]) - np.min(boxes[:, 0])), 1.0e-30)\n"
        "        total_height = max(float(np.max(boxes[:, 3]) - np.min(boxes[:, 1])), 1.0e-30)\n"
        "        contexts.append((\n"
        "            name, region_id, movable_area / region_area, target_density,\n"
        "            total_width / total_height,\n"
        "        ))\n"
        "    return tuple(contexts)\n"
    )
    delegated_method = (
        "    def _initialize_position(self, params, placedb):\n"
        "        die_contexts = _agentic3d_build_die_contexts(placedb)\n"
        "        placedb.agentic3d_die_contexts = die_contexts\n"
        "        return initialize_position(params, placedb, die_contexts)\n"
    )
    basic_after = replace_definition(basic_before, init_methods[0], delegated_method)
    delegated_tree = ast.parse(basic_after)
    delegated_classes = [
        item for item in delegated_tree.body
        if isinstance(item, ast.ClassDef) and item.name == "BasicPlace"
    ]
    lines = basic_after.splitlines(keepends=True)
    insert_at = delegated_classes[0].lineno - 1
    basic_after = (
        "".join(lines[:insert_at])
        + context_builder
        + "\n"
        + initializer
        + "\n"
        + "".join(lines[insert_at:])
    )

    placeobj_before = read_repo_file(open3dbench_root, PLACE_OBJ)
    placeobj_tree = ast.parse(placeobj_before)
    targets = [
        item for item in placeobj_tree.body
        if isinstance(item, ast.ClassDef) and item.name == "PreconditionOp"
    ]
    if len(targets) != 1:
        raise ValueError("Open3DBench must contain exactly one PreconditionOp")
    placeobj_after = replace_definition(placeobj_before, targets[0], lower_torch_compat(preconditioner))
    constructor_call = "return PreconditionOp(placedb, data_collections, op_collections)"
    if placeobj_after.count(constructor_call) != 1:
        raise ValueError(
            "Open3DBench must contain exactly one trusted PreconditionOp constructor"
        )
    call_offset = placeobj_after.index(constructor_call)
    line_offset = placeobj_after.rfind("\n", 0, call_offset) + 1
    indentation = placeobj_after[line_offset:call_offset]
    placeobj_after = placeobj_after.replace(
        constructor_call,
        "return PreconditionOp(\n"
        + indentation
        + "    placedb, data_collections, op_collections,\n"
        + indentation
        + ("    _agentic3d_build_die_contexts(placedb),\n" if component == "preconditioner"
           else "    getattr(placedb, \"agentic3d_die_contexts\", None),\n")
        + indentation
        + ")",
        1,
    )
    concurrent_before = read_repo_file(open3dbench_root, CONCURRENT)
    if concurrent_before.count('        "optimizer": "nesterov",\n') != 1:
        raise ValueError("reference-20260923 requires the native Nesterov optimizer")
    # The baseline is an exact native no-op. Compare executable bodies, not
    # temperatures or an approximate numerical recipe. Candidate signatures add
    # an optional context argument but otherwise retain the native definitions.
    same_initializer = ast.dump(ast.Module(body=init_node.body, type_ignores=[])) == ast.dump(
        ast.Module(body=init_methods[0].body, type_ignores=[])
    )
    native_methods = {n.name: n for n in targets[0].body if isinstance(n, ast.FunctionDef)}
    candidate_methods = {n.name: n for n in preconditioner_node.body if isinstance(n, ast.FunctionDef)}
    same_preconditioner = set(candidate_methods) == set(native_methods) and all(
        ast.dump(ast.Module(body=candidate_methods[name].body, type_ignores=[]))
        == ast.dump(ast.Module(body=native_methods[name].body, type_ignores=[]))
        for name in EXPECTED_METHODS
    )
    if component == "initializer" and not same_preconditioner:
        raise ValueError("initializer task must preserve the native preconditioner")
    if component == "preconditioner" and not same_initializer:
        raise ValueError("preconditioner task must preserve native initialization")
    if component == "preconditioner":
        # Keep BasicPlace byte-identical, including the native initializer. Build
        # read-only die context at the existing preconditioner construction site.
        tree = ast.parse(placeobj_after)
        index = next(n.lineno - 1 for n in tree.body
                     if isinstance(n, ast.ClassDef) and n.name == "PreconditionOp")
        lines = placeobj_after.splitlines(keepends=True)
        placeobj_after = "".join(lines[:index]) + context_builder + "\n" + "".join(lines[index:])
    patch = ""
    if not (same_initializer and same_preconditioner):
        if component != "preconditioner":
            patch += unified_diff(BASIC_PLACE, basic_before, basic_after)
        if component != "initializer":
            patch += unified_diff(PLACE_OBJ, placeobj_before, placeobj_after)
    result = {
        "schema_version": 1,
        "config_overrides": {"placement": {"target_density": 0.6}},
        "candidate_files": {
            "placement.json": {
                "concurrent_dreamplace": {
                    "cross_die_net_weight": 1.0,
                    "iterations": 4000,
                    "num_bins": 512,
                    "target_density": 0.6,
                    "stop_overflow": 0.01,
                    "density_weight": 0.00003,
                    "macro_place_flag": True,
                    "macro_halo_x": 10.0,
                    "macro_halo_y": 10.0,
                    "learning_rate": 0.01,
                    "die_params": {},
                    "hide_fillers_in_plot": False,
                }
            }
        },
        "source_patch": patch,
        "notes": (
            "Joint initialization and PreconditionOp replacement under the fixed "
            "native Nesterov, 4000-iteration reference-20260923 recipe."
        ),
    }
    # Physical macro halos come from the selected PDK, not model output.
    params = result["candidate_files"]["placement.json"]["concurrent_dreamplace"]
    del params["macro_halo_x"]
    del params["macro_halo_y"]
    result["notes"] = "Single-component {} evolution; other component is native.".format(component)
    return result
