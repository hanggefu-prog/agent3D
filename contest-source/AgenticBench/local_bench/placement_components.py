"""Single-component adapters; the complementary definition is never model-owned."""
import ast
from pathlib import Path

from .program_safety import normalized_program


def materialize_component(program_text, open3dbench_root, *, component):
    if component not in {"initializer", "preconditioner"}:
        raise ValueError("unknown placement component")
    from local_tasks import placement_source_common as adapter
    directory = Path(__file__).parents[1] / "local_tasks"
    init = normalized_program((directory / "placement_initialization/initial_program.py").read_text())
    precond = normalized_program((directory / "placement_preconditioner/initial_program.py").read_text())
    candidate = normalized_program(program_text)
    tree = ast.parse(candidate)
    expected_type, expected_name = ((ast.FunctionDef, "initialize_position")
                                   if component == "initializer" else (ast.ClassDef, "PreconditionOp"))
    if len(tree.body) != 1 or not isinstance(tree.body[0], expected_type) or tree.body[0].name != expected_name:
        raise ValueError("candidate must contain only " + expected_name)
    combined = candidate + "\n\n" + precond if component == "initializer" else init + "\n\n" + candidate
    return adapter.materialize(combined, open3dbench_root, component=component)
