from pathlib import Path
from local_bench.placement_components import materialize_component


def materialize(program_text: str, open3dbench_root: Path) -> dict:
    return materialize_component(program_text, open3dbench_root, component="preconditioner")
