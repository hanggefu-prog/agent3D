"""Offline loss packaging; native execution requires task qualification."""
from pathlib import Path

from local_tasks.tp_gnn_local_task_definitions import materialize_binary_loss


def materialize(program_text: str, open3dbench_root: Path) -> dict:
    return materialize_binary_loss(program_text)
