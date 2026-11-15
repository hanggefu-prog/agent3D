"""Validate and package the source-connected joint TP-GNN program capsule."""
from local_tasks.tp_gnn_local_task_definitions import materialize_loss_decoder


def materialize(program_text, open3dbench_root):
    return materialize_loss_decoder(program_text, open3dbench_root)
