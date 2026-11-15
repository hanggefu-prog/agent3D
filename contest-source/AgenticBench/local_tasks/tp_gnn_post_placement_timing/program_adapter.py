from pathlib import Path
from local_tasks.tp_gnn_timing_context import materialize as materialize_loss


def materialize(program_text, open3dbench_root):
    result = materialize_loss(program_text, open3dbench_root)
    result["notes"] = ("Timing-context-v2 loss with global-2way decoding; fitness is no-OPT "
                       "post-placement/HBT STA. Materialization is not native qualification.")
    return result
