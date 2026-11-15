"""Host-sealed TP-GNN handoff after placement, before CPU-only PDN/HBT/STA.

The GPU process never releases its own lease. The host confirms container
termination, hashes this continuation, then launches a GPU-free continuation.
This is a same-source/image split of one evaluation, not cross-run replay.
"""
import os
from pathlib import Path

from grt_bench.common import digest, read, sha, write
from .model import load_task

PATH_KEYS = ("workspace", "run_dir", "command_cwd", "output_dir", "reference_run",
             "baseline_run", "dreamplace_overlay")
CONTINUATION = "gpu-continuation.json"


def request_binding(spec, candidate, source, output, **options):
    return digest({"task": spec.task_id, "task_sha256": sha(spec.root / "task.yaml"),
        "candidate_sha256": sha(Path(candidate) / "candidate.json"),
        "source": str(source), "output": str(output),
        "image": os.environ.get("OPEN3DBENCH_IMAGE_ID"),
        "base_config_sha256": sha(options["base_config"] or spec.base_config),
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in options.items()}})


def split_command(command, phase):
    command = list(command)
    if "--from" not in command or "--to" not in command or "--resume" not in command:
        raise ValueError("Local phase requires the maintained resumable flow command")
    if command[command.index("--to") + 1] not in {"hbt", "opt"}:
        raise ValueError("Local TP-GNN endpoint must remain HBT plus post-placement STA")
    if phase == "frontend":
        command[command.index("--to") + 1] = "placement"
    elif phase == "backend":
        command[command.index("--from") + 1] = "pdn"
    else:
        raise ValueError("Unknown local execution phase")
    return command


def save_continuation(state):
    output, run = state["output_dir"], state["run_dir"]
    inventory = {str(p): sha(p) for p in (
        run / "20_partition/stage_manifest.json", run / "30_placement/stage_manifest.json",
        run / "30_placement/checkpoint.json", output / "resolved_config.yaml",
        output / "candidate_snapshot.json")}
    for stage in ("partition", "placement"):
        method = Path(state["resolved_config"][stage]["config"])
        inventory[str(method)] = sha(method)
    serial = dict(state)
    serial["spec"] = str(state["spec"].root)
    for key in PATH_KEYS:
        serial[key] = str(state[key]) if state[key] is not None else None
    # Carry only evaluator-created overrides. Never serialize inherited secrets
    # or CUDA device visibility into a continuation launched without a GPU.
    overrides = {k: v for k, v in state["run_environment"].items()
                 if os.environ.get(k) != v}
    if any(k not in {"PLACE3D_DREAMPLACE_INSTALL", "OPEN3D_OPT_PASSES", "OPENROAD_EXE",
                     "LD_LIBRARY_PATH", "PYTHONPATH"} for k in overrides):
        raise ValueError("Unexpected local evaluator environment override")
    serial["run_environment"] = overrides
    serial["artifacts"]["execution_mode"] = "gpu-frontend-cpu-backend"
    path = output / CONTINUATION
    write(path, {"schema_version": 1, "state": serial, "inventory": inventory})
    return path


def load_continuation(output, expected_sha256, binding):
    path = output / CONTINUATION
    if not expected_sha256 or sha(path) != expected_sha256:
        raise ValueError("Unsealed or modified local GPU continuation")
    value = read(path)
    state = value["state"]
    if value.get("schema_version") != 1 or state["phase_binding"] != binding:
        raise ValueError("Local continuation request differs")
    if any(sha(p) != expected for p, expected in value["inventory"].items()):
        raise ValueError("Local continuation checkpoint/configuration changed")
    for key in PATH_KEYS:
        state[key] = Path(state[key]) if state[key] is not None else None
    if state["output_dir"] != output or state["run_dir"] != output / "run":
        raise ValueError("Local continuation output differs")
    state["spec"] = load_task(Path(state["spec"]))
    if state["spec"].task_id not in {"tp_gnn_post_placement_timing", "tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}:
        raise ValueError("Local continuation belongs to another task")
    state["run_environment"] = {**os.environ, **state["run_environment"]}
    # Native resume validates the complete stage manifests and physical hashes.
    # The handoff hashes above bind exactly which checkpoint the host released.
    return state
