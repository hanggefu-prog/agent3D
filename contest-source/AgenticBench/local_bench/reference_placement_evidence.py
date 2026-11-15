"""Read-only native evidence gate for the versioned placement reference tasks.

This supplements, never replaces, the native output/metrics validator. A recipe
receipt is not signoff, an optimization improvement, or a scoring reference.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


DYNAMIC_TASK = "placement_dynamic_parameters_reference_20260923"
SOURCE_TASK = "placement_preconditioner_source_reference_20260923"
TASKS = {DYNAMIC_TASK, SOURCE_TASK}
STAGE = Path("30_placement/stages/02_global_place")
CONTROL_MARKER = "NesterovStepControl"
CONTROL_ROW = re.compile(
    r"NesterovStepControl\s+step=(\d+)\s+call=(\d+)\s+"
    r"native_step=(\S+)\s+shared=(\S+)\s+bottom=(\S+)\s+top=(\S+)\s*$"
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not path.stat().st_size:
        raise ValueError("Reference placement requires a nonempty regular evidence file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _number(mapping, key, expected, context):
    value = mapping.get(key)
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Reference placement {}.{} must equal {}".format(context, key, expected))
    return value


def _flag(mapping, key, expected, context, *, default=None):
    value = mapping.get(key, default)
    if not isinstance(value, (bool, int)) or int(value) != expected:
        raise ValueError("Reference placement {}.{} must equal {}".format(context, key, expected))
    return int(value)


def validate_reference_placement_params(params: dict, *, dynamic: bool, deterministic_flag: int = 0, placement_seed: int = 1000) -> dict:
    if not isinstance(params, dict):
        raise ValueError("Reference placement params.json must be an object")
    numbers = {
        "random_seed": placement_seed, "num_threads": 4, "num_bins_x": 512, "num_bins_y": 512,
        "target_density": 0.6, "macro_halo_x": 10, "macro_halo_y": 10,
        "stop_overflow": 0.01, "density_weight": 0.00003,
    }
    checked = {key: _number(params, key, expected, "params") for key, expected in numbers.items()}
    checked["macro_place_flag"] = _flag(params, "macro_place_flag", 1, "params")
    checked["enable_fillers"] = _flag(params, "enable_fillers", 1, "params")
    checked["random_center_init_flag"] = _flag(params, "random_center_init_flag", 0, "params")
    # Frozen native parameters omit this option; DREAMPlace's native default is
    # zero. Do not silently strengthen it to one for this new task family.
    checked["deterministic_flag"] = _flag(params, "deterministic_flag", deterministic_flag, "params", default=0)
    checked["deterministic_flag_explicit"] = "deterministic_flag" in params
    stages = params.get("global_place_stages")
    if not isinstance(stages, list) or len(stages) != 1 or not isinstance(stages[0], dict):
        raise ValueError("Reference placement requires exactly one configured global_place_stages entry")
    stage = stages[0]
    if stage.get("optimizer") != "nesterov":
        raise ValueError("Reference placement optimizer must be native nesterov")
    checked_stage = {"optimizer": "nesterov"}
    for key, expected in {"iteration": 4000, "learning_rate": 0.01, "num_bins_x": 512, "num_bins_y": 512}.items():
        checked_stage[key] = _number(stage, key, expected, "global_place_stages[0]")
    if stage.get("wirelength") != "weighted_average":
        raise ValueError("Reference placement wirelength must be weighted_average")
    checked_stage["wirelength"] = stage["wirelength"]
    checked["global_place_stages"] = [checked_stage]
    concurrent = params.get("concurrent_lol")
    if not isinstance(concurrent, dict):
        raise ValueError("Reference placement requires concurrent_lol parameters")
    checked_concurrent = {key: _number(concurrent, key, expected, "concurrent_lol")
                          for key, expected in {
                              "macro_snap_guard_um": 1.001, "macro_halo_x": 10,
                              "macro_halo_y": 10, "initialization_seed": placement_seed,
                              "cross_die_net_weight": 1,
                          }.items()}
    checked_concurrent["enabled"] = _flag(concurrent, "enabled", 1, "concurrent_lol")
    if concurrent.get("initialization_mode") != "die-center":
        raise ValueError("Reference placement initialization_mode must be die-center")
    if concurrent.get("macro_halo_units") != "micron":
        raise ValueError("Reference placement macro halo units must be micron")
    checked_concurrent.update(initialization_mode="die-center", macro_halo_units="micron")
    checked["concurrent_lol"] = checked_concurrent
    # use_bb has a native default of zero in the frozen reference. The new
    # controller uses that native no-BB backtracking path, never an implicit BB
    # algorithm change. Missing means the same audited native default.
    use_bb = params.get("use_bb", 0)
    if not isinstance(use_bb, (int, bool)) or int(use_bb) not in (0, 1):
        raise ValueError("Reference placement params.use_bb must be 0 or 1")
    if use_bb:
        raise ValueError("Reference Nesterov evidence requires use_bb missing/0; BB differs from the frozen recipe")
    checked["use_bb"] = int(use_bb)
    checked["use_bb_explicit"] = "use_bb" in params
    return checked


def parse_nesterov_step_controls(text: str) -> dict:
    """Require every marked row to parse and belong to a real optimizer stage.

    With macro placement enabled, native DREAMPlace duplicates the one
    configured stage internally. Call indices restart at one only after a new
    native optimizer-selection header; the global step index does not reset.
    """
    rows, segments = [], []
    active = None
    previous_step = None
    for line_number, original in enumerate(text.splitlines(), 1):
        line = ANSI_ESCAPE.sub("", original)
        if re.search(r"\buse nesterov optimizer\s*$", line):
            if active is not None and active["calls"] == 0:
                raise ValueError("Nesterov stage has no active step-controller calls")
            active = {"selection_line": line_number, "calls": 0}
            segments.append(active)
            if len(segments) > 2:
                raise ValueError("Reference placement has unexpected extra Nesterov optimizer stages")
        if CONTROL_MARKER not in line:
            continue
        match = CONTROL_ROW.search(line)
        if not match:
            raise ValueError("Malformed NesterovStepControl row at line {}".format(line_number))
        if active is None:
            raise ValueError("Nesterov step control lacks a preceding native optimizer-selection header")
        step, call = int(match[1]), int(match[2])
        if call != active["calls"] + 1:
            raise ValueError("Nesterov control call sequence is not contiguous at line {}".format(line_number))
        expected_step = 0 if previous_step is None else previous_step + 1
        if step != expected_step:
            raise ValueError("Nesterov control global step sequence is not contiguous at line {}".format(line_number))
        values = {}
        for key, token in zip(("native_step", "shared", "bottom", "top"), match.groups()[2:]):
            try:
                value = float(token)
            except ValueError as error:
                raise ValueError("Malformed Nesterov control {} at line {}".format(key, line_number)) from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Nesterov control {} must be finite and positive at line {}".format(key, line_number))
            values[key] = value
        active["calls"] = call
        previous_step = step
        rows.append(dict(line=line_number, segment=len(segments), step=step, call=call, **values))
    if not rows or active is None or active["calls"] == 0:
        raise ValueError("Nesterov dynamic task has zero active step-controller calls")
    return {"call_count": len(rows), "segments": segments, "controls": rows,
            "native_step_interpretation": "alpha_k hint before native no-BB backtracking; not the final accepted trial alpha",
            "identity_controls_only": all(row["shared"] == row["bottom"] == row["top"] == 1 for row in rows)}


def validate_reference_placement_evidence(run_dir: Path, task_id: str, *, runtime_overlay: Path | None = None,
                                         profile: str = "reference_20260923", placement_seed: int = 1000) -> dict:
    if task_id not in TASKS:
        raise ValueError("Reference placement evidence requested for an unsupported task: " + str(task_id))
    stage = Path(run_dir) / STAGE
    params_path, log_path = stage / "params.json", stage / "run.log"
    evidence = {"params": _artifact(params_path), "log": _artifact(log_path)}
    params = json.loads(params_path.read_text(encoding="utf-8"))
    dynamic = task_id == DYNAMIC_TASK
    if profile not in {"reference_20260923", "current-source-nesterov-v1"} or (
            profile == "current-source-nesterov-v1" and not dynamic):
        raise ValueError("Unsupported placement evidence profile")
    if profile != "current-source-nesterov-v1" and placement_seed != 1000:
        raise ValueError("Dated placement references have a fixed seed")
    recipe = validate_reference_placement_params(params, dynamic=dynamic,
        deterministic_flag=1 if profile == "current-source-nesterov-v1" else 0,
        placement_seed=placement_seed)
    controls = parse_nesterov_step_controls(log_path.read_text(encoding="utf-8")) if dynamic else {
        "call_count": 0, "segments": [], "controls": [], "required": False,
    }
    if runtime_overlay is not None:
        names = ("NonLinearPlace.py", "NesterovAcceleratedGradientOptimizer.py") if dynamic else ("BasicPlace.py", "PlaceObj.py")
        evidence["runtime_overlay_files"] = {
            name: _artifact(Path(runtime_overlay) / "dreamplace" / name) for name in names
        }
    return {
        "schema_version": 1, "baseline_id": profile, "task_id": task_id,
        "status": "passed", "recipe": recipe, "dynamic_controls": controls, "evidence": evidence,
        "interpretation": "Native recipe and control-log evidence only; overlay hashes are inventory, not independent proof of runtime loading. Native output/metric validation remains mandatory.",
    }
