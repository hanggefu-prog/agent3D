"""Admission for decoder-bound timing and independent placement references."""
import os
from pathlib import Path

from .program_bridge import materialize_program
from .reference_latest import read_json, require, sha256

TASKS = frozenset({"hbt_assignment_policy", "tp_gnn_bin_timing", "tp_gnn_post_placement_timing",
                   "tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing",
                   "placement_initialization", "placement_preconditioner"})


def implementation_identity(spec, source, *, runtime_image=None):
    if spec.kind == "hbt_assignment_policy":
        from .hbt_native import implementation_identity as hbt_identity
        return hbt_identity(spec, source)
    # Scoring and native Python implementations are part of the comparison.
    # A source synchronization never promotes an older measurement silently.
    paths = ("Place-3D/src/partition/lol/methods/tp_gnn.py",
             "Place-3D/src/partition/lol/methods/spatial_bins.py",
             "Place-3D/src/partition/lol/methods/spatial_geometry.py",
             "Place-3D/src/placement/concurrent_dreamplace.py",
             "Place-3D/thirdparty/DREAMPlace/dreamplace/BasicPlace.py",
             "Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py",
             "Place-3D/thirdparty/DREAMPlace/dreamplace/NonLinearPlace.py",
             "Place-3D/thirdparty/DREAMPlace/dreamplace/NesterovAcceleratedGradientOptimizer.py",
             "open3dbench/backend/post_placement_sta.py", "open3dbench/pdks.py")
    result = {name: sha256(Path(source) / name) for name in paths}
    if spec.task_id in {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}:
        result["runtime_image"] = (runtime_image if runtime_image is not None
                                   else os.environ.get("OPEN3DBENCH_IMAGE_ID", ""))
        for relative in ("open3dbench/backend/three_d_opt.py",
                         "open3dbench/backend/opt_handoff.py", "open3dbench/runner.py",
                         "open3dbench/config.py", "open3dbench/stage_manifest.py"):
            result[relative] = sha256(Path(source) / relative)
        for name in ("tp_gnn_binary.py", "tp_gnn_binary_loss.py", "tp_gnn_encoder.py", "tp_gnn_evolution.py"):
            relative = "Place-3D/src/partition/lol/methods/" + name
            result[relative] = sha256(Path(source) / relative)
        recipe = ("tp_gnn_binary_regularized.json" if spec.task_id == "tp_gnn_binary_loss_timing"
                  else "tp_gnn_fast.json")
        relative = "Place-3D/config/partition/" + recipe
        result[relative] = sha256(Path(source) / relative)
        result["harness/gnn_evolution_timing.py"] = sha256(Path(__file__).parent / "gnn_evolution_timing.py")
        result["harness/packed_gnn_reference.py"] = sha256(Path(__file__).parent / "packed_gnn_reference.py")
        result["harness/tp_gnn_local_task_definitions.py"] = sha256(
            Path(__file__).resolve().parents[1] / "local_tasks/tp_gnn_local_task_definitions.py")
        result["harness/tp_gnn_evolution_loss_language.py"] = sha256(
            Path(__file__).resolve().parents[1] / "local_tasks/tp_gnn_evolution_loss_language.py")
    for name in ("metrics.py", "evaluation.py", "post_placement_timing.py", "placement_components.py",
                 "scoped_reference.py"):
        result["harness/" + name] = sha256(Path(__file__).parent / name)
    if spec.task_id == "tp_gnn_post_placement_timing":
        result["harness/phased.py"] = sha256(Path(__file__).parent / "phased.py")
    for name in ("task.yaml", "program_adapter.py", "initial_program.py"):
        result["task/" + name] = sha256(spec.root / name)
    return result


def admit_reference(spec, candidate, source, baseline_run, reference_run, *, execution_contract=None,
                    runtime_image=None):
    """Do not let missing references silently turn arbitrary candidates into score 1."""
    native = materialize_program(spec, spec.program_template.read_text(), source)
    identity = implementation_identity(spec, source, runtime_image=runtime_image)
    if execution_contract is not None:
        # Bind the explicit current profile without changing dated reference admission.
        identity = {**identity, "execution_contract": execution_contract}
        for name in ("current_nesterov.py", "reference_placement_evidence.py"):
            identity["harness/" + name] = sha256(Path(__file__).parent / name)
        identity["task/nesterov_runtime.py"] = sha256(spec.root / "nesterov_runtime.py")
    if reference_run is None:
        require(candidate == native, "Only the exact native initial program may create a new reference")
        require(baseline_run is not None, "Native reference needs an independently validated upstream input")
        return {"mode": "FRESH", "task_id": spec.task_id, "endpoint_qualified": False,
                "implementation_sha256": identity}
    require(baseline_run is not None, "Candidate requires its frozen baseline input")
    run = Path(reference_run).resolve(strict=True)
    receipt = read_json(run.parent / "evaluation_result.json")
    from .packed_gnn_reference import TASKS as PACKED_TASKS, admit as admit_packed
    if spec.task_id in PACKED_TASKS and (run / "reference_summary.json").is_file():
        return admit_packed(spec, native, identity, baseline_run, run, receipt)
    require(receipt.get("artifacts", {}).get("task") == spec.task_id,
            "Reference belongs to a different task")
    require(receipt.get("metrics", {}).get("runs_successfully") == 1.0,
            "Reference evaluation did not succeed")
    require(receipt.get("artifacts", {}).get("reference_admission", {}).get("mode") == "FRESH",
            "Reference is not a native initial-program evaluation")
    require(receipt["artifacts"]["reference_admission"].get("implementation_sha256") == identity,
            "Reference implementation differs; new same-contract qualification is required")
    initial = read_json(run.parent / "candidate_snapshot.json")
    require(initial == native, "Reference initial program or trusted adapter differs")
    for stage, directory in (("input", "00_input"), ("floorplan", "10_floorplan"), ("partition", "20_partition"),
                             ("placement", "30_placement"), ("pdn", "40_pdn")):
        if stage not in spec.upstream_stages:
            continue
        require(sha256(run / directory / "stage_manifest.json")
                == sha256(Path(baseline_run) / directory / "stage_manifest.json"),
                "Candidate/reference upstream identity differs: " + stage)
    # Re-extract authoritative evidence, not just the score from the receipt.
    from .metrics import extract_metrics
    extract_metrics(spec.kind, run, None, task_id=spec.task_id, frozen_input_run=baseline_run)
    return {"mode": "ADOPT", "task_id": spec.task_id,
            "reference_receipt_sha256": sha256(run.parent / "evaluation_result.json")}
