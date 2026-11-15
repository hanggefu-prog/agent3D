"""Decoder-bound loss evaluation: placement + HBT + measurement, never OPT."""
import math
from pathlib import Path

from .reference_latest import read_json, require, sha256, validate_artifacts

STA_DIRECTORY = "55_post_placement_sta"
TASK_DECODERS = {
    "tp_gnn_bin_timing": "spatial-bin",
    "tp_gnn_post_placement_timing": "global-2way",
    "tp_gnn_binary_loss_timing": "binary-probability",
    "tp_gnn_loss_decoder_timing": "global-2way",
}


def validate_recipe(config, source, *, expected_decoder):
    """Reject the wrong decoder before cloning, training or physical execution."""
    require(expected_decoder in TASK_DECODERS.values(), "Unsupported timing-task decoder")
    partition = config.get("partition", {})
    require(partition.get("algorithm") == "tp-gnn-flat", "Timing task requires tp-gnn-flat")
    configured = partition.get("config")
    require(isinstance(configured, str) and bool(configured), "Timing task requires an explicit partition recipe")
    path = Path(configured)
    if not path.is_absolute():
        path = Path(source) / "Place-3D" / path
    recipe = read_json(path).get("partition", {})
    # This is the maintained TPGNNConfig default; actual native reports must
    # still record the decoder explicitly and pass validate_snapshot below.
    require(recipe.get("tp_gnn_decoder", "global-2way") == expected_decoder,
            "Timing task requires " + expected_decoder + " decoding")


def snapshot_command(workspace, run_dir, config, python, environment):
    from open3dbench.pdks import get_pdk
    platform = (Path(workspace) / "OpenROAD-3D/flow/platforms"
                / (get_pdk(config["pdk"]).openroad_platform + "_3D"))
    return [python, "-m", "open3dbench.backend.post_placement_sta",
            "--checkpoint", str(Path(run_dir) / "50_hbt/checkpoint.json"),
            "--platform-dir", str(platform), "--output-dir", str(Path(run_dir) / STA_DIRECTORY),
            "--openroad", environment.get("OPENROAD_EXE", "/usr/local/bin/openroad")]


def validate_physical_snapshot(run_dir):
    """Verify the shared no-OPT physical boundary, without a training contract."""
    from open3dbench.stage_manifest import validate_stage_manifest
    from open3dbench.backend.opt_handoff import load_opt_handoff

    run = Path(run_dir).resolve(strict=True)
    for name in ("60_opt", "70_cts", "75_global_route", "80_route", "90_report", "95_thermal"):
        require(not (run / name).exists(), "No-OPT timing run contains a later physical stage: " + name)
    for stage, directory in (("partition", "20_partition"), ("placement", "30_placement"),
                             ("pdn", "40_pdn"), ("hbt", "50_hbt")):
        validate_stage_manifest(run / directory / "stage_manifest.json", expected_stage=stage,
                                verify_artifacts=True)
    checkpoint = run / "50_hbt/checkpoint.json"
    handoff = load_opt_handoff(checkpoint)
    row = read_json(run / STA_DIRECTORY / "metrics.json")
    require(row.get("stage") == "post-placement-sta" and row.get("status") == "success",
            "Missing successful post-placement STA")
    source = row.get("input", {})
    require(source.get("required_checkpoint_stage") == "hbt"
            and source.get("physical_placement_mutated") is False
            and source.get("checkpoint_sha256") == sha256(checkpoint), "STA checkpoint binding differs")
    analysis = row.get("analysis", {})
    require(analysis.get("timing_view") == "unified-flat-two-die"
            and analysis.get("parasitics") == "placement-estimated-die-rc-plus-passive-hbt-pi"
            and analysis.get("setup") == "max-delay" and analysis.get("signoff_ready") is False,
            "STA timing boundary differs")
    records = row.get("artifact_records", {})
    artifacts = row.get("artifacts", {})
    required = {"timing_snapshot", "setup_report", "hold_report", "check_setup_report",
                "parasitic_annotation_report", "passive_assembly_verilog",
                "passive_assembly_spef", "hbt_boundary_timing", "replay_tcl"}
    require(required <= set(records) and required <= set(artifacts), "Incomplete STA evidence")
    require(all(records[k].get("path") == artifacts[k] for k in required), "STA artifact aliases differ")
    validate_artifacts(run / STA_DIRECTORY, records)
    require(row.get("hbt_counts", {}).get("total") == len(handoff.boundaries), "HBT count differs")
    require(row.get("boundary_coverage", {}).get("sta_unconstrained_endpoints") == 0,
            "STA has unknown or unconstrained endpoints")
    require(row.get("reported_paths", {}).get("setup", {}).get("reported_path_count", 0) > 0,
            "STA has no constrained setup paths")
    for name in ("wns_setup", "tns_setup"):
        value = row.get("metrics", {}).get(name)
        require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value),
                "Missing finite global " + name)
    require(row["metrics"]["tns_setup"] <= 0, "TNS must be non-positive")
    return row


def validate_snapshot(run_dir, *, expected_decoder="spatial-bin", evolution_task=None):
    require(expected_decoder in TASK_DECODERS.values(), "Unsupported timing-task decoder")
    run = Path(run_dir).resolve(strict=True)
    row = validate_physical_snapshot(run)
    report = read_json(run / "20_partition/tp_gnn_report.json")
    if evolution_task is not None:
        from .gnn_evolution_timing import validate_native_evolution, TASK_DECODERS as EVOLUTION_DECODERS
        require(EVOLUTION_DECODERS.get(evolution_task) == expected_decoder,
                "Evolution task identity differs from its decoder")
        validate_native_evolution(run, task_id=evolution_task)
    else:
        require(expected_decoder != "binary-probability", "Binary timing requires its explicit evolution task")
        require(report.get("training", {}).get("loss_context_interface") == "timing-context-v2",
                "Reference must use the current timing-context-v2 native loss hook")
    from .placement_baseline import validate_tp_gnn_partition
    validate_tp_gnn_partition(run, expected_method="tp-gnn-flat", require_bound_report=True,
                              expected_decoder=expected_decoder)
    require(report.get("config", {}).get("decoder") == expected_decoder,
            "Reference must use " + expected_decoder + " decoding")
    if expected_decoder == "spatial-bin":
        bins = report.get("clustering", {}).get("spatial_bins", {})
        require(bins.get("postclosure", {}).get("accepted") is True
                and bins.get("handoff_constraints", {}).get("accepted") is True,
                "Spatial-bin feasibility did not pass")
        row["bin_max_area_error"] = bins["postclosure"]["max_area_error"]
    row["partition_config"] = {k: v for k, v in report["config"].items()
                               if k not in {"output_dir", "timing_input", "prototype_def", "prototype_params",
                                            "timing_features", "timing_net_weights", "timing_connections",
                                            "prototype_gpdef", "prototype_params_json", "prototype_die_regions_json",
                                            "evolution_program"}}
    return row


def timing_metrics(run_dir, reference_run, *, expected_decoder="spatial-bin"):
    from .metrics import tp_gnn_learning_objective_metrics, _relative_cost, RATIO_DENOMINATOR_FLOOR

    row = validate_snapshot(run_dir, expected_decoder=expected_decoder)
    reference = validate_snapshot(reference_run, expected_decoder=expected_decoder) if reference_run else None
    # Retain current native diagnostics and all existing training/assignment gates.
    result = tp_gnn_learning_objective_metrics(run_dir, reference_run)
    require(result["partition_constraints_passed"] == 1.0, "Global partition balance failed")
    wns, tns = (row["metrics"][key] for key in ("wns_setup", "tns_setup"))
    wcost = tcost = 1.0
    if reference is not None:
        require(row["hbt_model"] == reference["hbt_model"], "Candidate/reference HBT models differ")
        require(row["partition_config"] == reference["partition_config"], "Candidate/reference partition recipes differ")
        wcost = _relative_cost(max(0., -wns), max(0., -reference["metrics"]["wns_setup"]))
        tcost = _relative_cost(max(0., -tns), max(0., -reference["metrics"]["tns_setup"]))
    cost = 0.6 * wcost + 0.4 * tcost
    result.pop("timing_weighted_cut_score", None)
    result.update(combined_score=1. / max(cost, RATIO_DENOMINATOR_FLOOR),
                  normalized_sta_cost=cost, normalized_wns_cost=wcost, normalized_tns_cost=tcost,
                  sta_wns_weight=0.6, sta_tns_weight=0.4, final_global_wns_setup_ns=wns,
                  final_global_tns_setup_ns=tns, opt_passes=0.0, sta_graph_complete=1.0,
                  partition_decoder=expected_decoder,
                  timing_evaluation_boundary="post-placement-hbt-no-opt")
    if expected_decoder == "spatial-bin":
        result["spatial_bin_max_area_error"] = row["bin_max_area_error"]
    from .observed_diagnostics import assignment_comparison
    result.update(assignment_comparison(run_dir, reference_run))
    return result
