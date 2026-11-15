"""Artifact-only collectors for the two contest TP-GNN evolution tasks.

The endpoint is the native one-pass OPT checkpoint, before CTS. Candidate
source is never executed by this collector.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import yaml

from .reference_latest import read_json, require, sha256, validate_artifacts
from local_tasks.tp_gnn_local_task_definitions import PACKAGES, validate_candidate

BINARY_TASK = "tp_gnn_binary_loss_timing"
LEGACY_TASK = "tp_gnn_loss_decoder_timing"
TASK_DECODERS = {BINARY_TASK: "binary-probability", LEGACY_TASK: "global-2way"}
BASELINE_RECIPES = {
    BINARY_TASK: "config/partition/tp_gnn_binary_regularized.json",
    LEGACY_TASK: "config/partition/tp_gnn_fast.json",
}
# Canonical JSON hashes bind every field while allowing whitespace-only edits.
BASELINE_RECIPE_SHA256 = {
    BINARY_TASK: "8920103bbeea4b46ed863cdc0121672ba774707e638563df976fbdaf2e46f982",
    LEGACY_TASK: "7c69dbfaa55f24134808191efe9ea8301ccb5d416a356d6d136f18a4889a48ec",
}
TIMING_FILES = ("timing_features.csv", "timing_net_weights.csv", "timing_manifest.json",
                "openroad_full_partition_last.connections.tsv")


def _source_root(mounted_source=Path("/workspace")):
    """Find the pinned physical source in both checkout and evaluator mounts."""
    for root in (Path(__file__).resolve().parents[2], mounted_source):
        if (root / "Place-3D/config/partition/tp_gnn_fast.json").is_file():
            return root
    raise ValueError("Pinned physical source is unavailable to the GNN result reader")


def _recipe(source, task_id):
    require(task_id in TASK_DECODERS, "Unknown GNN evolution timing task")
    path = Path(source) / "Place-3D" / BASELINE_RECIPES[task_id]
    value = read_json(path)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(hashlib.sha256(encoded).hexdigest() == BASELINE_RECIPE_SHA256[task_id],
            "GNN evolution baseline recipe changed; new task qualification is required")
    return path.resolve(strict=True), value["partition"]


def validate_recipe(config, source, *, task_id):
    """Check the trusted baseline before any workspace or physical execution."""
    expected, _ = _recipe(source, task_id)
    partition = config.get("partition", {})
    require(partition.get("algorithm") == "tp-gnn-flat", "GNN evolution requires tp-gnn-flat")
    value = partition.get("config")
    require(isinstance(value, str) and bool(value), "GNN evolution requires an explicit baseline recipe")
    configured = Path(value)
    if not configured.is_absolute():
        configured = Path(source) / "Place-3D" / configured
    configured = configured.resolve(strict=True)
    recipe = read_json(configured)
    baseline = read_json(expected)
    seed = partition.get("seed")
    require(type(seed) is int and 0 <= seed < 2**31,
            "GNN evolution task has an invalid paired seed")
    baseline["partition"]["seed"] = seed
    require(recipe == baseline, "GNN evolution task changed a field other than the native seed")
    if configured != expected:
        require(configured.name == "partition.json" and configured.parent.name == "method_seeds",
                "GNN evolution native seed file is not the trusted matrix adapter")
    require(config.get("placement", {}).get("seed") == seed,
            "GNN evolution needs paired partition and placement seeds")
    program = partition.get("evolution_program")
    require(isinstance(program, str) and bool(program), "GNN evolution requires its explicit program capsule")


def _finite(value, message):
    require(type(value) in (int, float) and math.isfinite(value), message)
    return value


def _integer(value, message, *, minimum=1):
    require(type(value) is int and value >= minimum, message)
    return value


def _bound(stage, manifest, key, filename):
    records = manifest.get("artifacts", {})
    record = records.get(key)
    require(isinstance(record, dict), "Missing hash-bound native artifact: " + key)
    require(record.get("path") == filename, "Native artifact path differs: " + key)
    validate_artifacts(stage, {key: record})
    return stage / filename


def _provenance(receipt, capsule, task_id):
    require(isinstance(receipt, dict), "Missing native evolution receipt")
    interface = PACKAGES[task_id][1]
    source = read_json(capsule)["source"]
    require(receipt.get("interface_version") == interface, "Native evolution interface differs")
    require(receipt.get("program_sha256") == sha256(capsule)
            and receipt.get("capsule_sha256") == sha256(capsule), "Native evolution capsule hash differs")
    require(receipt.get("source_sha256") == hashlib.sha256(source.encode()).hexdigest()
            and receipt.get("source_bytes") == len(source.encode()), "Native evolution source hash differs")


def _native_config(report, task_id, expected_seed):
    _, recipe = _recipe(_source_root(), task_id)
    recipe = dict(recipe, seed=expected_seed)
    config = report.get("config", {})
    require(isinstance(config, dict), "Missing native partition recipe")
    for key, value in recipe.items():
        if key == "method":
            continue
        field = key[len("tp_gnn_"):] if key.startswith("tp_gnn_") else key
        require(type(config.get(field)) is type(value) and config[field] == value,
                "Native partition recipe changed: " + field)
    require(config.get("decoder") == TASK_DECODERS[task_id], "Native partition decoder differs")
    if task_id == LEGACY_TASK:
        require(config.get("learning_rate") == .001, "Legacy GNN learning rate changed")
    return config


def _csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _validate_binary(stage, manifest, report, evolution, steps):
    _, recipe = _recipe(_source_root(), BINARY_TASK)
    updates = recipe["tp_gnn_epochs"]
    selection_start = max(1, math.ceil(updates * recipe["tp_gnn_binary_ramp_end_fraction"]))
    # Match the frozen trainer's final interpolation, including float rounding.
    final_temperature = recipe["tp_gnn_binary_temperature"] + (
        recipe["tp_gnn_binary_temperature_final"] - recipe["tp_gnn_binary_temperature"])
    training = report["training"]
    require(training.get("loss_context_interface") == evolution.get("loss_context_interface") == "binary-loss-v1",
            "Binary loss context differs")
    require(steps == training.get("effective_epochs") == training.get("effective_min_epochs")
            == training.get("epochs_completed") == updates,
            f"Binary GNN must complete exactly {updates} updates")
    require(evolution.get("training_loss_calls") == updates
            and evolution.get("scoring_loss_calls") == updates + 2
            and evolution.get("loss_calls") == 2 * updates + 2
            and evolution.get("score_epoch") == updates,
            "Binary loss hook-call or fixed score-epoch evidence differs")
    require(evolution.get("canonical_diagnostics_protected") is True, "Canonical binary diagnostics are not protected")
    regularization = training.get("regularization", {})
    require(regularization.get("noise_used_in_scoring_or_decoding") is False
            and regularization.get("selection_start_epoch") == selection_start,
            "Binary checkpoint scoring must use the clean final phase")
    selected = _integer(training.get("selected_epoch"), "Missing selected binary epoch", minimum=selection_start)
    require(selected <= updates, "Selected binary epoch exceeds the training budget")
    rows = _csv(_bound(stage, manifest, "binary_training", "binary_training.csv"))
    history = training.get("history")
    require(len(rows) == updates + 1 and isinstance(history, list) and len(history) == updates + 1,
            "Binary training history does not cover all updates")
    for epoch, (row, native) in enumerate(zip(rows, history)):
        require(int(row.get("epoch", -1)) == native.get("epoch") == epoch, "Binary epoch evidence differs")
        for field in ("loss", "canonical_loss", "expected_cut", "area_balance_loss", "binarization_loss"):
            actual = float(row.get(field, "nan"))
            expected = _finite(native.get(field), "Missing binary canonical history field: " + field)
            require(math.isfinite(actual) and actual == expected, "Binary history field differs: " + field)
        require(float(row.get("temperature", "nan")) == final_temperature
                and float(row.get("binarization_weight", "nan")) == recipe["tp_gnn_binary_binarization_weight"],
                "Binary checkpoint comparison settings changed")
        if epoch >= selection_start:
            require(float(row.get("training_logit_noise_scale", "nan")) == 0.,
                    "Binary final training phase must be noise-free")
    final = _finite(training.get("final_loss"), "Missing finite selected binary loss")
    require(final == history[selected]["loss"] == training.get("selected_metrics", {}).get("loss"),
            "Selected binary loss differs from its history")
    require(history[selected]["loss"] == min(row["loss"] for row in history[selection_start:]),
            "Binary checkpoint was not selected using the fixed final objective")


def _validate_legacy(stage, report, evolution, steps):
    training = report["training"]
    require(training.get("cuda_available") is True, "Legacy native CUDA availability was not measured")
    require(training.get("loss_context_interface") == evolution.get("loss_context_interface") == "timing-context-v2",
            "Legacy loss context differs")
    require(steps >= 100 and evolution.get("loss_calls") == steps,
            "Legacy loss hook must run once per optimizer update")
    count = _integer(training.get("epochs_completed"), "Missing legacy completed epoch count")
    per_epoch = _integer(training.get("optimizer_steps_per_epoch"), "Missing legacy steps per epoch")
    effective = max(20, math.ceil(100 / per_epoch))
    minimum = max(10, math.ceil(100 / per_epoch))
    require(count * per_epoch == steps and minimum <= count <= effective
            and training.get("max_epochs") == effective, "Legacy minimum-step training budget changed")
    rows = _csv(stage / "training_log.csv")
    require(len(rows) == count, "Legacy training history does not cover completed epochs")
    for epoch, row in enumerate(rows, 1):
        require(int(row.get("epoch", -1)) == epoch and math.isfinite(float(row.get("loss", "nan"))),
                "Legacy epoch evidence is invalid")
    require(float(rows[-1]["loss"]) == training.get("final_loss"), "Legacy final training loss differs")
    decoder = report.get("evolution_decoder", {})
    _provenance(decoder, stage / "evolution_program.json", LEGACY_TASK)
    require(type(decoder.get("decoder_calls")) is int and decoder["decoder_calls"] == 1
            and decoder.get("labels_validated") is True
            and decoder.get("protected_closure_affinities") is True and decoder.get("iterations_limit") == 50,
            "Evolved decoder call, label validation or protected closure evidence differs")


def validate_native_evolution(run_dir, *, task_id):
    """Validate hash-bound code and actual native hook evidence after stages."""
    require(task_id in TASK_DECODERS, "Unknown GNN evolution timing task")
    stage = Path(run_dir) / "20_partition"
    manifest = read_json(stage / "stage_manifest.json")
    capsule = _bound(stage, manifest, "evolution_program", "evolution_program.json")
    filename, _, _ = PACKAGES[task_id]
    validate_candidate({"config_overrides": {}, "source_patch": "",
                        "candidate_files": {filename: read_json(capsule)}}, task_id)
    report = read_json(stage / "tp_gnn_report.json")
    run_config = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())
    seed = run_config.get("partition", {}).get("seed")
    require(type(seed) is int and 0 <= seed < 2**31,
            "Missing native partition seed in the completed run")
    config = _native_config(report, task_id, seed)
    training = report.get("training", {})
    require(training.get("status") == "trained" and str(training.get("device", "")).startswith("cuda"),
            "GNN evolution requires completed native CUDA training")
    evolution = training.get("evolution")
    _provenance(evolution, capsule, task_id)
    steps = _integer(training.get("optimizer_steps_completed"), "Missing actual optimizer-step evidence")
    handoff = report.get("lol_handoff_contract", {})
    require(handoff.get("constraints", {}).get("accepted") is True
            and handoff.get("supernode_integrity", {}).get("preserved") is True,
            "Native protected feasibility closure did not pass")
    if task_id == BINARY_TASK:
        _validate_binary(stage, manifest, report, evolution, steps)
    else:
        _validate_legacy(stage, report, evolution, steps)
    return dict(evolution, task_id=task_id, optimizer_steps_completed=steps,
                partition_decoder=config["decoder"])


def _independent(run):
    from .reference_local_partition import independent_partition_metrics
    return independent_partition_metrics(run, source_root=_source_root(),
                                         timing_directory="20_partition/timing_input")


def _binary_diagnostics(run, result):
    report = read_json(Path(run) / "20_partition/tp_gnn_report.json")
    training = report["training"]
    for source, target in (("probability_uncertain_fraction", "binary_probability_uncertain_fraction"),
                           ("probability_uncertain_area_fraction", "binary_probability_uncertain_area_fraction")):
        value = _finite(training.get(source), "Missing measured binary probability uncertainty")
        require(0 <= value <= 1, "Binary uncertainty fraction is outside [0,1]")
        result[target] = value
    stages = report.get("assignment_stages", {})
    for name, prefix in (("raw_binary_threshold", "binary_raw"), ("lol_feasible_supernodes", "binary_final")):
        row = stages.get(name, {})
        ratio = _finite(row.get("top_area_ratio"), "Missing measured binary assignment area ratio")
        require(0 <= ratio <= 1, "Binary assignment area ratio is outside [0,1]")
        result[prefix + "_top_area_ratio"] = ratio
    moved = report.get("lol_handoff_contract", {}).get("supernode_integrity", {}).get("moved_area_fraction")
    moved = _finite(moved, "Missing measured binary closure movement")
    require(0 <= moved <= 1, "Binary closure moved area fraction is outside [0,1]")
    result["binary_closure_moved_area_fraction"] = moved


def _opt_snapshot(run_dir, task_id):
    """Read the hash-bound, single-pass native OPT measurement."""
    from open3dbench.stage_manifest import validate_stage_manifest

    run = Path(run_dir).resolve(strict=True)
    for later in ("70_cts", "75_global_route", "80_route", "90_report", "95_thermal"):
        require(not (run / later).exists(), "Contest OPT run contains a later stage")
    for stage, directory in (("partition", "20_partition"), ("placement", "30_placement"),
                             ("pdn", "40_pdn"), ("hbt", "50_hbt"), ("opt", "60_opt")):
        validate_stage_manifest(run / directory / "stage_manifest.json", expected_stage=stage,
                                verify_artifacts=True)
    validate_native_evolution(run, task_id=task_id)
    stage = run / "60_opt"
    manifest = read_json(stage / "stage_manifest.json")
    metrics = read_json(_bound(stage, manifest, "metrics", "metrics.json"))
    checkpoint = read_json(_bound(stage, manifest, "checkpoint", "checkpoint.json"))
    require(checkpoint.get("stage") == "opt" and
            checkpoint.get("source_checkpoint") == "../50_hbt/checkpoint.json",
            "OPT checkpoint does not follow the HBT result")
    require(metrics.get("status") == "success" and metrics.get("iterations_run") == 1
            and metrics.get("passes_run") == 1 and
            metrics.get("die_passes") == {"bottom": 1, "top": 1},
            "Contest requires exactly one native OPT pass per die")
    validity = metrics.get("timing_validity", {})
    require(validity.get("graph_completeness") == "passed" and
            validity.get("global_wns_role") == "pre-cts-placement-estimated" and
            validity.get("authoritative_timing_view") == "continuous-passive-hbt" and
            validity.get("unclocked_or_unconstrained_endpoints") == 0 and
            validity.get("combinational_loops") == 0 and
            validity.get("missing_liberty_cells") == 0 and
            validity.get("signoff_ready") is False and
            metrics.get("parasitics", {}).get("assembly") ==
            "continuous-passive-hbt-placement-estimated-rc",
            "OPT timing view differs from the official endpoint")
    timing = metrics.get("final_global", {})
    wns = _finite(timing.get("wns_setup"), "Missing post-OPT WNS")
    tns = _finite(timing.get("tns_setup"), "Missing post-OPT TNS")
    require(tns <= 0, "Post-OPT TNS must be non-positive")
    report = read_json(run / "20_partition/tp_gnn_report.json")
    require(report.get("config", {}).get("decoder") == TASK_DECODERS[task_id],
            "Native partition decoder differs")
    from .placement_baseline import validate_tp_gnn_partition
    validate_tp_gnn_partition(run, expected_method="tp-gnn-flat", require_bound_report=True,
                              expected_decoder=TASK_DECODERS[task_id])
    return {"wns": wns, "tns": tns, "metrics": metrics,
            "partition_config": {k: v for k, v in report["config"].items()
                                 if k not in {"output_dir", "timing_input", "prototype_def",
                                              "prototype_params", "timing_features", "timing_net_weights",
                                              "timing_connections", "prototype_gpdef", "prototype_params_json",
                                              "prototype_die_regions_json", "evolution_program"}}}


def _packed_reference(run_dir, task_id):
    """Read a qualified reference summary without shipping the native run."""
    root = Path(run_dir).resolve(strict=True)
    summary = read_json(root / "reference_summary.json")
    require(isinstance(summary, dict) and set(summary) == {
        "schema_version", "task_id", "wns", "tns", "partition_config",
        "timing_sha256", "assignment_sha256", "native_stage_sha256",
        "independent_partition_metrics"}, "Packed GNN reference schema differs")
    require(summary["schema_version"] == 1 and summary["task_id"] == task_id,
            "Packed GNN reference task differs")
    require(isinstance(summary["partition_config"], dict),
            "Packed GNN partition recipe is missing")
    require(isinstance(summary["timing_sha256"], dict)
            and set(summary["timing_sha256"]) == set(TIMING_FILES)
            and all(isinstance(value, str) and len(value) == 64
                    for value in summary["timing_sha256"].values()),
            "Packed GNN timing input hashes differ")
    require(isinstance(summary["native_stage_sha256"], dict)
            and set(summary["native_stage_sha256"]) == {
                "20_partition", "30_placement", "40_pdn", "50_hbt", "60_opt"}
            and all(isinstance(value, str) and len(value) == 64
                    for value in summary["native_stage_sha256"].values()),
            "Packed GNN native stage evidence is missing")
    require(isinstance(summary["independent_partition_metrics"], dict),
            "Packed GNN independent partition evidence is missing")
    assignment = root / "20_partition/partition_assignment.csv"
    require(sha256(assignment) == summary["assignment_sha256"],
            "Packed GNN reference assignment differs")
    wns = _finite(summary["wns"], "Packed GNN WNS is invalid")
    tns = _finite(summary["tns"], "Packed GNN TNS is invalid")
    require(tns <= 0, "Packed GNN TNS must be non-positive")
    return {"wns": wns, "tns": tns,
            "partition_config": summary["partition_config"],
            "timing_sha256": summary["timing_sha256"]}


def timing_metrics(run_dir, reference_run, *, task_id):
    """Score official 0.4 WNS + 0.6 TNS improvements after one OPT pass."""
    from .observed_diagnostics import assignment_comparison
    require(task_id in TASK_DECODERS, "Unknown GNN evolution timing task")
    decoder = TASK_DECODERS[task_id]
    row = _opt_snapshot(run_dir, task_id)
    packed = bool(reference_run and (Path(reference_run) / "reference_summary.json").is_file())
    reference = (_packed_reference(reference_run, task_id) if packed else
                 _opt_snapshot(reference_run, task_id) if reference_run else None)
    result = _independent(run_dir)
    if reference_run:
        if not packed:
            _independent(reference_run)
        require(row["partition_config"] == reference["partition_config"],
                "Candidate/reference partition recipes differ")
        for name in TIMING_FILES:
            relative = Path("20_partition/timing_input") / name
            expected = (reference["timing_sha256"][name] if packed else
                        sha256(Path(reference_run) / relative))
            require(sha256(Path(run_dir) / relative) == expected,
                    "Candidate/reference frozen timing differs: " + name)
    wns, tns = row["wns"], row["tns"]
    w_improvement = t_improvement = 0.0
    if reference is not None:
        w_ref, t_ref = max(0.0, -reference["wns"]), max(0.0, -reference["tns"])
        require(w_ref > 0.0 and t_ref > 0.0,
                "Official zero-deficit reference rule must be frozen before scoring")
        w_improvement = (w_ref - max(0.0, -wns)) / w_ref
        t_improvement = (t_ref - max(0.0, -tns)) / t_ref
    score = 100.0 * (0.4 * w_improvement + 0.6 * t_improvement)
    result.pop("timing_weighted_cut_score", None)
    result.update(combined_score=score, wns_deficit_improvement=w_improvement,
                  tns_deficit_improvement=t_improvement, sta_wns_weight=0.4,
                  sta_tns_weight=0.6, final_global_wns_setup_ns=wns,
                  final_global_tns_setup_ns=tns, opt_passes=1.0, sta_graph_complete=1.0,
                  partition_decoder=decoder, timing_evaluation_boundary="post-opt-pre-cts")
    result.update(assignment_comparison(run_dir, reference_run))
    if task_id == BINARY_TASK:
        _binary_diagnostics(run_dir, result)
    return result
