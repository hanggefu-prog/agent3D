"""Read-only qualification of FRESH, same-runtime TP-GNN-flat references.

The initial loss, training recipe and frozen partition-last STA inputs are
controller-owned. Native assignments are rescored from the immutable hypergraph,
not from evaluator success flags. This module never dispatches training or EDA.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

import yaml

from . import reference_baseline as baseline
from .benchmark_input import validate_benchmark_input
from .model import load_candidate, load_task, resolve_task_path
from .program_bridge import initial_program_text, materialize_program
from .reference_local_placement import (_expected_overlay, _json, _regular, _source_revision,
    _implementation_name, _implementation_path)
from .reference_store import digest, file_sha

FRAMEWORK = Path(__file__).resolve().parents[1]
SOURCE = FRAMEWORK.parent
TASK = "tp_gnn_learning_objective_reference_20260923"
TP_GNN = "Place-3D/src/partition/lol/methods/tp_gnn.py"
CORE = "Place-3D/src/partition/lol/core.py"
STAGE = "20_partition"
TIMING = "15_timing_extract"
RUNTIME_INPUTS = ("10_floorplan/partition.input", "10_floorplan/partition.names.csv",
                  "10_floorplan/design_params.json")
PARAMS = "10_floorplan/design_params.json"


def _validate_rebased_params(baseline_run, run, physical_sha256):
    """Accept only the three native copy-time path substitutions, not physics edits."""
    expected = _json(baseline_run / PARAMS)
    for key, filename in (("def_input", "design.def"), ("verilog_input", "design.v"),
                          ("sdc_input", "design.sdc")):
        relative = "10_floorplan/" + filename
        sha = physical_sha256.get(relative)
        if not sha:
            raise ValueError("Rebased partition input lacks independently pinned physical identity: " + relative)
        for root in (baseline_run, run):
            if file_sha(_regular(root / relative)) != sha:
                raise ValueError("Rebased partition input changed physical bytes: " + relative)
        expected[key] = str(_regular(run / relative))
    if _json(run / PARAMS) != expected:
        raise ValueError("Partition parameters changed beyond the three copy-time input path substitutions")


def _context(source_root, task_id, case, benchmark_input):
    from .evaluation import _validate_frozen_input
    from .reference_latest import baseline_for_bundle, BASELINE_ID
    source = Path(source_root).resolve(strict=True)
    if task_id != TASK:
        raise ValueError("Native partition reference requires the trusted versioned task checkout")
    spec = load_task(resolve_task_path(FRAMEWORK / "local_tasks", task_id))
    bundle = Path(benchmark_input).resolve(strict=True)
    selected_baseline = baseline_for_bundle(bundle)
    if source != SOURCE and selected_baseline.BASELINE_ID != BASELINE_ID:
        raise ValueError("Legacy partition requires its coupled controller checkout")
    validated = validate_benchmark_input(bundle, task_id=task_id)
    aligned = selected_baseline.validate_reference_bundle(bundle, case, fixed_partition=False)
    config = Path(validated["config"]).resolve(strict=True)
    recipe = yaml.safe_load(config.read_text())
    selected_baseline.validate_reference_config(recipe, case)
    spec = replace(spec, base_config=config)
    candidate = materialize_program(spec, initial_program_text(spec, load_candidate(spec.initial_candidate)), source)
    run = Path(validated["run_dir"]).resolve(strict=True)
    timing = _validate_frozen_input(run / spec.frozen_inputs[0].path, spec.frozen_inputs[0])
    manifest = bundle / "benchmark_input.json" if bundle.is_dir() else bundle
    frozen = {"benchmark_input_manifest_sha256": file_sha(_regular(manifest)),
              "physical_prefix_sha256": aligned["physical_input_sha256"],
              "partition_runtime_inputs_sha256": {name: file_sha(_regular(run / name)) for name in RUNTIME_INPUTS},
              "timing_input": timing}
    return source, spec, candidate, recipe, frozen, run, manifest


def _physical(source, spec, candidate, recipe, revision, image):
    from .reference_latest import baseline_for_config, BASELINE_ID, SOURCE_REVISION
    selected_baseline = baseline_for_config(recipe)
    selected_baseline.validate_reference_image(image)
    if selected_baseline.BASELINE_ID == BASELINE_ID and revision != SOURCE_REVISION:
        raise ValueError("Latest partition source revision differs")
    recipe_file = (FRAMEWORK / "local_bench/reference_configs/partition.json"
                   if selected_baseline.BASELINE_ID == BASELINE_ID else source / selected_baseline.PARTITION_CONFIG)
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError("Partition reference requires a full source commit")
    files = [*sorted((source / "open3dbench").rglob("*.py")),
             *sorted((source / "Place-3D/src/partition/lol").rglob("*.py")),
             source / "Place-3D/src/api.py", recipe_file,
             spec.root / "task.yaml", spec.program_template, spec.program_adapter,
             FRAMEWORK / "local_tasks/tp_gnn_loss_patch.py",
             FRAMEWORK / "local_tasks/tp_gnn_timing_context.py",
             FRAMEWORK / "local_bench/evaluation.py", FRAMEWORK / "local_bench/workspace.py",
             FRAMEWORK / "local_bench/program_bridge.py", FRAMEWORK / "scripts/local_reference.py",
             FRAMEWORK / "scripts/run_local_task_smoke.py", FRAMEWORK / "scripts/codesign_runtime.py"]
    return {"producer_contract": "local-partition-same-runtime-v1", "task_id": TASK,
            "source_revision": revision, "image": image, "configuration": recipe,
            "physical_config_sha256": digest(recipe), "initial_candidate_sha256": digest(candidate),
            "partition_config": _json(recipe_file),
            "implementation_sha256": {_implementation_name(source, path): file_sha(_regular(path)) for path in files},
            "runtime_loading": "isolated-source-python-with-verified-initial-loss-v1",
            "evaluation_boundary": "partition-only-frozen-partition-last-sta", "threads": 4,
            "archive_source_bridge": "unsupported"}


def expected_local_contract(source_root, task_id, case, benchmark_input, *, source_revision, image):
    from .reference_catalog import evaluator_digest, task_digest
    source, spec, candidate, recipe, frozen, _, _ = _context(source_root, task_id, case, benchmark_input)
    if _source_revision(source, clean=True) != source_revision:
        raise ValueError("Partition reference source revision differs from the controller")
    return {"case": case, "endpoint": "partition", "task_digest": task_digest(spec.root),
            "evaluator_digest": evaluator_digest(), "frozen_inputs": frozen,
            "physical": _physical(source, spec, candidate, recipe, source_revision, image)}


def audit_runtime(receipt_path, output_path):
    """Container-only source/image preflight; no model, training or EDA calls."""
    receipt = _json(receipt_path)
    physical = receipt.get("contract", {}).get("physical", {})
    if (receipt.get("reference_purpose") not in {"FRESH", "INTERFACE_PROBE"} or physical.get("task_id") != TASK
            or physical.get("image") != os.environ.get("OPEN3DBENCH_IMAGE_ID")):
        raise ValueError("Partition runtime audit differs from its FRESH image/task contract")
    source = Path(receipt["source_root"]).resolve(strict=True)
    measured = {name: file_sha(_regular(_implementation_path(source, name))) for name in physical["implementation_sha256"]}
    if measured != physical["implementation_sha256"]:
        raise ValueError("Partition runtime source differs; no physical process was dispatched")
    value = {"schema_version": 1, "image": physical["image"], "source_root": str(source),
             "reference_purpose_sha256": file_sha(receipt_path), "implementation_sha256": measured,
             "python_executable": sys.executable}
    with Path(output_path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _trusted_core(source):
    # Load only the pure trusted parser/metric module, never the candidate's
    # training module or its import path. No torch/CUDA imports are necessary.
    name = "_agentic3d_reference_partition_core"
    path = _regular(source / CORE)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def independent_partition_metrics(run, source_root=SOURCE, *, timing_directory=TIMING):
    """Recompute counts, balance and cut from canonical assignment + frozen nets."""
    from .partition_assignment import read_assignment
    from .metrics import tp_gnn_learning_objective_metrics
    source, run = Path(source_root), Path(run)
    core = _trusted_core(source)
    stage = run / STAGE
    raw = _json(stage / "partition_metrics.json")
    tiers = read_assignment(_regular(stage / "partition_assignment.csv"), raw)
    _, instances, nets = core.parse_lol_input(_regular(run / "10_floorplan/partition.input"))
    names = core.load_name_map(_regular(run / "10_floorplan/partition.names.csv"))
    if (not instances or not nets or len({i.name for i in instances}) != len(instances)
            or set(names) != {i.name for i in instances} or set(names.values()) != set(tiers)):
        raise ValueError("Native assignment does not cover the immutable instance inventory exactly")
    weights = core.load_timing_net_weights(_regular(run / timing_directory / "timing_net_weights.csv"), len(nets))
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Frozen timing weights must be finite and positive")
    criticalities = core.load_timing_net_criticalities(run / timing_directory / "timing_net_weights.csv", len(nets))
    actual = core.compute_metrics(instances, nets, {name: tiers[public] for name, public in names.items()},
                                  weights, criticalities)
    with _regular(stage / "partition_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("Native partition metrics must contain exactly one data row")
    method = _json(stage / "partition_manifest.json")
    for key, expected in actual.items():
        published_key = key + "_native" if key in {
            "top_area", "bottom_area", "top_macro_area", "bottom_macro_area",
            "top_standard_cell_area", "bottom_standard_cell_area"} else key
        for row, column in ((rows[0], key), (method.get("metrics", {}), key),
                            (raw.get("method_metrics", {}), key), (raw, published_key)):
            value = row.get(column)
            if (isinstance(value, bool) or value is None or not math.isfinite(float(value))
                    or not math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-9)):
                raise ValueError("Native partition metrics disagree with independent assignment scoring: " + key)
    result = tp_gnn_learning_objective_metrics(run, run)
    if result.get("partition_constraints_passed") != 1 or result.get("combined_score") != 1:
        raise ValueError("Native initial partition fails the unchanged task constraints")
    return result


def _training_evidence(stage, physical):
    report = _json(stage / "tp_gnn_report.json")
    training, config = report.get("training", {}), _json(stage / "training_config.json")
    recipe = physical["partition_config"]["partition"]
    required = {"seed": recipe["seed"], "embedding_dim": recipe["tp_gnn_embedding_dim"],
                "epochs": recipe["tp_gnn_epochs"], "min_epochs": recipe["tp_gnn_min_epochs"],
                "min_optimizer_steps": recipe["tp_gnn_min_optimizer_steps"],
                "batch_size": recipe["tp_gnn_batch_size"], "random_walks": recipe["tp_gnn_random_walks"],
                "walk_length": recipe["tp_gnn_walk_length"], "negative_samples": recipe["tp_gnn_negative_samples"],
                "sample_size_1": recipe["tp_gnn_sample_size_1"], "sample_size_2": recipe["tp_gnn_sample_size_2"],
                "max_train_nodes": recipe["tp_gnn_max_train_nodes"],
                "convergence_patience": recipe["tp_gnn_convergence_patience"],
                "convergence_tolerance": recipe["tp_gnn_convergence_tolerance"],
                "mode": "random-walk-negative-sampling", "loss": "OpenAgentic3DBench-evolved compute_tp_gnn_loss"}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Native training did not use the fixed reference recipe and initial loss hook")
    if (training.get("status") != "trained" or training.get("seed") != recipe["seed"]
            or training.get("requested_epochs") != recipe["tp_gnn_epochs"]
            or training.get("min_optimizer_steps") != recipe["tp_gnn_min_optimizer_steps"]
            or training.get("cuda_available") is not True or not str(training.get("device")).startswith("cuda")
            or training.get("loss_semantics") != "OpenAgentic3DBench-evolved objective over frozen positive/negative samples"):
        raise ValueError("Native partition lacks completed CUDA training under the trusted loss interface")
    with _regular(stage / "training_log.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    count, steps, per_epoch = (training.get(key) for key in
        ("epochs_completed", "optimizer_steps_completed", "optimizer_steps_per_epoch"))
    if (any(type(value) is not int or value <= 0 for value in (count, steps, per_epoch))
            or count != len(rows) or steps < recipe["tp_gnn_min_optimizer_steps"]
            or count * per_epoch != steps or count < config.get("effective_min_epochs", math.inf)
            or count > config.get("effective_epochs", -1)
            or training.get("max_epochs") != config.get("effective_epochs")
            or training.get("optimizer_steps_per_epoch") != config.get("optimizer_steps_per_epoch")):
        raise ValueError("Native partition training budget/optimizer evidence is incomplete")
    minimum_epochs = math.ceil(recipe["tp_gnn_min_optimizer_steps"] / per_epoch)
    if (config["effective_epochs"] != max(1, recipe["tp_gnn_epochs"], minimum_epochs)
            or config["effective_min_epochs"] != max(1, recipe["tp_gnn_min_epochs"], minimum_epochs)):
        raise ValueError("Native partition effective training budget differs from the fixed minimum-step policy")
    for epoch, row in enumerate(rows, 1):
        if int(row.get("epoch", -1)) != epoch or not math.isfinite(float(row.get("loss", "nan"))):
            raise ValueError("Native partition epoch log is invalid")
    if not rows or float(rows[-1]["loss"]) != training.get("final_loss"):
        raise ValueError("Native partition final loss disagrees with its epoch log")
    diagnostics = training.get("final_embedding_diagnostics", {})
    for key in ("total_dimension_variance", "sample_pair_cosine_median"):
        value = diagnostics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Native embedding diagnostics are missing or nonfinite")
    epoch_rows = re.findall(r"\[tp-gnn\] epoch (\d+)/(\d+): loss=([^\s]+) batches=(\d+) optimizer_steps=(\d+)",
                           _regular(stage / "run.log").read_text())
    if (len(epoch_rows) != count or any(int(item[0]) != index or int(item[1]) != config["effective_epochs"]
            or int(item[3]) != per_epoch or int(item[4]) != index * per_epoch
            or not math.isfinite(float(item[2]))
            or not math.isclose(float(item[2]), float(rows[index - 1]["loss"]), rel_tol=1e-7, abs_tol=1e-8)
            for index, item in enumerate(epoch_rows, 1))):
        raise ValueError("Native training log does not evidence the reported loss/optimizer calls")
    for name in ("initial_embeddings.npy", "embeddings.npy"):
        if _regular(stage / name).stat().st_size == 0:
            raise ValueError("Native partition training embeddings are empty")
    return {"epochs": count, "optimizer_steps": steps}


def probe_candidate(spec, source):
    """The bounded, controller-authored nonidentity interface probe; not a model candidate."""
    initial = initial_program_text(spec, load_candidate(spec.initial_candidate))
    if initial.count("return -(") != 1:
        raise ValueError("Initial loss anchor differs; refusing to invent a probe")
    return materialize_program(spec, initial.replace("return -(", "return -1.1 * (", 1), source)


def _qualify_native_partition(evaluation_root, task_id, case, *, source_root, benchmark_input, probe_reference=None):
    from .placement_baseline import validate_tp_gnn_partition
    from .reference_catalog import evaluator_digest, task_digest
    evaluation = Path(evaluation_root).resolve(strict=True)
    source, spec, candidate, recipe, frozen, baseline_run, manifest_path = _context(
        source_root, task_id, case, benchmark_input)
    from .reference_latest import baseline_for_config, BASELINE_ID, bind_recipe_paths
    latest = baseline_for_config(recipe).BASELINE_ID == BASELINE_ID
    timing_directory = STAGE + "/timing_input" if latest else TIMING
    parent, run = evaluation.parent, evaluation / "run"
    workspace = evaluation / "workspace/Open3DBench-V2.0"
    purpose_path, ledger_path = parent / "reference-purpose.json", parent / (evaluation.name + ".attempt.json")
    purpose, ledger = _json(purpose_path), _json(ledger_path)
    physical = _physical(source, spec, candidate, recipe, _source_revision(source, clean=True), purpose.get("image"))
    contract = {"case": case, "endpoint": "partition", "task_digest": task_digest(spec.root),
                "evaluator_digest": evaluator_digest(), "frozen_inputs": frozen, "physical": physical}
    expected_purpose = "FRESH" if probe_reference is None else "INTERFACE_PROBE"
    reference_run = None if probe_reference is None else probe_reference["qualification"]["baseline"]["source_run"]
    if probe_reference is not None:
        if probe_reference["contract"] != contract:
            raise ValueError("Probe authority differs from the same-runtime initial reference contract")
        candidate = probe_candidate(spec, source)
    if (purpose.get("schema_version") != 1 or purpose.get("reference_purpose") != expected_purpose
            or purpose.get("reference_run", "missing") != reference_run or purpose.get("contract") != contract
            or purpose.get("evaluation_root") != str(evaluation) or purpose.get("source_root") != str(source)):
        raise ValueError("Partition reference lacks exact controller FRESH provenance")
    candidate_path = parent / "candidate/candidate.json"
    if _json(candidate_path) != candidate or purpose.get("candidate_sha256") != file_sha(candidate_path):
        raise ValueError("Partition reference must be the unmodified initial loss, not an evolved candidate")
    required = {"kind": "local-task-smoke-evaluation", "task": task_id,
                "candidate_sha256": file_sha(candidate_path), "source_commit": physical["source_revision"],
                "image": physical["image"], "reference": reference_run, "create_routing_reference": False,
                "reference_purpose": expected_purpose, "reference_purpose_sha256": file_sha(purpose_path)}
    if (ledger.get("schema_version") != 2 or ledger.get("state") != "exited" or ledger.get("exit_code") != 0
            or ledger.get("evaluator_process_started") is not True
            or any(ledger.get("contract", {}).get(key) != value for key, value in required.items())):
        raise ValueError("Partition reference lacks a successful controller-owned native attempt")
    audit_path = parent / "runtime-source.json"
    if _json(audit_path) != {"schema_version": 1, "image": physical["image"], "source_root": str(source),
            "reference_purpose_sha256": file_sha(purpose_path), "implementation_sha256": physical["implementation_sha256"],
            "python_executable": "/usr/bin/python3"}:
        raise ValueError("Partition reference source/runtime audit differs")
    evidence = {}
    def add(path, role):
        path = _regular(path)
        evidence[str(path)] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path), "role": role}
    for path, role in ((purpose_path, "reference-purpose"), (ledger_path, "controller-execution"),
                       (audit_path, "runtime-source-audit"), (candidate_path, "initial-loss"),
                       (manifest_path, "frozen-input-manifest")):
        add(path, role)
    for group in ("physical_prefix_sha256", "partition_runtime_inputs_sha256"):
        for name, expected in frozen[group].items():
            for root in (baseline_run, run):
                path = _regular(root / name)
                if latest and root == run and name in {"00_input/design_params.json", PARAMS}:
                    from .reference_local_placement import _validate_copied_input_params
                    _validate_copied_input_params(baseline_run, run, name, frozen["physical_prefix_sha256"])
                    add(path, "rebased-frozen-parameters")
                    continue
                if name == PARAMS and root == run:
                    _validate_rebased_params(baseline_run, run, frozen["physical_prefix_sha256"])
                    add(path, "rebased-frozen-parameters")
                    continue
                if file_sha(path) != expected:
                    raise ValueError("Partition frozen input changed: " + name)
                add(path, "immutable-partition-input")
    frozen_dir = evaluation / "harness_inputs/frozen/tp_gnn_timing"
    for name, record in frozen["timing_input"].items():
        for root in (baseline_run / spec.frozen_inputs[0].path, frozen_dir, run / timing_directory):
            path = _regular(root / name)
            if path.stat().st_size != record["bytes"] or file_sha(path) != record["sha256"]:
                raise ValueError("Partition frozen STA artifact changed: " + name)
            add(path, "immutable-partition-last-sta")
    # The maintained native runner publishes exactly this local compatibility
    # link. It is not evidence itself and cannot point outside the current run.
    link = run / STAGE / "timing_input"
    if latest and (link.is_symlink() or not link.is_dir() or (run / TIMING).exists()):
        raise ValueError("Latest partition must own its timing_input directory, without a stage 15 alias")
    if not latest and (not link.is_symlink() or os.readlink(link) != "../" + TIMING):
        raise ValueError("Partition timing compatibility view does not reference this run's frozen stage 15")
    result_path = evaluation / "evaluation_result.json"
    artifacts = _json(result_path).get("artifacts", {})
    if any(artifacts.get(key) != value for key, value in {
            "task": task_id, "run_dir": str(run), "workspace": str(workspace),
            "candidate": str(candidate_path.parent), "output_dir": str(evaluation), "exit_code": 0}.items()):
        raise ValueError("Partition native evaluator provenance paths/task differ")
    add(result_path, "native-process-provenance")
    config_path = evaluation / "resolved_config.yaml"
    expected_config = json.loads(json.dumps(recipe))
    if latest:
        expected_config = bind_recipe_paths(expected_config, source)
    expected_config["partition"]["timing_input"] = str(frozen_dir)
    if yaml.safe_load(_regular(config_path).read_text()) != expected_config:
        raise ValueError("Partition native resolved physical recipe differs")
    add(config_path, "resolved-physical-configuration")
    expected_command = ["/usr/bin/python3", "-m", "open3dbench", "--repo", str(workspace), "run",
                        str(config_path), "--from", "partition" if latest else "timing_extract", "--to", "partition", "--resume", "--run-dir", str(run)]
    invocations = artifacts.get("native_flow_execution")
    if not isinstance(invocations, list) or len(invocations) != 1:
        raise ValueError("Partition reference requires exactly one evidenced native flow invocation")
    invocation = invocations[0]
    if (artifacts.get("command") != expected_command or invocation.get("command") != expected_command
            or invocation.get("cwd") != str(workspace) or invocation.get("process_started") is not True
            or invocation.get("returncode") != 0 or invocation.get("status") != "completed"):
        raise ValueError("Partition reference native command differs or did not succeed")
    seconds = invocation.get("wall_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Partition reference native runtime is missing")
    patched = _expected_overlay(source, candidate)
    if set(patched) != {TP_GNN}:
        raise ValueError("Initial partition reference must change only the trusted loss hook")
    for name, expected in physical["implementation_sha256"].items():
        if name.startswith("AgenticBench/"):
            continue  # Controller code is not part of the physical clone.
        path = _regular(workspace / name)
        differs = path.read_text() != patched[name] if name in patched else file_sha(path) != expected
        if differs:
            raise ValueError("Executed partition source differs from the frozen initial implementation: " + name)
        add(path, "active-native-source")
    stage_pairs = ((STAGE, "partition"),) if latest else ((TIMING, "timing_extract"), (STAGE, "partition"))
    for directory, stage_name in stage_pairs:
        path = run / directory / "stage_manifest.json"
        stage = _json(path)
        if (stage.get("schema_version") != 1 or stage.get("stage") != stage_name or stage.get("status") != "success"
                or stage.get("design") != case or stage.get("integration") != "lol" or stage.get("pdk") != "ng45"
                or stage.get("provenance", {}).get("container_image_id") != physical["image"]):
            raise ValueError("Native partition/timing-import stage manifest identity differs")
        records = stage.get("artifacts", {})
        if not isinstance(records, dict) or not records:
            raise ValueError("Native partition stage manifest has no artifacts")
        required_paths = ({"timing_features.csv", "timing_net_weights.csv", "timing_manifest.json",
                           "openroad_full_partition_last.connections.tsv"} if stage_name == "timing_extract" else
                          {"partition_assignment.csv", "cross_die_nets.csv", "die_regions.json", "partition_manifest.json",
                           "partition_metrics.json", "partition_metrics.csv", "config.json"})
        if not required_paths.issubset({item.get("path") for item in records.values()}):
            raise ValueError("Native partition stage manifest omits required artifacts")
        if latest and not {"timing_input/" + name for name in frozen["timing_input"]}.issubset(
                {item.get("path") for item in records.values()}):
            raise ValueError("Latest partition manifest omits its frozen timing artifacts")
        for item in records.values():
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Partition manifest artifact escapes its stage")
            artifact = _regular(path.parent / relative)
            if artifact.stat().st_size != item.get("bytes") or file_sha(artifact) != item.get("sha256"):
                raise ValueError("Partition native artifact size/hash differs")
            add(artifact, "native-stage-artifact")
        add(path, "native-stage-manifest")
    if _json(run / STAGE / "config.json") != physical["partition_config"]:
        raise ValueError("Native partition changed the fixed training/decoder recipe")
    validate_tp_gnn_partition(run, expected_method="tp-gnn-flat", require_bound_report=True)
    trained = _training_evidence(run / STAGE, physical)
    for name in ("training_config.json", "training_log.csv", "initial_embeddings.npy", "embeddings.npy",
                 "tp_gnn_report.json", "run.log"):
        add(run / STAGE / name, "native-loss-training-evidence")
    raw = independent_partition_metrics(run, source, timing_directory=timing_directory)
    return {"qualified": True, "producer_profile": "native-local-partition-v1", "case": case,
            "endpoint": "partition", "baseline": {"raw_metrics": raw, "source_run": str(run),
                "native_runtime_historical": {"seconds": seconds, "scope": "native partition flow process"}},
            "source_identity": {"purpose_sha256": file_sha(purpose_path), "image": physical["image"],
                "source_revision": physical["source_revision"], "runtime_source_audit_sha256": file_sha(audit_path)},
            "source_contract": physical, "frozen_inputs": frozen, "training_evidence": trained,
            "evidence": [evidence[key] for key in sorted(evidence)], "executed_native_stages": []}


def prepare_local_partition_reference(evaluation_root, task_id, case, *, source_root, benchmark_input):
    """Only FRESH initial evaluations can be published as ADOPT authorities."""
    return _qualify_native_partition(evaluation_root, task_id, case, source_root=source_root,
                                    benchmark_input=benchmark_input)


def qualify_partition_probe(evaluation_root, *, source_root, benchmark_input, reference_record, contract):
    """Requalify one explicit loss-times-1.1 run; never publish it as a reference."""
    from .reference_catalog import qualify_producer
    from .reference_store import load_reference
    before = file_sha(_regular(reference_record))
    record = load_reference(reference_record, contract=contract, validate=qualify_producer)
    if record["producer"].get("kind") != "native-local-partition-v1":
        raise ValueError("Partition probe needs a native same-runtime initial reference")
    value = _qualify_native_partition(evaluation_root, TASK, contract["case"], source_root=source_root,
                                      benchmark_input=benchmark_input, probe_reference=record)
    reference_stage = Path(record["qualification"]["baseline"]["source_run"]) / STAGE
    probe_stage = Path(evaluation_root) / "run" / STAGE
    def first_loss(stage):
        with (stage / "training_log.csv").open(newline="") as stream:
            return float(next(csv.DictReader(stream))["loss"])
    old, new = first_loss(reference_stage), first_loss(probe_stage)
    if math.isclose(old, new, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("Nonidentity loss probe has no measured loss-trajectory change")
    load_reference(reference_record, contract=contract, validate=qualify_producer)
    if file_sha(reference_record) != before:
        raise ValueError("Partition authority changed during probe qualification")
    return {"schema_version": 1, "qualified": True, "kind": "partition-loss-interface-probe-v1",
            "source_commit": contract["physical"]["source_revision"], "image": contract["physical"]["image"],
            "reference_record": str(reference_record), "reference_record_sha256": before,
            "reference_id": record["reference_id"], "case": contract["case"],
            "native_evidence": value, "loss_multiplier": 1.1,
            "measured_first_epoch_loss": {"reference": old, "probe": new},
            "interpretation": "Loss hook exercised with changed measured loss; no assignment or QoR improvement claim.",
            "registry_publications": 0, "model_calls": 0, "candidate_finals": 0}
