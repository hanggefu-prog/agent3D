"""Offline native-evidence rejection and score tests; no EDA or model calls."""
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from local_bench import gnn_evolution_timing as timing
from local_tasks.tp_gnn_local_task_definitions import PACKAGES

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "AgenticBench/local_tasks"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_result_reader_finds_separate_physical_source_mount(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    source = tmp_path / "workspace"
    recipe = source / "Place-3D/config/partition/tp_gnn_fast.json"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("{}")
    monkeypatch.setattr(timing, "__file__", str(trusted / "local_bench/gnn_evolution_timing.py"))
    assert timing._source_root(mounted_source=source) == source


def record(path):
    return {"path": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def native_fixture(run, task):
    write(run / "config.yaml", {"partition": {"seed": 1001}})
    stage = run / "20_partition"
    filename, interface, _ = PACKAGES[task]
    capsule = json.loads((TASKS / task / "initial_candidate/candidate.json").read_text())[
        "candidate_files"][filename]
    path = stage / "evolution_program.json"
    write(path, capsule)
    manifest = {"artifacts": {"evolution_program": record(path)}}
    recipe = json.loads((ROOT / "Place-3D" / timing.BASELINE_RECIPES[task]).read_text())["partition"]
    config = {(key[7:] if key.startswith("tp_gnn_") else key): value
              for key, value in recipe.items() if key != "method"}
    config.update(decoder=timing.TASK_DECODERS[task], evolution_program=str(path))
    proof = {"interface_version": interface, "program_sha256": record(path)["sha256"],
             "capsule_sha256": record(path)["sha256"],
             "source_sha256": hashlib.sha256(capsule["source"].encode()).hexdigest(),
             "source_bytes": len(capsule["source"].encode())}
    training = {"status": "trained", "device": "cuda:0", "evolution": proof}
    report = {"config": config, "training": training, "lol_handoff_contract": {
        "constraints": {"accepted": True}, "supernode_integrity": {"preserved": True}}}
    if task == timing.BINARY_TASK:
        training.update(optimizer_steps_completed=400, effective_epochs=400,
                        effective_min_epochs=400, epochs_completed=400, selected_epoch=400,
                        loss_context_interface="binary-loss-v1", final_loss=1 / 401,
                        regularization={"noise_used_in_scoring_or_decoding": False,
                                        "selection_start_epoch": 300})
        proof.update(loss_context_interface="binary-loss-v1", loss_calls=802,
                     training_loss_calls=400, scoring_loss_calls=402, score_epoch=400,
                     canonical_diagnostics_protected=True)
        rows = [{"epoch": epoch, "loss": 1 / (epoch + 1), "canonical_loss": .5,
                 "expected_cut": .1, "area_balance_loss": .01, "binarization_loss": .2,
                 "temperature": 1. + (.1 - 1.), "binarization_weight": 10., "training_logit_noise_scale": 0.}
                for epoch in range(401)]
        training["history"] = rows
        training["selected_metrics"] = rows[-1]
        path = stage / "binary_training.csv"
        manifest["artifacts"]["binary_training"] = None
    else:
        config["learning_rate"] = .001
        training.update(optimizer_steps_completed=100, epochs_completed=100, cuda_available=True,
                        optimizer_steps_per_epoch=1, max_epochs=100, final_loss=.25,
                        loss_context_interface="timing-context-v2")
        proof.update(loss_context_interface="timing-context-v2", loss_calls=100)
        report["evolution_decoder"] = dict(proof, decoder_calls=1, labels_validated=True,
                                            protected_closure_affinities=True, iterations_limit=50)
        rows = [{"epoch": epoch, "loss": .25} for epoch in range(1, 101)]
        path = stage / "training_log.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if task == timing.BINARY_TASK:
        manifest["artifacts"]["binary_training"] = record(path)
    write(stage / "stage_manifest.json", manifest)
    write(stage / "tp_gnn_report.json", report)
    return report, manifest


@pytest.mark.parametrize("task", list(timing.TASK_DECODERS))
def test_recipe_is_bound_to_task_and_frozen_json(tmp_path, task):
    source = tmp_path / "source"
    relative = Path("Place-3D") / timing.BASELINE_RECIPES[task]
    (source / relative).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / relative, source / relative)
    config = {"partition": {"algorithm": "tp-gnn-flat", "config": timing.BASELINE_RECIPES[task],
                             "evolution_program": "candidate.json", "seed": 1001},
              "placement": {"seed": 1001}}
    timing.validate_recipe(config, source, task_id=task)
    bad = copy.deepcopy(config)
    bad["partition"]["config"] = "wrong.json"
    (source / "Place-3D/wrong.json").write_bytes((source / relative).read_bytes())
    with pytest.raises(ValueError, match="trusted matrix adapter"):
        timing.validate_recipe(bad, source, task_id=task)
    mutated = json.loads((source / relative).read_text())
    mutated["partition"]["tp_gnn_epochs"] += 1
    write(source / relative, mutated)
    with pytest.raises(ValueError, match="recipe changed"):
        timing.validate_recipe(config, source, task_id=task)


@pytest.mark.parametrize("task", list(timing.TASK_DECODERS))
def test_valid_native_hook_receipt_and_exact_capsule_are_accepted(tmp_path, task):
    native_fixture(tmp_path, task)
    receipt = timing.validate_native_evolution(tmp_path, task_id=task)
    assert receipt["task_id"] == task
    assert receipt["optimizer_steps_completed"] == (400 if task == timing.BINARY_TASK else 100)


@pytest.mark.parametrize("field,value,match", [
    ("training.device", "cpu", "CUDA"),
    ("config.learning_rate", .02, "recipe changed"),
    ("training.optimizer_steps_completed", 399, "400 updates"),
    ("training.evolution.loss_calls", 801, "hook-call"),
    ("training.evolution.training_loss_calls", 399, "hook-call"),
    ("training.evolution.scoring_loss_calls", 401, "hook-call"),
    ("training.evolution.score_epoch", 399, "hook-call"),
    ("training.evolution.canonical_diagnostics_protected", False, "not protected"),
    ("training.evolution.program_sha256", "0" * 64, "hash differs"),
    ("training.evolution.interface_version", "old", "interface differs"),
    ("training.regularization.noise_used_in_scoring_or_decoding", True, "clean final"),
    ("training.selected_epoch", 299, "selected binary epoch"),
    ("lol_handoff_contract.supernode_integrity.preserved", False, "closure"),
])
def test_binary_rejects_forged_or_changed_native_evidence(tmp_path, field, value, match):
    report, _ = native_fixture(tmp_path, timing.BINARY_TASK)
    node = report
    parts = field.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    write(tmp_path / "20_partition/tp_gnn_report.json", report)
    with pytest.raises(ValueError, match=match):
        timing.validate_native_evolution(tmp_path, task_id=timing.BINARY_TASK)


@pytest.mark.parametrize("field,value,match", [
    ("training.evolution.loss_calls", 99, "once per optimizer"),
    ("training.optimizer_steps_completed", 99, "once per optimizer"),
    ("evolution_decoder.decoder_calls", 2, "decoder call"),
    ("evolution_decoder.labels_validated", False, "decoder call"),
    ("evolution_decoder.protected_closure_affinities", False, "decoder call"),
    ("evolution_decoder.iterations_limit", 51, "decoder call"),
])
def test_legacy_rejects_wrong_calls_or_decoder_protection(tmp_path, field, value, match):
    report, _ = native_fixture(tmp_path, timing.LEGACY_TASK)
    node = report
    parts = field.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    write(tmp_path / "20_partition/tp_gnn_report.json", report)
    with pytest.raises(ValueError, match=match):
        timing.validate_native_evolution(tmp_path, task_id=timing.LEGACY_TASK)


def test_capsule_must_be_hash_published_and_pass_current_grammar(tmp_path):
    _, manifest = native_fixture(tmp_path, timing.BINARY_TASK)
    path = tmp_path / "20_partition/evolution_program.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="size changed|hash changed"):
        timing.validate_native_evolution(tmp_path, task_id=timing.BINARY_TASK)
    capsule = json.loads(path.read_text())
    capsule["source"] = capsule["source"].replace("node_areas.sum()", "node_areas.detach().sum()")
    write(path, capsule)
    manifest["artifacts"]["evolution_program"] = record(path)
    write(path.parent / "stage_manifest.json", manifest)
    with pytest.raises(ValueError, match="forbidden"):
        timing.validate_native_evolution(tmp_path, task_id=timing.BINARY_TASK)


def mocked_measurements(tmp_path, monkeypatch, task, candidate, reference):
    run, ref = tmp_path / "candidate", tmp_path / "reference"
    for root in (run, ref):
        directory = root / "20_partition/timing_input"
        directory.mkdir(parents=True)
        for name in timing.TIMING_FILES:
            (directory / name).write_text("frozen-identical-input")
    calls = []
    def snapshot(path, task_id):
        assert task_id == task
        wns, tns = candidate if path == run else reference
        return {"wns": wns, "tns": tns,
                "partition_config": {"decoder": timing.TASK_DECODERS[task]}}
    def independent(path):
        calls.append(path)
        return {"combined_score": 1., "partition_constraints_passed": 1., "timing_weighted_cut_score": 99.}
    monkeypatch.setattr(timing, "_opt_snapshot", snapshot)
    monkeypatch.setattr(timing, "_independent", independent)
    monkeypatch.setattr(timing, "_binary_diagnostics", lambda *args: None)
    return run, ref, calls


@pytest.mark.parametrize("candidate,reference,expected", [
    ((-1., -10.), (-1., -10.), 0.),
    ((-.5, -5.), (-1., -10.), 50.),
    ((-2., -5.), (-1., -10.), -10.),
    ((0., 0.), (-1., -10.), 100.),
])
@pytest.mark.parametrize("task", list(timing.TASK_DECODERS))
def test_fitness_preserves_timing_tradeoff_and_independent_partition_checks(
        tmp_path, monkeypatch, task, candidate, reference, expected):
    run, ref, calls = mocked_measurements(tmp_path, monkeypatch, task, candidate, reference)
    result = timing.timing_metrics(run, ref, task_id=task)
    assert result["combined_score"] == pytest.approx(expected)
    assert result["sta_wns_weight"] == .4 and result["sta_tns_weight"] == .6
    assert result["opt_passes"] == 1
    assert "timing_weighted_cut_score" not in result
    assert calls == [run, ref]


@pytest.mark.parametrize("reference", [(0., -1.), (-1., 0.), (0., 0.)])
def test_zero_reference_deficit_does_not_get_a_synthetic_score(tmp_path, monkeypatch, reference):
    run, ref, _ = mocked_measurements(tmp_path, monkeypatch, timing.BINARY_TASK, (0., 0.), reference)
    with pytest.raises(ValueError, match="zero-deficit"):
        timing.timing_metrics(run, ref, task_id=timing.BINARY_TASK)


def test_frozen_timing_mutation_cannot_be_compensated_by_better_timing(tmp_path, monkeypatch):
    run, ref, _ = mocked_measurements(tmp_path, monkeypatch, timing.LEGACY_TASK, (0., 0.), (-1., -10.))
    (run / "20_partition/timing_input/timing_net_weights.csv").write_text("changed")
    with pytest.raises(ValueError, match="frozen timing differs"):
        timing.timing_metrics(run, ref, task_id=timing.LEGACY_TASK)


def test_binary_diagnostics_preserve_measured_zero_and_reject_missing_values(tmp_path):
    report = {"training": {"probability_uncertain_fraction": 0., "probability_uncertain_area_fraction": .1},
              "assignment_stages": {"raw_binary_threshold": {"top_area_ratio": .7},
                                    "lol_feasible_supernodes": {"top_area_ratio": .5}},
              "lol_handoff_contract": {"supernode_integrity": {"moved_area_fraction": .2}}}
    write(tmp_path / "20_partition/tp_gnn_report.json", report)
    result = {}
    timing._binary_diagnostics(tmp_path, result)
    assert result == {"binary_probability_uncertain_fraction": 0.,
                      "binary_probability_uncertain_area_fraction": .1,
                      "binary_raw_top_area_ratio": .7, "binary_final_top_area_ratio": .5,
                      "binary_closure_moved_area_fraction": .2}
    del report["training"]["probability_uncertain_fraction"]
    write(tmp_path / "20_partition/tp_gnn_report.json", report)
    with pytest.raises(ValueError, match="Missing measured"):
        timing._binary_diagnostics(tmp_path, {})


def test_independent_scoring_recomputes_real_frozen_net_assignment(tmp_path):
    from local_bench.reference_local_partition import _trusted_core
    core = _trusted_core(ROOT)
    floorplan = tmp_path / "10_floorplan"
    stage = tmp_path / "20_partition"
    timing_input = stage / "timing_input"
    floorplan.mkdir()
    timing_input.mkdir(parents=True)
    (floorplan / "partition.input").write_text(
        "LibCell N CELL 1 1 1\nPin A 0 0\n"
        "Inst C1 CELL\nInst C2 CELL\nNet N1 2\nPin C1/A\nPin C2/A\n")
    (floorplan / "partition.names.csv").write_text("internal_name,inst_name\nC1,cell_a\nC2,cell_b\n")
    (timing_input / "timing_net_weights.csv").write_text("net_index,weight,paper_weight_normalized\n0,2,0.5\n")
    (stage / "partition_assignment.csv").write_text("inst_name,die\ncell_a,top\ncell_b,bottom\n")
    _, instances, nets = core.parse_lol_input(floorplan / "partition.input")
    actual = core.compute_metrics(instances, nets, {"C1": "top", "C2": "bottom"}, [2.], [.5])
    with (stage / "partition_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(actual))
        writer.writeheader()
        writer.writerow(actual)
    raw = dict(actual, method_metrics=actual, timing_weight_mode="paper_delay_slack_hyperedge_v1")
    for key in ("top_area", "bottom_area", "top_macro_area", "bottom_macro_area",
                "top_standard_cell_area", "bottom_standard_cell_area"):
        raw[key + "_native"] = actual[key]
    write(stage / "partition_metrics.json", raw)
    write(stage / "partition_manifest.json", {"metrics": actual})
    write(stage / "tp_gnn_report.json", {
        "method": "tp-gnn-flat", "hierarchy": {"mode": "flat", "consumed_as_feature": False},
        "config": {"decoder": "binary-probability"}, "training": {"status": "trained"}})
    result = timing._independent(tmp_path)
    assert result["partition_constraints_passed"] == 1
    assert result["cut_net_count"] == 1 and result["timing_weighted_cut"] == 2
    raw["cut_net_count"] = 0
    write(stage / "partition_metrics.json", raw)
    with pytest.raises(ValueError, match="independent assignment"):
        timing._independent(tmp_path)
