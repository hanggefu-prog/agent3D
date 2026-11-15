import hashlib
import json
from pathlib import Path

import pytest

from local_bench.reference_placement_evidence import (
    DYNAMIC_TASK, SOURCE_TASK, STAGE, parse_nesterov_step_controls,
    validate_reference_placement_evidence, validate_reference_placement_params,
)


def native_params():
    return {
        "random_seed": 1000, "num_threads": 4, "num_bins_x": 512, "num_bins_y": 512,
        "target_density": 0.6, "macro_halo_x": 10, "macro_halo_y": 10,
        "stop_overflow": 0.01, "density_weight": 0.00003, "macro_place_flag": 1,
        "enable_fillers": 1, "random_center_init_flag": 0,
        "global_place_stages": [{"optimizer": "nesterov", "iteration": 4000,
                                 "learning_rate": 0.01, "num_bins_x": 512,
                                 "num_bins_y": 512, "wirelength": "weighted_average"}],
        "concurrent_lol": {"enabled": True, "initialization_mode": "die-center",
                           "initialization_seed": 1000, "cross_die_net_weight": 1,
                           "macro_halo_x": 10, "macro_halo_y": 10,
                           "macro_halo_units": "micron", "macro_snap_guard_um": 1.001},
    }


def row(step=0, call=1, native_step="0.2", shared="1", bottom="1", top="1"):
    return "[INFO] NesterovStepControl step={} call={} native_step={} shared={} bottom={} top={}\n".format(step, call, native_step, shared, bottom, top)


HEADER = "[INFO] use nesterov optimizer\n"


def make_run(tmp_path, *, dynamic=False, log=None):
    stage = tmp_path / STAGE
    stage.mkdir(parents=True)
    params = native_params()
    (stage / "params.json").write_text(json.dumps(params))
    (stage / "run.log").write_text(log if log is not None else HEADER + row())
    return tmp_path


@pytest.mark.parametrize("flag", [None, 0, False])
def test_native_determinism_default_zero_is_preserved(flag):
    params = native_params()
    if flag is not None:
        params["deterministic_flag"] = flag
    receipt = validate_reference_placement_params(params, dynamic=False)
    assert receipt["deterministic_flag"] == 0
    assert receipt["deterministic_flag_explicit"] == (flag is not None)


def test_current_nesterov_profile_requires_determinism_without_widening_dated_profile(tmp_path):
    run = make_run(tmp_path, dynamic=True)
    path = run / STAGE / "params.json"
    with pytest.raises(ValueError, match="deterministic_flag"):
        validate_reference_placement_evidence(run, DYNAMIC_TASK, profile="current-source-nesterov-v1")
    params = json.loads(path.read_text())
    params["deterministic_flag"] = 1
    path.write_text(json.dumps(params))
    result = validate_reference_placement_evidence(run, DYNAMIC_TASK, profile="current-source-nesterov-v1")
    assert result["baseline_id"] == "current-source-nesterov-v1"
    with pytest.raises(ValueError, match="deterministic_flag"):
        validate_reference_placement_evidence(run, DYNAMIC_TASK)
    with pytest.raises(ValueError, match="Unsupported placement evidence profile"):
        validate_reference_placement_evidence(run, SOURCE_TASK, profile="current-source-nesterov-v1")


@pytest.mark.parametrize("key,value", [
    ("random_seed", 1001), ("num_threads", 32), ("num_bins_x", 256), ("num_bins_y", 256),
    ("target_density", 0.7), ("macro_place_flag", 0), ("macro_halo_x", 0),
    ("macro_halo_y", 0), ("deterministic_flag", 1), ("deterministic_flag", None),
    ("enable_fillers", 0), ("random_center_init_flag", 1), ("density_weight", float("nan")),
])
def test_changed_recipe_rejected(key, value):
    params = native_params()
    params[key] = value
    with pytest.raises(ValueError, match=key):
        validate_reference_placement_params(params, dynamic=False)


@pytest.mark.parametrize("key,value", [
    ("optimizer", "adam"), ("iteration", 2000), ("learning_rate", 0.02),
    ("num_bins_x", 256), ("num_bins_y", 256), ("wirelength", "logsumexp"),
])
def test_changed_global_stage_rejected(key, value):
    params = native_params()
    params["global_place_stages"][0][key] = value
    with pytest.raises(ValueError):
        validate_reference_placement_params(params, dynamic=False)


def test_exactly_one_configured_stage_and_reference_snap_guard_required():
    params = native_params()
    params["global_place_stages"] *= 2
    with pytest.raises(ValueError, match="exactly one"):
        validate_reference_placement_params(params, dynamic=False)
    params = native_params()
    params["concurrent_lol"]["macro_snap_guard_um"] = 0
    with pytest.raises(ValueError, match="macro_snap_guard"):
        validate_reference_placement_params(params, dynamic=False)


@pytest.mark.parametrize("use_bb", [1, True, "1", 2])
def test_dynamic_bb_or_unknown_bb_rejected(use_bb):
    params = native_params()
    if use_bb is not None:
        params["use_bb"] = use_bb
    with pytest.raises(ValueError, match="use_bb"):
        validate_reference_placement_params(params, dynamic=True)


@pytest.mark.parametrize("use_bb", [None, 0, False])
def test_dynamic_native_no_bb_preserved(use_bb):
    params = native_params()
    if use_bb is not None:
        params["use_bb"] = use_bb
    assert validate_reference_placement_params(params, dynamic=True)["use_bb"] == 0


def test_active_control_sequence_allows_native_macro_stage_reset():
    text = HEADER + row() + row(1, 2, shared="2", bottom="1.25", top="0.8") + HEADER + row(2, 1)
    result = parse_nesterov_step_controls(text)
    assert result["call_count"] == 3
    assert [segment["calls"] for segment in result["segments"]] == [2, 1]
    assert not result["identity_controls_only"]  # Noninitial factors are valid.
    assert result["controls"][1]["shared"] == 2


@pytest.mark.parametrize("text", [
    "native output without controller\n", HEADER, row(),
    HEADER + row(call=2), HEADER + row(step=1), HEADER + row() + row(step=1, call=1),
    HEADER + row() + row(step=2, call=2), HEADER + row() + row(step=0, call=2),
    HEADER + row() + HEADER, HEADER + row() + HEADER + row(1) + HEADER + row(2),
    HEADER + "NesterovStepControl step=0 call=1 native_step=0.2\n",
    HEADER + row(native_step="nan"), HEADER + row(shared="inf"), HEADER + row(bottom="0"),
    HEADER + row(top="-1"), HEADER + row(native_step="not-a-number"),
])
def test_zero_malformed_nonfinite_or_bad_sequence_rejected(text):
    with pytest.raises(ValueError):
        parse_nesterov_step_controls(text)


def test_receipt_retains_hashes_and_controls_without_modifying_files(tmp_path):
    run = make_run(tmp_path, dynamic=True, log=HEADER + row(shared="0.5"))
    before = {path: path.read_bytes() for path in (run / STAGE).iterdir()}
    receipt = validate_reference_placement_evidence(run, DYNAMIC_TASK)
    assert receipt["status"] == "passed" and receipt["dynamic_controls"]["call_count"] == 1
    for name in ("params", "log"):
        evidence = receipt["evidence"][name]
        assert evidence["sha256"] == hashlib.sha256(Path(evidence["path"]).read_bytes()).hexdigest()
    assert before == {path: path.read_bytes() for path in (run / STAGE).iterdir()}


def test_source_task_requires_recipe_but_not_a_dynamic_controller(tmp_path):
    run = make_run(tmp_path, log="[INFO] native placement complete\n")
    receipt = validate_reference_placement_evidence(run, SOURCE_TASK)
    assert receipt["dynamic_controls"]["call_count"] == 0
    assert receipt["dynamic_controls"]["required"] is False


def test_optional_overlay_files_are_hashed_as_inventory_only(tmp_path):
    run = make_run(tmp_path / "run", dynamic=True)
    overlay = tmp_path / "overlay" / "dreamplace"
    overlay.mkdir(parents=True)
    for name in ("NonLinearPlace.py", "NesterovAcceleratedGradientOptimizer.py"):
        (overlay / name).write_text("# candidate runtime overlay\n")
    receipt = validate_reference_placement_evidence(run, DYNAMIC_TASK, runtime_overlay=overlay.parent)
    assert len(receipt["evidence"]["runtime_overlay_files"]) == 2
    assert "not independent proof" in receipt["interpretation"]


def test_empty_missing_and_symlink_evidence_fail(tmp_path):
    run = make_run(tmp_path)
    log = run / STAGE / "run.log"
    log.write_text("")
    with pytest.raises(ValueError, match="nonempty regular"):
        validate_reference_placement_evidence(run, SOURCE_TASK)
    log.unlink()
    target = tmp_path / "outside.log"
    target.write_text("some log\n")
    log.symlink_to(target)
    with pytest.raises(ValueError, match="nonempty regular"):
        validate_reference_placement_evidence(run, SOURCE_TASK)


def test_unsupported_task_is_not_silently_treated_as_source(tmp_path):
    with pytest.raises(ValueError, match="unsupported task"):
        validate_reference_placement_evidence(tmp_path, "placement_dynamic_parameters")
