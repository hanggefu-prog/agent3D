"""Exercise the real bounded Python planner and independent geometry verifier."""
import csv
import json
from pathlib import Path
import sys

import pytest

from local_bench.hbt_native import validate_assignment
from local_bench.hbt_policy import materialize_policy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Place-3D"))
from src.hbt.planner import load_problem, write_result

INITIAL = (ROOT / "AgenticBench/local_tasks/hbt_assignment_policy/initial_program.py").read_text()


def make_assignment(root, program=INITIAL):
    root.mkdir()
    problem = {"schema_version": 1, "units": "dbu", "dbu_per_micron": 1000,
               "site_grid": {"die": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
                             "origin_x": 0, "origin_y": 0, "pitch_x": 10, "pitch_y": 10,
                             "hbt_width": 2, "hbt_height": 2}, "blockages": [],
               "nets": [{"net_name": "n0", "bottom_pins": [[1, 1], [21, 21]], "top_pins": [[5, 5]]},
                        {"net_name": "n1", "bottom_pins": [[11, 11]], "top_pins": [[15, 15]]},
                        {"net_name": "n2", "bottom_pins": [[41, 41]], "top_pins": [[51, 51]]}]}
    path = root / "problem.json"
    path.write_text(json.dumps(problem))
    policy = materialize_policy(program)["candidate_files"]["hbt_policy.json"]
    (root / "policy.json").write_text(json.dumps(policy))
    native, context = load_problem(path)
    write_result(root / "assignment", path, "track-aware-greedy", native, context,
                 candidate_limit=None, max_cost_entries=1000, policy_path=root / "policy.json")
    rows = list(csv.DictReader((root / "assignment/hbt.csv").open()))
    with (root / "hbt.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["net_name", "x_dbu", "y_dbu"])
        writer.writeheader()
        writer.writerows({"net_name": r["net_name"], "x_dbu": r["legal_x"], "y_dbu": r["legal_y"]} for r in rows)
    return path, native, context


def test_initial_policy_is_exactly_native_and_changed_policy_executes(tmp_path):
    path, problem, context = make_assignment(tmp_path / "initial")
    write_result(tmp_path / "native", path, "track-aware-greedy", problem, context,
                 candidate_limit=None, max_cost_entries=1000)
    assert (tmp_path / "native/hbt.csv").read_bytes() == (tmp_path / "initial/assignment/hbt.csv").read_bytes()
    assert validate_assignment(tmp_path / "initial")["assigned_count"] == 3
    changed = INITIAL.replace('return context["pin_count"]', 'return -context["pin_count"]')
    assert changed != INITIAL
    make_assignment(tmp_path / "changed", changed)
    validate_assignment(tmp_path / "changed")
    assert (tmp_path / "changed/assignment/hbt.csv").read_bytes() != (tmp_path / "initial/assignment/hbt.csv").read_bytes()


@pytest.mark.parametrize("fault", ["count", "hpwl", "hook", "policy", "physical", "missing", "overlap"])
def test_corrupt_evidence_is_rejected(tmp_path, fault):
    root = tmp_path / "run"
    make_assignment(root)
    metric_path = root / "assignment/metrics.json"
    metrics = json.loads(metric_path.read_text())
    if fault == "count":
        metrics["assigned_count"] = 4
    elif fault == "hpwl":
        metrics["legal_combined_hpwl_dbu"] += 1
    elif fault == "hook":
        metrics["policy"]["calls"]["net_priority"] = 0
    elif fault == "policy":
        (root / "policy.json").write_text('{"source": "invalid"}')
    elif fault == "physical":
        (root / "hbt.csv").write_text("net_name,x_dbu,y_dbu\nn0,0,0\n")
    else:
        path = root / "assignment/hbt.csv"
        rows = list(csv.DictReader(path.open()))
        if fault == "missing":
            rows.pop()
        else:
            rows[1]["legal_x"], rows[1]["legal_y"] = rows[0]["legal_x"], rows[0]["legal_y"]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    metric_path.write_text(json.dumps(metrics))
    with pytest.raises((ValueError, KeyError, SyntaxError)):
        validate_assignment(root)


def test_policy_exception_never_falls_back(tmp_path):
    bad = INITIAL.replace('return context["pin_count"]', 'return 1 / 0')
    with pytest.raises(ValueError, match="failed"):
        make_assignment(tmp_path / "bad", bad)
    assert not (tmp_path / "bad/assignment/manifest.json").exists()


@pytest.mark.parametrize("fault", [None, "duplicate", "missing", "nan"])
def test_native_sta_text_diagnostics_and_numeric_contract(tmp_path, fault):
    from local_bench.hbt_native import native_timing_values
    text = "analysis_mode\telmore_cutpoint\nwns_setup\t-1.2\ntns_setup\t-30\nclock_count\t1\n"
    if fault == "duplicate":
        text += "wns_setup\t-2\n"
    elif fault == "missing":
        text = text.replace("wns_setup\t-1.2\n", "")
    elif fault == "nan":
        text = text.replace("-1.2", "nan")
    path = tmp_path / "metrics.tsv"
    path.write_text(text)
    if fault is None:
        assert native_timing_values(path) == {"wns_setup": -1.2, "tns_setup": -30., "clock_count": 1.}
    else:
        with pytest.raises(ValueError):
            native_timing_values(path)


@pytest.mark.parametrize("change", [None, "design", "type", "pdk"])
def test_frozen_prefix_must_match_evaluation_case(tmp_path, monkeypatch, change):
    from local_bench import hbt_native as native
    monkeypatch.setattr(native, "prefix_identity", lambda root: {"pdn": "frozen"})
    stage = tmp_path / "40_pdn"
    stage.mkdir()
    (stage / "stage_manifest.json").write_text(json.dumps(
        {"design": "bp_fe", "integration": "lol", "pdk": "ng45"}))
    config = {"design": "bp_fe", "type": "lol", "pdk": "ng45"}
    if change is None:
        assert native.validate_input_identity(tmp_path, config) == {"pdn": "frozen"}
    else:
        config[change] = "wrong"
        with pytest.raises(ValueError, match="frozen design/platform"):
            native.validate_input_identity(tmp_path, config)


def test_renamed_hbt_task_cannot_bypass_reference_admission(tmp_path, monkeypatch):
    import shutil
    import yaml
    from local_bench import scoped_reference
    from local_bench.evaluation import evaluate_task
    from local_bench.model import load_task
    task = tmp_path / "renamed"
    shutil.copytree(ROOT / "AgenticBench/local_tasks/hbt_assignment_policy", task)
    raw = yaml.safe_load((task / "task.yaml").read_text())
    raw["id"] = "hbt-round-1"
    raw["base_config"] = str(ROOT / "AgenticBench/local_tasks/tp_gnn_post_placement_timing/reference_config.yaml")
    (task / "task.yaml").write_text(yaml.safe_dump(raw))
    def reject(*args):
        raise ValueError("reference-admission-was-reached")
    monkeypatch.setattr(scoped_reference, "admit_reference", reject)
    result = evaluate_task(task, load_task(task).initial_candidate, ROOT, tmp_path / "result")
    assert result.artifacts["error_message"] == "reference-admission-was-reached"
    assert not result.artifacts["native_flow_execution"]
