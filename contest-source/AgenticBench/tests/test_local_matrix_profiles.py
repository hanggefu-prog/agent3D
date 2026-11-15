"""Declarative default matrix profile: loader, validation and schema-1 rendering."""
import json
from pathlib import Path

import pytest
import yaml

from local_bench import cli
from local_bench.evaluation import UPSTREAM_STAGE_DIRS
from local_bench.matrix import LocalEvaluationMatrix
from local_bench.matrix_profile import (
    DEFAULT_MATRIX_PROFILE,
    cell_config,
    load_matrix_profile,
    render_matrix,
    task_settings,
)
from local_bench.model import load_task
from local_bench.task_inventory import MATRIX_TASK_DIRECTORIES


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "releases/local-matrix-evolution-20261018"
CASES = ["bp_fe-ng45", "ariane133-ng45"]
SEEDS = ["s1", "s2", "s3"]


def baseline_runs(tmp_path, spec):
    runs = {}
    for case in CASES:
        for seed in SEEDS:
            run = tmp_path / "baselines" / case / seed
            for stage in spec.upstream_stages:
                manifest = run / UPSTREAM_STAGE_DIRS[stage] / "stage_manifest.json"
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps({"stage": stage, "case": case, "seed": seed}))
            runs.setdefault(case, {})[seed] = str(run)
    return runs


def write_profile(tmp_path, mutate):
    data = yaml.safe_load(DEFAULT_MATRIX_PROFILE.read_text())
    mutate(data)
    path = tmp_path / "profile.yaml"
    # Keep relative base_config entries resolvable from the copy.
    for entry in data["tasks"].values():
        if isinstance(entry, dict) and isinstance(entry.get("base_config"), str):
            entry["base_config"] = str((DEFAULT_MATRIX_PROFILE.parent / entry["base_config"]).resolve())
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_default_profile_matches_the_20261018_release_conventions():
    profile = load_matrix_profile()
    assert profile["qualified"] is False
    assert [c["id"] for c in profile["cases"]] == CASES
    assert all(c["design"] == c["id"][:-5] for c in profile["cases"])
    assert profile["seeds"] == [
        {"id": "s" + str(n), "io": 1, "partition": 1000 + n, "placement": 1000 + n, "routing": -1}
        for n in (1, 2, 3)
    ]
    assert set(profile["tasks"]) == set(MATRIX_TASK_DIRECTORIES)
    for name in MATRIX_TASK_DIRECTORIES:
        expected = 2 if name.startswith("global_routing_") else 6
        assert task_settings(name, profile) == {"workers": expected, "timeout_seconds": 3600}
    nesterov = profile["tasks"]["placement_parameters_reference_20260923"]["base_config"]
    assert nesterov == ROOT / "local_bench/reference_configs/nesterov_current.yaml"
    # The release scripts that define these conventions are still present.
    for script in ("audit_runner.py", "routing_audit_runner.py"):
        text = (RELEASE / script).read_text()
        assert "seeds=[dict(id='s'+str(n),io=1,partition=1000+n,placement=1000+n,routing=-1)" in text
        assert "cases=['bp_fe-ng45','ariane133-ng45']" in text
    assert ("config['routing']['resource_reduction']=0.5 if config['design']=='bp_fe' else 0.0"
            in (RELEASE / "routing_audit_runner.py").read_text())


@pytest.mark.parametrize("task", MATRIX_TASK_DIRECTORIES)
def test_rendered_matrix_loads_in_the_existing_matrix_schema(tmp_path, task):
    spec = load_task(ROOT / "local_tasks" / task)
    runs = baseline_runs(tmp_path, spec)
    matrix_path = render_matrix(task, runs, tmp_path / "matrix")
    data = json.loads(matrix_path.read_text())
    assert set(data) == {"schema_version", "task", "cases", "seeds", "cells"}
    assert all(cell["reference_run"] is None for cell in data["cells"])
    matrix = LocalEvaluationMatrix(matrix_path, spec, references_required=False)
    assert [(c["case"], c["seed"]) for c in matrix.cells] == [(c, s) for c in CASES for s in SEEDS]
    # An unqualified rendering cannot score candidates.
    with pytest.raises(ValueError, match="qualified reference"):
        LocalEvaluationMatrix(matrix_path, spec)
    for cell in matrix.cells:
        config = yaml.safe_load(Path(cell["base_config"]).read_text())
        assert config["design"] == cell["case"][:-5]
        seed = cell["seeds"]
        assert [config[s]["seed"] for s in ("pins", "partition", "placement", "routing")] == [
            seed["io"], seed["partition"], seed["placement"], seed["routing"]]
        if task.startswith("global_routing_"):
            assert config["threads"] == config["routing"]["threads"] == 16
            assert config["routing"]["resource_reduction"] == (0.5 if config["design"] == "bp_fe" else 0.0)


def test_cell_config_keeps_task_base_config_otherwise_unchanged():
    profile = load_matrix_profile()
    spec = load_task(ROOT / "local_tasks/placement_initialization")
    base = yaml.safe_load(Path(spec.base_config).read_text())
    config = cell_config(profile, "placement_initialization", "ariane133-ng45", "s2")
    for key in ("pins", "partition", "placement", "routing"):
        base[key]["seed"] = config[key]["seed"]
    base["design"] = "ariane133"
    assert config == base
    nesterov = cell_config(profile, "placement_parameters_reference_20260923", "bp_fe-ng45", "s1")
    current = yaml.safe_load((ROOT / "local_bench/reference_configs/nesterov_current.yaml").read_text())
    for key, value in (("pins", 1), ("partition", 1001), ("placement", 1001), ("routing", -1)):
        current[key]["seed"] = value
    assert nesterov == current

@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(qualified=True), "qualified: false"),
    (lambda d: d.update(extra=1), "exactly the keys"),
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d["tasks"].pop("hbt_assignment_policy"), "maintained task directories"),
    (lambda d: d["tasks"].update(netlist_partition={"workers": 1}), "maintained task directories"),
    (lambda d: d["seeds"][0].update(routing=0), "routing=-1"),
    (lambda d: d["seeds"].append(dict(d["seeds"][0])), "duplicate seed"),
    (lambda d: d["cases"].append(dict(d["cases"][0])), "duplicate case"),
    (lambda d: d["tasks"]["hbt_assignment_policy"].update(workers=0), "positive integer"),
    (lambda d: d["tasks"]["hbt_assignment_policy"].update(config_overrides={"placement.seed": 7}),
     "may not override"),
    (lambda d: d["tasks"]["hbt_assignment_policy"].update(config_overrides={"design": "ibex"}),
     "may not override"),
    (lambda d: d["tasks"]["global_routing_maze_cost"].update(
        case_config_overrides={"unknown-case": {"threads": 1}}), "keyed by case id"),
])
def test_invalid_profiles_are_rejected(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        load_matrix_profile(write_profile(tmp_path, mutate))


def test_render_requires_every_baseline_run(tmp_path):
    spec = load_task(ROOT / "local_tasks/hbt_assignment_policy")
    runs = baseline_runs(tmp_path, spec)
    del runs["ariane133-ng45"]["s3"]
    with pytest.raises(ValueError, match="missing baseline run for cell ariane133-ng45/s3"):
        render_matrix("hbt_assignment_policy", runs, tmp_path / "matrix")


def test_cli_render_matrix(tmp_path, capsys):
    spec = load_task(ROOT / "local_tasks/global_routing_maze_cost")
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps(baseline_runs(tmp_path, spec)))
    assert cli.main(["render-matrix", "global_routing_maze_cost_source",
                     "--baseline-runs", str(runs), "--output-dir", str(tmp_path / "out")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["qualified"] is False
    assert report["settings"] == {"workers": 2, "timeout_seconds": 3600}
    LocalEvaluationMatrix(report["matrix"], spec, references_required=False)


def test_matrix_and_task_docs_match_code():
    """Gate hook: the maintained gate collects test_local_matrix*.py, not the
    exact-named docs test, so the repository-level doc check also runs here."""
    import importlib.util

    script = ROOT / "scripts" / "check_doc_consistency.py"
    spec = importlib.util.spec_from_file_location("check_doc_consistency_gate", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check(ROOT.parent) == []


@pytest.mark.parametrize('name', ['tp_gnn_binary_loss_timing', 'tp_gnn_loss_decoder_timing'])
def test_new_single_seed_gnn_tasks_cannot_render_an_implicit_matrix(tmp_path, name):
    profile = load_matrix_profile()
    assert len(profile['tasks']) == 8
    assert name not in profile['tasks']
    output = tmp_path / 'forbidden-matrix'
    with pytest.raises(ValueError, match='no defined seed-matrix contract'):
        render_matrix(name, {}, output, profile=profile)
    assert not output.exists()
