"""Retired code stays absent while shared evidence and current tasks stay usable."""
from pathlib import Path

import pytest
import yaml

from local_bench.benchmark_input import load_benchmark_input_profile
from local_bench.metrics import EXTRACTORS
from local_bench.model import load_candidate, load_task
from local_bench.partition_assignment import read_assignment
from local_bench.task_inventory import CORE_TASK_DIRECTORIES

ROOT = Path(__file__).resolve().parents[1]
RETIRED_PROFILES = (
    "bp_fe_lol_ng45_mtkahypar_frontend", "bp_fe_lol_ng45_mtkahypar_parameter_cts",
    "bp_fe_lol_ng45_tp_gnn", "bp_lol_ng45_tp_gnn_ta3d_portable",
    *(case + "_lol_ng45_tp_gnn_pin_cut_v1" for case in ("ariane133", "bp", "bp_multi", "swerv_wrapper")),
    *(case + "_lol_ng45_tp_gnn_ta3d_20260904" for case in ("ariane133", "bp_multi", "swerv_wrapper")),
)


@pytest.mark.parametrize("profile", RETIRED_PROFILES)
def test_retired_only_input_is_not_loadable(profile):
    with pytest.raises(FileNotFoundError):
        load_benchmark_input_profile(profile)


@pytest.mark.parametrize("kind", ["tp_gnn_ta3d_learning_objective", "tp_gnn_pin_cut_learning_objective"])
def test_retired_scoring_cannot_be_selected_even_by_external_task(tmp_path, kind):
    assert kind not in EXTRACTORS
    (tmp_path / "task.yaml").write_text(yaml.safe_dump({"schema_version": 1, "kind": kind}))
    with pytest.raises(ValueError, match="Retired cut-score"):
        load_task(tmp_path)


@pytest.mark.parametrize("name", CORE_TASK_DIRECTORIES)
def test_current_task_files_are_intact(name):
    spec = load_task(ROOT / "local_tasks" / name)
    load_candidate(spec.initial_candidate)
    for path in (spec.base_config, spec.prompt, spec.program_template, spec.program_adapter):
        if path is not None:
            assert path.is_file(), path


def test_all_shipped_task_and_evaluation_profile_input_dependencies_exist():
    definitions = list((ROOT / "local_tasks").glob("*/task.yaml"))
    definitions += list((ROOT / "evaluation_profiles").glob("*/profile.yaml"))
    for path in definitions:
        data = yaml.safe_load(path.read_text())
        assert (path.parent / data["base_config"]).is_file(), path
        if data.get("benchmark_input"):
            assert load_benchmark_input_profile(data["benchmark_input"]).config.is_file()


def test_canonical_assignment_reader_preserves_validation(tmp_path):
    path = tmp_path / "assignment.csv"
    path.write_text("inst_name,die\na,top\nb,bottom\n")
    raw = {"instance_count": 2, "top_count": 1, "bottom_count": 1}
    assert read_assignment(path, raw) == {"a": "top", "b": "bottom"}
    with pytest.raises(ValueError, match="count mismatch"):
        read_assignment(path, {**raw, "instance_count": 3})


@pytest.mark.parametrize("body", ["inst_name,die\n", "name,die\na,top\n",
    "inst_name,die\na,left\n", "inst_name,die\na,top\na,bottom\n", "inst_name,die\n,top\n"])
def test_invalid_assignments_still_fail_closed(tmp_path, body):
    path = tmp_path / "assignment.csv"
    path.write_text(body)
    with pytest.raises(ValueError):
        read_assignment(path, {"instance_count": 2, "top_count": 1, "bottom_count": 1})


def test_assignment_reader_is_part_of_evaluator_identity(monkeypatch):
    from local_bench import reference_catalog as catalog
    before = catalog.evaluator_digest()
    original = catalog.file_sha
    monkeypatch.setattr(catalog, "file_sha", lambda path: "changed" if path.name == "partition_assignment.py" else original(path))
    assert catalog.evaluator_digest() != before
