"""Current-source admission is explicit and cannot relabel a dated reference."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from local_bench import current_nesterov as current, scoped_reference
from local_bench.model import load_task

ROOT = Path(__file__).resolve().parents[2]
SPEC = load_task(ROOT / "AgenticBench/local_tasks/placement_parameters_reference_20260923")
CONFIG = ROOT / "AgenticBench/local_bench/reference_configs/nesterov_current.yaml"


def test_existing_task_does_not_silently_select_new_reference():
    assert not current.selected(SPEC, None)
    assert current.selected(SPEC, CONFIG)
    assert not current.selected(SimpleNamespace(task_id="placement_initialization"), CONFIG)


def test_current_profile_binds_actual_runtime(monkeypatch):
    monkeypatch.setenv("OPEN3DBENCH_IMAGE_ID", "sha256:" + "a" * 64)
    bound = current.contract(CONFIG)
    assert bound["profile"] == current.PROFILE
    assert bound["image"] == "sha256:" + "a" * 64
    assert bound["config"]["placement"]["seed"] == 1000


@pytest.mark.parametrize("image", ["", "open3dbench:release", "sha256:abc"])
def test_current_profile_rejects_unpinned_runtime(monkeypatch, image):
    monkeypatch.setenv("OPEN3DBENCH_IMAGE_ID", image)
    with pytest.raises(ValueError, match="pinned"):
        current.contract(CONFIG)


@pytest.mark.parametrize("section,key,value", [
    ("placement", "seed", 1001), ("placement", "macro_halo_um", 2),
    ("partition", "algorithm", "random-balanced"), ("input", "mode", "checkpoint"),
])
def test_current_profile_rejects_changed_recipe(tmp_path, monkeypatch, section, key, value):
    monkeypatch.setenv("OPEN3DBENCH_IMAGE_ID", "sha256:" + "a" * 64)
    config = yaml.safe_load(CONFIG.read_text())
    config[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="recipe differs"):
        current.contract(path)


def test_current_reference_rejects_other_runtime_and_old_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(scoped_reference, "materialize_program", lambda *args: {"initial": True})
    monkeypatch.setattr(scoped_reference, "implementation_identity", lambda *args: {"source": "abc"})
    monkeypatch.setattr(scoped_reference, "sha256", lambda *args: "file-hash")
    context = {"image": "sha256:" + "a" * 64}
    initial = {"initial": True}
    fresh = scoped_reference.admit_reference(SPEC, initial, ROOT, tmp_path, None,
                                              execution_contract=context)
    assert fresh["mode"] == "FRESH"
    run = tmp_path / "reference/run"
    run.mkdir(parents=True)
    receipt = {"artifacts": {"task": SPEC.task_id, "reference_admission": fresh},
               "metrics": {"runs_successfully": 1.0}}
    (run.parent / "evaluation_result.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="implementation differs"):
        scoped_reference.admit_reference(SPEC, initial, ROOT, tmp_path, run,
                                          execution_contract={"image": "sha256:" + "b" * 64})
    old = copy.deepcopy(receipt)
    del old["artifacts"]["reference_admission"]["implementation_sha256"]["execution_contract"]
    (run.parent / "evaluation_result.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="implementation differs"):
        scoped_reference.admit_reference(SPEC, initial, ROOT, tmp_path, run, execution_contract=context)
    with pytest.raises(ValueError, match="exact native initial"):
        scoped_reference.admit_reference(SPEC, {"initial": False}, ROOT, tmp_path, None,
                                          execution_contract=context)
