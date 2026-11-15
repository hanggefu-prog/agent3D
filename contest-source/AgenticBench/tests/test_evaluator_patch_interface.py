"""Select trusted adapters from source compatibility without altering policy."""
from pathlib import Path
import shutil

import pytest

from local_bench import evaluation


def test_current_placement_promotion_uses_current_two_pass_interface(tmp_path):
    root = Path(__file__).resolve().parents[2]
    relative = Path("open3dbench/backend/three_d_opt.py")
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    shutil.copyfile(root / relative, destination)
    before = destination.read_bytes()
    for kind in ("placement_two_pass_sta", "tp_gnn_placement_sta", "tp_gnn_placement_cts_thermal"):
        selected = evaluation.select_evaluator_source_patch(kind, tmp_path)
        assert selected == evaluation.EVALUATOR_SOURCE_PATCHES["tp_gnn_placement_sta"]
        assert destination.read_bytes() == before


def fixture_patch(path, old):
    path.write_text("--- a/core.txt\n+++ b/core.txt\n@@ -1 +1 @@\n-{}\n+adapted\n".format(old))
    return path


def test_historical_interface_is_explicitly_matched_and_unknown_rejected(tmp_path, monkeypatch):
    modern = fixture_patch(tmp_path / "modern.patch", "modern")
    legacy = fixture_patch(tmp_path / "legacy.patch", "legacy")
    monkeypatch.setattr(evaluation, "EVALUATOR_SOURCE_PATCHES", {
        "tp_gnn_placement_sta": modern, "placement_two_pass_sta": legacy})
    core = tmp_path / "core.txt"
    core.write_text("legacy\n")
    assert evaluation.select_evaluator_source_patch("placement_two_pass_sta", tmp_path) == legacy
    assert core.read_text() == "legacy\n"
    core.write_text("unknown\n")
    with pytest.raises(ValueError, match="exactly one compatible source interface"):
        evaluation.select_evaluator_source_patch("placement_two_pass_sta", tmp_path)
    assert evaluation.select_evaluator_source_patch("placement_source", tmp_path) is None


def test_ambiguous_interface_is_not_guessed(tmp_path, monkeypatch):
    one = fixture_patch(tmp_path / "one.patch", "same")
    two = fixture_patch(tmp_path / "two.patch", "same")
    monkeypatch.setattr(evaluation, "EVALUATOR_SOURCE_PATCHES", {
        "tp_gnn_placement_sta": one, "placement_two_pass_sta": two})
    (tmp_path / "core.txt").write_text("same\n")
    with pytest.raises(ValueError, match="found 2"):
        evaluation.select_evaluator_source_patch("placement_two_pass_sta", tmp_path)
