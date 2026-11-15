"""Exercise the qualification copy paths without post-3.8 pathlib APIs.

These are local synthetic filesystem checks: no compiler, Docker, model or EDA.
Removing the APIs prevents a newer host Python from masking container failures.
"""
import json
import os
from pathlib import Path, PurePath
import stat

import pytest

from local_bench import benchmark_input, evaluation, reference_archive, workspace
from scripts import qualify_archive_grt as grt
from test_reference_evaluated import (
    make_evaluation,
    test_fresh_then_extend_publications_are_read_only_adoptable as _exercise_registry_extension,
)


def _relative_to_python38(self, *other):
    """The old lexical implementation, without 3.12's is_relative_to call.

    Removing only is_relative_to on 3.12 would break relative_to itself rather
    than emulate 3.8. Keep this test-only shim separate from production code.
    """
    base = type(self)(*other)
    count = len(base.parts)
    if self.anchor != base.anchor or type(self)(*self.parts[:count]) != base:
        raise ValueError("{} is not beneath {}".format(self, base))
    return type(self)(*self.parts[count:])


@pytest.fixture
def python38_paths(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(PurePath, "relative_to", _relative_to_python38)
        patch.delattr(PurePath, "is_relative_to", raising=False)
        patch.delattr(Path, "readlink", raising=False)
        assert not hasattr(Path("."), "is_relative_to")
        assert not hasattr(Path("."), "readlink")
        yield


def test_container_extend_adoption_requalifies_prefix_without_new_pathlib(
        make_evaluation, tmp_path, python38_paths):
    # This existing synthetic lifecycle exercises real catalog publication,
    # recursive parent requalification, byte-identical prefix checks and the
    # final rejection of a mutated parent. No evaluator/tool dispatch is used.
    _exercise_registry_extension(make_evaluation, tmp_path)


def evidence(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": grt.file_sha(path)}


def prefix_fixture(root):
    for stage, directory in grt.STAGES.items():
        folder = root / directory
        folder.mkdir(parents=True)
        (folder / "stage_manifest.json").write_text(json.dumps({"stage": stage}))
    folder = root / "10_floorplan"
    for name in ("design.def", "design.v", "design.sdc"):
        (folder / name).write_text("frozen " + name + "\n")
    (folder / "design_params.json").write_text(json.dumps({
        "def_input": "/old/design.def", "verilog_input": "/old/design.v",
        "sdc_input": "/old/design.sdc", "lef_input": ["/fixed/tech.lef"]}))
    return [evidence(path) for path in root.rglob("*") if path.is_file()]


def test_archive_prefix_copy_preserves_membership_and_bytes_without_new_pathlib(
        tmp_path, monkeypatch, python38_paths):
    monkeypatch.delenv("OPEN3D_LINK_FROZEN_UPSTREAM", raising=False)
    root = tmp_path / "sealed-run"
    rows = prefix_fixture(root)
    # Archive-wide evidence outside this run is validated but never copied.
    sibling = tmp_path / "sealed-run-other/70_cts/stage_manifest.json"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("external archive evidence\n")
    rows.append(evidence(sibling))
    before = {row["path"]: Path(row["path"]).read_bytes() for row in rows}
    target = tmp_path / "new-run"
    copied = grt.copy_prefix({"baseline": {"source_run": str(root)}, "evidence": rows}, target)
    assert len(copied) == len(rows) - 1
    assert all(not name.startswith("..") for name in copied)
    assert not (target / "15_timing_extract").exists()
    assert not (target / "75_global_route").exists()
    assert before == {row["path"]: Path(row["path"]).read_bytes() for row in rows}
    params = json.loads((target / "10_floorplan/design_params.json").read_text())
    assert params["def_input"] == str(target / "10_floorplan/design.def")
    assert params["lef_input"] == ["/fixed/tech.lef"]
    for row in rows[:-1]:
        relative = Path(row["path"]).relative_to(root)
        if relative.name != "design_params.json":
            assert grt.file_sha(target / relative) == row["sha256"]


@pytest.mark.parametrize("fault", ["changed-bytes", "symlink"])
def test_archive_copy_keeps_source_evidence_guards(tmp_path, python38_paths, fault):
    root = tmp_path / "sealed-run"
    rows = prefix_fixture(root)
    path = root / "70_cts/stage_manifest.json"
    if fault == "changed-bytes":
        path.write_text("changed\n")
    else:
        external = tmp_path / "external.json"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
    with pytest.raises(ValueError, match="Authenticated evidence changed"):
        grt.copy_prefix({"baseline": {"source_run": str(root)}, "evidence": rows}, tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("owner", [benchmark_input, evaluation])
def test_placement_input_and_evaluator_use_compatible_sealed_copy_helper(
        tmp_path, monkeypatch, python38_paths, owner):
    assert owner.copy_upstream_stages is workspace.copy_upstream_stages
    monkeypatch.delenv("OPEN3D_LINK_FROZEN_UPSTREAM", raising=False)
    root = tmp_path / "sealed-run"
    rows = prefix_fixture(root)
    for row in rows:
        Path(row["path"]).chmod(0o444)
    folders = [root, *(p for p in root.rglob("*") if p.is_dir())]
    for folder in folders:
        folder.chmod(0o555)
    before = {row["path"]: (Path(row["path"]).read_bytes(), Path(row["path"]).stat().st_mode)
              for row in rows}
    target = tmp_path / "new-run"
    try:
        owner.copy_upstream_stages(root, target, ["input", "floorplan", "partition"])
        assert before == {row["path"]: (Path(row["path"]).read_bytes(), Path(row["path"]).stat().st_mode)
                          for row in rows}
        assert json.loads((target / "10_floorplan/design_params.json").read_text())["def_input"] == str(
            target / "10_floorplan/design.def")
        assert not (target / "10_floorplan/design.def").stat().st_mode & stat.S_IWUSR
        assert (target / "10_floorplan/stage_manifest.json").stat().st_mode & stat.S_IWUSR
    finally:
        # Restore only the synthetic fixture's directories for pytest cleanup.
        for folder in folders:
            folder.chmod(0o755)


@pytest.mark.parametrize("fault", ["sibling", "parent", "symlink"])
def test_mutable_copy_containment_is_unchanged_without_new_pathlib(tmp_path, python38_paths, fault):
    root = tmp_path / "copy"
    root.mkdir()
    if fault == "sibling":
        path = tmp_path / "copy-other/metadata.json"
    elif fault == "parent":
        path = root / "../metadata.json"
    else:
        external = tmp_path / "external"
        external.mkdir()
        (root / "alias").symlink_to(external, target_is_directory=True)
        path = root / "alias/metadata.json"
    with pytest.raises(workspace.WorkspaceError, match="escapes|symlink"):
        workspace._check_copied_path(root, path)


@pytest.mark.parametrize("fault", [None, "changed-target", "escape"])
def test_authenticated_archive_symlink_uses_compatible_readlink(
        tmp_path, python38_paths, fault):
    root = tmp_path / "archive"
    root.mkdir()
    target = root / "native.log"
    target.write_text("authenticated native log\n")
    name = "alias.log"
    expected = "native.log"
    actual = expected
    if fault == "changed-target":
        actual = "other.log"
    elif fault == "escape":
        target = tmp_path / "outside.log"
        target.write_text("outside\n")
        actual = expected = "../outside.log"
    (root / name).symlink_to(actual)
    archive = reference_archive._Archive.__new__(reference_archive._Archive)
    archive.root, archive.evidence = root, {}
    archive.inventory = {name: {"kind": "symlink", "target": expected},
                         "native.log": {"kind": "file", "bytes": target.stat().st_size,
                                        "sha256": grt.file_sha(target)}}
    assert os.readlink(root / name) == actual
    if fault is None:
        assert archive.file(name) == target
        assert archive.evidence[str(target)]["sha256"] == grt.file_sha(target)
    else:
        with pytest.raises(ValueError, match="symlink identity differs|symlink escapes"):
            archive.file(name)
