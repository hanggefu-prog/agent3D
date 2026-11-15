"""Protect input boundaries without relying on Python 3.9 pathlib methods."""
from pathlib import Path

import pytest

from engineering_bench.protocol import path_within


def test_protected_containment_handles_equal_child_and_sibling():
    assert path_within(Path("/input/run"), Path("/input/run"))
    assert path_within(Path("/input/run/out"), Path("/input/run"))
    assert not path_within(Path("/input/run-other"), Path("/input/run"))


def test_snapshot_rejects_output_symlink_into_input_before_docker(tmp_path, monkeypatch):
    from engineering_bench.native_snapshot import snapshot
    run, platform = tmp_path / "input", tmp_path / "platform"
    run.mkdir(); platform.mkdir()
    alias = tmp_path / "output"
    alias.symlink_to(run, target_is_directory=True)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("must reject before Docker"))
    with pytest.raises(ValueError, match="protected input"):
        snapshot(run=run, platform=platform, image="sha256:" + "a" * 64,
                 stage="60_opt", die="bottom", output=alias / "new")
