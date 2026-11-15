"""Published source archives use independent copies without requiring Git history."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from local_bench.workspace import WorkspaceError, isolated_clone, apply_candidate_patch


def test_archive_copy_is_independent_patchable_and_preserves_internal_links(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    code = source / "algorithm.py"
    code.write_text("VALUE = 1\n")
    code.chmod(0o444)
    (source / "alias.py").symlink_to("algorithm.py")
    nested = source / "nested"
    nested.mkdir()
    (nested / "tool").write_text("#!/bin/sh\n")
    (nested / "tool").chmod(0o555)
    nested.chmod(0o555)
    target = isolated_clone(source, tmp_path / "candidate")
    assert not (target / ".git").exists()
    assert (target / "algorithm.py").stat().st_ino != code.stat().st_ino
    assert (target / "alias.py").resolve() == target / "algorithm.py"
    assert os.access(target / "nested/tool", os.X_OK)
    patch = tmp_path / "edit.patch"
    patch.write_text("--- a/algorithm.py\n+++ b/algorithm.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n")
    apply_candidate_patch(target, patch)
    assert (target / "algorithm.py").read_text() == "VALUE = 2\n"
    assert code.read_text() == "VALUE = 1\n"
    receipt = json.loads((tmp_path / "candidate.archive-source.json").read_text())
    assert receipt["git_revision"] is None and receipt["files"] == 3
    assert len(receipt["tree_sha256"]) == 64


@pytest.mark.parametrize("link", ["../outside", "/etc/passwd", "missing"])
def test_archive_external_absolute_and_broken_links_are_rejected(tmp_path, link):
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "outside").write_text("private")
    (source / "alias").symlink_to(link)
    with pytest.raises(WorkspaceError, match="Unsafe source archive link"):
        isolated_clone(source, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_archive_special_files_and_nested_destination_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(WorkspaceError, match="Invalid source archive"):
        isolated_clone(source, source / "candidate")
    os.mkfifo(source / "pipe")
    with pytest.raises(WorkspaceError, match="Unsupported source archive entry"):
        isolated_clone(source, tmp_path / "candidate")


def test_git_checkout_still_clones_committed_state(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])
    git("init", "-q")
    (source / "algorithm.py").write_text("VALUE = 1\n")
    git("add", "algorithm.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
    (source / "algorithm.py").write_text("VALUE = 2\n")
    candidate = isolated_clone(source, tmp_path / "candidate")
    assert (candidate / "algorithm.py").read_text() == "VALUE = 1\n"
    assert (candidate / ".git").is_dir()
