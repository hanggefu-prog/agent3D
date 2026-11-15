"""Evaluator identity v2: engineering_bench plus its import closure only."""
from pathlib import Path

import pytest

from engineering_bench.identity import (PATH_REFERENCED, current_identity, evaluator_files, evaluator_identity,
    identity_version, imported_files, verify_evaluator)

FRAMEWORK = Path(__file__).resolve().parents[1]


def tree(tmp_path):
    files = {
        "engineering_bench/__init__.py": "",
        "engineering_bench/core.py": "from .helper import value\nimport shared.util\n",
        "engineering_bench/helper.py": "def value():\n    from shared import lazy\n",
        "engineering_bench/physical/flat.py": "import sibling\nfrom layout import x\n",
        "engineering_bench/physical/sibling.py": "import json\n",
        "engineering_bench/physical/layout.py": "",
        "engineering_bench/stage.tcl": "puts ok\n",
        "shared/__init__.py": "",
        "shared/util.py": "",
        "shared/lazy.py": "",
        "shared/unused.py": "",
        "unrelated_bench/model.py": "",
        "scripts/build_openroad_candidate.sh": "#!/bin/sh\n",
        "scripts/other_track.sh": "#!/bin/sh\n",
        "global_bench/container_terminal.py": "",
        "global_bench/other.py": "",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path / "engineering_bench"


def test_closure_inventory(tmp_path):
    root = tree(tmp_path)
    files = evaluator_files(root)
    assert files == sorted([
        "engineering_bench/__init__.py", "engineering_bench/core.py", "engineering_bench/helper.py",
        "engineering_bench/physical/flat.py", "engineering_bench/physical/layout.py",
        "engineering_bench/physical/sibling.py", "engineering_bench/stage.tcl",
        "global_bench/container_terminal.py", "scripts/build_openroad_candidate.sh",
        "shared/__init__.py", "shared/lazy.py", "shared/util.py"])
    assert imported_files(root / "helper.py", tmp_path) == {(tmp_path / "shared/lazy.py").resolve(),
                                                            (tmp_path / "shared/__init__.py").resolve()}


def test_v2_ignores_unrelated_tracks_but_v1_does_not(tmp_path):
    root = tree(tmp_path)
    v1, v2 = evaluator_identity(root, version=1), evaluator_identity(root)
    (tmp_path / "scripts/other_track.sh").write_text("#!/bin/sh\necho changed\n")
    (tmp_path / "unrelated_bench/model.py").write_text("changed = True\n")
    (tmp_path / "shared/unused.py").write_text("changed = True\n")
    assert evaluator_identity(root) == v2
    # v1 hashed every script in the shared directories; it is kept as frozen.
    assert evaluator_identity(root, version=1) != v1
    for name in ("shared/lazy.py", "engineering_bench/stage.tcl", "scripts/build_openroad_candidate.sh",
                 "engineering_bench/physical/sibling.py"):
        before = evaluator_identity(root)
        (tmp_path / name).write_text((tmp_path / name).read_text() + "\n# changed\n")
        assert evaluator_identity(root) != before, name
    with pytest.raises(ValueError, match="Unknown"):
        evaluator_identity(root, version=3)


def test_missing_path_referenced_runtime_fails(tmp_path):
    root = tree(tmp_path)
    (tmp_path / "global_bench/container_terminal.py").unlink()
    with pytest.raises(ValueError, match="path-referenced"):
        evaluator_files(root)


def test_record_versions_and_verification(tmp_path):
    root = tree(tmp_path)
    assert identity_version({}) == 1 and identity_version({"identity_version": 2}) == 2
    for bad in (3, "2", True):
        with pytest.raises(ValueError):
            identity_version({"identity_version": bad})
    old = {"evaluator_sha256": evaluator_identity(root, version=1)}
    new = {"identity_version": 2, "evaluator_sha256": evaluator_identity(root)}
    assert verify_evaluator(old, root) and verify_evaluator(new, root)
    with pytest.raises(ValueError, match="changed"):
        verify_evaluator(dict(old, identity_version=2), root)
    (tmp_path / "shared/util.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="changed"):
        verify_evaluator(new, root)


def test_real_inventory_is_engineering_closure():
    files = evaluator_files()
    assert set(PATH_REFERENCED) <= set(files)
    assert "engineering_bench/identity.py" in files and "engineering_bench/physical/cts_final_queue.py" in files
    assert all((FRAMEWORK / name).is_file() for name in files)
    outside = [f for f in files if not f.startswith("engineering_bench/")]
    assert outside and all(f.endswith(".py") or f in PATH_REFERENCED for f in outside)
    assert not any(f.startswith("local_bench/") and f.endswith(".sh") for f in files)
    assert current_identity() == {"identity_version": 2, "evaluator_sha256": evaluator_identity()}
