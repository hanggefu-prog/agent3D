"""Evaluator wrapper hunks must equal the trusted routing adapter output."""
import json
import difflib
import subprocess
from pathlib import Path

import pytest

from local_bench.model import load_task
from local_bench.openevolve_adapter import _materialize_candidate
from local_bench.program_safety import unified_diff
from local_bench.routing_wrapper_integrity import (
    RUNNER,
    ROUTE_BACKEND,
    split_patch_sections,
    trusted_wrapper_diffs,
    verify_wrapper_hunks,
)
from local_bench.verifier import verify_candidate
from local_bench.workspace import apply_candidate_patch, _partition_patch
from local_tasks.routing_source_common import grt_only_patches


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
TASKS = ("global_routing_ripup_policy", "global_routing_maze_cost",
         "global_routing_iteration_controller")
ANCHOR = '        grt_env = dict(env, ROUTE_GUIDE=env["ROUTE_GRT_GUIDE"])\n'
ROUTE = (
    "import os\n\n\n"
    "def global_route(env, die, die_dir):\n"
    '    checkpoint = "global_route_checkpoint.json"\n'
    '    metrics = "grt_metrics.json"\n'
    "    if die:\n"
    + ANCHOR +
    "        return grt_env, checkpoint, metrics\n"
    "    return None\n"
)
RUNNER_TEXT = (
    "class Runner:\n"
    "    def global_route(self, destination):\n"
    "        return self.run(grt_only=True, output_dir=destination)\n"
)


def fake_repo(tmp_path):
    root = tmp_path / "open3dbench-root"
    for relative, text in ((ROUTE_BACKEND, ROUTE), (RUNNER, RUNNER_TEXT)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def source_hunk(spec):
    path = spec.allowed_source_paths[0]
    return unified_diff(path, "int a = 1;\nint b = 1;\n", "int a = 2;\nint b = 1;\n")


def candidate(patch):
    return {"schema_version": 1, "config_overrides": {}, "candidate_files": {},
            "source_patch": patch, "notes": "wrapper integrity test"}


@pytest.mark.parametrize("task", TASKS)
def test_trusted_wrapper_hunk_is_accepted(tmp_path, task):
    spec = load_task(ROOT / "local_tasks" / task)
    root = fake_repo(tmp_path)
    trusted = trusted_wrapper_diffs(root)
    assert trusted[ROUTE_BACKEND] and not trusted[RUNNER]
    patch = source_hunk(spec) + trusted[ROUTE_BACKEND]
    assert verify_candidate(spec, candidate(patch), root)["patch"]["paths"] == sorted(
        [spec.allowed_source_paths[0], ROUTE_BACKEND])
    assert verify_wrapper_hunks(spec, patch, root) == {
        ROUTE_BACKEND: "trusted", RUNNER: "absent"}


@pytest.mark.parametrize("task", TASKS)
def test_hand_edited_wrapper_hunk_is_rejected(tmp_path, task):
    spec = load_task(ROOT / "local_tasks" / task)
    root = fake_repo(tmp_path)
    trusted = trusted_wrapper_diffs(root)[ROUTE_BACKEND]
    edited = trusted.replace(
        "OPEN3D_ROUTING_DIE=die,\n",
        "OPEN3D_ROUTING_DIE=die, OPEN3D_SKIP_OVERFLOW_CHECK='1',\n")
    assert edited != trusted
    patch = source_hunk(spec) + edited
    # Missing trusted source must fail closed, rather than silently doing scope only.
    with pytest.raises(ValueError, match="require the Open3DBench checkout"):
        verify_candidate(spec, candidate(patch))
    with pytest.raises(ValueError, match="not byte-identical"):
        verify_candidate(spec, candidate(patch), root)


@pytest.mark.parametrize("task", TASKS)
def test_unrequested_runner_hunk_is_rejected(tmp_path, task):
    spec = load_task(ROOT / "local_tasks" / task)
    root = fake_repo(tmp_path)
    runner_hunk = unified_diff(RUNNER, RUNNER_TEXT, RUNNER_TEXT.replace(
        "grt_only=True", "grt_only=False"))
    with pytest.raises(ValueError, match=r"open3dbench/runner.py is not byte-identical"):
        verify_candidate(spec, candidate(source_hunk(spec) + runner_hunk), root)


def test_wrapper_hunk_split_or_with_preamble_is_rejected(tmp_path):
    spec = load_task(ROOT / "local_tasks" / TASKS[1])
    root = fake_repo(tmp_path)
    trusted = trusted_wrapper_diffs(root)[ROUTE_BACKEND]
    preamble = "diff --git a/{0} b/{0}\n".format(ROUTE_BACKEND)
    with pytest.raises(ValueError, match="not byte-identical"):
        verify_wrapper_hunks(spec, source_hunk(spec) + preamble + trusted, root)
    with pytest.raises(ValueError, match="not byte-identical"):
        verify_wrapper_hunks(spec, source_hunk(spec) + trusted + trusted, root)


def test_source_only_patch_and_missing_checkout():
    spec = load_task(ROOT / "local_tasks" / TASKS[0])
    patch = source_hunk(spec)
    assert verify_wrapper_hunks(spec, patch, None) == {
        ROUTE_BACKEND: "absent", RUNNER: "absent"}
    wrapper = unified_diff(ROUTE_BACKEND, "a\n", "b\n")
    with pytest.raises(ValueError, match="require the Open3DBench checkout"):
        verify_wrapper_hunks(spec, patch + wrapper, None)
    assert verify_candidate(spec, candidate(patch))["patch"]["present"]
    with pytest.raises(ValueError, match="require the Open3DBench checkout"):
        verify_candidate(spec, candidate(patch + wrapper))


@pytest.mark.parametrize("path", [
    "open3dbench//backend/three_d_route.py",
    "open3dbench/./backend/three_d_route.py",
    "./open3dbench/backend/three_d_route.py",
    "open3dbench/backend/three_d_route.py/.",
    "open3dbench/backend/../backend/three_d_route.py",
])
def test_wrapper_path_alias_is_rejected_by_verifier_and_apply(tmp_path, path):
    spec = load_task(ROOT / "local_tasks" / TASKS[1])
    root = fake_repo(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    patch = unified_diff(path, ROUTE, ROUTE.replace("grt_metrics.json", "altered_metrics.json"))
    patch_path = tmp_path / "candidate.patch"
    patch_path.write_text(patch)
    with pytest.raises(ValueError, match="non-canonical"):
        verify_candidate(spec, candidate(patch), root)
    with pytest.raises(ValueError, match="non-canonical"):
        apply_candidate_patch(root, patch_path)
    assert (root / ROUTE_BACKEND).read_text() == ROUTE


@pytest.mark.parametrize("zero_context", [False, True])
def test_canonical_trusted_plus_alias_mutation_cannot_hide_behind_trusted_status(tmp_path, zero_context):
    """Regress the actual strict and unidiff-zero two-section bypasses."""
    spec = load_task(ROOT / "local_tasks" / TASKS[1])
    root = fake_repo(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    trusted = trusted_wrapper_diffs(root)[ROUTE_BACKEND]
    # Strict Git application succeeds for the old implementation when the
    # second section's context describes the trusted first section's postimage.
    before = ROUTE if zero_context else grt_only_patches(root)[0]
    after = before.replace("grt_metrics.json", "altered_metrics.json")
    alias = "open3dbench//backend/three_d_route.py"
    if zero_context:
        mutation = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="a/" + alias, tofile="b/" + alias, n=0))
    else:
        mutation = unified_diff(alias, before, after)
    patch = trusted + mutation
    patch_path = tmp_path / "candidate.patch"
    patch_path.write_text(patch)
    with pytest.raises(ValueError, match="non-canonical"):
        verify_candidate(spec, candidate(patch), root)
    with pytest.raises(ValueError, match="non-canonical"):
        apply_candidate_patch(root, patch_path)
    assert (root / ROUTE_BACKEND).read_text() == ROUTE


def test_duplicate_canonical_wrapper_sections_fail_formal_verifier(tmp_path):
    spec = load_task(ROOT / "local_tasks" / TASKS[1])
    root = fake_repo(tmp_path)
    trusted = trusted_wrapper_diffs(root)[ROUTE_BACKEND]
    with pytest.raises(ValueError, match="not byte-identical"):
        verify_candidate(spec, candidate(trusted + trusted), root)


def test_trusted_canonical_wrapper_still_verifies_and_applies(tmp_path):
    spec = load_task(ROOT / "local_tasks" / TASKS[1])
    root = fake_repo(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    patch = trusted_wrapper_diffs(root)[ROUTE_BACKEND]
    assert verify_candidate(spec, candidate(patch), root)["patch"]["present"]
    patch_path = tmp_path / "candidate.patch"
    patch_path.write_text(patch)
    assert apply_candidate_patch(root, patch_path)["status"] == "applied"
    assert (root / ROUTE_BACKEND).read_text() == grt_only_patches(fake_repo(tmp_path / "second"))[0]


@pytest.mark.parametrize("before,after", [
    ("open3dbench/backend/three_d_route.py", "b/open3dbench/backend/three_d_route.py"),
    ("a/open3dbench/backend/three_d_route.py", "open3dbench/backend/three_d_route.py"),
    ("a/open3dbench/backend/three_d_route.py", "b/open3dbench/runner.py"),
])
def test_header_path_interpretation_cannot_differ_from_git_strip_level(before, after):
    patch = "--- {}\n+++ {}\n@@ -1 +1 @@\n-old\n+new\n".format(before, after)
    with pytest.raises(ValueError, match="prefixes|renames"):
        split_patch_sections(patch)


@pytest.mark.parametrize("path", [
    "OpenROAD-3D//openroad/src/grt/src/fastroute/src/maze.cpp",
    "OpenROAD-3D/./openroad/src/grt/src/fastroute/src/maze.cpp",
    "OpenROAD-3D/openroad/src/grt/src/fastroute/src/maze.cpp/.",
])
def test_nested_alias_never_reaches_outer_apply_partition(path):
    patch = unified_diff(path, "int a = 1;\n", "int a = 2;\n")
    with pytest.raises(ValueError, match="non-canonical"):
        _partition_patch(patch, "OpenROAD-3D/openroad/")


def test_nested_git_preamble_and_header_like_source_are_preserved():
    path = "OpenROAD-3D/openroad/src/grt/src/fastroute/src/maze.cpp"
    # A removed source line itself begins with '-- a/<prefix>'; it must not
    # become a new file section or have its apparent path rewritten.
    before = "-- a/OpenROAD-3D/openroad/unchanged.txt\nvalue = 1\n"
    patch = "diff --git a/{0} b/{0}\n".format(path) + unified_diff(path, before, "value = 2\n")
    outer, nested = _partition_patch(patch, "OpenROAD-3D/openroad/")
    assert not outer
    assert nested.startswith("diff --git a/src/grt/src/fastroute/src/maze.cpp b/src/grt/src/fastroute/src/maze.cpp\n")
    assert "--- a/OpenROAD-3D/openroad/unchanged.txt\n" in nested
    assert [path for path, _ in split_patch_sections(nested)] == ["src/grt/src/fastroute/src/maze.cpp"]


def test_git_preamble_alias_is_rejected_even_with_canonical_unified_headers(tmp_path):
    spec = load_task(ROOT / "local_tasks" / TASKS[1])
    root = fake_repo(tmp_path)
    patch = "diff --git a/open3dbench//backend/three_d_route.py b/open3dbench//backend/three_d_route.py\n"
    patch += unified_diff(ROUTE_BACKEND, ROUTE, ROUTE.replace("grt_metrics.json", "altered_metrics.json"))
    with pytest.raises(ValueError, match="non-canonical"):
        verify_candidate(spec, candidate(patch), root)


def test_hunk_bodies_resembling_headers_do_not_split_sections():
    patch = unified_diff("x.cpp", "a\n-- b\n++ c\n", "a\n--- b\n+++ c\n")
    sections = split_patch_sections(patch + unified_diff(RUNNER, "a\n", "b\n"))
    assert [path for path, _ in sections] == ["x.cpp", RUNNER]
    assert "".join(text for _, text in sections) == patch + unified_diff(RUNNER, "a\n", "b\n")


@pytest.mark.parametrize("task", TASKS)
def test_raw_candidate_json_cannot_bypass_the_adapter(tmp_path, task):
    spec = load_task(ROOT / "local_tasks" / task)
    root = fake_repo(tmp_path)
    trusted = trusted_wrapper_diffs(root)[ROUTE_BACKEND]
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "candidate.json").write_text(json.dumps(candidate(
        source_hunk(spec) + trusted.replace("hbt_ports.tsv", "other_ports.tsv"))))
    with pytest.raises(ValueError, match="not byte-identical"):
        _materialize_candidate(spec.root, raw / "candidate.json", tmp_path / "eval", root)
    (raw / "candidate.json").write_text(json.dumps(candidate(source_hunk(spec) + trusted)))
    out = _materialize_candidate(spec.root, raw / "candidate.json", tmp_path / "eval2", root)
    assert json.loads((out / "candidate.json").read_text())["source_patch"].endswith(trusted)


def test_current_checkout_wrapper_diffs_are_derivable():
    # The maintained checkout must support re-deriving the trusted wrapper.
    diffs = trusted_wrapper_diffs(REPO)
    assert set(diffs) == {ROUTE_BACKEND, RUNNER}
    spec = load_task(ROOT / "local_tasks" / TASKS[1])
    patch = source_hunk(spec) + diffs[ROUTE_BACKEND] + diffs[RUNNER]
    verify_candidate(spec, candidate(patch), REPO)
    if not diffs[RUNNER]:
        # The current standalone boundary needs no runner change, so any
        # runner hunk is outside the trusted envelope.
        before = (REPO / RUNNER).read_text(encoding="utf-8")
        after = before.replace("\n", "\n# injected\n", 1)
        with pytest.raises(ValueError, match="not byte-identical"):
            verify_candidate(spec, candidate(
                source_hunk(spec) + unified_diff(RUNNER, before, after)), REPO)
