"""Synthetic round boundaries only; no model, Docker, SSH or physical flow."""
import fcntl
import json
from contextlib import contextmanager

import pytest

from scripts import experiment_retention as retention
from test_experiment_retention import global_run, write


def read(path):
    return json.loads(path.read_text())


def prepared(tmp_path, *, agentic=False):
    root = global_run(tmp_path)
    config = read(root / "run_config.json")
    config.update(retention="best-per-round", method="agentic3d" if agentic else "codex")
    write(root / "run_config.json", config)
    for folder in (root / "results").iterdir():
        source = folder / "submission/source_changes/new.py"
        source.parent.mkdir()
        source.write_text("# Complete added candidate source\n")
        write(folder / "submission/submission.json", {
            "patch_path": "patch.diff", "patch_sha256": retention._sha(folder / "submission/patch.diff"),
            "changes": [{"status": "added", "path": "new.py", "sha256": retention._sha(source),
                         "submitted_path": "source_changes/new.py"},
                        {"status": "deleted", "path": "old.py", "sha256": None, "submitted_path": None}],
            "artifacts": []})
        write(folder / "structural-verification.json", {"passed": True})
    boundary(root)
    if agentic:
        from scripts.run_workspace_agentic3d import PHASE_ROLES
        write(root / "role_turns.json", [
            {"round": n, "phase": phase, "state": "completed", "validated": True}
            for n in range(1, 5) for phase in PHASE_ROLES])
    return root


def boundary(root, *, final=False):
    state = "completed" if final else "round-completed"
    for name in ("status.json", "summary.json"):
        value = read(root / name)
        value.update(state=state, round=4)
        write(root / name, value)


@contextmanager
def locked(root):
    with (root / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock


def compact(root, *, final=False):
    with locked(root) as lock:
        return retention.finish_round_retention(root, "best-per-round", 4,
                                                driver_lock=lock, final=final)


@pytest.mark.parametrize("agentic", [False, True])
def test_round_keeps_best_incumbent_reference_failures_and_exact_code(tmp_path, agentic):
    root = prepared(tmp_path, agentic=agentic)
    before = {p: (root / p).read_bytes() for p in ("summary.json", "status.json", "history.json")}
    result = compact(root)
    assert result["state"] == "applied"
    assert result["pruned"] == ["results/round-03/" + name for name in retention.GLOBAL_TREES]
    assert compact(root) == result  # Exact same boundary is idempotent.
    for p, content in before.items():
        assert (root / p).read_bytes() == content
    for name in ("reference", "round-01", "round-02", "round-04"):
        assert (root / "results" / name / "run/design.odb").is_file()
    assert (root / "subject/design.odb").is_file()
    assert (root / "scratch/design.odb").is_file()
    assert (root / "results/round-03/submission/source_changes/new.py").read_text().startswith("# Complete")
    assert (root / "results/round-03/submission/patch.diff").is_file()
    archived = root / "results/retention/round-04/diagnostics/results/round-03/run/stage.log"
    assert archived.read_text() == "Native original evidence\n"
    assert not (archived.parent / "design.odb").exists()


def latest_success(root):
    history = read(root / "history.json")
    history[-1].update(success=True, metrics={"combined_score": 1.2})
    write(root / "history.json", history)
    summary = read(root / "summary.json")
    summary["history"] = history
    write(root / "summary.json", summary)
    write(root / "results/round-04/feedback.json", history[-1])
    write(root / "results/round-04/verification/verification.json", {"passed": True, "metrics": history[-1]["metrics"]})


def test_latest_success_is_available_for_next_turn_then_compacted_at_final(tmp_path):
    root = prepared(tmp_path)
    latest_success(root)
    assert compact(root)["state"] == "applied"
    assert (root / "results/round-04/run/design.odb").exists()
    boundary(root, final=True)
    result = compact(root, final=True)
    assert result["state"] == "applied"
    assert result["pruned"] == ["results/round-04/" + name for name in retention.GLOBAL_TREES]
    assert (root / "results/retention/final/receipt.json").is_file()


def test_new_best_releases_previous_best_but_not_distinct_incumbent(tmp_path):
    root = prepared(tmp_path)
    history = read(root / "history.json")
    history[-1].update(success=True, metrics={"combined_score": 4.0})
    best = {"round": 4, "score": 4.0, "run": history[-1]["run"]}
    write(root / "history.json", history)
    write(root / "best.json", best)
    summary = read(root / "summary.json")
    summary.update(history=history, best=best)
    write(root / "summary.json", summary)
    write(root / "results/round-04/feedback.json", history[-1])
    write(root / "results/round-04/verification/verification.json", {"passed": True, "metrics": history[-1]["metrics"]})
    result = compact(root)
    assert result["state"] == "applied"
    assert not (root / "results/round-01/run").exists()
    assert (root / "results/round-02/run/design.odb").exists()
    assert (root / "results/round-04/run/design.odb").exists()


@pytest.mark.parametrize("mode", ["full", "best"])
def test_round_hook_does_nothing_in_existing_modes(tmp_path, mode):
    root = tmp_path / "absent"
    assert retention.finish_round_retention(root, mode, 1, driver_lock=None)["state"] == "unchanged"
    assert not root.exists()


@pytest.mark.parametrize("change", ["live", "history", "score", "patch", "source", "structural", "adjudication"])
def test_uncertain_evidence_blocks_deletion(tmp_path, change):
    root = prepared(tmp_path)
    if change == "live":
        write(root / "status.json", {"state": "final-evaluation", "final_evaluations": 4})
    elif change == "history":
        write(root / "history.json", read(root / "history.json")[:-1])
    elif change == "score":
        write(root / "results/round-03/verification/verification.json", {"passed": False})
    elif change == "patch":
        (root / "results/round-03/submission/patch.diff").write_text("changed")
    elif change == "source":
        (root / "results/round-03/submission/source_changes/new.py").unlink()
    elif change == "structural":
        write(root / "results/round-03/structural-verification.json", {"passed": False})
    else:
        rows = read(root / "history.json")
        rows[-1]["adjudication"] = "needs_evidence"
        write(root / "history.json", rows)
        summary = read(root / "summary.json")
        summary["history"] = rows
        write(root / "summary.json", summary)
    assert compact(root)["state"] == "storage-error"
    assert (root / "results/round-03/run/design.odb").exists()


@pytest.mark.parametrize("change", ["missing", "unvalidated", "running", "duplicate"])
def test_agentic_requires_all_role_receipts(tmp_path, change):
    root = prepared(tmp_path, agentic=True)
    turns = read(root / "role_turns.json")
    if change == "missing": turns.pop()
    elif change == "unvalidated": turns[-1]["validated"] = False
    elif change == "running": turns[-1]["state"] = "running"
    else: turns.append(turns[-1])
    write(root / "role_turns.json", turns)
    assert compact(root)["state"] == "storage-error"
    assert (root / "results/round-03/run/design.odb").exists()


def test_driver_lock_is_mandatory_and_other_owner_is_respected(tmp_path):
    root = prepared(tmp_path)
    assert retention.finish_round_retention(root, "best-per-round", 4, driver_lock=None)["state"] == "storage-error"
    with locked(root):
        with (root / "driver.lock").open("a") as other:
            assert retention.finish_round_retention(root, "best-per-round", 4, driver_lock=other)["state"] == "storage-error"
    assert (root / "results/round-03/run/design.odb").exists()


def test_failed_archive_blocks_this_and_later_boundaries(tmp_path, monkeypatch):
    root = prepared(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("synthetic disk full")
    with monkeypatch.context() as m:
        m.setattr(retention.shutil, "copyfile", fail)
        assert compact(root)["state"] == "storage-error"
    assert compact(root)["state"] == "storage-error"
    boundary(root, final=True)
    assert compact(root, final=True)["state"] == "storage-error"
    assert (root / "results/round-03/run/design.odb").exists()


def test_archived_evidence_cannot_be_changed_before_next_cleanup(tmp_path):
    root = prepared(tmp_path)
    latest_success(root)
    assert compact(root)["state"] == "applied"
    path = root / "results/retention/round-04/diagnostics/results/round-03/run/stage.log"
    path.write_text("changed")
    boundary(root, final=True)
    assert compact(root, final=True)["state"] == "storage-error"
    assert (root / "results/round-04/run/design.odb").exists()


def test_changed_state_during_archive_preserves_all_payloads(tmp_path, monkeypatch):
    root = prepared(tmp_path)
    copy = retention.shutil.copyfile
    def change(*args, **kwargs):
        result = copy(*args, **kwargs)
        write(root / "status.json", {"state": "running"})
        return result
    monkeypatch.setattr(retention.shutil, "copyfile", change)
    assert compact(root)["state"] == "storage-error"
    assert (root / "results/round-03/run/design.odb").exists()


def test_public_cli_cannot_compact_a_live_search(tmp_path):
    root = prepared(tmp_path)
    with pytest.raises(retention.RetentionError):
        retention.finalize_retention(root, "best-per-round", apply=True)
    with pytest.raises(retention.RetentionError, match="completed"):
        retention.finalize_retention(root, "best", apply=True)


def test_partial_removal_blocks_later_cleanup_without_replaying(tmp_path, monkeypatch):
    root = prepared(tmp_path)
    latest_success(root)
    remove = retention.shutil.rmtree
    count = 0
    def interrupted(path):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("synthetic interrupted cleanup")
        return remove(path)
    interrupted.avoids_symlink_attacks = remove.avoids_symlink_attacks
    with monkeypatch.context() as m:
        m.setattr(retention.shutil, "rmtree", interrupted)
        assert compact(root)["state"] == "storage-error"
    assert not (root / "results/round-03/run").exists()
    assert (root / "results/round-03/source/design.odb").exists()
    assert (root / "results/retention/round-04/diagnostics/results/round-03/run/stage.log").is_file()
    boundary(root, final=True)
    assert compact(root, final=True)["state"] == "storage-error"
    assert (root / "results/round-04/run/design.odb").exists()


def test_unsafe_submission_link_does_not_escape_deletion_scope(tmp_path):
    root = prepared(tmp_path)
    path = root / "results/round-03/submission/source_changes/new.py"
    outside = tmp_path / "external.py"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    assert compact(root)["state"] == "storage-error"
    assert (root / "results/round-03/source/design.odb").exists()
    assert outside.is_file()


def test_missing_final_evidence_is_storage_error_not_physical_failure(tmp_path):
    root = prepared(tmp_path)
    (root / "status.json").unlink()
    with locked(root) as lock:
        assert retention.finish_retention(root, "best-per-round", driver_lock=lock)["state"] == "storage-error"
    assert (root / "results/round-03/run/design.odb").exists()
