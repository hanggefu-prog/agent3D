"""Retention tests use synthetic files only: no Docker, EDA or model calls."""
import json
from pathlib import Path

import pytest

from scripts import experiment_retention as retention


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bulk(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "design.odb").write_bytes(b"physical database" * 20)
    (folder / "stage.log").write_text("Native original evidence\n")
    write(folder / "stage_manifest.json", {"artifacts": [{"path": "design.odb", "sha256": "original"}]})
    (folder / "config.yaml").write_text("seed: 123\n")


def local(tmp_path, state="completed"):
    root = (tmp_path / "local").resolve()
    root.mkdir()
    write(root / "run_config.json", {"source_commit": "pinned", "image": "sha256:base"})
    write(root / "status.json", {"state": state, "candidate_finals": 4})
    rows = []
    for order, score, ok in ((1, 2., True), (2, 1., True), (3, 2., True), (4, None, False)):
        folder = root / "evolution/evaluations" / f"{order:04d}-abcd"
        write(folder / "candidate/candidate.json", {"source_patch": "candidate patch"})
        metrics = {"runs_successfully": 1 if ok else 0, "combined_score": score}
        write(folder / "result/evaluation_result.json", {"metrics": metrics, "artifacts": {}})
        for tree in retention.LOCAL_TREES:
            bulk(folder / tree)
        write(folder / "image-build/receipt.json", {"image_id": f"sha256:candidate{order}"})
        rows.append({"order": order, "status": "accepted" if ok else "failed", "score": score,
                     "metrics": metrics, "candidate_sha256": str(order),
                     "candidate": str(folder / "candidate/candidate.json")})
    (root / "evolution/history.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    write(root / "summary.json", {"state": state, "candidate_finals": 4, "harness": {"best": rows[0]}})
    bulk(root / "reference/result/run")
    bulk(root / "unknown-directory")
    return root


def global_run(tmp_path, incumbent=2, reference_files=True):
    root = (tmp_path / "global").resolve()
    root.mkdir()
    write(root / "run_config.json", {"source_commit": "pinned"})
    write(root / "status.json", {"state": "completed", "final_evaluations": 4})
    history = []
    for number, score, success in ((0, 1., True), (1, 3., True), (2, 2., True), (3, 1.5, True), (4, None, False)):
        folder = root / "results" / ("reference" if number == 0 else f"round-{number:02d}")
        if number != 0 or reference_files:
            for tree in retention.GLOBAL_TREES:
                bulk(folder / tree)
        metrics = {"combined_score": score} if success else {}
        row = {"round": number, "success": success, "metrics": metrics, "run": str(folder / "run"),
               "adjudication": "completed"}
        history.append(row)
        if number == 0 and not reference_files:
            continue
        write(folder / "feedback.json", row)
        write(folder / "verification/verification.json", {"passed": success, "metrics": metrics})
        write(folder / "submission/submission.json", {"source_commit": "base"})
        (folder / "submission/patch.diff").write_text("Reproducible source patch\n")
    best = {"round": 1, "score": 3., "run": history[1]["run"]}
    inc = {"round": incumbent, "score": history[incumbent]["metrics"]["combined_score"], "run": history[incumbent]["run"]}
    write(root / "best.json", best)
    write(root / "incumbent.json", inc)
    write(root / "history.json", history)
    write(root / "summary.json", {"state": "completed", "final_evaluations": 4,
                                 "history": history, "best": best, "incumbent": inc})
    bulk(root / "subject")
    bulk(root / "scratch")
    return root


def test_full_is_exact_noop_even_without_root(tmp_path):
    root = tmp_path / "does-not-exist"
    assert retention.finalize_retention(root, "full", apply=True)["state"] == "unchanged"
    assert not root.exists()


def test_best_dry_run_is_readonly_and_deterministic_tie_preserves_recorded_best(tmp_path):
    root = local(tmp_path)
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    result = retention.finalize_retention(root, "best")
    assert result["state"] == "dry-run"
    assert len(result["pruned"]) == 10
    assert "evolution/evaluations/0001-abcd" in result["protected"]
    assert sorted(str(path.relative_to(root)) for path in root.rglob("*")) == before


def test_local_apply_keeps_reference_best_failures_configs_receipts_and_archive(tmp_path):
    root = local(tmp_path)
    original_summary = (root / "summary.json").read_bytes()
    result = retention.finalize_retention(root, "best", apply=True)
    assert result["state"] == "applied"
    assert (root / "summary.json").read_bytes() == original_summary
    for folder in ("reference/result/run", "evolution/evaluations/0001-abcd/result/run",
                   "evolution/evaluations/0004-abcd/result/run", "unknown-directory"):
        assert (root / folder / "design.odb").is_file()
    loser = "evolution/evaluations/0002-abcd"
    assert not (root / loser / "result/run").exists()
    assert (root / loser / "candidate/candidate.json").is_file()
    assert (root / loser / "image-build/receipt.json").is_file()
    archive = root / "retention/diagnostics" / loser / "result/run"
    assert (archive / "stage.log").read_text() == "Native original evidence\n"
    assert (archive / "config.yaml").read_text() == "seed: 123\n"
    assert not (archive / "design.odb").exists()
    assert retention.finalize_retention(root, "best", apply=True) == result
    assert "not intact native artifacts" in result["interpretation"]


def test_global_protects_measured_best_distinct_incumbent_reference_failure_and_working_state(tmp_path):
    root = global_run(tmp_path)
    result = retention.finalize_retention(root, "best", apply=True)
    assert result["pruned"] == ["results/round-03/run", "results/round-03/source", "results/round-03/dreamplace-install"]
    for folder in ("results/reference", "results/round-01", "results/round-02", "results/round-04"):
        assert (root / folder / "run/design.odb").exists()
    assert (root / "subject/design.odb").exists()
    assert (root / "scratch/design.odb").exists()
    assert (root / "results/round-03/submission/patch.diff").exists()


@pytest.mark.parametrize("state", ["running", "needs-attention", "reference_failed"])
def test_incomplete_or_failed_experiment_refused_and_driver_hook_noop(tmp_path, state):
    root = local(tmp_path, state)
    with pytest.raises(retention.RetentionError, match="completed"):
        retention.finalize_retention(root, "best", apply=True)
    assert retention.finish_retention(root, "best")["state"] == "skipped"
    assert not (root / "retention").exists()


def test_target_symlink_rejected_without_touching_external_path(tmp_path):
    root = local(tmp_path)
    target = root / "evolution/evaluations/0002-abcd/result/runtime"
    import shutil
    shutil.rmtree(target)
    external = tmp_path / "external"
    bulk(external)
    target.symlink_to(external, target_is_directory=True)
    with pytest.raises(retention.RetentionError, match="Symlinked"):
        retention.finalize_retention(root, "best", apply=True)
    assert (external / "design.odb").exists()
    assert not (root / "retention").exists()


def test_nested_symlink_unlinked_without_following_external_data(tmp_path):
    root = local(tmp_path)
    external = tmp_path / "external"
    bulk(external)
    target = root / "evolution/evaluations/0002-abcd/result/run"
    (target / "external").symlink_to(external, target_is_directory=True)
    result = retention.finalize_retention(root, "best", apply=True)
    assert not target.exists() and (external / "design.odb").exists()
    links = [row for row in result["inventory"] if row["kind"] == "symlink"]
    assert len(links) == 1 and links[0]["target"] == str(external)
    assert not any("/external/" in row["path"] for row in result["inventory"])
    assert not (root / "retention/diagnostics" / target.relative_to(root) / "external").exists()


def test_untrusted_candidate_escape_cannot_expand_deletion_scope(tmp_path):
    root = local(tmp_path)
    history = root / "evolution/history.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    rows[1]["candidate"] = str(tmp_path / "outside/candidate/candidate.json")
    history.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(retention.RetentionError, match="escapes"):
        retention.finalize_retention(root, "best", apply=True)
    assert not (root / "retention").exists()


def test_best_must_match_independent_success_not_llm_claim(tmp_path):
    root = global_run(tmp_path)
    write(root / "results/round-03/verification/verification.json", {"passed": False, "metrics": {}})
    with pytest.raises(retention.RetentionError, match="verification"):
        retention.finalize_retention(root, "best", apply=True)
    assert not (root / "retention").exists()


def test_pending_adjudication_cannot_lose_evidence(tmp_path):
    root = global_run(tmp_path)
    history = json.loads((root / "history.json").read_text())
    history[3]["adjudication"] = "pending"
    write(root / "history.json", history)
    summary = json.loads((root / "summary.json").read_text())
    summary["history"] = history
    write(root / "summary.json", summary)
    with pytest.raises(retention.RetentionError, match="adjudication"):
        retention.finalize_retention(root, "best", apply=True)


def test_archive_failure_cannot_delete_original_or_rewrite_physical_success(tmp_path, monkeypatch):
    root = local(tmp_path)
    original = (root / "summary.json").read_bytes()
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(retention.shutil, "copyfile", fail)
    result = retention.finish_retention(root, "best")
    assert result["state"] == "storage-error"
    assert (root / "summary.json").read_bytes() == original
    assert (root / "evolution/evaluations/0002-abcd/result/run/design.odb").exists()
    assert list(root.glob("retention-error-*.json"))
    with pytest.raises(retention.RetentionError, match="archive requires inspection"):
        retention.finalize_retention(root, "best", apply=True)


def test_frozen_input_overlap_and_source_checkout_refused(tmp_path):
    root = local(tmp_path)
    write(root / "run_config.json", {"baseline_run": str(root / "reference")})
    with pytest.raises(retention.RetentionError, match="frozen source/input"):
        retention.finalize_retention(root, "best")
    write(root / "run_config.json", {})
    (root / ".git").mkdir()
    with pytest.raises(retention.RetentionError, match="source checkout"):
        retention.finalize_retention(root, "best")


def test_cli_default_dry_run_and_unknown_mode_rejected(tmp_path, capsys):
    root = local(tmp_path)
    assert retention.main([str(root), "--mode", "best"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "dry-run"
    assert not (root / "retention").exists()
    with pytest.raises(retention.RetentionError):
        retention.validate_mode("compact")


def test_alias_cannot_prune_the_best_directory(tmp_path):
    root = local(tmp_path)
    history = root / "evolution/history.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    rows[2]["candidate"] = rows[0]["candidate"]
    history.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(retention.RetentionError, match="alias"):
        retention.finalize_retention(root, "best", apply=True)
    assert (root / "evolution/evaluations/0001-abcd/result/run/design.odb").exists()


def test_build_recipe_is_archived_before_context_removal(tmp_path):
    root = local(tmp_path)
    context = root / "evolution/evaluations/0002-abcd/image-build/context"
    for name in retention.BUILD_RECIPE_NAMES:
        (context / name).write_text("trusted build recipe\n")
    retention.finalize_retention(root, "best", apply=True)
    for name in retention.BUILD_RECIPE_NAMES:
        archived = root / "retention/diagnostics" / context.relative_to(root) / name
        assert archived.read_text() == "trusted build recipe\n"


def test_global_selected_path_cannot_alias_a_loser(tmp_path):
    root = global_run(tmp_path)
    history = json.loads((root / "history.json").read_text())
    history[1]["run"] = history[3]["run"]
    write(root / "history.json", history)
    best = json.loads((root / "best.json").read_text())
    best["run"] = history[3]["run"]
    write(root / "best.json", best)
    summary = json.loads((root / "summary.json").read_text())
    summary.update(history=history, best=best)
    write(root / "summary.json", summary)
    with pytest.raises(retention.RetentionError, match="native round"):
        retention.finalize_retention(root, "best", apply=True)


def test_initial_reference_incumbent_can_lack_separate_record(tmp_path):
    root = global_run(tmp_path, incumbent=0)
    (root / "incumbent.json").unlink()
    result = retention.finalize_retention(root, "best", apply=True)
    assert "results/reference" in result["protected"]
    assert (root / "results/reference/run/design.odb").exists()


def test_apply_requires_symlink_safe_rmtree(tmp_path, monkeypatch):
    root = local(tmp_path)
    monkeypatch.setattr(retention.shutil.rmtree, "avoids_symlink_attacks", False)
    with pytest.raises(retention.RetentionError, match="symlink-safe"):
        retention.finalize_retention(root, "best", apply=True)
    assert (root / "evolution/evaluations/0002-abcd/result/run/design.odb").exists()


def adopted_global(tmp_path, monkeypatch, *, best_reference=False):
    """Real record seal/adoption receipt, synthetic independently checked producer."""
    import copy
    from global_bench.verifiers.open3dbench_codesign_verifier import reference_self_metrics
    from local_bench import reference_catalog
    from local_bench.reference_store import adopt_reference, file_sha, publish_reference

    root = global_run(tmp_path, incumbent=0, reference_files=False)
    run = (tmp_path / "external-producer/run").resolve()
    bulk(run)
    artifact = run / "design.odb"
    contract = {"case": "bp_fe", "endpoint": "timing", "task_digest": "a" * 64,
                "evaluator_digest": "b" * 64, "frozen_inputs": {"fixture": "c" * 64},
                "physical": {"profile": "tp_gnn_placement_one_pass"}}
    raw = {"boundary_budget_model": "passive-cutpoint", "opt_passes": 1,
           "wns_setup_ns": -1., "tns_setup_ns": -10., "wns_deficit": 1., "tns_deficit": 10.}
    payload = {"qualified": True, "case": "bp_fe", "endpoint": "timing",
               "source_contract": contract["physical"], "frozen_inputs": contract["frozen_inputs"],
               "source_identity": {"fixture": "synthetic, not physical qualification"},
               "baseline": {"source_run": str(run), "raw_metrics": raw},
               "evidence": [{"path": str(artifact), "bytes": artifact.stat().st_size,
                             "sha256": file_sha(artifact), "role": "fixture"}]}
    producer = {"kind": "native-evaluation-v1", "run_root": str(run),
                "evaluation_root": str(run.parent), "case": "bp_fe", "endpoint": "timing",
                "profile": "tp_gnn_placement_one_pass"}
    def qualify(_):
        return copy.deepcopy(payload)
    monkeypatch.setattr(reference_catalog, "qualify_producer", qualify)
    record = publish_reference(tmp_path / "store", contract=contract, origin="FRESH",
                               producer=producer, validate=qualify)
    folder = root / "results/reference"
    receipt = adopt_reference(record, contract=contract, validate=qualify, output=folder)
    feedback = {"success": True, "metrics": reference_self_metrics("timing", raw, producer["profile"]),
                "run": str(run), "reference_origin": "adopted", "reference_id": receipt["reference_id"],
                "reference_record": str(record), "reference_record_sha256": file_sha(record),
                "reference_evaluator_invocations": 0, "new_reference_native_attempts": 0,
                "verification": str(folder / "reference_adoption.json")}
    write(folder / "feedback.json", feedback)
    write(root / "run_config.json", {"reference_policy": "ADOPT", "reference_selection": {
        "policy": "ADOPT", "record": str(record), "record_sha256": file_sha(record), "contract": contract}})
    history = json.loads((root / "history.json").read_text())
    history[0] = {"round": 0, "kind": "reference", **feedback}
    if best_reference:
        for row in history[1:]:
            if row["success"]:
                row["metrics"]["combined_score"] = row["round"] / 10.
                candidate_folder = root / "results" / f"round-{row['round']:02d}"
                write(candidate_folder / "feedback.json", row)
                write(candidate_folder / "verification/verification.json", {"passed": True, "metrics": row["metrics"]})
    incumbent = {"round": 0, "score": feedback["metrics"]["combined_score"], "run": str(run)}
    best = incumbent if best_reference else json.loads((root / "best.json").read_text())
    write(root / "best.json", best)
    write(root / "incumbent.json", incumbent)
    write(root / "history.json", history)
    summary = json.loads((root / "summary.json").read_text())
    summary.update(history=history, best=best, incumbent=incumbent)
    write(root / "summary.json", summary)
    return root, run, record


@pytest.mark.parametrize("best_reference", [False, True])
@pytest.mark.parametrize("apply", [False, True])
def test_adopted_external_reference_is_protected_and_never_a_prune_target(tmp_path, monkeypatch, best_reference, apply):
    root, run, record = adopted_global(tmp_path, monkeypatch, best_reference=best_reference)
    paths = [record, *[p for p in run.rglob("*") if p.is_file()],
             *[p for p in (root / "results/reference").rglob("*") if p.is_file()]]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    result = retention.finalize_retention(root, "best", apply=apply)
    assert "results/reference" in result["protected"]
    assert result["pruned"]
    assert all(path.startswith("results/round-") for path in result["pruned"])
    assert all(not Path(row["path"]).is_absolute() for row in result["inventory"])
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    assert not (root / "results/reference/run").exists()


@pytest.mark.parametrize("fault", ["record_sha", "receipt_sha", "receipt_qualified", "receipt_source",
    "feedback_source", "feedback_metrics", "both_metrics", "missing_receipt", "native_attempt", "artifact"])
def test_adopted_reference_tamper_blocks_all_pruning(tmp_path, monkeypatch, fault):
    root, run, record = adopted_global(tmp_path, monkeypatch)
    folder = root / "results/reference"
    if fault in {"record_sha", "artifact"}:
        target = record if fault == "record_sha" else run / "design.odb"
        target.chmod(0o644)
        target.write_bytes(target.read_bytes() + b" ")
    elif fault == "missing_receipt":
        (folder / "reference_adoption.json").unlink()
    elif fault.startswith("receipt_") or fault == "native_attempt":
        receipt = json.loads((folder / "reference_adoption.json").read_text())
        if fault == "receipt_sha": receipt["record_sha256"] = "0" * 64
        elif fault == "receipt_qualified": receipt["status"] = "failed"
        elif fault == "receipt_source": receipt["qualification"]["baseline"]["source_run"] = str(tmp_path / "wrong")
        else: receipt["new_reference_native_attempts"] = 1
        write(folder / "reference_adoption.json", receipt)
    else:
        feedback = json.loads((folder / "feedback.json").read_text())
        if fault == "feedback_source": feedback["run"] = str(tmp_path / "wrong")
        else: feedback["metrics"]["combined_score"] = 999.
        write(folder / "feedback.json", feedback)
        if fault == "both_metrics":
            history = json.loads((root / "history.json").read_text())
            history[0]["metrics"] = feedback["metrics"]
            write(root / "history.json", history)
            summary = json.loads((root / "summary.json").read_text())
            summary["history"] = history
            write(root / "summary.json", summary)
    with pytest.raises(retention.RetentionError):
        retention.finalize_retention(root, "best", apply=True)
    assert not (root / "retention").exists()
    assert (root / "results/round-03/run/design.odb").exists()


def test_adoption_does_not_relax_candidate_path_containment(tmp_path, monkeypatch):
    root, external, _ = adopted_global(tmp_path, monkeypatch)
    history = json.loads((root / "history.json").read_text())
    history[1]["run"] = str(external)
    write(root / "history.json", history)
    best = json.loads((root / "best.json").read_text())
    best["run"] = str(external)
    write(root / "best.json", best)
    summary = json.loads((root / "summary.json").read_text())
    summary.update(history=history, best=best)
    write(root / "summary.json", summary)
    with pytest.raises(retention.RetentionError, match="escapes"):
        retention.finalize_retention(root, "best", apply=True)
    assert not (root / "retention").exists()


def test_adopted_record_change_during_archival_blocks_pruning(tmp_path, monkeypatch):
    root, _, record = adopted_global(tmp_path, monkeypatch)
    original_copy = retention.shutil.copyfile
    changed = False
    def mutate_after_copy(*args, **kwargs):
        nonlocal changed
        result = original_copy(*args, **kwargs)
        if not changed:
            record.chmod(0o644)
            record.write_bytes(record.read_bytes() + b" ")
            changed = True
        return result
    monkeypatch.setattr(retention.shutil, "copyfile", mutate_after_copy)
    with pytest.raises(retention.RetentionError, match="pinned digest"):
        retention.finalize_retention(root, "best", apply=True)
    assert changed
    assert (root / "results/round-03/run/design.odb").exists()
    assert not (root / "retention/receipt.json").exists()


def test_extended_adoption_cannot_hide_parent_authority_in_experiment(tmp_path, monkeypatch):
    from local_bench.reference_store import file_sha, publish_reference
    root, _, record = adopted_global(tmp_path, monkeypatch)
    raw = json.loads(record.read_text())
    parent = root / "results/round-03/source" / record.name
    parent.write_bytes(record.read_bytes())
    producer = {**raw["producer"], "parent_record": str(parent), "parent_reference_id": raw["reference_id"]}
    child = publish_reference(tmp_path / "child-store", contract=raw["contract"], origin="EXTEND",
                              producer=producer, validate=lambda _: raw["qualification"])
    config = json.loads((root / "run_config.json").read_text())
    config["reference_selection"].update(record=str(child), record_sha256=file_sha(child))
    write(root / "run_config.json", config)
    with pytest.raises(retention.RetentionError, match="authority overlaps"):
        retention.finalize_retention(root, "best", apply=True)
    assert parent.exists() and not (root / "retention").exists()
