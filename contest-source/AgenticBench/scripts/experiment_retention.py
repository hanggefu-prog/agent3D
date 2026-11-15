"""Evidence-preserving storage retention, independent of search and scoring.

``full`` is an exact no-op. ``best`` may compact only successful non-selected
evaluations after an experiment has completed. The reference, recorded measured
best, distinct accepted incumbent and every failed/uncertain attempt are kept.
Workspace drivers can opt into ``best-per-round`` at quiescent round boundaries;
the latest candidate is also protected until the next round finishes.
This is storage retention, not result caching: no physical evaluation is run and
no candidate score is changed. Adopted reference feedback is checked against its
sealed raw metrics using the original objective normalization.

The CLI is dry-run by default. Native manifests are never rewritten to hide
deleted artifacts. A separate inventory states that compacted runs no longer
support full native hash replay or resume. No Docker images are removed here.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import time


class RetentionError(ValueError):
    """The attempt does not provide enough evidence for safe compaction."""


def validate_mode(mode):
    if mode not in {"full", "best"}:
        raise RetentionError("Retention mode must be full or best")
    return mode


def validate_workspace_mode(mode):
    if mode == "best-per-round":
        return mode
    return validate_mode(mode)


# Only these trees, under successful *losing* evaluation folders, are disposable.
LOCAL_TREES = ("result/run", "result/workspace", "result/runtime",
               "image-build/workspace", "image-build/context")
GLOBAL_TREES = ("run", "source", "dreamplace-install")
DIAGNOSTIC_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".toml", ".log",
                       ".rpt", ".md", ".csv", ".tsv", ".txt", ".patch",
                       ".diff", ".sdc", ".tcl", ".cfg", ".ini"}
BUILD_RECIPE_NAMES = {"Dockerfile", "build_openroad_candidate.sh", "seal_runtime.py"}


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RetentionError("Missing or symlinked evidence: " + str(path))
    try:
        return json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise RetentionError("Invalid JSON evidence: " + str(path)) from error


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _inside(root, path):
    """Resolve no links; every existing ancestor must be a real directory."""
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise RetentionError("Path escapes experiment: " + str(path)) from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RetentionError("Symlinked retention path: " + str(current))
    return path


def _inventory(root, target):
    """Never follow symlinks, devices, sockets or a mounted child filesystem."""
    target = _inside(root, target)
    if not target.is_dir():
        raise RetentionError("Prune target is not a directory: " + str(target))
    device = target.stat().st_dev
    entries = []
    for folder, directories, files in os.walk(target, followlinks=False):
        for name in sorted(directories + files):
            path = Path(folder) / name
            info = path.lstat()
            relative = str(path.relative_to(root))
            if stat.S_ISLNK(info.st_mode):
                # Keep only the link's literal payload. In particular, do not
                # stat, hash, open, copy or descend into its external target.
                target_text = os.readlink(path)
                entries.append({"path": relative, "kind": "symlink", "target": target_text,
                                "link_text_sha256": hashlib.sha256(os.fsencode(target_text)).hexdigest()})
                continue
            if info.st_dev != device:
                raise RetentionError("Refusing a nested filesystem: " + str(path))
            if stat.S_ISDIR(info.st_mode):
                entries.append({"path": relative, "kind": "directory"})
            elif stat.S_ISREG(info.st_mode):
                entries.append({"path": relative, "kind": "file", "size_bytes": info.st_size,
                                "sha256": _sha(path)})
            else:
                raise RetentionError("Refusing a non-regular artifact: " + str(path))
    return sorted(entries, key=lambda row: row["path"])


def _local_plan(root, summary):
    harness = summary.get("harness")
    if not isinstance(harness, dict) or not isinstance(harness.get("best"), dict):
        raise RetentionError("Local summary has no recorded measured best")
    best = harness["best"]
    if best.get("status") not in {"accepted", "cached"} or not _number(best.get("score")):
        raise RetentionError("Local best is not a successful finite-score candidate")
    history_path = _inside(root, root / "evolution/history.jsonl")
    if not history_path.is_file():
        raise RetentionError("Missing local candidate history")
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if len(rows) != summary.get("candidate_finals"):
        raise RetentionError("Local summary/history final counts disagree")
    best_matches = [row for row in rows if row.get("order") == best.get("order")
                    and row.get("candidate_sha256") == best.get("candidate_sha256")
                    and row.get("score") == best.get("score")]
    if len(best_matches) != 1:
        raise RetentionError("Local recorded best does not match history")
    selected = []
    losing = []
    evaluation_paths = set()
    for row in rows:
        # Malformed/rejected/cached records are never pruned.
        if row.get("status") != "accepted":
            continue
        if not _number(row.get("score")) or row.get("metrics", {}).get("runs_successfully") != 1:
            raise RetentionError("Accepted local record has inconsistent success evidence")
        candidate = _inside(root, Path(row.get("candidate", "")))
        relative = candidate.relative_to(root)
        if (len(relative.parts) != 5 or relative.parts[:2] != ("evolution", "evaluations")
                or relative.parts[-2:] != ("candidate", "candidate.json")):
            raise RetentionError("Unexpected local candidate layout")
        evaluation = candidate.parent.parent
        if evaluation in evaluation_paths:
            raise RetentionError("Multiple local records alias one physical evaluation")
        evaluation_paths.add(evaluation)
        if row["score"] > best["score"]:
            raise RetentionError("Recorded best is worse than a successful history row")
        result = _read(evaluation / "result/evaluation_result.json")
        if (result.get("metrics", {}).get("runs_successfully") != 1
                or result.get("metrics", {}).get("combined_score") != row["score"]):
            raise RetentionError("Local native result disagrees with recorded score")
        if row["order"] == best["order"]:
            selected.append(str(evaluation.relative_to(root)))
        else:
            losing.append((evaluation, LOCAL_TREES))
    if not selected:
        raise RetentionError("Local measured best has no retained successful evaluation")
    return ["reference", *selected], losing


def _adopted_reference(root, row):
    """Authenticate an external round-zero baseline without adding prune paths."""
    from local_bench.reference_catalog import qualify_producer
    from local_bench.reference_store import SCHEMA, load_reference

    folder = _inside(root, root / "results/reference")
    config = _read(root / "run_config.json")
    selected = config.get("reference_selection")
    if (config.get("reference_policy") != "ADOPT" or not isinstance(selected, dict)
            or selected.get("policy") != "ADOPT" or row.get("round") != 0
            or row.get("kind") != "reference"):
        raise RetentionError("Adopted reference lacks a pinned ADOPT selection")
    value, sha = selected.get("record"), selected.get("record_sha256")
    if not isinstance(value, str) or not isinstance(sha, str) or len(sha) != 64:
        raise RetentionError("Adopted reference record identity is missing")
    record_path = Path(value)
    if (not record_path.is_absolute() or record_path.resolve() != record_path
            or record_path.is_symlink() or any(p.is_symlink() for p in record_path.parents)
            or not record_path.is_file() or _sha(record_path) != sha):
        raise RetentionError("Adopted reference record differs from its pinned digest/path")
    try:
        record = load_reference(record_path, contract=selected.get("contract"), validate=qualify_producer)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RetentionError("Adopted reference no longer independently qualifies: " + str(error)) from error
    qualification = record["qualification"]
    source_run = Path(qualification["baseline"]["source_run"])
    # This exception is only for external authority. It must never permit an
    # adopted record to alias a successful candidate that could be compacted.
    external, seen = [], set()
    def collect(path, sealed):
        if path in seen or len(seen) >= 8:
            raise RetentionError("Invalid adopted reference lineage")
        seen.add(path)
        payload = sealed["qualification"]
        external.extend([path, Path(payload["baseline"]["source_run"]),
                         *(Path(item["path"]) for item in payload["evidence"])])
        producer = sealed["producer"]
        if producer.get("parent_record"):
            parent_path = Path(producer["parent_record"])
            try:
                parent_raw = _read(parent_path)
                parent = load_reference(parent_path, contract=parent_raw["contract"], validate=qualify_producer)
                if parent["reference_id"] != producer.get("parent_reference_id"):
                    raise ValueError("parent ID mismatch")
            except (KeyError, OSError, TypeError, ValueError) as error:
                raise RetentionError("Adopted reference parent no longer qualifies") from error
            collect(parent_path, parent)
    collect(record_path, record)
    for path in external:
        if (not path.is_absolute() or path.resolve() != path
                or path.is_symlink() or any(p.is_symlink() for p in path.parents)
                or path == root or root in path.parents or path in root.parents):
            raise RetentionError("Adopted reference authority overlaps the experiment or uses an unsafe path")
    receipt = _read(_inside(root, folder / "reference_adoption.json"))
    feedback = _read(_inside(root, folder / "feedback.json"))
    if (not isinstance(receipt, dict) or not isinstance(feedback, dict)
            or receipt.get("schema") != SCHEMA or receipt.get("operation") != "ADOPT"
            or receipt.get("status") != "qualified" or receipt.get("reference_origin") != "adopted"
            or receipt.get("record") != value or receipt.get("record_sha256") != sha
            or receipt.get("reference_id") != record["reference_id"]
            or receipt.get("original_origin") != record["origin"]
            or receipt.get("contract") != selected["contract"]
            or receipt.get("qualification") != qualification):
        raise RetentionError("Adopted reference receipt differs from its sealed qualification")
    for key in ("new_reference_native_attempts", "model_calls", "candidate_finals"):
        if type(receipt.get(key)) is not int or receipt[key] != 0:
            raise RetentionError("Adopted reference receipt has inconsistent execution accounting")
    from global_bench.verifiers.open3dbench_codesign_verifier import reference_self_metrics
    try:
        expected_metrics = reference_self_metrics(record["contract"]["endpoint"],
            qualification["baseline"]["raw_metrics"], record["contract"]["physical"]["profile"])
    except (KeyError, TypeError, ValueError) as error:
        raise RetentionError("Adopted reference has no supported scoring identity") from error
    for item in (feedback, row):
        if (item.get("success") is not True or item.get("reference_origin") != "adopted"
                or item.get("metrics") != expected_metrics
                or item.get("run") != str(source_run)
                or item.get("reference_record") != value or item.get("reference_record_sha256") != sha
                or item.get("reference_id") != record["reference_id"]
                or item.get("verification") != str(folder / "reference_adoption.json")):
            raise RetentionError("Adopted reference feedback/history differs from its qualified source")
        for key in ("reference_evaluator_invocations", "new_reference_native_attempts"):
            if type(item.get(key)) is not int or item[key] != 0:
                raise RetentionError("Adopted reference feedback has inconsistent execution accounting")


def _global_plan(root, summary, *, keep_latest=False):
    history = _read(root / "history.json")
    if not isinstance(history, list) or summary.get("history") != history:
        raise RetentionError("Global summary/history disagree")
    best = _read(root / "best.json")
    if summary.get("best") != best or not _number(best.get("score")):
        raise RetentionError("Global measured-best record is inconsistent")
    by_round = {row.get("round"): row for row in history}
    if len(by_round) != len(history):
        raise RetentionError("Duplicate global round records")
    selected_numbers = {0, best.get("round")}
    if keep_latest:
        selected_numbers.add(max(by_round))
    incumbent = summary.get("incumbent")
    if incumbent is not None:
        incumbent_file = root / "incumbent.json"
        if (not isinstance(incumbent, dict)
                or (incumbent_file.exists() and _read(incumbent_file) != incumbent)
                or (not incumbent_file.exists() and incumbent.get("round") != 0)):
            raise RetentionError("Global incumbent record is inconsistent")
        selected_numbers.add(incumbent.get("round"))
    for number in selected_numbers:
        row = by_round.get(number)
        # The latest failed candidate is retained for diagnosis, never selected
        # as a qualified best/incumbent merely because it is recent.
        if keep_latest and number == max(by_round) and number not in {0, best.get("round"),
                incumbent.get("round") if incumbent else None} and row.get("success") is not True:
            continue
        if not row or row.get("success") is not True or not _number(row.get("metrics", {}).get("combined_score")):
            raise RetentionError("Selected global result is not qualified")
        expected_run = root / "results" / ("reference" if number == 0 else f"round-{number:02d}") / "run"
        if number == 0 and row.get("reference_origin") == "adopted":
            _adopted_reference(root, row)
        elif _inside(root, Path(row.get("run", ""))) != expected_run:
            raise RetentionError("Selected global run path differs from its native round")
    if (by_round[best["round"]]["metrics"]["combined_score"] != best["score"]
            or by_round[best["round"]].get("run") != best.get("run")):
        raise RetentionError("Global best does not match measured result")
    if incumbent is not None:
        row = by_round[incumbent["round"]]
        if (row["metrics"]["combined_score"] != incumbent.get("score")
                or row.get("run") != incumbent.get("run")):
            raise RetentionError("Global incumbent does not match measured result")
    finals = [row for row in history if row.get("round", 0) > 0]
    if len(finals) != summary.get("final_evaluations"):
        raise RetentionError("Global final count disagrees with history")
    protected, losing = [], []
    for row in history:
        number = row.get("round")
        if type(number) is not int or number < 0:
            raise RetentionError("Invalid global round number")
        folder = root / "results" / ("reference" if number == 0 else f"round-{number:02d}")
        _inside(root, folder)
        if number in selected_numbers:
            protected.append(str(folder.relative_to(root)))
            continue
        if row.get("success") is not True:
            continue
        score = row.get("metrics", {}).get("combined_score")
        if not _number(score) or score > best["score"]:
            raise RetentionError("Global success/best score evidence is inconsistent")
        feedback = _read(folder / "feedback.json")
        verification = _read(folder / "verification/verification.json")
        if (feedback.get("success") is not True or feedback.get("metrics") != row.get("metrics")
                or verification.get("passed") is not True
                or verification.get("metrics") != row.get("metrics")):
            raise RetentionError("Global feedback/independent verification disagree")
        # A completed Agentic3D run must have finished the evidence-consuming roles.
        if "adjudication" in row and row["adjudication"] != "completed":
            raise RetentionError("Global role adjudication has not completed")
        losing.append((folder, GLOBAL_TREES))
    return protected, losing


def _write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def finalize_retention(root, mode="full", *, apply=False):
    """Plan/apply retention for one completed native local or workspace search.

    ``apply=False`` is read-only, including no creation of lock or report files.
    Applying twice returns the existing completed receipt, never repeats work.
    A partial receipt blocks retries so a human can inspect the retained evidence.
    Target/ancestor symlinks are rejected. Nested links are inventoried by literal
    link text and unlinked without following/copying/hashing their targets.
    """
    validate_mode(mode)
    return _finalize_retention(root, mode, apply=apply)


def _round_snapshot(root, summary, status, config, boundary):
    number, final = boundary
    expected_state = "completed" if final else "round-completed"
    if (config.get("retention") != "best-per-round"
            or summary.get("state") != expected_state or status.get("state") != expected_state
            or type(number) is not int or number < 1
            or summary.get("final_evaluations") != number
            or status.get("final_evaluations") != number
            or (not final and status.get("round") != number)):
        raise RetentionError("No matching driver-owned completed round boundary")
    history = _read(root / "history.json")
    if (summary.get("history") != history or [row.get("round") for row in history] != list(range(number + 1))
            or any(type(row.get("round")) is not int for row in history)):
        raise RetentionError("Round history must be complete and contiguous")
    for row in history[1:]:
        if "adjudication" in row and row["adjudication"] != "completed":
            raise RetentionError("Round role adjudication has not completed")
    if config.get("method") == "agentic3d":
        if not isinstance(summary.get("incumbent"), dict):
            raise RetentionError("Agentic3D incumbent is missing")
        from scripts.run_workspace_agentic3d import PHASE_ROLES
        turns = _read(root / "role_turns.json")
        expected = {(n, phase) for n in range(1, number + 1) for phase in PHASE_ROLES}
        actual = {(r.get("round"), r.get("phase")) for r in turns}
        if (actual != expected or len(turns) != len(expected)
                or any(r.get("state") != "completed" or r.get("validated") is not True for r in turns)
                or any(r.get("adjudication") != "completed" for r in history[1:])):
            raise RetentionError("Agentic3D role receipts are incomplete")


def _reproduction_evidence(root, losing):
    """Retain exact submitted code (including added files) before source pruning."""
    hashes = {}
    for folder, _ in losing:
        submission = _inside(root, folder / "submission")
        data = _read(submission / "submission.json")
        structural = _read(folder / "structural-verification.json")
        if structural.get("passed") is not True:
            raise RetentionError("Candidate submission did not pass structural verification")
        def checked(relative, expected):
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RetentionError("Unsafe reproduction path")
            path = _inside(submission, submission / relative)
            if not path.is_file() or _sha(path) != expected:
                raise RetentionError("Candidate reproduction bytes changed: " + relative)
            hashes[str(path.relative_to(root))] = expected
        checked(data["patch_path"], data["patch_sha256"])
        for row in data["changes"]:
            if row["sha256"] is not None:
                checked(row["submitted_path"], row["sha256"])
            elif row.get("submitted_path") is not None:
                raise RetentionError("Deleted candidate file has inconsistent reproduction metadata")
        for row in data["artifacts"]:
            checked(row["submitted_path"], row["sha256"])
        for path in (submission / "submission.json", folder / "structural-verification.json",
                     folder / "feedback.json", folder / "verification/verification.json"):
            _inside(root, path)
            hashes[str(path.relative_to(root))] = _sha(path)
    return hashes


def _finalize_retention(root, mode, *, apply=False, boundary=None):
    if mode == "full":
        return {"schema_version": 1, "mode": "full", "state": "unchanged", "pruned": []}
    root = Path(os.path.abspath(Path(root).expanduser()))
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise RetentionError("Experiment root must be an existing non-symlink directory")
    summary = _read(root / "summary.json")
    status = _read(root / "status.json")
    config = _read(root / "run_config.json")
    if boundary is not None:
        _round_snapshot(root, summary, status, config, boundary)
    elif summary.get("state") != "completed" or status.get("state") != "completed":
        raise RetentionError("Retention requires a successfully completed experiment")
    if (root / ".git").exists():
        raise RetentionError("An experiment root cannot be a source checkout")
    for key in ("source", "baseline_run", "reference_run", "benchmark_input", "normalization_reference"):
        value = config.get(key)
        if isinstance(value, str) and value:
            protected = Path(value).expanduser().resolve()
            if root == protected or root in protected.parents or protected in root.parents:
                raise RetentionError("Experiment overlaps frozen source/input: " + key)
    if boundary is None:
        archive = _inside(root, root / "retention")
    else:
        base = _inside(root, root / "results/retention")
        if base.exists():
            for previous in base.iterdir():
                if not previous.is_dir() or previous.is_symlink():
                    raise RetentionError("Unexpected round retention archive")
                receipt = _read(previous / "receipt.json")
                if receipt.get("state") != "applied" or _sha(previous / "pruned_inventory.json") != receipt.get("inventory_sha256"):
                    raise RetentionError("Prior round retention requires inspection")
                diagnostics = set(receipt.get("diagnostics", []))
                for row in receipt.get("inventory", []):
                    if row.get("path") in diagnostics:
                        path = _inside(root, previous / "diagnostics" / row["path"])
                        if not path.is_file() or _sha(path) != row["sha256"]:
                            raise RetentionError("Archived round diagnostic changed")
        number, final = boundary
        archive = _inside(root, base / ("final" if final else f"round-{number:02d}"))
    receipt_path = archive / "receipt.json"
    if receipt_path.exists():
        receipt = _read(receipt_path)
        if receipt.get("state") != "applied":
            raise RetentionError("Prior retention did not complete; inspect it before any retry")
        if boundary is not None and any(_sha(root / filename) != receipt.get(key) for key, filename in
                (("summary_sha256", "summary.json"), ("status_sha256", "status.json"),
                 ("run_config_sha256", "run_config.json"))):
            raise RetentionError("Round boundary changed after retention")
        return receipt
    if archive.exists():
        raise RetentionError("Existing retention archive requires inspection")
    if "harness" in summary:
        protected, losing = _local_plan(root, summary)
        kind = "local"
    else:
        protected, losing = _global_plan(root, summary, keep_latest=boundary is not None and not boundary[1])
        kind = "workspace"
    if boundary is not None and kind != "workspace":
        raise RetentionError("Per-round retention supports workspace drivers only")
    evidence = _reproduction_evidence(root, losing) if boundary is not None else {}
    inventory, targets, skipped = [], [], []
    for folder, names in losing:
        for name in names:
            target = _inside(root, folder / name)
            if not target.exists():
                continue
            try:
                entries = _inventory(root, target)
            except RetentionError as error:
                skipped.append({"path": str(target.relative_to(root)), "reason": str(error)})
                continue
            targets.append(str(target.relative_to(root)))
            inventory.extend(entries)
    diagnostic_rows = [row for row in inventory if row["kind"] == "file"
                       and (Path(row["path"]).suffix.lower() in DIAGNOSTIC_SUFFIXES
                            or Path(row["path"]).name in BUILD_RECIPE_NAMES)]
    report = {"schema_version": 1, "mode": "best-per-round" if boundary else "best", "kind": kind,
              "state": "dry-run", "protected": protected, "pruned": targets,
              "skipped": skipped, "inventory": inventory,
              "diagnostics": [row["path"] for row in diagnostic_rows],
              "bytes_in_prunable_trees": sum(row.get("size_bytes", 0) for row in inventory),
              "diagnostic_bytes_retained": sum(row["size_bytes"] for row in diagnostic_rows),
              "summary_sha256": _sha(root / "summary.json"),
              "status_sha256": _sha(root / "status.json"),
              "run_config_sha256": _sha(root / "run_config.json"),
              "reproduction_sha256": evidence,
              "interpretation": "Pruned successful non-selected runs are diagnostic-only, not intact native artifacts or resumable runs. Original manifests/results are unchanged. Reference, selected best/incumbent and all failed or uncertain payloads remain protected. Docker images and shared caches are untouched."}
    if not apply:
        return report
    lock_path = _inside(root, root / ".retention.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RetentionError("Another retention operation owns this experiment") from error
        if archive.exists():
            raise RetentionError("Retention archive appeared concurrently")
        archive.mkdir(parents=True)
        _write_new(archive / "plan.json", report)
        # Seal all compact evidence before deleting even one original file.
        for row in diagnostic_rows:
            destination = archive / "diagnostics" / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / row["path"], destination)
            if destination.stat().st_size != row["size_bytes"] or _sha(destination) != row["sha256"]:
                raise RetentionError("Diagnostic copy differs from inventory: " + row["path"])
        for key, filename in (("summary_sha256", "summary.json"), ("status_sha256", "status.json"),
                              ("run_config_sha256", "run_config.json")):
            if _sha(root / filename) != report[key]:
                raise RetentionError("Experiment state changed during retention")
        if kind == "workspace":
            if boundary is not None:
                _round_snapshot(root, _read(root / "summary.json"), _read(root / "status.json"), config, boundary)
                _global_plan(root, summary, keep_latest=not boundary[1])
                for path, expected in evidence.items():
                    if _sha(_inside(root, root / path)) != expected:
                        raise RetentionError("Candidate reproduction/feedback changed during retention")
            reference = next((row for row in summary["history"] if row.get("round") == 0), None)
            if reference and reference.get("reference_origin") == "adopted":
                _adopted_reference(root, reference)
        for target in targets:
            expected = [row for row in inventory if row["path"].startswith(target + "/")]
            if _inventory(root, root / target) != expected:
                raise RetentionError("Prune tree changed after evidence snapshot: " + target)
        _write_new(archive / "pruned_inventory.json", inventory)
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RetentionError("This platform lacks symlink-safe recursive deletion")
        for target in targets:
            shutil.rmtree(_inside(root, root / target))
        result = {**report, "state": "applied", "finished_at": time.time(),
                  "inventory_sha256": _sha(archive / "pruned_inventory.json")}
        _write_new(receipt_path, result)
        return result


def finish_retention(root, mode="full", *, driver_lock=None):
    """Best-effort driver hook; storage issues never change physical success.

    Healthy or failed searches are untouched. Only a completed experiment is
    passed to the strict finalizer. A separate storage-error receipt preserves
    any failure without editing summary, status, results or previous receipts.
    """
    if mode == "best-per-round":
        return finish_round_retention(root, mode, None, driver_lock=driver_lock, final=True)
    validate_mode(mode)
    if mode == "full":
        return finalize_retention(root, mode)
    root = Path(os.path.abspath(Path(root).expanduser()))
    try:
        if root.is_symlink() or root.resolve() != root or not root.is_dir():
            raise RetentionError("Unsafe experiment root")
        summary, status = _read(root / "summary.json"), _read(root / "status.json")
        if summary.get("state") != "completed" or status.get("state") != "completed":
            return {"mode": mode, "state": "skipped", "reason": "Experiment is not completed"}
        result = finalize_retention(root, mode, apply=True)
        print(json.dumps({"event": "experiment-retention", "mode": mode,
                          "state": result["state"], "root": str(root),
                          "receipt": str(root / "retention/receipt.json"),
                          "pruned_tree_count": len(result.get("pruned", []))}), flush=True)
        return result
    except Exception as error:
        result = {"schema_version": 1, "mode": mode, "state": "storage-error",
                  "error": f"{type(error).__name__}: {error}",
                  "physical_result_unchanged": True, "recorded_at": time.time()}
        try:
            if not root.is_symlink() and root.resolve() == root and root.is_dir():
                # Never replace a prior failed cleanup receipt.
                receipt = root / f"retention-error-{time.time_ns()}.json"
                _write_new(receipt, result)
        except OSError:
            # A full disk can prevent even the diagnostic write; returning the
            # warning still must not turn a valid physical result into failure.
            pass
        print(json.dumps({"event": "experiment-retention", "root": str(root),
                          **result}), flush=True)
        return result


def finish_round_retention(root, mode, number, *, driver_lock, final=False):
    """Synchronous workspace-driver hook, never an external live-run cleaner."""
    validate_workspace_mode(mode)
    if mode != "best-per-round":
        return {"mode": mode, "state": "unchanged"}
    root = Path(os.path.abspath(Path(root).expanduser()))
    try:
        if root.resolve() != root or not root.is_dir():
            raise RetentionError("Unsafe experiment root")
        path = _inside(root, root / "driver.lock")
        if driver_lock is None or driver_lock.closed:
            raise RetentionError("Per-round retention requires the active driver lock")
        actual, expected = os.fstat(driver_lock.fileno()), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise RetentionError("Per-round retention requires the active driver lock")
        # Reaffirm exclusive ownership on this open description. A separate
        # process cannot acquire it while the driver is running this hook.
        fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if final and number is None:
            number = _read(root / "status.json").get("final_evaluations")
        result = _finalize_retention(root, "best", apply=True, boundary=(number, final))
        print(json.dumps({"event": "round-retention", "round": number, "final": final,
                          "state": result["state"], "pruned_tree_count": len(result.get("pruned", []))}), flush=True)
        return result
    except Exception as error:
        result = {"state": "storage-error", "mode": mode, "round": number,
                  "error": f"{type(error).__name__}: {error}", "physical_result_unchanged": True}
        if root.is_dir() and root.resolve() == root:
            try:
                _write_new(root / f"retention-error-{time.time_ns()}.json", result)
            except OSError:
                pass
        print(json.dumps({"event": "round-retention", **result}), flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="One completed experiment, not a batch/source root")
    parser.add_argument("--mode", choices=("full", "best"), default="full")
    parser.add_argument("--apply", action="store_true", help="Apply planned pruning; default is read-only")
    args = parser.parse_args(argv)
    print(json.dumps(finalize_retention(args.root, args.mode, apply=args.apply), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
