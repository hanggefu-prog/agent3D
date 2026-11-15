"""Synthetic control-plane retention tests; none is native EDA qualification."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.archive import completed_outcomes
from grt_bench.common import digest, read, records, sha, verify_records, write
from global_bench.grt_trace_retention import TraceRetention
from scripts.pilot_trace_retention import SyntheticCampaign, disk_usage, pins_for, prune_paths


@pytest.fixture
def pilot(tmp_path):
    fixture = SyntheticCampaign(tmp_path / "isolated-pilot-campaign")
    fixture.branches()
    origin = tmp_path / "readonly-origin"
    origin.mkdir()
    write(origin / "identity.json", {"synthetic_only": True})
    retention = TraceRetention.initialize(fixture.root, origin, fixture.task, acknowledge_isolated_copy=True)
    for index, name in enumerate(("baseline", "improved", "weak", "tie", "derived")):
        retention.seal(fixture.names[name], "event-%d" % index)
    return fixture, retention


def test_three_cases_three_seeds_keep_strict_best_trace_not_parent_ancestry(pilot):
    fixture, retention = pilot
    assert len(fixture.queue.rows()) == 45
    trace = retention.trace()
    assert [row["candidate"] for row in trace] == [fixture.names[name] for name in ("baseline", "improved", "derived")]
    assert fixture.names["tie"] not in {row["candidate"] for row in trace}
    assert fixture.archive.admit(fixture.names["derived"])["parent"] == fixture.names["tie"]
    assert len({payload["case"] for payload in [__import__("json").loads(row["payload"]) for row in fixture.queue.rows()]}) == 3


def test_pruning_preserves_budget_archive_receipts_and_history(pilot):
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    budget_before = sha(fixture.budget_path)
    queue_before = digest(fixture.queue.rows())
    archive_before = digest(fixture.archive.show())
    histories_before = retention.history(candidate)
    paths = prune_paths(fixture, candidate)
    before = disk_usage(fixture.root)
    plan = retention.plan_prune(candidate, paths, pins=pins_for(fixture))
    receipt = retention.apply(plan)
    after = disk_usage(fixture.root)
    assert before["allocated_bytes_unique_inodes"] > after["allocated_bytes_unique_inodes"]
    assert any(not (fixture.root / path).exists() for path in paths)
    assert sha(fixture.budget_path) == budget_before
    assert digest(fixture.queue.rows()) == queue_before
    assert digest(fixture.archive.show()) == archive_before
    history = retention.history(candidate)
    assert history["raw_evidence_state"] == "pruned by retention policy"
    assert histories_before["raw_evidence_state"] == "retained"
    assert "not a fresh physical verification" in history["verification_scope"]
    assert retention.apply(plan) == receipt
    # Host archive verification still sees all authoritative compact evidence.
    outcomes = completed_outcomes(fixture.root, fixture.task, candidate, fixture.archive.admit(candidate))
    assert len(outcomes) == 9
    for job in fixture.jobs[candidate]:
        attempt = fixture.root / "runs" / job
        # Full original raw inventory cannot be relabelled as a fresh verification.
        with pytest.raises((ValueError, FileNotFoundError)):
            verify_records(attempt / "output", read(attempt / "physical-inventory.json"))
    reading = retention.reading_index()
    assert candidate in str(reading)
    assert "pruned by retention policy" in str(reading)


def test_best_logs_survive_while_old_best_physical_artifacts_can_retire(pilot):
    fixture, retention = pilot
    candidate = fixture.names["improved"]
    original_logs = {path: sha(fixture.root / path) for path in prune_paths(fixture, candidate) if path.endswith(".log")}
    physical = [path for path in prune_paths(fixture, candidate) if path.endswith((".odb", ".def", ".spef"))]
    retention.apply(retention.plan_prune(candidate, physical, pins=pins_for(fixture)))
    assert physical and all(not (fixture.root / path).exists() for path in physical)
    assert all(sha(fixture.root / path) == checksum for path, checksum in original_logs.items())
    # Full trace bytes live in the sealed bundle; redundant native-path copies
    # can then retire without losing the best-so-far log trajectory.
    retention.apply(retention.plan_prune(candidate, list(original_logs), pins=pins_for(fixture)))
    history = retention.history(candidate)
    assert original_logs
    for item in history["logs"]:
        assert sha(fixture.root / item["blob"]) == item["file"]["sha256"]


def test_checkpoint_pin_protects_whole_stage_and_its_logs(pilot):
    fixture, retention = pilot
    candidate = fixture.names["tie"]
    job = fixture.jobs[candidate][0]
    checkpoint = "runs/%s/output/run/80_route/design.odb" % job
    sibling_log = "runs/%s/output/run/80_route/stage.log" % job
    with pytest.raises(ValueError):
        retention.plan_prune(candidate, [sibling_log], pins=pins_for(fixture, [checkpoint]))
    # An unrelated stage can still be removed once all explicit dependencies are known.
    unrelated = "runs/%s/output/run/60_opt/stage.log" % job
    retention.apply(retention.plan_prune(candidate, [unrelated], pins=pins_for(fixture, [checkpoint])))
    assert (fixture.root / checkpoint).exists()
    assert (fixture.root / sibling_log).exists()
    assert not (fixture.root / unrelated).exists()


def test_best_record_retains_actual_upstream_producer_logs(pilot):
    fixture, retention = pilot
    source_candidate, destination_candidate = fixture.names["tie"], fixture.names["derived"]
    for source_job, destination_job in zip(fixture.jobs[source_candidate], fixture.jobs[destination_candidate]):
        attempt = fixture.root / "runs" / destination_job
        proof = read(attempt / "control/incremental.json")
        assert proof == read(attempt / "verified/result.json")["incremental"]
        assert proof["source_job"] == source_job
    upstream_log = "runs/%s/output/run/30_placement/stage.log" % fixture.jobs[source_candidate][0]
    source_checksum = sha(fixture.root / upstream_log)
    retained = [item for item in retention.history(destination_candidate)["logs"] if item["path"] == upstream_log]
    assert len(retained) == 1
    assert retained[0]["reused"] is True
    assert sha(fixture.root / retained[0]["blob"]) == source_checksum
    retention.apply(retention.plan_prune(source_candidate, [upstream_log], pins=pins_for(fixture)))
    assert not (fixture.root / upstream_log).exists()
    assert sha(fixture.root / retained[0]["blob"]) == source_checksum


@pytest.mark.parametrize("tamper", ["control-proof", "empty-stages", "missing-proof",
                                     "unplanned-sidecar", "source-log", "source-plan"])
def test_untrusted_proof_or_missing_manifest_log_cannot_publish_trace(tmp_path, tamper):
    fixture = SyntheticCampaign(tmp_path / "upstream-negative-pilot-campaign")
    fixture.branches()
    origin = tmp_path / "readonly-origin"
    origin.mkdir()
    retention = TraceRetention.initialize(fixture.root, origin, fixture.task, acknowledge_isolated_copy=True)
    for index, name in enumerate(("baseline", "improved", "weak", "tie")):
        retention.seal(fixture.names[name], "event-%d" % index)
    source_job = fixture.jobs[fixture.names["tie"]][0]
    destination_job = fixture.jobs[fixture.names["derived"]][0]
    attempt = fixture.root / "runs" / destination_job
    proof = read(attempt / "control/incremental.json")
    assert proof == read(attempt / "verified/result.json")["incremental"]
    if tamper == "control-proof":
        # This alternative source still matches the full case/seed, so checking
        # pairing alone would accept it. Its unverified provenance must not win.
        proof["source_job"] = fixture.jobs[fixture.baseline][0]
        write(attempt / "control/incremental.json", proof, replace=True)
    elif tamper == "missing-proof":
        (attempt / "control/incremental.json").unlink()
    elif tamper == "empty-stages":
        proof["stages"] = {}
        write(attempt / "control/incremental.json", proof, replace=True)
    elif tamper == "unplanned-sidecar":
        source_attempt = fixture.root / "runs" / source_job
        assert not read(source_attempt / "control/plan.json").get("incremental")
        assert "incremental" not in read(source_attempt / "verified/result.json")
        # An empty stage set avoids an unrelated self-cycle rejection: only
        # the unplanned sidecar itself must be enough to reject publication.
        write(source_attempt / "control/incremental.json", dict(proof, stages={}))
    elif tamper == "source-log":
        source_log = fixture.root / "runs" / source_job / "output/run/30_placement/stage.log"
        manifest = read(source_log.parent / "stage_manifest.json")
        assert "stage.log" in manifest["artifacts"]
        source_log.unlink()
    else:
        source_plan = fixture.root / "runs" / source_job / "control/plan.json"
        value = read(source_plan)
        value["task_digest"] = "f" * 64
        write(source_plan, value, replace=True)
    before = digest(fixture.queue.rows())
    with pytest.raises((ValueError, FileNotFoundError)):
        retention.seal(fixture.names["derived"], "event-derived")
    assert digest(fixture.queue.rows()) == before
    assert not (fixture.root / "control/trace-retention-pilot/history" /
                (fixture.names["derived"] + ".json")).exists()


def test_baseline_prefix_and_current_frontier_are_pinned(pilot):
    fixture, retention = pilot
    for candidate, relative in ((fixture.baseline, "output/run/00_input/stage.log"),
                                (fixture.names["derived"], "output/run/80_route/design.odb")):
        job = fixture.jobs[candidate][0]
        with pytest.raises(ValueError):
            retention.plan_prune(candidate, ["runs/%s/%s" % (job, relative)], pins=pins_for(fixture))


def test_incomplete_or_mismatched_dependency_inventory_is_rejected(pilot):
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    for field, value in (("complete", False), ("root", "/elsewhere"), ("task_digest", "f" * 64)):
        pins = pins_for(fixture)
        pins[field] = value
        with pytest.raises(ValueError):
            retention.plan_prune(candidate, prune_paths(fixture, candidate), pins=pins)


def test_failed_and_uncertain_matrices_cannot_be_sealed_or_pruned(pilot):
    fixture, retention = pilot
    for name, uncertain in (("failed", False), ("uncertain", True)):
        candidate, job = fixture.terminal_branch(name, uncertain=uncertain)
        before = digest(fixture.queue.rows())
        with pytest.raises(ValueError):
            retention.seal(candidate, name + "-event")
        with pytest.raises((ValueError, FileNotFoundError)):
            retention.plan_prune(candidate, prune_paths(fixture, candidate), pins=pins_for(fixture))
        assert digest(fixture.queue.rows()) == before
        assert (fixture.root / "runs" / job / "output/run/30_placement/diagnostic.log").exists()
    assert next(row for row in fixture.queue.rows() if row["id"] == job)["state"] == "uncertain"


@pytest.mark.parametrize("bad", ["../outside.log", "/tmp/outside.log", "control/pilot-budget.json", "candidates/noop/tree/algo.py"])
def test_scope_and_path_escape_are_rejected(pilot, bad):
    fixture, retention = pilot
    with pytest.raises(ValueError):
        retention.plan_prune(fixture.names["weak"], [bad], pins=pins_for(fixture))


def test_symlink_and_artifact_tamper_fail_without_deletion(pilot, tmp_path):
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    paths = prune_paths(fixture, candidate)
    original = fixture.root / paths[0]
    outside = tmp_path / "outside.log"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(ValueError):
        retention.plan_prune(candidate, paths, pins=pins_for(fixture))
    assert outside.exists()
    original.unlink()
    original.write_bytes(outside.read_bytes())
    plan = retention.plan_prune(candidate, paths, pins=pins_for(fixture))
    original.write_bytes(b"changed after planning")
    with pytest.raises(ValueError):
        retention.apply(plan)
    assert original.read_bytes() == b"changed after planning"


def test_hash_bound_plan_rejects_path_mutation(pilot):
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    plan = retention.plan_prune(candidate, prune_paths(fixture, candidate), pins=pins_for(fixture))
    forged = copy.deepcopy(plan)
    forged["candidate"] = fixture.names["derived"]
    with pytest.raises(ValueError):
        retention.apply(forged)


def test_tampered_history_seal_cannot_authorize_deletion(pilot):
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    record = fixture.root / "control/trace-retention-pilot/history" / (candidate + ".json")
    value = read(record)
    value["body"]["score"] = -100
    write(record, value, replace=True)
    with pytest.raises(ValueError):
        retention.history(candidate)
    with pytest.raises(ValueError):
        retention.plan_prune(candidate, prune_paths(fixture, candidate), pins=pins_for(fixture))


def test_published_best_log_blob_tamper_is_detected(pilot):
    fixture, retention = pilot
    record = retention.history(fixture.names["improved"])
    blob = fixture.root / record["logs"][0]["blob"]
    blob.chmod(0o600)
    blob.write_bytes(b"tampered trace log")
    with pytest.raises(ValueError):
        retention.trace()


def test_queue_change_after_plan_does_not_start_deletion(pilot):
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    paths = prune_paths(fixture, candidate)
    plan = retention.plan_prune(candidate, paths, pins=pins_for(fixture))
    fixture.terminal_branch("new-failure")
    with pytest.raises(ValueError):
        retention.apply(plan)
    assert all((fixture.root / path).exists() for path in paths)


def test_new_pin_generation_invalidates_old_plan(pilot):
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    paths = prune_paths(fixture, candidate)
    plan = retention.plan_prune(candidate, paths, pins=pins_for(fixture))
    changed = pins_for(fixture, [paths[0]])
    changed["generation"] = "new-host-dependency-generation"
    retention.publish_pins(changed)
    with pytest.raises(ValueError, match="pin generation"):
        retention.apply(plan)
    assert all((fixture.root / path).exists() for path in paths)


def test_agent_development_phase_blocks_retention_even_with_idle_queue(pilot):
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    paths = prune_paths(fixture, candidate)
    plan = retention.plan_prune(candidate, paths, pins=pins_for(fixture))
    write(fixture.root / "control/campaign.json", {"rounds": [{"round": 6, "status": "editing"}]})
    assert all(row["state"] == "succeeded" for row in fixture.queue.rows())
    with pytest.raises(ValueError, match="campaign"):
        retention.plan_prune(candidate, paths, pins=pins_for(fixture))
    with pytest.raises(ValueError, match="campaign"):
        retention.apply(plan)
    assert all((fixture.root / path).exists() for path in paths)


def test_isolated_copy_marker_is_required(tmp_path):
    fixture = SyntheticCampaign(tmp_path / "production-root")
    origin = tmp_path / "origin"
    origin.mkdir()
    with pytest.raises(ValueError):
        TraceRetention.initialize(fixture.root, origin, fixture.task, acknowledge_isolated_copy=True)
    safe = tmp_path / "unmarked-pilot-root"
    safe.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        TraceRetention(safe, fixture.task)
    with pytest.raises(ValueError):
        TraceRetention.initialize(safe, origin, fixture.task, acknowledge_isolated_copy=False)


def test_interrupted_delete_resumes_exact_plan_without_double_accounting(pilot, monkeypatch):
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    paths = prune_paths(fixture, candidate)
    plan = retention.plan_prune(candidate, paths, pins=pins_for(fixture))
    original_unlink = Path.unlink
    removed = []

    def stop_after_first_file(path, *args, **kwargs):
        if str(path).startswith(str(fixture.root / "runs")):
            if removed:
                raise OSError("Synthetic interrupted removal")
            removed.append(str(path))
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "unlink", stop_after_first_file)
        with pytest.raises(OSError, match="interrupted"):
            retention.apply(plan)
    assert len(removed) == 1
    receipt = retention.apply(plan)
    assert retention.apply(plan) == receipt
    assert all(not (fixture.root / path).exists() for path in paths)


def test_crash_between_unlink_and_progress_is_not_guessed_success(pilot, monkeypatch):
    from global_bench import grt_trace_retention as module
    fixture, retention = pilot
    candidate = fixture.names["weak"]
    paths = prune_paths(fixture, candidate)
    plan = retention.plan_prune(candidate, paths, pins=pins_for(fixture))
    original_write = module.write

    def fail_progress(path, value, **kwargs):
        if str(path).endswith(".progress.json"):
            raise OSError("Synthetic crash before durable removal progress")
        return original_write(path, value, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(module, "write", fail_progress)
        with pytest.raises(OSError, match="durable removal"):
            retention.apply(plan)
    assert sum(not (fixture.root / path).exists() for path in paths) == 1
    with pytest.raises((ValueError, FileNotFoundError)):
        retention.apply(plan)
    assert sum(not (fixture.root / path).exists() for path in paths) == 1
    # An unresolved deletion is never advertised as completed historical pruning.
    history = retention.history(candidate)
    assert history["raw_evidence_state"] == "pruning incomplete; reconciliation required"
    assert history["incomplete_pruning_plans"]
