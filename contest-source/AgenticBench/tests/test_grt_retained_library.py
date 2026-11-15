"""Consumer regressions; synthetic fixtures never qualify native EDA results."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from global_bench.grt_candidate_library import export_library, load_library, RETENTION_POLICY
from grt_bench.archive import completed_outcomes
from grt_bench.common import read, records, sha, write
import test_grt_optimization as fixtures


class RetainedLibraryTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.OptimizationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.bootstrap()
        self.root, self.task = self.f.root, self.f.task
        self.serial = 0
        self.proofs = {}

    def retain(self, candidate, *, record=True, log=False):
        audit = self.f.archive.admit(candidate)
        evaluation = next(r for r in self.f.archive.show()["evaluated"] if r["candidate"] == candidate)
        outcomes = completed_outcomes(self.root, self.task, candidate, audit)
        proof = {"source_audit": audit, "evaluation": evaluation, "outcomes": outcomes,
                 "record_setter": record, "logs": [], "verification_scope": "authenticated historical reading only"}
        if log:
            jobs = [r for r in self.f.queue.rows() if json.loads(r["payload"])["candidate"] == candidate]
            job = jobs[0]["id"]
            blob = self.root / "control/trace-retention/log-blobs" / ("a" * 64)
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(b"Native OPT log\r\n")
            blob.chmod(0o444)
            proof["logs"].append({"path": "runs/" + job + "/output/run/60_opt/native.log", "job": job,
                                  "stage": "60_opt", "reused": False, "requested_jobs": [job],
                                  "blob": blob.relative_to(self.root).as_posix(),
                                  "file": {"bytes": blob.stat().st_size, "sha256": sha(blob), "mode": 0o444}})
        self.proofs[candidate] = proof
        write(self.root / "control/trace-retention/authorization.json", {"consumer_fixture": True}, replace=True)
        return proof

    def patches(self, *, fail=None):
        owner = self
        class History:
            def __init__(self, root, task):
                if fail:
                    raise ValueError(fail)
            def has(self, candidate):
                return candidate in owner.proofs
            def history(self, candidate):
                proof = owner.proofs[candidate]
                if isinstance(proof, BaseException):
                    raise proof
                return proof
        return patch("global_bench.grt_history_retention.RetainedHistory", History)

    def export(self):
        self.serial += 1
        self.output = self.root / "libraries" / str(self.serial)
        self.index = export_library(self.root, self.task, self.output)
        return {e["candidate"]: e for e in self.index["candidates"]}

    def reseal(self):
        artifacts = records(self.output)
        artifacts.pop("manifest.json")
        write(self.output / "manifest.json", {"schema_version": 1, "artifacts": artifacts}, replace=True)
        for path in self.output.rglob("*"):
            if path.is_file():
                path.chmod(0o444)

    def test_retained_exact_metrics_survive_absent_bulk_analysis(self):
        self.retain(self.f.baseline)
        for row in self.f.queue.rows():
            shutil.rmtree(self.root / "runs" / row["id"] / "output/analysis")
        with self.patches():
            entries = self.export()
        self.assertEqual(self.index["schema_version"], 2)
        self.assertEqual(self.index["retention_policy"], RETENTION_POLICY)
        self.assertEqual(entries[self.f.baseline]["retention"]["history"], "authenticated_retained")
        detail = read(self.output / self.f.baseline / "evaluation/metrics.json")
        self.assertEqual(detail["aggregate"], self.proofs[self.f.baseline]["evaluation"])
        load_library(self.output, self.task, self.f.archive.show())

    def test_present_corrupt_seal_does_not_fall_back_to_raw(self):
        self.retain(self.f.baseline)
        self.proofs[self.f.baseline] = ValueError("seal authentication failed")
        with self.patches(), self.assertRaisesRegex(ValueError, "seal authentication"):
            self.export()

    def test_bad_authorization_does_not_silently_disable_policy(self):
        self.retain(self.f.baseline)
        with self.patches(fail="invalid authorization"), self.assertRaisesRegex(ValueError, "invalid authorization"):
            self.export()

    def test_no_seal_fallback_requires_intact_independent_evidence(self):
        self.retain(self.f.baseline)
        candidate = self.f.child()
        self.f.receipts(candidate, [12] * 5)
        self.f.archive.ingest(candidate)
        row = next(r for r in self.f.queue.rows() if json.loads(r["payload"])["candidate"] == candidate)
        (self.root / "runs" / row["id"] / "verified/result.json").unlink()
        with self.patches(), self.assertRaises(FileNotFoundError):
            self.export()

    def test_source_audit_and_archive_aggregate_remain_strict(self):
        for key in ("evaluation", "source_audit"):
            self.proofs.clear()
            proof = self.retain(self.f.baseline)
            proof[key] = dict(proof[key], unexpected=True)
            with self.patches(), self.assertRaisesRegex(ValueError, "source/archive"):
                self.export()

    def test_nonrecord_keeps_source_diff_metrics_diagnostics_without_bulk_logs(self):
        self.retain(self.f.baseline)
        candidate = self.f.child()
        attempts = self.f.receipts(candidate, [12] * 5)
        self.f.archive.ingest(candidate)
        for attempt in attempts:
            (attempt / "output/large.log").write_bytes(b"x" * 10000)
        self.retain(candidate, record=False)
        with self.patches():
            entry = self.export()[candidate]
        folder = self.output / candidate
        self.assertEqual(entry["retention"]["native_logs"], "summary_only")
        self.assertTrue((folder / "from_parent.diff").read_text())
        self.assertEqual(read(folder / "logs/index.json"), [])
        self.assertTrue(read(folder / "evaluation/diagnostics.json")["cells"])
        self.assertEqual(set(records(folder / "code")), set(self.task["optimization"]["allowed_files"]))
        load_library(self.output, self.task, self.f.archive.show())

    def test_partial_failed_candidate_keeps_explicit_diagnostics_without_claimed_score(self):
        self.retain(self.f.baseline)
        candidate = self.f.child()
        attempts = self.f.receipts(candidate, skip_last=True, failed=True)
        for attempt in attempts:
            (attempt / "output/failed.log").write_text("Native failure diagnostics\n")
        with self.patches():
            entry = self.export()[candidate]
        self.assertEqual(entry["status"], "incomplete_or_failed")
        self.assertEqual(entry["retention"]["history"], "not_evaluated")
        self.assertEqual(entry["retention"]["native_logs"], "summary_only")
        detail = read(self.output / candidate / "evaluation/metrics.json")
        self.assertIsNone(detail["aggregate"])
        self.assertEqual(detail["verified_training_cells"], [])
        cells = read(self.output / candidate / "evaluation/cells.json")
        self.assertEqual(len(cells), 5)
        self.assertTrue(all(cell["state"] == "failed" and not cell["logs"] for cell in cells))
        load_library(self.output, self.task, self.f.archive.show())

    def test_new_unsealed_evaluation_uses_strict_native_fallback(self):
        self.retain(self.f.baseline)
        candidate = self.f.child()
        attempts = self.f.receipts(candidate, [12] * 5)
        self.f.archive.ingest(candidate)
        (attempts[0] / "output/native.log").write_bytes(b"Original unsealed log\n")
        with self.patches():
            entry = self.export()[candidate]
        self.assertEqual(entry["retention"]["history"], "strict_native")
        self.assertEqual(entry["retention"]["native_logs"], "available")
        self.assertEqual(len(read(self.output / candidate / "logs/index.json")), 1)
        load_library(self.output, self.task, self.f.archive.show())

    def test_record_logs_and_code_share_only_immutable_blobs_across_libraries(self):
        proof = self.retain(self.f.baseline, log=True)
        original = self.f.store / self.f.baseline / "tree" / fixtures.PARTITION
        original_mode = original.stat().st_mode
        with self.patches():
            self.export()
            first = self.output
            self.export()
        second = self.output
        code = Path(self.f.baseline) / "code" / fixtures.PARTITION
        self.assertEqual((first / code).stat().st_ino, (second / code).stat().st_ino)
        self.assertNotEqual(original.stat().st_ino, (second / code).stat().st_ino)
        self.assertEqual(original.stat().st_mode, original_mode)
        log = read(second / self.f.baseline / "logs/index.json")[0]
        shared = second / self.f.baseline / log["path"]
        self.assertEqual(shared.stat().st_ino, (self.root / proof["logs"][0]["blob"]).stat().st_ino)
        load_library(second, self.task, self.f.archive.show())

    def test_reused_upstream_logs_keep_actual_producer_and_consumer_cell(self):
        proof = self.retain(self.f.baseline, log=True)
        log = proof["logs"][0]
        producer = "b" * 64
        log["path"] = "runs/" + producer + "/output/run/30_placement/native.log"
        log.update(job=producer, stage="30_placement", reused=True)
        with self.patches():
            self.export()
        exported = read(self.output / self.f.baseline / "logs/index.json")[0]
        self.assertEqual(exported["job"], producer)
        self.assertTrue(exported["reused"])
        self.assertIn("reused-stage-sources/" + producer, exported["path"])
        self.assertIn("/30_placement/", exported["path"])
        load_library(self.output, self.task, self.f.archive.show())

    def test_mutable_or_changed_retained_log_blob_is_rejected(self):
        proof = self.retain(self.f.baseline, log=True)
        blob = self.root / proof["logs"][0]["blob"]
        blob.chmod(0o644)
        with self.patches(), self.assertRaisesRegex(ValueError, "not immutable"):
            self.export()
        blob.write_bytes(b"tampered")
        blob.chmod(0o444)
        with self.patches(), self.assertRaisesRegex(ValueError, "blob changed"):
            self.export()

    def test_original_log_mode_is_preserved_separately_from_immutable_blob(self):
        proof = self.retain(self.f.baseline, log=True)
        proof["logs"][0]["file"]["mode"] = 0o644  # The sealed original was writable; its published blob is not.
        with self.patches():
            self.export()
        load_library(self.output, self.task, self.f.archive.show())

    def test_summary_schema_and_bulk_inventory_stay_strict_if_resealed(self):
        self.retain(self.f.baseline, record=False)
        with self.patches():
            self.export()
        folder = self.output / self.f.baseline
        (folder / "logs/injected.log").write_bytes(b"Not retained")
        self.reseal()
        with self.assertRaisesRegex(ValueError, "log inventory"):
            load_library(self.output, self.task, self.f.archive.show())

    def test_resealed_raw_metric_changes_cannot_keep_original_score(self):
        self.retain(self.f.baseline)
        candidate = self.f.child()
        self.f.receipts(candidate, [12] * 5)
        self.f.archive.ingest(candidate)
        self.retain(candidate, record=False)
        with self.patches():
            self.export()
        path = self.output / candidate / "evaluation/metrics.json"
        detail = read(path)
        detail["verified_training_cells"][0]["metrics"][self.task["metrics"][0]] *= 2
        from global_bench.grt_candidate_library import raw_statistics
        detail["raw_per_case"] = raw_statistics(self.task, detail["verified_training_cells"])
        write(path, detail, replace=True)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "case/seed metrics"):
            load_library(self.output, self.task, self.f.archive.show())

    def test_readonly_blob_permissions_are_part_of_new_reading_contract(self):
        self.retain(self.f.baseline)
        with self.patches():
            self.export()
        (self.output / "README.md").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "must stay immutable"):
            load_library(self.output, self.task, self.f.archive.show())

    def test_legacy_uninitialized_export_remains_schema_one(self):
        self.export()
        self.assertEqual(self.index["schema_version"], 1)
        self.assertNotIn("retention", self.index["candidates"][0])
        load_library(self.output, self.task, self.f.archive.show())


if __name__ == "__main__":
    unittest.main()


def test_real_history_seal_and_compact_library_after_raw_retirement(tmp_path):
    from test_grt_history_retention import SyntheticCampaign, authorization
    from global_bench.grt_history_retention import RetainedHistory
    fixture = SyntheticCampaign(tmp_path / "real-library", artifact_bytes=16)
    fixture.branches()
    history = RetainedHistory.initialize(fixture.root, fixture.task, authorization=authorization(fixture))
    for name in ("baseline", "improved", "weak", "tie", "derived"):
        history.seal(fixture.names[name], "terminal-" + name)
    # Native admission must still fail, but the authenticated historical reader remains useful.
    weak = fixture.names["weak"]
    for job in fixture.jobs[weak]:
        attempt = fixture.root / "runs" / job
        (attempt / "output/analysis/metrics.json").unlink()
        for log in (attempt / "output/run").rglob("*.log"):
            log.unlink()
    output = fixture.root / "libraries/compact-real"
    index = export_library(fixture.root, fixture.task, output)
    entries = {entry["candidate"]: entry for entry in index["candidates"]}
    assert entries[weak]["retention"]["native_logs"] == "summary_only"
    derived = fixture.names["derived"]
    logs = read(output / derived / "logs/index.json")
    assert any(log["reused"] and "/30_placement/" in log["path"] for log in logs)
    assert all(log["job"] is not None for log in logs)
    assert entries[derived]["retention"]["best_so_far_record_setter"] is True
    load_library(output, fixture.task, fixture.archive.show())
