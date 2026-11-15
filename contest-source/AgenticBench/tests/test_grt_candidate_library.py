"""Candidate reading exports use synthetic receipts, never native qualification."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_bench.grt_candidate_library import export_library, load_library, main
from global_bench.grt_round_prompt import round_prompt
from grt_bench.common import read, records, sha, write
from grt_bench.contract import matrix
import test_grt_optimization as fixtures


class CandidateLibraryTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.OptimizationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.bootstrap()
        self.serial = 0

    def export(self, **kwargs):
        self.serial += 1
        self.output = self.f.root / "libraries" / str(self.serial)
        self.index = export_library(self.f.root, self.f.task, self.output, **kwargs)
        return {e["candidate"]: e for e in self.index["candidates"]}

    def render(self, **kwargs):
        f = self.f
        required = {f.baseline, *(r["candidate"] for r in f.archive.show()["frontier"])}
        outcomes = [json.loads(r["result"]) for r in f.queue.rows()
                    if json.loads(r["payload"])["candidate"] in required
                    and json.loads(r["payload"])["split"] == "train"]
        budget = {"round_number": 1, "round_limit": 10, "evaluation_limit": 10,
                  "evaluations_used": 0, "round_reserved": True}
        return round_prompt(f.task, f.archive.show(), budget, outcomes, library=self.output, **kwargs)

    def test_cumulative_and_parent_diffs_and_measured_deltas_are_distinct(self):
        f = self.f
        parent = f.child()
        f.receipts(parent, [8, 8, 10, 12, 10]); f.archive.ingest(parent)
        child = f.child(parent, native=True)
        f.receipts(child, [7, 7, 11, 11, 10]); f.archive.ingest(child)
        entries = self.export()
        change = entries[child]["changes"]
        self.assertEqual(change["from_parent"]["changed_files"], [fixtures.NATIVE])
        self.assertEqual(change["from_reference"]["changed_files"], sorted([fixtures.PARTITION, fixtures.NATIVE]))
        self.assertIn("value: 1 -> 2", change["from_reference"]["summary"])
        self.assertNotIn("value: 1 -> 2", change["from_parent"]["summary"])
        folder = self.output / child
        self.assertNotIn(fixtures.PARTITION, (folder / "from_parent.diff").read_text())
        self.assertIn(fixtures.PARTITION, (folder / "from_reference.diff").read_text())
        row = {r["candidate"]: r for r in f.archive.show()["evaluated"]}
        score = entries[child]["evaluation_summary"]
        for metric, value in row[child]["vector"].items():
            self.assertEqual(score["delta_from_reference"][metric], value - row[f.baseline]["vector"][metric])
            self.assertEqual(score["delta_from_parent"][metric], value - row[parent]["vector"][metric])
        self.assertNotEqual(score["delta_from_parent"], score["delta_from_reference"])
        bundle = self.render()
        visible = next(e for e in bundle["context"]["candidates"] if e["candidate"] == child)
        self.assertEqual(visible["evaluation_summary"], score)
        self.assertNotIn("verified_training_cells", bundle["prompt"])
        self.assertNotIn(str(f.root), bundle["prompt"])

    def test_source_is_readable_before_checkout_and_is_an_independent_copy(self):
        f = self.f
        entries = self.export()
        self.assertEqual(f.archive.show()["selections"], [])
        self.assertEqual(entries[f.baseline]["changes"]["from_reference"]["summary"], "No source changes.")
        code = self.output / f.baseline / "code"
        self.assertEqual(set(records(code)), set(f.task["optimization"]["allowed_files"]))
        original = f.store / f.baseline / "tree" / fixtures.PARTITION
        copied = code / fixtures.PARTITION
        self.assertEqual(copied.read_bytes(), original.read_bytes())
        self.assertNotEqual(copied.stat().st_ino, original.stat().st_ino)
        self.assertEqual(copied.stat().st_mode & 0o222, 0)
        selection = f.archive.select(f.baseline, "Inspect copied code first")["id"]
        workspace = f.root / "edit-later"
        f.archive.checkout(workspace, selection)
        (workspace / fixtures.PARTITION).write_text("value = 99\n")
        self.assertEqual(copied.read_text(), "value = 1\n")
        load_library(self.output, f.task, f.archive.show())  # Selecting alone does not stale the view.

    def test_logs_keep_bytes_and_exclude_heldout_and_host_controls(self):
        f = self.f
        job = f.queue.rows()[0]
        attempt = f.root / "runs" / job["id"]
        logs = ("output/flow-0.log", "output/run/30_placement/run.log", "output/grt/timing.rpt",
                "execution/frontend/container.log", "verification/container.log")
        native = b"Original line\r\nPPA report\n"
        for name in logs:
            path = attempt / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(native)
        for name in ("execution/environment.json", "control/hidden.log", "command.json"):
            path = attempt / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("DO_NOT_EXPORT_CONTROL_SECRET")
        for heldout in f.receipts(f.baseline, split="test"):
            (heldout / "output/heldout.log").write_text("DO_NOT_EXPORT_HELDOUT_RESULT")
        self.export()
        folder = self.output / f.baseline
        inventory = read(folder / "logs/index.json")
        self.assertEqual(len(inventory), len(logs))
        for log in inventory:
            self.assertEqual((folder / log["path"]).read_bytes(), native)
            self.assertEqual(sha(folder / log["path"]), log["sha256"])
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"DO_NOT_EXPORT", path.read_bytes())
                self.assertNotIn("held-out", path.as_posix())
        self.assertEqual(len(read(folder / "evaluation/cells.json")), 6)

    def test_dominated_and_infeasible_candidates_remain_visible(self):
        f = self.f
        dominated = f.child(); f.receipts(dominated, [12, 12, 12, 12, 10]); f.archive.ingest(dominated)
        infeasible = f.child(); f.receipts(infeasible, [1, 1, 1, 1, 12]); f.archive.ingest(infeasible)
        entries = self.export()
        self.assertEqual(entries[dominated]["status"], "dominated")
        self.assertFalse(entries[dominated]["selectable"])
        self.assertTrue(entries[dominated]["evaluation_summary"]["power_feasible"])
        self.assertEqual(entries[infeasible]["status"], "infeasible")
        self.assertFalse(entries[infeasible]["selectable"])
        self.assertEqual(entries[infeasible]["evaluation_summary"]["power_failed_cases"], ["bp", "bp_fe"])
        self.assertEqual(set(self.render()["context"]["frontier"]), {f.baseline})

    def test_internal_native_aliases_export_one_canonical_log(self):
        f = self.f
        attempt = f.root / "runs" / f.queue.rows()[0]["id"]
        native = attempt / "output/run/60_opt/iterations/01/global/passive/sta.log"
        native.parent.mkdir(parents=True)
        native.write_bytes(b"Native STA output\n")
        stage = attempt / "output/run/60_opt"
        (stage / "assembly").symlink_to("iterations/01/global", target_is_directory=True)
        (stage / "final.log").symlink_to("iterations/01/global/passive/sta.log")
        self.export()
        folder = self.output / f.baseline
        logs = read(folder / "logs/index.json")
        self.assertEqual(len(logs), 1)
        self.assertTrue(logs[0]["path"].endswith(native.relative_to(attempt).as_posix()))
        self.assertEqual((folder / logs[0]["path"]).read_bytes(), native.read_bytes())
        self.assertFalse(any(p.is_symlink() for p in self.output.rglob("*")))
        load_library(self.output, f.task, f.archive.show())

    def test_internal_alias_cannot_export_host_controls_or_missing_logs(self):
        f = self.f
        attempt = f.root / "runs" / f.queue.rows()[0]["id"]
        hidden = attempt / "control/secret.log"
        hidden.write_text("DO_NOT_EXPORT")
        link = attempt / "output/native.log"
        for target in ("../control/secret.log", "missing.log"):
            with self.subTest(target=target):
                link.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "Symlink"):
                    self.export()
                link.unlink()

    def test_terminal_partial_failure_has_logs_but_no_fabricated_score(self):
        f = self.f
        candidate = f.child()
        attempts = f.receipts(candidate, skip_last=True, failed=True)
        for attempt in attempts: (attempt / "output/failed.log").write_text("Placement failed\n")
        entries = self.export()
        entry = entries[candidate]
        self.assertEqual(entry["status"], "incomplete_or_failed")
        self.assertIsNone(entry["evaluation_summary"])
        folder = self.output / candidate
        detail = read(folder / "evaluation/metrics.json")
        self.assertIsNone(detail["aggregate"])
        self.assertIsNone(detail["raw_per_case"])
        self.assertEqual(detail["verified_training_cells"], [])
        cells = read(folder / "evaluation/cells.json")
        self.assertEqual(len(cells), 5)
        self.assertTrue(all(c["state"] == "failed" and c["logs"] for c in cells))
        self.assertIsNone(next(e for e in self.render()["context"]["candidates"]
                               if e["candidate"] == candidate)["evaluation_summary"])

    def test_unevaluated_candidate_and_absent_logs_remain_explicit(self):
        f = self.f
        candidate = f.child()
        entry = self.export()[candidate]
        self.assertEqual(entry["status"], "not_evaluated")
        self.assertIsNone(entry["evaluation_summary"])
        self.assertEqual(read(self.output / candidate / "logs/index.json"), [])
        self.assertEqual(read(self.output / candidate / "evaluation/cells.json"), [])

    def test_running_or_duplicate_training_jobs_are_rejected(self):
        f = self.f
        child = f.child()
        payload = matrix(f.task, child)[0]
        job = f.queue.submit(payload, f.resources)
        with self.assertRaisesRegex(ValueError, "Finish or reconcile"): self.export()
        f.queue.cancel(job)
        other = f.queue.submit(payload, dict(f.resources, cpu=2))
        f.queue.cancel(other)
        with self.assertRaisesRegex(ValueError, "Duplicate training cell"): self.export()

    def test_symlinked_attempt_or_native_log_is_rejected(self):
        f = self.f
        attempt = f.root / "runs" / f.queue.rows()[0]["id"]
        outside = f.root / "outside.log"; outside.write_text("Outside data")
        link = attempt / "output/native.log"; link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink"): self.export()
        link.unlink()
        renamed = f.root / "relocated-run"; attempt.rename(renamed); attempt.symlink_to(renamed, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"): self.export()

    def test_changed_evidence_and_stale_archive_are_rejected(self):
        f = self.f
        self.export()
        f.child()
        with self.assertRaisesRegex(ValueError, "snapshot"): load_library(self.output, f.task, f.archive.show())
        self.export()
        source = self.output / f.baseline / "code" / fixtures.PARTITION
        source.chmod(0o644); source.write_text("value = 99\n")
        with self.assertRaisesRegex(ValueError, "Artifact changed"): self.render()

    def test_resealed_false_scores_are_rejected_against_archive(self):
        f = self.f
        self.export()
        index = copy.deepcopy(self.index)
        index["candidates"][0]["evaluation_summary"]["objectives"]["timing"] = 0.1
        write(self.output / "index.json", index, replace=True)
        write(self.output / f.baseline / "candidate.json", index["candidates"][0], replace=True)
        artifacts = records(self.output); artifacts.pop("manifest.json")
        write(self.output / "manifest.json", {"schema_version": 1, "artifacts": artifacts}, replace=True)
        with self.assertRaisesRegex(ValueError, "summary differs"): self.render()

    def test_readable_scope_is_bound_to_task_even_if_resealed(self):
        self.export()
        path = self.output / "task-context.json"
        details = read(path)
        self.assertEqual(details["allowed_files"], self.f.task["optimization"]["allowed_files"])
        self.assertEqual({c["id"] for c in details["evaluation_cases"]}, {"bp", "bp_fe"})
        self.assertNotIn("prompt", details)
        self.assertEqual(path.stat().st_mode & 0o222, 0)
        details["allowed_files"].append(fixtures.PROTECTED)
        write(path, details, replace=True)
        artifacts = records(self.output); artifacts.pop("manifest.json")
        write(self.output / "manifest.json", {"schema_version": 1, "artifacts": artifacts}, replace=True)
        with self.assertRaisesRegex(ValueError, "context differs from frozen task"):
            self.render()

    def test_legacy_library_keeps_exact_scope_inline(self):
        self.export()
        (self.output / "task-context.json").unlink()
        artifacts = records(self.output); artifacts.pop("manifest.json")
        write(self.output / "manifest.json", {"schema_version": 1, "artifacts": artifacts}, replace=True)
        bundle = self.render()
        self.assertNotIn("task_details", bundle["agent_context"])
        self.assertEqual(bundle["agent_context"]["allowed_files"],
                         self.f.task["optimization"]["allowed_files"])

    def test_extra_files_and_overlapping_agent_mounts_are_rejected(self):
        self.export()
        for workspace in ("/workspace/candidates", "/workspace", "/workspace/candidates/edit"):
            with self.subTest(workspace=workspace), self.assertRaisesRegex(ValueError, "must be separate"):
                self.render(workspace=workspace)
        (self.output / "extra.txt").write_text("Unexpected material")
        with self.assertRaisesRegex(ValueError, "Unexpected reading-library files"): self.render()

    def test_cli_creates_a_new_library_without_mutating_source_or_runs(self):
        f = self.f
        before = records(f.root / "runs")
        task = f.root / "task-export.json"; write(task, f.task)
        output = f.root / "cli-library"
        args = ["library", "export", str(f.root), str(task), str(output)]
        with patch.object(sys, "argv", args): main()
        load_library(output, f.task, f.archive.show())
        self.assertEqual(before, records(f.root / "runs"))
        with patch.object(sys, "argv", args), self.assertRaises(FileExistsError): main()
        with self.assertRaisesRegex(ValueError, "separate"):
            export_library(f.root, f.task, f.root / "runs/bad-export")


if __name__ == "__main__": unittest.main()
