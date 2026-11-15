"""Timing-accounting tests; synthetic fixtures are not native speed evidence."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.common import read, sha, write
from grt_bench.contract import matrix
from scripts.benchmark_grt_matrix import STAGES, freeze_timing_candidate, render, summarize
from test_grt_matrix import task_fixture


class TimingTests(unittest.TestCase):
    def test_fresh_candidate_uses_external_workspace_and_preserves_snapshot(self):
        from grt_bench.candidates import freeze, verify_snapshot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "run.sh").write_text("#!/bin/sh\nexit 0\n")
            (source / "run.sh").chmod(0o755)
            (source / "alias").symlink_to("run.sh")
            parent = freeze(source, root / "candidates")
            before = verify_snapshot(root / "candidates" / parent)
            trials = []
            for name in ("trial-one", "trial-two"):
                output = root / name
                output.mkdir()
                child = freeze_timing_candidate(root, parent, output, name)
                after = verify_snapshot(root / "candidates" / child)
                self.assertEqual(before["files"], after["files"])
                self.assertEqual(after["scope"]["changes"], [])
                self.assertEqual(after["parent"], parent)
                (output / "candidate-source/run.sh").write_text("changed workspace")
                self.assertEqual(verify_snapshot(root / "candidates" / parent), before)
                self.assertEqual(verify_snapshot(root / "candidates" / child), after)
                trials.append(child)
            self.assertEqual(len({parent, *trials}), 3)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root, self.task, self.candidate = Path(self.tmp.name), task_fixture(), "c" * 64
        self.rows = []
        for index, cell in enumerate(matrix(self.task, self.candidate)):
            job = str(index)
            folder = self.root / "runs" / job
            started = 1000 + index * 10
            write(folder / "control/plan.json", dict(cell, frontend_prefix="/inputs/frontend"))
            write(folder / "output/analysis/manifest.json", {"fixture": True})
            result = dict(cell, status="verified", evidence={
                "plan_sha256": sha(folder / "control/plan.json"),
                "analysis_manifest_sha256": sha(folder / "output/analysis/manifest.json")},
                accounting={"verification": {"wall_seconds": 1}})
            write(folder / "result.json", result)
            stages = {s: {"status": "success", "runtime_seconds": .5,
                "started_at": datetime.fromtimestamp(started + 2, timezone.utc).isoformat()} for s in STAGES}
            stages.update({s: {"status": "success", "source_manifest_sha256": "f" * 64,
                "runtime_seconds": 99999} for s in ("input", "floorplan")})
            write(folder / "output/run/run.json", {"frontend_prefix": {"boundary": "partition"}, "stages": stages})
            write(folder / "output/execution.json", {"commands": [
                {"command": ["run"], "wall_seconds": 3},
                {"command": ["global-route"], "wall_seconds": 1}]})
            write(folder / "output/analysis/metrics.json", {"runtime_seconds": 1})
            self.rows.append({"id": job, "state": "succeeded", "payload": json.dumps(cell),
                "submitted": 1000, "started": started, "finished": started + 8})
        self.jobs = [r["id"] for r in self.rows]

    def summary(self, rows=None):
        return summarize(self.root, self.task, self.candidate, self.rows if rows is None else rows, self.jobs)

    def test_measured_envelope_not_sum_of_cells_or_historical_prefix(self):
        report = self.summary()
        self.assertEqual(report["batch_submit_to_verified_seconds"], 58)
        self.assertEqual(report["first_partition_to_last_verified_seconds"], 56)
        self.assertAlmostEqual(report["average_busy_workers"], 48 / 58)
        self.assertFalse(report["historical_prefix_runtime_included"])
        self.assertTrue(all(c["partition_to_verified_seconds"] == 6 for c in report["cells"]))
        self.assertTrue(all("input" not in c["stage_seconds"] for c in report["cells"]))
        self.assertIn("Minimum (min)", render(report))

    def test_partial_failed_duplicate_and_wrong_identity_are_not_timings(self):
        variants = [self.rows[:-1], self.rows + self.rows[:1]]
        for field, value in (("state", "failed"), ("started", 999), ("finished", 1001)):
            rows = copy.deepcopy(self.rows); rows[0][field] = value; variants.append(rows)
        rows = copy.deepcopy(self.rows)
        payload = json.loads(rows[0]["payload"]); payload["candidate"] = "d" * 64
        rows[0]["payload"] = json.dumps(payload); variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(ValueError): self.summary(rows)

    def test_analysis_only_replay_is_rejected(self):
        path = self.root / "runs/0/result.json"
        result = read(path); result["physical_replay"] = {"source_job": "old"}
        write(path, result, replace=True)
        with self.assertRaisesRegex(ValueError, "Rescoring"): self.summary()

    def test_phased_timing_requires_handoff_and_gpu_release(self):
        for row in self.rows:
            folder = self.root / "runs" / row["id"]
            plan = read(folder / "control/plan.json")
            plan["execution_mode"] = "gpu-frontend-cpu-backend"
            write(folder / "control/plan.json", plan, replace=True)
            result = read(folder / "result.json")
            result["evidence"]["plan_sha256"] = sha(folder / "control/plan.json")
            write(folder / "result.json", result, replace=True)
            write(folder / "output/gpu_frontend.json", {"fixture": True})
            write(folder / "output/execution.json", {"execution_mode": plan["execution_mode"],
                "frontend_handoff_sha256": sha(folder / "output/gpu_frontend.json"), "commands": [
                    {"command": ["run", "--to", "placement"], "wall_seconds": 1},
                    {"command": ["run", "--from", "pdn", "--resume"], "wall_seconds": 2},
                    {"command": ["global-route"], "wall_seconds": 1}]}, replace=True)
            write(folder / "resource-release.json", {"allocation": {"gpus": []}, "evidence": {"terminated": True}})
        self.assertEqual(self.summary()["batch_submit_to_verified_seconds"], 58)
        write(self.root / "runs/0/resource-release.json", {"allocation": {"gpus": ["0"]},
            "evidence": {"terminated": True}}, replace=True)
        with self.assertRaisesRegex(ValueError, "release"): self.summary()

    def test_reused_partition_and_missing_prefix_proof_are_rejected(self):
        path = self.root / "runs/0/output/run/run.json"
        original = read(path)
        value = copy.deepcopy(original)
        value["stages"]["partition"]["source_manifest_sha256"] = "f" * 64
        write(path, value, replace=True)
        with self.assertRaisesRegex(ValueError, "freshly executed"): self.summary()
        value = copy.deepcopy(original)
        value["stages"]["input"].pop("source_manifest_sha256")
        write(path, value, replace=True)
        with self.assertRaisesRegex(ValueError, "prefix reuse"): self.summary()

    def test_changed_plan_and_nonfinite_runtime_are_rejected(self):
        path = self.root / "runs/0/output/analysis/metrics.json"
        path.write_text('{"runtime_seconds":NaN}')
        with self.assertRaisesRegex(ValueError, "nonfinite"): self.summary()
        write(path, {"runtime_seconds": 1}, replace=True)
        plan = self.root / "runs/0/control/plan.json"
        plan.write_text('{}')
        with self.assertRaisesRegex(ValueError, "evidence changed"): self.summary()
