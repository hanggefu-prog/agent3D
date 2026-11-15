"""Task/archive integration with synthetic host receipts, not native EDA qualification."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.archive import Archive
from grt_bench.candidates import freeze, materialize
from grt_bench.common import digest, evaluator_identity, read, records, sha, write
from grt_bench.contract import matrix, METRICS, OBJECTIVES, validate
from grt_bench.optimization import define_task, edit_policy, verify_candidate
from grt_bench.queue import Queue
from test_grt_matrix import task_fixture

PARTITION = "Place-3D/src/partition/lol/methods/tp_gnn.py"
PLACEMENT = "Place-3D/src/placement/concurrent_dreamplace.py"
NATIVE = "OpenROAD-3D/openroad/src/rsz/src/RepairSetup.cc"
PROTECTED = "open3dbench/runner.py"
PROMPT = "---\ncomponents: [partition, placement, timing_opt]\n---\nImprove generalizable post-GRT PPA.\n"


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        for name in (PARTITION, PLACEMENT, NATIVE, PROTECTED):
            path = self.source / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("int value = 1;\n" if name == NATIVE else "value = 1\n")
        self.store = self.root / "candidates"
        self.baseline = freeze(self.source, self.store)
        base = task_fixture(); base["evaluator_sha256"] = evaluator_identity()
        base["cases"].append(dict(base["cases"][0], id="held-out", split="test"))
        self.task = define_task(base, self.baseline, PROMPT, self.store)
        self.archive = Archive(self.root, self.task)
        write(self.root / "control/tasks" / (validate(self.task) + ".json"), self.task)
        self.queue = Queue(self.root / "control/queue.sqlite")
        self.queue.register_host("fixture", {"cpu": 8, "memory_gb": 16, "storage_gb": 40, "gpus": [], "build_slots": 1})
        self.resources = {"cpu": 1, "memory_gb": 1, "storage_gb": 1, "gpu_count": 0, "build_slots": 0}
        self.serial = 0

    def receipts(self, candidate, values=None, *, skip_last=False, failed=False, split="train"):
        values = values or [10.0] * 5
        payloads = matrix(self.task, candidate, split)
        if skip_last: payloads = payloads[:-1]
        attempts = []
        for payload in payloads:
            job = self.queue.submit(payload, self.resources)
            lease = self.queue.claim("fixture")
            self.assertEqual(lease["id"], job)
            attempt = self.root / "runs" / job; attempts.append(attempt)
            write(attempt / "control/task.json", self.task)
            write(attempt / "control/plan.json", dict(payload, output="/output"))
            analysis = attempt / "output/analysis"
            write(analysis / "metrics.json", dict(zip(METRICS, values)))
            write(analysis / "manifest.json", {"artifacts": records(analysis)})
            audit = self.archive.admit(candidate)
            write(attempt / "candidate-verification.json", audit)
            verified = {**payload, "status": "verified", "metrics": dict(zip(METRICS, values)),
                        "evidence": {"analysis_manifest_sha256": sha(analysis / "manifest.json"),
                                     "plan_sha256": sha(attempt / "control/plan.json")}}
            write(attempt / "verified/result.json", verified)
            result = {**verified, "accounting": {"fixture": True},
                      "candidate_verification_sha256": sha(attempt / "candidate-verification.json")}
            write(attempt / "result.json", result)
            self.queue.finish(job, lease["token"], result, not failed)
        return attempts

    def bootstrap(self):
        self.receipts(self.baseline)
        self.archive.ingest(self.baseline)

    def child(self, parent=None, *, native=False, selection=None):
        parent = parent or self.baseline
        if selection is None:
            selection = self.archive.select(parent, "Explore a timing/power tradeoff")["id"]
        self.serial += 1
        source = self.root / ("edit-" + str(self.serial))
        receipt = self.archive.checkout(source, selection)
        self.assertEqual(receipt["parent"], parent)
        relative = NATIVE if native else PARTITION
        (source / relative).write_text("int value = %d;\n" % (self.serial + 1) if native
                                      else "value = %d\n" % (self.serial + 1))
        return self.archive.freeze(source, selection)

    def test_prompt_scope_is_frozen_and_narrowable(self):
        prompt = "---\ncomponents: [partition]\nfiles: [" + PARTITION + "]\n---\nImprove cuts.\n"
        task = define_task(task_fixture(), self.baseline, prompt, self.store)
        self.assertEqual(edit_policy(task)["allowed_paths"], [PARTITION])
        self.assertFalse(edit_policy(task)["native"])
        for bad in (prompt.replace(PARTITION, PROTECTED), prompt.replace("[partition]", "[routing]"),
                    "Change whatever you like.", prompt.replace("components:", "permissions:")):
            with self.subTest(prompt=bad), self.assertRaises(ValueError):
                define_task(task_fixture(), self.baseline, bad, self.store)
        for field, value in (("epsilon", .001), ("metrics", ["power_w"])):
            bad = copy.deepcopy(self.task); bad[field] = value
            with self.assertRaises(ValueError): validate(bad)
        bad = copy.deepcopy(self.task); bad["optimization"]["allowed_files"].append(PROTECTED)
        with self.assertRaises(ValueError): validate(bad)

    def test_scope_protected_creation_deletion_mode_and_syntax(self):
        self.bootstrap()
        selection = self.archive.select(self.baseline, "Narrow legal edit")["id"]
        operations = [lambda s: (s / PROTECTED).write_text("value = 3\n"),
                      lambda s: (s / PARTITION).unlink(),
                      lambda s: (s / PARTITION).chmod(0o755),
                      lambda s: (s / PARTITION).write_text("def broken(\n"),
                      lambda s: (s / PARTITION).with_name("extra.py").write_text("value = 1\n")]
        for i, operation in enumerate(operations):
            source = materialize(self.store / self.baseline, self.root / ("bad-" + str(i)))
            operation(source)
            with self.subTest(operation=i), self.assertRaises((ValueError, SyntaxError)):
                self.archive.freeze(source, selection)

    def test_noop_unrelated_root_and_unsafe_ancestor_are_rejected(self):
        self.bootstrap()
        selection = self.archive.select(self.baseline, "Check ancestry")["id"]
        with self.assertRaisesRegex(ValueError, "no changes"):
            self.archive.freeze(self.source, selection)
        (self.source / PARTITION).write_text("value = 2\n")
        other_root = freeze(self.source, self.store)
        with self.assertRaises(ValueError): verify_candidate(self.task, self.store, other_root)
        legacy_policy = edit_policy(self.task); legacy_policy.pop("task_digest")
        ancestor = freeze(self.source, self.store, parent=self.baseline, policy=legacy_policy)
        (self.source / PARTITION).write_text("value = 3\n")
        child = freeze(self.source, self.store, parent=ancestor, policy=edit_policy(self.task))
        with self.assertRaisesRegex(ValueError, "policy differs"):
            verify_candidate(self.task, self.store, child)

    def test_paired_ratios_then_seed_and_case_weights(self):
        from grt_bench.pareto import aggregate
        task = copy.deepcopy(self.task)
        task["cases"][0]["weight"] = 2
        refs = [{**p, "status": "verified", "metrics": dict.fromkeys(METRICS, float(i + 1))}
                for i, p in enumerate(matrix(task, self.baseline))]
        rows = [{**p, "candidate": "c" * 64, "metrics": dict(p["metrics"], power_w=12)} for p in refs]
        summary = aggregate(task, "c" * 64, rows, refs)
        expected_first = sum(13 / denominator for denominator in (2, 3, 4)) / 3
        expected_second = sum(13 / denominator for denominator in (5, 6, 7)) / 3
        self.assertAlmostEqual(summary["metric_ratios"]["power_w"], (2 * expected_first + expected_second) / 3)
        self.assertNotIn("power_w", summary["vector"])
        self.assertFalse(summary["feasible"])
        self.assertEqual(summary["vector"]["overflow"], 1.0)
        self.assertGreater(summary["per_case"]["bp_fe"]["power_w"]["stdev"], 0)

    def test_full_ancestry_policy_and_native_receipt_checked(self):
        self.bootstrap()
        child = self.child(native=True)
        self.assertTrue(self.archive.admit(child)["native_build_required"])
        self.receipts(child, [8] * 5); self.archive.ingest(child)
        grandchild = self.child(child)
        audit = self.archive.admit(grandchild)
        self.assertEqual(audit["ancestry"], [grandchild, child, self.baseline])
        self.assertTrue(audit["native_build_required"])
        for field in ("policy", "scope", "parent"):
            data = read(self.store / grandchild / "candidate.json")
            if field == "policy": data[field]["allowed_paths"].append(PROTECTED)
            elif field == "scope": data[field]["native_build_required"] = False
            else: data[field] = "f" * 64
            forged = digest(data)
            import shutil
            shutil.copytree(self.store / grandchild / "tree", self.store / forged / "tree")
            write(self.store / forged / "candidate.json", data)
            with self.subTest(field=field), self.assertRaises((ValueError, FileNotFoundError)):
                verify_candidate(self.task, self.store, forged)
        # A task with the same files but different prose cannot import this lineage.
        other = copy.deepcopy(self.task); other["optimization"]["prompt"] += "Prefer power.\n"
        with self.assertRaises(ValueError): verify_candidate(other, self.store, grandchild)

    def test_source_only_freeze_cannot_bypass_parent_selection_or_worker_admission(self):
        (self.source / PARTITION).write_text("value = 3\n")
        candidate = freeze(self.source, self.store, parent=self.baseline, policy=edit_policy(self.task))
        self.assertEqual(verify_candidate(self.task, self.store, candidate)["status"], "legal")
        with self.assertRaisesRegex(ValueError, "parent selection"):
            self.archive.admit(candidate)
        from grt_bench.worker import evaluate
        with self.assertRaisesRegex(ValueError, "parent selection"):
            evaluate({"payload": matrix(self.task, candidate)[0]}, {"root": str(self.root)}, lambda: None)
        from grt_bench.__main__ import main
        args = ["grt", "submit", str(self.queue.path), str(self.root / "control/tasks" / (validate(self.task) + ".json")),
                candidate, "unused.json", "--root", str(self.root)]
        with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "parent selection"):
            main()
        self.assertEqual(self.queue.rows(), [])

    def test_tradeoffs_equivalent_candidates_history_and_parallel_ingest(self):
        self.bootstrap()
        children = [self.child() for _ in range(3)]
        for candidate, values in zip(children, ([8, 9, 10, 12, 10], [8, 9, 10, 12, 10], [12, 12, 12, 12, 10])):
            self.receipts(candidate, values)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(self.archive.ingest, children))
        state = Archive(self.root, self.task).show()
        self.assertEqual(len(state["evaluated"]), 4)
        self.assertEqual({r["candidate"] for r in state["frontier"]}, {self.baseline, *children[:2]})
        self.assertFalse(self.archive.ingest(children[2])["non_dominated"])
        self.assertTrue(self.archive.ingest(children[0])["non_dominated"])
        with self.assertRaisesRegex(ValueError, "current training frontier"):
            self.archive.select(children[2], "Dominated")

    def test_selected_parent_remains_valid_for_inflight_work(self):
        self.bootstrap()
        selection = self.archive.select(self.baseline, "Start before another job finishes")["id"]
        improved = self.child(); self.receipts(improved, [8] * 5)
        result = self.archive.ingest(improved)
        self.assertEqual(result["now_dominated"], [self.baseline])
        with self.assertRaises(ValueError): self.archive.select(self.baseline, "Too late")
        in_flight = self.child(selection=selection)
        self.archive.admit(in_flight)

    def test_over_power_candidate_is_recorded_but_cannot_be_a_parent(self):
        self.bootstrap()
        child = self.child()
        self.receipts(child, [1, 1, 1, 1, 11.1])
        receipt = self.archive.ingest(child)
        self.assertFalse(receipt["feasible"])
        self.assertFalse(receipt["non_dominated"])
        state = self.archive.show()
        self.assertEqual(len(state["evaluated"]), 2)
        self.assertEqual([r["candidate"] for r in state["frontier"]], [self.baseline])
        with self.assertRaisesRegex(ValueError, "current training frontier"):
            self.archive.select(child, "Cannot bypass power cap with better timing")

    def test_power_constraint_stays_bound_to_baseline_not_parent(self):
        self.bootstrap()
        parent = self.child(); self.receipts(parent, [8, 8, 8, 8, 11])
        self.assertTrue(self.archive.ingest(parent)["feasible"])
        child = self.child(parent); self.receipts(child, [7, 7, 7, 7, 12])
        self.assertFalse(self.archive.ingest(child)["feasible"])
        from grt_bench.pareto import aggregate
        results = [json.loads(r["result"]) for r in self.queue.rows()]
        with self.assertRaisesRegex(ValueError, "frozen baseline"):
            aggregate(self.task, child, [r for r in results if r["candidate"] == child],
                      [r for r in results if r["candidate"] == parent])

    def test_missing_failed_and_heldout_matrices_cannot_enter_frontier(self):
        self.receipts(self.baseline, skip_last=True)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.archive.ingest(self.baseline)
        self.assertEqual(self.archive.show()["frontier"], [])

    def test_failed_matrix_is_not_penalized_into_a_pareto_point(self):
        self.receipts(self.baseline, failed=True)
        with self.assertRaisesRegex(ValueError, "must succeed"):
            self.archive.ingest(self.baseline)

    def test_heldout_outcomes_do_not_drive_the_archive(self):
        self.receipts(self.baseline, split="test")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.archive.ingest(self.baseline)
        self.assertEqual(self.archive.show()["frontier"], [])

    def test_receipt_and_raw_evidence_tampering_rejected(self):
        attempts = self.receipts(self.baseline)
        attempt = attempts[0]
        path = attempt / "result.json"; original = path.read_bytes()
        result = read(path); result["metrics"]["power_w"] = 0
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "authoritative queue"):
            self.archive.ingest(self.baseline)
        path.write_bytes(original)
        (attempt / "output/analysis/metrics.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Artifact changed"):
            self.archive.ingest(self.baseline)
        self.assertEqual(self.archive.show()["frontier"], [])

    def test_complete_baseline_is_reference_and_task_cannot_change(self):
        self.bootstrap()
        row = self.archive.show()["frontier"][0]
        self.assertEqual(row["vector"], dict.fromkeys(OBJECTIVES, 1.0))
        with self.archive.transaction() as db:
            db.execute("UPDATE metadata SET value='{}' WHERE key='task'")
        with self.assertRaisesRegex(ValueError, "Archive task changed"):
            Archive(self.root, self.task)


if __name__ == "__main__":
    unittest.main()
