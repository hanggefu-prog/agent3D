"""Single-invocation prompt/checkout contracts; no model or native EDA calls."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_bench.grt_round_checkout import RoundParentCheckout
from global_bench.grt_candidate_library import export_library, record_summary
from global_bench import grt_round_prompt as prompts
from global_bench.grt_round_prompt import check_parent_request, check_response, round_prompt, save_bundle
from grt_bench.common import read, verify_records, write
from grt_bench.contract import OBJECTIVES
import test_grt_optimization as fixtures


class RoundPromptTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OptimizationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.bootstrap()
        self.library_serial = 0
        self.budget = {"round_number": 1, "round_limit": 10, "evaluation_limit": 10,
                       "evaluations_used": 0, "round_reserved": True}

    def export(self):
        self.library_serial += 1
        self.library = self.fixture.root / "libraries" / str(self.library_serial)
        export_library(self.fixture.root, self.fixture.task, self.library)
        return self.library

    def prepare(self, *, task=None, view=None, budget=None, history=(), outcomes=None, **paths):
        f = self.fixture
        return round_prompt(task or f.task, view or f.archive.show(), budget or self.budget,
            outcomes if outcomes is not None else [json.loads(r["result"]) for r in f.queue.rows()],
            history, library=self.export(), **paths)

    def proposal(self):
        return {"parent": self.fixture.baseline, "reason": "Only qualified frontier parent",
                "hypothesis": "Use a topology-aware cut penalty",
                "evidence": ["Baseline has nonzero timing deficits"],
                "files_to_modify": [fixtures.PARTITION],
                "expected_tradeoffs": dict.fromkeys((*OBJECTIVES, "power_w"), "uncertain")}

    def handler(self, bundle):
        f = self.fixture
        return RoundParentCheckout(f.root, f.task, bundle,
            f.root / "control/rounds/001", f.root / "workspaces/round-001")

    def response(self, bundle, receipt):
        return {**receipt["proposal"], "task_digest": bundle["context"]["task_digest"],
                "round": 1, "selection_id": receipt["selection_id"], "status": "ready",
                "summary": "Changed cut weighting",
                "change_summary": {"from_reference": "Change value from 1 to 2.", "from_parent": "Change value from 1 to 2."},
                "changed_files": [fixtures.PARTITION],
                "checks": [], "risks": ["Native evaluation not run"], "next_round_hint": "Inspect seed timing"}

    def test_single_round_checkout_edit_and_freeze(self):
        f = self.fixture
        bundle = self.prepare()
        self.assertEqual(bundle, self.prepare())
        self.assertEqual(bundle["context"]["maximum_model_invocations_this_round"], 1)
        self.assertEqual(bundle["context"]["cells_per_matrix"], 6)
        self.assertNotIn("raw_per_case", bundle["context"]["candidates"][0])
        self.assertEqual(bundle["context"]["candidates"][0]["evaluation_summary"]["objectives"], dict.fromkeys(OBJECTIVES, 1.0))
        handler = self.handler(bundle)
        receipt = handler.checkout(self.proposal())
        self.assertEqual(receipt["workspace"], "/workspace/subject")
        self.assertNotIn(str(f.root), json.dumps(receipt))
        (handler.workspace / fixtures.PARTITION).write_text("value = 2\n")
        response = self.response(bundle, receipt)
        check_response(bundle, response, handler.receipt())
        candidate = f.archive.freeze(handler.workspace, receipt["selection_id"])
        self.assertEqual(f.archive.admit(candidate)["parent"], f.baseline)
        note = record_summary(f.root, f.task, candidate, bundle, receipt, response)
        self.assertEqual(note, record_summary(f.root, f.task, candidate, bundle, receipt, response))
        self.export()
        entry = read(self.library / candidate / "candidate.json")
        self.assertEqual(entry["changes"]["from_parent"]["summary_origin"], "agent-description")
        self.assertEqual(entry["changes"]["from_reference"]["summary"], response["change_summary"]["from_reference"])
        with self.assertRaisesRegex(ValueError, "changed files differ"):
            record_summary(f.root, f.task, candidate, bundle, receipt, dict(response, changed_files=[fixtures.PLACEMENT]))
        output = f.root / "round-prompt"
        save_bundle(output, bundle)
        verify_records(output, read(output / "manifest.json")["artifacts"])
        with self.assertRaises(FileExistsError): save_bundle(output, bundle)
        self.assertEqual(len(f.queue.rows()), 6)
        self.assertEqual(len(f.archive.show()["evaluated"]), 1)

    def test_cli_emits_one_prompt_and_a_bound_parent_tool_schema(self):
        from global_bench.grt_round_prompt import main
        f = self.fixture
        paths = [f.root / (name + ".json") for name in ("task", "archive", "budget", "outcomes")]
        values = [f.task, f.archive.show(), self.budget, [json.loads(r["result"]) for r in f.queue.rows()]]
        for path, value in zip(paths, values): write(path, value)
        output = f.root / "cli-prompt"
        with patch.object(sys, "argv", ["prompt", *map(str, paths), str(output), "--library", str(self.export()),
                                       "--scratch-directory", "/workspace/local-checks"]): main()
        bundle = read(output / "bundle.json")
        self.assertEqual(bundle["phase"], "round")
        self.assertEqual(bundle["context"]["scratch_directory"], "/workspace/local-checks")
        self.assertEqual(read(output / "output.schema.json"), bundle["output_schema"])
        self.assertEqual(read(output / "parent-tool.json"), bundle["parent_tool"])
        self.assertEqual((output / "prompt.md").read_text(), bundle["prompt"])
        verify_records(output, read(output / "manifest.json")["artifacts"])

    def test_scratch_directory_cannot_overlap_source_or_reading_library(self):
        for scratch in ("/workspace/subject", "/workspace/subject/tests", "/workspace",
                        "/workspace/candidates", "/workspace/candidates/temporary", "/",
                        "relative", "/workspace/subject/../checks"):
            with self.subTest(scratch=scratch), self.assertRaises(ValueError):
                self.prepare(scratch_directory=scratch)
        bundle = self.prepare(scratch_directory="/workspace/local-checks")
        self.assertEqual(bundle["context"]["scratch_directory"], "/workspace/local-checks")
        self.assertEqual(bundle["context"]["direct_full_evaluation_allowance"], 0)
        self.assertEqual(bundle["context"]["maximum_matrices_this_round"], 1)

    def test_small_local_execution_does_not_enter_the_candidate_snapshot(self):
        f = self.fixture
        bundle = self.prepare()
        handler = self.handler(bundle)
        receipt = handler.checkout(self.proposal())
        source = handler.workspace / fixtures.PARTITION
        source.write_text("value = 2\n")
        scratch = f.root / "local-checks"
        scratch.mkdir()
        test = scratch / "check_value.py"
        test.write_text("import runpy, sys\n"
                        "assert runpy.run_path(sys.argv[1])['value'] == 2\n"
                        "print('small algorithm check passed')\n")
        result = subprocess.run([sys.executable, str(test), str(source)], cwd=scratch,
                                capture_output=True, text=True, check=True, timeout=10)
        (scratch / "result.log").write_text(result.stdout)
        response = self.response(bundle, receipt)
        response["checks"] = [{"name": "small algorithm check", "outcome": "passed",
                               "details": result.stdout.strip()}]
        check_response(bundle, response, receipt)
        candidate = f.archive.freeze(handler.workspace, receipt["selection_id"])
        self.assertEqual(f.archive.admit(candidate)["changes"], [fixtures.PARTITION])
        files = read(f.store / candidate / "candidate.json")["files"]
        self.assertEqual(set(files), set(read(f.store / f.baseline / "candidate.json")["files"]))
        self.assertEqual(len(f.queue.rows()), 6)  # No new full evaluation was submitted.

    def test_background_is_included_only_in_first_round(self):
        background = (prompts.TEMPLATES / "background.md").read_text()
        instructions = (prompts.TEMPLATES / "round.md").read_text()
        first = self.prepare()
        later = self.prepare(budget=dict(self.budget, round_number=2))
        self.assertTrue(first["prompt"].startswith(background + "\n\n" + instructions))
        self.assertTrue(later["prompt"].startswith(instructions))
        self.assertNotIn(background, later["prompt"])
        self.assertEqual(first["template_digest"], later["template_digest"])
        for bundle in (first, later):
            self.assertNotIn("research_brief", bundle["context"])
            self.assertNotIn('"research_brief":', bundle["prompt"])
            self.assertEqual(bundle["context"]["allowed_files"],
                             self.fixture.task["optimization"]["allowed_files"])
            self.assertEqual(bundle["context"]["maximum_model_invocations_this_round"], 1)

    def test_template_digest_tracks_background_even_after_first_round(self):
        first = self.prepare()
        later = self.prepare(budget=dict(self.budget, round_number=2))
        templates = self.fixture.root / "prompt-templates"
        templates.mkdir()
        for name in ("background.md", "round.md"):
            (templates / name).write_text((prompts.TEMPLATES / name).read_text())
        with (templates / "background.md").open("a") as f:
            f.write("\nChanged background for this test.\n")
        with patch.object(prompts, "TEMPLATES", templates):
            changed_first = self.prepare()
            changed_later = self.prepare(budget=dict(self.budget, round_number=2))
        self.assertNotEqual(first["prompt"], changed_first["prompt"])
        self.assertEqual(later["prompt"], changed_later["prompt"])
        self.assertNotEqual(first["template_digest"], changed_first["template_digest"])
        self.assertEqual(changed_first["template_digest"], changed_later["template_digest"])

    def test_budget_bounds_and_inconsistent_counters_rejected(self):
        for field, value in (("round_number", 0), ("round_number", 11), ("round_number", True),
                             ("evaluations_used", 10), ("evaluations_used", 1),
                             ("evaluation_limit", 0), ("round_reserved", False)):
            budget = dict(self.budget, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.prepare(budget=budget)

    def test_heldout_metadata_and_results_never_reach_prompt(self):
        bundle = self.prepare()
        self.assertNotIn("held-out", json.dumps(bundle["context"]))
        view = self.fixture.archive.show()
        view["evaluated"][0]["per_case"]["held-out"] = view["evaluated"][0]["per_case"]["bp"]
        with self.assertRaises(ValueError): self.prepare(view=view)
        outcomes = [json.loads(r["result"]) for r in self.fixture.queue.rows()]
        outcomes[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "only this task's training"): self.prepare(outcomes=outcomes)

    def test_raw_feedback_requires_admitted_complete_matrix(self):
        rows = [json.loads(r["result"]) for r in self.fixture.queue.rows()]
        with self.assertRaises(ValueError): self.prepare(outcomes=rows[:-1])
        rows[0]["metrics"]["power_w"] += 1
        with self.assertRaisesRegex(ValueError, "admitted archive"): self.prepare(outcomes=rows)

    def test_prompt_has_compact_results_and_details_keep_raw_timing(self):
        bundle = self.prepare()
        context = bundle["context"]
        self.assertEqual(context["contract"]["objectives"], list(OBJECTIVES))
        self.assertEqual(context["contract"]["scoring"]["power_max_ratio"], 1.1)
        entry = context["candidates"][0]
        self.assertEqual(set(entry["evaluation_summary"]["objectives"]), set(OBJECTIVES))
        self.assertTrue(entry["evaluation_summary"]["power_feasible"])
        self.assertEqual(entry["evaluation_summary"]["delta_from_reference"], dict.fromkeys(OBJECTIVES, 0))
        self.assertNotIn("raw_per_case", json.dumps(context))
        detail = read(self.library / self.fixture.baseline / "evaluation/metrics.json")
        self.assertIn("wns_deficit_ns", detail["raw_per_case"]["bp"])
        self.assertIn("tns_deficit_ns", detail["raw_per_case"]["bp"])
        self.assertEqual(detail["aggregate"]["constraints"]["power_w"]["per_case"]["bp"]["limit_w"], 11)
        self.assertEqual(set(bundle["parent_tool"]["inputSchema"]["properties"]["expected_tradeoffs"]["properties"]),
                         {*OBJECTIVES, "power_w"})

    def test_agent_view_keeps_decisions_and_full_schemas_keep_enforcement(self):
        bundle = self.prepare()
        view = bundle["agent_context"]
        self.assertIn(json.dumps(view, indent=2, sort_keys=True), bundle["prompt"])
        self.assertEqual(view["task_digest"], bundle["context"]["task_digest"])
        self.assertEqual(view["scoring"], self.fixture.task["scoring"])
        self.assertEqual(view["budget"]["evaluation_limit"], self.budget["evaluation_limit"])
        self.assertNotIn("round_reserved", view["budget"])
        self.assertEqual(view["task_details"], "/workspace/candidates/task-context.json")
        details = read(self.library / "task-context.json")
        files = self.fixture.task["optimization"]["allowed_files"]
        self.assertEqual(details["allowed_files"], files)
        self.assertEqual(details["contract"]["seeds"], self.fixture.task["seeds"])
        for key in ("allowed_files", "reference_digest", "frontier", "recent_history", "training_cases", "contract"):
            self.assertNotIn(key, view)
        self.assertNotIn("manifest_sha256", json.dumps(view))
        for name in files:
            self.assertNotIn(name, bundle["prompt"])
        for schema, field in ((bundle["parent_tool"]["inputSchema"], "files_to_modify"),
                              (bundle["output_schema"], "files_to_modify"),
                              (bundle["output_schema"], "changed_files")):
            self.assertEqual(schema["properties"][field]["items"]["enum"], files)
        before = copy.deepcopy(bundle["context"])
        self.assertEqual(prompts._agent_context(bundle["context"]), view)
        self.assertEqual(bundle["context"], before)

    def test_agent_view_preserves_small_deltas_missing_results_and_failures(self):
        bundle = self.prepare()
        context = copy.deepcopy(bundle["context"])
        entry = context["candidates"][0]
        entry["evaluation_summary"]["delta_from_reference"]["temperature_peak_k"] = -1.234567890123e-10
        entry["evaluation_summary"]["power_feasible"] = False
        entry["evaluation_summary"]["power_failed_cases"] = ["bp"]
        missing = copy.deepcopy(entry)
        missing.update(candidate="c" * 64, status="incomplete_or_failed", evaluation_summary=None)
        context["candidates"].append(missing)
        view = prompts._agent_context(context)
        for displayed, original in zip(view["candidates"], context["candidates"]):
            self.assertEqual(displayed["change_summary"], original["change_summary"])
            self.assertEqual(displayed["directory"], original["directory"])
        score = view["candidates"][0]["evaluation_summary"]
        self.assertEqual(score["delta_from_reference"]["temperature_peak_k"], -1.234567890123e-10)
        self.assertFalse(score["power_feasible"])
        self.assertEqual(score["power_failed_cases"], ["bp"])
        self.assertIsNone(view["candidates"][1]["evaluation_summary"])

    def test_infeasible_history_is_visible_but_parent_is_not_selectable(self):
        f = self.fixture
        child = f.child(); f.receipts(child, [1, 1, 1, 1, 12]); f.archive.ingest(child)
        history = [{"round": 1, "split": "train", "status": "infeasible", "parent": f.baseline,
                    "candidate": child, "hypothesis": "Improve timing", "feedback": "Power exceeds the cap"}]
        refs = [json.loads(r["result"]) for r in f.queue.rows() if json.loads(r["payload"])["candidate"] == f.baseline]
        bundle = self.prepare(budget=dict(self.budget, round_number=2, evaluations_used=1),
                              outcomes=refs, history=history)
        self.assertEqual(bundle["context"]["recent_history"], history)
        entry = next(r for r in bundle["context"]["candidates"] if r["candidate"] == child)
        self.assertEqual(entry["status"], "infeasible")
        self.assertFalse(entry["selectable"])
        self.assertFalse(entry["evaluation_summary"]["power_feasible"])
        self.assertEqual(entry["evaluation_summary"]["power_failed_cases"], ["bp", "bp_fe"])
        with self.assertRaises(ValueError):
            check_parent_request(bundle, dict(self.proposal(), parent=child))

    def test_parent_tool_rejects_unlisted_sources_scope_and_model_paths(self):
        bundle = self.prepare()
        for key, value in (("parent", "c" * 64), ("files_to_modify", [fixtures.PROTECTED]),
                           ("workspace", "/tmp/arbitrary"), ("hypothesis", ""), ("evidence", [])):
            with self.subTest(field=key), self.assertRaises(ValueError):
                check_parent_request(bundle, dict(self.proposal(), **{key: value}))

    def test_final_response_requires_the_host_receipt_and_recorded_proposal(self):
        bundle = self.prepare(); handler = self.handler(bundle)
        receipt = handler.checkout(self.proposal()); response = self.response(bundle, receipt)
        with self.assertRaisesRegex(ValueError, "No host checkout"): check_response(bundle, response)
        for key, value in (("round", 2), ("round", True), ("task_digest", "b" * 64),
                           ("selection_id", "invented"),
                           ("changed_files", [fixtures.PROTECTED]),
                           ("checks", [{"name": "AST", "outcome": "failed", "details": "Syntax error"}]),
                           ("change_summary", {"from_reference": "", "from_parent": "A change"}),
                           ("change_summary", {"from_reference": "x " * 101, "from_parent": "A change"})):
            with self.subTest(field=key), self.assertRaises(ValueError):
                check_response(bundle, dict(response, **{key: value}), receipt)
        with self.assertRaises(ValueError): check_response(bundle, dict(response, claimed_score=0), receipt)
        for key, value in (("task_digest", "b" * 64), ("bundle_digest", "a" * 64),
                           ("round", 2), ("workspace", "/wrong")):
            with self.subTest(receipt_field=key), self.assertRaises(ValueError):
                check_response(bundle, response, dict(receipt, **{key: value}))

    def test_blocked_before_checkout_is_valid_and_cannot_evaluate(self):
        bundle = self.prepare()
        response = {**self.proposal(), "task_digest": bundle["context"]["task_digest"],
                    "round": 1, "parent": None, "selection_id": None, "status": "blocked",
                    "summary": "Parent tool unavailable", "change_summary": {"from_reference": "", "from_parent": ""}, "files_to_modify": [], "changed_files": [],
                    "checks": [], "risks": [], "next_round_hint": "Host must restore the tool"}
        check_response(bundle, response)
        with self.assertRaises(ValueError): check_response(bundle, dict(response, status="ready"))
        self.assertIsNone(self.handler(bundle).receipt())

    def test_history_is_bounded_and_rejects_test_or_future_rounds(self):
        budget = dict(self.budget, round_number=10, evaluations_used=1)
        history = [{"round": i, "split": "train", "status": "rejected", "parent": self.fixture.baseline,
                    "candidate": None, "hypothesis": "Try a new penalty", "feedback": "Illegal edit"} for i in range(1, 10)]
        bundle = self.prepare(budget=budget, history=history)
        self.assertEqual(len(bundle["context"]["recent_history"]), 8)
        self.assertEqual(bundle["context"]["omitted_history_count"], 1)
        for key, value in (("split", "test"), ("round", 10)):
            bad = copy.deepcopy(history); bad[0][key] = value
            with self.assertRaises(ValueError): self.prepare(budget=budget, history=bad)

    def test_live_frontier_is_rechecked_after_prompt_preparation(self):
        f = self.fixture; bundle = self.prepare()
        child = f.child(); f.receipts(child, [8] * 5); f.archive.ingest(child)
        handler = self.handler(bundle)
        with self.assertRaisesRegex(ValueError, "current training frontier"): handler.checkout(self.proposal())
        self.assertFalse(handler.workspace.exists())
        view = f.archive.show(); view["frontier"] = view["evaluated"]
        with self.assertRaisesRegex(ValueError, "frontier differs"): self.prepare(view=view)

    def test_concurrent_replay_selects_once_and_never_overwrites_edits(self):
        f = self.fixture; bundle = self.prepare(); proposal = self.proposal()
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(lambda _: self.handler(bundle).checkout(proposal), range(2)))
        self.assertEqual(receipts[0], receipts[1])
        self.assertEqual(len(f.archive.show()["selections"]), 1)
        handler = self.handler(bundle)
        source = handler.workspace / fixtures.PARTITION
        source.write_text("value = 77\n")
        self.assertEqual(handler.checkout(proposal), receipts[0])
        self.assertEqual(source.read_text(), "value = 77\n")
        with self.assertRaisesRegex(ValueError, "already bound"):
            handler.checkout(dict(proposal, reason="Switch the proposal"))

    def test_failed_materialization_is_not_automatically_replayed(self):
        f = self.fixture; handler = self.handler(self.prepare())
        with patch("grt_bench.archive.Archive.checkout", side_effect=RuntimeError("Interrupted copy")):
            with self.assertRaises(RuntimeError): handler.checkout(self.proposal())
        with self.assertRaisesRegex(ValueError, "reconciliation"): handler.checkout(self.proposal())
        with self.assertRaisesRegex(ValueError, "reconciliation"): handler.receipt()
        self.assertEqual(len(f.archive.show()["selections"]), 1)

    def test_tool_binding_does_not_reuse_another_round_or_existing_workspace(self):
        f = self.fixture; bundle = self.prepare(); handler = self.handler(bundle)
        handler.checkout(self.proposal())
        another = self.prepare(budget=dict(self.budget, round_number=2))
        with self.assertRaisesRegex(ValueError, "already bound"):
            self.handler(another).checkout(self.proposal())
        with self.assertRaises(ValueError):
            RoundParentCheckout(f.root, f.task, bundle, f.root / "outside-control", f.root / "workspaces/new")
        with self.assertRaises(ValueError):
            RoundParentCheckout(f.root, f.task, bundle, f.root / "control/new", f.root / "candidates/overwrite")


if __name__ == "__main__": unittest.main()
