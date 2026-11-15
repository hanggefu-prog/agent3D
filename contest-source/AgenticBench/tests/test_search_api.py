from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_bench.model import load_candidate, load_task
from local_bench.program_bridge import initial_program_text
from local_bench.search_api import (
    AskTellRunner,
    CandidateFeedback,
    CandidateProposal,
    ProgramEvaluationHarness,
    SearchContext,
    SearchProblem,
    TaskEvaluationOptions,
)


ROOT = Path(__file__).resolve().parents[1]


class QueueBackend:
    name = "queue-test"

    def __init__(self, batches):
        self.batches = list(batches)
        self.feedback = []
        self.contexts = []

    def ask(self, context: SearchContext):
        self.contexts.append(context)
        return self.batches.pop(0)

    def tell(self, feedback):
        self.feedback.append(tuple(feedback))

    def done(self):
        return not self.batches


class MetadataEvaluator:
    def evaluate_proposal(self, proposal):
        score = float(proposal.metadata["score"])
        return CandidateFeedback(
            proposal_id=proposal.proposal_id,
            program_sha256=hashlib.sha256(proposal.program.encode("utf-8")).hexdigest(),
            status="accepted",
            score=score,
            metrics={"combined_score": score},
        )


class AskTellRunnerTest(unittest.TestCase):
    def test_batches_are_evaluated_told_and_ranked(self):
        backend = QueueBackend(
            [
                [
                    CandidateProposal("p1", "program one", {"score": 0.8}),
                    CandidateProposal("p2", "program two", {"score": 1.1}),
                ],
                [CandidateProposal("p3", "program three", {"score": 0.9})],
            ]
        )
        result = AskTellRunner().run(
            backend,
            MetadataEvaluator(),
            SearchProblem("task", "problem", "initial"),
            max_evaluations=3,
        )

        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(result.incumbent.proposal_id, "p2")
        self.assertEqual([len(batch) for batch in backend.feedback], [2, 1])
        self.assertEqual(backend.contexts[1].incumbent.proposal_id, "p2")
        self.assertEqual(backend.contexts[1].remaining_budget, 1)

    def test_duplicate_proposal_ids_are_rejected(self):
        backend = QueueBackend(
            [[
                CandidateProposal("same", "program one"),
                CandidateProposal("same", "program two"),
            ]]
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            AskTellRunner().run(
                backend,
                MetadataEvaluator(),
                SearchProblem("task", "problem", "initial"),
                max_evaluations=2,
            )

    def test_backend_must_honor_remaining_budget(self):
        backend = QueueBackend(
            [[
                CandidateProposal("p1", "program one", {"score": 0.8}),
                CandidateProposal("p2", "program two", {"score": 0.9}),
            ]]
        )
        with self.assertRaisesRegex(ValueError, "only 1 evaluations remaining"):
            AskTellRunner().run(
                backend,
                MetadataEvaluator(),
                SearchProblem("task", "problem", "initial"),
                max_evaluations=1,
            )


class ProgramEvaluationHarnessTest(unittest.TestCase):
    def test_compact_evaluation_can_redirect_workspace_and_runtime_to_scratch(self):
        task_dir = ROOT / "local_tasks" / "placement_initialization"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            evaluation = output / "evaluations" / "0001-test"
            evaluation.mkdir(parents=True)
            harness = ProgramEvaluationHarness(
                TaskEvaluationOptions(
                    task_dir=task_dir,
                    output_dir=output,
                    evaluation_mode="interface",
                    open3dbench_root=root,
                    compact_evaluations=True,
                )
            )
            with mock.patch.dict(
                os.environ,
                {"OPEN3D_EVALUATION_SCRATCH_ROOT": str(root / "scratch")},
            ):
                result, scratch = harness._prepare_evaluation_result_dir(evaluation)

            self.assertIsNotNone(scratch)
            self.assertTrue((result / "workspace").is_symlink())
            self.assertTrue((result / "runtime").is_symlink())
            self.assertTrue((result / "workspace").is_dir())
            harness._discard_evaluation_scratch(result, scratch)
            self.assertFalse((result / "workspace").exists())
            self.assertFalse((result / "runtime").exists())
            self.assertFalse(scratch.exists())

    def test_verify_mode_records_accepted_and_cached_proposals(self):
        task_dir = ROOT / "local_tasks" / "placement_initialization"
        spec = load_task(task_dir)
        program = initial_program_text(spec, load_candidate(spec.initial_candidate))
        with tempfile.TemporaryDirectory() as tmp:
            harness = ProgramEvaluationHarness(
                TaskEvaluationOptions(
                    task_dir=task_dir,
                    output_dir=Path(tmp),
                    evaluation_mode="verify",
                    open3dbench_root=ROOT.parent,
                )
            )
            first = harness.evaluate_proposal(CandidateProposal("first", program))
            second = harness.evaluate_proposal(CandidateProposal("second", program))

            self.assertEqual(first.status, "accepted")
            self.assertEqual(second.status, "cached")
            self.assertEqual(first.score, second.score)
            self.assertEqual(len(harness.records), 2)
            self.assertTrue((Path(tmp) / "history.jsonl").is_file())
            search_rows = (Path(tmp) / "search_history.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn('"proposal_id": "first"', search_rows)
            self.assertIn('"proposal_id": "second"', search_rows)

    def test_invalid_program_becomes_rejected_feedback(self):
        task_dir = ROOT / "local_tasks" / "placement_initialization"
        with tempfile.TemporaryDirectory() as tmp:
            harness = ProgramEvaluationHarness(
                TaskEvaluationOptions(
                    task_dir=task_dir,
                    output_dir=Path(tmp),
                    evaluation_mode="verify",
                    open3dbench_root=ROOT.parent,
                )
            )
            feedback = harness.evaluate_proposal(
                CandidateProposal("bad", "import os\n")
            )

            self.assertEqual(feedback.status, "rejected")
            self.assertIsNone(feedback.score)
            self.assertEqual(feedback.error_type, "ValueError")


if __name__ == "__main__":
    unittest.main()
