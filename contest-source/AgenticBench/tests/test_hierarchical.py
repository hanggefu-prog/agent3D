from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from agentic3d.hierarchical import (
    GlobalModelingContext,
    HierarchicalRoundCoordinator,
    IntegratedPPAEvaluator,
    IntegratedProposal,
    LocalCorrectionResult,
    StageTask,
    StageTaskMaterializer,
    aggregate_corrections,
    local_task_template_catalog,
    modeled_tasks_from_global_analysis,
)
from local_bench.search_api import CandidateFeedback


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "local_tasks"


def _feedback(proposal_id: str, score: float = 1.0) -> CandidateFeedback:
    return CandidateFeedback(
        proposal_id=proposal_id,
        program_sha256=hashlib.sha256(proposal_id.encode("utf-8")).hexdigest(),
        status="accepted",
        score=score,
        metrics={"combined_score": score},
    )


def _source_task(round_number: int = 0) -> StageTask:
    return StageTask(
        task_id="layout-source-{}".format(round_number),
        template="placement_preconditioner",
        template_dir=TASKS / "placement_preconditioner",
        stage="layout",
        global_round=round_number,
        hypothesis="change the preconditioner to reduce late overflow",
        objective={"goal": "improve placement", "metric_weights": {"hpwl": 0.9, "overflow": 0.1}},
        acceptance_gates=("combined score improves",),
        correctness_gates=("placement is legal",),
        evaluation_budget=2,
        source_scope=(
            "Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py",
        ),
        config_scope=(),
        context_artifacts=("placement.log",),
    )


def _dynamic_task(round_number: int = 0) -> StageTask:
    return StageTask(
        task_id="layout-dynamic-{}".format(round_number),
        template="placement_parameters_reference_20260923",
        template_dir=TASKS / "placement_parameters_reference_20260923",
        stage="layout",
        global_round=round_number,
        hypothesis="adapt the learning rate to the observed overflow trajectory",
        objective={"goal": "improve placement", "metric_weights": {"hpwl": 0.9, "overflow": 0.1}},
        acceptance_gates=("combined score improves",),
        correctness_gates=("placement is legal",),
        evaluation_budget=2,
        source_scope=(
            "Place-3D/thirdparty/DREAMPlace/dreamplace/NonLinearPlace.py",
        ),
        config_scope=("placement.target_density",),
    )


def _candidate(*, patch: str = "", config=None):
    return {
        "schema_version": 1,
        "config_overrides": config or {},
        "candidate_files": {},
        "source_patch": patch,
        "notes": "test correction",
    }


def _patch(path: str, before: str, after: str) -> str:
    return (
        "diff --git a/{0} b/{0}\n"
        "--- a/{0}\n"
        "+++ b/{0}\n"
        "@@ -1 +1 @@\n"
        "-{1}\n"
        "+{2}\n"
    ).format(path, before, after)


class StageTaskModelingTest(unittest.TestCase):
    def test_catalog_exposes_exact_template_scopes(self):
        catalog = local_task_template_catalog(TASKS)
        self.assertEqual(set(catalog), {
            "tp_gnn_post_placement_timing",
            "tp_gnn_binary_loss_timing",
            "tp_gnn_loss_decoder_timing",
            "placement_initialization",
            "placement_preconditioner",
            "placement_parameters_reference_20260923",
            "hbt_assignment_policy",
            "global_routing_ripup_policy",
            "global_routing_maze_cost",
            "global_routing_iteration_controller",
        })
        self.assertIn(
            "Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py",
            catalog["placement_preconditioner"]["allowed_source_paths"],
        )
        self.assertEqual(catalog["placement_parameters_reference_20260923"]["stage"], "layout")
        self.assertNotIn("global_routing_net_ordering", catalog)
        self.assertEqual(catalog["hbt_assignment_policy"]["stage"], "place_opt")
        self.assertEqual(catalog["global_routing_ripup_policy"]["stage"], "routing")
        self.assertEqual(catalog["global_routing_maze_cost"]["stage"], "routing")
        self.assertEqual(
            catalog["global_routing_iteration_controller"]["stage"], "routing"
        )

    def test_global_analysis_decodes_and_materializes_a_narrow_task(self):
        analysis = {
            "round": 3,
            "global_plan": {
                "local_tasks": [
                    {
                        "task_id": "layout-preconditioner-round-3",
                        "template": "placement_preconditioner",
                        "stage": "layout",
                        "hypothesis": "density gradients are underweighted late in placement",
                        "objective": {
                            "goal": "reduce HPWL and overflow",
                            "metric_weights": {"hpwl": 0.9, "overflow": 0.1},
                        },
                        "acceptance_gates": ["combined score improves"],
                        "correctness_gates": ["legal placement"],
                        "evaluation_budget": 10,
                        "source_scope": [
                            "Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"
                        ],
                        "config_scope": [],
                        "context_artifacts": ["placement.log"],
                    }
                ]
            },
        }
        task = modeled_tasks_from_global_analysis(analysis, TASKS)[0]
        with tempfile.TemporaryDirectory() as tmp:
            generated = StageTaskMaterializer().materialize(task, Path(tmp) / "task")
            self.assertEqual(generated.task_id, "layout-preconditioner-round-3")
            self.assertEqual(generated.allowed_source_paths, list(task.source_scope))
            self.assertEqual(generated.allowed_config_paths, [])
            prompt = generated.prompt.read_text(encoding="utf-8")
            self.assertIn("density gradients are underweighted", prompt)
            self.assertTrue((generated.root / "stage_task.json").is_file())
            self.assertTrue((generated.root / "program_adapter.py").is_file())

    def test_materializer_rejects_scope_expansion(self):
        task = _source_task()
        expanded = StageTask(
            **{
                **task.__dict__,
                "source_scope": ("OpenROAD-3D/flow/Makefile",),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "expands template scope"):
                StageTaskMaterializer().materialize(expanded, Path(tmp) / "task")


class CorrectionAggregationTest(unittest.TestCase):
    def _corrections(self):
        source = _source_task()
        dynamic = _dynamic_task()
        source_path = source.source_scope[0]
        dynamic_path = dynamic.source_scope[0]
        return (
            LocalCorrectionResult(
                correction_id="source-best",
                task=source,
                base_revision="abc123",
                candidate=_candidate(patch=_patch(source_path, "old_a", "new_a")),
                feedback=_feedback("source-best", 1.05),
                verification={"passed": True},
            ),
            LocalCorrectionResult(
                correction_id="dynamic-best",
                task=dynamic,
                base_revision="abc123",
                candidate=_candidate(
                    patch=_patch(dynamic_path, "old_b", "new_b"),
                    config={"placement": {"target_density": 0.6}},
                ),
                feedback=_feedback("dynamic-best", 1.03),
                verification={"passed": True},
            ),
        )

    def test_disjoint_local_corrections_form_one_integrated_proposal(self):
        integrated = aggregate_corrections(self._corrections(), "global-round-000")
        self.assertEqual(integrated.base_revision, "abc123")
        self.assertEqual(len(integrated.source_paths), 2)
        self.assertEqual(
            integrated.candidate["config_overrides"]["placement"]["target_density"],
            0.6,
        )
        decoded = IntegratedProposal.from_program(integrated.to_program())
        self.assertEqual(decoded.selected_corrections, ("source-best", "dynamic-best"))

    def test_same_source_file_is_a_deterministic_conflict(self):
        first = self._corrections()[0]
        second = LocalCorrectionResult(
            correction_id="source-other",
            task=first.task,
            base_revision=first.base_revision,
            candidate=_candidate(
                patch=_patch(first.task.source_scope[0], "old_c", "new_c")
            ),
            feedback=_feedback("source-other", 1.02),
            verification={"passed": True},
        )
        with self.assertRaisesRegex(ValueError, "source patch conflict"):
            aggregate_corrections((first, second), "global-round-000")


class HierarchicalCoordinatorTest(unittest.TestCase):
    def test_fixed_round_order_models_optimizes_aggregates_and_tells(self):
        context = GlobalModelingContext(
            user_command="reduce placement HPWL and congestion",
            global_round=0,
            global_ppa_report={"hpwl": 100.0},
            accepted_revision="abc123",
            remaining_budget=4,
        )
        tasks = (_source_task(), _dynamic_task())

        def modeler(received):
            self.assertIs(received, context)
            return tasks

        def optimizer(task):
            path = task.source_scope[0]
            return LocalCorrectionResult(
                correction_id=task.task_id + "-best",
                task=task,
                base_revision=context.accepted_revision,
                candidate=_candidate(
                    patch=_patch(path, "old", "new"),
                    config={"placement": {"target_density": 0.6}}
                    if task.config_scope
                    else {},
                ),
                feedback=_feedback(task.task_id + "-best", 1.01),
                verification={"passed": True},
            )

        def run_full_ppa(integrated):
            return {
                "score": 1.08,
                "metrics": {
                        "combined_score": 1.08,
                        "whole_flow_ppa": True,
                        "corrections": len(integrated.selected_corrections),
                },
            }

        result = HierarchicalRoundCoordinator().run_round(
            context, modeler, optimizer, IntegratedPPAEvaluator(run_full_ppa)
        )
        self.assertEqual(len(result.tasks), 2)
        self.assertEqual(len(result.corrections), 2)
        self.assertEqual(result.outer_search.incumbent.score, 1.08)
        self.assertTrue(result.outer_search.incumbent.metrics["whole_flow_ppa"])

    def test_whole_flow_exception_is_returned_through_outer_tell(self):
        context = GlobalModelingContext(
            user_command="reduce placement HPWL",
            global_round=0,
            global_ppa_report={},
            accepted_revision="abc123",
            remaining_budget=2,
        )
        task = _source_task()

        def optimizer(modeled):
            return LocalCorrectionResult(
                correction_id="source-best",
                task=modeled,
                base_revision="abc123",
                candidate=_candidate(
                    patch=_patch(modeled.source_scope[0], "old", "new")
                ),
                feedback=_feedback("source-best"),
                verification={"passed": True},
            )

        def fail(_integrated):
            raise RuntimeError("full flow failed")

        result = HierarchicalRoundCoordinator().run_round(
            context,
            lambda _context: [task],
            optimizer,
            IntegratedPPAEvaluator(fail),
        )
        feedback = result.outer_search.history[0]
        self.assertEqual(feedback.status, "failed")
        self.assertIsNone(feedback.score)
        self.assertEqual(feedback.error_type, "RuntimeError")


if __name__ == "__main__":
    unittest.main()
