"""Two-method integration with real OpenEvolve database/sampler, offline LLM."""
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from local_bench.model import EvaluationResult
from local_bench.search_api import (AskTellRunner, CandidateFeedback, CandidateProposal,
    ProgramEvaluationHarness, SearchBackend, SearchContext, SearchProblem, TaskEvaluationOptions)
from local_bench.search_session import prepare_search_problem, run_task_search, search_feedback
from local_bench.openevolve_search import OpenEvolveAskTell

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "AgenticBench/local_tasks/tp_gnn_post_placement_timing"


def options(tmp_path, task=TASK, mode="interface"):
    return TaskEvaluationOptions(task_dir=task, output_dir=tmp_path / "run",
                                 open3dbench_root=ROOT, evaluation_mode=mode)


def native_backend(generate):
    pytest.importorskip("openevolve")
    from openevolve.config import Config
    config = Config(diff_based_evolution=False)
    config.database.num_islands = 2
    config.database.migration_interval = 1
    return OpenEvolveAskTell(config, generate=generate)


def feedback(proposal, score=1.0, status="accepted"):
    return CandidateFeedback(proposal.proposal_id,
        hashlib.sha256(proposal.program.encode()).hexdigest(), status, score,
        metrics={"combined_score": score} if score is not None else {},
        error_message="invalid objective" if score is None else None)


def test_two_method_backend_needs_no_name_done_or_task_access(tmp_path):
    class TwoMethods:
        def __init__(self):
            self.contexts, self.results = [], []
        def ask(self, context):
            self.contexts.append(context)
            return [CandidateProposal(str(context.evaluations), context.problem.initial_program)]
        def tell(self, batch):
            self.results.extend(batch)
    backend = TwoMethods()
    assert isinstance(backend, SearchBackend)
    result = run_task_search(backend, options(tmp_path, mode="verify"), max_evaluations=2)
    assert len(result.history) == 2
    assert [x.status for x in result.history] == ["accepted", "cached"]
    first, second = backend.contexts
    assert first.problem.system_message and first.problem.initial_program
    assert first.problem.response_format == "complete_program"
    assert first.problem.prompt_templates["full_rewrite_user"]
    assert "source_path" not in json.dumps(asdict(first.problem))
    assert second.remaining_budget == 1 and second.incumbent is not None
    assert set(backend.results[0].artifacts) == {"optimization_feedback"}


def test_two_method_backend_can_stop_with_empty_ask():
    class Stop:
        def ask(self, context): return []
        def tell(self, feedback): raise AssertionError("Nothing to tell")
    result = AskTellRunner().run(Stop(), None, SearchProblem("t", "d", "code"), max_evaluations=2)
    assert result.stop_reason == "backend_done" and not result.history


def test_native_two_rounds_without_native_evaluator_or_export(tmp_path, monkeypatch):
    from local_bench import openevolve_adapter, prompting
    def forbidden(*args, **kwargs):
        raise AssertionError("Native evaluator/export must not be used")
    monkeypatch.setattr(openevolve_adapter, "evaluate_from_task_dir", forbidden)
    monkeypatch.setattr(prompting, "export_openevolve_bundle", forbidden)
    problem = prepare_search_problem(options(tmp_path))
    messages, evaluations = [], []
    def generate(system, user):
        messages.append((system, user))
        return "```python\n" + problem.initial_program + "\n```"
    def transport(*args, **kwargs):
        evaluations.append(args)
        return EvaluationResult(metrics={"combined_score": 1.02, "runs_successfully": 1.0,
            "timing_weighted_cut": 700, "area_balance_error": None},
            artifacts={"manifest": "DO_NOT_SEND", "error_message": None})
    backend = native_backend(generate)
    result = run_task_search(backend, options(tmp_path), max_evaluations=2, evaluation_transport=transport)
    assert len(result.history) == len(messages) == 2
    assert len(evaluations) == 1  # Second candidate is a cache hit, not a hidden re-evaluation.
    assert [item.status for item in result.history] == ["accepted", "cached"]
    assert "no measured score" in messages[0][1]
    assert "timing_weighted_cut" in messages[1][1]
    assert "DO_NOT_SEND" not in messages[1][1]
    for system, user in messages:
        assert "## Background" in system
        assert "## Implementation requirements" in system
        assert "### Numerical safety" in system
        for text in (system, user):
            plain = " ".join(text.split())
            assert "Try one main change at a time" in plain
            assert "causal" not in plain.lower()
        assert "You may combine ideas that have already improved comparable legal partitions" in " ".join(system.split())
        assert "combine ideas when comparable measured results support them" in " ".join(user.split())
    assert backend.database.best_program_id
    assert all(p.metrics["combined_score"] == 1.02 for p in backend.database.programs.values())
    assert "DO_NOT_SEND" in (tmp_path / "run/history.jsonl").read_text()
    assert len((tmp_path / "run/search_history.jsonl").read_text().splitlines()) == 2


def test_engine_uses_only_context_and_feedback_without_benchmark_reads(tmp_path, monkeypatch):
    problem = prepare_search_problem(options(tmp_path))
    backend = native_backend(lambda system, user: problem.initial_program)
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        if path.is_relative_to(ROOT):
            raise AssertionError("Search backend attempted to read benchmark files")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    first = backend.ask(SearchContext(problem, 0, 2, (), None))[0]
    good = feedback(first)
    backend.tell([good])
    second = backend.ask(SearchContext(problem, 1, 1, (good,), good))[0]
    backend.tell([feedback(second)])


def test_two_distinct_candidates_use_tell_for_population_ranking(tmp_path):
    opt = options(tmp_path)
    problem = prepare_search_problem(opt)
    generations, evaluations = [], []
    def generate(system, user):
        generations.append(user)
        return problem.initial_program.replace("    return -(",
            "    return {} * -(".format(1.0 + len(generations) / 100))
    def transport(*args, **kwargs):
        evaluations.append(1)
        return EvaluationResult(metrics={"combined_score": 1 + len(evaluations) / 100,
                                         "runs_successfully": 1}, artifacts={})
    backend = native_backend(generate)
    result = run_task_search(backend, opt, max_evaluations=2, evaluation_transport=transport)
    assert len(generations) == len(evaluations) == 2
    assert result.history[0].program_sha256 != result.history[1].program_sha256
    assert result.incumbent.score == 1.02
    assert backend.database.best_program_id == result.incumbent.proposal_id
    assert '"combined_score": 1.01' in generations[1]


def test_failure_reaches_next_prompt_but_never_population(tmp_path):
    problem = prepare_search_problem(options(tmp_path))
    seen = []
    def generate(system, user):
        seen.append(user)
        return problem.initial_program
    backend = native_backend(generate)
    first = backend.ask(SearchContext(problem, 0, 2, (), None))[0]
    bad = search_feedback(problem, feedback(first, None, "rejected"))
    backend.tell([bad])
    assert not backend.database.programs
    assert "combined_score" not in bad.metrics  # No fabricated zero reward.
    second = backend.ask(SearchContext(problem, 1, 1, (bad,), None))[0]
    assert "invalid objective" in seen[-1] and '"score": null' in seen[-1]
    assert "not a scored parent" in seen[-1]
    backend.tell([feedback(second)])
    assert backend.database.best_program_id == second.proposal_id


def test_protocol_identity_pending_and_contract_guards(tmp_path):
    problem = prepare_search_problem(options(tmp_path))
    backend = native_backend(lambda system, user: problem.initial_program)
    context = SearchContext(problem, 0, 2, (), None)
    first = backend.ask(context)[0]
    with pytest.raises(RuntimeError, match="tell"):
        backend.ask(context)
    for bad in [replace(feedback(first), proposal_id="wrong"),
                replace(feedback(first), program_sha256="wrong"),
                replace(feedback(first), score=float("nan")),
                replace(feedback(first), score=2),
                replace(feedback(first), status="failed")]:
        with pytest.raises(ValueError):
            backend.tell([bad])
    good = feedback(first)
    backend.tell([good])
    with pytest.raises(ValueError, match="pending"):
        backend.tell([good])
    with pytest.raises(ValueError, match="contract changed"):
        backend.ask(SearchContext(replace(problem, description="changed"), 1, 1, (good,), good))
    assert backend.ask(SearchContext(problem, 1, 0, (good,), good)) == ()


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), True])
def test_harness_does_not_turn_invalid_score_into_zero(tmp_path, score):
    task = TASK.parent / "placement_initialization"
    problem = prepare_search_problem(options(tmp_path, task))
    harness = ProgramEvaluationHarness(options(tmp_path, task), evaluation_transport=lambda *a, **k:
        EvaluationResult(metrics={"combined_score": score}, artifacts={}))
    item = harness.evaluate_proposal(CandidateProposal("invalid", problem.initial_program))
    assert item.status == "failed" and item.score is None and item.error_type == "InvalidScore"


def test_oversize_candidate_is_rejected_and_recorded(tmp_path):
    opt = options(tmp_path, mode="verify")
    problem = prepare_search_problem(opt)
    harness = ProgramEvaluationHarness(opt)
    item = harness.evaluate_proposal(CandidateProposal("big", "#" * (problem.max_program_chars + 1)))
    assert item.status == "rejected" and item.score is None
    assert "maximum" in item.error_message


def test_no_silent_provider_retry_or_fresh_backend_resume(tmp_path):
    problem = prepare_search_problem(options(tmp_path))
    calls = []
    def empty(system, user):
        calls.append(1)
        return ""
    backend = native_backend(empty)
    with pytest.raises(ValueError, match="no candidate"):
        backend.ask(SearchContext(problem, 0, 2, (), None))
    assert len(calls) == 1 and backend._pending is None
    fresh = native_backend(empty)
    with pytest.raises(ValueError, match="resume"):
        fresh.ask(SearchContext(problem, 1, 1, (), None))


def test_cached_failure_keeps_reason_and_is_not_a_valid_zero(tmp_path):
    task = TASK.parent / "placement_initialization"
    opt = options(tmp_path, task, "full")
    problem = prepare_search_problem(opt)
    count = []
    def failed(*args, **kwargs):
        count.append(1)
        return EvaluationResult(metrics={"combined_score": 0.0, "runs_successfully": 0.0},
            artifacts={"error_type": "NativeFailure", "error_message": "routing failed"})
    harness = ProgramEvaluationHarness(opt, evaluation_transport=failed)
    first = harness.evaluate_proposal(CandidateProposal("one", problem.initial_program))
    second = harness.evaluate_proposal(CandidateProposal("two", problem.initial_program))
    assert len(count) == 1
    assert first.status == second.status == "failed"
    assert second.score is None and second.error_message == "routing failed"
    assert second.artifacts["cached_from_order"] == 1


def test_native_ensemble_transport_without_evaluator(tmp_path):
    pytest.importorskip("openevolve")
    from openevolve.config import Config, LLMModelConfig
    problem = prepare_search_problem(options(tmp_path))
    calls = []
    class OfflineModel:
        def __init__(self):
            self.model = "offline-test-only"
        async def generate_with_context(self, system_message, messages, **kwargs):
            calls.append(messages)
            return problem.initial_program
    config = Config(diff_based_evolution=False)
    config.llm.models = [LLMModelConfig(name="offline-test-only", init_client=lambda cfg: OfflineModel())]
    backend = OpenEvolveAskTell(config)
    try:
        proposal = backend.ask(SearchContext(problem, 0, 1, (), None))[0]
        backend.tell([feedback(proposal)])
        assert len(calls) == 1 and backend.database.best_program_id == proposal.proposal_id
    finally:
        backend.close()


def test_existing_run_is_rejected_before_ask(tmp_path):
    opt = options(tmp_path)
    opt.output_dir.mkdir()
    (opt.output_dir / "history.jsonl").write_text("preserved\n")
    class NeverAsk:
        def ask(self, context): raise AssertionError("Must not generate into an old run")
        def tell(self, batch): raise AssertionError("Must not evaluate")
    with pytest.raises(FileExistsError, match="resume"):
        run_task_search(NeverAsk(), opt, max_evaluations=2)
    assert (opt.output_dir / "history.jsonl").read_text() == "preserved\n"


def test_unsupported_modes_fail_instead_of_falling_back():
    pytest.importorskip("openevolve")
    from openevolve.config import Config
    for config in [Config(), Config(diff_based_evolution=False)]:
        if not config.diff_based_evolution:
            config.database.db_path = "/nonexistent/old-database"
        with pytest.raises(ValueError):
            OpenEvolveAskTell(config)


def test_matrix_diagnostics_reach_next_native_prompt_without_nonscalar_database_metrics(tmp_path):
    task = TASK.parent / 'placement_initialization'
    problem = prepare_search_problem(options(tmp_path, task))
    messages = []
    def generate(system, user):
        messages.append(user)
        return problem.initial_program
    backend = native_backend(generate)
    first = backend.ask(SearchContext(problem, 0, 2, (), None))[0]
    matrix = {'complete': True,
              'cases': [{'case': 'bp_fe-ng45', 'mean': 1.02, 'stddev': .12,
                         'scores': {'s1': .90, 's2': 1.14}}],
              'cells': [{'case':'bp_fe-ng45', 'seed':'s1', 'status':'succeeded', 'score':.9,
                         'result_dir':'/results/bp_fe/s1', 'metrics':{'placement_native_hpwl':12345}}]}
    good = replace(feedback(first, 1.02), metrics={'combined_score':1.02, 'runs_successfully':1, 'matrix':matrix})
    backend.tell([good])
    assert 'matrix' not in backend.database.programs[first.proposal_id].metrics
    second = backend.ask(SearchContext(problem, 1, 1, (good,), good))[0]
    assert 'bp_fe-ng45' in messages[-1] and 'stddev' in messages[-1]
    assert 's1' in messages[-1] and 's2' in messages[-1]
    assert 'placement_native_hpwl' in messages[-1] and '/results/bp_fe/s1' in messages[-1]
    backend.tell([feedback(second)])
    backend.close()
