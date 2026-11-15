"""Display-only regressions with real OpenEvolve and no provider/EDA calls."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_bench.feedback_semantics import metric_definition
from local_bench.model import load_candidate, load_task
from local_bench.prompt_context import distinct_reference, feedback_text, prompt_metrics
from local_bench.prompting import build_problem_description
from local_bench.search_api import CandidateFeedback, SearchContext, TaskEvaluationOptions
from local_bench.search_session import prepare_search_problem

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("tp_gnn_post_placement_timing", "placement_initialization", "placement_preconditioner",
         "placement_parameters_reference_20260923", "global_routing_ripup_policy",
         "global_routing_maze_cost", "global_routing_iteration_controller", "hbt_assignment_policy")


@pytest.mark.parametrize("task", TASKS)
def test_initial_program_separate_from_instructions(tmp_path, task):
    spec = load_task(ROOT / "AgenticBench/local_tasks" / task)
    problem = prepare_search_problem(TaskEvaluationOptions(
        task_dir=spec.root, output_dir=tmp_path, open3dbench_root=ROOT))
    assert problem.initial_program not in problem.description
    assert spec.prompt.read_text().strip() in problem.description
    assert "## Background" in problem.description
    # Explicit legacy export keeps its old self-contained description.
    assert problem.initial_program in build_problem_description(spec, load_candidate(spec.initial_candidate), ROOT)


@pytest.mark.parametrize("task", TASKS)
def test_compact_metrics_keep_meanings_and_unknowns(task):
    spec = load_task(ROOT / "AgenticBench/local_tasks" / task)
    from local_bench.optimization_feedback import (PLACEMENT_METRICS, NESTEROV_METRICS,
        ROUTING_METRICS, POLICY_METRICS, TP_TIMING_GROUPS)
    from local_bench.feedback_semantics import DEFINITIONS
    keys = set(PLACEMENT_METRICS | NESTEROV_METRICS | ROUTING_METRICS | POLICY_METRICS | DEFINITIONS.keys())
    keys.update(key for group in TP_TIMING_GROUPS.values() for key in group)
    raw = dict.fromkeys(keys, 0.0)
    raw.update(combined_score=1.012345678901, runtime_seconds=None, legacy_noise=123)
    before = copy.deepcopy(raw)
    displayed = prompt_metrics(raw, spec.task_id)
    assert raw == before and 1 <= len(displayed) <= 20
    assert displayed["combined_score"] == raw["combined_score"]
    assert all(metric_definition(key, spec.task_id) for key in displayed)
    assert "runtime_seconds" not in displayed and "legacy_noise" not in displayed
    text = feedback_text(raw, spec.task_id)
    assert text.count('"combined_score":') == 1
    assert "higher is better" in text
    assert "runtime_seconds" not in text
    for invalid in (None, True, float("nan"), float("inf"), "0"):
        assert prompt_metrics({"combined_score": invalid}, spec.task_id) == {}


def test_reference_deduplicates_content_not_just_id():
    parent = SimpleNamespace(id="one", code="def f():\n    return 1\n")
    duplicate = SimpleNamespace(id="two", code="def f():  \n    return 1\n\n")
    other = SimpleNamespace(id="three", code="def f():\n    return 2\n")
    assert distinct_reference(parent, [parent, duplicate, other]) is other
    assert distinct_reference(parent, [duplicate]) is None


def test_real_sampler_three_rounds_one_parent_one_reference_and_bounded_failure(tmp_path):
    pytest.importorskip("openevolve")
    from openevolve.config import Config
    from local_bench.openevolve_search import OpenEvolveAskTell
    problem = prepare_search_problem(TaskEvaluationOptions(
        task_dir=ROOT / "AgenticBench/local_tasks/placement_initialization",
        output_dir=tmp_path, open3dbench_root=ROOT))
    messages = []
    def generate(system, user):
        messages.append((system, user))
        return problem.initial_program + "\n# candidate {}\n".format(len(messages))
    config = Config(diff_based_evolution=False)
    config.database.num_islands = 1
    backend = OpenEvolveAskTell(config, generate=generate)
    history, programs = [], []
    for i in range(3):
        proposal = backend.ask(SearchContext(problem, i, 3-i, tuple(history), history[0] if history else None))[0]
        programs.append(proposal.program)
        item = CandidateFeedback(proposal.proposal_id, hashlib.sha256(proposal.program.encode()).hexdigest(),
            "accepted" if i == 0 else "rejected", 1.0123456789 if i == 0 else None,
            metrics={"combined_score": 1.0123456789, "placement_native_hpwl": 123,
                     "placement_logged_rows": 900}, artifacts={"optimization_feedback": "RAW_LOG" * 10000},
            error_message=None if i == 0 else "Invalid arithmetic: " + "bad " * 10000)
        backend.tell([item])
        history.append(item)
    for system, user in messages:
        assert problem.initial_program not in system
        assert "RAW_LOG" not in user and "placement_logged_rows" not in user
        assert user.count("```python") == 1  # No different accepted source exists.
        assert user.count('"placement_native_hpwl":') <= 1
    assert "no measured score" in messages[0][1]
    assert '"combined_score": 1.0123456789' in messages[1][1]
    assert "not a scored parent" in messages[2][1] and "Invalid arithmetic" in messages[2][1]
    assert "# candidate 2" not in messages[2][1]  # No whole rejected-code dump.
    assert len(messages[2][1]) - len(messages[1][1]) < 1200
    stored = next(iter(backend.database.programs.values()))
    assert stored.metrics["placement_logged_rows"] == 900  # No loss of population data.
    assert "RAW_LOG" in backend._artifacts[stored.id]["optimization_feedback"]


@pytest.mark.parametrize("show_example", [True, False])
def test_real_sampler_caps_distinct_examples(tmp_path, show_example):
    pytest.importorskip("openevolve")
    from openevolve.config import Config
    from local_bench.openevolve_search import OpenEvolveAskTell
    from local_bench.search_api import SearchProblem
    problem = SearchProblem("generic", "Keep f().", "def f():\n    return 0\n")
    users = []
    def generate(system, user):
        users.append(user)
        return "def f():\n    return {}\n".format(len(users))
    config = Config(diff_based_evolution=False)
    config.database.num_islands = 1
    if not show_example:
        config.prompt.num_top_programs = config.prompt.num_diverse_programs = 0
    backend = OpenEvolveAskTell(config, generate=generate)
    history = []
    for i in range(5):
        p = backend.ask(SearchContext(problem, i, 5-i, tuple(history), history[-1] if history else None))[0]
        result = CandidateFeedback(p.proposal_id, hashlib.sha256(p.program.encode()).hexdigest(),
                                   "accepted", 1+i/100, metrics={"combined_score": 1+i/100})
        backend.tell([result])
        history.append(result)
    assert [u.count("```python") for u in users] == ([1, 1, 2, 2, 2] if show_example else [1]*5)
    for user in users:
        blocks = user.split("```python\n")[1:]
        programs = [block.split("```")[0].strip() for block in blocks]
        assert len(programs) == len(set(programs))
