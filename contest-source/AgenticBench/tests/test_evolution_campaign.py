"""Offline native OpenEvolve boundaries; no provider or physical execution."""
import hashlib
from types import SimpleNamespace
import pytest
from local_bench.evolution_campaign import MODELS, RecordedBackend, configuration, validate_configuration
from local_bench.openevolve_search import OpenEvolveAskTell
from local_bench.search_api import AskTellRunner, SearchProblem, CandidateFeedback


def test_settings_equal_except_model():
    pytest.importorskip("openevolve")
    settings = []
    for model in MODELS:
        c = configuration(model)
        row = validate_configuration(c, model, 30)
        assert row["database"]["population_size"] == 5
        row["llm"]["models"][0]["name"] = "same"
        settings.append(row)
    assert settings[0] == settings[1] == settings[2]
    with pytest.raises(ValueError): validate_configuration(c, model, 31)
    c.database.population_size = 6
    with pytest.raises(ValueError): validate_configuration(c, model, 30)


@pytest.mark.parametrize("consumed", [1, 2, 29])
def test_carryover_preserves_total_and_config(consumed):
    pytest.importorskip("openevolve")
    c = configuration(MODELS[0])
    result = validate_configuration(c, MODELS[0], 30-consumed, consumed_generation_slots=consumed)
    assert result["max_iterations"] == 30
    with pytest.raises(ValueError):
        validate_configuration(c, MODELS[0], 30, consumed_generation_slots=consumed)


@pytest.mark.parametrize("consumed", [-1, 30, True, 1.0])
def test_carryover_rejects_invalid_values(consumed):
    pytest.importorskip("openevolve")
    with pytest.raises(ValueError):
        validate_configuration(configuration(MODELS[0]), MODELS[0], 29,
                               consumed_generation_slots=consumed)


def test_native_thirty_slots_population_cap_feedback_and_initial(tmp_path):
    pytest.importorskip("openevolve")
    prompts, evaluated, acknowledged = [], [], []
    def generate(system, user):
        prompts.append((system, user))
        return "```python\ndef objective(x):\n    return x + %d\n```" % len(prompts)
    def evaluate(proposal):
        evaluated.append(proposal)
        return CandidateFeedback(proposal.proposal_id, hashlib.sha256(proposal.program.encode()).hexdigest(),
            "accepted", float(len(evaluated)), metrics={"combined_score": float(len(evaluated))},
            artifacts={"optimization_feedback": "measured physical result"})
    native = OpenEvolveAskTell(configuration(MODELS[0]), generate=generate)
    backend = RecordedBackend(native, tmp_path, acknowledged.extend)
    result = AskTellRunner().run(backend, SimpleNamespace(evaluate_proposal=evaluate),
        SearchProblem("test", "Task", "def objective(x):\n    return x", system_message="Test contract"),
        max_evaluations=30)
    assert len(prompts) == len(evaluated) == len(acknowledged) == len(result.history) == 30
    assert "return x" in prompts[0][1]
    assert '"combined_score": 1.0' in prompts[1][1]
    assert "measured physical result" not in prompts[1][1]  # Raw artifact text stays out.
    assert len(native.database.programs) <= 5
    assert len(list(tmp_path.glob("population-*.json"))) == 30
    backend.close()
