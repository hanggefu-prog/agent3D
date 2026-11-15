"""Real candidate guards with fake model/EDA calls; never calls external APIs."""
import json
from types import SimpleNamespace

import pytest

from agentic3d import sta_knowledge_runner as runner
from agentic3d import codesign_endpoint as endpoint
from agentic3d.knowledge import write_json, digest
from agentic3d.sta_protocol import ANALYZER_SCHEMA, LOCAL_SCHEMA, REFLECTION_SCHEMA, REVIEW_SCHEMA, PARTITION_PATH
from scripts import run_codesign_experiment as experiment
from scripts import run_sta_codesign_comparison as legacy

REAL_RESEARCHER = runner.StatelessResearcher


@pytest.fixture
def harness(tmp_path, monkeypatch):
    spec = legacy.load_task(legacy.STA_PROFILE)
    source = tmp_path / "source"
    for name in spec.allowed_source_paths:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("value = 0\n")
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Frozen source test fixture")
    root = tmp_path / "run"
    root.mkdir()
    args = SimpleNamespace(agent_workspace=source, knowledge_seed=None, knowledge_split=None,
        knowledge_phase="independent", knowledge_import=None, test_knowledge_adaptation=False,
        case="bp_fe", instructions=instructions, reference_run="/reference/run", rounds=2,
        model="fake", codex_bin="fake", inner_tolerance=1e-6, gpu=0, knowledge_export=None,
        objective="timing", output=root, bundle=tmp_path / "bundle")
    reference = {"metrics": dict(runs_successfully=1, combined_score=1., opt_passes=2,
        sta_graph_complete=1, sta_unconstrained_endpoints=0, final_global_wns_setup_ns=-1.,
        final_global_tns_setup_ns=-10., timing_weighted_cut=100.)}
    state = SimpleNamespace(bad=0, kind="patch", local_calls=0, prompts=[], physical=[],
                            primitive="final", inner=1, physical_error=None, model_error=None,
                            fail_inner=False, blocked_role=None, evidence_failure=None)
    driver = SimpleNamespace(**vars(legacy))
    driver._remote_result = lambda _: reference
    driver.final_score = runner.final_score
    driver.FINAL_TARGET = "tp_gnn_placement_sta"
    driver.COMMON_CONTRACT = "Fixed evaluator; no API in test"

    class Agent:
        def __init__(self, name, path, schema, *unused):
            self.schema = schema
        def ask(self, prompt):
            state.prompts.append((self.schema, prompt))
            if self.schema == state.blocked_role:
                raise runner.ResearchBlocked("fixture tools unavailable")
            if self.schema == ANALYZER_SCHEMA:
                context = json.loads(prompt.split("STATE:\n")[1])
                primitive = state.primitive if context["budget"]["remaining_inner_physical"] else "final"
                return dict(hypothesis="measure change", evidence="mock", evidence_paths=[],
                    local_task=dict(id="test-task", description="Test failure handling", stage="partition",
                        source_scope=[PARTITION_PATH], evaluation=primitive,
                        objective=[] if primitive == "final" else [
                            dict(metric="timing_weighted_cut", weight=1., scale_floor=1.)],
                        procedure="Inspect frozen code", validation="Unchanged physical judge"),
                    specialist_instruction="", max_inner_attempts=1 if primitive == "final" else state.inner,
                    knowledge_refs=[], skills=[], tasks=[])
            if self.schema == REFLECTION_SCHEMA:
                return dict(assessment="Observed candidate failure or measured outcome", observations=[], suggestions=[])
            if self.schema == REVIEW_SCHEMA:
                return dict(assessment="Keep task scope", decisions=[], skills=[], tasks=[])
            state.local_calls += 1
            if state.model_error:
                raise state.model_error
            initial = json.loads((driver.STA_PROFILE / "initial_candidate/candidate.json").read_text())
            old = 999 if state.local_calls <= state.bad else 0
            path = "open3dbench/runner.py" if state.kind == "scope" and old else PARTITION_PATH
            initial["source_patch"] = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-value = {old}\n+value = {state.local_calls}\n"
            if state.kind == "noop":
                initial["source_patch"] = ""
            program = "not python (" if state.kind == "syntax" and old else legacy.render_candidate_program(initial)
            answer = dict(hypothesis="candidate", program=program)
            if self.schema == LOCAL_SCHEMA:
                answer.update(observations=[], suggestions=[])
            return answer

    def evaluate(**kw):
        state.physical.append(kw)
        if state.physical_error:
            raise state.physical_error
        if state.fail_inner and kw["task"] != driver.FINAL_TARGET:
            return {"metrics": {"runs_successfully": 0}}, "/result/run"
        return {"metrics": {**reference["metrics"], "timing_weighted_cut": 80.,
                            "combined_score": 1. + len(state.physical) / 100}}, "/result/run"
    driver.evaluate_remote = evaluate
    driver.PersistentCodex = Agent
    monkeypatch.setattr(runner, "StatelessResearcher", Agent)
    def evidence(_driver, remote, dest):
        dest.mkdir(parents=True, exist_ok=True)
        if state.evidence_failure and state.evidence_failure in dest.name:
            report = {"files": [], "error": "fixture evidence unavailable"}
        else:
            path = dest / "00.txt"
            path.write_text("fixture evidence")
            report = {"remote_run": remote, "files": [{"local_file": str(path), "sha256": digest(path.read_text())}], "error": None}
        write_json(dest / "index.json", report)
        return report
    monkeypatch.setattr(runner, "collect_evidence", evidence)
    state.run = lambda method="agentic3d": runner.run(args, root, driver) if method == "agentic3d" else experiment.run_codex(args, driver, reference)
    state.root, state.args, state.driver, state.agent_factory = root, args, driver, Agent
    return state


def test_access_probe_precedes_roles_once_and_blocked_probe_is_resumable(harness, monkeypatch):
    import hashlib
    from agentic3d.research_io import RESPONSE_INSTRUCTION

    h = harness
    h.args.rounds = 1
    calls, blocked = [], [True]
    monkeypatch.setattr(runner, "StatelessResearcher", REAL_RESEARCHER)
    def model(prompt, config):
        calls.append(config.run_dir)
        schema = json.loads(config.output_schema.read_text())["properties"]["response"]["anyOf"][0]
        if "research-preflight-v2" in str(config.run_dir):
            if blocked[0]:
                answer = {"blocked": "fixture file tools unavailable", "response": None}
            else:
                paths = json.loads(prompt.splitlines()[1])
                answer = {"blocked": None, "response": {
                    key: hashlib.sha256((config.workspace / path).read_bytes()).hexdigest()
                    for key, path in paths.items()}}
                answer["response"]["nonce"] = json.loads((config.workspace / "research-access.json").read_text())["nonce"]
        else:
            role = h.agent_factory("fixture", config.run_dir, schema)
            answer = {"blocked": None, "response": role.ask(prompt.removesuffix(RESPONSE_INSTRUCTION))}
        return SimpleNamespace(success=True, final_message=json.dumps(answer))
    h.driver.run_codex = model
    with pytest.raises(runner.ResearchBlocked, match="file tools unavailable"):
        h.run()
    assert len(calls) == 1 and not h.prompts and not h.physical
    assert not json.loads((h.root / "rounds/01/plan.sampling.json").read_text())["pending"]
    blocked[0] = False
    h.run()
    assert sum("research-preflight-v2" in str(path) for path in calls) == 2
    assert len(h.prompts) == 4 and len(h.physical) == 1
    before = len(calls)
    h.run()  # fully cached completion needs neither model access nor another final
    assert len(calls) == before and len(h.physical) == 1


@pytest.mark.parametrize("method", ["agentic3d", "codex"])
@pytest.mark.parametrize("kind", ["patch", "scope", "syntax"])
def test_invalid_candidates_replan_then_finish_without_spending_final(harness, method, kind):
    h = harness
    h.bad, h.kind = 3, kind
    h.run(method)
    assert len(h.physical) == 2 and h.local_calls == 5
    summary = json.loads((h.root / "summary.json").read_text())
    assert summary["final_evaluations"] == 2 and summary["failed_final_evaluations"] == 0
    rejections = list(h.root.glob("rounds/**/rejected.json"))
    assert len(rejections) == 1
    text = json.dumps(json.loads(rejections[0].read_text()))
    assert "final_global_wns" not in text
    assert any("invalid" in prompt for _, prompt in h.prompts[1:])
    calls = len(h.prompts)
    h.run(method)
    assert len(h.prompts) == calls and len(h.physical) == 2


def test_inner_failure_is_feedback_for_next_sample(harness):
    h = harness
    h.primitive, h.inner, h.bad = "partition", 3, 3
    h.run()
    trials = json.loads((h.root / "rounds/01/inner_trials.json").read_text())
    assert trials[0]["selection_score"] == 0 and not trials[0]["physical_evaluated"]
    assert trials[0]["result"]["metrics"] == {}
    assert trials[1]["physical_evaluated"]
    prompts = [p for schema, p in h.prompts if schema == LOCAL_SCHEMA]
    assert '"prior_attempts": [{"attempt": 1' in prompts[3]
    assert len(h.physical) == 6  # round 1 improves early; round 2 uses its three cheap calls
    summary = json.loads((h.root / "summary.json").read_text())
    assert summary["inner_proxy_evaluations"] == 4 and summary["invalid_candidates"] == 1


@pytest.mark.parametrize("method", ["agentic3d", "codex"])
@pytest.mark.parametrize("kind", ["patch", "noop"])
def test_all_invalid_bounded_and_resume_does_not_reset_budget(harness, method, kind):
    h = harness
    h.bad, h.kind = 100, kind
    with pytest.raises(runner.SearchBudgetExhausted):
        h.run(method)
    assert h.local_calls == 9 and not h.physical
    status = json.loads((h.root / "search_status.json").read_text())
    assert status["completed_final"] == 0 and status["remaining_final"] == 2
    calls = len(h.prompts)
    with pytest.raises(runner.SearchBudgetExhausted):
        h.run(method)
    assert len(h.prompts) == calls and not h.physical


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("SSH failed")])
def test_infrastructure_failure_is_not_a_bad_candidate(harness, error):
    h = harness
    h.physical_error = error
    with pytest.raises(type(error), match=str(error)):
        h.run()
    assert len(h.physical) == 1
    assert not list(h.root.glob("rounds/**/rejected.json"))


def test_api_failure_not_auto_retried_on_resume(harness):
    h = harness
    h.model_error = RuntimeError("API unavailable")
    with pytest.raises(RuntimeError, match="API unavailable"):
        h.run()
    h.model_error = None
    with pytest.raises(RuntimeError, match="interrupted/failed"):
        h.run()
    assert h.local_calls == 1 and not h.physical


def test_replanning_cannot_expand_three_inner_physical_budget(harness):
    h = harness
    h.args.rounds = 1
    h.primitive, h.inner, h.fail_inner = "partition", 3, True
    h.run()
    assert len(h.physical) == 4
    assert sum(p["task"] == "tp_gnn_sta_inner" for p in h.physical) == 3
    summary = json.loads((h.root / "summary.json").read_text())
    assert summary["final_evaluations"] == 1 and summary["inner_proxy_evaluations"] == 3
    assert summary["rejected_task_plans"] == 1
    calls = len(h.prompts)
    h.run()
    assert len(h.prompts) == calls and len(h.physical) == 4


def test_malformed_model_json_is_rejected_not_infrastructure(tmp_path):
    calls = []
    def ask(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.loads("not JSON")
        return {"valid": True}
    response = runner.cached_call(SimpleNamespace(ask=ask), tmp_path / "r.json", "task",
                                  lambda r: None, resample=True)
    assert response["valid"] and len(calls) == 2
    assert "invalid model JSON" in calls[1]


def test_sampling_replays_durable_response_and_does_not_reset_errors(tmp_path):
    path = tmp_path / "response.json"
    calls = []
    def ask(prompt):
        calls.append(prompt)
        return {"value": len(calls)}
    interrupted = [False]
    def validate(response):
        if response["value"] == 1:
            raise ValueError("bad patch")
        if not interrupted[0]:
            interrupted[0] = True
            raise OSError("disk unavailable during check")
    agent = SimpleNamespace(ask=ask)
    with pytest.raises(OSError):
        runner.cached_call(agent, path, "task", validate, resample=True)
    assert len(calls) == 2
    assert runner.cached_call(agent, path, "task", validate, resample=True)["value"] == 2
    assert len(calls) == 2
    assert "abandon its hypothesis" in calls[1]


@pytest.mark.parametrize("role", [ANALYZER_SCHEMA, LOCAL_SCHEMA, REFLECTION_SCHEMA, REVIEW_SCHEMA])
def test_role_blocker_stops_without_fake_candidate_and_explicit_resume(harness, role):
    h = harness
    h.args.rounds = 1
    h.blocked_role = role
    with pytest.raises(runner.ResearchBlocked, match="tools unavailable"):
        h.run()
    final_done = role in (REFLECTION_SCHEMA, REVIEW_SCHEMA)
    assert len(h.physical) == int(final_done)
    assert not list(h.root.glob("rounds/**/rejected.json"))
    assert not list(h.root.glob("rounds/**/rejection.json"))
    status = json.loads((h.root / "search_status.json").read_text())
    assert status["reason"] == "research-blocked"
    assert status["completed_final"] == int(final_done)
    assert status["remaining_final"] == 1 - int(final_done)
    if role == ANALYZER_SCHEMA:
        assert h.local_calls == 0
        assert not list(h.root.glob("rounds/**/enhanced_task.json"))
        assert not list(h.root.glob("knowledge/evolved/*/record.json"))
    assert not list(h.root.glob("knowledge/evolved/*-review/record.json"))
    for path in h.root.glob("rounds/**/*.sampling.json"):
        assert not json.loads(path.read_text())["pending"]
    h.blocked_role = None
    h.run()
    assert len(h.physical) == 1
    assert len(list(h.root.glob("knowledge/evolved/*-review/record.json"))) == 1
    calls = len(h.prompts)
    h.run()
    assert len(h.physical) == 1 and len(h.prompts) == calls


@pytest.mark.parametrize("stage", ["reference", "round-01", "inner-01"])
def test_evidence_failure_resume_never_repeats_physical_evaluation(harness, stage):
    h = harness
    h.args.rounds = 1
    h.evidence_failure = stage
    if stage == "inner-01":
        h.primitive = "partition"
    with pytest.raises(runner.ResearchBlocked, match="Evidence unavailable"):
        h.run()
    assert len(h.physical) == (0 if stage == "reference" else 1)
    if stage == "reference":
        assert not h.prompts
    if stage == "round-01":
        assert len(json.loads((h.root / "history.json").read_text())) == 1
    h.evidence_failure = None
    h.run()
    assert len(h.physical) == (2 if stage == "inner-01" else 1)
    assert json.loads((h.root / "summary.json").read_text())["final_evaluations"] == 1


def test_blocked_call_preserves_previous_candidate_repair_budget(tmp_path):
    path = tmp_path / "response.json"
    values = iter([{"value": 0}, runner.ResearchBlocked("cannot read"), {"value": 1}])
    def ask(_):
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value
    def validate(value):
        if value["value"] == 0:
            raise ValueError("bad candidate")
    agent = SimpleNamespace(ask=ask)
    with pytest.raises(runner.ResearchBlocked):
        runner.cached_call(agent, path, "same task", validate, resample=True)
    ledger = json.loads(path.with_suffix(".sampling.json").read_text())
    assert len(ledger["samples"]) == 1 and not ledger["pending"]
    assert runner.cached_call(agent, path, "same task", validate, resample=True) == {"value": 1}
    assert len(json.loads(path.with_suffix(".sampling.json").read_text())["samples"]) == 2


@pytest.mark.parametrize("failed_inner", [False, True])
def test_retained_evaluator_evidence_allows_normal_inner_feedback(harness, tmp_path, monkeypatch, failed_inner):
    import subprocess
    from agentic3d.evidence import collect_evidence
    from local_bench.evaluation import _discard_heavy_evaluation_artifacts

    h = harness
    h.args.rounds, h.primitive, h.inner, h.fail_inner = 1, "partition", 3, failed_inner
    h.driver._remote = lambda *args: subprocess.run(
        ["/bin/sh", "-c", " ".join(args[2:])], capture_output=True, text=True)
    reference = tmp_path / "reference-result"
    write_json(reference / "run/run.json", {"status": "success"})
    h.args.reference_run = str(reference / "run")
    evaluate = h.driver.evaluate_remote
    def physical(**kw):
        result, _ = evaluate(**kw)
        envelope = tmp_path / "physical" / str(len(h.physical)) / "result"
        result["artifacts"] = {"output_dir": str(envelope)}
        if result["metrics"]["runs_successfully"] == 0:
            result["artifacts"].update(error_type="ValueError", error_message="candidate rejected before run")
        else:
            write_json(envelope / "run/run.json", {"status": "success"})
            if kw["retention"] == "compact":
                _discard_heavy_evaluation_artifacts(envelope, result["artifacts"])
        write_json(envelope / "evaluation_result.json", result)
        (envelope / "stage.log").write_text("structured candidate failure" if failed_inner else "completed inner")
        # Compact success and pre-run candidate failure lack run/; full final keeps it.
        return result, str(envelope / "run")
    h.driver.evaluate_remote = physical
    monkeypatch.setattr(runner, "collect_evidence", collect_evidence)
    h.run()
    summary = json.loads((h.root / "summary.json").read_text())
    assert summary["final_evaluations"] == 1
    assert summary["inner_proxy_evaluations"] == (3 if failed_inner else 1)
    assert len(h.physical) == (4 if failed_inner else 2)
    assert summary["rejected_task_plans"] == (1 if failed_inner else 0)
    before = len(h.physical)
    h.run()
    assert len(h.physical) == before
