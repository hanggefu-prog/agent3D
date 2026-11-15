"""Offline task-definition tests. No candidate execution, EDA or model calls."""
from copy import deepcopy
import shutil
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agentic3d.contracts import LOCAL_TASK_TEMPLATES
from local_bench.cli import main
from local_bench.evaluation import evaluate_task
from local_bench.hbt_policy import materialize_policy, score_timing, validate_policy
from local_bench.model import EvaluationResult, load_candidate, load_task, require_executable_task
from local_bench.optimization_feedback import optimization_result
from local_bench.program_bridge import materialize_program
from local_bench.prompting import export_openevolve_bundle, feedback_context, _diversity_themes
from local_bench.routing_baseline import TASKS as ROUTING_TASKS
from local_bench.search_api import CandidateFeedback, ProgramEvaluationHarness, TaskEvaluationOptions
from local_bench.search_session import prepare_search_problem, search_feedback, run_task_search
from local_bench.task_inventory import CORE_TASK_DIRECTORIES
from local_bench.verifier import verify_candidate

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "AgenticBench/local_tasks/hbt_assignment_policy"
INITIAL = (TASK / "initial_program.py").read_text()


def policy(priority_body, site_body='return site_context["combined_hpwl_dbu"]'):
    def indent(body):
        return "\n".join("    " + line for line in body.splitlines())
    return ("def net_priority(context):\n" + indent(priority_body)
            + "\n\ndef site_cost(net_context, site_context):\n" + indent(site_body) + "\n")


def test_initial_adapter_literal_and_scope_agree():
    spec = load_task(TASK)
    assert spec.availability == "available"
    initial = load_candidate(spec.initial_candidate)
    assert materialize_program(spec, INITIAL, ROOT) == initial == materialize_policy(INITIAL)
    assert verify_candidate(spec, initial)["candidate"] == initial
    assert not spec.allowed_source_paths
    assert not spec.allowed_config_paths
    raw = yaml.safe_load((TASK / "task.yaml").read_text())
    assert raw["execution"]["upstream_stages"] == ["input", "floorplan", "partition", "placement", "pdn"]
    definition = raw["definition"]
    assert definition["upstream_partition_reference"] == "global-2way"
    assert definition["endpoint"]["opt_passes"] == 0
    assert definition["endpoint"]["directory"] == "55_post_placement_sta"
    assert definition["candidate_search"]["expansion"] == "double_only_until_a_legal_candidate_exists"
    assert definition["admission"]["per_network_timing_context"] == "not_exposed_in_v1"


@pytest.mark.parametrize("body", [
    'return context["pin_count"] / max(1, context["die_span_dbu"])',
    'a = context["top_bbox_hpwl_dbu"]\nif a > 0:\n    return a\nelse:\n    return 0',
    'if context["pin_count"] > 3:\n    a = 2\nelse:\n    a = 1\nreturn a',
    'if context["pin_count"] > 3:\n    return 2\nelse:\n    a = 1\nreturn a',
    'return abs(context["top_bbox_hpwl_dbu"] - context["bottom_bbox_hpwl_dbu"])',
])
def test_bounded_geometry_policies_are_accepted(body):
    assert validate_policy(policy(body))


@pytest.mark.parametrize("body", [
    'import os\nreturn 1',
    'return open("/tmp/should-not-exist", "w")',
    'return context["timing_weight"]',
    'return context.get("pin_count")',
    'context = 1\nreturn context',
    'context["pin_count"] = 1\nreturn 1',
    'return context',
    'return context["pin_count"] ** 2',
    'return 1e99',
    'return 1e999',
    'for i in range(2):\n    a = i\nreturn a',
    'return net_priority(context)',
    'return [context["pin_count"]]',
    'return sum([1, 2])',
    'return max(1)',
    'return min(a=1, b=2)',
    'if context["pin_count"] > 3:\n    a = 2\nreturn a',
    'if context["pin_count"] > 3:\n    return 2',
    'a = a + 1\nreturn a',
    'return 1\na = 2',
    'def helper():\n    return 1\nreturn 1',
    '_private = 1\nreturn _private',
    'abs = 1\nreturn abs',
])
def test_unsupported_policies_are_rejected_without_execution(body):
    with pytest.raises(ValueError):
        validate_policy(policy(body))


@pytest.mark.parametrize("source", [
    INITIAL.replace("net_priority(context)", "net_priority(context=1)"),
    INITIAL.replace("net_priority(context)", "net_priority(context: dict)"),
    INITIAL.replace("net_priority(context)", "net_priority(ctx)"),
    INITIAL + "\ndef helper():\n    return 1\n",
    "@staticmethod\n" + policy("return 1"),
    "# " + "a" * 16001 + "\n" + INITIAL,
    policy("\n".join("a = 1" for _ in range(180)) + "\nreturn a"),
])
def test_signature_and_size_limits(source):
    with pytest.raises((ValueError, SyntaxError)):
        validate_policy(source)


@pytest.mark.parametrize("mutation", ["source", "interface", "extra", "override", "missing"])
def test_literal_candidate_cannot_bypass_policy_verifier(mutation):
    candidate = deepcopy(materialize_policy(INITIAL))
    data = candidate["candidate_files"]["hbt_policy.json"]
    if mutation == "source":
        data["source"] = policy('return __import__("os")')
    elif mutation == "interface":
        data["interface_version"] = "untrusted-v2"
    elif mutation == "extra":
        data["precomputed_assignments"] = {"net": 1}
    elif mutation == "override":
        candidate["config_overrides"] = {"hbt": {"legalizer": "another"}}
    else:
        candidate["candidate_files"] = {}
    with pytest.raises(ValueError):
        verify_candidate(load_task(TASK), candidate)


def timing(wns=-2, tns=-100):
    return {"wns_setup": wns, "tns_setup": tns}


@pytest.mark.parametrize("candidate, reference, expected", [
    (timing(), timing(), 1),
    (timing(-1, -50), timing(), 2),
    (timing(-1, -100), timing(), 1 / .7),
    (timing(-4, -50), timing(), 1 / 1.4),
    (timing(0, 0), timing(0, 0), 1),
    (timing(2, 0), timing(0, 0), 1),
    (timing(-1e-12, -1e-12), timing(0, 0), 1),
    (timing(-2e-12, 0), timing(0, 0), 1 / 2.2),
    (timing(0, 0), timing(), 1e12),
])
def test_score_contract(candidate, reference, expected):
    result = score_timing(candidate, reference)
    assert result["combined_score"] == pytest.approx(expected)
    assert result["final_global_wns_setup_ns"] == candidate["wns_setup"]
    assert "runs_successfully" not in result  # Pure arithmetic grants no physical qualification.
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [None, True, "0", float("nan"), float("inf")])
@pytest.mark.parametrize("side", ["candidate", "reference"])
def test_missing_or_invalid_timing_never_scores(bad, side):
    candidate, reference = timing(), timing()
    (candidate if side == "candidate" else reference)["wns_setup"] = bad
    with pytest.raises(ValueError):
        score_timing(candidate, reference)


def test_positive_tns_and_cost_overflow_are_rejected():
    with pytest.raises(ValueError):
        score_timing(timing(tns=1), timing())
    with pytest.raises(ValueError):
        score_timing(timing(-1e308, -1e308), timing(0, 0))


def test_model_context_style_and_feedback_contract(tmp_path):
    spec = load_task(TASK)
    options = TaskEvaluationOptions(TASK, tmp_path / "session", open3dbench_root=ROOT)
    problem = prepare_search_problem(options)
    assert problem.metadata["availability"] == "available"
    headers = ("## Background", "## Your task", "## How results are compared", "## Inputs and output", "## Constraints")
    assert [problem.description.index(h) for h in headers] == sorted(problem.description.index(h) for h in headers)
    for text in (problem.description, problem.system_message):
        for forbidden in (str(ROOT), "sha256", "ADOPT", "FRESH", "manifest", "causal", "versus frozen"):
            assert forbidden not in text
        assert "hybrid bonding terminal (hbt)" in " ".join(text.lower().split())
    assert "logic-on-logic (LoL)" in problem.description
    assert "timing_weight" not in problem.description
    assert "timing-valid" not in " ".join(_diversity_themes(spec))
    metrics = {"combined_score": 1.1, "final_global_wns_setup_ns": -1,
               "unassigned_count": 0, "legal_combined_hpwl_dbu": 100,
               "hbt_runtime_seconds": None, "policy_calls": 100,
               "grt_total_overflow": 0, "api_latency_seconds": 1}
    result = EvaluationResult(metrics=metrics, artifacts={"log": "/private/raw.log"})
    projected = optimization_result(result, task_id=spec.task_id)
    assert projected.metrics == {k: metrics[k] for k in (
        "combined_score", "final_global_wns_setup_ns", "unassigned_count", "legal_combined_hpwl_dbu")}
    feedback = CandidateFeedback("candidate", "sha", "accepted", 1.1, metrics=metrics, artifacts=result.artifacts)
    assert search_feedback(problem, feedback).metrics == projected.metrics
    rendered = feedback_context([feedback.to_dict()], spec)
    assert "/private/raw.log" not in rendered and "policy_calls" not in rendered
    assert "Missing metrics remain unknown" in rendered
    assert not options.output_dir.exists()


@pytest.mark.parametrize("mode", ["full", "interface"])
def test_definition_cannot_start_physical_harness(tmp_path, mode, definition_task):
    options = TaskEvaluationOptions(definition_task, tmp_path / "blocked", evaluation_mode=mode, open3dbench_root=ROOT)
    with pytest.raises(ValueError, match="definition-only"):
        ProgramEvaluationHarness(options, evaluation_transport=lambda **kw: pytest.fail("Native transport called"))
    assert not options.output_dir.exists()
    backend = SimpleNamespace(ask=lambda context: pytest.fail("Search backend called"))
    with pytest.raises(ValueError, match="definition-only"):
        run_task_search(backend, options, max_evaluations=1)


def test_definition_cannot_evaluate_or_export(tmp_path, definition_task):
    spec = load_task(definition_task)
    with pytest.raises(ValueError, match="definition-only"):
        evaluate_task(definition_task, spec.initial_candidate, ROOT, tmp_path / "eval", dry_run=True)
    with pytest.raises(ValueError, match="definition-only"):
        export_openevolve_bundle(spec, load_candidate(spec.initial_candidate), ROOT, tmp_path / "bundle")
    assert not (tmp_path / "eval").exists() and not (tmp_path / "bundle").exists()


def test_offline_verify_is_still_available(tmp_path):
    options = TaskEvaluationOptions(TASK, tmp_path / "verify", evaluation_mode="verify", open3dbench_root=ROOT)
    harness = ProgramEvaluationHarness(options, evaluation_transport=lambda **kw: pytest.fail("Native transport called"))
    assert harness.evaluate_program(INITIAL) is not None
    assert harness.records[-1]["status"] == "accepted"
    projected = optimization_result(EvaluationResult(
        metrics=harness.records[-1]["metrics"], artifacts={}), task_id="hbt_assignment_policy")
    assert projected.metrics["verifier_only"] == 1
    assert "synthetic protocol-test values" in projected.artifacts["optimization_feedback"]


def test_smoke_guard_precedes_provider_or_native_construction(tmp_path, definition_task):
    from scripts import run_local_task_smoke as smoke

    def forbidden(*args, **kwargs):
        pytest.fail("Definition-only smoke constructed a provider or native transport")

    with pytest.raises(ValueError, match="definition-only"):
        smoke.run_smoke(SimpleNamespace(framework=definition_task.parent.parent),
                        task="hbt_assignment_policy", output=tmp_path / "blocked", rounds=2,
                        provider_factory=forbidden, transport_factory=forbidden)
    assert not (tmp_path / "blocked").exists()


def test_core_inventory_and_legacy_visibility(capsys):
    assert len(CORE_TASK_DIRECTORIES) == 10
    assert "hbt_assignment_policy" in CORE_TASK_DIRECTORIES
    assert "global_routing_net_ordering" not in CORE_TASK_DIRECTORIES
    assert "global_routing_net_ordering" not in LOCAL_TASK_TEMPLATES
    assert "hbt_assignment_policy" in LOCAL_TASK_TEMPLATES
    assert "global_routing_net_ordering" not in ROUTING_TASKS
    assert main(["list", "--core"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 10
    assert next(r for r in rows if r["id"] == "hbt_assignment_policy")["availability"] == "available"
    main(["list"])
    assert all(r["id"] != "global_routing_net_ordering_source" for r in json.loads(capsys.readouterr().out))
    with pytest.raises(FileNotFoundError):
        load_task(ROOT / "AgenticBench/local_tasks/global_routing_net_ordering")


@pytest.fixture
def definition_task(tmp_path):
    """Keep the generic not-yet-implemented-task guard covered after HBT promotion."""
    target = tmp_path / "framework/local_tasks/hbt_assignment_policy"
    shutil.copytree(TASK, target)
    path = target / "task.yaml"
    data = yaml.safe_load(path.read_text())
    data["availability"] = "definition-only"
    data["base_config"] = str(load_task(TASK).base_config)
    path.write_text(yaml.safe_dump(data))
    return target
