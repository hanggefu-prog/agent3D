"""Real selection/store/verifier dispatch with synthetic physical evidence.

No Docker, model or physical flow is invoked. Only the producer/input validators
are fixture boundaries; source identity, contracts, record seals and native
candidate metric/pass checks use their production implementation.
"""
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from global_bench.model import load_global_task
from global_bench.verifiers import open3dbench_codesign_verifier as verifier
from local_bench import reference_baseline as baseline, reference_catalog as catalog
from local_bench.reference_store import file_sha, publish_reference
from scripts import workspace_reference as selection
from scripts.workspace_codesign_evaluator import WorkspaceEvaluator
from test_one_pass_codesign_contract import native_one_pass
from test_sta_codesign import _write_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    git("init", "-q")
    (source / "algorithm.py").write_text("value = 1\n")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "initial")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "benchmark_input.json").write_text('{"fixture":"frozen-input"}\n')
    monkeypatch.setattr(selection, "validate_benchmark_input", lambda p: {"run_dir": str(bundle)})
    monkeypatch.setattr(baseline, "validate_reference_bundle", lambda *a: {})
    monkeypatch.setattr(baseline, "validate_physical_prefix", lambda *a: {})
    runtime = SimpleNamespace(source=source, source_commit=git("rev-parse", "HEAD"),
        image=sorted(baseline.IMAGE_IDS)[0], batch=tmp_path / "batch", reference_policy="FRESH",
        reference_store=tmp_path / "registry", reference_record=None)
    task = load_global_task(ROOT / "global_tasks/lol_partition_placement_timing_one_pass")
    calls = []
    payloads = {}
    def qualify(producer):
        calls.append(producer)
        return copy.deepcopy(payloads[producer["run_root"]])
    monkeypatch.setattr(selection, "_qualify", qualify)
    monkeypatch.setattr(catalog, "qualify_producer", qualify)
    return SimpleNamespace(runtime=runtime, task=task, bundle=bundle, calls=calls,
                           payloads=payloads, tmp=tmp_path, git=git)


def choose(f):
    return selection.WorkspaceReference(f.runtime, "bp_fe", "timing", f.bundle, f.task)


def fresh(f):
    selected = choose(f)
    class Evaluator:
        calls = 0
        def evaluate(self, subject, output, reference):
            self.calls += 1
            assert reference is None and subject == f.runtime.source
            run = output / "run"
            run.mkdir(parents=True)
            artifact = run / "native.log"
            artifact.write_text("Fixture only, not EDA evidence.\n")
            raw = dict(wns_setup_ns=-1., tns_setup_ns=-10., wns_deficit=1., tns_deficit=10.,
                       boundary_budget_model="passive-cutpoint", opt_passes=1)
            f.payloads[str(run)] = {
                "qualified": True, "case": "bp_fe", "endpoint": "timing",
                "source_contract": selected.contract["physical"],
                "frozen_inputs": selected.contract["frozen_inputs"],
                "source_identity": {"fixture_revision": f.runtime.source_commit},
                "baseline": {"raw_metrics": raw, "source_run": str(run),
                             "native_runtime_historical": {"opt": 25.}},
                "evidence": [{"path": str(artifact), "bytes": artifact.stat().st_size,
                              "sha256": file_sha(artifact), "role": "fixture"}]}
            return {"success": True, "metrics": {"combined_score": 1.}, "run": str(run)}
    evaluator = Evaluator()
    result = selected.obtain(evaluator, f.tmp / "fresh")
    return selected, evaluator, result


@pytest.mark.parametrize("rounds", [5, 10])
def test_fresh_publishes_once_then_adopt_never_dispatches(fixture, rounds):
    f = fixture
    from scripts.workspace_task_selection import load_registered_task, workspace_task_id
    f.task = load_registered_task(workspace_task_id("timing", rounds), objective="timing")
    selected, evaluator, result = fresh(f)
    assert evaluator.calls == selected.evaluator_invocations == 1
    record = selected.record_path
    original = record.read_bytes()
    f.runtime.reference_policy = "ADOPT"
    f.runtime.batch = f.tmp / "different-batch"
    adopted = choose(f)
    assert adopted.to_dict()["record_sha256"] == file_sha(record)
    assert adopted.evaluator_invocations == 0
    class Forbidden:
        def evaluate(self, *a):
            pytest.fail("ADOPT must not dispatch physical evaluation")
    received = adopted.obtain(Forbidden(), f.tmp / "adoption")
    assert received["metrics"]["combined_score"] == 1.
    assert received["reference_evaluator_invocations"] == received["new_reference_native_attempts"] == 0
    assert received["historical_native_runtime"] == {"opt": 25.}
    assert record.read_bytes() == original
    assert len(f.calls) >= 4  # publication, post-publication, selection, adoption


def test_default_adopt_missing_record_has_no_fresh_fallback(fixture):
    f = fixture
    del f.runtime.reference_policy
    with pytest.raises(ValueError, match="no automatic FRESH/EXTEND"):
        choose(f)
    assert not f.calls


def test_extend_fails_before_source_input_or_native(fixture, monkeypatch):
    f = fixture
    f.runtime.reference_policy = "EXTEND"
    monkeypatch.setattr(selection, "_source_identity", lambda *a: pytest.fail("Unexpected source dispatch"))
    with pytest.raises(ValueError, match="No flow was dispatched"):
        choose(f)


def test_archive_algorithms_not_inferred_from_matching_config(fixture, monkeypatch):
    f = fixture
    path = f.tmp / "archive.json"
    path.write_text(json.dumps({"producer": {"kind": "frozen-20260923-v1"}}))
    f.runtime.reference_policy, f.runtime.reference_record = "ADOPT", path
    monkeypatch.setattr(selection, "_source_identity", lambda *a: pytest.fail("Unexpected source dispatch"))
    with pytest.raises(ValueError, match="stage-specific algorithm/backend"):
        choose(f)


@pytest.mark.parametrize("mutation", ["dirty", "commit", "image", "bundle"])
def test_current_source_and_frozen_identity_must_match(fixture, mutation):
    f = fixture
    fresh(f)
    f.runtime.reference_policy = "ADOPT"
    if mutation == "dirty":
        (f.runtime.source / "algorithm.py").write_text("value = 2\n")
    elif mutation == "commit":
        f.runtime.source_commit = "0" * 40
    elif mutation == "image":
        f.runtime.image = sorted(baseline.IMAGE_IDS)[1]
    else:
        (f.bundle / "benchmark_input.json").write_text("different frozen import")
    with pytest.raises(ValueError):
        choose(f)


def test_evaluator_rejects_reference_record_changed_after_selection(fixture):
    f = fixture
    selected, _, result = fresh(f)
    evaluator = object.__new__(WorkspaceEvaluator)
    evaluator.runtime = f.runtime
    assert evaluator._qualified_reference(result["run"]) == Path(result["run"])
    path = selected.record_path
    path.chmod(0o644)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed after run configuration"):
        evaluator._qualified_reference(result["run"])


@pytest.mark.parametrize("boundary", ["preflight", "adoption", "candidate"])
def test_record_bytes_cannot_change_during_qualification(fixture, monkeypatch, boundary):
    f = fixture
    selected, _, result = fresh(f)
    f.runtime.reference_policy = "ADOPT"
    adopted = choose(f) if boundary == "adoption" else None
    path = selected.record_path
    original = selection._qualify
    def mutate(producer):
        qualified = original(producer)
        path.chmod(0o644)
        path.write_text(path.read_text() + " ")
        return qualified
    monkeypatch.setattr(selection, "_qualify", mutate)
    monkeypatch.setattr(catalog, "qualify_producer", mutate)
    with pytest.raises(ValueError, match="changed during"):
        if boundary == "preflight":
            choose(f)
        elif boundary == "adoption":
            adopted.obtain(None, f.tmp / "interrupted-adoption")
        else:
            evaluator = object.__new__(WorkspaceEvaluator)
            evaluator.runtime = f.runtime
            evaluator._qualified_reference(result["run"])


def test_standalone_verifier_requalifies_reference_and_keeps_candidate_gates(fixture, monkeypatch):
    f = fixture
    selected, _, result = fresh(f)
    candidate = f.tmp / "candidate"
    _write_run(candidate, -0.5, -5.)
    native_one_pass(candidate)
    (candidate / "run.json").write_text(json.dumps({"design": "bp_fe"}))
    # Native hash/seed validators have separate exhaustive fixture suites.
    monkeypatch.setattr(verifier, "_validate_chain", lambda *a, **k: None)
    monkeypatch.setattr(verifier, "_validate_seed_contract", lambda *a, **k: None)
    contract = f.tmp / "evaluation-contract.json"
    contract.write_text(json.dumps({"case": "bp_fe", "task_digest": f.task.digest(),
        "evaluation_endpoint": {"profile": f.task.objective.evaluator_profile,
                                "task_id": f.task.task_id},
        "source_revision": f.runtime.source_commit, "image": f.runtime.image,
        "input_manifest_sha256": file_sha(f.bundle / "benchmark_input.json")}))
    submission = {"task_id": f.task.task_id, "task_digest": f.task.digest(),
                  "base_revision": f.runtime.source_commit}
    def score():
        return verifier.score_with_reference_record(candidate, Path(result["run"]), selected.record_path,
            contract, mode="timing", profile=f.task.objective.evaluator_profile, submission=submission)
    assert score()["combined_score"] == pytest.approx(2.)
    opt = candidate / "60_opt/metrics.json"
    value = json.loads(opt.read_text())
    value["passes_run"] = 2
    opt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exactly 1 OPT"):
        score()
    submission["task_digest"] = "0" * 64
    with pytest.raises(ValueError, match="trusted candidate task"):
        score()
