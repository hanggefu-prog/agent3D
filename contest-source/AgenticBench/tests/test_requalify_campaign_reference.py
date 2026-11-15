"""Zero-EDA recovery control-plane fixtures; no model/native tool processes."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import requalify_campaign_reference as module
from test_workspace_framework_identity import real_split_bundle


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / "compute"
    profile = source / "AgenticBench/local_bench/benchmark_inputs/bp_fe_lol_ng45_reference_20260923/config.yaml"
    write(profile, {"design": "bp_fe"})
    bundle = tmp_path / "v2/inputs/bp_fe/bundle"
    write(bundle / "benchmark_input.json", {})
    old = tmp_path / "v2/references/bp_fe/timing"
    evaluation = old / "reference/result"
    write(evaluation / "evaluation-contract.json", {})
    write(evaluation / "feedback.json", {"success": True})
    contract = {"case": "bp_fe", "endpoint": "timing", "task_digest": "a" * 64,
        "evaluator_digest": "b" * 64, "frozen_inputs": {"frozen": True},
        "physical": {"source_revision": "1" * 40, "image": "sha256:" + "2" * 64}}
    identity = {"physical_producer_revision": "1" * 40, "trusted_framework_revision": "3" * 40}
    monkeypatch.setattr(module, "validate_audit_framework", lambda *a, **kw: copy.deepcopy(identity))
    monkeypatch.setattr(module, "validate_benchmark_input", lambda *a, **kw:
                        {"config": str(profile), "run_dir": str(bundle / "run")})
    monkeypatch.setattr(module.baseline, "validate_reference_bundle", lambda *a, **kw: {})
    monkeypatch.setattr(module.baseline, "validate_audit_profile", lambda *a: profile.parent)
    monkeypatch.setattr(module.baseline, "validate_physical_prefix", lambda *a, **kw: {})
    monkeypatch.setattr(module, "task_digest", lambda *a: contract["task_digest"])
    monkeypatch.setattr(module, "evaluator_digest", lambda: contract["evaluator_digest"])
    monkeypatch.setattr(module, "expected_source_contract", lambda *a, **kw: copy.deepcopy(contract["physical"]))
    monkeypatch.setattr(module, "expected_frozen_inputs", lambda *a: copy.deepcopy(contract["frozen_inputs"]))
    old_contract = copy.deepcopy(contract)
    old_contract["evaluator_digest"] = "0" * 64
    config = {"kind": "campaign-reference-only-v1", "case": "bp_fe", "endpoint": "timing",
        "task_id": module.TIMING_TEN_TASK, "reference_policy": "FRESH", "source_commit": "1" * 40,
        "image": "sha256:" + "2" * 64, "benchmark_input": str(bundle), "search_dispatch": False,
        "candidate_finals": 0, "candidate_budget_consumed": 0, "model_calls": 0,
        "reference_selection": {"contract": old_contract}}
    summary = {"state": "failed", "error": module.REGISTRATION_ERROR, "case": "bp_fe",
               "endpoint": "timing", "accounting": {"registry_publications": 0}}
    write(old / "run_config.json", config)
    write(old / "summary.json", summary)
    calls = SimpleNamespace(qualify=0, register=0, load=0)
    options = SimpleNamespace(fail_qualify=False, fail_load=False, mutate_record=False, mutate_old=False,
                              uncertain_publication=False)
    payload = {"qualified": True, "executed_native_stages": [], "case": "bp_fe", "endpoint": "timing",
        "source_contract": contract["physical"], "frozen_inputs": contract["frozen_inputs"],
        "baseline": {"source_run": str(evaluation / "run"), "native_runtime_historical": {"opt": 2.5}}}
    def qualify(producer):
        calls.qualify += 1
        assert producer["evaluation_root"] == str(evaluation)
        if options.fail_qualify:
            raise ValueError("Native artifact hash mismatch")
        if options.mutate_old:
            (old / "summary.json").write_text(json.dumps(summary) + "\n")
        return copy.deepcopy(payload)
    def register(store, producer, *, origin, contract):
        calls.register += 1
        assert origin == "FRESH"
        assert contract["physical"]["source_revision"] == "1" * 40
        record = store / "records" / ("4" * 64 + ".json")
        write(record, {"origin": origin, "reference_id": "4" * 64, "qualification": payload})
        if options.uncertain_publication:
            raise OSError("Uncertain write completion")
        return record
    def load(path, *, contract, validate):
        calls.load += 1
        assert validate is qualify
        if options.fail_load:
            raise ValueError("Independent requalification failed")
        value = json.loads(path.read_text())
        if options.mutate_record:
            path.write_text(path.read_text() + "\n")
        return value
    monkeypatch.setattr(module, "qualify_producer", qualify)
    monkeypatch.setattr(module, "register_producer", register)
    monkeypatch.setattr(module, "load_reference", load)
    args = dict(source=source, source_revision="1" * 40, audit_framework_revision="3" * 40,
        case="bp_fe", image="sha256:" + "2" * 64, benchmark_input=bundle,
        existing_evaluation=evaluation, old_summary=old / "summary.json",
        reference_store=tmp_path / "v3-store", output=tmp_path / "v3/requalification/bp_fe")
    return SimpleNamespace(args=args, calls=calls, options=options, config=config, summary=summary,
                           old=old, contract=contract, payload=payload, identity=identity)


def test_requalification_preserves_old_failure_and_physical_identity(fixture):
    before = {p: p.read_bytes() for p in fixture.old.rglob("*") if p.is_file()}
    result = module.requalify(**fixture.args)
    assert result["state"] == "qualified"
    assert (fixture.calls.qualify, fixture.calls.register, fixture.calls.load) == (1, 1, 1)
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert result["contract"]["physical"]["source_revision"] == "1" * 40
    assert result["trusted_framework_revision"] == "3" * 40
    assert result["contract"]["evaluator_digest"] == "b" * 64
    for field in ("build_dispatches", "evaluator_container_dispatches", "model_calls", "candidate_finals",
                  "generated_model_candidates", "native_physical_attempts", "successful_evaluations",
                  "candidate_budget_consumed", "improvements"):
        assert result["accounting"][field] == 0
    assert result["accounting"]["registry_publications"] == 1
    assert result["accounting"]["historical_native_attempts_requalified"] == 1
    assert result["accounting"]["api_latency_seconds"] is None
    assert result["historical_native_runtime"] == {"opt": 2.5}


@pytest.mark.parametrize("field,value", [("case", "bp"), ("source_commit", "9" * 40),
    ("reference_policy", "ADOPT"), ("candidate_finals", 1), ("search_dispatch", True)])
def test_mismatched_old_attempt_rejected_before_output(fixture, field, value):
    fixture.config[field] = value
    write(fixture.old / "run_config.json", fixture.config)
    with pytest.raises(ValueError, match="identity differs"):
        module.requalify(**fixture.args)
    assert not fixture.args["output"].exists() and fixture.calls.qualify == 0


@pytest.mark.parametrize("field,value", [("state", "running"), ("error", "Native solver failed")])
def test_only_terminal_specific_registration_failure_can_be_recovered(fixture, field, value):
    fixture.summary[field] = value
    write(fixture.old / "summary.json", fixture.summary)
    with pytest.raises(ValueError, match="preserved, unpublished"):
        module.requalify(**fixture.args)


def test_physics_or_task_contract_cannot_change(fixture):
    fixture.config["reference_selection"]["contract"]["physical"]["source_revision"] = "9" * 40
    write(fixture.old / "run_config.json", fixture.config)
    with pytest.raises(ValueError, match="original task, frozen input or physical"):
        module.requalify(**fixture.args)


def test_recovery_failure_is_preserved_and_never_retried(fixture):
    fixture.options.fail_qualify = True
    result = module.requalify(**fixture.args)
    assert result["state"] == "failed" and fixture.calls.register == 0
    before = (fixture.args["output"] / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="no implicit retry"):
        module.requalify(**fixture.args)
    assert (fixture.args["output"] / "summary.json").read_bytes() == before
    assert fixture.calls.qualify == 1


@pytest.mark.parametrize("flag", ["fail_load", "mutate_record"])
def test_postpublication_failure_is_not_qualified(fixture, flag):
    setattr(fixture.options, flag, True)
    result = module.requalify(**fixture.args)
    assert result["state"] == "failed"
    assert result["accounting"]["registry_publications"] == 1
    assert not (fixture.args["output"] / "authority.json").exists()


def test_old_failure_mutation_prevents_publication(fixture):
    fixture.options.mutate_old = True
    result = module.requalify(**fixture.args)
    assert result["state"] == "failed" and "changed during recovery" in result["error"]
    assert fixture.calls.register == 0


def test_uncertain_publication_is_not_reported_as_zero_or_retried(fixture):
    fixture.options.uncertain_publication = True
    result = module.requalify(**fixture.args)
    assert result["state"] == "failed"
    assert result["accounting"]["registry_publications"] is None
    assert result["accounting"]["registry_publications_unknown"] == 1
    assert fixture.calls.register == 1 and fixture.calls.load == 0
    assert not (fixture.args["output"] / "authority.json").exists()
    with pytest.raises(FileExistsError):
        module.requalify(**fixture.args)


@pytest.mark.parametrize("target", ["output", "reference_store"])
def test_destinations_must_not_overlap_preserved_results(fixture, target):
    fixture.args[target] = fixture.old / "new"
    with pytest.raises(ValueError, match="disjoint"):
        module.requalify(**fixture.args)


def test_symlink_destination_fails_closed(fixture):
    output = fixture.args["output"]
    output.parent.mkdir(parents=True)
    output.symlink_to(fixture.old, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        module.requalify(**fixture.args)


def test_parent_traversal_cannot_bypass_disjointness(fixture):
    fixture.args["output"] = fixture.args["output"].parent / "extra/../bp_fe"
    with pytest.raises(ValueError, match="parent traversal"):
        module.requalify(**fixture.args)


def test_wrapper_has_no_native_or_model_dispatch_imports():
    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and ("runtime" in name or "workspace_codesign_evaluator" in name or
                             "qualify_campaign_reference" in name or "subprocess" in name) for name in imports)


def test_real_input_resolver_reaches_qualifier_across_checkouts(real_split_bundle, monkeypatch):
    """Regression for v3: real resolver returns audit-local, not compute-local.

    Only the independent native-qualification boundary is stopped here; input,
    profile, Git identity, archive and original physical-contract checks run.
    Synthetic fixture files are not claims about a native physical evaluation.
    """
    pair = real_split_bundle
    monkeypatch.setattr(module, "ROOT", pair.framework)
    image = sorted(module.baseline.IMAGE_IDS)[0]
    task = module.load_global_task(pair.framework / "global_tasks" / module.TIMING_TEN_TASK)
    contract = {"case": "bp_fe", "endpoint": "timing", "task_digest": module.task_digest(task.root),
        "evaluator_digest": "0" * 64,
        "frozen_inputs": module.expected_frozen_inputs("bp_fe", module.file_sha(pair.manifest)),
        "physical": module.expected_source_contract("bp_fe", "timing",
            profile=task.objective.evaluator_profile, source_revision=pair.source_revision, image=image)}
    old = pair.batch / "v2/references/bp_fe/timing"
    evaluation = old / "reference/result"
    write(evaluation / "evaluation-contract.json", {"source_revision": pair.source_revision})
    write(evaluation / "feedback.json", {"success": True})
    write(old / "run_config.json", {"kind": "campaign-reference-only-v1", "case": "bp_fe",
        "endpoint": "timing", "task_id": module.TIMING_TEN_TASK, "reference_policy": "FRESH",
        "source_commit": pair.source_revision, "image": image, "benchmark_input": str(pair.bundle),
        "search_dispatch": False, "candidate_finals": 0, "candidate_budget_consumed": 0,
        "model_calls": 0, "reference_selection": {"contract": contract}})
    write(old / "summary.json", {"state": "failed", "case": "bp_fe", "endpoint": "timing",
        "error": module.REGISTRATION_ERROR, "accounting": {"registry_publications": 0}})
    old_bytes = (old / "summary.json").read_bytes()
    validated = module.validate_benchmark_input(pair.bundle, task_id=task.objective.evaluator_profile)
    assert Path(validated["config"]) == pair.profile / "config.yaml"
    assert Path(validated["config"]) != pair.physical_profile / "config.yaml"
    calls = []
    def stop_at_native_qualifier(producer):
        calls.append(producer)
        raise RuntimeError("Synthetic test boundary: no native qualification requested")
    monkeypatch.setattr(module, "qualify_producer", stop_at_native_qualifier)
    result = module.requalify(source=pair.source, source_revision=pair.source_revision,
        audit_framework_revision=pair.audit_revision, case="bp_fe", image=image,
        benchmark_input=pair.bundle, existing_evaluation=evaluation, old_summary=old / "summary.json",
        reference_store=pair.batch / "v4-store", output=pair.batch / "v4/requalification/bp_fe")
    assert len(calls) == 1 and "Synthetic test boundary" in result["error"]
    assert result["accounting"]["registry_publications"] == 0
    assert result["accounting"]["native_physical_attempts"] == 0
    assert (old / "summary.json").read_bytes() == old_bytes
