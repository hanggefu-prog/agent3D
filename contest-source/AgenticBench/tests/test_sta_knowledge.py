import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic3d.knowledge import KnowledgeStore, digest, validate_skill, write_json, read_package
from agentic3d.sta_protocol import (validate_task, proxy_score, PARTITION_PATH,
                                  ANALYZER_SCHEMA, LOCAL_SCHEMA, REFLECTION_SCHEMA, REVIEW_SCHEMA)
from agentic3d import sta_knowledge_runner as runner
from scripts import run_sta_codesign_comparison as driver


SEED = {"skills/initial/SKILL.md": "immutable seed"}
SPLIT = {"train": ["bp", "swerv_wrapper"], "test": ["ariane133"]}


def skill():
    return {"id": "critical-tail-diagnosis", "description": "Diagnose critical-tail proxy mismatch",
            "files": {"SKILL.md": "---\nname: critical-tail-diagnosis\ndescription: Diagnose critical-tail proxy mismatch\n---\n"
                      "Inspect pass1/pass2 logs; distinguish observed effects from causal hypotheses.\n"}}


def task(stage="partition", evaluation="partition", task_id="new-critical-tail-task"):
    spec = driver.load_task(driver.STA_PROFILE)
    paths = list(spec.allowed_source_paths) if stage == "joint" else (
        [PARTITION_PATH] if stage == "partition" else [p for p in spec.allowed_source_paths if p != PARTITION_PATH])
    metric = "timing_weighted_cut" if evaluation == "partition" else "placement_native_hpwl"
    return {"id": task_id, "description": "New reusable tail/geometry task", "stage": stage,
            "source_scope": paths, "evaluation": evaluation,
            "objective": [] if evaluation == "final" else [{"metric": metric, "weight": 1., "scale_floor": 1e-9}],
            "procedure": "Inspect measured evidence, refine mechanism, compare outcomes",
            "validation": "Unchanged full global gates; require actual pass2 timing"}


def test_fresh_repeats_never_load_previous_evolution(tmp_path):
    first = KnowledgeStore.initialize(tmp_path / "first", seed_files=SEED, case="bp")
    first.save("round-01", [skill()], [task()], {"validation": "proposed"})
    second = KnowledgeStore.initialize(tmp_path / "second", seed_files=SEED, case="bp")
    assert len(first.records()) == 1 and second.records() == []
    assert first.manifest["seed_sha256"] == second.manifest["seed_sha256"]
    assert (first.root / "evolved/round-01/skills/critical-tail-diagnosis/SKILL.md").is_file()
    assert (first.root / "evolved/round-01/tasks/new-critical-tail-task.json").is_file()
    assert json.loads((first.root / "seed/snapshot.json").read_text()) == SEED


def training_package(tmp_path):
    train = KnowledgeStore.initialize(tmp_path / "train", seed_files=SEED,
                                     case="bp", phase="train", split=SPLIT)
    train.save("round-01", [skill()], [task()], {"validation": "observed", "outer_score": 1.1})
    # These must never be discovered/exported.
    (train.root / "session.json").write_text('"session-secret"')
    (train.root / "candidate.py").write_text("candidate-secret")
    train.export(tmp_path / "package.json")
    return tmp_path / "package.json"


def test_train_export_frozen_test_and_optional_adaptation(tmp_path):
    package = training_package(tmp_path)
    original = package.read_bytes()
    assert b"session-secret" not in original and b"candidate-secret" not in original
    test = KnowledgeStore.initialize(tmp_path / "test", seed_files=SEED, case="ariane133",
                                     phase="test", split=SPLIT, imported=package)
    test.save("round-01", [skill()], [task(task_id="test-local")], {})
    assert len(test.records()) == 1
    assert (test.root / "test-proposals/round-01/record.json").exists()
    with pytest.raises(ValueError, match="only training"):
        test.export(tmp_path / "leak.json")
    adapted = KnowledgeStore.initialize(tmp_path / "adapted", seed_files=SEED,
        case="ariane133", phase="test", split=SPLIT, imported=package, adapt_test=True)
    adapted.save("round-01", [], [task(task_id="test-local")], {})
    assert len(adapted.records()) == 2 and package.read_bytes() == original
    again = KnowledgeStore.initialize(tmp_path / "another-test", seed_files=SEED,
        case="ariane133", phase="test", split=SPLIT, imported=package)
    assert len(again.records()) == 1


def test_sequential_training_preserves_lineage(tmp_path):
    package = training_package(tmp_path)
    train2 = KnowledgeStore.initialize(tmp_path / "train2", seed_files=SEED,
        case="swerv_wrapper", phase="train", split=SPLIT, imported=package)
    train2.save("round-01", [], [task(task_id="another-task")], {})
    train2.export(tmp_path / "package2.json")
    assert read_package(tmp_path / "package2.json")["payload"]["training_cases"] == ["bp", "swerv_wrapper"]


def test_independent_cannot_import_and_exports_cannot_overwrite(tmp_path):
    package = training_package(tmp_path)
    with pytest.raises(ValueError, match="independent"):
        KnowledgeStore.initialize(tmp_path / "independent", seed_files=SEED, case="bp", imported=package)
    store = KnowledgeStore(tmp_path / "train")
    store.save("round-02", [], [task(task_id="another-task")], {})
    with pytest.raises(ValueError, match="overwrite"):
        store.export(package)


def test_explicit_seed_selects_only_pregiven_snapshot(tmp_path):
    training_package(tmp_path)
    source = tmp_path / "train/seed/snapshot.json"
    actual = runner.configured_seed(SimpleNamespace(knowledge_seed=source), tmp_path)
    assert actual == SEED
    assert "critical-tail-diagnosis" not in json.dumps(actual)


def test_overlap_tamper_seed_drift_and_resume_rejected(tmp_path):
    package = training_package(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        KnowledgeStore.initialize(tmp_path / "bad", seed_files=SEED, case="bp", phase="test",
            split={"train": ["bp"], "test": ["bp"]}, imported=package)
    with pytest.raises(ValueError, match="lineage"):
        KnowledgeStore.initialize(tmp_path / "bad2", seed_files=SEED, case="bp", phase="test",
            split={"train": ["swerv_wrapper"], "test": ["bp"]}, imported=package)
    with pytest.raises(ValueError, match="different seed"):
        KnowledgeStore.initialize(tmp_path / "bad3", seed_files={"changed.md": "x"},
            case="ariane133", phase="test", split=SPLIT, imported=package)
    first = KnowledgeStore.initialize(tmp_path / "fresh", seed_files=SEED, case="bp")
    with pytest.raises(ValueError, match="resume"):
        KnowledgeStore.initialize(first.root, seed_files=SEED, case="ariane133")
    (first.root / "seed/snapshot.json").write_text('{"changed": "x"}')
    with pytest.raises(ValueError, match="seed snapshot"):
        KnowledgeStore(first.root)
    envelope = json.loads(package.read_text())
    envelope["payload"]["training_cases"] = ["ariane133"]
    package.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="checksum"):
        read_package(package)


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/x.py", "scripts/../../x.py", "scripts/x.sh"])
def test_skill_resource_paths_are_bounded(path):
    value = skill()
    value["files"][path] = "print(1)"
    with pytest.raises(ValueError):
        validate_skill(value)


def test_new_tasks_are_not_limited_to_seed_ids_and_joint_is_supported():
    spec = driver.load_task(driver.STA_PROFILE)
    assert validate_task(task(), spec)["id"] == "new-critical-tail-task"
    assert validate_task(task("joint", "final"), spec)["stage"] == "joint"
    with pytest.raises(ValueError, match="require final"):
        validate_task(task("joint", "partition"), spec)
    bad = task()
    bad["source_scope"].append("OpenROAD/STA.cpp")
    with pytest.raises(ValueError, match="global boundary"):
        validate_task(bad, spec)
    bad = task()
    bad["objective"][0]["metric"] = "final_global_wns_setup_ns"
    with pytest.raises(ValueError, match="unsupported"):
        validate_task(bad, spec)


@pytest.mark.parametrize("invalid", [float("nan"), -1., True, 0.])
def test_proxy_objective_weights_must_be_finite_positive(invalid):
    value = task()
    value["objective"][0]["weight"] = invalid
    with pytest.raises(ValueError):
        validate_task(value, driver.load_task(driver.STA_PROFILE))


def test_proxy_scoring_cannot_accept_failed_or_missing_evidence():
    value = task()
    reference = {"timing_weighted_cut": 100.}
    assert proxy_score(value, {"metrics": {"runs_successfully": 1, "timing_weighted_cut": 80.}}, reference) == 1.25
    assert proxy_score(value, {"metrics": {"runs_successfully": 0, "timing_weighted_cut": 1.}}, reference) is None
    assert proxy_score(value, {"metrics": {"runs_successfully": 1}}, reference) is None


def test_evidence_collector_reads_only_run_artifacts_and_marks_truncation(tmp_path):
    import subprocess
    run = tmp_path / "run"
    stage = run / "30_placement"
    stage.mkdir(parents=True)
    (stage / "run.log").write_text("x" * 20000 + "\napi_key=secret\nlast iteration 2000\n")
    external = tmp_path / "private.log"
    external.write_text("must-not-leak")
    (stage / "escape.log").symlink_to(external)
    fake = SimpleNamespace(_remote=lambda *args: subprocess.run(
        ["/bin/sh", "-c", " ".join(args[2:])],
        capture_output=True, text=True))
    report = runner.collect_evidence(fake, str(run), tmp_path / "evidence")
    assert report["error"] is None and len(report["files"]) == 1
    assert report["files"][0]["truncated"]
    text = Path(report["files"][0]["local_file"]).read_text()
    assert "secret" not in text and "must-not-leak" not in text and "last iteration 2000" in text


@pytest.mark.parametrize("failed_round", [None, 2])
@pytest.mark.parametrize("phase", ["independent", "train", "test"])
def test_five_round_v2_mock_end_to_end_and_resume(tmp_path, monkeypatch, failed_round, phase):
    """Exercise real materialization, scope verification, merge and git apply checks.

    Model and physical evaluator alone are fakes; this does not claim STA gains.
    """
    spec = driver.load_task(driver.STA_PROFILE)
    source = tmp_path / "source"
    for name in spec.allowed_source_paths:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("value = 0\n")
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Exact frozen task fixture")
    root = tmp_path / "experiment"
    root.mkdir()
    args = SimpleNamespace(agent_workspace=source, knowledge_split=None,
        knowledge_phase="independent", knowledge_import=None, test_knowledge_adaptation=False,
        case="bp", instructions=instructions, reference_run="/isolated/reference/run",
        rounds=5, model="fake-model", codex_bin="fake-codex", inner_tolerance=1e-6,
        gpu=0, knowledge_export=None)
    if phase != "independent":
        split = tmp_path / "split.json"
        write_json(split, SPLIT)
        args.knowledge_phase, args.knowledge_split = phase, split
        if phase == "train":
            args.knowledge_export = tmp_path / "export.json"
        else:
            train = KnowledgeStore.initialize(tmp_path / "train", seed_files=runner.seed_snapshot(driver.ROOT),
                                              case="bp", phase="train", split=SPLIT)
            train.save("trained", [skill()], [], {"author": "global-analyzer"})
            args.knowledge_import = tmp_path / "import.json"
            train.export(args.knowledge_import)
            args.case = "ariane133"
    calls, evaluations = [], []
    plan_count = 0
    local_count = 0
    review_interrupted = False
    class Agent:
        def __init__(self, name, path, schema, *unused):
            self.name, self.root, self.schema = name, path, schema
        def ask(self, prompt):
            nonlocal plan_count, local_count, review_interrupted
            calls.append((self.name, prompt))
            if self.schema == ANALYZER_SCHEMA:
                plan_count += 1
                state = json.loads(prompt.split("STATE:\n", 1)[1])
                selection = []
                if plan_count > 1:
                    learned = next(r for r in state["knowledge_catalog"] if r["id"] == "critical-tail-diagnosis")
                    selection = [{"ref": learned["ref"], "reason": "Reuse measured diagnosis",
                                  "attachments": [learned["resources"][0]["path"]]}]
                active = task("joint", "final") if plan_count in (3, 5) else task(
                    "placement", "placement") if plan_count == 2 else task()
                return dict(hypothesis="test mechanism", evidence="fixture evidence",
                    evidence_paths=["evidence/index.json"], local_task=active,
                    specialist_instruction="Inspect logs and improve algorithm",
                    max_inner_attempts=3 if plan_count == 4 else 1,
                    knowledge_refs=selection, skills=[], tasks=[])
            if self.schema == REFLECTION_SCHEMA:
                context = json.loads(prompt.split("\n")[-1])
                evidence = json.loads((root / "agent-workspace" / context["candidate_evidence"]["path"]).read_text())
                assert digest(evidence) == context["candidate_evidence"]["sha256"]
                assert evidence["selected_candidate"]["source_patch"]
                assert evidence["local_attempts"][0]["candidate_sha256"]
                return dict(assessment="mock outcome; no physical claim", observations=[], suggestions=[
                    dict(target="skill", change="Distinguish proxy gains from final timing",
                         rationale="Different physical mechanisms", evidence_paths=[context["candidate_evidence"]["path"]],
                         uncertainty="Not independently generalized")])
            if self.schema == REVIEW_SCHEMA:
                context = json.loads(prompt.split("\n")[-1])
                if failed_round and context["budget"]["remaining_final"] == 0 and not review_interrupted:
                    review_interrupted = True
                    raise RuntimeError("fixture review interruption")
                raw = skill()
                raw["files"] = [{"path": p, "content": c} for p, c in raw["files"].items()]
                decisions = [dict(suggestion_id=s["suggestion_id"],
                                  decision="adopt" if s["source"] == "reflector" else "defer",
                                  reason="Test global review provenance",
                                  skill_ids=[raw["id"]] if s["source"] == "reflector" else [], task_ids=[])
                             for s in context["suggestions"]]
                return dict(assessment="Globally reviewed, still provisional", decisions=decisions, skills=[raw], tasks=[])
            assert self.schema == LOCAL_SCHEMA
            local_count += 1
            context = json.loads(prompt.split("\n")[-1])
            active = context["enhanced_task"]["task"]
            assert "knowledge" not in context and "plan" not in context
            assert "knowledge_catalog" not in context["state"]
            assert "global_plan" not in json.dumps(context["recent_outcomes"])
            if plan_count > 1:
                assert context["enhanced_task"]["knowledge_sources"][0]["id"] == "critical-tail-diagnosis"
                assert len(context["enhanced_task"]["optional_attachments"]) == 1
            assert "Inspect pass1/pass2 logs; distinguish observed effects" not in prompt
            path = active["source_scope"][0]
            patch = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-value = 0\n+value = {local_count}\n"
            candidate = dict(schema_version=1, config_overrides={"partition": {"algorithm": "tp-gnn"}},
                             candidate_files={}, source_patch=patch, notes="mock candidate")
            return dict(hypothesis="fixture", program=driver.render_candidate_program(candidate), observations=[],
                        suggestions=[dict(target="task", change="Consider a final-only diagnostic task",
                                          rationale="Proxy can miss timing effects", evidence_paths=[],
                                          uncertainty="Physical outcome not yet available")])
    metrics = dict(runs_successfully=1, opt_passes=2, sta_graph_complete=1,
        sta_unconstrained_endpoints=0, final_global_wns_setup_ns=-1., final_global_tns_setup_ns=-10.,
        combined_score=1., timing_weighted_cut=100., placement_native_hpwl=100.,
        placement_overflow_bottom=.1, placement_overflow_top=.1, placement_overflow_max=.1)
    monkeypatch.setattr(runner, "StatelessResearcher", Agent)
    monkeypatch.setattr(driver, "_remote_result", lambda _: {"metrics": metrics})
    def evaluate(**kw):
        evaluations.append(kw)
        cand = json.loads((kw["local_candidate"] / "candidate.json").read_text())
        definition = (
            driver.STA_PROFILE
            if kw["task"] == "tp_gnn_placement_sta"
            else (
                driver.TP_GNN_PROXY_PROFILE
                if kw["task"] == "tp_gnn_sta_inner"
                else driver.PLACEMENT_PROXY_PROFILE
            )
        )
        driver.verify_candidate(driver.load_task(definition), cand)
        n = sum(x["task"] == "tp_gnn_placement_sta" for x in evaluations)
        if kw["task"] == "tp_gnn_placement_sta" and n == failed_round:
            return {"metrics": {"runs_successfully": 0, "combined_score": 0.}}, "/isolated/result/run"
        cut = 80. if "/round-04/inner/02" in kw["remote_root"] else 90.
        return {"metrics": {**metrics, "combined_score": 1. + n*.01,
                            "timing_weighted_cut": cut, "placement_native_hpwl": 90.}}, "/isolated/result/run"
    monkeypatch.setattr(driver, "evaluate_remote", evaluate)
    def evidence(_driver, remote, dest):
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "00.txt"
        path.write_text("fixture evidence")
        report = {"remote_run": remote, "files": [{"local_file": str(path), "sha256": digest(path.read_text())}], "error": None}
        write_json(dest / "index.json", report)
        return report
    monkeypatch.setattr(runner, "collect_evidence", evidence)
    if failed_round:
        with pytest.raises(RuntimeError, match="review interruption"):
            runner.run(args, root, driver)
        assert len(evaluations) == 9
    runner.run(args, root, driver)
    assert len(evaluations) == 9  # 5 final + 4 inner; round4 stops at its second strict improvement
    assert len([e for e in evaluations if e["task"] == "tp_gnn_placement_sta"]) == 5
    summary = json.loads((root / "summary.json").read_text())
    assert summary["final_sta_evaluations"] == 5 and summary["inner_proxy_evaluations"] == 4
    assert summary["context_mode"] == "enhanced-task-v1"
    assert summary["knowledge_authority"] == "global-analyzer-only-v1"
    store = KnowledgeStore(root / "knowledge")
    learned_records = list((store.root / ("test-proposals" if phase == "test" else "evolved")).glob("*/record.json"))
    assert len(learned_records) == 10  # 5 plans + 5 Analyzer reviews, no Local/Reflector publishers
    assert not list(store.root.glob("*/round-*-reflection"))
    assert not list(store.root.glob("*/round-*-attempt-*"))
    for path in learned_records:
        record = json.loads(path.read_text())
        assert record["record"]["provenance"]["author"] == "global-analyzer"
    if phase == "test":
        assert len(store.records()) == 1  # imported skill only; no new formal knowledge activated
    elif phase == "train":
        exported = args.knowledge_export.read_text()
        assert "mock candidate" not in exported and "source_patch" not in exported
        assert "fixture review interruption" not in exported
    for number in range(1, 6):
        packet = json.loads((root / "rounds" / f"{number:02d}" / "enhanced_task.json").read_text())
        checksum = packet.pop("sha256")
        assert checksum == digest(packet)
    assert summary["valid_final_sta_evaluations"] == (4 if failed_round else 5)
    prompts = [p for name, p in calls if name == "global-analyzer-v2"]
    assert all("0.60" in p and "remaining_final_including_this" in p and "evidence_index" in p for p in prompts)
    assert "critical-tail-diagnosis" in prompts[1]  # learned artifact actually reused next round
    assert "local-01-01" in prompts[1] and '"decision": "defer"' in prompts[1]
    assert len([p for name, p in calls if name == "knowledge-reflector-v2"]) == 5
    previous_calls = len(calls)
    runner.run(args, root, driver)
    assert len(evaluations) == 9 and len(calls) == previous_calls
    assert source.joinpath(PARTITION_PATH).read_text() == "value = 0\n"
    packet_path = root / "rounds/01/enhanced_task.json"
    packet = json.loads(packet_path.read_text())
    packet["task"]["objective"][0]["weight"] = 0.5
    write_json(packet_path, packet)
    with pytest.raises(ValueError, match="checksum"):
        runner.run(args, root, driver)
    assert len(evaluations) == 9


def test_legacy_is_default_and_knowledge_is_explicit():
    args = driver.build_parser().parse_args(["--mode", "agentic3d", "--output-dir", "/tmp/unused", "--gpu", "0"])
    assert args.protocol == "legacy" and args.knowledge_phase == "independent"


def test_ephemeral_calls_do_not_reuse_hidden_model_sessions(tmp_path):
    configs = []
    def run(prompt, config):
        configs.append(config)
        config.run_dir.mkdir(parents=True)
        return SimpleNamespace(success=True, final_message='{"blocked": null, "response": {"ok": true}}')
    fake = SimpleNamespace(run_codex=run, CodexExecConfig=driver.CodexExecConfig)
    researcher = runner.StatelessResearcher("local", tmp_path / "local", {}, "fake-model",
                                          "fake-bin", tmp_path, fake)
    researcher.ask("first")
    researcher.ask("second")
    assert all(c.ephemeral and c.resume_thread_id is None and c.sandbox == "read-only" for c in configs)
    assert configs[0].run_dir != configs[1].run_dir


def test_final_gate_rejects_missing_pass2_even_with_positive_score():
    metrics = dict(runs_successfully=1, opt_passes=1, sta_graph_complete=1,
                   sta_unconstrained_endpoints=0, combined_score=2.,
                   final_global_wns_setup_ns=-1., final_global_tns_setup_ns=-10.)
    assert runner.final_score({"metrics": metrics}) is None
    metrics["opt_passes"] = 2
    assert runner.final_score({"metrics": metrics}) == 2.
    metrics["combined_score"] = float("nan")
    assert runner.final_score({"metrics": metrics}) is None


def test_model_usage_counts_actual_traces_and_marks_missing(tmp_path):
    assert runner.model_usage(tmp_path)["tokens"] is None
    path = tmp_path / "local-v2/calls/001"
    path.mkdir(parents=True)
    write_json(path / "result.json", {"duration_seconds": 2.5})
    (path / "events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 15}}) + "\n")
    result = runner.model_usage(tmp_path)
    assert result["completed_model_calls"] == 1 and result["model_wall_seconds"] == 2.5
    assert result["tokens"]["output_tokens"] == 15
