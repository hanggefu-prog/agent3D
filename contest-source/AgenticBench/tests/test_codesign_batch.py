import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic3d import codesign_endpoint as endpoint
from agentic3d import sta_knowledge_runner as knowledge
from agentic3d.sta_protocol import ANALYZER_SCHEMA, LOCAL_SCHEMA, REFLECTION_SCHEMA, REVIEW_SCHEMA, PARTITION_PATH
from agentic3d.knowledge import write_json
from scripts import run_sta_codesign_comparison as legacy
from scripts import run_codesign_experiment as experiment
from scripts import launch_codesign_batch as batch
from scripts.codesign_runtime import Runtime
from scripts.launch_codesign_batch import matrix


def test_restart_method_subset_is_explicit_and_validated():
    from scripts.launch_codesign_batch import selected_methods
    assert selected_methods({}) == ["codex", "agentic3d"]
    assert selected_methods({"methods": ["agentic3d"]}) == ["agentic3d"]
    for values in ([], ["codex", "codex"], ["unknown"]):
        with pytest.raises(ValueError):
            selected_methods({"methods": values})


def result(objective, score=1.):
    metrics = {"runs_successfully": 1, "combined_score": score}
    if objective == "timing":
        metrics.update(opt_passes=2, sta_graph_complete=1, sta_unconstrained_endpoints=0,
                       final_global_wns_setup_ns=-1., final_global_tns_setup_ns=-10.)
    else:
        metrics.update(global_average_temperature_k=325./score,
                       opt_passes=2, cts_completed=1,
                       thermal_evaluation_boundary="two-opt-cts-cell-power-v1",
                       power_model="die-local-openroad-placement-estimated",
                       normalized_average_temperature_cost=1./score,
                       thermal_grid_size=10, thermal_ambient_k=318.15)
    return {"metrics": metrics}


def test_matrix_has_four_cases_with_configurable_final_budgets():
    groups = matrix()
    assert len(groups) == 16 and sum(x["final_budget"] for x in groups) == 160
    assert len({(g["case"], g["objective"], g["method"]) for g in groups}) == 16
    assert len(matrix(["bp_fe"])) == 4
    assert {g["case"] for g in matrix()} == {"bp_fe", "bp", "ariane133", "swerv_wrapper"}
    assert all(g["final_budget"] == 3 for g in matrix(["bp"], rounds=3))
    assert batch.selected_rounds({}) == 10
    assert batch.selected_rounds({"rounds": 51}) == 51
    with pytest.raises(ValueError):
        matrix(["bp_fe", "bp_fe"])


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "3", None])
def test_budget_requires_a_positive_integer_without_coercion(value, monkeypatch):
    with pytest.raises(ValueError, match="positive integer"):
        matrix(["bp"], rounds=value)
    for objective in endpoint.PROFILES:
        with pytest.raises(ValueError, match="positive integer"):
            endpoint.common(objective, value)
    # Even a directly invoked worker fails before Runtime/model/EDA preflight.
    monkeypatch.setattr(batch, "Runtime", lambda *a, **kw: pytest.fail("must not start runtime"))
    with pytest.raises(ValueError, match="positive integer"):
        batch.supervise({"rounds": value})


def cli_args(case="bp"):
    return ["--mode", "codex", "--objective", "timing", "--case", case,
            "--output", "/unused/output", "--batch-root", "/unused", "--source", "/unused/source",
            "--bundle", "/unused/input", "--reference-run", "/unused/reference/run", "--gpus", "0",
            "--image", "sha256:" + "1" * 64, "--codex-bin", "/unused/codex"]


@pytest.mark.parametrize("case", ["bp_fe", "bp", "ariane133", "swerv_wrapper"])
def test_cli_and_runtime_share_four_cases(case):
    args = experiment.parser().parse_args(cli_args(case))
    assert args.case == case and args.rounds == 10 and args.model == "gpt-5.6-sol"
    assert experiment.parser().parse_args(cli_args(case) + ["--rounds", "3"]).rounds == 3
    runtime = Runtime.__new__(Runtime)
    runtime.framework = legacy.ROOT
    assert runtime.profile(case).name == case + "_lol_ng45_tp_gnn_sta"
    assert (runtime.profile(case) / "config.yaml").is_file()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_budget_before_runtime(value, monkeypatch):
    monkeypatch.setattr(experiment.sys, "argv", ["run_codesign_experiment.py", *cli_args(), "--rounds", value])
    monkeypatch.setattr(experiment, "Runtime", lambda *a, **kw: pytest.fail("must not start runtime"))
    with pytest.raises(ValueError, match="positive integer"):
        experiment.main()


@pytest.mark.parametrize("rounds", [None, 3, 51])
def test_batch_plan_reports_resolved_budget_without_launch(tmp_path, monkeypatch, capsys, rounds):
    config = {"cases": ["bp_fe", "bp", "ariane133", "swerv_wrapper"],
              "model": "gpt-5.6-sol", "session": "codesign-test"}
    if rounds is not None:
        config["rounds"] = rounds
    path = tmp_path / "server.json"
    write_json(path, config)
    monkeypatch.setattr(batch.sys, "argv", ["launch_codesign_batch.py", "--config", str(path)])
    monkeypatch.setattr(batch, "Runtime", lambda *a, **kw: pytest.fail("plan must not start runtime"))
    batch.main()
    plan = json.loads(capsys.readouterr().out)
    resolved = 10 if rounds is None else rounds
    assert plan["candidate_final_budget"] == 16 * resolved
    assert plan["reference_budget"] == 8
    assert all(g["final_budget"] == resolved for g in plan["groups"])


def test_resume_cannot_change_final_budget(tmp_path):
    path = tmp_path / "run_config.json"
    config = {"case": "bp", "rounds": 3, "model": "gpt-5.6-sol"}
    experiment.immutable_json(path, config)
    experiment.immutable_json(path, dict(config))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="immutable run contract differs"):
        experiment.immutable_json(path, {**config, "rounds": 4})
    assert path.read_bytes() == original


def test_endpoint_gates_and_budgets_are_not_interchanged():
    assert endpoint.final_score(result("thermal"), "thermal") == 1.
    assert endpoint.final_score(result("thermal"), "timing") is None
    assert endpoint.final_score(result("timing"), "thermal") is None
    assert "Exactly 10 final evaluations" in endpoint.common("thermal", 10)
    assert "Exactly 10 final evaluations" in endpoint.common("timing", 10)
    assert "Cost = 0.60" not in endpoint.common("thermal", 10)
    assert "FIVE" not in endpoint.common("thermal", 10)
    for objective in ("thermal", "timing"):
        assert "Exactly 51 final evaluations" in endpoint.common(objective, 51)
        assert "at most THREE cheap inner evaluations" in endpoint.common(objective, 51)
        assert "remaining_inner_physical" in endpoint.common(objective, 10)
        assert "selection score 0" in endpoint.common(objective, 10)
    bad = result("thermal")
    bad["metrics"]["combined_score"] = float("nan")
    assert endpoint.final_score(bad, "thermal") is None
    with pytest.raises(ValueError, match="another case"):
        endpoint.validate_reference("bp_fe", {"benchmark_id": "bp_lol_ng45_tp_gnn_sta"}, result("thermal"), "thermal")


@pytest.mark.parametrize("objective", ["timing", "thermal"])
@pytest.mark.parametrize("method", ["codex", "agentic3d"])
@pytest.mark.parametrize("rounds", [3, 10])
def test_configurable_round_mock_including_failure_and_resume(tmp_path, monkeypatch, objective, method, rounds):
    """Real schemas/materialization/patch checks, fake models and physical results."""
    profile = legacy.ROOT / "evaluation_profiles" / endpoint.PROFILES[objective]
    spec = legacy.load_task(profile)
    source = tmp_path / "source"
    for name in spec.allowed_source_paths:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("value = 0\n")
    root = tmp_path / "experiment"
    root.mkdir()
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Mock frozen contract")
    args = SimpleNamespace(agent_workspace=source, knowledge_split=None, knowledge_seed=None,
        knowledge_phase="independent", knowledge_import=None, test_knowledge_adaptation=False,
        case="bp_fe", instructions=instructions, reference_run="/isolated/reference/run",
        rounds=rounds, model="fake-model", codex_bin="fake-codex", inner_tolerance=1e-6,
        gpu=0, knowledge_export=None, objective=objective, output=root, bundle=tmp_path / "bundle")
    driver = SimpleNamespace(**vars(legacy))
    driver.STA_PROFILE = profile
    driver.FINAL_TARGET = endpoint.TARGETS[objective]
    driver.COMMON_CONTRACT = endpoint.common(objective, rounds)
    driver.role_prompt = lambda role: endpoint.role_prompt(role, objective)
    driver.final_score = lambda r: endpoint.final_score(r, objective)
    driver.validate_case_reference = lambda c, b, r: endpoint.validate_reference(c, b, r, objective)
    driver._remote_result = lambda _: result(objective)
    calls, evaluations, proposals = [], [], []
    initial = json.loads((profile / "initial_candidate/candidate.json").read_text())
    initial["source_patch"] = f"--- a/{PARTITION_PATH}\n+++ b/{PARTITION_PATH}\n@@ -1 +1 @@\n-value = 0\n+value = 1\n"
    class FakeAgent:
        def __init__(self, name, root, schema, *unused):
            self.schema, self.workspace = schema, source
        def ask(self, prompt):
            calls.append(prompt)
            assert f"Exactly {rounds} final evaluations" in prompt
            if self.schema == ANALYZER_SCHEMA:
                return dict(hypothesis="mechanism", evidence="mock only", evidence_paths=[],
                    local_task=dict(id="joint-final-test", description="test final task", stage="joint",
                        source_scope=list(spec.allowed_source_paths), evaluation="final", objective=[],
                        procedure="Develop and check a mechanism", validation="Trusted unchanged final judge"),
                    specialist_instruction="", max_inner_attempts=1, knowledge_refs=[], skills=[], tasks=[])
            if self.schema == REFLECTION_SCHEMA:
                return dict(assessment="mock only", observations=[], suggestions=[])
            if self.schema == REVIEW_SCHEMA:
                return dict(assessment="mock only", decisions=[], skills=[], tasks=[])
            proposals.append(1)
            candidate = {**initial, "source_patch": initial["source_patch"].replace(
                "+value = 1", f"+value = {len(proposals)}")}
            answer = dict(hypothesis="mock", program=legacy.render_candidate_program(candidate))
            if self.schema == LOCAL_SCHEMA:
                answer.update(observations=[], suggestions=[])
            return answer
    def evaluate(**kw):
        evaluations.append(kw)
        assert kw["task"] == endpoint.TARGETS[objective]
        legacy.verify_candidate(spec, json.loads((kw["local_candidate"] / "candidate.json").read_text()))
        value = {"metrics": {"runs_successfully": 0, "combined_score": 0}} if len(evaluations) == 2 else result(objective, 1.01)
        return value, "/isolated/run"
    driver.evaluate_remote = evaluate
    driver.PersistentCodex = FakeAgent
    monkeypatch.setattr(knowledge, "StatelessResearcher", FakeAgent)
    def evidence(_driver, remote, dest):
        from agentic3d.knowledge import digest, write_json
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "00.txt"
        path.write_text("fixture evidence")
        report = {"remote_run": remote, "files": [{"local_file": str(path), "sha256": digest(path.read_text())}], "error": None}
        write_json(dest / "index.json", report)
        return report
    monkeypatch.setattr(knowledge, "collect_evidence", evidence)
    run = lambda: knowledge.run(args, root, driver) if method == "agentic3d" else experiment.run_codex(args, driver, result(objective))
    run()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["final_evaluations"] == rounds
    assert summary["valid_final_evaluations"] == rounds - 1
    assert summary["failed_final_evaluations"] == 1
    count = len(calls)
    run()
    assert len(evaluations) == rounds and len(calls) == count
    if method == "agentic3d":
        assert summary["knowledge_authority"] == "global-analyzer-only-v1"
        assert summary["context_mode"] == "enhanced-task-v1"


def fake_runtime(tmp_path):
    runtime = Runtime.__new__(Runtime)
    runtime.batch = tmp_path
    runtime.source = tmp_path / "source"
    runtime.framework = runtime.source / "AgenticBench"
    runtime.source_commit = "frozen"
    runtime.image = "sha256:" + "1" * 64
    runtime.gpus, runtime.cpus, runtime.memory_gb = [6, 7], 32, 96
    runtime.share_gpu = False
    runtime.min_gpu_free_mb = 16000
    config = runtime.profile("bp_fe") / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("design: bp_fe\n")
    return runtime


@pytest.mark.parametrize("rounds", [None, 3])
@pytest.mark.parametrize("share_gpu", [False, True])
def test_batch_passes_same_final_budget_to_both_methods(tmp_path, monkeypatch, rounds, share_gpu):
    runtime = fake_runtime(tmp_path)
    runtime.share_gpu = share_gpu
    runtime.min_gpu_free_mb = 2048
    config = {"batch_root": str(tmp_path), "gpus": "6,7",
              "codex_bin": "/unused/codex", "model": "gpt-5.6-sol"}
    if rounds is not None:
        config["rounds"] = rounds
    candidate = runtime.framework / "evaluation_profiles" / endpoint.PROFILES["timing"] / "initial_candidate/candidate.json"
    write_json(candidate, {})
    runtime.evaluate = lambda *a, **kw: None
    monkeypatch.setattr(batch, "qualify_reference", lambda *a: result("timing"))
    commands = []
    def start(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=len(commands), wait=lambda: 0)
    monkeypatch.setattr(batch.subprocess, "Popen", start)
    batch.run_group(config, runtime, "bp", "timing")
    assert len(commands) == 2
    assert {command[command.index("--mode") + 1] for command in commands} == {"codex", "agentic3d"}
    for command in commands:
        assert command[command.index("--rounds") + 1] == str(10 if rounds is None else rounds)
        assert command[command.index("--case") + 1] == "bp"
        assert command[command.index("--model") + 1] == "gpt-5.6-sol"
        assert ("--share-gpu" in command) == share_gpu
        assert command[command.index("--min-gpu-free-mb") + 1] == "2048"


@pytest.mark.parametrize("share_gpu", [False, True])
@pytest.mark.parametrize("min_gpu_free_mb", [16000, 2048])
def test_slot_waits_for_foreign_gpu_process_unless_shared(tmp_path, monkeypatch, share_gpu, min_gpu_free_mb):
    from scripts import codesign_runtime as module
    runtime = fake_runtime(tmp_path)
    runtime.gpus = [6]
    runtime.share_gpu = share_gpu
    runtime.min_gpu_free_mb = min_gpu_free_mb
    (tmp_path / "locks").mkdir()
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw:
                        "MemAvailable: 300000000 kB\n" if str(path) == "/proc/meminfo"
                        else original_read(path, *a, **kw))
    processes = iter(["12345\n", ""])
    monkeypatch.setattr(module.subprocess, "check_output", lambda cmd, **kw:
                        next(processes) if "--query-compute-apps=pid" in cmd else
                        ("7100\n" if min_gpu_free_mb == 2048 else "48000\n"))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=100 * 1024**3))
    waits = []
    monkeypatch.setattr(module.time, "sleep", lambda seconds: waits.append(seconds))
    with runtime.slot() as gpu:
        assert gpu == 6
    assert waits == ([] if share_gpu else [30])


@pytest.mark.parametrize("shortage", ["gpu", "ram", "root-disk", "output-disk"])
def test_shared_gpu_keeps_all_resource_admission_checks(tmp_path, monkeypatch, shortage):
    from scripts import codesign_runtime as module
    runtime = fake_runtime(tmp_path)
    runtime.gpus, runtime.share_gpu = [6], True
    (tmp_path / "locks").mkdir()
    ready = [False]
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw:
                        "MemAvailable: {} kB\n".format(1 if shortage == "ram" and not ready[0] else 300000000)
                        if str(path) == "/proc/meminfo" else original_read(path, *a, **kw))
    monkeypatch.setattr(module.subprocess, "check_output", lambda cmd, **kw:
                        "12345\n" if "--query-compute-apps=pid" in cmd else
                        ("1\n" if shortage == "gpu" and not ready[0] else "48000\n"))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=
                        1 if not ready[0] and ((shortage == "root-disk" and str(path) == "/") or
                        (shortage == "output-disk" and Path(path) == tmp_path)) else 100 * 1024**3))
    waits = []
    def wait(seconds):
        waits.append(seconds)
        ready[0] = True
    monkeypatch.setattr(module.time, "sleep", wait)
    with runtime.slot() as gpu:
        assert gpu == 6
    assert waits == [30]


def test_gpu_sharing_policy_is_explicit_and_immutable(tmp_path):
    assert batch.selected_share_gpu({}) is False
    assert batch.selected_share_gpu({"share_gpu": True}) is True
    assert experiment.parser().parse_args(cli_args()).share_gpu is False
    assert experiment.parser().parse_args(cli_args() + ["--share-gpu"]).share_gpu is True
    assert batch.selected_min_gpu_free_mb({}) == 16000
    assert batch.selected_min_gpu_free_mb({"min_gpu_free_mb": 2048}) == 2048
    assert experiment.parser().parse_args(cli_args() + ["--min-gpu-free-mb", "2048"]).min_gpu_free_mb == 2048
    for value in (1, "true", None, []):
        with pytest.raises(ValueError, match="JSON boolean"):
            batch.selected_share_gpu({"share_gpu": value})
    path = tmp_path / "run_config.json"
    experiment.immutable_json(path, {"share_gpu": False})
    with pytest.raises(ValueError, match="immutable run contract differs"):
        experiment.immutable_json(path, {"share_gpu": True})


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2048", None])
def test_gpu_free_memory_threshold_requires_a_positive_integer(value):
    from scripts.codesign_runtime import validate_min_gpu_free_mb
    with pytest.raises(ValueError, match="positive integer"):
        validate_min_gpu_free_mb(value)
    with pytest.raises(ValueError, match="positive integer"):
        batch.selected_min_gpu_free_mb({"min_gpu_free_mb": value})


def test_gpu_sharing_policy_cannot_change_cached_evaluation_identity(tmp_path):
    runtime = fake_runtime(tmp_path)
    candidate = tmp_path / "candidate"
    write_json(candidate / "candidate.json", {"test": True})
    runtime.execute = lambda command, output, log, contract: write_json(
        output / "evaluation_result.json", result("thermal"))
    args = ("tp_gnn_placement_thermal", candidate, tmp_path / "eval", "bundle",
            tmp_path / "input", None, "bp_fe", tmp_path / "log")
    runtime.evaluate(*args)
    saved = json.loads((tmp_path / "eval/evaluation_contract.json").read_text())
    assert saved["share_gpu"] is False
    assert saved["min_gpu_free_mb"] == 16000
    runtime.min_gpu_free_mb = 2048
    with pytest.raises(ValueError, match="contract changed"):
        runtime.evaluate(*args)
    runtime.min_gpu_free_mb = 16000
    runtime.share_gpu = True
    with pytest.raises(ValueError, match="contract changed"):
        runtime.evaluate(*args)


def test_input_preparation_uses_private_writable_source(tmp_path):
    runtime = fake_runtime(tmp_path)
    commands = []
    def execute(command, output, log, contract):
        commands.append(command)
        write_json(output / "benchmark_input.json", {"mock": True})
        return 0
    runtime.execute = execute
    for case in ("bp_fe", "bp", "ariane133", "swerv_wrapper"):
        bundle = tmp_path / case / "input"
        runtime.prepare(case, bundle)
        command = commands[-1]
        assert command[:2] == ["/usr/bin/python3", "-c"]
        assert "isolated_clone" in command[2]
        assert command[3] == str(runtime.source)
        private = str(tmp_path / case / "input-source")
        assert command[4] == private
        assert command[command.index("--open3dbench-root") + 1] == private
        assert private != str(runtime.source)
    assert commands[0][4] != commands[1][4]


def test_docker_resource_isolation_and_timing_only_passes(tmp_path):
    runtime = fake_runtime(tmp_path)
    commands = []
    def execute(command, output, log, contract):
        commands.append(command)
        write_json(output / "evaluation_result.json", result("timing"))
    runtime.execute = execute
    candidate = tmp_path / "candidate"
    write_json(candidate / "candidate.json", {"test": True})
    for target in ("tp_gnn_placement_sta", "tp_gnn_placement_thermal", "tp_gnn_placement_cts_thermal"):
        runtime.evaluate(target, candidate, tmp_path / target, "bundle", tmp_path / "input",
                         None, "bp_fe", tmp_path / "log")
    assert commands[0][:2] == ["env", "OPEN3D_OPT_PASSES=2"]
    assert "OPEN3D_OPT_PASSES=2" not in commands[1]
    assert commands[2][:2] == ["env", "OPEN3D_OPT_PASSES=2"]
    docker = runtime.docker(["true"], 7)
    assert "device=7" in docker and "none" in docker and "96g" in docker
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(docker)
    assert not any("auth.json" in item or "OPENAI_API_KEY" in item for item in docker)


def test_cache_rejects_changed_candidate_and_incomplete_attempt(tmp_path):
    runtime = fake_runtime(tmp_path)
    candidate = tmp_path / "candidate"
    write_json(candidate / "candidate.json", {"test": True})
    count = []
    def execute(command, output, log, contract):
        count.append(1)
        write_json(output / "evaluation_result.json", result("thermal"))
    runtime.execute = execute
    args = ("tp_gnn_placement_thermal", candidate, tmp_path / "eval", "bundle", tmp_path / "input", None, "bp_fe", tmp_path / "log")
    runtime.evaluate(*args)
    runtime.evaluate(*args)
    assert len(count) == 1
    write_json(candidate / "candidate.json", {"test": False})
    with pytest.raises(ValueError, match="contract changed"):
        runtime.evaluate(*args)
    runtime = fake_runtime(tmp_path / "second")
    output = runtime.batch / "incomplete"
    write_json(output.parent / "incomplete.attempt.json", {"state": "running"})
    with pytest.raises(RuntimeError, match="silently retry"):
        runtime.execute(["true"], output, runtime.batch / "log", {})


def test_thermal_seed_has_correct_final_contract():
    seed = knowledge.seed_snapshot(legacy.ROOT, "thermal", 10)
    assert "tp_gnn_placement_cts_thermal" in json.loads(seed["tools.json"])["final"]
    assert "evaluation_profiles/tp_gnn_placement_two_pass/prompt.md" not in seed
    assert "Exactly 10 final evaluations" in seed["protocol/common.md"]


@pytest.mark.parametrize("objective", ["timing", "thermal"])
@pytest.mark.parametrize("method", ["tp-gnn", "tp-gnn-flat"])
def test_reference_qualification_preserves_verified_process_envelope(tmp_path, monkeypatch, objective, method):
    from local_bench import benchmark_input, metrics
    bundle, run = tmp_path / "input", tmp_path / "reference/run"
    write_json(bundle / "benchmark_input.json", {"benchmark_id": "bp_fe_lol_ng45_tp_gnn_sta"})
    for stage, recorded in (("20_partition", method), ("30_placement", "concurrent-dreamplace")):
        write_json(run / stage / "stage_manifest.json", {"method": recorded})
    report = {"method": method, "config": {"decoder": "global-2way"}}
    if method == "tp-gnn-flat":
        report["hierarchy"] = {"mode": "flat", "consumed_as_feature": False}
    report_path = run / "20_partition/tp_gnn_report.json"
    write_json(report_path, report)
    for stage in ("00_input", "10_floorplan"):
        for base in (bundle / "run", run):
            write_json(base / stage / "stage_manifest.json", {"frozen": stage})
    saved = result(objective)
    saved["artifacts"] = {"exit_code": 0}
    path = run.parent / "evaluation_result.json"
    write_json(path, saved)
    raw = {k: v for k, v in saved["metrics"].items() if k != "runs_successfully"}
    monkeypatch.setattr(benchmark_input, "validate_benchmark_input", lambda *a, **kw: {"run_dir": str(bundle / "run")})
    def extract(*args, **kwargs):
        if objective == "thermal":
            assert kwargs["frozen_input_run"] == bundle / "run"
        return dict(raw)
    monkeypatch.setitem(metrics.EXTRACTORS, endpoint.TARGETS[objective], extract)
    assert experiment.qualify_reference("bp_fe", objective, bundle, run) == saved
    # Method names must match their native report; aliases cannot manufacture
    # flat provenance. The decoder gate remains independent of that identity.
    other_method = "tp-gnn-flat" if method == "tp-gnn" else "tp-gnn"
    write_json(run / "20_partition/stage_manifest.json", {"method": other_method})
    with pytest.raises(ValueError, match="identities disagree"):
        experiment.qualify_reference("bp_fe", objective, bundle, run)
    write_json(run / "20_partition/stage_manifest.json", {"method": method})
    if method == "tp-gnn-flat":
        for hierarchy in (None, {"mode": "flat", "consumed_as_feature": True}):
            write_json(report_path, {**report, "hierarchy": hierarchy})
            with pytest.raises(ValueError, match="flat-hierarchy evidence"):
                experiment.qualify_reference("bp_fe", objective, bundle, run)
    else:
        write_json(report_path, {**report, "hierarchy": {"mode": "flat", "consumed_as_feature": False}})
        with pytest.raises(ValueError, match="cannot be relabeled"):
            experiment.qualify_reference("bp_fe", objective, bundle, run)
    write_json(report_path, {**report, "config": {"decoder": "other"}})
    with pytest.raises(ValueError, match="global-2way decoder"):
        experiment.qualify_reference("bp_fe", objective, bundle, run)
    write_json(report_path, report)
    assert experiment.qualify_reference("bp_fe", objective, bundle, run) == saved
    raw["combined_score"] = 0.9
    with pytest.raises(ValueError, match="differs from raw"):
        experiment.qualify_reference("bp_fe", objective, bundle, run)
    raw["combined_score"] = 1.
    saved["artifacts"]["exit_code"] = 1
    write_json(path, saved)
    with pytest.raises(ValueError, match="process evidence"):
        experiment.qualify_reference("bp_fe", objective, bundle, run)
    saved["artifacts"]["exit_code"] = 0
    saved["metrics"]["runs_successfully"] = 0
    write_json(path, saved)
    with pytest.raises(ValueError, match="unqualified"):
        experiment.qualify_reference("bp_fe", objective, bundle, run)


def test_docker_tmp_uses_unique_output_volume_directories(tmp_path):
    runtime = fake_runtime(tmp_path)
    commands = [runtime.docker(["true"], 6) for _ in range(2)]
    mounts = [next(x for x in command if x.endswith(":/tmp")) for command in commands]
    assert mounts[0] != mounts[1]
    for command, mount in zip(commands, mounts):
        scratch = Path(mount[:-5])
        scratch.relative_to(runtime.batch)
        assert scratch.is_dir()
        assert "TMPDIR=/tmp" in command and "MPLCONFIGDIR=/tmp/matplotlib" in command
    with pytest.raises(ValueError):
        runtime.docker(["true"], 6, tmp_path.parent / "foreign")
