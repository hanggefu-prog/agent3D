from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from global_bench.cli import DEFAULT_TASK_ROOT, main as global_main
from global_bench.coordinator import GlobalCandidateFeedback, GlobalRoundCoordinator
from global_bench.evaluation import AdaptiveEvaluationRunner
from global_bench.impact import ImpactResolver, compute_stage_signature
from global_bench.intervention import (
    GlobalSearchContext,
    Intervention,
    InterventionResult,
    build_global_analyzer_prompt,
)
from global_bench.model import load_global_task
from global_bench.runtime import DockerTrialRuntime
from global_bench.scheduler import FileResourcePool, GlobalTrialScheduler, ResourceUnavailable
from global_bench.submission import collect_submission
from global_bench.verifier import (
    build_verifier_container_command,
    load_trusted_verifier,
    verify_submission_structure,
)
from global_bench.workspace import TrialWorkspaceManager


ROOT = Path(__file__).resolve().parents[1]


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def source_repository(tmp_path: Path, *, control_plane: bool = False) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    (source / "open3dbench").mkdir()
    (source / "open3dbench/backend").mkdir()
    (source / "open3dbench/backend/three_d_route.py").write_text("BASE = 1\n")
    if control_plane:
        (source / "AgenticBench").mkdir()
        (source / "AgenticBench/verifier.py").write_text("TRUSTED = True\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "base")
    return source


def task():
    return load_global_task(DEFAULT_TASK_ROOT / "lol_full_flow_ppa")


def prepare_trial(tmp_path: Path, *, control_plane: bool = False):
    source = source_repository(tmp_path, control_plane=control_plane)
    layout = TrialWorkspaceManager(tmp_path / "trials").create(
        task(), source, trial_id="trial-001", user_command="reduce routed wire length"
    )
    return source, layout


def write_artifacts(layout) -> dict:
    run = layout.runs / "final"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "pins": {"seed": 1},
                "partition": {"seed": 1001},
                "placement": {"seed": 1000},
                "routing": {"seed": -1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    stages = {
        "input": "00_input",
        "floorplan": "10_floorplan",
        "timing_extract": "15_timing_extract",
        "partition": "20_partition",
        "placement": "30_placement",
        "pdn": "40_pdn",
        "hbt": "50_hbt",
        "opt": "60_opt",
        "cts": "70_cts",
        "global_routing": "75_global_route",
        "detailed_routing": "80_route",
        "report": "90_report",
        "thermal": "95_thermal",
    }
    status = {"status": "success", "stages": {}}
    for stage, directory in stages.items():
        stage_dir = run / directory
        stage_dir.mkdir()
        artifact = stage_dir / "artifact.json"
        artifact.write_text("{}\n")
        manifest = {
            "schema_version": 1,
            "stage": stage,
            "status": "success",
            "signature": {"implementation": stage},
            "artifacts": {
                "fixture": {
                    "path": artifact.name,
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            },
        }
        (stage_dir / "stage_manifest.json").write_text(json.dumps(manifest))
        status["stages"][stage] = {"status": "success"}
    checkpoint = run / "95_thermal/checkpoint.json"
    checkpoint.write_text(
        json.dumps({"stage": "thermal", "thermal": {"metrics": "metrics.json"}}) + "\n"
    )
    metrics = run / "90_report/metrics.json"
    metrics.write_text(json.dumps({"wire_length": 123.0, "drc": 0}) + "\n")
    run_manifest = run / "run.json"
    run_manifest.write_text(json.dumps(status) + "\n")
    return {
        "run_manifest": run_manifest,
        "final_checkpoint": checkpoint,
        "metrics": metrics,
    }


def test_global_task_is_scope_free_and_has_a_valid_dag():
    spec = task()
    encoded = json.dumps(spec.to_dict())
    assert "source_scope" not in encoded
    assert "config_scope" not in encoded
    assert "AgenticBench" in spec.workspace.protected_paths
    assert "open3dbench/runner.py" in spec.workspace.protected_paths
    assert spec.evaluation.stage_ids()[0] == "00_input"
    assert spec.evaluation.stage_ids()[-1] == "95_thermal"
    assert spec.seed_policy.io_placement == 1
    assert spec.seed_policy.partition == 1001
    assert spec.seed_policy.placement == 1000
    assert spec.seed_policy.routing == -1


def test_codesign_tasks_fix_objectives_but_not_algorithm_file_scopes():
    timing = load_global_task(DEFAULT_TASK_ROOT / "lol_partition_placement_timing")
    thermal = load_global_task(DEFAULT_TASK_ROOT / "lol_partition_placement_thermal")
    assert timing.objective.kind == "sta_wns_tns"
    assert timing.objective.parameters == {
        "wns_weight": 0.6,
        "tns_weight": 0.4,
        "deficit_floor": 1.0e-12,
        "opt_passes": 2,
    }
    assert timing.evaluation.stage_ids()[-1] == "60_opt"
    assert thermal.objective.primary_metric == "global_average_temperature_k"
    assert thermal.objective.parameters["metric_path"] == "temperature.global_average_k"
    assert thermal.evaluation.stage_ids()[-1] == "95_thermal"
    assert "source_scope" not in json.dumps(timing.to_dict())
    assert "source_scope" not in json.dumps(thermal.to_dict())
    assert load_trusted_verifier(timing.verifier.verifier_id).verifier_id.endswith("timing-v1")
    assert load_trusted_verifier(thermal.verifier.verifier_id).verifier_id.endswith("thermal-v1")


def test_global_analyzer_models_an_intervention_without_file_scope():
    spec = task()
    context = GlobalSearchContext(
        task_id=spec.task_id,
        user_command="reduce HPWL and congestion",
        global_round=0,
        accepted_revision="abc123",
        objectives={"hpwl": "min", "congestion": "min"},
        hard_constraints=("legal placement",),
        stage_artifacts={"70_cts": "trusted://cts"},
        observations=(),
        remaining_budget={"trials": 4},
    )
    prompt = build_global_analyzer_prompt(spec, context)
    assert "Do not define source_scope" in prompt
    assert "reduce HPWL and congestion" in prompt
    assert '"partition": 1001' in prompt
    assert "not optimization parameters" in prompt
    intervention = Intervention.from_mapping(
        {
            "schema_version": 1,
            "intervention_id": "route-round-0",
            "global_round": 0,
            "decision": "explore",
            "hypothesis": "route ordering causes detours",
            "target_stages": ["80_route"],
            "objectives": {"wire_length": "min"},
            "hard_constraints": ["routing completes"],
            "context_artifacts": ["70_cts/checkpoint.json"],
            "evaluation_policy": "suffix",
            "execution_backend": "workspace_agent",
            "upstream_checkpoint": "trusted://70_cts",
            "budget": {"walltime_seconds": 300},
        },
        spec,
    )
    assert intervention.target_stages == ("80_route",)
    assert "scope" not in intervention.to_dict()


def test_parallel_trial_workspaces_are_independent(tmp_path):
    source = source_repository(tmp_path)
    manager = TrialWorkspaceManager(tmp_path / "trials")
    first = manager.create(task(), source, trial_id="trial-a")
    second = manager.create(task(), source, trial_id="trial-b")
    first_file = first.workspace / "open3dbench/backend/three_d_route.py"
    second_file = second.workspace / "open3dbench/backend/three_d_route.py"
    first_file.write_text("BASE = 2\n")
    assert second_file.read_text() == "BASE = 1\n"
    assert (first.build, first.runs, first.logs, first.scratch) != (
        second.build,
        second.runs,
        second.logs,
        second.scratch,
    )
    assert (first.control / "user-command.md").read_text().strip() == task().instruction


def test_route_change_reuses_verified_cts_prefix():
    plan = ImpactResolver(task().evaluation).resolve(
        ["open3dbench/backend/three_d_route.py"],
        claimed_stages=["80_route"],
        policy="suffix",
    )
    assert plan.start_stage == "75_global_route"
    assert plan.rerun_stages == (
        "75_global_route",
        "80_route",
        "90_report",
        "95_thermal",
    )
    assert "70_cts" in plan.reused_stages
    assert not plan.conservative_fallback


def test_unknown_shared_change_conservatively_invalidates_every_stage():
    plan = ImpactResolver(task().evaluation).resolve(
        ["open3dbench/runner.py"], policy="suffix"
    )
    assert plan.conservative_fallback
    assert plan.rerun_stages == task().evaluation.stage_ids()


def test_stage_signature_covers_source_config_upstream_and_seed(tmp_path):
    source = tmp_path / "algorithm.py"
    source.write_text("VALUE = 1\n")
    first = compute_stage_signature(
        stage_id="80_route",
        upstream_artifacts={"70_cts": "abc"},
        resolved_config={"routing": {"seed": 1}},
        implementation_files=[source],
        environment_identity="sha256:image",
        seed=1,
    )
    source.write_text("VALUE = 2\n")
    second = compute_stage_signature(
        stage_id="80_route",
        upstream_artifacts={"70_cts": "abc"},
        resolved_config={"routing": {"seed": 1}},
        implementation_files=[source],
        environment_identity="sha256:image",
        seed=1,
    )
    assert first != second


def test_runtime_plan_mounts_trial_state_and_inputs_without_a_new_agent_api(tmp_path):
    _, layout = prepare_trial(tmp_path)
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    plan = DockerTrialRuntime().plan(
        task(),
        layout,
        ["codex", "exec", "--full-auto", "/task/instruction.md"],
        skills_root=ROOT / "skills",
        host_environment={"OPENAGENTIC3D_GLOBAL_INPUT": str(benchmark)},
        assigned_gpus=["0"],
    )
    command = list(plan.command)
    assert "--read-only" in command
    assert "no-new-privileges" in command
    assert command[command.index("--network") + 1] == "none"
    assert any("dst=/workspace,rw" in item for item in command)
    assert any("dst=/inputs/benchmark,readonly" in item for item in command)
    assert any("dst=/git/common,readonly" in item for item in command)
    assert any("dst=/git/worktree,rw" in item for item in command)
    assert plan.environment["GIT_COMMON_DIR"] == "/git/common"
    assert plan.agent_command[0] == "codex"


def test_adaptive_runner_materializes_only_the_verified_prefix(tmp_path):
    spec = task()
    accepted = tmp_path / "accepted"
    accepted.mkdir()
    (accepted / "config.yaml").write_text("design: bp_fe\n")
    status = {"status": "completed", "stages": {}}
    for stage in spec.evaluation.stages:
        stage_dir = accepted / stage.directory
        stage_dir.mkdir()
        artifact = stage_dir / "artifact.json"
        artifact.write_text("{}\n")
        manifest = {
            "schema_version": 1,
            "stage": stage.runner_name,
            "status": "success",
            "signature": {"implementation": stage.stage_id},
            "artifacts": {
                "fixture": {
                    "path": artifact.name,
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            },
        }
        (stage_dir / "stage_manifest.json").write_text(json.dumps(manifest))
        status["stages"][stage.runner_name] = {"status": "success"}
    (accepted / "run.json").write_text(json.dumps(status))
    repository = tmp_path / "candidate-source"
    (repository / "open3dbench").mkdir(parents=True)
    plan = ImpactResolver(spec.evaluation).resolve(
        ["open3dbench/backend/three_d_route.py"], policy="suffix"
    )
    candidate = tmp_path / "candidate-run"
    replay = AdaptiveEvaluationRunner().prepare(
        spec,
        plan,
        accepted_run=accepted,
        candidate_run=candidate,
        candidate_repository=repository,
    )
    assert (candidate / "70_cts/stage_manifest.json").is_file()
    assert not (candidate / "80_route").exists()
    assert replay.reused_prefix[-1] == "70_cts"
    assert replay.command[replay.command.index("--from") + 1] == "global_routing"
    assert replay.command[replay.command.index("--to") + 1] == "thermal"
    resolved = yaml.safe_load((candidate / "config.yaml").read_text())
    assert resolved["pins"]["seed"] == 1
    assert resolved["partition"]["seed"] == 1001
    assert resolved["placement"]["seed"] == 1000
    assert resolved["routing"]["seed"] == -1


def test_global_round_order_is_fixed_without_fixing_the_agent_implementation():
    spec = task()
    context = GlobalSearchContext(
        task_id=spec.task_id,
        user_command="reduce routing cost",
        global_round=1,
        accepted_revision="abc123",
        objectives={"wire_length": "min"},
        hard_constraints=("legal route",),
        stage_artifacts={"70_cts": "trusted://cts"},
        observations=(),
        remaining_budget={"trials": 2},
    )
    events = []

    def modeler(received):
        events.append("model")
        assert received is context
        return Intervention(
            intervention_id="route-1",
            global_round=1,
            decision="explore",
            hypothesis="reroute order causes detours",
            target_stages=("80_route",),
            objectives={"wire_length": "min"},
            hard_constraints=("legal route",),
            context_artifacts=("70_cts/checkpoint.json",),
            evaluation_policy="suffix",
            execution_backend="workspace_agent",
            upstream_checkpoint="trusted://cts",
            budget={"walltime_seconds": 60},
        )

    def executor(intervention, received):
        events.append("execute")
        assert received is context
        return InterventionResult(
            intervention_id=intervention.intervention_id,
            trial_id="trial-route-1",
            base_revision="abc123",
            candidate_revision=None,
            changed_files=("open3dbench/backend/three_d_route.py",),
            claimed_affected_stages=("80_route",),
            submission="/data/submission",
        )

    def evaluator(result, plan):
        events.append("evaluate")
        assert plan.rerun_stages == (
            "75_global_route",
            "80_route",
            "90_report",
            "95_thermal",
        )
        return GlobalCandidateFeedback(
            trial_id=result.trial_id,
            passed=True,
            metrics={"wire_length": 1.0},
            artifacts={"verification": "/data/verification.json"},
            reason="trusted gates passed",
        )

    outcome = GlobalRoundCoordinator(spec, modeler, executor, evaluator).run_round(context)
    assert events == ["model", "execute", "evaluate"]
    assert outcome.feedback.passed


def test_submission_is_hashed_and_verified_outside_workspace(tmp_path):
    _, layout = prepare_trial(tmp_path)
    candidate = layout.workspace / "open3dbench/backend/three_d_route.py"
    candidate.write_text("BASE = 2\n")
    manifest = collect_submission(task(), layout, write_artifacts(layout))
    assert {item.path for item in manifest.changes} == {
        "open3dbench/backend/three_d_route.py"
    }
    report = verify_submission_structure(task(), layout.submission)
    assert report.passed, report.errors
    verifier_output = layout.root / "verifier-output"
    verifier_script = ROOT / "global_bench/verifiers/open3dbench_global_verifier.py"
    process = subprocess.run(
        [
            sys.executable,
            str(verifier_script),
            "--submission",
            str(layout.submission),
            "--output",
            str(verifier_output),
            "--evaluated-run",
            str(layout.runs / "final"),
        ],
        check=False,
    )
    assert process.returncode == 0
    assert json.loads((verifier_output / "verification.json").read_text())["passed"]
    metrics = next(item for item in manifest.artifacts if item.name == "metrics")
    (layout.submission / metrics.submitted_path).write_text("{}\n")
    tampered = verify_submission_structure(task(), layout.submission)
    assert not tampered.passed
    assert not tampered.checks["artifact_integrity"]


def test_submission_rejects_control_plane_changes(tmp_path):
    _, layout = prepare_trial(tmp_path, control_plane=True)
    (layout.workspace / "AgenticBench/verifier.py").write_text("TRUSTED = False\n")
    with pytest.raises(ValueError, match="control-plane"):
        collect_submission(task(), layout, write_artifacts(layout))


def test_verifier_container_sees_only_frozen_inputs_and_trusted_results(tmp_path):
    _, layout = prepare_trial(tmp_path)
    collect_submission(task(), layout, write_artifacts(layout))
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    output = tmp_path / "verifier-output"
    scratch = tmp_path / "verifier-scratch"
    output.mkdir()
    scratch.mkdir()
    command = build_verifier_container_command(
        task(),
        layout.submission,
        layout.runs / "final",
        {"benchmark_input": benchmark},
        output,
        scratch,
    )
    assert "--read-only" in command
    assert "none" == command[command.index("--network") + 1]
    assert any("dst=/submission,readonly" in item for item in command)
    assert any("dst=/evaluation-run,readonly" in item for item in command)
    assert any("dst=/inputs/benchmark,readonly" in item for item in command)
    assert not any("dst=/workspace" in item for item in command)


def test_named_resource_leases_serialize_parallel_trials(tmp_path):
    pool = FileResourcePool(tmp_path / "leases")
    first = pool.acquire(["gpu0", "gpu1"], "trial-a")
    second = pool.acquire(["gpu0", "gpu1"], "trial-b")
    assert {first.resource_id, second.resource_id} == {"gpu0", "gpu1"}
    with pytest.raises(ResourceUnavailable):
        pool.acquire(["gpu0", "gpu1"], "trial-c")
    pool.release(first)
    third = pool.acquire(["gpu0"], "trial-c")
    assert third.resource_id == "gpu0"


def test_scheduler_enforces_the_task_parallelism_budget(tmp_path):
    source = source_repository(tmp_path)
    spec = replace(task(), budget=replace(task().budget, concurrent_trials=1))
    scheduler = GlobalTrialScheduler(tmp_path / "scheduled")
    first = scheduler.prepare(spec, source, trial_id="scheduled-a")
    with pytest.raises(ResourceUnavailable):
        scheduler.prepare(spec, source, trial_id="scheduled-b")
    scheduler.release(first.layout)
    second = scheduler.prepare(spec, source, trial_id="scheduled-b")
    assert second.slot.resource_id == "slot-000"


def test_global_cli_lists_and_resolves_routing_suffix(capsys):
    assert global_main(["list"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert any(item["id"] == "lol_full_flow_ppa" for item in listing)
    assert global_main(
        [
            "impact",
            "lol_full_flow_ppa",
            "--changed-file",
            "open3dbench/backend/three_d_route.py",
            "--policy",
            "suffix",
        ]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["start_stage"] == "75_global_route"
