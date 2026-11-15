"""Reference-aligned local integration gates; no Docker, EDA, or model calls."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from local_bench import evaluation
from local_bench import reference_baseline as ref
from local_bench.model import EvaluationResult
from scripts import run_local_task_smoke as smoke
from scripts.codesign_runtime import Runtime
# Reuse the actual native-evidence/registry fixture, not a mocked selection.
from test_reference_local_placement import setup as native_reference_environment

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "placement_dynamic_parameters_reference_20260923": "placement_parameters_reference_20260923",
}
IMAGE = sorted(ref.IMAGE_IDS)[0]


def forbidden(*args, **kwargs):
    raise AssertionError("No physical/model process may start in this test")


@pytest.fixture
def smoke_environment(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    batch.mkdir()
    bundle = batch / "bundle"
    (bundle / "run").mkdir(parents=True)
    # Family selection is now an explicit on-disk boundary, while the native
    # qualification below remains mocked for these pre-evaluation tests.
    (bundle / "benchmark_input.json").write_text(json.dumps({"baseline_id": ref.BASELINE_ID}))
    calls = SimpleNamespace(bundle=[], transport=[], model=[])
    def contained(path):
        path = Path(path).resolve()
        path.relative_to(batch)
        return path
    runtime = SimpleNamespace(source=ROOT.parent, framework=ROOT, batch=batch,
        source_commit="fixture", image=IMAGE, cpus=16, memory_gb=48,
        share_gpu=True, min_gpu_free_mb=2048, contained=contained)
    monkeypatch.setattr(smoke, "validate_benchmark_input", lambda *args, **kwargs: {
        "run_dir": str(bundle / "run"), "config": str(ref.profile_path("bp_fe") / "config.yaml")})
    monkeypatch.setattr(smoke, "materialize_program", lambda *args: {"schema_version": 1})
    archive = tmp_path / "archive"
    def validate(path, case, *, fixed_partition=False):
        calls.bundle.append((path, case, fixed_partition))
        return {"baseline_id": ref.BASELINE_ID, "reference_root": str(archive)}
    monkeypatch.setattr(ref, "validate_reference_bundle", validate)
    class FailedReferenceTransport:
        def __init__(self, received):
            assert received is runtime
            calls.transport.append(self)
            self.physical_calls = 0
            self.receipts = []
        def __call__(self, task, candidate, source, output, **kwargs):
            self.physical_calls += 1
            output.mkdir(parents=True)
            result = EvaluationResult({"runs_successfully": 0, "combined_score": 0.0},
                                      {"error_type": "FixtureReferenceRejected"})
            (output / "evaluation_result.json").write_text(json.dumps(result.to_dict()))
            self.receipts.append({"path": str(output / "evaluation_result.json")})
            return result
    return SimpleNamespace(runtime=runtime, bundle=bundle, archive=archive,
                           calls=calls, transport=FailedReferenceTransport)


@pytest.mark.parametrize("task", TASKS)
def test_local_smoke_requires_adopted_bundle_before_model(smoke_environment, task):
    env = smoke_environment
    with pytest.raises(ValueError, match="adopted benchmark-input bundle"):
        smoke.run_smoke(env.runtime, task=task, output=env.runtime.batch / "missing-bundle",
                        provider_factory=forbidden, transport_factory=env.transport)
    assert not env.calls.transport and not env.calls.bundle
    assert not (env.runtime.batch / "missing-bundle").exists()


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("fault", ["old_bundle", "wrong_image"])
def test_local_smoke_rejects_identity_mismatch_before_reference(smoke_environment, monkeypatch, task, fault):
    env = smoke_environment
    if fault == "wrong_image":
        env.runtime.image = "sha256:" + "0" * 64
    else:
        def old_bundle(*args, **kwargs):
            raise ValueError("Input bundle is not the versioned reference-aligned profile")
        monkeypatch.setattr(ref, "validate_reference_bundle", old_bundle)
    with pytest.raises(ValueError, match="pinned|versioned"):
        smoke.run_smoke(env.runtime, task=task, benchmark_input=env.bundle,
                        output=env.runtime.batch / "rejected",
                        provider_factory=forbidden, transport_factory=env.transport)
    assert not env.calls.transport
    assert not (env.runtime.batch / "rejected").exists()


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("case", ["bp_fe", "bp", "ariane133", "swerv_wrapper"])
def test_local_smoke_records_fixed_partition_archive_before_fresh_reference(smoke_environment, monkeypatch, task, case):
    env = smoke_environment
    # This test isolates pre-evaluation failure accounting. The full reference
    # producer/registry path has independent native-artifact fixture coverage.
    from scripts import local_reference
    monkeypatch.setattr(local_reference, "expected_local_contract", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(smoke, "validate_benchmark_input", lambda *args, **kwargs: {
        "run_dir": str(env.bundle / "run"), "config": str(ref.profile_path(case) / "config.yaml")})
    output = env.runtime.batch / "smoke"
    result = smoke.run_smoke(env.runtime, task=task, benchmark_input=env.bundle, output=output, reference_policy="FRESH",
                            provider_factory=forbidden, transport_factory=env.transport)
    assert env.calls.bundle == [(env.bundle, case, True)]
    assert env.runtime.reference_baseline == ref.BASELINE_ID
    assert env.runtime.reference_root == env.archive
    assert len(env.calls.transport) == 1
    assert result["state"] == "reference_failed" and result["reference_evaluations"] == 1
    assert result["model_requests"] == result["candidate_finals"] == 0
    saved = json.loads((output / "run_config.json").read_text())
    assert saved["reference_alignment"]["reference_root"] == str(env.archive)
    assert saved["reference_alignment"]["case"] == case
    assert saved["reference_excluded_from_budget"] is True
    assert saved["normalization_reference"] is None
    assert all(saved["reference_selection"][key] is None
               for key in ("record", "record_sha256", "reference_id"))


@pytest.mark.parametrize("task", TASKS)
def test_local_driver_pins_fresh_and_adopted_identity_before_search(native_reference_environment, task):
    """Real driver/qualification/store with synthetic native and model I/O."""
    from local_bench import reference_local_placement as local
    from local_bench.reference_store import file_sha
    from test_reference_local_placement import native_fixture, write

    env = native_reference_environment
    snapshots = []
    _, spec, candidate, *_ = local._context(local.SOURCE, task, "bp_fe", env.bundle)
    initial = local.initial_program_text(spec, candidate)

    def assert_pinned(output):
        saved = json.loads((output / "run_config.json").read_text())
        selection = saved["reference_selection"]
        assert selection == env.runtime.local_reference_selection.to_dict()
        record = Path(selection["record"])
        assert selection["record_sha256"] == file_sha(record)
        assert selection["reference_id"] == json.loads(record.read_text())["reference_id"]
        assert saved["normalization_reference"] == str(env.runtime.adopted_reference_run)
        assert saved["generated_candidate_budget"] == 5
        assert saved["reference_excluded_from_budget"] is True
        return selection

    class FixtureTransport:
        def __init__(self, runtime):
            self.receipts, self.attempts = [], []

        def __call__(self, task_dir, candidate_dir, source, output, **kwargs):
            if kwargs.get("reference_run") is None:
                fixture = native_fixture(env, task, output)
                result = EvaluationResult(**json.loads((fixture.evaluation / "evaluation_result.json").read_text()))
            else:
                env.runtime.local_reference_selection.verify_selected()
                result = EvaluationResult({"combined_score": 1.0, "runs_successfully": 1}, {})
                write(output / "evaluation_result.json", result.to_dict())
            self.attempts.append({"output_dir": str(output), "evaluator_dispatch_attempted": True,
                                  "native_flow_started": True})
            self.receipts.append({"path": str(output / "evaluation_result.json")})
            return result

    class FixtureProvider:
        def __init__(self, model_root, *args):
            self.output = Path(model_root).parent
            self.calls = self.completed_calls = 0
            # The authority must be durable even if the first model call fails.
            snapshots.append(assert_pinned(self.output))

        def draw_sample(self, messages):
            assert assert_pinned(self.output) == snapshots[-1]
            self.calls += 1
            self.completed_calls += 1
            return initial

    for name, policy in (("fresh", "FRESH"), ("adopted", "ADOPT")):
        output = env.batch / name
        result = smoke.run_smoke(env.runtime, task=task, benchmark_input=env.bundle,
            output=output, reference_policy=policy,
            provider_factory=FixtureProvider, transport_factory=FixtureTransport)
        assert result["state"] == "completed", result["error"]
        assert result["reference_evaluations"] == (1 if policy == "FRESH" else 0)
        assert result["candidate_finals"] == result["model_requests"] == 5
        qualification = json.loads((output / "reference/qualification.json").read_text())
        assert assert_pinned(output) == qualification["reference_selection"] == result["reference_selection"]
    assert snapshots[0]["policy"] == "FRESH" and snapshots[1]["policy"] == "ADOPT"
    for key in ("record", "record_sha256", "reference_id", "contract"):
        assert snapshots[0][key] == snapshots[1][key]


@pytest.fixture
def local_evaluator(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    calls = SimpleNamespace(partition=[], physical=[], subprocess=[])
    monkeypatch.setattr(evaluation, "load_candidate", lambda path: {})
    monkeypatch.setattr(evaluation, "verify_candidate", lambda *args: {})
    monkeypatch.setattr(evaluation, "validate_patch_scope", lambda *args: {"paths": []})
    def resolved(spec, candidate_dir, output, candidate, *, base_config=None):
        value = yaml.safe_load((base_config or spec.base_config).read_text())
        output.write_text(yaml.safe_dump(value))
        return value
    monkeypatch.setattr(evaluation, "_write_resolved_config", resolved)
    def partition(spec, run, config):
        calls.partition.append((spec.task_id, run, config))
        return {"status": "passed", "partition_method": "tp-gnn-flat"}
    monkeypatch.setattr(evaluation, "_validate_placement_partition_reference", partition)
    def physical(run, case, *, fixed_partition=False):
        calls.physical.append((run, case, fixed_partition))
        return {"20_partition/partition_assignment.csv": "verified-fixture-hash"}
    monkeypatch.setattr(ref, "validate_physical_prefix", physical)
    def subprocess_gate(command, **kwargs):
        calls.subprocess.append(command)
        return SimpleNamespace(returncode=1, stdout="Fixture stop after validated native plan dispatch")
    monkeypatch.setattr(evaluation.subprocess, "run", subprocess_gate)
    monkeypatch.setenv("OPEN3DBENCH_IMAGE_ID", IMAGE)
    return SimpleNamespace(candidate=candidate, baseline=baseline, calls=calls, output=tmp_path / "result")


@pytest.mark.parametrize("task,directory", TASKS.items())
@pytest.mark.parametrize("fault", ["config", "image", "partition_hash", "missing_baseline"])
def test_reference_local_evaluator_fails_before_subprocess(local_evaluator, monkeypatch, tmp_path,
                                                          task, directory, fault):
    env = local_evaluator
    base_config = ref.profile_path("bp_fe") / "config.yaml"
    if fault == "config":
        value = yaml.safe_load(base_config.read_text())
        value["threads"] = 32
        base_config = tmp_path / "wrong-config.yaml"
        base_config.write_text(yaml.safe_dump(value))
    elif fault == "image":
        monkeypatch.setenv("OPEN3DBENCH_IMAGE_ID", "sha256:" + "0" * 64)
    elif fault == "partition_hash":
        def reject(*args, **kwargs):
            raise ValueError("Fixed placement partition differs from frozen reference")
        monkeypatch.setattr(ref, "validate_physical_prefix", reject)
    else:
        env.baseline = None
    result = evaluation.evaluate_task(ROOT / "local_tasks" / directory, env.candidate,
        ROOT.parent, env.output, baseline_run=env.baseline, base_config=base_config)
    assert result.metrics["runs_successfully"] == 0
    assert result.artifacts["error_type"] == "ValueError"
    assert not env.calls.subprocess
    assert not (env.output / "run").exists()
    if fault in ("config", "image", "missing_baseline"):
        assert not env.calls.physical


@pytest.mark.parametrize("task,directory", TASKS.items())
@pytest.mark.parametrize("case", ["bp_fe", "bp", "ariane133", "swerv_wrapper"])
def test_reference_local_evaluator_checks_initial_config_image_and_fixed_hash_before_plan(local_evaluator,
                                                                                       task, directory, case):
    env = local_evaluator
    result = evaluation.evaluate_task(ROOT / "local_tasks" / directory, env.candidate,
        ROOT.parent, env.output, baseline_run=env.baseline,
        base_config=ref.profile_path(case) / "config.yaml")
    assert env.calls.partition[0][0] == task
    assert env.calls.physical == [(env.baseline, case, True)]
    assert len(env.calls.subprocess) == 1 and "plan" in env.calls.subprocess[0]
    assert result.artifacts["reference_alignment"]["baseline_id"] == ref.BASELINE_ID
    assert result.artifacts["reference_alignment"]["case"] == case
    assert result.artifacts["reference_alignment"]["physical_input_sha256"] == {
        "20_partition/partition_assignment.csv": "verified-fixture-hash"}
    # Deliberate fixture stop: no successful evaluation or native metrics fabricated.
    assert result.metrics["runs_successfully"] == 0


def test_missing_system_config_fails_before_reference(smoke_environment, monkeypatch, tmp_path):
    """The real config reader runs before output creation or any transport."""
    env = smoke_environment
    spec = smoke.load_task(ROOT / "local_tasks" / TASKS[next(iter(TASKS))])
    env.runtime.framework = tmp_path / "framework"
    missing = env.runtime.framework / "local_tasks" / "missing-config"
    missing.mkdir(parents=True)
    monkeypatch.setattr(smoke, "load_task", lambda _: replace(spec, root=missing))
    monkeypatch.setattr(smoke, "resolve_task_path", lambda *args: missing)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        smoke.run_smoke(env.runtime, task=spec.task_id, benchmark_input=env.bundle,
            output=env.runtime.batch / "missing-system-config",
            provider_factory=forbidden, transport_factory=env.transport)
    assert not env.calls.transport and not env.calls.bundle
    assert not (env.runtime.batch / "missing-system-config").exists()


@pytest.mark.parametrize("bad_case", [None, "", "unregistered-design"])
def test_reference_local_bad_case_fails_before_native_plan(local_evaluator, tmp_path, bad_case):
    env = local_evaluator
    config = yaml.safe_load((ref.profile_path("bp_fe") / "config.yaml").read_text())
    config["design"] = bad_case
    path = tmp_path / "bad-case.yaml"
    path.write_text(yaml.safe_dump(config))
    result = evaluation.evaluate_task(ROOT / "local_tasks" / "placement_parameters_reference_20260923",
        env.candidate, ROOT.parent, env.output, baseline_run=env.baseline, base_config=path)
    assert result.metrics["runs_successfully"] == 0
    assert not env.calls.subprocess and not env.calls.physical


def test_nesterov_evaluator_reaches_real_evidence_and_metric_dispatch(local_evaluator, monkeypatch):
    """Synthetic native artifacts exercise real post-run gates and dispatch.

    Execution/setup are stubbed; neither native evidence parsing nor metrics is
    mocked. This is a wiring test, not an EDA qualification result.
    """
    from test_reference_placement_evidence import HEADER, native_params, row

    env = local_evaluator
    for name in ("apply_candidate_patch", "validate_candidate_workspace"):
        monkeypatch.setattr(evaluation, name, lambda *args: {})
    for name in ("select_evaluator_source_patch", "materialize_dreamplace_source_tree",
                 "prepare_dreamplace_overlay", "prepare_openroad_runtime"):
        monkeypatch.setattr(evaluation, name, lambda *args: None)
    monkeypatch.setattr(evaluation, "isolated_clone", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation, "copy_upstream_stages", lambda *args: None)
    monkeypatch.setattr(evaluation, "_deterministic_dreamplace_params", lambda *args, **kwargs: env.output / "fixture-params.json")
    monkeypatch.setattr(evaluation, "_dreamplace_global_placement_command",
                        lambda *args, **kwargs: (["synthetic-native-fixture"], ROOT))
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""))
    def native_fixture(*args, **kwargs):
        stage = env.output / "run/30_placement/stages/02_global_place"
        stage.mkdir(parents=True)
        (stage / "params.json").write_text(json.dumps(native_params()))
        (stage / "run.log").write_text(HEADER + row() + row(1, 2, shared="1.5") +
            "[INFO] iteration 1, wHPWL 1000, Overflow [0.1, 0.2], MaxDensity [0.6, 0.7]\n")
        (stage / "stage_manifest.json").write_text('{"returncode": 0, "target_density": 0.6}')
        (stage / "placement.gp.def").write_text("synthetic placement fixture\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(evaluation, "_run_native_flow", native_fixture)
    result = evaluation.evaluate_task(ROOT / "local_tasks" / "placement_parameters_reference_20260923",
        env.candidate, ROOT.parent, env.output, baseline_run=env.baseline)
    assert result.metrics["runs_successfully"] == 1
    assert result.metrics["nesterov_control_updates"] == 2
    assert result.metrics["nesterov_shared_multiplier_final"] == 1.5
    assert result.metrics["placement_native_hpwl"] == 1000
    assert "dynamic_lr_updates" not in result.metrics
    evidence = result.artifacts["reference_placement_evidence"]
    assert evidence["dynamic_controls"]["call_count"] == 2
    assert result.artifacts["cascade"][-1] == {"name": "metrics", "status": "passed"}


@pytest.fixture
def docker_runtime(tmp_path, monkeypatch):
    runtime = object.__new__(Runtime)
    runtime.source = tmp_path / "source"
    runtime.framework = ROOT
    runtime.batch = tmp_path / "batch"
    runtime.batch.mkdir()
    runtime.image, runtime.cpus, runtime.memory_gb, runtime.gpus = IMAGE, 16, 48, [1]
    archive = tmp_path / "archive"
    archive.mkdir()
    original = tmp_path / "historical-source"
    index = archive / "reference.json"
    index.write_text(json.dumps({"original_root": str(original)}))
    expected = hashlib.sha256(index.read_bytes()).hexdigest()
    monkeypatch.setattr(ref, "identity", lambda: {"reference_json_sha256": expected})
    runtime.reference_root = archive
    return runtime, original


def test_runtime_docker_copy_mode_and_archive_mounts_are_explicit(docker_runtime):
    runtime, original = docker_runtime
    legacy = runtime.docker(["true"], 1, runtime.batch / "legacy-scratch")
    assert "OPEN3D_LINK_FROZEN_UPSTREAM=1" in legacy
    assert not any(str(runtime.reference_root) in value for value in legacy)
    runtime.reference_baseline = ref.BASELINE_ID
    aligned = runtime.docker(["true"], 1, runtime.batch / "new-scratch")
    assert "OPEN3D_LINK_FROZEN_UPSTREAM=0" in aligned
    assert "OPEN3D_LINK_FROZEN_UPSTREAM=1" not in aligned
    assert f"{runtime.reference_root}:{runtime.reference_root}:ro" in aligned
    assert f"{runtime.reference_root}:{original}:ro" in aligned


def test_local_runtime_preserves_read_only_batch_and_archive(docker_runtime):
    runtime, original = docker_runtime
    local = object.__new__(smoke.LocalSmokeRuntime)
    local.__dict__.update(runtime.__dict__)
    local.reference_baseline = ref.BASELINE_ID
    local.write_root = local.batch / "private-evaluation"
    local.write_root.mkdir()
    command = local.docker(["true"], 1, local.batch / "local-scratch")
    assert f"{local.batch}:{local.batch}:ro" in command
    assert f"{local.write_root}:{local.write_root}" in command
    assert f"{local.reference_root}:{original}:ro" in command
    assert "OPEN3DBENCH_IMAGE_ID=" + IMAGE in command
    assert "OPEN3D_LINK_FROZEN_UPSTREAM=0" in command


def test_local_runtime_mounts_external_qualified_reference_read_only(docker_runtime, tmp_path):
    runtime, original = docker_runtime
    local = object.__new__(smoke.LocalSmokeRuntime)
    local.__dict__.update(runtime.__dict__)
    local.write_root = local.batch / "candidate-evaluation"
    local.write_root.mkdir()
    local.adopted_reference_run = tmp_path / "external-qualified-run"
    local.adopted_reference_run.mkdir()
    command = local.docker(["true"], 1, local.batch / "local-scratch")
    assert f"{local.adopted_reference_run}:{local.adopted_reference_run}:ro" in command
    assert f"{local.adopted_reference_run}:{local.adopted_reference_run}" not in command
    assert f"{local.write_root}:{local.write_root}" in command


def test_runtime_prepare_rejects_old_existing_bundle_without_subprocess(docker_runtime, monkeypatch):
    runtime, original = docker_runtime
    runtime.reference_baseline = ref.BASELINE_ID
    bundle = runtime.batch / "existing"
    bundle.mkdir()
    (bundle / "benchmark_input.json").write_text('{"benchmark_id":"old"}')
    monkeypatch.setattr(runtime, "profile", lambda case: ref.PROFILE_ROOT / "bp_fe_lol_ng45_reference_20260923")
    monkeypatch.setattr(ref, "validate_reference_root", lambda *args: None)
    calls = []
    def reject(path, case, *, fixed_partition=False):
        calls.append((path, case, fixed_partition))
        raise ValueError("Input bundle is not the versioned reference-aligned profile")
    monkeypatch.setattr(ref, "validate_reference_bundle", reject)
    monkeypatch.setattr("scripts.codesign_runtime.subprocess.run", forbidden)
    monkeypatch.setattr(runtime, "execute", forbidden)
    with pytest.raises(ValueError, match="versioned"):
        runtime.prepare("bp_fe", bundle)
    assert calls == [(bundle, "bp_fe", True)]
