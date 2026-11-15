"""Real importer/evaluator wiring, with native commands replaced by fixtures.

No Docker, physical tools, model calls or real evaluation metrics are used.
The archive fingerprint fixture replaces data identities, not bundle validators.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from global_bench.model import load_global_task
from local_bench import reference_baseline as baseline
from local_bench.workspace import isolated_clone
from scripts import workspace_codesign_evaluator as bridge
from scripts import workspace_reference as references
from scripts.run_placement_interface_qualification import QualificationRuntime
from scripts.workspace_task_selection import TIMING_TEN_TASK
from test_adopt_reference_20260923 import adopt, frozen, write

ROOT = Path(__file__).resolve().parents[1]


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


@pytest.fixture
def imported(frozen, monkeypatch, tmp_path):
    source = frozen["source"]
    (source / "AgenticBench").mkdir()
    (source / "AgenticBench/trusted-control.txt").write_text("trusted fixture\n")
    (source / "algorithm.py").write_text("VALUE = 1\n")
    for relative in (baseline.PARTITION_CONFIG, baseline.PLACEMENT_CONFIG):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT.parent / relative, target)
    git(source, "add", ".")
    git(source, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "complete importer source fixture")
    runs = tmp_path / "runs"
    runs.mkdir()
    bundle = runs / "inputs/bp_fe/bundle"
    adopt(frozen, output=bundle)
    runtime = object.__new__(QualificationRuntime)
    runtime.source, runtime.batch, runtime.framework = source, runs, source / "AgenticBench"
    runtime.source_commit = git(source, "rev-parse", "HEAD")
    runtime.cpus, runtime.memory_gb = 16, 48
    runtime.min_gpu_free_mb, runtime.min_data_free_gb = 2048, 64
    runtime.image = "sha256:3b8e1793c0e780308101b480fa17f05771ad97d728209146e842bf6dea471583"
    runtime.reference_baseline = "reference_20260923"
    runtime.reference_policy, runtime.reference_store = "FRESH", runs / "reference-store"
    runtime.reference_root = frozen["archive"]
    runtime.reference_record = None
    admissions = []
    @contextmanager
    def slot():
        admissions.append(True)
        yield 1
    runtime.slot = slot
    def forbidden(*args, **kwargs):
        raise AssertionError("Global evaluator must not call the local Docker/execute adapter")
    runtime.docker = runtime.execute = forbidden
    task = load_global_task(ROOT / "global_tasks" / TIMING_TEN_TASK)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp_fe", "timing", bundle, task)
    return {**frozen, "runtime": runtime, "bundle": bundle, "task": task,
            "evaluator": evaluator, "admissions": admissions,
            "manifest": json.loads((bundle / "benchmark_input.json").read_text()),
            "output": runs / "references/bp_fe/timing/result"}


def alias(value, manifest=None, validation=None, output=None):
    return bridge.input_source_alias(value["runtime"], manifest or value["manifest"], value["bundle"],
        output or value["output"], baseline_validation=(
            {"baseline_id": "reference_20260923"} if validation is None else validation))


def stage(run, name, directory):
    path = run / directory
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({"stage": name}))
    write(path / "stage_manifest.json", {"schema_version": 1, "stage": name, "status": "success",
        "signature": {}, "artifacts": {"metrics": {"path": "metrics.json",
        "bytes": (path / "metrics.json").stat().st_size, "sha256": bridge._sha(path / "metrics.json")}}})


def fake_native(value, monkeypatch):
    observed = []
    def call(phase, argv, output, deadline):
        observed.append((phase, argv))
        run = output / "run"
        if phase in {"frontend", "backend"}:
            stages = ({"timing_extract": "15_timing_extract", "partition": "20_partition", "placement": "30_placement"}
                      if phase == "frontend" else {"pdn": "40_pdn", "hbt": "50_hbt", "opt": "60_opt"})
            status = bridge._read(run / "run.json")
            for name, directory in stages.items():
                stage(run, name, directory)
                status["stages"][name] = {"status": "success", "directory": directory}
            write(run / "run.json", status)
            if phase == "frontend":
                for name in ("design.v", "design.sdc", "floorplan.json", "io_plan.csv", "preplan_manifest.json", "pdn_plan.json"):
                    shutil.copyfile(run / "10_floorplan" / name, run / "30_placement" / name)
                for die in ("bottom", "top"):
                    destination = run / "30_placement/die_init" / die / "design.sdc"
                    destination.parent.mkdir(parents=True)
                    shutil.copyfile(run / "10_floorplan/design.sdc", destination)
                for directory in ("20_partition", "30_placement"):
                    (run / directory / "partition_assignment.csv").write_text("inst_name,die\nu1,bottom\nu2,top\n")
            else:
                write(run / "60_opt/checkpoint.json", {"stage": "opt"})
        if phase == "verifier":
            write(output / "verification/verification.json", {"passed": True, "metrics": {"combined_score": 1.0}})
    monkeypatch.setattr(value["evaluator"], "_call", call)
    return observed


def mounts(argv):
    rows = [argv[index + 1].split(":") for index, arg in enumerate(argv) if arg == "-v"]
    assert len(rows) == len({row[1] for row in rows}), "Docker mount targets must be unique"
    return {target: (source, mode) for source, target, mode in rows}


def test_real_adopter_manifest_reaches_evaluator_with_distinct_trusted_mounts(imported, monkeypatch):
    value, runtime = imported, imported["runtime"]
    observed = fake_native(value, monkeypatch)
    assert not runtime.source.is_relative_to(runtime.batch)
    assert alias(value) == runtime.source
    result = value["evaluator"].evaluate(runtime.source, value["output"])
    assert result["success"], result
    assert [phase for phase, _ in observed] == ["frontend", "backend", "verifier"]
    for phase, argv in observed:
        mapping = mounts(argv)
        expected = value["output"] / ("source" if phase == "frontend" else "backend-source")
        assert mapping[str(runtime.source)] == (str(expected), "ro")
        assert mapping[str(runtime.framework)] == (str(runtime.framework), "ro")
        assert mapping[str(runtime.batch)] == (str(runtime.batch), "ro")
        assert "OPEN3D_OPT_PASSES=1" in argv
        assert not any(mode == "rw" and target in {str(runtime.source), str(runtime.framework), str(value["bundle"])}
                       for target, (_, mode) in mapping.items())
    contract = bridge._read(value["output"] / "evaluation-contract.json")
    assert contract["source_alias"] == str(runtime.source)
    assert contract["source_revision"] == runtime.source_commit
    assert contract["task_digest"] == value["task"].digest()
    assert contract["input_manifest_sha256"] == bridge._sha(value["bundle"] / "benchmark_input.json")
    assert contract["reference_purpose"] == "FRESH" and contract["opt_passes"] == 1
    assert runtime.write_root is None and len(value["admissions"]) == 1
    with pytest.raises(FileExistsError):
        value["evaluator"].evaluate(runtime.source, value["output"])
    assert len(observed) == 3


def test_legacy_batch_alias_has_no_new_importer_requirement(imported, monkeypatch):
    value = imported
    legacy = value["runtime"].batch / "input-source"
    manifest = copy.deepcopy(value["manifest"])
    manifest["environment"]["open3dbench"] = {"root": str(legacy)}
    manifest.pop("current_importer_source")
    manifest.pop("import_kind")
    write(value["bundle"] / "benchmark_input.json", manifest)
    observed = fake_native(value, monkeypatch)
    result = value["evaluator"].evaluate(value["source"], value["output"])
    assert result["success"], result
    for phase, argv in observed:
        mapping = mounts(argv)
        assert mapping[str(value["source"])] == (str(value["source"]), "ro")
        assert mapping[str(legacy)][1] == "ro"


@pytest.mark.parametrize("has_receipt", [False, True])
def test_exact_source_inside_batch_still_requires_verified_importer(imported, has_receipt):
    runtime = copy.copy(imported["runtime"])
    runtime.source = runtime.batch / "nested-frozen-source"
    runtime.framework = runtime.source / "AgenticBench"
    runtime.framework.mkdir(parents=True)
    manifest = copy.deepcopy(imported["manifest"])
    for record in (manifest["environment"]["open3dbench"], manifest["current_importer_source"]):
        record["root"] = str(runtime.source)
    if not has_receipt:
        manifest.pop("current_importer_source")
        with pytest.raises(ValueError, match="importer identity"):
            alias({**imported, "runtime": runtime}, manifest)
    else:
        assert alias({**imported, "runtime": runtime}, manifest) == runtime.source


def test_trusted_framework_submount_does_not_shadow_editable_frontend(imported):
    runtime, evaluator = imported["runtime"], imported["evaluator"]
    subject = runtime.batch / "editable-subject"
    isolated_clone(runtime.source, subject)
    editable = Path("Place-3D/src/partition/fixture_algorithm.py")
    (subject / editable).parent.mkdir(parents=True)
    (subject / editable).write_text("VALUE = 2\n")
    changes = bridge.source_changes(subject, runtime.source_commit, imported["task"].workspace.protected_paths)
    output = runtime.batch / "editable-mount-fixture"
    output.mkdir()
    snapshot = evaluator._snapshot(subject, output, changes)
    mapping = mounts(evaluator._docker([], 1, output, snapshot, runtime.source,
        writable=(output / "run",), name="offline-editable-frontend"))
    def mapped_file(target):
        destination = max((Path(path) for path in mapping if target.is_relative_to(path)), key=lambda path: len(path.parts))
        return Path(mapping[str(destination)][0]) / target.relative_to(destination)
    assert mapped_file(runtime.source / editable) == snapshot / editable
    assert mapped_file(runtime.source / editable).read_text() == "VALUE = 2\n"
    assert mapped_file(runtime.framework / "trusted-control.txt") == runtime.framework / "trusted-control.txt"


@pytest.mark.parametrize("candidate_edit", [False, True])
def test_object_submount_preserves_native_git_provenance_when_source_is_hidden(imported, candidate_edit):
    """Emulate the source-alias overlay locally; never invoke Docker or EDA."""
    runtime, evaluator = imported["runtime"], imported["evaluator"]
    output = runtime.batch / "git-mount-fixture"
    output.mkdir()
    snapshot = isolated_clone(runtime.source, output / "source")
    if candidate_edit:
        (snapshot / "algorithm.py").write_text("VALUE = 2\n")
    argv = evaluator._docker([], 1, output, snapshot, runtime.source,
        writable=(output / "run",), name="offline-object-mount")
    mapping = mounts(argv)
    objects = runtime.source / ".git/objects"
    assert mapping[str(objects)] == (str(objects), "ro")
    assert str(runtime.source / ".git") not in mapping
    assert not any(str(runtime.source / (".git/" + name)) in mapping for name in ("HEAD", "index", "refs"))
    # First reproduce the broken view: clone replaces the source, so its
    # alternates file points back to its own empty object directory.
    hidden = runtime.source.with_name("hidden-frozen-source")
    runtime.source.rename(hidden)
    snapshot.rename(runtime.source)
    command = ["git", "--git-dir=" + str(runtime.source / ".git"),
               "--work-tree=" + str(runtime.source), "status", "--porcelain"]
    broken = subprocess.run(command, cwd=runtime.source, capture_output=True, text=True)
    assert broken.returncode != 0
    # A local symlink models only the added read-only objects submount. The
    # evaluated snapshot's own HEAD, index and working files remain untouched.
    shutil.rmtree(objects)
    objects.symlink_to(hidden / ".git/objects", target_is_directory=True)
    observed = subprocess.check_output(command, cwd=runtime.source, text=True)
    assert bool(observed.strip()) is candidate_edit
    assert git(runtime.source, "rev-parse", "HEAD") == runtime.source_commit
    assert git(runtime.source, "show", "HEAD:algorithm.py") == "VALUE = 1"


def test_worktree_object_store_is_mounted_without_exposing_git_administration(imported):
    runtime = copy.copy(imported["runtime"])
    checkout = runtime.source.with_name("worktree-source")
    git(runtime.source, "worktree", "add", "--detach", str(checkout), runtime.source_commit)
    expected_objects = runtime.source / ".git/objects"
    runtime.source, runtime.framework = checkout, checkout / "AgenticBench"
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp_fe", "timing", imported["bundle"], imported["task"])
    output = runtime.batch / "worktree-mount-fixture"
    output.mkdir()
    snapshot = isolated_clone(checkout, output / "source")
    mapping = mounts(evaluator._docker([], 1, output, snapshot, checkout,
        writable=(output / "run",), name="offline-worktree-objects"))
    assert mapping[str(expected_objects)] == (str(expected_objects), "ro")
    assert str(checkout / ".git") not in mapping
    assert str(expected_objects.parent) not in mapping
    assert (snapshot / ".git/objects/info/alternates").read_text().strip() == str(expected_objects)


@pytest.mark.parametrize("name", ["alternates", "http-alternates"])
def test_source_object_store_never_expands_an_external_alternate_chain(imported, name):
    runtime, evaluator = imported["runtime"], imported["evaluator"]
    alternate = runtime.source / ".git/objects/info" / name
    alternate.parent.mkdir(parents=True, exist_ok=True)
    alternate.write_text(str(runtime.source.with_name("unreviewed-objects")) + "\n")
    output = runtime.batch / "external-objects-fixture"
    output.mkdir()
    with pytest.raises(ValueError, match="self-contained Git object store"):
        evaluator._docker([], 1, output, output / "source", runtime.source,
            writable=(output / "run",), name="offline-reject-alternates")


def test_object_store_ignores_process_git_path_overrides(imported, monkeypatch):
    source = imported["source"]
    unexpected = source.with_name("untrusted-objects")
    unexpected.mkdir()
    for name in ("GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_DIR"):
        monkeypatch.setenv(name, str(unexpected))
    assert bridge._source_git_objects(source) == source / ".git/objects"


def test_object_store_cannot_be_a_link_to_an_external_directory(imported):
    source = imported["source"]
    objects = source / ".git/objects"
    outside = source.with_name("external-object-store")
    objects.rename(outside)
    objects.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink common directory"):
        bridge._source_git_objects(source)


@pytest.mark.parametrize("target_kind", ["file", "dependency", "source-root"])
def test_exact_alias_rejects_snapshot_links_back_into_shadowed_source(imported, target_kind):
    runtime, evaluator = imported["runtime"], imported["evaluator"]
    output = runtime.batch / "shadowed-link-fixture"
    output.mkdir()
    snapshot = isolated_clone(runtime.source, output / "source")
    if target_kind == "dependency":
        target = runtime.source / "OpenROAD-3D/openroad"
        target.mkdir(parents=True)
        link = snapshot / "OpenROAD-3D/openroad"
        link.parent.mkdir(parents=True)
    else:
        target = runtime.source / "algorithm.py" if target_kind == "file" else runtime.source
        link = snapshot / "source-link"
    link.symlink_to(target, target_is_directory=target_kind != "file")
    with pytest.raises(ValueError, match="shadowed frozen source"):
        evaluator._docker([], 1, output, snapshot, runtime.source,
            writable=(output / "run",), name="offline-reject-shadowed-link")


def test_standard_empty_gitlink_and_nonshadowed_snapshot_links_stay_usable(imported):
    runtime, evaluator = imported["runtime"], imported["evaluator"]
    output = runtime.batch / "empty-gitlink-fixture"
    output.mkdir()
    snapshot = isolated_clone(runtime.source, output / "source")
    dependency = snapshot / "OpenROAD-3D/openroad"
    dependency.mkdir(parents=True)
    (snapshot / "internal-link").symlink_to("algorithm.py")
    (snapshot / "archive-link").symlink_to(imported["archive"], target_is_directory=True)
    mapping = mounts(evaluator._docker([], 1, output, snapshot, runtime.source,
        writable=(output / "run",), name="offline-empty-gitlink"))
    assert dependency.is_dir() and not any(dependency.iterdir())
    assert str(runtime.source / "OpenROAD-3D/openroad") not in mapping
    assert mapping[str(runtime.source)] == (str(snapshot), "ro")


@pytest.mark.parametrize("section,key,wrong", [
    ("environment", "root", "/outside/old-source"),
    ("environment", "revision", "0" * 40), ("environment", "dirty", True),
    ("environment", "dirty", 0), ("environment", "dirty", None),
    ("current_importer_source", "root", "/outside/old-source"),
    ("current_importer_source", "revision", "0" * 40),
    ("current_importer_source", "dirty", True), ("current_importer_source", "dirty", 0),
    ("current_importer_source", "dirty", None),
])
def test_importer_identity_is_fail_closed(imported, section, key, wrong):
    manifest = copy.deepcopy(imported["manifest"])
    record = manifest["environment"]["open3dbench"] if section == "environment" else manifest[section]
    record[key] = wrong
    with pytest.raises(ValueError):
        alias(imported, manifest)


@pytest.mark.parametrize("mutation", ["missing-importer", "missing-kind", "wrong-kind", "wrong-baseline", "unvalidated"])
def test_exact_source_exception_requires_adopted_validated_bundle(imported, mutation):
    manifest = copy.deepcopy(imported["manifest"])
    validation = {"baseline_id": "reference_20260923"}
    if mutation == "missing-importer":
        manifest.pop("current_importer_source")
    elif mutation == "missing-kind":
        manifest.pop("import_kind")
    elif mutation == "wrong-kind":
        manifest["import_kind"] = "untrusted"
    elif mutation == "wrong-baseline":
        manifest["baseline_id"] = "untrusted"
    else:
        validation = {}
    with pytest.raises(ValueError, match="importer identity"):
        alias(imported, manifest, validation)


@pytest.mark.parametrize("form", ["parent", "child", "sibling", "relative", "dotdot", "dot", "colon", "newline", "tab", "nul", "symlink"])
def test_unsafe_or_external_aliases_rejected_before_resolution(imported, form):
    root = imported["source"]
    values = {"parent": str(root.parent), "child": str(root / "AgenticBench"),
        "sibling": str(root.with_name("other-source")), "relative": "isolated-source",
        "dotdot": str(root.parent / "unused/..") + "/" + root.name,
        "dot": str(root.parent) + "/./" + root.name, "colon": str(root) + ":ro",
        "newline": str(root) + "\n", "tab": str(root) + "\t", "nul": str(root) + "\0"}
    if form == "symlink":
        link = root.with_name("linked-source")
        link.symlink_to(root, target_is_directory=True)
        values[form] = str(link)
    manifest = copy.deepcopy(imported["manifest"])
    manifest["environment"]["open3dbench"]["root"] = values[form]
    with pytest.raises(ValueError):
        alias(imported, manifest)


@pytest.mark.parametrize("relation", ["equal", "parent", "child"])
@pytest.mark.parametrize("target", ["bundle", "output"])
def test_contained_alias_cannot_shadow_inputs_or_outputs(imported, relation, target):
    path = imported[target]
    path = path if relation == "equal" else path.parent if relation == "parent" else path / "child"
    manifest = copy.deepcopy(imported["manifest"])
    manifest["environment"]["open3dbench"]["root"] = str(path)
    with pytest.raises(ValueError, match="overlaps"):
        alias(imported, manifest)


def test_bundle_artifacts_are_validated_before_alias_exception(imported, monkeypatch):
    value = imported
    observed = fake_native(value, monkeypatch)
    (value["bundle"] / "run/00_input/design.def").write_text("corrupt\n")
    monkeypatch.setattr(bridge, "input_source_alias", lambda *a, **k: pytest.fail("Alias must follow native input validation"))
    result = value["evaluator"].evaluate(value["source"], value["output"])
    assert not result["success"] and "artifact" in result["error"]
    assert not observed and not value["admissions"]


def test_conflicting_duplicate_mount_is_rejected(imported):
    value, runtime = imported, imported["runtime"]
    output = runtime.batch / "mount-fixture"
    output.mkdir()
    with pytest.raises(ValueError, match="Conflicting evaluator mount target"):
        value["evaluator"]._docker([], 1, output, output / "source", runtime.source,
            writable=(runtime.source,), name="offline-conflict")


@pytest.mark.parametrize("relative", [".", "locks", "locks/gpu-1.lock"])
def test_alias_cannot_cover_shared_control_state(imported, relative):
    manifest = copy.deepcopy(imported["manifest"])
    manifest["environment"]["open3dbench"]["root"] = str(imported["runtime"].batch / relative)
    with pytest.raises(ValueError):
        alias(imported, manifest)


@pytest.mark.parametrize("phase", ["frontend", "backend", "verifier", "runtime-overlay"])
@pytest.mark.parametrize("kind", ["legacy", "importer"])
def test_all_phase_mount_targets_are_unique_and_control_is_trusted(imported, phase, kind):
    value, runtime = imported, imported["runtime"]
    output = runtime.batch / "phase-mount-fixture"
    output.mkdir()
    source_alias = runtime.source if kind == "importer" else runtime.batch / "legacy-alias"
    selected = output / ("source" if phase == "frontend" else "backend-source")
    writable = output / ("verification" if phase == "verifier" else "runtime" if phase == "runtime-overlay" else "run")
    argv = value["evaluator"]._docker([], 1, output, selected, source_alias,
        writable=(writable,), readonly=(output / "run/00_input",), name="offline-" + phase)
    mapping = mounts(argv)
    assert mapping[str(source_alias)] == (str(selected), "ro")
    if kind == "importer":
        assert mapping[str(runtime.framework)] == (str(runtime.framework), "ro")
    else:
        assert mapping[str(runtime.source)] == (str(runtime.source), "ro")
    assert {target for target, (_, mode) in mapping.items() if mode == "rw"} == {"/tmp", str(writable)}


def test_fresh_and_adopt_bind_identical_new_input_source_and_ten_round_task(imported, monkeypatch):
    value, runtime = imported, imported["runtime"]
    fresh = references.WorkspaceReference(runtime, "bp_fe", "timing", value["bundle"], value["task"])
    record = runtime.batch / "offline-record.json"
    write(record, {"producer": {"kind": "native-evaluation-v1"}})
    contracts = []
    def lookup(store, *, contract):
        contracts.append(contract)
        return record
    monkeypatch.setattr(references, "find_reference", lookup)
    monkeypatch.setattr(references, "load_reference", lambda *a, **k: json.loads(record.read_text()))
    runtime.reference_policy = "ADOPT"
    adopted = references.WorkspaceReference(runtime, "bp_fe", "timing", value["bundle"], value["task"])
    assert adopted.contract == fresh.contract == contracts[0]
    assert adopted.contract["physical"]["source_revision"] == runtime.source_commit
    assert adopted.contract["physical"]["opt_passes"] == 1
    assert adopted.contract["frozen_inputs"]["benchmark_input_manifest_sha256"] == bridge._sha(value["bundle"] / "benchmark_input.json")
    assert value["task"].task_id == TIMING_TEN_TASK
    assert alias(value) == runtime.source
