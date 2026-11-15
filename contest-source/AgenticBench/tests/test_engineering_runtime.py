import copy
import json
from pathlib import Path
import pytest

from engineering_bench.tasks import TASK_ROOT, load_task, read_json
from engineering_bench.protocol import sha
from engineering_bench.campaign import load_campaign, preflight
from engineering_bench.evaluation import validate_native_result
from engineering_bench import model
from engineering_bench.input_audit import clock_table
from global_bench.workspace_codex import build_workspace_command


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": path.name, "sha256": sha(path)}


def native_fixture(tmp_path, task_id="cross_die_cts"):
    # Deliberately synthetic. Nothing here launches or qualifies an EDA run.
    task = load_task(task_id)
    evidence = dump(tmp_path / "proof.json", {"test_fixture_only": True})
    checks = {name: {"passed": True, "evidence": evidence} for name in task.data["required_capabilities"]}
    timing = {"setup_wns_ns": -.5, "setup_tns_ns": -2., "hold_wns_ns": -.1,
              "hold_tns_ns": -.3, "hold_violating_endpoints": ["x"]}
    ref = {"legal": True, "case": "bp", "endpoint": task.data["endpoint"],
        "primary_metric": 1., "measurements": {"clock/rise/max": {"a": 0., "b": 1.}},
        "expected_inventory": {"clock/rise/max": ["a", "b"]}, "timing": timing,
        "timing_tolerances": {k: 0. for k in timing if k != "hold_violating_endpoints"}}
    reference = dump(tmp_path / "reference.json", ref)
    q = {"kind": "engineering-native-qualification-v1", "task": task.task_id,
         "task_digest": task.digest(), "case": "bp", "seed": 1001,
         "input_stage": task.data["input_stage"], "endpoint": task.data["endpoint"],
         "physical_success": True, "source_sha256": "a" * 64, "runtime_sha256": "b" * 64,
         "input_sha256": "c" * 64, "evaluator_sha256": "d" * 64,
         "capabilities": checks, "reference": reference}
    qpath = tmp_path / "qualification.json"
    dump(qpath, q)
    measurements = dump(tmp_path / "measurements.json", {"timing": timing,
                        "values": {"clock/rise/max": {"a": 0., "b": .5}}})
    result = {"kind": "engineering-native-result-v1", "task": task.task_id,
        "task_digest": task.digest(), "case": "bp", "seed": 1001,
        "input_sha256": "c" * 64, "evaluator_sha256": "d" * 64,
        "endpoint": task.data["endpoint"], "submission_sha256": "e" * 64,
        "runtime_sha256": "f" * 64, "qualification_sha256": sha(qpath),
        "checks": checks, "measurements": measurements}
    rpath = tmp_path / "result.json"
    dump(rpath, result)
    return task, qpath, rpath


def validate(task, qpath, rpath):
    return validate_native_result(task, "bp", qpath, rpath, submission_sha256="e" * 64,
                                  runtime_sha256="f" * 64)


def test_reducer_uses_raw_data_not_candidate_score(tmp_path):
    task, qpath, rpath = native_fixture(tmp_path)
    result = read_json(rpath)
    result["candidate_claimed_score"] = 100000
    dump(rpath, result)
    assert validate(task, qpath, rpath)["candidate_metric"] == .5


@pytest.mark.parametrize("timing", [None, {"setup_wns_ns": -100.0}])
def test_cts_skew_only_does_not_require_timing_closure(tmp_path, timing):
    task, qpath, rpath = native_fixture(tmp_path)
    measurements = read_json(tmp_path / "measurements.json")
    if timing is None:
        measurements.pop("timing")
    else:
        measurements["timing"] = timing
    result = read_json(rpath)
    result["measurements"] = dump(tmp_path / "measurements.json", measurements)
    dump(rpath, result)
    checked = validate(task, qpath, rpath)
    assert checked["candidate_metric"] == .5
    assert checked["timing_guard"] == {"required": False, "passed": None, "failures": []}
    assert checked["timing_diagnostic"] == timing
    assert checked["endpoint"] == "post_cts_placement_rc_joint_skew"


@pytest.mark.parametrize("task_id", ["clock_compensation", "multi_hbt_fanout"])
def test_current_stage_sta_tasks_keep_their_timing_guard(tmp_path, task_id):
    task, qpath, rpath = native_fixture(tmp_path, task_id)
    measurements = read_json(tmp_path / "measurements.json")
    measurements["timing"]["hold_wns_ns"] = -5.0
    result = read_json(rpath)
    result["measurements"] = dump(tmp_path / "measurements.json", measurements)
    dump(rpath, result)
    with pytest.raises(ValueError, match="setup/hold guard failed"):
        validate(task, qpath, rpath)


def test_former_grt_endpoint_cannot_be_reused_for_stage_sta(tmp_path):
    task, qpath, rpath = native_fixture(tmp_path, "multi_hbt_fanout")
    qualification = read_json(qpath)
    qualification["endpoint"] = "grt_rc_fixed_sink_joint_sta"
    dump(qpath, qualification)
    with pytest.raises(ValueError, match="Qualification mismatch: endpoint"):
        validate(task, qpath, rpath)


@pytest.mark.parametrize("field", ["case", "endpoint", "runtime_sha256", "submission_sha256", "input_sha256", "evaluator_sha256"])
def test_identity_mismatch_rejected(tmp_path, field):
    task, qpath, rpath = native_fixture(tmp_path)
    result = read_json(rpath)
    result[field] = "incorrect"
    dump(rpath, result)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate(task, qpath, rpath)


def test_missing_native_gate_and_dropped_sink(tmp_path):
    task, qpath, rpath = native_fixture(tmp_path)
    result = read_json(rpath)
    result["checks"].pop("physical_equivalence")
    dump(rpath, result)
    with pytest.raises(ValueError, match="physical gate"):
        validate(task, qpath, rpath)
    task, qpath, rpath = native_fixture(tmp_path)
    result = read_json(rpath)
    measurement = read_json(tmp_path / "measurements.json")
    del measurement["values"]["clock/rise/max"]["b"]
    result["measurements"] = dump(tmp_path / "measurements.json", measurement)
    dump(rpath, result)
    with pytest.raises(ValueError, match="sink inventory"):
        validate(task, qpath, rpath)


def test_campaign_has_twelve_finals_no_repetitions(tmp_path):
    plan = load_campaign(TASK_ROOT / "campaign.gpt56.json")
    assert len(plan["cases"]) * len(plan["tasks"]) == 12
    result = preflight(plan, tmp_path)
    assert not result["ready"]
    assert result["model_calls_dispatched"] == result["candidate_finals_dispatched"] == 0
    assert all(c["status"] == "blocked" for t in result["tasks"] for c in t["cases"])


def test_missing_interface_never_launches_paid_development(tmp_path, monkeypatch):
    def should_not_call(**kwargs):
        pytest.fail("An unqualified task must not call the developer")
    monkeypatch.setattr(model, "run_workspace_turn", should_not_call)
    with pytest.raises(FileNotFoundError):
        model.develop(load_task("clock_compensation"), tmp_path / "missing",
                      model="gpt-5.6-sol", output=tmp_path / "development")
    assert not (tmp_path / "development").exists()


def test_explicit_source_phase_is_not_a_qualification_or_final(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(model, "run_workspace_turn", lambda **kw: calls.append(kw) or {"success": True})
    model.develop_source(load_task("clock_compensation"), model="gpt-5.6-sol",
                         output=tmp_path / "draft")
    dispatch = read_json(tmp_path / "draft/dispatch.json")
    assert dispatch["phase"] == "preliminary_source_development"
    assert dispatch["candidate_finals"] == 0
    assert dispatch["physical_qualification"] is False
    assert dispatch["submission_sealed"] is False
    assert len(calls) == 1 and calls[0]["thread_id"] is None
    assert "not the sealed final" in calls[0]["prompt"]
    assert "Do not invent missing native APIs" in calls[0]["prompt"]
    assert not (tmp_path / "draft/qualification.json").exists()
    assert not (tmp_path / "draft/submission.json").exists()
    with pytest.raises(FileExistsError):
        model.develop_source(load_task("clock_compensation"), model="gpt-5.6-sol",
                             output=tmp_path / "draft")
    assert len(calls) == 1


def test_development_source_mounts_are_readonly_except_scope(tmp_path):
    from engineering_bench.prepare_development import private_command
    subject = tmp_path / "source"
    subject.mkdir()
    (subject / ".git").mkdir()
    (subject / "engineering").mkdir()
    dirs = {key: tmp_path / key for key in ("control", "scratch", "results", "benchmark", "reference")}
    for path in dirs.values():
        path.mkdir()
    command = private_command(load_task("clock_compensation"), subject,
        name="o3db-agent-test", image="sha256:" + "a" * 64, protected=(), cpus=8, memory_gb=32, **dirs)
    assert "--gpus" not in command
    assert command[-3:] == ["sha256:" + "a" * 64, "sleep", "infinity"]
    assert f"type=bind,src={subject},dst=/workspace,readonly" in command
    assert f"type=bind,src={subject / 'engineering'},dst=/workspace/engineering,readonly=false" in command
    assert f"type=bind,src={subject / '.git'},dst=/workspace/.git,readonly" in command


@pytest.mark.parametrize("vendored", [False, True])
def test_clone_uses_gitlinks_not_stale_gitmodules_entries(tmp_path, vendored):
    import subprocess
    from engineering_bench.submission import clone_source
    source = tmp_path / "original"
    source.mkdir()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Engineering fixture")
    git("config", "user.email", "fixture@example.invalid")
    (source / "README").write_text("synthetic fixture")
    git("add", "README")
    git("commit", "-qm", "base")
    child_revision = git("rev-parse", "HEAD")
    (source / ".gitmodules").write_text('[submodule "optional"]\n\tpath = optional\n\turl = https://example.invalid/optional\n')
    git("add", ".gitmodules")
    (source / "optional").mkdir()
    if vendored:
        (source / "optional/source.c").write_text("int main() { return 0; }\n")
        git("add", "optional/source.c")
    else:
        git("update-index", "--add", "--cacheinfo", "160000", child_revision, "optional")
    git("commit", "-qm", "dependency")
    if not vendored:
        with pytest.raises(ValueError, match="uninitialized"):
            clone_source(source, tmp_path / "strict")
    else:
        clone_source(source, tmp_path / "draft")
        assert (tmp_path / "draft/optional/source.c").read_text() == "int main() { return 0; }\n"
        assert not (tmp_path / "draft/optional/.git").exists()


def test_model_is_explicit_without_changing_defaults(tmp_path):
    command = build_workspace_command(binary="codex", control=tmp_path, final=tmp_path / "out",
        overrides=(), model="test-model")
    assert command[command.index("--model") + 1] == "test-model"
    for bad in ("x\n", "--unsafe", "", None):
        with pytest.raises(ValueError):
            build_workspace_command(binary="codex", control=tmp_path, final=tmp_path / "out",
                                    overrides=(), model=bad)


def test_clock_parser_does_not_mix_auxiliary_falling_edge(tmp_path):
    path = tmp_path / "clocks.tsv"
    path.write_text("clock\tdie\tpin\tactive_edge\tarrival_max_rise_ns\tarrival_max_fall_ns\tarrival_min_rise_ns\tarrival_min_fall_ns\tclock_domain_count\n"
                    "clk\tbottom\ta/CK\trise\t0.1\t5.0\t0.1\t5.0\t1\n"
                    "aux\ttop\tb/CK\tfall\t10\t20\t10\t20\t1\n")
    groups = clock_table(path)
    assert groups["clk/rise/max/default"] == {"a/CK": .1}
    assert groups["aux/fall/max/default"] == {"b/CK": 20.}


def test_clock_parser_rejects_ambiguous_sinks(tmp_path):
    path = tmp_path / "clocks.tsv"
    path.write_text("clock\tdie\tpin\tactive_edge\tarrival_max_rise_ns\tarrival_max_fall_ns\tarrival_min_rise_ns\tarrival_min_fall_ns\tclock_domain_count\n"
                    "clk\tbottom\ta/CK\trise\t0.1\t5.0\t0.1\t5.0\t2\n")
    with pytest.raises(ValueError, match="Ambiguous"):
        clock_table(path)
