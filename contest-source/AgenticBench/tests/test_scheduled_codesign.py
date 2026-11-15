import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

from scripts import run_scheduled_codesign as scheduled
from scripts.run_scheduled_codesign import bind_task_contracts, check_model_preflight, task_contract

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sha256:" + "a" * 64


def preflight_value():
    return {"success": True, "exit_code": 0, "timed_out": False,
            "final_message": "OK\n", "model": "gpt-5.6-sol", "image": IMAGE,
            "qualification": "workspace-terminal-and-resume-v1", "eda_executed": False,
            "checks": {name: True for name in scheduled.REQUIRED_PREFLIGHT_CHECKS}}


@pytest.mark.parametrize("objective", ["timing", "thermal"])
def test_schedule_uses_global_task_budget_not_batch_default(objective):
    contract = task_contract(ROOT, objective)
    assert contract["rounds"] == 5
    assert len(contract["task_sha256"]) == 64
    assert contract["seed_policy"]["partition"] == 1001
    assert "one-pass" in contract["reference_policy"]
    assert contract["task_id"].endswith("_one_pass")
    assert contract["adapter"] == "codex-container-workspace-v1"


def test_actual_workspace_qualification_metadata_format(tmp_path):
    path = tmp_path / "result.json"
    value = preflight_value()
    path.write_text(json.dumps(value))
    result = check_model_preflight(path, IMAGE)
    assert len(result["result_sha256"]) == 64
    assert result["image"] == IMAGE
    assert result["qualification"] == "workspace-terminal-and-resume-v1"
    for update in ({"success": False}, {"exit_code": 1}, {"timed_out": True},
                   {"final_message": "failed"}, {"model": "different"},
                   {"qualification": "simple-api-probe"}, {"image": "sha256:" + "b" * 64},
                   {"eda_executed": True}, {"exit_code": False}, {"final_message": None}):
        path.write_text(json.dumps({**value, **update}))
        with pytest.raises(ValueError):
            check_model_preflight(path, IMAGE)


def test_simple_ok_api_probe_cannot_qualify_workspace_execution(tmp_path):
    path = tmp_path / "old-result.json"
    path.write_text(json.dumps({"success": True, "exit_code": 0, "timed_out": False,
                                "final_message": "OK", "model": "gpt-5.6-sol"}))
    with pytest.raises(ValueError, match="workspace terminal"):
        check_model_preflight(path, IMAGE)


@pytest.mark.parametrize("key", sorted(scheduled.REQUIRED_PREFLIGHT_CHECKS))
def test_each_workspace_boundary_and_resume_check_is_required(tmp_path, key):
    path = tmp_path / "result.json"
    value = preflight_value()
    del value["checks"][key]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="every required"):
        check_model_preflight(path, IMAGE)
    value["checks"][key] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="every required"):
        check_model_preflight(path, IMAGE)


@pytest.mark.parametrize("value", [None, [], "OK"])
def test_preflight_is_a_structured_qualification_not_a_scalar(tmp_path, value):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="receipt must be an object"):
        check_model_preflight(path, IMAGE)


def test_prepare_publishes_contract_before_objectives(tmp_path):
    path = tmp_path / "bp/task_contracts.json"
    value = {"case": "bp", "rounds": 5}
    bind_task_contracts(path, value, "prepare")
    assert json.loads(path.read_text()) == value
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("phase", ["timing", "thermal"])
def test_objective_requires_matching_contract_without_creating_or_rewriting(tmp_path, phase, monkeypatch):
    path = tmp_path / "bp/task_contracts.json"
    value = {"case": "bp", "rounds": 5}
    monkeypatch.setattr(scheduled, "immutable_json", lambda *args: pytest.fail("Objective must not write"))
    with pytest.raises(ValueError, match="missing"):
        bind_task_contracts(path, value, phase)
    assert not path.parent.exists()
    path.parent.mkdir()
    path.write_text(json.dumps({**value, "rounds": 10}))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    with pytest.raises(ValueError, match="differs"):
        bind_task_contracts(path, value, phase)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert not path.with_suffix(".json.tmp").exists()


def test_real_concurrent_objective_reads_preserve_contract_and_never_share_temporary(tmp_path, monkeypatch):
    path = tmp_path / "bp/task_contracts.json"
    value = {"case": "bp", "tasks": {phase: task_contract(ROOT, phase) for phase in ("timing", "thermal")}}
    bind_task_contracts(path, value, "prepare")
    path.chmod(0o444)
    before = (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
    monkeypatch.setattr(scheduled, "immutable_json", lambda *args: pytest.fail("Concurrent objectives must not write"))
    def verify(phase):
        for _ in range(40):
            bind_task_contracts(path, value, phase)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(verify, ["timing", "thermal"]))
    assert (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode) == before
    assert not path.with_suffix(".json.tmp").exists()


def scheduled_fixture(tmp_path, monkeypatch):
    preflight = tmp_path / "qualified.json"
    preflight.write_text(json.dumps(preflight_value()))
    batch = tmp_path / "batch"
    instances = []
    class FakeRuntime:
        def __init__(self, source, batch_root, image, gpu, cpus, memory_gb, **options):
            self.framework = ROOT
            self.source_commit = "f" * 40
            self.source, self.batch, self.image = source, batch_root, image
            self.gpus, self.cpus, self.memory_gb = [int(gpu)], cpus, memory_gb
            self.options = options
            self.prepared = []
            instances.append(self)
        def prepare(self, case, bundle):
            self.prepared.append((case, bundle))
            bundle.mkdir(parents=True)
            (bundle / "benchmark_input.json").write_text("{}")
    monkeypatch.setattr(scheduled, "Runtime", FakeRuntime)
    base = ["run_scheduled_codesign.py", "--case", "bp_fe", "--source", str(ROOT.parent),
            "--batch-root", str(batch), "--image", IMAGE, "--gpu", "1", "--cpus", "4",
            "--memory-gb", "32", "--min-gpu-free-mb", "2048", "--codex-bin", "/trusted/codex",
            "--preflight-result", str(preflight)]
    def invoke(phase, *extra):
        monkeypatch.setattr(sys, "argv", [*base, "--phase", phase, *extra])
        scheduled.main()
    return invoke, batch, instances, preflight


@pytest.mark.parametrize("objective", ["timing", "thermal"])
@pytest.mark.parametrize("policy", ["ADOPT", "FRESH", "EXTEND"])
def test_scheduled_job_dispatches_workspace_driver_and_uses_its_status_root(tmp_path, monkeypatch, capsys, objective, policy):
    invoke, batch, runtimes, _ = scheduled_fixture(tmp_path, monkeypatch)
    calls = []
    def workspace(runtime, case, phase, bundle, output, codex_bin):
        calls.append((runtime, case, phase, bundle, output, codex_bin))
        output.mkdir(parents=True)
        (output / "status.json").write_text(json.dumps({"state": "completed"}))
        (output / "summary.json").write_text(json.dumps({"state": "completed", "case": case,
            "objective": phase, "final_evaluations": 5,
            "reference_evaluations": 0 if runtime.reference_policy == "ADOPT" else 1,
            "reference_adoptions": int(runtime.reference_policy == "ADOPT")}))
    monkeypatch.setattr(scheduled, "run_workspace_experiment", workspace)
    invoke("prepare")
    assert not calls
    assert runtimes[0].prepared == [("bp_fe", batch / "bp_fe/input")]
    contract = batch / "bp_fe/task_contracts.json"
    before = contract.read_bytes(), contract.stat().st_mtime_ns
    store = tmp_path / "shared-reference-store"
    record = store / "records" / ("a" * 64 + ".json")
    extra = ["--reference-policy", policy, "--reference-store", str(store)]
    if policy != "FRESH":
        extra += ["--reference-record", str(record)]
    invoke(objective, *extra)
    assert len(calls) == 1
    assert calls[0][1:] == ("bp_fe", objective, batch / "bp_fe/input",
                           batch / "bp_fe" / objective / "codex", "/trusted/codex")
    assert runtimes[-1].options == {"share_gpu": True, "min_gpu_free_mb": 2048}
    assert runtimes[-1].cpus == 4 and runtimes[-1].memory_gb == 32
    assert runtimes[-1].reference_policy == policy
    assert runtimes[-1].reference_store == store
    assert runtimes[-1].reference_record == (record if policy != "FRESH" else None)
    assert before == (contract.read_bytes(), contract.stat().st_mtime_ns)
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event["summary"] == str(batch / "bp_fe" / objective / "codex/summary.json")


def test_bad_workspace_preflight_stops_before_runtime_or_physical_preparation(tmp_path, monkeypatch):
    invoke, batch, runtimes, preflight = scheduled_fixture(tmp_path, monkeypatch)
    bad = preflight_value()
    bad["qualification"] = "simple-api-probe"
    preflight.write_text(json.dumps(bad))
    monkeypatch.setattr(scheduled, "run_workspace_experiment", lambda *args: pytest.fail("Must not dispatch"))
    with pytest.raises(ValueError, match="workspace terminal"):
        invoke("prepare")
    assert not runtimes and not batch.exists()


@pytest.mark.parametrize("change", [{"final_evaluations": 4}, {"objective": "thermal"},
                                    {"case": "bp"}, {"state": "needs-attention"}])
def test_completed_workspace_summary_must_match_immutable_budget_and_endpoint(tmp_path, monkeypatch, change):
    invoke, _, _, _ = scheduled_fixture(tmp_path, monkeypatch)
    invoke("prepare")
    def workspace(runtime, case, phase, bundle, output, codex_bin):
        output.mkdir(parents=True)
        (output / "status.json").write_text(json.dumps({"state": "completed"}))
        (output / "summary.json").write_text(json.dumps({"state": "completed", "case": case,
            "objective": phase, "final_evaluations": 5, **change}))
    monkeypatch.setattr(scheduled, "run_workspace_experiment", workspace)
    with pytest.raises(ValueError, match="fixed task budget/endpoint"):
        invoke("timing")


def test_old_focused_contract_cannot_be_silently_reused(tmp_path, monkeypatch):
    invoke, batch, _, _ = scheduled_fixture(tmp_path, monkeypatch)
    invoke("prepare")
    path = batch / "bp_fe/task_contracts.json"
    old = json.loads(path.read_text())
    old["tasks"]["timing"]["adapter"] = "focused-candidate-program-codex-v1"
    path.write_text(json.dumps(old))
    before = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(scheduled, "run_workspace_experiment", lambda *args: pytest.fail("Must not dispatch"))
    with pytest.raises(ValueError, match="differs"):
        invoke("timing")
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


def test_workspace_attention_state_is_not_reported_as_completed(tmp_path, monkeypatch):
    invoke, _, _, _ = scheduled_fixture(tmp_path, monkeypatch)
    invoke("prepare")
    def workspace(runtime, case, phase, bundle, output, codex_bin):
        output.mkdir(parents=True)
        (output / "status.json").write_text(json.dumps({"state": "needs-attention"}))
        # No summary is published on a failed driver: inspect status first.
    monkeypatch.setattr(scheduled, "run_workspace_experiment", workspace)
    with pytest.raises(RuntimeError, match="needs inspection"):
        invoke("timing")
