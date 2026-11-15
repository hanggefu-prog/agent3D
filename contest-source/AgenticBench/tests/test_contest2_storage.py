"""Training retention must preserve scores/feedback and require host termination."""
from pathlib import Path
import gzip
from types import SimpleNamespace

import pytest

from contest2 import storage
from contest2.controller import Task2Controller, TASK_IDS
from grt_bench.common import digest, read, sha, write
from grt_bench.queue import Queue
from grt_bench.runtime import UncertainExecution
from local_bench.failure_feedback import native_failure_diagnostics
from local_bench.model import EvaluationResult

TASK = TASK_IDS["binary_loss"]


def terminal_cell(root, *, output=None, queue=None, success=True, exit_code=0, finish=True, task_id=TASK):
    root = root.resolve()
    output = output or root / "evaluations/one/cells/bp_fe-ng45/1003/result"
    control = output.parent / "container"
    queue = queue or Queue(root / "queue.sqlite")
    queue.register_host("local", dict(cpu=4, memory_gb=24, storage_gb=40, gpus=[], build_slots=0))
    job = queue.submit({"kind": "local-matrix", "output": str(control)},
                      dict(cpu=1, memory_gb=1, storage_gb=1, gpu_count=0, build_slots=0))
    claim = queue.claim("local", kind="local-matrix", job_id=job)
    if finish:
        queue.finish(job, claim["token"], {"output": str(control)}, success)
    write(control / "request.json", {"task": task_id, "output": str(output)})
    write(control / "lease.json", {"job": job, "uncertain": False, "success": success})
    name = "grt-local-matrix-" + digest(str(output))[:24]
    write(control / "execution/terminal.json", {"container": name, "exit_code": exit_code})
    if exit_code:
        write(control / "execution/stop.json", {"container": name, "terminated": True})
    metrics = {"runs_successfully": int(success), "combined_score": 7.25,
               "final_global_wns_setup_ns": -0.2, "final_global_tns_setup_ns": -23.5}
    write(output / "evaluation_result.json", {"metrics": metrics, "artifacts": {}})
    write(output / "candidate_snapshot.json", {"candidate": "immutable"})
    write(output / "run/60_opt/stage_manifest.json", {"status": "success" if success else "failed"})
    write(output / "run/20_partition/partition_metrics.json", {"cut_net_count": 42})
    (output / "run/20_partition/partition_assignment.csv").write_text("inst_name,die\na,top\nb,bottom\n")
    (output / "run/60_opt/top.odb").write_bytes(b"physical" * 8192)
    (output / "run/60_opt/native.log").write_text("native command\nERROR: PDN disconnected\n")
    (output / "run/60_opt/alias.log").symlink_to("native.log")
    (output / "stage.log").write_text("RuntimeError: failed; see " + str(output / "run/60_opt/native.log") + "\n")
    for name in ("workspace", "runtime"):
        (output / name).mkdir()
        (output / name / "large.bin").write_bytes(b"x" * 4096)
    return SimpleNamespace(output=output, evaluations=root / "evaluations", queue=queue,
                           task_id=task_id, metrics=metrics, control=control, claim=claim)


def compact(cell):
    return storage.compact_cell(cell.output, evaluations=cell.evaluations,
                                queue=cell.queue, task_id=cell.task_id)


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("task_id", TASK_IDS.values())
def test_terminal_cells_keep_score_assignment_and_native_failure_evidence(tmp_path, success, task_id):
    cell = terminal_cell(tmp_path, success=success, task_id=task_id)
    receipt_hash = sha(cell.output / "evaluation_result.json")
    before = native_failure_diagnostics(cell.output)
    record = compact(cell)
    assert record["status"] == "complete"
    assert not any((cell.output / name).exists() for name in storage.HEAVY)
    assert sha(cell.output / "evaluation_result.json") == receipt_hash
    retained = cell.output / "retained_artifacts"
    assert (retained / "algorithm/20_partition/partition_assignment.csv").read_text().endswith("b,bottom\n")
    assert not (retained / "algorithm/60_opt/alias.log").is_symlink()
    assert read(retained / "diagnostics.json") == before
    after = native_failure_diagnostics(cell.output)
    assert [r["excerpt"] for r in before] == [r["excerpt"] for r in after]
    assert after[0]["source"] == "retained_artifacts/algorithm/60_opt/native.log"
    assert compact(cell) == record


@pytest.mark.parametrize("problem", ["live", "uncertain", "missing_terminal", "missing_stop",
                                      "wrong_container", "reference", "other_task", "heavy_link",
                                      "external_evidence", "outside_root"])
def test_unconfirmed_or_unowned_cells_are_preserved(tmp_path, problem):
    cell = terminal_cell(tmp_path, finish=problem != "live", exit_code=1 if problem == "missing_stop" else 0)
    if problem == "uncertain":
        value = read(cell.control / "lease.json")
        write(cell.control / "lease.json", {**value, "uncertain": True}, replace=True)
    elif problem == "missing_terminal":
        (cell.control / "execution/terminal.json").unlink()
    elif problem == "missing_stop":
        (cell.control / "execution/stop.json").unlink()
    elif problem == "wrong_container":
        write(cell.control / "execution/terminal.json", {"container": "other", "exit_code": 0}, replace=True)
    elif problem == "reference":
        value = read(cell.output / "evaluation_result.json")
        value["artifacts"]["matrix_reference"] = True
        write(cell.output / "evaluation_result.json", value, replace=True)
    elif problem == "other_task":
        cell.task_id = "other"
    elif problem == "heavy_link":
        (cell.output / "runtime").rename(tmp_path / "outside")
        (cell.output / "runtime").symlink_to(tmp_path / "outside", target_is_directory=True)
    elif problem == "external_evidence":
        (tmp_path / "outside.log").write_text("outside")
        (cell.output / "run/outside.log").symlink_to(tmp_path / "outside.log")
    elif problem == "outside_root":
        cell.evaluations = tmp_path / "different"
        cell.evaluations.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        compact(cell)
    assert (cell.output / "run/60_opt/top.odb").is_file()


def test_complete_logs_are_retained_losslessly(tmp_path, monkeypatch):
    import gzip
    cell = terminal_cell(tmp_path)
    original = (cell.output / "run/60_opt/native.log").read_bytes()
    monkeypatch.setattr(storage, "COMPRESS_BYTES", 32)
    record = compact(cell)
    logs = [(name, row) for name, row in record["retained_files"].items()
            if row["source"].endswith(".log")]
    assert len(logs) == 2 and not record["omitted_logs"]
    for name, row in logs:
        assert row["offset"] == 0
        assert gzip.decompress((cell.output / "retained_artifacts" / name).read_bytes()) == original


def test_metadata_budget_failure_leaves_original_files_and_score(tmp_path, monkeypatch):
    cell = terminal_cell(tmp_path)
    monkeypatch.setattr(storage, "METADATA_BUDGET", 1)
    original = sha(cell.output / "evaluation_result.json")
    with pytest.raises(ValueError, match="Metadata"):
        compact(cell)
    assert (cell.output / "run/60_opt/top.odb").exists()
    assert not list(cell.output.glob(".retention-*"))
    assert sha(cell.output / "evaluation_result.json") == original


def test_large_sdc_audits_are_losslessly_compressed_with_training_history_retained(tmp_path, monkeypatch):
    cell = terminal_cell(tmp_path)
    content = b'{"source_paths": [' + b'"repeated-instance-name",' * 30000 + b'"last"]}'
    audit = cell.output / "run/60_opt/top.sdc.audit.json"
    audit.write_bytes(content)
    history = cell.output / "run/20_partition/binary_training.csv"
    history.write_text("epoch,loss\n1,0.25\n")
    monkeypatch.setattr(storage, "METADATA_BUDGET", 10000)
    record = compact(cell)
    name = "algorithm/60_opt/top.sdc.audit.json.gz"
    archived = cell.output / "retained_artifacts" / name
    assert gzip.decompress(archived.read_bytes()) == content
    assert record["retained_files"][name]["encoding"] == "gzip"
    assert record["retained_files"][name]["source_bytes"] == len(content)
    assert record["retained_files"][name]["bytes"] < len(content) // 10
    assert (cell.output / "retained_artifacts/algorithm/20_partition/binary_training.csv").read_text() == "epoch,loss\n1,0.25\n"


def test_interrupted_removal_can_resume_from_verified_retained_evidence(tmp_path, monkeypatch):
    cell = terminal_cell(tmp_path)
    remove = storage.shutil.rmtree
    def interrupt(path):
        if path.name == "workspace":
            raise OSError("Interrupted cleanup")
        remove(path)
    monkeypatch.setattr(storage.shutil, "rmtree", interrupt)
    with pytest.raises(OSError, match="Interrupted"):
        compact(cell)
    assert read(cell.output / "storage-retention.json")["status"] == "prepared"
    assert not (cell.output / "run").exists()
    monkeypatch.setattr(storage.shutil, "rmtree", remove)
    assert compact(cell)["status"] == "complete"


def test_changed_retained_evidence_stops_further_deletion(tmp_path, monkeypatch):
    cell = terminal_cell(tmp_path)
    remove = storage.shutil.rmtree
    monkeypatch.setattr(storage.shutil, "rmtree", lambda _: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError):
        compact(cell)
    (cell.output / "retained_artifacts/algorithm/60_opt/native.log").write_text("changed")
    monkeypatch.setattr(storage.shutil, "rmtree", remove)
    with pytest.raises(ValueError, match="evidence changed"):
        compact(cell)
    assert (cell.output / "run/60_opt/top.odb").exists()


def test_failed_directory_sync_never_removes_original_evidence(tmp_path, monkeypatch):
    cell = terminal_cell(tmp_path)
    synchronize = storage._sync_directory
    def fail(path):
        if Path(path) == cell.output:
            raise OSError("Directory sync failed")
        synchronize(path)
    monkeypatch.setattr(storage, "_sync_directory", fail)
    with pytest.raises(OSError, match="sync failed"):
        compact(cell)
    assert (cell.output / "run/60_opt/top.odb").exists()
    monkeypatch.setattr(storage, "_sync_directory", synchronize)
    assert compact(cell)["status"] == "complete"


@pytest.mark.parametrize("outcome", ["success", "failure", "uncertain", "cleanup_error"])
def test_transport_preserves_native_result_or_exception(tmp_path, monkeypatch, outcome):
    cell = terminal_cell(tmp_path, success=outcome != "failure")
    transport = storage.TrainingStorageTransport(SimpleNamespace(queue=cell.queue), cell.evaluations)
    result = EvaluationResult(cell.metrics, {})
    error = UncertainExecution("unknown") if outcome == "uncertain" else RuntimeError("native error")
    if outcome == "cleanup_error":
        monkeypatch.setattr(storage, "METADATA_BUDGET", 1)
    def execute(*args, **kwargs):
        assert not kwargs["compact_artifacts"]
        if outcome in {"failure", "uncertain"}:
            raise error
        return result
    args = (execute, Path(TASK), tmp_path, tmp_path, cell.output)
    if outcome in {"failure", "uncertain"}:
        with pytest.raises(type(error)) as raised:
            transport._cell(*args, compact_artifacts=False)
        assert raised.value is error
    else:
        assert transport._cell(*args, compact_artifacts=False) is result
    assert (cell.output / "run").exists() == (outcome in {"uncertain", "cleanup_error"})
    assert (cell.output / "storage-maintenance-error.json").exists() == (outcome == "cleanup_error")


def test_controller_cleans_each_cell_before_next_and_aggregation_is_unchanged(tmp_path):
    from test_contest2_sampling import matrix_fixture, native_metrics
    from local_bench.matrix import LocalEvaluationMatrix
    path, spec = matrix_fixture(tmp_path / "inputs")
    matrix = LocalEvaluationMatrix(path, spec)
    evaluations = tmp_path.resolve() / "evaluations"
    queue = Queue(tmp_path / "queue.sqlite")
    outputs = []
    class Transport:
        def prepare_candidate(self, *args):
            return self.execute
        def execute(self, task, candidate, source, output, **options):
            assert not options["compact_artifacts"]
            assert all(not (previous / "run").exists() for previous in outputs)
            cell = terminal_cell(tmp_path, output=output, queue=queue)
            cell.metrics.update(native_metrics(output.parent.parent.name, 7.25))
            write(output / "evaluation_result.json", {"metrics": cell.metrics, "artifacts": {}}, replace=True)
            outputs.append(output)
            return EvaluationResult(cell.metrics, {})
    transport = Transport()
    transport.queue = queue
    c = Task2Controller.__new__(Task2Controller)
    c.transport, c.matrix = transport, matrix
    c.retain_physical_files = False
    c.source_root, c.evaluations = tmp_path, evaluations
    terminal = []
    c.ledger = SimpleNamespace(finish_evaluation=lambda *a: terminal.append(a))
    c._evaluate("one", "candidate", tmp_path, evaluations / "one")
    assert terminal == [("one", "succeeded")]
    result = read(evaluations / "one/evaluation_result.json")
    assert result["metrics"]["combined_score"] == pytest.approx(7.25)
    assert result["metrics"]["matrix"]["succeeded_cells"] == 7
    assert len(outputs) == 7 and all(not (p / "run").exists() for p in outputs)


@pytest.mark.parametrize('incomplete', [False, True])
def test_phased_retention_requires_both_containers_and_gpu_release(tmp_path, incomplete):
    cell = terminal_cell(tmp_path)
    name = 'grt-' + cell.claim['id'][:32]
    write(cell.control / 'execution/frontend/terminal.json', {'container': name+'-frontend', 'exit_code': 0})
    write(cell.control / 'execution/frontend/stop.json', {'container': name+'-frontend', 'terminated': True})
    write(cell.control / 'resource-release.json', {
        'evidence': {'container': name+'-frontend', 'terminated': True},
        'allocation': {'gpus': [], 'resources': {'gpu_count': 0}}})
    if incomplete:
        with pytest.raises(ValueError, match='regular file'):
            compact(cell)
        assert (cell.output / 'run').exists()
    else:
        write(cell.control / 'execution/backend/terminal.json', {'container': name+'-backend', 'exit_code': 0})
        assert compact(cell)['status'] == 'complete'


@pytest.mark.parametrize('retain', [False, True])
def test_storage_option_keeps_or_retires_physical_files(tmp_path, retain):
    cell = terminal_cell(tmp_path)
    before = (cell.output / 'evaluation_result.json').read_bytes()
    transport = SimpleNamespace(queue=cell.queue)
    wrapper = storage.TrainingStorageTransport(transport, cell.evaluations,
                                               retain_physical_files=retain)
    wrapper._maintain(cell.output, TASK)
    assert (cell.output / 'run/60_opt/top.odb').exists() is retain
    assert (cell.output / 'evaluation_result.json').read_bytes() == before
    if not retain:
        assert (cell.output / 'retained_artifacts/algorithm/60_opt/native.log').is_file()


def test_storage_option_rejects_ambiguous_string(tmp_path):
    with pytest.raises(ValueError, match='boolean'):
        storage.TrainingStorageTransport(None, tmp_path, retain_physical_files='false')
