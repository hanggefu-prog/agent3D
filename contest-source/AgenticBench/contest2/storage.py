"""Host-side retention for completed evaluation cells; scores remain immutable.

Reference qualification does not use this wrapper.
Only queue-confirmed terminal cells with a native termination receipt may retire
reproducible run/workspace/runtime directories. Unknown executions stay intact.
"""
from __future__ import annotations

import fcntl
from functools import partial
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import uuid

from grt_bench.common import digest, read, sha, write
from grt_bench.runtime import UncertainExecution

TASKS = {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}
HEAVY = ("run", "workspace", "runtime")
METADATA_BUDGET = 64 * 1024 * 1024
COMPRESS_BYTES = 256 * 1024
# Infrastructure, identity and interface failures need investigation/rejection.
UNQUALIFIED = re.compile(
    r"out of memory|MemoryError|No space left|disk quota|CUDA error|driver version|"
    r"permission denied|No such file or directory|ModuleNotFoundError|ImportError|cannot connect.*docker|"
    r"checksum|sha256|hash (?:mismatch|changed)|artifact.*(?:changed|differs)|source.*(?:changed|differs|mutat)|"
    r"signature.*(?:changed|differs)|input.*(?:changed|differs)|"
    r"unexpected keyword argument|missing .*required positional argument", re.I)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Storage evidence must be a regular file: " + path.name)
    return path


def _bound(output, evaluations, queue, task_id):
    """Use host-only control files, never a participant claim of completion."""
    output, evaluations = Path(output), Path(evaluations).resolve(strict=True)
    relative = output.relative_to(evaluations)
    if len(relative.parts) != 5 or relative.parts[1] != "cells" or relative.parts[-1] != "result":
        raise ValueError("Retention is restricted to owned evaluation cells")
    if output.resolve(strict=True) != output or any(p.is_symlink() for p in output.parents):
        raise ValueError("Retention output cannot traverse a symlink")
    control = output.parent / "container"
    if control.is_symlink():
        raise ValueError("Retention control directory cannot be a symlink")
    request = read(_regular(control / "request.json"))
    lease = read(_regular(control / "lease.json"))
    if (task_id not in TASKS or request.get("task") != task_id
            or request.get("output") != str(output) or lease.get("uncertain") is not False):
        raise ValueError("Retention task, output or execution ownership differs")
    jobs = [r for r in queue.rows() if r["id"] == lease["job"]]
    if (len(jobs) != 1 or jobs[0]["state"] not in {"succeeded", "failed"}
            or json.loads(jobs[0]["payload"]).get("output") != str(control)
            or json.loads(jobs[0]["payload"]).get("kind") != "local-matrix"):
        raise ValueError("Native queue has not confirmed terminal ownership")
    evidence = [control / "request.json", control / "lease.json"]
    if (control / "execution/frontend").exists():
        name = "grt-" + lease["job"][:32]
        phases = [("frontend", name + "-frontend")]
        release = control / "resource-release.json"
        if release.exists():
            released = read(_regular(release))
            if (released.get("evidence", {}).get("container") != name + "-frontend"
                    or released["evidence"].get("terminated") is not True
                    or released.get("allocation", {}).get("gpus") != []
                    or released["allocation"].get("resources", {}).get("gpu_count") != 0):
                raise ValueError("GPU release evidence differs")
            evidence.append(release)
            phases.append(("backend", name + "-backend"))
        elif jobs[0]["state"] != "failed":
            raise ValueError("Completed phased cell has no GPU release")
        terminals = [(control / "execution" / phase, container) for phase, container in phases]
    else:
        terminals = [(control / "execution", "grt-local-matrix-" + digest(str(output))[:24])]
    for folder, name in terminals:
        path = folder / "terminal.json"
        terminal = read(_regular(path))
        if terminal.get("container") != name or type(terminal.get("exit_code")) is not int:
            raise ValueError("Native termination receipt differs")
        evidence.append(path)
        if terminal["exit_code"] or folder.name == "frontend":
            stop = folder / "stop.json"
            value = read(_regular(stop))
            if value.get("container") != name or value.get("terminated") is not True:
                raise ValueError("Failed container termination is unconfirmed")
            evidence.append(stop)
    if (output / "evaluation_result.json").exists():
        result = read(_regular(output / "evaluation_result.json"))
        if result.get("artifacts", {}).get("matrix_reference"):
            raise ValueError("Reference results cannot use training retention")
        evidence.append(output / "evaluation_result.json")
    elif terminal["exit_code"] == 0:
        raise ValueError("Completed native cell has no evaluation receipt")
    for name in ("candidate_snapshot.json", "resolved_config.yaml"):
        if (output / name).exists():
            evidence.append(_regular(output / name))
    return {str(p.relative_to(output.parent)): sha(p) for p in evidence}


def _physical_failure(output, evaluations, queue, task_id):
    """Classify only a finished physical stage with matching native/host receipts.

    Missing measurements, verifier/identity errors, launch failures and unknown
    container state never become a physical penalty just because they raised.
    """
    from .scoring import PhysicalCellFailure
    from local_bench.model import EvaluationResult
    try:
        evidence = _bound(output, evaluations, queue, task_id)
        result = read(_regular(output / "evaluation_result.json"))
        metrics, artifacts = result["metrics"], result["artifacts"]
        if metrics.get("runs_successfully") != 0 or artifacts.get("evaluation_pending"):
            return None
        executions = artifacts.get("native_flow_execution", [])
        if not executions:
            return None
        last = executions[-1]
        if (last.get("invocation") != "stage" or last.get("process_started") is not True
                or type(last.get("returncode")) is not int or not last.get("completed_at_utc")):
            return None
        run_path = _regular(output / "run/run.json")
        run = read(run_path)
        stages = run.get("stages", {})
        failed = [(name, row) for name, row in stages.items()
                  if isinstance(row, dict) and row.get("status") == "failed"]
        reason = None
        if (last.get("status") == "failed" and 0 < last["returncode"] < 125
                and artifacts.get("error_type") == "RuntimeError"
                and artifacts.get("error_message") == "Open3DBench stage failed; see stage.log"
                and run.get("status") == "failed" and len(failed) == 1
                and failed[0][0] in {"partition", "placement", "pdn", "hbt", "opt"}
                and failed[0][1].get("started_at") and failed[0][1].get("completed_at")
                and failed[0][1].get("reason")):
            reason = failed[0][0] + ": " + failed[0][1]["reason"]
        elif (last.get("status") == "completed" and last["returncode"] == 0
                and run.get("status") == "success" and not failed
                and artifacts.get("error_type") == "ValueError"):
            message = artifacts.get("error_message")
            # These errors occur only after artifact validation and independent
            # recomputation; mismatched native measurements remain unscored.
            if message in {"Native assignment does not cover the immutable instance inventory exactly",
                           "Native initial partition fails the unchanged task constraints"}:
                reason = message
            elif message == "OPT timing view differs from the official endpoint":
                path = _regular(output / "run/60_opt/metrics.json")
                measured = read(path)
                validity = measured.get("timing_validity", {})
                counts = [validity.get(name) for name in ("unclocked_or_unconstrained_endpoints",
                          "combinational_loops", "missing_liberty_cells")]
                if (validity.get("global_wns_role") == "pre-cts-placement-estimated"
                        and validity.get("authoritative_timing_view") == "continuous-passive-hbt"
                        and validity.get("signoff_ready") is False
                        and measured.get("parasitics", {}).get("assembly") ==
                        "continuous-passive-hbt-placement-estimated-rc"
                        and validity.get("graph_completeness") in {"passed", "failed"}
                        and all(type(value) is int and value >= 0 for value in counts)
                        and (validity["graph_completeness"] == "failed" or any(counts))):
                    reason = "Physical timing graph failed: " + json.dumps(validity, sort_keys=True)
                    evidence["result/run/60_opt/metrics.json"] = sha(path)
        if reason is None:
            return None
        from local_bench.failure_feedback import native_failure_diagnostics
        diagnostics = native_failure_diagnostics(output)
        messages = [reason] + [item["excerpt"] for item in diagnostics]
        # Native stage logs may carry a cause hidden behind the wrapper error.
        for name in ("stage.log", "backend-stage.log"):
            path = output / name
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    messages.append(stream.read().decode("utf-8", errors="replace"))
        if any(UNQUALIFIED.search(message) for message in messages):
            return None
        evidence["result/run/run.json"] = sha(run_path)
        return PhysicalCellFailure(reason, EvaluationResult(metrics=metrics, artifacts=artifacts), evidence)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _inventory(output):
    records, paths = {}, {}
    device = output.stat().st_dev
    for name in HEAVY:
        root = output / name
        if root.is_symlink():
            raise ValueError("Heavy directory is an external link: " + name)
        if not root.exists():
            continue
        if not root.is_dir():
            raise ValueError("Heavy directory became a file: " + name)
        for directory, folders, files in os.walk(root, followlinks=False):
            if Path(directory).stat().st_dev != device:
                raise ValueError("Retention cannot cross a filesystem")
            for leaf in sorted(files + [n for n in folders if (Path(directory) / n).is_symlink()]):
                path = Path(directory) / leaf
                relative = path.relative_to(output).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    records[relative] = {"symlink": os.readlink(path)}
                elif stat.S_ISREG(info.st_mode):
                    records[relative] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                        "inode": info.st_ino, "device": info.st_dev,
                        "links": info.st_nlink, "blocks": info.st_blocks * 512}
                else:
                    raise ValueError("Unexpected special file in evaluation output")
                paths[relative] = path
    return records, paths


def _select(relative):
    path = Path(relative)
    if path.parts[0] != "run":
        return None
    if path.suffix in {".log", ".rpt"}:
        return "log"
    if path.suffix in {".json", ".yaml", ".yml"}:
        return "metadata"
    if path.name in {"partition_assignment.csv", "hbt.csv", "io_plan.csv",
                     "binary_training.csv", "training_log.csv"}:
        return "metadata"
    return None


def _retain(output, paths, stage):
    kept, omitted = {}, []
    remaining = METADATA_BUDGET
    from local_bench.failure_feedback import native_failure_diagnostics
    diagnostics = native_failure_diagnostics(output)
    for relative in sorted(paths):
        kind = _select(relative)
        if kind is None:
            continue
        source = paths[relative]
        resolved = source.resolve(strict=True)
        if output not in resolved.parents or not resolved.is_file():
            raise ValueError("Retained evidence points outside this cell")
        size = resolved.stat().st_size
        target = stage / "algorithm" / Path(relative).relative_to("run")
        compressed = size > COMPRESS_BYTES
        if compressed:
            target = target.with_name(target.name + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("rb") as src, target.open("xb") as dst:
            if compressed:
                with gzip.GzipFile(filename="", fileobj=dst, mode="wb", mtime=0) as archive:
                    shutil.copyfileobj(src, archive, length=1024 * 1024)
            else:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        archived_bytes = target.stat().st_size
        if kind == "metadata":
            if archived_bytes > remaining:
                raise ValueError("Metadata exceeds retention allowance; preserve original run")
            remaining -= archived_bytes
        kept[target.relative_to(stage).as_posix()] = {"bytes": archived_bytes, "offset": 0,
            "encoding": "gzip" if compressed else "identity", "source_bytes": size,
            "source": relative, "source_sha256": sha(resolved), "sha256": sha(target)}
    write(stage / "diagnostics.json", diagnostics)
    return kept, omitted


def _verify_retained(output, record):
    for name, row in record["retained_files"].items():
        path = _regular(output / "retained_artifacts" / name)
        if sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Retained evidence changed before cleanup")
    if sha(_regular(output / "retained_artifacts/diagnostics.json")) != record["diagnostics_sha256"]:
        raise ValueError("Retained diagnostics changed")


def _same_file_identity(before, after):
    # Allocated blocks can settle after process exit without a content change.
    # Keep this statistic in receipts, but do not use it as file identity.
    return before is not None and (
        {key: value for key, value in before.items() if key != "blocks"}
        == {key: value for key, value in after.items() if key != "blocks"})


def compact_cell(output, *, evaluations, queue, task_id):
    """Durably retain evidence before deleting only this cell's bulk outputs."""
    output = Path(output)
    binding = _bound(output, evaluations, queue, task_id)
    lock = output / ".storage.lock"
    if lock.is_symlink():
        raise ValueError("Retention lock cannot be a symlink")
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        receipt = output / "storage-retention.json"
        if receipt.exists() or receipt.is_symlink():
            record = read(_regular(receipt))
            if record.get("binding") != binding or record.get("task") != task_id:
                raise ValueError("Retention request identities changed")
            _verify_retained(output, record)
            current, _ = _inventory(output)
            if any(not _same_file_identity(record["removed_files"].get(k), v) for k, v in current.items()):
                raise ValueError("Evaluation output changed after retention was prepared")
            if record["status"] == "complete":
                if current:
                    raise ValueError("Retired output unexpectedly reappeared")
                return record
        else:
            records, paths = _inventory(output)
            retained = output / "retained_artifacts"
            if retained.exists() or retained.is_symlink():
                raise ValueError("Existing retained evidence requires explicit inspection")
            stage = output / (".retention-" + uuid.uuid4().hex)
            stage.mkdir()
            try:
                kept, omitted = _retain(output, paths, stage)
                record = {"schema_version": 1, "mode": "compact-training", "status": "prepared",
                    "task": task_id, "binding": binding, "removed_files": records,
                    "retained_files": kept, "omitted_logs": omitted,
                    "diagnostics_sha256": sha(stage / "diagnostics.json"),
                    "logical_bytes_removed": sum(r.get("bytes", 0) for r in records.values()),
                    "retained_bytes": sum(r["bytes"] for r in kept.values()),
                    "full_native_revalidation": "requires a new physical run"}
                for directory, _, _ in os.walk(stage, topdown=False):
                    _sync_directory(directory)
                stage.rename(retained)
                write(receipt, record)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            _verify_retained(output, record)
        # Recheck host completion and file identity immediately before removal.
        current, _ = _inventory(output)
        if any(not _same_file_identity(record["removed_files"].get(k), v) for k, v in current.items()):
            raise ValueError("Evaluation output changed after retention was prepared")
        if _bound(output, evaluations, queue, task_id) != binding:
            raise ValueError("Native execution evidence changed")
        _sync_directory(output)  # Also persist a journal resumed after fsync failure.
        for name in HEAVY:
            target = output / name
            if target.is_symlink():
                raise ValueError("Heavy directory became a symlink")
            if target.exists():
                shutil.rmtree(target)
        record = {**record, "status": "complete"}
        write(receipt, record, replace=True)
        _sync_directory(output)
        return record


class TrainingStorageTransport:
    """Keep storage maintenance outside the native evaluator and its lease."""
    def __init__(self, transport, evaluations, *, retain_physical_files=False):
        if type(retain_physical_files) is not bool:
            raise ValueError("retain_physical_files must be a boolean")
        self.transport, self.evaluations = transport, Path(evaluations).resolve()
        self.retain_physical_files = retain_physical_files

    def prepare_candidate(self, task, candidate, source, output):
        execute = self.transport.prepare_candidate(task, candidate, source, output)
        return partial(self._cell, execute)

    def _cell(self, execute, task, candidate, source, output, **options):
        # The frozen evaluator must finish and publish its result before cleanup.
        if options.get("compact_artifacts"):
            raise ValueError("Training retention must run after native termination")
        try:
            result = execute(task, candidate, source, output, **options)
        except UncertainExecution:
            raise
        except Exception:
            failure = _physical_failure(Path(output), self.evaluations,
                                        self.transport.queue, Path(task).name)
            self._maintain(Path(output), Path(task).name)
            if failure is not None:
                raise failure
            raise
        self._maintain(Path(output), Path(task).name)
        return result

    def _maintain(self, output, task_id):
        if self.retain_physical_files:
            return
        try:
            compact_cell(output, evaluations=self.evaluations, queue=self.transport.queue, task_id=task_id)
        except Exception as error:
            # A cleanup problem never changes an already measured score or failure.
            if (self.evaluations in output.parents and output.resolve() == output
                    and output.is_dir() and not output.is_symlink()):
                try:
                    write(output / "storage-maintenance-error.json",
                          {"error_type": type(error).__name__, "error": str(error),
                           "evaluation_unchanged": True}, replace=True)
                except OSError:
                    pass
            print("Training storage retained for inspection: " + type(error).__name__ + ": " + str(error), file=sys.stderr)
