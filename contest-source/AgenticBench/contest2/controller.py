"""Host-owned task-2 boundary: function capsules, budget and native matrices."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import sqlite3
import urllib.request

from grt_bench.common import digest, read, sha, write
from local_bench.matrix import LocalEvaluationMatrix
from local_bench.model import load_task
from local_bench.seeds import cell_contract
from local_bench.scoped_reference import admit_reference
from local_tasks.tp_gnn_local_task_definitions import (
    BINARY_TASK, DECODER_TASK, materialize_binary_loss, materialize_loss_decoder,
    validate_program, PACKAGES,
)

from .model_gateway import validate_request
from .halo_policy import validate_matrix_halos
from .sampling import smoke_matrix, validate_public_matrix


TASK_IDS = {"binary_loss": BINARY_TASK, "loss_decoder": DECODER_TASK}
REQUEST = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")


def _retained_diagnostics(output):
    """Use the host's saved excerpts only after its native run was retired."""
    if (output / "run").exists():
        return []
    try:
        cached = output / "retained_artifacts/diagnostics.json"
        receipt = read(output / "storage-retention.json")
        if cached.is_symlink() or sha(cached) != receipt["diagnostics_sha256"]:
            return []
        result = []
        for row in read(cached):
            source = Path(row["source"])
            if source.is_absolute() or ".." in source.parts:
                continue
            if source.parts and source.parts[0] == "run":
                source = Path("retained_artifacts/algorithm") / source.relative_to("run")
            if not (output / source).is_file():
                source = source.with_name(source.name + ".gz")
            if (output / source).is_file():
                result.append({"source": source.as_posix(), "excerpt": row["excerpt"]})
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return []  # Missing maintenance evidence must not hide existing feedback.


class Task2Controller:
    def __init__(self, *, baseline, matrix_path, source_root, work_root, ledger,
                 transport, gateway_url="", gateway_token="", codex_binary="", retain_physical_files=False):
        if baseline not in TASK_IDS or ledger.task_id != baseline:
            raise ValueError("Unknown or mismatched GNN baseline")
        self.baseline = baseline
        self.evaluation_mode = ledger.evaluation_mode
        self.task_id = TASK_IDS[baseline]
        self.source_root = Path(source_root).resolve(strict=True)
        self.task_dir = Path(__file__).resolve().parents[1] / "local_tasks" / self.task_id
        self.spec = load_task(self.task_dir)
        if self.spec.stop_stage != "opt" or self.spec.allowed_source_paths or self.spec.allowed_config_paths:
            raise ValueError("Contest GNN task must end at OPT and expose only a function capsule")
        self.matrix = LocalEvaluationMatrix(matrix_path, self.spec)
        validate_matrix_halos(self.matrix)
        validate_public_matrix(self.matrix, baseline, "train")
        from .scoring import validate_scoring_references
        validate_scoring_references(self.matrix, baseline)
        if ([seed["partition"] for seed in self.matrix.seeds] != [1001, 1002, 1003] or
                [seed["placement"] for seed in self.matrix.seeds] != [1001, 1002, 1003]):
            raise ValueError("Official training case and seed matrix differs")
        self.root = Path(work_root).resolve()
        if self.root.is_symlink():
            raise ValueError("Contest work root must be a real directory")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.evaluation_mode == "smoke":
            self.matrix = smoke_matrix(self.matrix, baseline, self.root / "control/smoke-matrix.json")
        self.candidates = self.root / "candidates"
        self.evaluations = self.root / "evaluations"
        if self.candidates.is_symlink() or self.evaluations.is_symlink():
            raise ValueError("Protected candidate and evaluation stores must be real directories")
        self.candidates.mkdir(exist_ok=True)
        self.evaluations.mkdir(exist_ok=True)
        self.ledger = ledger
        self.transport = transport
        if type(retain_physical_files) is not bool:
            raise ValueError("retain_physical_files must be a boolean")
        self.retain_physical_files = retain_physical_files
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token
        self.codex_binary = codex_binary
        self.reading_directory = self.root / "agent-evaluations" / self.baseline
        self.reading_directory.mkdir(parents=True, exist_ok=True)
        self.container_name = None
        self.codex_directory = self.root / "control/codex" / self.baseline
        self.codex_directory.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gnn-matrix")
        self._futures = {}
        self._lock = threading.RLock()
        initial = (self.task_dir / "initial_program.py").read_text()
        self.initial_source = validate_program(initial, self.task_id)
        self.baseline_candidate = self._candidate_id(self.initial_source)
        capsule = (materialize_binary_loss(self.initial_source) if baseline == "binary_loss"
                   else materialize_loss_decoder(self.initial_source))
        for cell in self.matrix.cells:
            cell_contract(self.spec, capsule, self.source_root, cell["baseline_run"],
                          cell["reference_run"], cell["base_config"], cell["seeds"])
            admit_reference(self.spec, capsule, self.source_root, cell["baseline_run"],
                            cell["reference_run"], runtime_image=getattr(transport, "image", None))

    def close(self):
        self._pool.shutdown(wait=True)

    def _candidate_id(self, source):
        value = {"task": self.task_id, "matrix": self.matrix.identity, "source": source}
        if self.evaluation_mode == "smoke":
            value["evaluation_mode"] = "smoke"
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def dispatch(self, method, params):
        if not isinstance(params, dict):
            raise ValueError("Invalid Agent request")
        if method == "initial" and not params:
            return {"task": self.baseline, "baseline_candidate": self.baseline_candidate,
                    "evaluation_mode": self.evaluation_mode,
                    "files": {"guide": "/workspace/README.md", "source": "/opt/contest/source", "agent": "/workspace/agent",
                              "framework": "/opt/contest/AgenticBench",
                              "evaluations": "/workspace/evaluations"},
                    "evaluation_cells": [{"case": row["case"], "seed": row["seed"]}
                                         for row in self.matrix.cells]}
        if method == "status" and not params:
            return self.status()
        if method == "submit_evaluation" and set(params) == {"source", "request_id"}:
            return self.submit_evaluation(params["source"], params["request_id"])
        if method == "evaluation_status" and set(params) == {"request_id"}:
            return self.evaluation_status(params["request_id"])
        if method == "model_responses_create" and set(params) == {"payload"}:
            return self.model_responses_create(params["payload"])
        if method == "codex_turn" and set(params) == {"prompt", "request_id"}:
            return self.codex_turn(params["prompt"], params["request_id"])
        raise ValueError("Unknown or malformed Agent request")

    def status(self):
        state = self.ledger.snapshot()
        history = []
        for row in state["evaluations"]:
            baseline = (row["candidate_id"].split("-", 2)[1] if row["status"] == "rejected"
                        else self._candidate_program(row["candidate_id"])[0])
            if baseline == self.baseline:
                history.append({**{k: row[k] for k in
                    ("request_id", "candidate_id", "status", "reason")}, "baseline": baseline})
        return {"task": self.baseline,
                "evaluation_mode": self.evaluation_mode,
                "evaluations_remaining": state["evaluations_remaining"],
                "model_cost_available_rmb": state["model_cost_available_rmb"],
                "model_dispatch_allowed": state["model_dispatch_allowed"],
                "evaluation_dispatch_allowed": state["evaluation_dispatch_allowed"],
                "model_stop_reasons": state["model_stop_reasons"],
                "unresolved_model_calls": state["unresolved_model_calls"],
                "codex_available": bool(self.codex_binary),
                "dispatch_allowed": state["dispatch_allowed"],
                "history": history}

    def _candidate_program(self, candidate_id):
        capsule = read(self.candidates / candidate_id / "candidate.json")
        files = capsule.get("candidate_files", {})
        for baseline, task_id in TASK_IDS.items():
            filename = PACKAGES[task_id][0]
            if set(files) == {filename}:
                source = validate_program(files[filename]["source"], task_id)
                if baseline == self.baseline and self._candidate_id(source) != candidate_id:
                    raise ValueError("Candidate source differs from its frozen identity")
                return baseline, source
        raise ValueError("Unknown candidate baseline")

    def submit_evaluation(self, source, request_id):
        if not isinstance(request_id, str) or not REQUEST.fullmatch(request_id):
            raise ValueError("Invalid evaluation request ID")
        payload = digest({"submission": source})
        with self._lock:
            previous = next((r for r in self.ledger.snapshot()["evaluations"]
                             if r["request_id"] == request_id), None)
            if previous and previous["status"] == "rejected":
                if previous["payload_digest"] != payload:
                    raise ValueError("Request ID belongs to another rejected submission")
                return self.evaluation_status(request_id)
            if not previous and not self.ledger.snapshot()["evaluation_dispatch_allowed"]:
                raise ValueError("Contest budget is closed")
            try:
                return self._submit_evaluation(source, request_id)
            except (ValueError, OSError, RuntimeError) as error:
                # An accepted request is polled, never charged a second time.
                if any(r["request_id"] == request_id for r in self.ledger.snapshot()["evaluations"]):
                    raise
                self.ledger.reject_submission(request_id, payload,
                    type(error).__name__ + ": " + str(error), baseline=self.baseline)
                return self.evaluation_status(request_id)

    def _submit_evaluation(self, source, request_id):
        if not isinstance(request_id, str) or not REQUEST.fullmatch(request_id):
            raise ValueError("Invalid evaluation request ID")
        source = validate_program(source, self.task_id)
        candidate_id = self._candidate_id(source)
        with self._lock:
            previous = [row for row in self.ledger.snapshot()["evaluations"]
                        if row["request_id"] == request_id]
            if previous:
                if previous[0]["candidate_id"] != candidate_id:
                    raise ValueError("Request ID already belongs to another candidate")
                return self.evaluation_status(request_id)
            if any(row["status"] == "accepted" for row in self.ledger.snapshot()["evaluations"]):
                raise ValueError("The previous accepted matrix must finish first")
            self.matrix.assert_unchanged()
            folder = self.candidates / candidate_id
            capsule = (materialize_binary_loss(source) if self.baseline == "binary_loss"
                       else materialize_loss_decoder(source))
            if folder.exists():
                if read(folder / "candidate.json") != capsule:
                    raise ValueError("Candidate content differs from its frozen identity")
            else:
                folder.mkdir()
                write(folder / "candidate.json", capsule)
            payload = digest({"task": self.task_id, "matrix": self.matrix.identity,
                              "candidate": candidate_id})
            self.ledger.accept_matrix(request_id, candidate_id, payload, baseline=self.baseline)
            output = self.evaluations / request_id
            from .reconciliation import record_owner
            record_owner(self.evaluations / (request_id + "-owner.json"))
            self._futures[request_id] = self._pool.submit(
                self._evaluate, request_id, candidate_id, folder, output)
            return {"request_id": request_id, "candidate_id": candidate_id, "status": "accepted"}

    def _evaluate(self, request_id, candidate_id, folder, output):
        try:
            from .storage import TrainingStorageTransport
            result = self.matrix.evaluate(folder, self.source_root, output,
                transport=TrainingStorageTransport(self.transport, self.evaluations,
                    retain_physical_files=self.retain_physical_files),
                workers=getattr(self.transport, "workers", 1), timeout_seconds=None)
            from .scoring import complete_score
            if not complete_score(result.metrics, self.matrix):
                self.ledger.finish_evaluation(request_id, "failed", "Incomplete native matrix")
            else:
                self.ledger.finish_evaluation(request_id, "succeeded")
        except Exception as error:
            from grt_bench.runtime import UncertainExecution
            if not isinstance(error, UncertainExecution):
                self.ledger.finish_evaluation(request_id, "failed", type(error).__name__ + ": " + str(error))
            raise

    def evaluation_status(self, request_id):
        if not isinstance(request_id, str) or not REQUEST.fullmatch(request_id):
            raise ValueError("Invalid evaluation request ID")
        rows = [row for row in self.ledger.snapshot()["evaluations"]
                if row["request_id"] == request_id]
        if not rows:
            raise KeyError("Unknown evaluation request")
        row = rows[0]
        if row["status"] == "rejected":
            if not row["candidate_id"].startswith("rejected-" + self.baseline + "-"):
                raise ValueError("Evaluation belongs to another GNN baseline")
            from .evaluation_files import publish
            return publish(self.root / "agent-evaluations" / self.baseline, request_id,
                {k: row[k] for k in ("request_id", "candidate_id", "status", "reason")}, [])
        baseline, source = self._candidate_program(row["candidate_id"])
        if baseline != self.baseline:
            raise ValueError("Evaluation belongs to another GNN baseline")
        status = row["status"]
        if status == "accepted":
            future = self._futures.get(request_id)
            if future is not None and not future.done():
                status = "running"
            else:
                try:
                    inspection = self._reconcile_evaluation(request_id, row)
                except (OSError, ValueError, sqlite3.Error) as error:
                    inspection = {"terminated": False,
                        "reason": "Inspection needs another poll: " + type(error).__name__ + ": " + str(error)}
                if not inspection["terminated"]:
                    return {"request_id": request_id, "candidate_id": row["candidate_id"],
                            "status": "reconciling", "reason": inspection["reason"],
                            "inspection": inspection}
                return self.evaluation_status(request_id)
        reply = {"request_id": request_id, "candidate_id": row["candidate_id"],
                 "status": status, "reason": row["reason"]}
        if status == "failed":
            reply["feedback"] = self._failure_feedback(request_id)
        if status == "succeeded":
            reply["source"] = source
            result = read(self.evaluations / request_id / "evaluation_result.json")
            from .scoring import complete_score
            if not complete_score(result["metrics"], self.matrix):
                raise ValueError("Success receipt differs from native matrix")
            matrix = result["metrics"]["matrix"]
            visible = ("final_global_wns_setup_ns", "final_global_tns_setup_ns",
                       "wns_deficit_improvement", "tns_deficit_improvement",
                       "area_balance_error", "macro_count_error", "macro_area_error",
                       "cut_net_count", "timing_weighted_cut", "tp_gnn_final_loss")
            reply["feedback"] = {"score": result["metrics"]["combined_score"],
                "valid": result["metrics"].get("valid", True),
                "cases": matrix["cases"],
                "cells": [{"case": cell["case"], "seed": cell["seed"],
                           "score": cell["score"],
                           "raw_score": cell.get("raw_score"), "valid": cell.get("valid", True),
                           "status": cell["status"], "reason": cell.get("error_message"),
                           "metrics": {key: cell["metrics"][key] for key in visible
                                       if key in cell["metrics"]}}
                          for cell in matrix["cells"]]}
        if status in {"succeeded", "failed"}:
            from .evaluation_files import publish
            output = self.evaluations / request_id
            trees = []
            for cell in self.matrix.cells:
                prefix = "cells/" + cell["case"] + "/" + str(cell["seed"])
                # Docker transcripts live under container/execution. Publish
                # that subtree only; request/lease control files stay private.
                for name, source in (("result", "result"),
                                     ("execution", "container/execution"),
                                     ("verification", "verification")):
                    trees.append((prefix + "/" + name, output / prefix / source))
            return publish(self.root / "agent-evaluations" / self.baseline, request_id, reply, trees)
        return reply

    def _reconcile_evaluation(self, request_id, row):
        from .reconciliation import inspect_execution
        queue = self.transport.queue
        output = self.evaluations / request_id
        jobs = [r for r in queue.rows() if json.loads(r["payload"]).get("kind") == "local-matrix"
                and output in Path(json.loads(r["payload"]).get("output", "/")).parents]
        names = {"grt-" + r["id"][:32] + suffix for r in jobs for suffix in ("-frontend", "-backend")}
        names.update("grt-local-matrix-" + digest(str(Path(json.loads(r["payload"])["output"]).parent / "result"))[:24]
                     for r in jobs)
        audit = inspect_execution(queue, jobs, names,
            self.evaluations / (request_id + "-owner.json"),
            self.evaluations / (request_id + "-reconciliation.json"))
        if audit["terminated"]:
            path = output / "evaluation_result.json"
            try:
                result = read(path) if path.exists() else {}
                if not isinstance(result, dict):
                    raise ValueError("Matrix summary is not an object")
                metrics = result.get("metrics", {})
                if not isinstance(metrics, dict):
                    raise ValueError("Matrix metrics are not an object")
                matrix = metrics.get("matrix", {})
                if not isinstance(matrix, dict):
                    raise ValueError("Matrix details are not an object")
            except (OSError, ValueError):
                metrics, matrix = {}, {}
            from .scoring import complete_score
            complete = complete_score(metrics, self.matrix)
            self.ledger.finish_evaluation(request_id, "succeeded" if complete else "failed",
                None if complete else "Execution ended without a complete verified matrix; original attempt retained, not replayed")
        return audit

    def _failure_feedback(self, request_id):
        path = self.evaluations / request_id / "evaluation_result.json"
        try:
            matrix = read(path).get("metrics", {}).get("matrix", {}) if path.is_file() else {}
            if not isinstance(matrix, dict):
                raise ValueError("Matrix details are not an object")
        except (OSError, ValueError, AttributeError):
            matrix = {}
        rows = matrix.get("cells")
        if (not isinstance(rows, list) or any(not isinstance(r, dict)
                or not isinstance(r.get("case"), str) or not isinstance(r.get("seed"), str) for r in rows)):
            from local_bench.failure_feedback import native_failure_diagnostics
            rows = []
            for cell in self.matrix.cells:
                folder = path.parent / "cells" / cell["case"] / cell["seed"]
                record = folder / "cell.json"
                row = {"case": cell["case"], "seed": cell["seed"], "status": "not_completed",
                       "error_message": "This cell did not publish a completed evaluation"}
                if record.is_file():
                    try:
                        value = read(record)
                        if not isinstance(value, dict):
                            raise ValueError("Cell summary is not an object")
                        row = value
                    except (OSError, ValueError) as error:
                        row.update(error_type=type(error).__name__, error_message="Unreadable cell summary: " + str(error))
                if not row.get("native_failure_diagnostics"):
                    row["native_failure_diagnostics"] = native_failure_diagnostics(folder / "result")
                rows.append(row)
        cells = []
        for row in rows:
            item = {k: row[k] for k in ("case", "seed", "status", "score", "error_type") if k in row}
            item["error_message"] = str(row.get("error_message", ""))[:4000]
            diagnostics = list(row.get("native_failure_diagnostics", []))
            output = path.parent / "cells" / row["case"] / row["seed"] / "result"
            for saved in _retained_diagnostics(output):
                if not any(r.get("excerpt") == saved["excerpt"] for r in diagnostics):
                    diagnostics.append(saved)
            item["diagnostics"] = [{"source": str(r.get("source", ""))[:400],
                                    "excerpt": str(r.get("excerpt", ""))[:1800]}
                                   for r in diagnostics[:2]]
            cells.append(item)
        return {"complete": False, "expected_cells": matrix.get("expected_cells", len(self.matrix.cells)),
                "succeeded_cells": matrix.get("succeeded_cells", sum(r.get("status") == "succeeded" for r in rows)), "cells": cells}

    def model_responses_create(self, payload):
        if not self.gateway_url or not self.gateway_token:
            raise ValueError("Metered model gateway is unavailable")
        if not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{4,5}/v1", self.gateway_url):
            raise ValueError("Model gateway must be a host loopback service")
        if not isinstance(payload, dict) or payload.get("stream", False):
            raise ValueError("Only non-streaming model requests are supported")
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        validate_request(raw)
        request = urllib.request.Request(self.gateway_url + "/responses", data=raw,
            headers={"Authorization": "Bearer " + self.gateway_token,
                     "Content-Type": "application/json"}, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=1800) as response:
            return json.load(response)

    from .runtime_recovery import recoverable_turn

    @recoverable_turn
    def codex_turn(self, prompt, request_id):
        if (not isinstance(request_id, str) or not REQUEST.fullmatch(request_id) or
                not isinstance(prompt, str) or not prompt.strip() or
                len(prompt.encode()) > 256 * 1024):
            raise ValueError("Invalid Codex turn")
        if not self.codex_binary or not self.container_name or not self.gateway_url:
            raise ValueError("Metered Codex adapter is unavailable")
        if not self.ledger.snapshot()["dispatch_allowed"]:
            raise ValueError("Contest budget is closed")
        home = self.codex_directory / "home"
        codex_home = home / ".codex"
        for directory in (home, codex_home):
            if directory.is_symlink():
                raise ValueError("Codex state must be a real directory")
            directory.mkdir(mode=0o700, exist_ok=True)
            directory.chmod(0o700)
        from .codex_runtime import run_workspace_turn
        with self._lock:
            state_path = self.codex_directory / "state.json"
            state = read(state_path) if state_path.exists() else {"thread_id": None, "turns": {}}
            previous = state["turns"].get(request_id)
            if previous is not None:
                if previous["prompt_digest"] != digest(prompt) or previous["status"] != "finished":
                    raise ValueError("Codex request differs or needs reconciliation")
                return previous["result"]
            if any(row["status"] == "pending" for row in state["turns"].values()):
                raise ValueError("Another Codex turn needs reconciliation")
            state["turns"][request_id] = {"prompt_digest": digest(prompt), "status": "pending",
                "ownership": "inherited-process-lock"}
            write(state_path, state, replace=state_path.exists())
            folder = self.codex_directory / request_id
            public = folder / "public"
            public.mkdir(parents=True, exist_ok=False)
            old_thread = state["thread_id"]
        environment = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home),
                       "CODEX_HOME": str(codex_home),
                       "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                       "CONTEST_GATEWAY_URL": self.gateway_url,
                       "CONTEST_GATEWAY_TOKEN": self.gateway_token,
                       "CONTEST_CODEX_BINARY": self.codex_binary,
            "CONTEST_TURN_LEASE_FD": str(self._turn_lease_fd)}
        try:
            result = run_workspace_turn(binary=str(Path(__file__).with_name("codex_shim.py")),
                control=public, output=folder / "agent", container=self.container_name,
                prompt=prompt, terminal_script=Path(__file__).with_name("container_terminal.py"),
                python=sys.executable, thread_id=old_thread, timeout_seconds=None,
                environment=environment, model="deepseek-flash")
        except Exception as error:
            result = {"success": False, "error_type": type(error).__name__,
                      "error": "Model adapter failed; inspect the retained turn log"}
        with self._lock:
            state = read(state_path)
            state["turns"][request_id] = {"prompt_digest": digest(prompt),
                                           "status": "finished", "result": result}
            if result["success"]:
                state["thread_id"] = result["thread_id"]
            write(state_path, state, replace=True)
        return result
