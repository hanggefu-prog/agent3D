"""Local matrix transport using the global scheduler and installed image builder.

No model credentials enter containers. Builds reserve CPU/build slots only;
HBT and routing cells reserve no GPU. Each lease is returned after termination,
never after an unconfirmed disconnect. The caller registers host capacity.
"""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import json
import os
import threading
import time

from grt_bench.common import digest, read, sha, write
from grt_bench.queue import Queue
from grt_bench.runtime import docker_command, stop_and_confirm, UncertainExecution
from contest2.execution_policy import run_container, admission
from .model import EvaluationResult, ROUTING_SOURCE_TASK_KINDS, load_task


class DockerMatrixTransport:
    def __init__(self, *, queue, host, image, compiler_cache=None, admission_timeout=600):
        self.queue = Queue(queue)
        self.host, self.image = host, image
        self.compiler_cache = Path(compiler_cache) if compiler_cache else None
        self.admission_timeout = admission_timeout
        if admission_timeout is not None and (type(admission_timeout) is not int or admission_timeout < 1):
            raise ValueError("admission_timeout must be positive")
        # Do not create a second queue silently or change a live host capacity.
        with self.queue.transaction() as db:
            if not db.execute("SELECT 1 FROM hosts WHERE id=?", (host,)).fetchone():
                raise ValueError("Register this host in the shared resource queue first")

    @property
    def workers(self):
        # GPU slots are managed independently: while a cell runs CPU backend,
        # another admitted cell may use its released GPU.
        with self.queue.transaction() as db:
            row = db.execute("SELECT capacity FROM hosts WHERE id=?", (self.host,)).fetchone()
        capacity = json.loads(row[0])
        return max(1, min(12, capacity["cpu"] // 4, capacity["memory_gb"] // 24,
                          capacity["storage_gb"] // 16))

    @contextmanager
    def lease(self, folder, resources):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"kind": "local-matrix", "output": str(folder.resolve())}
        job = self.queue.submit(payload, resources)
        claim, begin = None, time.monotonic()
        while claim is None:
            claim = self.queue.claim(self.host, kind="local-matrix", job_id=job)
            if claim is None:
                if self.admission_timeout is not None and time.monotonic() - begin >= self.admission_timeout:
                    self.queue.cancel(job)
                    raise TimeoutError("Local matrix resource admission timed out")
                time.sleep(1)
        done, errors = threading.Event(), []
        def heartbeat():
            while not done.wait(15):
                try:
                    self.queue.heartbeat(job, claim["token"])
                except Exception as error:
                    errors.append(error)
                    return
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        success = uncertain = False
        try:
            admission(folder, resources, claim["gpus"])
            yield claim
            if errors:
                raise errors[0]
            success = True
        except UncertainExecution:
            uncertain = True
            raise
        finally:
            done.set()
            thread.join()
            if not uncertain:
                self.queue.finish(job, claim["token"], {"output": str(folder), "success": success}, success)
            write(folder / "lease.json", {"job": job, "resources": resources, "gpus": claim["gpus"],
                                         "success": success, "uncertain": uncertain})

    def prepare_candidate(self, task, candidate, source, output):
        """Called exactly once per matrix, including native reference qualification."""
        spec = load_task(task)
        image, receipt = self.image, None
        if spec.kind in ROUTING_SOURCE_TASK_KINDS:
            from scripts.build_routing_candidate_image import build_candidate_image
            resources = dict(cpu=16, memory_gb=48, storage_gb=45, gpu_count=0, build_slots=1)
            with self.lease(Path(output).parent / "build-admission", resources):
                runtime = build_candidate_image(task, candidate, source, image, output,
                    threads=8, compiler_cache_dir=self.compiler_cache)
                image, receipt = runtime["image"], Path(runtime["receipt"])
        return partial(self.evaluate_cell, image=image, runtime_receipt=receipt)

    def evaluate_cell(self, task, candidate, source, output, *, image, runtime_receipt=None, **options):
        spec = load_task(task)
        routing = spec.kind in ROUTING_SOURCE_TASK_KINDS
        resources = dict(cpu=16 if routing else 4, memory_gb=48 if routing else 24,
                         storage_gb=45 if routing else 16,
                         gpu_count=int(spec.start_stage in {"partition", "placement"}), build_slots=0)
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        run = output.parent / "container"
        # Preserve paths in receipts; all immutable inputs are mounted read-only.
        framework = Path(__file__).resolve().parents[1]
        roots = [Path(source).resolve(), Path(candidate).resolve(), Path(options["baseline_run"]).resolve(),
                 Path(options["base_config"]).resolve()]
        if options["reference_run"] is not None:
            roots.append(Path(options["reference_run"]).resolve().parent)
        if runtime_receipt:
            roots.append(Path(runtime_receipt).resolve())
        # Avoid duplicate mount targets, including nested references to source.
        roots = sorted(set(roots), key=lambda p: len(p.parts))
        roots = [p for i, p in enumerate(roots) if not any(parent in p.parents for parent in roots[:i])]
        mounts = [(p, str(p), True) for p in roots]
        mounts += [(source, "/workspace", True), (framework, "/trusted", True), (output, str(output), False)]
        request = {"task": spec.root.name, "candidate": str(candidate), "source": str(source),
                   "output": str(output), "runtime_receipt": str(runtime_receipt) if runtime_receipt else None,
                   "options": {k: str(v) if isinstance(v, Path) else v for k, v in options.items()}}
        write(run / "request.json", request)
        with self.lease(run, resources) as claim:
            if spec.task_id in {"tp_gnn_post_placement_timing", "tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}:
                self.run_phased(request, run, image, resources, claim, mounts)
            else:
                name = "grt-local-matrix-" + digest(str(output))[:24]
                command = docker_command(image, name, resources, claim["gpus"],
                    mounts + [(run / "request.json", "/request.json", True)],
                    ["-m", "contest2.native_entry", "--cell", "/request.json"])
                run_container(command, run / "execution", name, timeout=None)
            value = read(output / "evaluation_result.json")
            if value.get("metrics", {}).get("runs_successfully") != 1:
                raise ValueError(value.get("artifacts", {}).get("error_message", "Native cell failed"))
        return EvaluationResult(value["metrics"], value["artifacts"])

    def run_phased(self, request, run, image, resources, claim, mounts):
        """Release the GPU only after the frontend container is proven stopped."""
        from .phased import CONTINUATION
        started = time.monotonic()
        name = "grt-" + claim["id"][:32]
        frontend = {**request, "options": {**request["options"], "execution_phase": "frontend"}}
        write(run / "frontend-request.json", frontend)
        command = docker_command(image, name + "-frontend", resources, claim["gpus"],
            mounts + [(run / "frontend-request.json", "/request.json", True)],
            ["-m", "contest2.native_entry", "--cell", "/request.json"])
        run_container(command, run / "execution/frontend", name + "-frontend",
                      timeout=None)
        stop_and_confirm(name + "-frontend", run / "execution/frontend")
        output = Path(request["output"])
        continuation = output / CONTINUATION
        if not continuation.is_file():
            error = read(output / "evaluation_result.json").get("artifacts", {}).get("error_message")
            raise ValueError(error or "Frontend did not publish its continuation")
        state = read(continuation)
        checkpoint = output / "run/30_placement/checkpoint.json"
        checkpoint_sha = sha(checkpoint)
        if state["inventory"].get(str(checkpoint)) != checkpoint_sha:
            raise ValueError("Frontend continuation/checkpoint mismatch")
        sealed_sha = sha(continuation)
        backend_resources = {**resources, "gpu_count": 0}
        evidence = {"container": name + "-frontend", "terminated": True,
                    "checkpoint_sha256": checkpoint_sha}
        allocation = self.queue.release(claim["id"], claim["token"], backend_resources, evidence)
        if allocation["resources"] != backend_resources or allocation["gpus"]:
            raise ValueError("Queue did not confirm local CPU allocation")
        write(run / "resource-release.json", {"allocation": allocation, "evidence": evidence,
            "continuation_sha256": sealed_sha, "elapsed_seconds": time.monotonic() - started})
        admission(run, backend_resources, [])
        backend = {**request, "options": {**request["options"], "execution_phase": "backend",
                                          "continuation_sha256": sealed_sha}}
        write(run / "backend-request.json", backend)
        command = docker_command(image, name + "-backend", backend_resources, [],
            mounts + [(run / "backend-request.json", "/request.json", True)],
            ["-m", "contest2.native_entry", "--cell", "/request.json"])
        run_container(command, run / "execution/backend", name + "-backend", timeout=None)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=Path, required=True)
    args = parser.parse_args()
    request = read(args.cell)
    os.environ["PLACE3D_DREAMPLACE_INSTALL"] = "/opt/dreamplace"
    os.environ["OPEN3D_LOCAL_DREAMPLACE_INSTALL"] = "/opt/dreamplace"
    if request["runtime_receipt"]:
        os.environ["OPEN3D_LOCAL_OPENROAD_PREBUILT_RECEIPT"] = request["runtime_receipt"]
    options = request["options"]
    for key in ("baseline_run", "reference_run", "base_config"):
        options[key] = Path(options[key]) if options[key] is not None else None
    from .evaluation import evaluate_task
    result = evaluate_task(Path(__file__).resolve().parents[1] / "local_tasks" / request["task"],
        Path(request["candidate"]), Path(request["source"]), Path(request["output"]), **options)
    print(json.dumps({"metrics": result.metrics, "error": result.artifacts.get("error_message")}))


if __name__ == "__main__":
    main()
