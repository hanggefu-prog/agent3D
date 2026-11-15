"""Execute host-authored matrix jobs, heartbeat leases, preserve failed attempts."""
import argparse
import json
from pathlib import Path
import shutil
import time
import threading
from urllib.request import Request, urlopen

from .candidates import materialize, verify_snapshot
from .common import read, sha, write, evaluator_identity
from .contract import validate
from .runtime import admission, docker_command, run_container, stop_and_confirm, UncertainExecution


class CandidateExecutionError(RuntimeError):
    pass


def native_execution(call, *args, **kwargs):
    # run_container has confirmed cgroup termination before reporting RuntimeError.
    # Timeout/uncertain execution and admission/verification errors are not retriable.
    try:
        return call(*args, **kwargs)
    except RuntimeError as error:
        if isinstance(error, UncertainExecution):
            raise
        # This exact runtime error means a terminated nonzero native process.
        if str(error) == "Native container failed; inspect retained log":
            raise CandidateExecutionError(str(error)) from error
        raise


def run_phased(lease, image, mounts, attempt, backend_resources, heartbeat, release):
    if release is None or lease["resources"]["gpu_count"] != 1:
        raise ValueError("Phased execution requires one GPU slot and a release callback")
    if set(backend_resources) != set(lease["resources"]) or backend_resources["gpu_count"] != 0:
        raise ValueError("Backend must use the same resource schema without GPU")
    if any(backend_resources[k] > lease["resources"][k] for k in backend_resources):
        raise ValueError("Backend allocation cannot exceed the frontend reservation")
    started = time.monotonic()
    name = "grt-" + lease["id"][:32]
    phases = []
    command = docker_command(image, name + "-frontend", lease["resources"], lease["gpus"], mounts,
                             ["-m", "grt_bench.flow", "/control/plan.json", "--phase", "frontend"])
    phases.append(run_container(command, attempt / "execution/frontend", name + "-frontend", heartbeat=heartbeat))
    stop_and_confirm(name + "-frontend", attempt / "execution/frontend")
    evidence = {"container": name + "-frontend", "terminated": True,
                "checkpoint_sha256": sha(attempt / "output/run/30_placement/checkpoint.json")}
    allocation = release(backend_resources, evidence)
    if allocation["resources"] != backend_resources or allocation["gpus"]:
        raise ValueError("Broker did not confirm backend allocation")
    write(attempt / "resource-release.json", {"allocation": allocation, "evidence": evidence,
        "elapsed_seconds": time.monotonic() - started})
    admission(attempt, backend_resources, [])
    remaining = 21600 - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Physical evaluation exceeded its frozen budget")
    command = docker_command(image, name + "-backend", backend_resources, [], mounts,
                             ["-m", "grt_bench.flow", "/control/plan.json", "--phase", "backend"])
    phases.append(run_container(command, attempt / "execution/backend", name + "-backend",
                                timeout=remaining, heartbeat=heartbeat))
    return {"exit_code": 0, "wall_seconds": time.monotonic() - started, "phases": phases,
            "execution_mode": "gpu-frontend-cpu-backend"}


def evaluate(lease, environment, heartbeat, release=None):
    payload = lease["payload"]
    root = Path(environment["root"]).resolve(strict=True)
    task_path = root / "control/tasks" / (payload["task_digest"] + ".json")
    task = read(task_path)
    if validate(task) != payload["task_digest"]:
        raise ValueError("Unbound task")
    if task["evaluator_sha256"] != evaluator_identity():
        raise ValueError("Evaluator differs from frozen task")
    if payload["seed"] not in task["seeds"]:
        raise ValueError("Unregistered seed")
    case = next(c for c in task["cases"] if c["id"] == payload["case"])
    if payload["split"] != case["split"]:
        raise ValueError("Wrong split")
    # Independent test release is host-owned, never accepted from a model turn.
    if case["split"] != "train":
        test_release = read(root / "control/test-releases" / (payload["candidate"] + ".json"))
        if test_release != {"candidate":payload["candidate"], "task_digest":payload["task_digest"], "frozen":True}:
            raise ValueError("Held-out evaluation requires a frozen candidate release")
    source = root / "candidates" / payload["candidate"]
    candidate_audit = None
    if "optimization" in task:
        from .archive import Archive
        candidate_audit = Archive(root, task).admit(payload["candidate"])
        frozen = read(source / "candidate.json")  # admit already hashes this snapshot
    else:
        frozen = verify_snapshot(source)
    inputs = environment["cases"][case["id"]]
    config = Path(inputs["config"]).resolve(strict=True)
    if sha(config) != case["config_sha256"]:
        raise ValueError("Case config changed")
    from .common import digest, verify_records
    input_manifest = read(inputs["manifest"])
    if digest(input_manifest) != case["input_identity"]:
        raise ValueError("Input identity differs")
    verify_records(Path(inputs["manifest"]).parent, input_manifest["artifacts"])
    runtime = read(root / "control/runtimes" / (payload["candidate"] + ".json"))
    if runtime["candidate"] != payload["candidate"]:
        raise ValueError("Runtime belongs to another candidate")
    if (frozen.get("scope") or {}).get("native_build_required"):
        if runtime.get("native") is not True:
            raise ValueError("Native edits require an installed candidate runtime")
        from .builder import verify_runtime
        from local_bench.workspace import openroad_source_fingerprint
        if runtime["manifest"]["source_fingerprint"] != openroad_source_fingerprint(source / "tree/OpenROAD-3D/openroad"):
            raise ValueError("Installed runtime was built from another native source")
        verify_runtime(runtime)
    else:
        if runtime["image"] != environment["base_image"]:
            raise ValueError("Unexpected Python-only runtime")
    admitted = admission(root, lease["resources"], lease["gpus"])
    attempt = root / "runs" / lease["id"]
    attempt.mkdir(parents=True, exist_ok=False)
    write(attempt / "admission.json", admitted)
    if candidate_audit is not None:
        write(attempt / "candidate-verification.json", candidate_audit)
    replay = environment.get("rescore_sources", {}).get(lease["id"])
    phased = environment.get("phase_resources")
    incremental = environment.get('incremental_plans', {}).get(lease['id'])
    if incremental:
        from .incremental import ORDER
        if ORDER.index(incremental['impact']['start']) > ORDER.index('placement'):
            if lease['resources']['gpu_count'] != 0:
                raise ValueError('CPU stage suffix must not reserve a GPU')
            phased = None
        if replay:
            raise ValueError('Analysis replay and physical suffix are distinct operations')
    if replay and phased:
        raise ValueError("An analysis replay cannot be a fresh phased execution")
    previous = None
    physical = None
    if replay:
        from .queue import Queue
        from .rescore import validate_transition, validate_drivers
        import re
        if not re.fullmatch(r"[0-9a-f]{64}", replay):
            raise ValueError("Invalid host-bound physical source")
        previous = root / "runs" / replay
        old_row = next(r for r in Queue(root / "control/queue.sqlite").rows() if r["id"] == replay)
        if old_row["state"] not in {"succeeded", "failed"}:
            raise ValueError("Physical source has not terminated")
        validate_transition(task, read(previous / "control/task.json"), payload, read(previous / "control/plan.json"))
        validate_drivers(environment["rescore_previous_framework"], environment["framework"],
                         read(previous / "control/task.json"), task)
        from .incremental import physical_source
        physical_job, physical_output = physical_source(root, replay)
        physical = root / 'runs' / physical_job
        prior_plan = read(physical / 'control/plan.json')
        commands = ('execution/frontend/command.json', 'execution/backend/command.json') if (
            prior_plan.get('execution_mode') == 'gpu-frontend-cpu-backend') else ('execution/command.json',)
        if any('OPEN3DBENCH_IMAGE_ID=' + runtime['image'] not in read(physical / name) for name in commands):
            raise ValueError('Physical replay runtime differs')
        workspace = physical / "workspace"
    else:
        workspace = materialize(source, attempt / "workspace")
    output = attempt / "output"; output.mkdir()
    control = attempt / "control"; control.mkdir()
    verified = attempt / "verified"; verified.mkdir()
    shutil.copyfile(task_path, control / "task.json")
    shutil.copyfile(config, control / "case.yaml")
    plan = {**payload, "task":"/control/task.json", "config":"/control/case.yaml", "source":"/workspace",
            "output":"/output", "hotspot":"/opt/open3dbench-assets/hotspot", "timeout_seconds":21600}
    if phased:
        plan.update(execution_mode="gpu-frontend-cpu-backend", phase_resources=phased)
    prefix = inputs.get("frontend_prefix")
    if 'frontend_inputs' in task and not prefix and not replay:
        raise ValueError('Task-frozen frontend is required; regeneration is forbidden')
    if prefix:
        plan["frontend_prefix"] = "/inputs/frontend"
    framework = Path(environment["framework"]).resolve(strict=True)
    mounts = [(workspace,"/workspace",bool(previous)),(framework,"/trusted",True),(control,"/control",True),(output,"/output",False)]
    from .algorithm_scope import python_mounts
    # The installed launcher imports these paths. Mounting the repository alone
    # does not cause DREAMPlace to execute a modified optimizer.
    algorithm_mounts = python_mounts(workspace, frozen['files'])
    mounts += algorithm_mounts
    python_runtime = {
        'candidate': payload['candidate'], 'image': runtime['image'],
        'modules': {target: {'source': str(path.relative_to(workspace)), 'sha256': sha(path)}
                    for path, target, _ in algorithm_mounts}}
    write(control / 'python-runtime.json', python_runtime)
    plan.update(python_runtime='/control/python-runtime.json', python_runtime_digest=digest(python_runtime))
    mounts.append((Path(inputs["manifest"]).resolve().parent,"/inputs/case",True))
    if incremental:
        from .incremental import bind_worker
        parent_output = bind_worker(root, control, payload, incremental, runtime)
        plan.update(incremental='/control/incremental.json', incremental_source='/parent-output')
        mounts.append((parent_output, '/parent-output', True))
    if previous:
        plan["physical_output"] = "/physical"
        plan["rescore"] = {"source_job":replay, "previous_plan":"/previous-control/plan.json",
            "previous_plan_sha256":sha(previous / "control/plan.json"), "previous_task":"/previous-control/task.json"}
        plan["rescore"]["physical_source_job"] = physical_job
        mounts += [(physical_output,"/physical",True),(previous / "control","/previous-control",True)]
        prefix = None
        plan.pop("frontend_prefix", None)
        # Failed scoring may not have reached the original worker's source audit.
        for relative, row in frozen["files"].items():
            path = workspace / relative
            if "symlink" in row:
                import os
                valid = path.is_symlink() and os.readlink(path) == row["symlink"]
            else:
                valid = not path.is_symlink() and sha(path) == row["sha256"]
            if not valid:
                raise ValueError("Physical replay source changed: " + relative)
        for path in workspace.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".so", ".pyc", ".pyo"} and path.relative_to(workspace).as_posix() not in frozen["files"]:
                raise ValueError("Physical replay contains new executable source: " + str(path))
    elif not prefix and environment.get("cache_frontend_prefix", True):
        from .prefix import prepare
        prefix, reused = prepare(root, payload, runtime["image"], lease["resources"], lease["gpus"],
                                 mounts, control, plan, heartbeat)
        plan["frontend_prefix"] = "/inputs/frontend"
        write(attempt / "prefix.json", {"source":str(prefix), "reused":reused,
                                       "validation":"native-stage-signatures-and-artifact-hashes"})
    if prefix:
        mounts.append((Path(prefix),"/inputs/frontend",True))
    write(control / "plan.json", plan)
    name = "grt-" + lease["id"][:32]
    verification_resources = lease["resources"]
    if phased:
        native = native_execution(run_phased, lease, runtime["image"], mounts, attempt, phased["backend"], heartbeat, release)
        verification_resources = phased["backend"]
    else:
        command = docker_command(runtime["image"], name, lease["resources"], lease["gpus"], mounts,
                                 ["-m","grt_bench.rescore" if previous else "grt_bench.flow","/control/plan.json"])
        native = native_execution(run_container, command, attempt / "execution", name, heartbeat=heartbeat)
    # Check source bytes before starting the read-only independent consumer.
    for relative, row in frozen["files"].items():
        path = workspace / relative
        if "symlink" in row:
            import os
            matches = path.is_symlink() and os.readlink(path) == row["symlink"]
        else:
            matches = not path.is_symlink() and sha(path) == row["sha256"]
        if not matches:
            raise ValueError("Evaluation mutated a source input: " + relative)
    # Generated data may be added, but cannot shadow a trusted Python import.
    for path in workspace.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".so", ".pyc", ".pyo"} and path.relative_to(workspace).as_posix() not in frozen["files"]:
            raise ValueError("Evaluation introduced executable source: " + str(path))
    readonly = [(p,t,True) for p,t,_ in mounts] + [(verified,"/verified",False)]
    command = docker_command(runtime["image"], name + "-verify", dict(verification_resources,gpu_count=0), [], readonly,
                             ["-m","grt_bench.verify","/control/plan.json","/verified/result.json"])
    verification = run_container(command, attempt / "verification", name + "-verify", timeout=1200, heartbeat=heartbeat)
    result = read(verified / "result.json")
    result["accounting"] = {"execution":native,"verification":verification}
    if candidate_audit is not None:
        result["candidate_verification_sha256"] = sha(attempt / "candidate-verification.json")
    write(attempt / "result.json", result)
    return result


class Client:
    def __init__(self, url, host, token):
        if not url.startswith("http://127.0.0.1:"):
            raise ValueError("Use loopback/SSH forwarding for the worker API")
        self.url, self.host, self.token = url.rstrip("/"), host, token

    def call(self, action, value):
        request = Request(self.url + "/" + action, data=json.dumps(value).encode(), method="POST",
            headers={"Authorization":"Bearer " + self.token,"X-Worker-Host":self.host,"Content-Type":"application/json"})
        with urlopen(request,timeout=20) as response:
            return json.load(response)


def loop(client, environment, *, once=False):
    while True:
        lease = client.call("claim", {})
        if lease is None:
            if once: return
            time.sleep(5)
            continue
        stop, lost = threading.Event(), threading.Event()
        def keep_lease():
            while not stop.wait(20):
                try:
                    client.call("heartbeat", {"job":lease["id"],"token":lease["token"]})
                except Exception:
                    lost.set()
                    return
        thread = threading.Thread(target=keep_lease, daemon=True)
        thread.start()
        def beat():
            if lost.is_set():
                raise RuntimeError("Lease heartbeat lost; stop container and reconcile")
        try:
            def release(resources, evidence):
                beat()
                return client.call("release", {"job": lease["id"], "token": lease["token"],
                                              "resources": resources, "evidence": evidence})
            result = evaluate(lease, environment, beat, release)
            success = True
        except UncertainExecution:
            # Do not acknowledge failure and release a potentially live GPU.
            # The expired lease remains reserved for host reconciliation.
            raise
        except Exception as error:
            result = {"status":"failed","error":type(error).__name__ + ": " + str(error),
                      "failure_kind": "candidate_execution" if isinstance(error, CandidateExecutionError) else "host_or_integrity"}
            success = False
        finally:
            stop.set()
            thread.join(timeout=25)
        # If acknowledgement is lost, leave the lease for host reconciliation.
        # Never dispatch the physical attempt again on an uncertain POST.
        client.call("finish", {"job":lease["id"],"token":lease["token"],"result":result,"success":success})
        if once: return


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--environment",type=Path,required=True)
    p.add_argument("--token",type=Path,required=True)
    p.add_argument("--host",required=True)
    p.add_argument("--url",default="http://127.0.0.1:8765")
    p.add_argument("--once",action="store_true")
    a=p.parse_args()
    if a.token.is_symlink() or a.token.stat().st_mode & 0o077:
        raise ValueError("Worker token must be private")
    loop(Client(a.url,a.host,a.token.read_text().strip()),read(a.environment),once=a.once)


if __name__ == "__main__": main()
