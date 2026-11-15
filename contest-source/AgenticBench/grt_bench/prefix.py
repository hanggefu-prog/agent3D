"""Single-flight, same-candidate frontend reuse through the native validator."""
import fcntl
from pathlib import Path
import time
from .common import digest, read, sha, write
from .runtime import docker_command, run_container


def prepare(root, payload, image, resources, gpus, mounts, control, plan, heartbeat):
    # Partition/placement seeds do not affect these two stages. Candidate and
    # image remain in the key: no cross-implementation compatibility is inferred.
    identity = digest({"candidate": payload["candidate"], "task": payload["task_digest"],
                       "case": payload["case"], "io_seed": payload["seed"]["io"], "image": image})
    store = Path(root) / "cache/frontend-prefixes"
    store.mkdir(parents=True, exist_ok=True)
    destination = store / identity
    with (store / (identity + ".lock")).open("a+") as lock:
        while True:
            heartbeat()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(1)
        receipt = destination / "ready.json"
        if receipt.exists():
            for relative, expected in read(receipt)["manifests"].items():
                if sha(destination / "output/run" / relative) != expected:
                    raise ValueError("Cached prefix manifest changed")
            return destination / "output/run", True
        if destination.exists():
            raise ValueError("Incomplete frontend prefix retained; explicit new attempt required")
        destination.mkdir()
        output = destination / "output"
        output.mkdir()
        write(Path(control) / "prefix-plan.json", plan)
        prefix_mounts = [(output if target == "/output" else path, target, readonly)
                         for path, target, readonly in mounts]
        name = "grt-prefix-" + identity[:32]
        command = docker_command(image, name, resources, gpus, prefix_mounts,
                                 ["-m", "grt_bench.flow", "/control/prefix-plan.json", "--prefix-only"])
        execution = run_container(command, destination / "execution", name, heartbeat=heartbeat)
        manifests = {stage + "/stage_manifest.json": sha(output / "run" / stage / "stage_manifest.json")
                     for stage in ("00_input", "10_floorplan")}
        write(receipt, {"identity": identity, "execution": execution, "manifests": manifests})
        return output / "run", False
