"""Pinned Docker launch plans and admission; no host EDA binary fallback."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from .common import digest, read, sha, write


class UncertainExecution(RuntimeError):
    """The host cannot prove termination; retain the resource lease."""


def stop_and_confirm(name, output):
    try:
        stopped = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=30)
        inspected = subprocess.run(["docker", "container", "inspect", "--format", "{{.State.Running}}", name],
                                   capture_output=True, text=True, timeout=20)
    except Exception as error:
        raise UncertainExecution("Cannot confirm container termination: " + name) from error
    absent = inspected.returncode != 0 and name in inspected.stderr and re.search(r"No such (?:container|object):", inspected.stderr) is not None
    confirmed = absent or (inspected.returncode == 0 and inspected.stdout.strip() == "false")
    try:
        write(Path(output) / "stop.json", {"container": name, "returncode": stopped.returncode,
            "stdout": stopped.stdout, "stderr": stopped.stderr, "terminated": confirmed,
            "inspection": {"returncode": inspected.returncode, "stdout": inspected.stdout, "stderr": inspected.stderr}})
    finally:
        if not confirmed:
            raise UncertainExecution("Container termination unconfirmed; retain lease: " + name)


def mount(source, destination, readonly=True):
    source = Path(source).resolve(strict=True)
    if any(c in str(source) for c in (",", "\n", "\r")):
        raise ValueError("Unsupported mount path")
    return ["--mount", "type=bind,source={},target={}{}".format(source, destination, ",readonly" if readonly else "")]


def docker_command(image, name, resources, gpus, mounts, arguments):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("A pinned local image ID is required")
    if not re.fullmatch(r"grt-[a-z0-9-]{1,90}", name):
        raise ValueError("Invalid task-owned container name")
    command = ["docker", "run", "--rm", "--pull=never", "--name", name,
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt", "no-new-privileges",
        "--cpus", str(resources["cpu"]), "--memory", str(resources["memory_gb"]) + "g",
        "--memory-swap", str(resources["memory_gb"]) + "g", "--pids-limit", "4096",
        "--user", "{}:{}".format(os.getuid(), os.getgid()), "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONPYCACHEPREFIX=/tmp/python-cache", "-e", "HOME=/tmp", "-e", "PYTHONPATH=/trusted:/workspace:/opt/dreamplace:/opt/dreamplace/dreamplace",
        "-e", "OPEN3DBENCH_IMAGE_ID=" + image, "--workdir", "/trusted"]
    threads = str(max(1, min(4, int(resources["cpu"]))))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPEN3DBENCH_THREADS"):
        command += ["-e", variable + "=" + threads]
    if gpus:
        if len(gpus) != resources["gpu_count"] or any(not str(g).isdigit() for g in gpus):
            raise ValueError("GPU assignment differs from lease")
        command += ["--gpus", '"device=' + ",".join(gpus) + '"']
    elif resources["gpu_count"]:
        raise ValueError("Missing GPU lease")
    for source, target, readonly in mounts:
        command += mount(source, target, readonly)
    return command + ["--entrypoint", "python3", image] + list(arguments)


def admission(root, resources, gpus):
    root = Path(root)
    free = shutil.disk_usage(root).free
    required = (resources["storage_gb"] + 10) * 1024**3
    if free < required:
        raise RuntimeError("Insufficient free disk: {} < {} bytes".format(free, required))
    data = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        data[key] = int(value.strip().split()[0]) * 1024
    if data["MemAvailable"] < (resources["memory_gb"] + 4) * 1024**3:
        raise RuntimeError("Insufficient available memory")
    subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, check=True, timeout=20)
    for gpu in gpus:
        result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=10)
        if int(result.stdout.strip()) < 4096:
            raise RuntimeError("Insufficient GPU memory")
    return {"free_disk_bytes": free, "memory_available_bytes": data["MemAvailable"], "gpus": gpus}


def run_container(command, output, name, timeout=21600, heartbeat=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "command.json", command)
    start = time.monotonic()
    process = None
    try:
        with (output / "container.log").open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while True:
                if heartbeat:
                    heartbeat()
                try:
                    code = process.wait(timeout=min(15, max(0.1, timeout - (time.monotonic() - start))))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - start >= timeout:
                        raise TimeoutError("Physical evaluation exceeded its frozen budget")
    except BaseException:
        # A disconnected docker client is not proof its container stopped.
        try:
            stop_and_confirm(name, output)
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except Exception as error:
                        raise UncertainExecution("Docker client cleanup did not complete") from error
        raise
    receipt = {"exit_code": code, "wall_seconds": time.monotonic() - start, "container": name}
    write(output / "terminal.json", receipt)
    if code:
        stop_and_confirm(name, output)
        raise RuntimeError("Native container failed; inspect retained log")
    return receipt
