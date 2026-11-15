"""Address frozen image contents consistently on classic and containerd Docker.

The native worker keeps its original config-digest identities. A host-only CLI
adapter translates those identities to verified local manifest addresses and
normalizes image-inspect output. Candidate images and container operations pass
through unchanged. No Docker socket or adapter authority enters the Agent.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


REGISTRY = Path(__file__).parent / "manifests/runtime-images.json"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def records():
    value = json.loads(REGISTRY.read_text())
    if value.get("schema_version") != 1 or not value.get("images"):
        raise ValueError("Missing frozen runtime identity registry")
    result = {}
    for row in value["images"]:
        if (not DIGEST.fullmatch(row["config_id"]) or not row["rootfs_layers"]
                or any(not DIGEST.fullmatch(v) for v in row["rootfs_layers"] + row["manifest_ids"])
                or row["platform"] != {"os": "linux", "architecture": "amd64"}
                or row["config_id"] in result):
            raise ValueError("Invalid frozen runtime identity record")
        result[row["config_id"]] = row
    return result


def real_docker():
    previous = os.environ.get("CONTEST_DOCKER_REAL")
    binary = previous or shutil.which("docker")
    if not binary or not Path(binary).is_file():
        raise ValueError("Docker CLI is unavailable")
    return str(Path(binary).resolve(strict=True))


def validate_inspection(value, record):
    if (value.get("Id") not in [record["config_id"], *record["manifest_ids"]]
            or value.get("RootFS", {}).get("Type") != "layers"
            or value.get("RootFS", {}).get("Layers") != record["rootfs_layers"]
            or value.get("Os") != record["platform"]["os"]
            or value.get("Architecture") != record["platform"]["architecture"]):
        raise ValueError("Loaded Docker image contents differ from the frozen runtime")
    return record["config_id"]


def resolve(image=None, *, docker=None):
    registry = records()
    image = image or next(iter(registry))
    if image not in registry:
        raise ValueError("Runtime is not in the frozen image registry")
    record = registry[image]
    binary = docker or real_docker()
    for address in [image, *record["manifest_ids"], "open3dbench-contest:runtime"]:
        result = subprocess.run([binary, "image", "inspect", address],
                                capture_output=True, text=True)
        if result.returncode:
            continue
        values = json.loads(result.stdout)
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError("Docker returned an ambiguous runtime identity")
        validate_inspection(values[0], record)
        return {"config_id": image, "docker_address": values[0]["Id"],
                "rootfs_layers": record["rootfs_layers"]}
    raise ValueError("Official runtime is not loaded; load the supplied Docker archive first")


def _create(path, content, mode):
    if path.is_symlink():
        raise ValueError("Docker compatibility files cannot be symlinks")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError("Docker compatibility binding changed")
    else:
        with path.open("xb") as stream:
            stream.write(content)
        path.chmod(mode)


def install(image, root):
    """Install an immutable, per-binding adapter in a host-owned work directory."""
    binary = real_docker()
    binding = resolve(image, docker=binary)
    value = {"docker": binary, "binding": binding, "record": records()[image],
             "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    directory = Path(root) / "control/docker-compat" / hashlib.sha256(raw).hexdigest()
    for path in (directory, *directory.parents):
        if path.is_symlink():
            raise ValueError("Docker compatibility directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    mapping = directory / "binding.json"
    _create(mapping, raw, 0o400)
    launcher = ("#!" + sys.executable + "\nimport os, sys\n"
                "os.execv(" + repr(sys.executable) + ", [" + repr(sys.executable) + ", "
                + repr(str(Path(__file__).resolve())) + ", '--docker-shim', "
                + repr(str(mapping.resolve())) + ", *sys.argv[1:]])\n").encode()
    _create(directory / "docker", launcher, 0o500)
    os.environ["CONTEST_DOCKER_REAL"] = binary
    os.environ["PATH"] = str(directory.resolve()) + os.pathsep + os.environ.get("PATH", os.defpath)
    return binding


GPU_PROBE = r'''
import json, torch
from dreamplace.ops.hpwl.hpwl import HPWL
assert torch.cuda.is_available(), 'CUDA is unavailable inside the official runtime'
p = torch.tensor([0.,2.,1.,0.,1.,3.], dtype=torch.float64, device='cuda', requires_grad=True)
(p.square().sum()).backward()
op = HPWL(flat_netpin=torch.tensor([0,1,2], dtype=torch.int32, device='cuda'),
          netpin_start=torch.tensor([0,3], dtype=torch.int32, device='cuda'),
          net_weights=torch.ones(1, dtype=torch.float64, device='cuda'),
          net_mask=torch.ones(1, dtype=torch.uint8, device='cuda'), algorithm='net-by-net')
assert abs(op(p.detach()).item()-5.) < 1e-9, 'DREAMPlace CUDA HPWL check failed'
torch.cuda.synchronize()
prop = torch.cuda.get_device_properties(0)
print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda,
 'gpu':torch.cuda.get_device_name(0), 'compute_capability':list(torch.cuda.get_device_capability(0)),
 'multiprocessors':prop.multi_processor_count, 'custom_cuda_operator':'hpwl', 'passed':True}))
'''


def gpu_preflight(image, root, gpus):
    """Check actual installed Torch and DREAMPlace CUDA execution before billing."""
    from grt_bench.common import write
    from grt_bench.runtime import stop_and_confirm
    binding = resolve(image)
    results = []
    if not gpus:
        raise ValueError("Official evaluation requires an NVIDIA GPU")
    for gpu in gpus:
        if not str(gpu).isdigit():
            raise ValueError("Invalid preflight GPU index")
        name = "o3db-gpu-check-" + uuid.uuid4().hex[:16]
        folder = Path(root) / "control/gpu-checks" / name
        folder.mkdir(parents=True, exist_ok=False)
        command = [real_docker(), "run", "--rm", "--pull=never", "--name", name,
                   "--network=none", "--read-only", "--cap-drop=ALL",
                   "--security-opt", "no-new-privileges", "--cpus", "1", "--memory", "4g",
                   "--pids-limit", "128", "--gpus", "device=" + str(gpu),
                   "--user", str(os.getuid()) + ":" + str(os.getgid()),
                   "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m",
                   "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
                   "--env", "OMP_NUM_THREADS=1",
                   "--env", "PYTHONPATH=/opt/dreamplace:/opt/dreamplace/dreamplace",
                   "--entrypoint", "python3", binding["docker_address"], "-c", GPU_PROBE]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            write(folder / "execution.json", {"image": binding, "gpu": str(gpu),
                  "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            if result.returncode:
                raise ValueError("The official CUDA runtime check failed on GPU " + str(gpu)
                                 + " (exit " + str(result.returncode) + ")"
                                 + "; see " + str(folder / "execution.json")
                                 + ". Check free memory, driver and GPU support; runtime upgrades require official qualification.")
            value = json.loads(result.stdout.strip().splitlines()[-1])
            if value.get("passed") is not True:
                raise ValueError("Invalid CUDA preflight result")
            results.append(value)
        finally:
            stop_and_confirm(name, folder)
    return results


def shim(mapping, args):
    value = json.loads(Path(mapping).read_text())
    if value["module_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Docker compatibility code differs from its binding")
    record = value["record"]
    canonical = record["config_id"]
    address = value["binding"]["docker_address"]
    args = [address if arg == canonical else arg for arg in args]
    inspect = args[:2] == ["image", "inspect"]
    if not inspect:
        os.execv(value["docker"], [value["docker"], *args])
    output_id = False
    for flag in ("--format", "-f"):
        if flag in args:
            i = args.index(flag)
            if args[i + 1] not in ("{{.Id}}", "{{ .Id }}"):
                os.execv(value["docker"], [value["docker"], *args])
            output_id = True
            args = args[:i] + args[i + 2:]
            break
    result = subprocess.run([value["docker"], *args], capture_output=True)
    sys.stderr.buffer.write(result.stderr)
    if result.returncode:
        sys.stdout.buffer.write(result.stdout)
        return result.returncode
    values = json.loads(result.stdout)
    for row in values:
        if row.get("Id") in [canonical, *record["manifest_ids"]]:
            validate_inspection(row, record)
            row["Id"] = canonical
    if output_id:
        print("\n".join(row["Id"] for row in values))
    else:
        print(json.dumps(values))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "--docker-shim":
        raise SystemExit("Host Docker compatibility adapter")
    raise SystemExit(shim(sys.argv[2], sys.argv[3:]))
