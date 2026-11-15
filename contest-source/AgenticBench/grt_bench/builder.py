"""Single-flight native builds shared by all case/seed jobs of a candidate.

Uses the maintained bounded private-tree/shared-ccache builder. Native runtime
is installed at the normal image path; host binaries are never injected at run
time. Physical algorithms use candidate OpenROAD; scoring uses the fixed
standalone OpenSTA from the pinned base. OpenSTA source is outside edit scope.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from .candidates import verify_snapshot
from .common import digest, read, sha, write


def runtime_key(candidate, base_image, recipe_sha256):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", base_image):
        raise ValueError("Pinned base image required")
    return digest({"candidate": candidate, "base_image": base_image, "recipe": recipe_sha256, "schema": 1})


def verify_runtime(receipt):
    image = receipt["image"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Unpinned runtime")
    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--read-only", "--cpus", "1",
        "--memory", "1g", "--cap-drop=ALL", "--security-opt", "no-new-privileges", "--entrypoint", "python3", image,
        "-c", "import hashlib,json,pathlib,subprocess; p=pathlib.Path('/usr/local/bin/openroad'); "
        "print(json.dumps({'binary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),"
        "'manifest':json.loads(pathlib.Path('/opt/agentic3d-grt-runtime.json').read_text()),"
        "'version_exit_code':subprocess.run([str(p),'-version'],stdout=subprocess.DEVNULL).returncode}))"]
    value = json.loads(subprocess.check_output(command, text=True, timeout=60))
    if value["binary_sha256"] != receipt["binary_sha256"] or value["manifest"] != receipt["manifest"] or value["version_exit_code"] != 0:
        raise ValueError("Installed runtime does not match candidate receipt")
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image, receipt["manifest"]["base_image"]],text=True))
    base_layers = inspected[1]["RootFS"]["Layers"]
    if inspected[0]["RootFS"]["Layers"][:len(base_layers)] != base_layers:
        raise ValueError("Candidate runtime does not preserve the pinned base layers")
    return value


def local_base_tag(image):
    """Dockerfile FROM needs a name; a bare sha256 ID is parsed as a tag."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Pinned base image required")
    tag = "agentic3d-grt-base:" + image.split(":")[1]
    subprocess.run(["docker", "tag", image, tag], check=True)
    if subprocess.check_output(["docker", "image", "inspect", tag, "--format", "{{.Id}}"],text=True).strip() != image:
        raise ValueError("Local base tag differs from pinned image")
    return tag


def build(snapshot, base_image, store, cache, threads=8):
    from scripts.openroad_compiler_cache import compile_cached
    from local_bench.workspace import openroad_source_fingerprint
    snapshot = Path(snapshot).resolve()
    source = verify_snapshot(snapshot)
    if not (source.get("scope") or {}).get("native_build_required"):
        return {"image": base_image, "candidate": snapshot.name, "native": False, "mode": "base-runtime-reuse"}
    framework = Path(__file__).resolve().parents[1]
    recipe = framework / "scripts/build_openroad_candidate.sh"
    native_source = snapshot / "tree/OpenROAD-3D/openroad"
    fingerprint = openroad_source_fingerprint(native_source)
    identity = runtime_key(fingerprint["sha256"], base_image, digest({"recipe":sha(recipe), "builder":sha(__file__)}))
    store = Path(store); store.mkdir(parents=True, exist_ok=True)
    lock = store / (identity + ".lock")
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        attempt = store / identity
        if (attempt / "receipt.json").exists():
            receipt = read(attempt / "receipt.json")
            if receipt["manifest"]["source_fingerprint"] != fingerprint or receipt["base_image"] != base_image:
                raise ValueError("Runtime cache identity changed")
            verify_runtime(receipt)
            if receipt["candidate"] == snapshot.name:
                return receipt
            return {**receipt, "candidate":snapshot.name, "reused_build_candidate":receipt["candidate"],
                    "mode":"source-identical-runtime-reuse"}
        if attempt.exists():
            raise ValueError("Prior incomplete build retained; no implicit retry")
        attempt.mkdir()
        started = time.monotonic()
        compiled = compile_cached(base_image=base_image, source_tree=native_source, builder=recipe,
            output=attempt / "compile", cache_root=Path(cache), threads=threads)
        if compiled["source_fingerprint"] != openroad_source_fingerprint(native_source):
            raise ValueError("Built native source differs")
        package = attempt / "package"; package.mkdir()
        shutil.copyfile(compiled["binary_path"], package / "openroad")
        (package / "openroad").chmod(0o755)
        manifest = {"kind": "grt-matrix-installed-openroad-v1", "candidate": snapshot.name,
            "binary_sha256": compiled["binary_sha256"], "source_fingerprint": compiled["source_fingerprint"],
            "base_image": base_image, "cache_key": compiled["cache_key"], "consumer": "candidate-openroad-physical/fixed-opensta-scoring"}
        write(package / "runtime.json", manifest)
        (package / "Dockerfile").write_text("FROM " + local_base_tag(base_image) + "\nUSER root\nCOPY openroad /usr/local/bin/openroad\nCOPY runtime.json /opt/agentic3d-grt-runtime.json\n")
        iid = attempt / "image.id"
        command = ["docker", "build", "--pull=false", "--network=none", "--iidfile", str(iid), str(package)]
        with (attempt / "package.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        receipt = {"candidate": snapshot.name, "base_image": base_image, "image": iid.read_text().strip(),
            "native": True, "mode": "private-tree-shared-ccache", "manifest": manifest,
            "binary_sha256": compiled["binary_sha256"], "cache_key": compiled["cache_key"],
            "wall_seconds": time.monotonic() - started}
        receipt["verification"] = verify_runtime(receipt)
        verify_snapshot(snapshot)
        write(attempt / "receipt.json", receipt)
        return receipt
