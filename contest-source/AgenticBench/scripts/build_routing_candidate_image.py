#!/usr/bin/env python3
"""Build a verified routing candidate into a pinned, image-installed runtime.

The caller owns scheduling/admission. No Docker build or model call happens on
import. This tool never mounts host credentials or accepts candidate commands.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_bench.model import ROUTING_SOURCE_TASK_KINDS, load_candidate, load_task
from local_bench.verifier import verify_candidate
from local_bench.workspace import (
    OPENROAD_SUBMODULE, PREBUILT_OPENROAD_BINARY, PREBUILT_OPENROAD_MANIFEST,
    WorkspaceError, apply_candidate_patch, isolated_clone, openroad_source_fingerprint,
)

CPU_QUOTA = 1600000  # 16 CPUs, same admission as the local evaluation worker.
MEMORY_LIMIT = "48g"


@contextmanager
def _phase(output: Path, name: str):
    """Append boundaries; never overwrite queue/build timestamps with a total."""
    start = time.monotonic()
    def event(state, **extra):
        with (output / "image-build-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"phase": name, "state": state,
                "at_utc": datetime.now(timezone.utc).isoformat(), **extra}, sort_keys=True) + "\n")
    event("started")
    try:
        yield
    except BaseException as error:
        event("failed", duration_seconds=time.monotonic() - start,
              error_type=type(error).__name__, error=str(error))
        raise
    else:
        event("completed", duration_seconds=time.monotonic() - start)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _run_json(command: list[str]) -> Any:
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(process.stdout)


def _local_image_id(value: str) -> str:
    # A mutable tag cannot be a reproducibility boundary. Inspect never pulls.
    if not (re.fullmatch(r"sha256:[0-9a-f]{64}", value)
            or re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", value)):
        raise ValueError("base_image must be a pinned local image ID or repository digest")
    inspected = _run_json(["docker", "image", "inspect", value])
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise WorkspaceError("Docker did not identify exactly one local base image")
    result = inspected[0].get("Id")
    if not isinstance(result, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", result):
        raise WorkspaceError("Docker returned an invalid image ID")
    return result


def _dockerfile(base_id: str, threads: int, build_cache: str = "discard") -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", base_id):
        raise ValueError("Dockerfile requires an exact image ID")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
        raise ValueError("build threads must be an integer in [1, 16]")
    if build_cache not in {"discard", "retain"}:
        raise ValueError("build_cache must be discard or retain")
    # There is no shared incremental cache in these per-candidate legacy builds.
    # Do not commit compiler objects and a second ccache copy into a Docker layer.
    build_env = "CCACHE_DIR=/build/ccache CCACHE_DISABLE=1 " if build_cache == "discard" else ""
    cleanup = " && rm -rf /build/openroad /build/ccache" if build_cache == "discard" else ""
    return """FROM {base} AS candidate_builder
USER root
COPY openroad /src/OpenROAD
COPY build_openroad_candidate.sh seal_runtime.py runtime-template.json /trusted/
RUN command -v cmake && command -v ccache && command -v c++ && \\
    {build_env}bash /trusted/build_openroad_candidate.sh /src/OpenROAD /build/openroad {threads} && \\
    install -D -m 0755 /build/openroad/bin/openroad /candidate/openroad && \\
    python3 /trusted/seal_runtime.py /candidate/openroad /trusted/runtime-template.json /candidate/runtime.json{cleanup}
FROM {base}
USER root
COPY --from=candidate_builder /candidate/openroad {binary}
COPY --from=candidate_builder /candidate/runtime.json {manifest}
RUN ln -sf {binary} /usr/local/bin/openroad
ENV OPENROAD_EXE={binary}
""".format(base=base_id, threads=threads, build_env=build_env, cleanup=cleanup, binary=PREBUILT_OPENROAD_BINARY,
           manifest=PREBUILT_OPENROAD_MANIFEST)


_SEAL_SCRIPT = """import hashlib,json,sys
from pathlib import Path
binary,template,output=map(Path,sys.argv[1:])
data=json.loads(template.read_text())
h=hashlib.sha256()
with binary.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):
        h.update(block)
data['binary_sha256']=h.hexdigest()
output.write_text(json.dumps(data,indent=2,sort_keys=True)+'\\n')
"""


_VERIFY_SCRIPT = """import hashlib,json,os,subprocess,sys
from pathlib import Path
p=Path(sys.argv[1])
if p.is_symlink() or not p.is_file() or not os.access(p,os.X_OK):
    raise SystemExit('invalid installed OpenROAD executable')
if Path('/usr/local/bin/openroad').resolve()!=p:
    raise SystemExit('OpenROAD command does not select the installed candidate')
h=hashlib.sha256()
with p.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):
        h.update(block)
deps=subprocess.run(['ldd',str(p)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
if deps.returncode or 'not found' in deps.stdout:
    raise SystemExit('installed OpenROAD dependencies failed: '+deps.stdout)
version=subprocess.run([str(p),'-version'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
if version.returncode:
    raise SystemExit('installed OpenROAD version check failed: '+version.stdout)
print(json.dumps({'binary_sha256':h.hexdigest(),'binary_path':str(p),
                  'resolved_command':str(Path('/usr/local/bin/openroad').resolve()),
                  'dependencies':deps.stdout,'version':version.stdout,'version_exit_code':version.returncode}))
"""


def _package_dockerfile(base_id: str) -> str:
    # Reuse input validation; the cached backend only changes compilation,
    # never the image-installed execution contract.
    _dockerfile(base_id, 1)
    return """FROM {base}
USER root
COPY openroad {binary}
COPY runtime.json {manifest}
RUN chmod 0755 {binary} && ln -sf {binary} /usr/local/bin/openroad
ENV OPENROAD_EXE={binary}
""".format(base=base_id, binary=PREBUILT_OPENROAD_BINARY, manifest=PREBUILT_OPENROAD_MANIFEST)


def _discard_build_copies(output: Path) -> None:
    """Discard only successful build-local copies, after saving the recipe.

    This never removes an image, run, installed binary, log, patch or receipt.
    Failed builds remain untouched. The patch and pinned source fingerprint in
    receipt.json reproduce the copies; reproduction/ contains the build recipe.
    """
    context = output / "context"
    targets = [output / "workspace", context / "openroad"]
    # Only private, regenerable object files belong to this successful attempt.
    # The explicitly selected shared cache and all failed attempts are untouched.
    if (output / "compiler-cache-result.json").is_file():
        targets.extend(output / "compiler-cache-attempt" / name for name in ("build", "build-source"))
    recipes = [context / name for name in (
        "Dockerfile", "runtime-template.json", "seal_runtime.py", "build_openroad_candidate.sh")]
    for path in [output, context, *targets, *recipes, output / "receipt.json"]:
        if path.is_symlink() or not path.exists():
            raise WorkspaceError("unsafe or missing build cache path: " + str(path))
        path.resolve(strict=True).relative_to(output.resolve(strict=True))
    evidence = output / "reproduction"
    evidence.mkdir(exist_ok=False)
    hashes = {}
    for path in recipes:
        shutil.copy2(path, evidence / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"schema_version": 1, "policy": "discard", "state": "prepared",
              "deleted_paths": [], "planned_paths": [str(path.relative_to(output)) for path in targets],
              "recipe_sha256": hashes,
              "receipt_sha256": hashlib.sha256((output / "receipt.json").read_bytes()).hexdigest(),
              "interpretation": "Regenerable build copies only; installed image and physical evidence retained"}
    ledger = output / "cache-retention.json"
    ledger.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for path in targets:
        shutil.rmtree(path)
        record["deleted_paths"].append(str(path.relative_to(output)))
        ledger.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    record["state"] = "completed"
    ledger.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _finish_build_cache(output: Path, image_id: str) -> None:
    """Storage errors cannot invalidate an already verified installed image."""
    try:
        _discard_build_copies(output)
    except Exception as error:
        warning = {"state": "storage-error", "image_id": image_id,
                   "error": f"{type(error).__name__}: {error}",
                   "installed_image_verified": True}
        try:
            with (output / "cache-retention-error.json").open("x") as stream:
                json.dump(warning, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except OSError:
            pass
        print(json.dumps({"event": "build-cache-retention", **warning}), file=sys.stderr, flush=True)


def build_candidate_image(
    task_dir: Path, candidate_dir: Path, source: Path, base_image: str,
    output_dir: Path, threads: int = 8, build_cache: str = "discard",
    *, compiler_cache_dir: Path | None = None, compiler_cache_gb: int = 8,
    base_runtime_equivalence: dict | None = None,
) -> Dict[str, str]:
    """Return ``{image, receipt}``; compile only in a quota-limited container.

    candidate_dir must contain the complete materialized candidate.json emitted
    by the trusted task adapter. output_dir must not exist. A failed build is
    retained and never automatically retried. The caller must run evaluation
    with OPEN3DBENCH_IMAGE_ID=image and the receipt environment variable, while
    retaining the same scheduler slot across build and evaluation.
    """
    if build_cache not in {"discard", "retain"}:
        raise ValueError("build_cache must be discard or retain")
    if type(compiler_cache_gb) is not int or not 1 <= compiler_cache_gb <= 64:
        raise ValueError("compiler_cache_gb must be an integer in [1, 64]")
    spec = load_task(Path(task_dir))
    if spec.kind not in ROUTING_SOURCE_TASK_KINDS:
        raise ValueError("candidate images are only supported for routing-source tasks")
    source = Path(source).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("candidate image output must be outside the source checkout")
    if compiler_cache_dir is not None:
        cache = Path(compiler_cache_dir).expanduser().absolute()
        if any(path.is_symlink() for path in (cache, *cache.parents)):
            raise ValueError("compiler cache path cannot contain symlinks")
        cache = cache.resolve()
        for protected in (source, output, Path(candidate_dir).expanduser().resolve(),
                          Path(task_dir).expanduser().resolve(), ROOT):
            if cache == protected or cache in protected.parents or protected in cache.parents:
                raise ValueError("compiler cache must be separate from source, task, candidate and attempt output")
        compiler_cache_dir = cache
    candidate = load_candidate(Path(candidate_dir))
    verified = verify_candidate(spec, candidate, source)
    patch = candidate.get("source_patch", "")
    if not patch or not any(path.startswith(OPENROAD_SUBMODULE + "/")
                            for path in verified["patch"]["paths"]):
        raise ValueError("routing-source image requires an embedded verified OpenROAD patch")
    # Clone committed state only; refusing dirty inputs prevents a fingerprint
    # from describing different bytes than the adapter originally inspected.
    status = subprocess.run(["git", "-C", str(source), "status", "--porcelain",
                             "--untracked-files=no"], check=True, capture_output=True, text=True)
    if status.stdout.strip():
        raise WorkspaceError("candidate image source must be a clean pinned checkout")
    revisions = {}
    for key, repository in (("source_commit", source),
                            ("openroad_commit", source / OPENROAD_SUBMODULE)):
        revision = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise WorkspaceError("cannot record pinned build source revision: " + key)
        revisions[key] = revision
    base_id = _local_image_id(base_image)
    if base_runtime_equivalence is not None:
        from local_bench.runtime_equivalence import validate_base_equivalence
        base_runtime_equivalence = validate_base_equivalence(base_runtime_equivalence, base_id,
            _run_json(['docker', 'image', 'inspect', base_id]))
    dockerfile = _dockerfile(base_id, threads, build_cache)
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    with _phase(output, "source_clone"):
        isolated_clone(source, workspace, mutable_submodules=[OPENROAD_SUBMODULE])
    patch_path = output / "candidate.patch"
    patch_path.write_text(patch, encoding="utf-8")
    (output / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    apply_candidate_patch(workspace, patch_path)
    openroad = workspace / OPENROAD_SUBMODULE
    fingerprint = openroad_source_fingerprint(openroad)
    context = output / "context"
    context.mkdir()
    # Git metadata/alternates can reveal host paths and are not build inputs.
    shutil.copytree(openroad, context / "openroad", symlinks=True,
                    ignore=shutil.ignore_patterns(".git"))
    if openroad_source_fingerprint(context / "openroad") != fingerprint:
        raise WorkspaceError("candidate build context differs from verified source")
    builder = ROOT / "scripts" / "build_openroad_candidate.sh"
    shutil.copy2(builder, context / builder.name)
    (context / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (context / "seal_runtime.py").write_text(_SEAL_SCRIPT, encoding="utf-8")
    template = {
        "schema_version": 1, "kind": "routing-source-installed-openroad",
        **revisions,
        "task": spec.task_id, "candidate_sha256": _canonical_digest(candidate),
        "source_fingerprint": fingerprint, "base_image_id": base_id,
        "binary_path": str(PREBUILT_OPENROAD_BINARY),
        "builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "dockerfile_sha256": hashlib.sha256(dockerfile.encode()).hexdigest(),
        "build_threads": threads, "build_cpu_quota": CPU_QUOTA,
        "build_memory": MEMORY_LIMIT, "build_network": "none",
        "build_cache_policy": build_cache,
        "compilation_mode": "shared-ccache" if compiler_cache_dir else "private-cold-build",
    }
    if base_runtime_equivalence is not None:
        template['base_runtime_equivalence'] = base_runtime_equivalence
    package_context = context
    if compiler_cache_dir is not None:
        from scripts.openroad_compiler_cache import compile_cached
        with _phase(output, "cached_compilation"):
            compiled = compile_cached(base_image=base_id, source_tree=context / "openroad",
                builder=builder, output=output / "compiler-cache-attempt", cache_root=compiler_cache_dir,
                threads=threads, max_size_gb=compiler_cache_gb)
        binary = Path(compiled["binary_path"])
        binary.resolve(strict=True).relative_to((output / "compiler-cache-attempt").resolve())
        if binary.is_symlink() or not binary.is_file():
            raise WorkspaceError("cached compiler returned an invalid artifact")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != compiled["binary_sha256"]:
            raise WorkspaceError("cached compiler binary hash changed before packaging")
        if compiled.get("source_fingerprint") != fingerprint:
            raise WorkspaceError("cached compiler source differs from the verified candidate")
        template.update({"compiler_cache_key": compiled["cache_key"],
                         "compiler_cache_identity": compiled["cache_identity"],
                         "compiler_cache_max_size_gb": compiler_cache_gb,
                         "build_source_fingerprint": compiled["build_source_fingerprint"],
                         "source_generation": compiled["source_generation"],
                         "binary_sha256": compiled["binary_sha256"]})
        package_context = output / "package-context"
        package_context.mkdir()
        shutil.copy2(binary, package_context / "openroad")
        (package_context / "runtime.json").write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
        dockerfile = _package_dockerfile(base_id)
        template["dockerfile_sha256"] = hashlib.sha256(dockerfile.encode()).hexdigest()
        (package_context / "runtime.json").write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
        (package_context / "Dockerfile").write_text(dockerfile)
        (context / "Dockerfile").write_text(dockerfile)
        (output / "compiler-cache-result.json").write_text(json.dumps(compiled, indent=2, sort_keys=True) + "\n")
    (context / "runtime-template.json").write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
    iidfile = output / "image.id"
    command = ["docker", "build", "--pull=false", "--network=none", "--memory", MEMORY_LIMIT,
               "--cpu-period", "100000", "--cpu-quota", str(CPU_QUOTA),
               "--iidfile", str(iidfile), "-f", str(package_context / "Dockerfile"), str(package_context)]
    (output / "build_command.json").write_text(json.dumps(command, indent=2) + "\n")
    # Docker's legacy builder applies these child-container resource quotas.
    # BuildKit ignores --memory/--cpu-quota; never silently use it here.
    environment = dict(os.environ)
    environment["DOCKER_BUILDKIT"] = "0"
    with _phase(output, "image_packaging" if compiler_cache_dir else "legacy_compile_and_package"):
        with (output / "image-build.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=environment, check=False)
        if process.returncode:
            raise WorkspaceError("candidate Docker image build failed; see {}".format(output / "image-build.log"))
    with _phase(output, "installed_runtime_verification"):
        image_id = _local_image_id(iidfile.read_text().strip())
        installed = _run_json(["docker", "run", "--rm", "--network=none", "--read-only",
                           "--cpus", "1", "--memory", "1g", "--entrypoint", "cat",
                           image_id, str(PREBUILT_OPENROAD_MANIFEST)])
        if (not isinstance(installed, dict)
            or any(installed.get(key) != value for key, value in template.items())
            or not re.fullmatch(r"[0-9a-f]{64}", str(installed.get("binary_sha256", "")))):
            raise WorkspaceError("built image runtime manifest does not match the verified candidate")
        verification = _run_json(["docker", "run", "--rm", "--network=none", "--read-only",
            "--cpus", "1", "--memory", "1g", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", "python3", image_id,
            "-B", "-c", _VERIFY_SCRIPT, str(PREBUILT_OPENROAD_BINARY)])
        if (not isinstance(verification, dict)
            or verification.get("binary_sha256") != installed["binary_sha256"]
            or verification.get("resolved_command") != str(PREBUILT_OPENROAD_BINARY)
            or verification.get("binary_path") != str(PREBUILT_OPENROAD_BINARY)):
            raise WorkspaceError("installed binary verification does not match the image manifest")
    receipt = output / "receipt.json"
    receipt.write_text(json.dumps({**installed, "image_id": image_id,
        "runtime_verification": verification}, indent=2, sort_keys=True) + "\n")
    if build_cache == "discard":
        _finish_build_cache(output, image_id)
    return {"image": image_id, "receipt": str(receipt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True, help="verified materialized candidate directory")
    parser.add_argument("--open3dbench-root", type=Path, required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--build-cache", choices=("discard", "retain"), default="discard",
                        help="Discard successful private build copies, or retain for build debugging")
    parser.add_argument("--compiler-cache-dir", type=Path,
                        help="Opt-in shared ccache root, separate from source and candidate outputs")
    parser.add_argument("--compiler-cache-gb", type=int, default=8,
                        help="ccache maximum size per compatible namespace (1-64 GiB)")
    args = parser.parse_args()
    print(json.dumps(build_candidate_image(args.task, args.candidate, args.open3dbench_root,
                                          args.base_image, args.output_dir, args.threads,
                                          args.build_cache, compiler_cache_dir=args.compiler_cache_dir,
                                          compiler_cache_gb=args.compiler_cache_gb), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
