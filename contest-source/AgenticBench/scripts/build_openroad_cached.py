#!/usr/bin/env python3
"""Trusted container-only driver; preserve the original shell build recipe.

CMake interception records separate configure and compile/link wall times. It
does not alter the recipe's arguments or replace the native dependency graph.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_environment() -> dict[str, str]:
    # No host environment is forwarded by the launcher. Keep the pinned image's
    # compiler/dependency settings, but never its user/global ccache shortcuts.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CCACHE_")}
    env["HOME"] = "/tmp/openroad-home"
    env["TMPDIR"] = "/tmp"
    return env


def tool(command: str, env: dict[str, str]) -> dict[str, str]:
    pieces = shlex.split(command)
    if len(pieces) != 1:
        raise RuntimeError("Compiler/tool identity must name one executable")
    executable = shutil.which(pieces[0], path=env.get("PATH"))
    if executable is None:
        raise RuntimeError("Missing required compiler-cache tool: " + command)
    resolved = Path(executable).resolve(strict=True)
    version = subprocess.check_output([str(resolved), "--version"], env=env, text=True,
                                      stderr=subprocess.STDOUT)
    return {"path": str(resolved), "sha256": sha(resolved), "version": version.strip()}


def probe() -> dict:
    env = clean_environment()
    tools = {name: tool(command, env) for name, command in
             (("c", env.get("CC", "cc")), ("cxx", env.get("CXX", "c++")),
              ("cmake", "cmake"), ("ccache", "ccache"), ("make", "make"))}
    tools["ninja"] = tool("ninja", env) if shutil.which("ninja", path=env.get("PATH")) else None
    deps = Path("/etc/openroad_deps_prefixes.txt")
    package_tool = shutil.which("dpkg-query", path=env.get("PATH"))
    packages = None
    if package_tool:
        raw = subprocess.check_output([package_tool, "-W", "-f=${Package}=${Version}\n"], env=env)
        packages = {"format": "dpkg-query-name-version", "sha256": hashlib.sha256(raw).hexdigest()}
    return {"schema_version": 1, "architecture": platform.machine(), "tools": tools,
            "dependency_prefixes": {"path": str(deps), "sha256": sha(deps),
                                    "content": deps.read_text()} if deps.is_file() else None,
            "packages": packages,
            "build_environment": {key: env.get(key) for key in
                                  ("CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
                                   "CMAKE_GENERATOR", "CMAKE_PREFIX_PATH", "LD_LIBRARY_PATH")}}


def event(phase: str, state: str, **extra) -> None:
    with Path("/candidate/logs/phases.jsonl").open("a") as stream:
        stream.write(json.dumps({"schema_version": 1, "phase": phase, "state": state,
                                 "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                 **extra}, sort_keys=True) + "\n")


def cmake(arguments: list[str]) -> int:
    phase = "compile_link" if "--build" in arguments else "configure"
    started = time.monotonic()
    event(phase, "started", arguments=arguments)
    try:
        process = subprocess.run([os.environ["OPENROAD_TRUSTED_CMAKE"], *arguments], check=False)
    except BaseException as error:
        event(phase, "failed", duration_seconds=time.monotonic() - started,
              error=f"{type(error).__name__}: {error}")
        raise
    event(phase, "completed", exit_code=process.returncode,
          duration_seconds=time.monotonic() - started)
    return process.returncode


def statistics(name: str, env: dict[str, str]) -> None:
    result = subprocess.run(["ccache", "-s"], capture_output=True, text=True, env=env, check=False)
    Path("/candidate/logs/" + name + ".txt").write_text(result.stdout + result.stderr)
    event("cache_statistics", "completed", snapshot=name, exit_code=result.returncode,
          interpretation="Shared namespace counters; per-attempt hits/misses unknown")


def stage_artifact() -> None:
    start = time.monotonic()
    event("compiled_artifact_stage", "started")
    try:
        binary = Path("/candidate/build/bin/openroad")
        if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError("Missing or unsafe compiled OpenROAD executable")
        target = Path("/candidate/artifacts/openroad")
        with binary.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        target.chmod(0o755)
        receipt = {"schema_version": 1, "binary_sha256": sha(target),
                   "size_bytes": target.stat().st_size, "candidate_executed": False}
        with Path("/candidate/artifacts/binary.json").open("x") as stream:
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
    except BaseException as error:
        event("compiled_artifact_stage", "failed", duration_seconds=time.monotonic() - start,
              error=f"{type(error).__name__}: {error}")
        raise
    event("compiled_artifact_stage", "completed", exit_code=0,
          duration_seconds=time.monotonic() - start)


def verify_artifact() -> dict:
    """Only called in a separate container with a single read-only binary mount."""
    binary = Path("/candidate/openroad")
    env = clean_environment()
    Path(env["HOME"]).mkdir(exist_ok=True)
    version = subprocess.run([str(binary), "-version"], capture_output=True, text=True,
                             env=env, check=False)
    dependencies = subprocess.run(["ldd", str(binary)], capture_output=True, text=True,
                                  env=env, check=False)
    return {"schema_version": 1, "binary_sha256": sha(binary), "size_bytes": binary.stat().st_size,
            "version_exit_code": version.returncode, "version_stdout": version.stdout,
            "version_stderr": version.stderr, "dependencies_exit_code": dependencies.returncode,
            "dependencies_stdout": dependencies.stdout, "dependencies_stderr": dependencies.stderr}


def build(threads: int) -> int:
    if not 1 <= threads <= 16:
        raise ValueError("Invalid build thread count")
    env = clean_environment()
    Path(env["HOME"]).mkdir()
    Path("/tmp/ccache-tmp").mkdir()
    wrapper = Path("/tmp/openroad-tool-wrapper")
    wrapper.mkdir()
    native_cmake = shutil.which("cmake", path=env.get("PATH"))
    if not native_cmake:
        raise RuntimeError("Missing native CMake")
    launcher = wrapper / "cmake"
    launcher.write_text("#!/bin/sh\nexec python3 /trusted/driver.py --cmake \"$@\"\n")
    launcher.chmod(0o755)
    env.update({"PATH": str(wrapper) + os.pathsep + env["PATH"],
                "OPENROAD_TRUSTED_CMAKE": native_cmake,
                "CCACHE_CONFIGPATH": "/trusted/ccache.conf", "CCACHE_DIR": "/ccache",
                "CCACHE_TEMPDIR": "/tmp/ccache-tmp"})
    configuration = subprocess.run(["ccache", "-p"], capture_output=True, text=True,
                                   env=env, check=False)
    Path("/candidate/logs/ccache-effective-config.txt").write_text(
        configuration.stdout + configuration.stderr)
    if configuration.returncode:
        raise RuntimeError("Cannot verify effective ccache configuration")
    statistics("ccache-before", env)
    try:
        process = subprocess.run(["bash", "/trusted/build_openroad_candidate.sh",
                                  "/candidate/source", "/candidate/build", str(threads)],
                                 env=env, check=False)
        if process.returncode:
            return process.returncode
        stage_artifact()
        return 0
    finally:
        statistics("ccache-after", env)


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--probe":
        print(json.dumps(probe(), sort_keys=True))
        return 0
    if len(sys.argv) == 2 and sys.argv[1] == "--verify":
        print(json.dumps(verify_artifact(), sort_keys=True))
        return 0
    if len(sys.argv) >= 2 and sys.argv[1] == "--cmake":
        return cmake(sys.argv[2:])
    if len(sys.argv) == 3 and sys.argv[1] == "--build":
        return build(int(sys.argv[2]))
    raise SystemExit("Expected --probe, --verify, --build THREADS, or internal --cmake arguments")


if __name__ == "__main__":
    raise SystemExit(main())
