"""Opt-in, bounded compiler-cache reuse for a private OpenROAD candidate build.

This stages a binary for subsequent *image packaging*, never host evaluation.
The caller retains the scheduler slot and is responsible for final-image checks.
No Docker command is executed on import. Failed attempts are never retried.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

from local_bench.workspace import openroad_source_fingerprint


class CompilerCacheError(RuntimeError):
    pass


SCHEMA = 1
DRIVER = Path(__file__).with_name("build_openroad_cached.py")
HELPER = Path(__file__).resolve()
COUNTER_NOTE = ("Raw shared-namespace ccache statistics, not per-attempt hits or misses; "
                "concurrent writers can contribute to both snapshots.")
# Audited native outputs for this distribution's unchanged CMake recipe:
# CMakeLists.txt configures Version.hh in-source; src/cmake/messages.cmake
# writes message catalogs/markers; embedded OpenSTA has its own catalog rule.
# This is intentionally a finite list, not a wildcard exclusion from identity.
_MESSAGE_DIRECTORIES = (
    "", "src/ant", "src/cgt", "src/cts", "src/cut/src", "src/dbSta", "src/dft",
    "src/dpl", "src/drt", "src/dst", "src/est", "src/exa", "src/fin", "src/gpl",
    "src/grt", "src/gui", "src/ifp", "src/mpl", "src/odb", "src/pad", "src/par",
    "src/pdn", "src/ppl", "src/psm", "src/ram", "src/rcx", "src/rmp", "src/rsz",
    "src/stt", "src/tap", "src/upf", "src/utl", "src/web",
)
GENERATED_SOURCE_PATHS = tuple(sorted({"include/ord/Version.hh", "src/sta/doc/messages.txt"}
    | {str(Path(folder) / name) for folder in _MESSAGE_DIRECTORIES
       for name in ("messages.txt", "messages.checked")}))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_path(value: Path) -> Path:
    """Reject symlinks in every existing ancestor, before resolving anything."""
    path = Path(os.path.abspath(Path(value).expanduser()))
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            raise CompilerCacheError("Symlinked cache/build path: " + str(part))
    # Docker --mount uses comma-separated fields; do not reinterpret a host path.
    if any(c in str(path) for c in (",", "\n", "\r", "\x00")):
        raise CompilerCacheError("Unsupported Docker mount path: " + str(path))
    return path


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _read_regular(path: Path) -> Any:
    _safe_path(path)
    if not path.is_file():
        raise CompilerCacheError("Missing cache metadata: " + str(path))
    try:
        return json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise CompilerCacheError("Corrupt cache metadata: " + str(path)) from error


def _source_inventory(root: Path) -> dict[str, Any]:
    """Exact regular/link/directory records; generated sources remain included."""
    records = {}
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(directories)
        for name in sorted(directories + files):
            path = Path(folder) / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                records[relative] = {"kind": "symlink", "target": os.readlink(path), "mode": mode}
            elif stat.S_ISREG(info.st_mode):
                records[relative] = {"kind": "file", "sha256": _sha(path),
                                     "size_bytes": info.st_size, "mode": mode}
            elif stat.S_ISDIR(info.st_mode):
                records[relative] = {"kind": "directory", "mode": mode}
            else:
                raise CompilerCacheError("Unsupported build-source entry: " + relative)
    return records


def _archive_generated_file(source: Path, destination: Path, expected: dict[str, Any]) -> None:
    """Copy only a validated regular file into a new host-owned destination."""
    source, destination = _safe_path(source), _safe_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _safe_path(destination)  # check the created/existing ancestors again
    flags = getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(source, os.O_RDONLY | flags), "rb") as incoming:
        info = os.fstat(incoming.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != expected["mode"]
                or info.st_size != expected["size_bytes"]):
            raise CompilerCacheError("Generated source no longer matches inventory: " + str(source))
        with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | flags, 0o600), "wb") as outgoing:
            digest = hashlib.sha256()
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(chunk)
                digest.update(chunk)
            if digest.hexdigest() != expected["sha256"]:
                raise CompilerCacheError("Generated source hash changed while archiving")
            os.fchmod(outgoing.fileno(), expected["mode"])


def _audit_generated_sources(source: Path, before: dict[str, Any], output: Path) -> dict[str, Any]:
    after = _source_inventory(source)
    _write_new(output / "build-source-after.json", after)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    unexpected = [name for name in changed if name not in GENERATED_SOURCE_PATHS
                  or after.get(name, {}).get("kind") != "file"
                  or before.get(name, {}).get("kind", "file") != "file"]
    generated = []
    for name in GENERATED_SOURCE_PATHS:
        if name not in after:
            continue
        if after[name]["kind"] != "file":
            unexpected.append(name)
            continue
        # Unlike artifacts/, this tree is never mounted writable in a container.
        destination = output / "generated-source" / name
        _archive_generated_file(source / name, destination, after[name])
        generated.append({"path": name, "before": before.get(name), "after": after[name],
                          "changed": name in changed,
                          "retained_path": str(destination.relative_to(output))})
    audit = {"schema_version": SCHEMA, "policy": "audited-native-generated-source-v1",
             "allowed_paths": list(GENERATED_SOURCE_PATHS), "changed_paths": changed,
             "unexpected_paths": sorted(set(unexpected)), "generated_files": generated,
             "before_inventory_sha256": _digest(before), "after_inventory_sha256": _digest(after)}
    _write_new(output / "source-generation.json", audit)
    if unexpected:
        raise CompilerCacheError("Unexpected source mutation during compilation: " + ", ".join(unexpected))
    return audit


def _event(path: Path, phase: str, state: str, **extra: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema_version": SCHEMA, "phase": phase,
                                 "state": state,
                                 "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                 **extra}, sort_keys=True, allow_nan=False) + "\n")


def _container(base_image: str) -> list[str]:
    return ["docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
            "--cpus", "16", "--memory", "48g", "--memory-swap", "48g",
            "--pids-limit", "512", "--cap-drop=ALL", "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g", "--entrypoint", "python3"]


def _mount(path: Path, destination: str, readonly: bool = False) -> list[str]:
    return ["--mount", f"type=bind,source={path},target={destination}" +
            (",readonly" if readonly else "")]


def _namespace(root: Path, identity: dict[str, Any]) -> tuple[str, Path]:
    """Create immutable metadata under a short host-side initialization lock."""
    root.mkdir(parents=True, exist_ok=True)
    _safe_path(root)
    lock = root / ".namespace.lock"
    _safe_path(lock)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(str(lock), flags, 0o600), "r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        marker = root / "cache-root.json"
        expected = {"schema_version": SCHEMA, "kind": "openroad-compiler-cache-root"}
        if marker.exists():
            if _read_regular(marker) != expected:
                raise CompilerCacheError("Incompatible compiler-cache root metadata")
        elif set(p.name for p in root.iterdir()) != {lock.name}:
            raise CompilerCacheError("Refusing to adopt a nonempty unmarked cache root")
        else:
            _write_new(marker, expected)
        key = _digest(identity)
        namespace = root / key
        _safe_path(namespace)
        metadata = {"schema_version": SCHEMA, "cache_key": key, "identity": identity}
        if namespace.exists():
            if _read_regular(namespace / "identity.json") != metadata:
                raise CompilerCacheError("Incompatible/corrupt cache namespace metadata")
            if {p.name for p in namespace.iterdir()} != {"identity.json", "objects"}:
                raise CompilerCacheError("Unexpected cache namespace contents")
            _safe_path(namespace / "objects")
            if not (namespace / "objects").is_dir():
                raise CompilerCacheError("Missing compiler-cache object directory")
        else:
            namespace.mkdir()
            _write_new(namespace / "identity.json", metadata)
            (namespace / "objects").mkdir()
        return key, namespace / "objects"


def compile_cached(*, base_image: str, source_tree: Path, builder: Path, output: Path,
                   cache_root: Path, threads: int, max_size_gb: int = 8) -> dict[str, Any]:
    """Build once in a bounded container with a fresh object tree and shared ccache.

    ``output`` must be new. ``source_tree`` is a private, verified source copy
    which stays immutable; native generated headers/catalogs are audited in a
    second attempt-owned writable copy under ``build-source``.
    The size limit applies to each compatibility namespace (ccache's normal
    approximate size bound); old namespaces are preserved, never auto-pruned.
    Native/Docker qualification is separate from this helper's offline tests.
    """
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", base_image):
        raise ValueError("Compiler cache requires a pinned local image ID")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
        raise ValueError("threads must be an integer in [1, 16]")
    if isinstance(max_size_gb, bool) or not isinstance(max_size_gb, int) or not 1 <= max_size_gb <= 64:
        raise ValueError("max_size_gb must be an integer in [1, 64]")
    source_tree, builder, output, cache_root = map(_safe_path,
                                                (source_tree, builder, output, cache_root))
    if cache_root == Path(cache_root.anchor):
        raise CompilerCacheError("A filesystem root cannot be a compiler cache")
    for a, b in ((source_tree, output), (source_tree, cache_root), (output, cache_root),
                 (builder, cache_root), (builder, output)):
        if _overlap(a, b):
            raise CompilerCacheError("Cache, source, recipe and attempt paths must not overlap")
    if not source_tree.is_dir() or not (source_tree / "CMakeLists.txt").is_file():
        raise CompilerCacheError("Incomplete private OpenROAD source tree")
    if not builder.is_file() or not DRIVER.is_file():
        raise CompilerCacheError("Missing trusted compiler-cache recipe/driver")
    output.mkdir(parents=True, exist_ok=False)
    events = output / "phases.jsonl"
    started = time.monotonic()
    _event(events, "compiler_cache_attempt", "started")
    try:
        prepare_started = time.monotonic()
        _event(events, "source_materialization", "started")
        source_fingerprint = openroad_source_fingerprint(source_tree)
        _write_new(output / "source-fingerprint.json", source_fingerprint)
        build_source = output / "build-source"
        shutil.copytree(source_tree, build_source, symlinks=True,
                        ignore=shutil.ignore_patterns(".git"))
        if openroad_source_fingerprint(build_source) != source_fingerprint:
            raise CompilerCacheError("Private build-source copy differs from verified source")
        source_inventory = _source_inventory(build_source)
        _write_new(output / "build-source-before.json", source_inventory)
        _event(events, "source_materialization", "completed", exit_code=0,
               duration_seconds=time.monotonic() - prepare_started)
        probe_command = (_container(base_image) + _mount(DRIVER, "/trusted/driver.py", True)
                         + [base_image, "/trusted/driver.py", "--probe"])
        _write_new(output / "probe-command.json", probe_command)
        probe_start = time.monotonic()
        _event(events, "toolchain_probe", "started")
        probe = subprocess.run(probe_command, capture_output=True, text=True, check=False)
        (output / "probe.stdout").write_text(probe.stdout)
        (output / "probe.stderr").write_text(probe.stderr)
        _event(events, "toolchain_probe", "completed", exit_code=probe.returncode,
               duration_seconds=time.monotonic() - probe_start)
        if probe.returncode:
            raise CompilerCacheError("Compiler-cache toolchain probe failed; see probe.stderr")
        try:
            observed = json.loads(probe.stdout)
        except ValueError as error:
            raise CompilerCacheError("Invalid compiler-cache toolchain probe") from error
        if (not isinstance(observed, dict) or observed.get("schema_version") != SCHEMA
                or not isinstance(observed.get("tools"), dict)
                or not observed.get("architecture")):
            raise CompilerCacheError("Incomplete compiler-cache toolchain identity")
        for name in ("c", "cxx", "cmake", "ccache", "make"):
            tool = observed["tools"].get(name, {})
            if not (isinstance(tool, dict) and tool.get("path") and tool.get("version")
                    and re.fullmatch(r"[0-9a-f]{64}", str(tool.get("sha256", "")))):
                raise CompilerCacheError("Missing observed build tool: " + name)
        identity = {"schema_version": SCHEMA, "base_image_id": base_image,
                    "toolchain": observed, "builder_sha256": _sha(builder),
                    "driver_sha256": _sha(DRIVER), "helper_sha256": _sha(HELPER),
                    "mode": "private-tree-shared-ccache",
                    "generated_source_policy": list(GENERATED_SOURCE_PATHS),
                    "ccache_policy": {"max_size_gb": max_size_gb,
                                      "compiler_check": "content", "sloppiness": ""}}
        key, cache_objects = _namespace(cache_root, identity)
        _write_new(output / "cache-identity.json", {"cache_key": key, **identity})
        config = output / "ccache.conf"
        config.write_text(f"max_size = {max_size_gb}G\ncompiler_check = content\n"
                          "sloppiness =\nstats = true\ncompression = true\n"
                          "direct_mode = true\nhard_link = false\nread_only = false\n"
                          "read_only_direct = false\nrecache = false\nrun_second_cpp = true\n"
                          "compiler =\nprefix_command =\n")
        for directory in ("build", "artifacts", "logs"):
            (output / directory).mkdir()
        command = (_container(base_image)
                   + _mount(build_source, "/candidate/source")
                   + _mount(output / "build", "/candidate/build")
                   + _mount(output / "artifacts", "/candidate/artifacts")
                   + _mount(output / "logs", "/candidate/logs")
                   + _mount(cache_objects, "/ccache")
                   + _mount(config, "/trusted/ccache.conf", True)
                   + _mount(builder, "/trusted/build_openroad_candidate.sh", True)
                   + _mount(DRIVER, "/trusted/driver.py", True)
                   + [base_image, "/trusted/driver.py", "--build", str(threads)])
        _write_new(output / "build-command.json", command)
        build_start = time.monotonic()
        _event(events, "build_container", "started", cache_key=key)
        with (output / "build.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        _event(events, "build_container", "completed", exit_code=process.returncode,
               duration_seconds=time.monotonic() - build_start)
        generation = _audit_generated_sources(build_source, source_inventory, output)
        if process.returncode:
            raise CompilerCacheError("Cached OpenROAD build failed; see build.log (no retry)")
        receipt = _read_regular(output / "artifacts" / "binary.json")
        if not isinstance(receipt, dict):
            raise CompilerCacheError("Invalid compiled OpenROAD receipt")
        binary = _safe_path(output / "artifacts" / "openroad")
        if (not binary.is_file() or not stat.S_ISREG(binary.stat().st_mode)
                or not binary.stat().st_mode & 0o111 or binary.stat().st_size == 0
                or receipt.get("binary_sha256") != _sha(binary)
                or receipt.get("size_bytes") != binary.stat().st_size
                or receipt.get("candidate_executed") is not False):
            raise CompilerCacheError("Compiled OpenROAD artifact verification failed")
        if openroad_source_fingerprint(source_tree) != source_fingerprint:
            raise CompilerCacheError("Private OpenROAD source changed during compilation")
        if (_sha(builder) != identity["builder_sha256"] or _sha(DRIVER) != identity["driver_sha256"]
                or _sha(HELPER) != identity["helper_sha256"]):
            raise CompilerCacheError("Trusted build recipe/driver changed during compilation")
        # Never execute candidate code while shared compiler state or provenance
        # mounts are writable. Verification has just two read-only bind mounts.
        verify_command = (_container(base_image)
                          + _mount(DRIVER, "/trusted/driver.py", True)
                          + _mount(binary, "/candidate/openroad", True)
                          + [base_image, "/trusted/driver.py", "--verify"])
        _write_new(output / "verify-command.json", verify_command)
        verify_started = time.monotonic()
        _event(events, "compiled_artifact_verify", "started")
        verification = subprocess.run(verify_command, capture_output=True, text=True, check=False)
        (output / "verify.stdout").write_text(verification.stdout)
        (output / "verify.stderr").write_text(verification.stderr)
        _event(events, "compiled_artifact_verify", "completed", exit_code=verification.returncode,
               duration_seconds=time.monotonic() - verify_started)
        if verification.returncode:
            raise CompilerCacheError("Isolated compiled OpenROAD verification failed; no retry")
        try:
            verified = json.loads(verification.stdout)
        except ValueError as error:
            raise CompilerCacheError("Invalid isolated binary verification evidence") from error
        if (not isinstance(verified, dict) or verified.get("binary_sha256") != receipt["binary_sha256"]
                or verified.get("size_bytes") != receipt["size_bytes"]
                or verified.get("version_exit_code") != 0 or verified.get("dependencies_exit_code") != 0
                or "not found" in str(verified.get("dependencies_stdout", ""))
                or _sha(binary) != receipt["binary_sha256"]):
            raise CompilerCacheError("Isolated compiled OpenROAD evidence mismatch")
        _write_new(output / "binary-verification.json", verified)
        result = {"binary_path": str(binary), "binary_sha256": receipt["binary_sha256"],
                  "source_fingerprint": source_fingerprint,
                  "build_source_fingerprint": openroad_source_fingerprint(build_source),
                  "source_generation": generation,
                  "cache_key": key, "cache_identity": identity,
                  "binary_verification": str(output / "binary-verification.json"),
                  "phases_file": str(events),
                  "container_phases_file": str(output / "logs" / "phases.jsonl"),
                  "stats_before": str(output / "logs" / "ccache-before.txt"),
                  "stats_after": str(output / "logs" / "ccache-after.txt"),
                  "cache_statistics_interpretation": COUNTER_NOTE,
                  "cache_size_scope": "approximate bound per compatibility namespace",
                  "native_evaluation_performed": False}
        _write_new(output / "result.json", result)
        _event(events, "compiler_cache_attempt", "completed", exit_code=0,
               duration_seconds=time.monotonic() - started)
        return result
    except BaseException as error:
        _event(events, "compiler_cache_attempt", "failed",
               duration_seconds=time.monotonic() - started,
               error=f"{type(error).__name__}: {error}")
        raise
