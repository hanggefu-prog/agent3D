"""Content-addressed snapshots and host-owned scope verification.

The host explicitly inventories source files. Candidate code never supplies its
own inventory, policy, runtime receipt or metrics verifier. No Git checkout is
reset and no existing snapshot is edited.
"""
import os
from pathlib import Path
import shutil
import stat
import subprocess
from .common import confined, digest, read, sha, write

PROTECTED = ("AgenticBench/", "tests/", "docker/", "open3dbench/",
             "OpenROAD-3D/openroad/src/sta/",
             "OpenROAD-3D/flow/", "Place-3D/src/api.py", "Place-3D/src/validation.py",
             "Place-3D/src/evaluation/", "Place-3D/src/hbt/", "Place-3D/scripts/")
HBT_ALGORITHM = "Place-3D/src/hbt/assignment.py"
NATIVE = (".cc", ".cpp", ".c", ".h", ".hh", ".hpp", ".i", ".cu", ".cuh", ".cmake")


def _under(name, prefix):
    return name == prefix.rstrip("/") or name.startswith(prefix.rstrip("/") + "/")


def inventory(root):
    root = Path(root).resolve(strict=True)
    entries = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute():
                raise ValueError("Absolute source symlink: " + name)
            try:
                path.resolve(strict=True).relative_to(root)
            except (ValueError, RuntimeError, FileNotFoundError) as error:
                raise ValueError("Escaping, cyclic or broken source symlink: " + name) from error
            entries[name] = {"symlink": target}
            continue
        if path.is_dir():
            continue
        st = path.stat()
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("Candidate must contain regular files")
        entries[name] = {"sha256": sha(path), "bytes": st.st_size, "executable": bool(st.st_mode & 0o111)}
    return entries


def verify_scope(before, after, policy):
    if not policy.get("allowed_paths") or not isinstance(policy.get("native"), bool):
        raise ValueError("Host must freeze an explicit edit policy")
    for name in policy["allowed_paths"]:
        if Path(name).is_absolute() or ".." in Path(name).parts or name in ("", "."):
            raise ValueError("Unsafe scope")
    changes = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    native = False
    for name in changes:
        if "symlink" in before.get(name, {}) or "symlink" in after.get(name, {}):
            raise ValueError("Candidate may not alter source symlinks: " + name)
        protected = any(_under(name, p) for p in PROTECTED)
        # Only an exact host grant unlocks the geometry/assignment algorithm.
        # HBT adapters, serialization and broad directory grants stay protected.
        if name == HBT_ALGORITHM and name in policy["allowed_paths"]:
            protected = False
        if protected or not any(_under(name, p) for p in policy["allowed_paths"]):
            raise ValueError("Protected/out-of-scope change: " + name)
        if name not in before and not policy.get("allow_create", False):
            raise ValueError("File creation forbidden")
        if name not in after and not policy.get("allow_delete", False):
            raise ValueError("File deletion forbidden")
        if name in before and name in after and before[name]["executable"] != after[name]["executable"]:
            raise ValueError("File mode change forbidden")
        in_openroad = _under(name, "OpenROAD-3D/openroad")
        compiled = in_openroad or Path(name).suffix in NATIVE or Path(name).name in {"CMakeLists.txt", "Makefile"}
        if compiled and not in_openroad:
            raise ValueError("No installed native runtime builder for this component: " + name)
        native |= compiled
        if compiled and not policy["native"]:
            raise ValueError("Native changes require a candidate image")
    return {"changes": changes, "native_build_required": native, "policy_digest": digest(policy)}


def freeze(source, store, *, paths=None, parent=None, policy=None):
    source, store = Path(source).resolve(strict=True), Path(store).resolve()
    if source == store or source in store.parents or store in source.parents:
        raise ValueError("Snapshot store and source must be separate")
    store.mkdir(parents=True, exist_ok=True)
    # Explicit paths allow tracked-source imports without copying run outputs.
    if paths is None:
        entries = inventory(source)
    else:
        entries = {}
        for name in sorted(set(paths)):
            path = confined(source, name)
            if not path.is_file():
                raise ValueError("Missing source file: " + name)
            entries[name] = {"sha256": sha(path), "bytes": path.stat().st_size,
                             "executable": bool(path.stat().st_mode & 0o111)}
    if not entries:
        raise ValueError("Empty source snapshot")
    audit = None
    if parent:
        parent_data = verify_snapshot(store / parent)
        audit = verify_scope(parent_data["files"], entries, policy)
        from .algorithm_scope import verify_functions
        verify_functions(store / parent / 'tree', source, policy)
        audit["native_build_required"] |= bool((parent_data.get("scope") or {}).get("native_build_required"))
    manifest = {"schema_version": 1, "parent": parent, "files": entries, "scope": audit,
                "policy": policy if parent else None}
    identity = digest(manifest)
    destination = store / identity
    if destination.exists():
        verify_snapshot(destination)
        return identity
    import tempfile
    temporary = Path(tempfile.mkdtemp(prefix=".freeze-", dir=store))
    try:
        tree = temporary / "tree"
        tree.mkdir()
        for name, row in entries.items():
            dst = tree / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if "symlink" in row:
                dst.symlink_to(row["symlink"])
                continue
            src = confined(source, name)
            shutil.copyfile(src, dst)
            dst.chmod(0o555 if row["executable"] else 0o444)
        if inventory(tree) != entries:
            raise ValueError("Source changed while freezing")
        write(temporary / "candidate.json", manifest)
        os.rename(temporary, destination)
    except BaseException:
        # Preserve the failed materialization for diagnosis.
        raise
    return identity


def verify_snapshot(path):
    path = Path(path)
    data = read(path / "candidate.json")
    if digest(data) != path.name or inventory(path / "tree") != data["files"]:
        raise ValueError("Candidate snapshot changed")
    return data


def tracked_paths(source):
    result = subprocess.run(["git", "-C", str(source), "ls-files", "-z", "--recurse-submodules"],
                            capture_output=True, check=True)
    return [p.decode() for p in result.stdout.split(b"\0") if p]


def materialize(snapshot, output):
    verify_snapshot(snapshot)
    output = Path(output)
    if output.exists():
        raise ValueError("Candidate workspace must be new")
    shutil.copytree(Path(snapshot) / "tree", output, symlinks=True)
    for path in output.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
    return output
