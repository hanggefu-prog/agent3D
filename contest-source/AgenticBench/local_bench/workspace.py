from __future__ import annotations

import json
import hashlib
import re
import shutil
import stat
import subprocess
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence
from .patch_paths import split_patch_sections, strip_section_prefix


class WorkspaceError(RuntimeError):
    pass


OPENROAD_SUBMODULE = "OpenROAD-3D/openroad"
DREAMPLACE_SOURCE = "Place-3D/thirdparty/DREAMPlace"
DREAMPLACE_PYTHON_PREFIX = DREAMPLACE_SOURCE + "/dreamplace/"
PREBUILT_OPENROAD_ENV = "OPEN3D_LOCAL_OPENROAD_PREBUILT_RECEIPT"
PREBUILT_OPENROAD_BINARY = Path("/opt/openagentic3d-routing/bin/openroad")
PREBUILT_OPENROAD_MANIFEST = Path("/opt/openagentic3d-routing/runtime.json")


def openroad_source_fingerprint(source: Path) -> Dict[str, object]:
    """Hash the complete build input, excluding Git administration only.

    Paths and executable bits are included. Links must remain inside the
    source bundle; build contexts must never follow a link into host data.
    Generated/ignored files are not silently omitted from a build identity.
    """
    source = source.resolve()
    if not (source / "CMakeLists.txt").is_file():
        raise WorkspaceError("OpenROAD source is incomplete: {}".format(source))
    records = []
    total_bytes = 0
    for directory, subdirs, files in os.walk(source, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs if name != ".git")
        base = Path(directory)
        links = [name for name in subdirs if (base / name).is_symlink()]
        subdirs[:] = [name for name in subdirs if name not in links]
        for name in sorted([item for item in files if item != ".git"] + links):
            path = base / name
            relative = path.relative_to(source).as_posix()
            if path.is_symlink():
                target = os.readlink(path)
                if Path(target).is_absolute():
                    raise WorkspaceError("absolute OpenROAD source link: {}".format(relative))
                try:
                    path.resolve(strict=True).relative_to(source)
                except (ValueError, OSError, RuntimeError) as error:
                    raise WorkspaceError("escaping/broken OpenROAD source link: {}".format(relative)) from error
                records.append([relative, "link", target])
            elif path.is_file():
                digest = hashlib.sha256()
                size = 0
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        size += len(chunk)
                        digest.update(chunk)
                total_bytes += size
                records.append([relative, "file", bool(path.stat().st_mode & 0o111), size, digest.hexdigest()])
            else:
                raise WorkspaceError("unsupported OpenROAD source entry: {}".format(relative))
    records.sort(key=lambda row: row[0])
    encoded = json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode()
    return {"algorithm": "openroad-tree-v1", "sha256": hashlib.sha256(encoded).hexdigest(),
            "entries": len(records), "bytes": total_bytes}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_docker() -> bool:
    return Path("/.dockerenv").is_file()


def _validated_prebuilt_openroad(source: Path, output_dir: Path, receipt_path: Path) -> Path:
    """Accept only the installed runtime paired with this exact source tree."""
    if not _inside_docker():
        raise WorkspaceError("prebuilt OpenROAD receipts must be consumed inside Docker")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    installed = json.loads(PREBUILT_OPENROAD_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or not isinstance(installed, dict):
        raise WorkspaceError("prebuilt OpenROAD receipt/manifest must be objects")
    for item in (receipt, installed):
        if item.get("schema_version") != 1 or item.get("kind") != "routing-source-installed-openroad":
            raise WorkspaceError("unsupported prebuilt OpenROAD receipt")
    image_id = receipt.get("image_id")
    if (not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
            or os.environ.get("OPEN3DBENCH_IMAGE_ID") != image_id):
        raise WorkspaceError("prebuilt OpenROAD image ID does not match this container")
    # The image-owned manifest has no self-referential image ID. Every one of
    # its fields must be present byte-for-byte in the external launch receipt.
    if any(receipt.get(key) != value for key, value in installed.items()):
        raise WorkspaceError("prebuilt OpenROAD receipt differs from the installed manifest")
    fingerprint = openroad_source_fingerprint(source)
    if installed.get("source_fingerprint") != fingerprint:
        raise WorkspaceError("prebuilt OpenROAD source fingerprint does not match candidate")
    binary = PREBUILT_OPENROAD_BINARY
    if installed.get("binary_path") != str(binary):
        raise WorkspaceError("prebuilt OpenROAD binary is not the installed routing runtime")
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise WorkspaceError("prebuilt OpenROAD installed binary is missing or not executable")
    expected_hash = installed.get("binary_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise WorkspaceError("prebuilt OpenROAD binary hash is invalid")
    if _sha256_file(binary) != expected_hash:
        raise WorkspaceError("prebuilt OpenROAD binary hash does not match receipt")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "openroad-prebuilt-receipt.json").write_text(
        json.dumps({**receipt, "verification": "passed"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "openroad-build.log").write_text(
        "No runtime build: verified candidate source and image-installed OpenROAD.\n"
        "Image: {}\nReceipt: {}\nBinary SHA256: {}\n".format(image_id, receipt_path, expected_hash),
        encoding="utf-8",
    )
    return binary


def _git_safe_directories(source: Path) -> tuple[Path, ...]:
    """Return exact Git paths needed for a normal checkout or worktree."""

    source = source.resolve()
    dotgit = source / ".git"
    trusted = [source]
    if dotgit.is_dir():
        trusted.append(dotgit.resolve())
    elif dotgit.is_file():
        marker = dotgit.read_text(encoding="utf-8", errors="replace").strip()
        prefix = "gitdir:"
        if not marker.lower().startswith(prefix):
            raise WorkspaceError("unsupported worktree .git pointer: {}".format(dotgit))
        gitdir = Path(marker[len(prefix) :].strip()).expanduser()
        if not gitdir.is_absolute():
            gitdir = dotgit.parent / gitdir
        gitdir = gitdir.resolve()
        if not gitdir.is_dir():
            raise WorkspaceError("worktree gitdir is missing: {}".format(gitdir))
        trusted.append(gitdir)
    return tuple(dict.fromkeys(trusted))


def _clone_local_with_gitlinks(source: Path, destination: Path, environment, config_path: Path) -> None:
    """Clone exact initialized gitlinks from local checkouts, never remote URLs."""
    source = source.resolve(strict=True)
    for trusted in _git_safe_directories(source):
        subprocess.run(['git', 'config', '--file', str(config_path), '--add',
                        'safe.directory', str(trusted)], check=True, capture_output=True, text=True)
    env = {**environment, 'GIT_ALLOW_PROTOCOL': 'file', 'GIT_CONFIG_NOSYSTEM': '1'}
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], env=env)
    if not (source / '.git').exists() or Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve() != source:
        raise WorkspaceError('An initialized local dependency checkout is required: ' + str(source))
    revision = git('rev-parse', 'HEAD').decode().strip()
    rows = git('ls-tree', '-r', '-z', 'HEAD').split(b'\0')
    children = []
    for row in rows:
        if not row:
            continue
        metadata, name = row.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        if mode != '160000':
            continue
        relative = Path(os.fsdecode(name))
        if relative.is_absolute() or '..' in relative.parts:
            raise WorkspaceError('Unsafe local gitlink path')
        child = source / relative
        if any(p.is_symlink() for p in (child, *child.parents)) or not (child / '.git').exists():
            raise WorkspaceError('Missing initialized local gitlink: ' + str(child))
        for trusted in _git_safe_directories(child):
            subprocess.run(['git', 'config', '--file', str(config_path), '--add',
                            'safe.directory', str(trusted)], check=True, capture_output=True, text=True)
        actual = subprocess.check_output(['git', '-C', str(child), 'rev-parse', 'HEAD'], env=env).decode().strip()
        if actual != oid:
            raise WorkspaceError('Local dependency differs from its parent gitlink: ' + str(child))
        children.append(relative)
    proc = subprocess.run(['git', 'clone', '--shared', '--no-hardlinks', str(source), str(destination)],
                          env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        raise WorkspaceError('Isolated local dependency clone failed:\n' + proc.stdout)
    cloned = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], env=env).decode().strip()
    if cloned != revision:
        raise WorkspaceError('Local clone changed its pinned revision')
    for relative in children:
        _clone_local_with_gitlinks(source / relative, destination / relative, env, config_path)


def isolated_clone(
    source: Path,
    destination: Path,
    *,
    mutable_submodules: Sequence[str] = (),
    include_worktree: bool = False,
) -> Path:
    """Create a disposable checkout without modifying the source repository."""

    source = source.resolve()
    if destination.exists():
        raise WorkspaceError("workspace already exists: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").exists() and not (source / ".git").is_symlink():
        return _copy_source_archive(source, destination)
    # Benchmark sources are commonly bind-mounted into a container under a
    # different numeric UID. ``git clone`` starts a local upload-pack process,
    # and command-scope ``git -c safe.directory=...`` is not inherited by that
    # child on all supported Git releases. Give the complete clone process a
    # private global config that trusts only this explicitly supplied source.
    config_fd, config_name = tempfile.mkstemp(
        prefix=".openagentic3d-gitconfig-", dir=str(destination.parent)
    )
    os.close(config_fd)
    config_path = Path(config_name)
    try:
        for trusted in _git_safe_directories(source):
            configured = subprocess.run(
                [
                    "git",
                    "config",
                    "--file",
                    str(config_path),
                    "--add",
                    "safe.directory",
                    str(trusted),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if configured.returncode:
                raise WorkspaceError(
                    "isolated clone trust configuration failed:\n{}".format(
                        configured.stdout
                    )
                )
        clone_env = os.environ.copy()
        clone_env["GIT_CONFIG_GLOBAL"] = str(config_path)
        proc = subprocess.run(
            [
                "git",
                "clone",
                "--shared",
                "--no-hardlinks",
                str(source),
                str(destination),
            ],
            env=clone_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    finally:
        config_path.unlink(missing_ok=True)
    if proc.returncode:
        raise WorkspaceError("isolated clone failed:\n{}".format(proc.stdout))

    # Most tasks need only read-only OpenROAD content. A source-evolution task
    # requests an isolated local clone so applying its nested patch can never
    # mutate the dependency checkout through the usual symlink.
    source_submodule = source / "OpenROAD-3D" / "openroad"
    clone_submodule = destination / "OpenROAD-3D" / "openroad"
    independent_openroad = (source_submodule / ".git").exists()
    if (
        source_submodule.is_dir()
        and independent_openroad
        and OPENROAD_SUBMODULE in set(mutable_submodules)
    ):
        if clone_submodule.exists() or clone_submodule.is_symlink():
            if clone_submodule.is_symlink() or clone_submodule.is_file():
                clone_submodule.unlink()
            else:
                shutil.rmtree(clone_submodule)
        clone_submodule.parent.mkdir(parents=True, exist_ok=True)
        nested_fd, nested_name = tempfile.mkstemp(
            prefix=".openagentic3d-openroad-gitconfig-",
            dir=str(destination.parent),
        )
        os.close(nested_fd)
        nested_config = Path(nested_name)
        try:
            for trusted in _git_safe_directories(source_submodule):
                configured = subprocess.run(
                    [
                        "git", "config", "--file", str(nested_config),
                        "--add", "safe.directory", str(trusted),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if configured.returncode:
                    raise WorkspaceError(
                        "OpenROAD clone trust configuration failed:\n{}".format(
                            configured.stdout
                        )
                    )
            nested_environment = os.environ.copy()
            nested_environment["GIT_CONFIG_GLOBAL"] = str(nested_config)
            _clone_local_with_gitlinks(source_submodule, clone_submodule,
                                      nested_environment, nested_config)
        finally:
            nested_config.unlink(missing_ok=True)
    elif (
        source_submodule.is_dir()
        and independent_openroad
        and not clone_submodule.exists()
    ):
        clone_submodule.symlink_to(source_submodule, target_is_directory=True)
    if include_worktree:
        copied, removed = _overlay_worktree_snapshot(source, destination)
        print(
            "[workspace] captured current working tree in isolated source: "
            "{} files copied, {} files removed".format(copied, removed),
            flush=True,
        )
    return destination


def _copy_source_archive(source: Path, destination: Path) -> Path:
    """Materialize an unpacked release without inventing Git provenance.

    Every regular source file is copied independently. Relative links must
    resolve inside the package; sockets, devices and external links are refused.
    The receipt describes the actual copied bytes, not a fabricated revision.
    """
    if not source.is_dir() or source == destination or source in destination.resolve().parents:
        raise WorkspaceError("Invalid source archive or destination")
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo")
    records = []
    for directory, folders, files in os.walk(source, followlinks=False):
        base = Path(directory)
        ignored = ignore(directory, folders + files)
        folders[:] = sorted(name for name in folders if name not in ignored)
        links = [name for name in folders if (base / name).is_symlink()]
        folders[:] = [name for name in folders if name not in links]
        for name in sorted([name for name in files if name not in ignored] + links):
            path = base / name
            relative = path.relative_to(source).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                target = os.readlink(path)
                try:
                    if Path(target).is_absolute():
                        raise ValueError("absolute link")
                    path.resolve(strict=True).relative_to(source)
                except (ValueError, OSError, RuntimeError) as error:
                    raise WorkspaceError("Unsafe source archive link: " + relative) from error
                records.append([relative, "link", target])
            elif stat.S_ISREG(mode):
                records.append([relative, "file", bool(mode & 0o111), _sha256_file(path)])
            else:
                raise WorkspaceError("Unsupported source archive entry: " + relative)
    shutil.copytree(source, destination, symlinks=True, copy_function=shutil.copy2,
                    ignore=ignore)
    _make_copied_directories_writable(destination)
    for record in records:
        path = destination / record[0]
        if record[1] == "file":
            if _sha256_file(path) != record[3]:
                raise WorkspaceError("Source archive changed during copy: " + record[0])
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        elif os.readlink(path) != record[2]:
            raise WorkspaceError("Source archive link changed during copy: " + record[0])
    encoded = json.dumps(sorted(records), separators=(",", ":"), ensure_ascii=True).encode()
    (destination.parent / (destination.name + ".archive-source.json")).write_text(
        json.dumps({"kind": "source-archive", "git_revision": None,
                    "files": len(records), "tree_sha256": hashlib.sha256(encoded).hexdigest()},
                   indent=2, sort_keys=True) + "\n")
    print("[workspace] copied unpacked source archive: {} entries".format(len(records)), flush=True)
    return destination


def _overlay_worktree_snapshot(source: Path, destination: Path) -> tuple[int, int]:
    """Copy non-committed outer-repository state into an isolated clone.

    The normal benchmark path intentionally clones committed Git state only.
    Manual experiments opt into this snapshot so edits made before ``init`` are
    reproducible inside their private source tree. Ignored files and nested
    repository working trees remain outside this outer-repository contract.
    """

    def git_paths(*arguments: str) -> list[Path]:
        completed = subprocess.run(
            ["git", "-C", str(source), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode:
            raise WorkspaceError(
                "cannot snapshot working tree with git {}: {}".format(
                    " ".join(arguments),
                    completed.stderr.decode("utf-8", errors="replace").strip(),
                )
            )
        paths = []
        for raw in completed.stdout.split(b"\0"):
            if not raw:
                continue
            relative = Path(raw.decode("utf-8", errors="surrogateescape"))
            if relative.is_absolute() or ".." in relative.parts:
                raise WorkspaceError(
                    "unsafe path in working-tree snapshot: {}".format(relative)
                )
            paths.append(relative)
        return paths

    changed = git_paths(
        "diff", "--no-renames", "--name-only", "-z",
        "--diff-filter=ACMRTUXB", "HEAD", "--",
    )
    changed.extend(git_paths("ls-files", "--others", "--exclude-standard", "-z"))
    deleted = git_paths(
        "diff", "--no-renames", "--name-only", "-z",
        "--diff-filter=D", "HEAD", "--",
    )
    copied = 0
    for relative in dict.fromkeys(changed):
        source_path = source / relative
        destination_path = destination / relative
        # Nested repositories/submodules own their own snapshot contract.
        if source_path.is_dir() and not source_path.is_symlink():
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists() or destination_path.is_symlink():
            if destination_path.is_dir() and not destination_path.is_symlink():
                shutil.rmtree(destination_path)
            else:
                destination_path.unlink()
        if source_path.is_symlink():
            destination_path.symlink_to(os.readlink(source_path))
        elif source_path.is_file():
            shutil.copy2(source_path, destination_path)
        else:
            continue
        copied += 1

    removed = 0
    for relative in dict.fromkeys(deleted):
        destination_path = destination / relative
        if destination_path.is_symlink() or destination_path.is_file():
            destination_path.unlink()
            removed += 1
        elif destination_path.is_dir():
            shutil.rmtree(destination_path)
            removed += 1
    return copied, removed


def apply_candidate_patch(repo: Path, patch_path: Path) -> Dict[str, str]:
    if not patch_path.is_file():
        return {"status": "absent"}
    patch_text = patch_path.read_text(encoding="utf-8")
    nested_prefix = OPENROAD_SUBMODULE + "/"
    outer_text, nested_text = _partition_patch(patch_text, nested_prefix)
    if nested_text:
        nested_repo = repo / OPENROAD_SUBMODULE
        if not nested_repo.is_dir() or nested_repo.is_symlink():
            raise WorkspaceError(
                "OpenROAD source patch requires an isolated mutable submodule"
            )
        # Current Open3DBench releases vendor OpenROAD in the main repository,
        # while older research checkouts keep it as an independent nested Git
        # repository.  A vendored checkout must receive the original full-path
        # patch at the outer root; an independent checkout receives the
        # prefix-stripped section at its own root.
        if not (nested_repo / ".git").exists():
            return _apply_patch_file(repo, patch_path)
        modes = []
        if outer_text.strip():
            modes.append(_apply_patch_text(repo, outer_text, patch_path.parent, "outer"))
        modes.append(
            _apply_patch_text(
                nested_repo, nested_text, patch_path.parent, "openroad"
            )
        )
        return {
            "status": "applied",
            "patch": str(patch_path),
            "mode": "+".join(modes),
        }
    return _apply_patch_file(repo, patch_path)


def materialize_dreamplace_source_tree(
    source_repo: Path,
    candidate_repo: Path,
    changed_paths: Sequence[str],
) -> Optional[Path]:
    """Replace the tracked DREAMPlace link only inside a candidate checkout."""

    if not any(path.startswith(DREAMPLACE_PYTHON_PREFIX) for path in changed_paths):
        return None
    source = source_repo / DREAMPLACE_SOURCE
    destination = candidate_repo / DREAMPLACE_SOURCE
    resolved_source = source.resolve()
    if not resolved_source.is_dir():
        raise FileNotFoundError(
            "patched DREAMPlace source tree is unavailable: {}".format(source)
        )
    if destination.is_symlink():
        destination.unlink()
    elif destination.is_dir():
        return destination
    elif destination.exists():
        raise WorkspaceError(
            "candidate DREAMPlace path is neither a directory nor symlink: {}".format(
                destination
            )
        )
    shutil.copytree(
        resolved_source,
        destination,
        symlinks=True,
        # This tree is mutable by definition. Hard links would let an editor
        # or patch implementation that writes in place modify the shared
        # dependency behind the tracked symlink.
        copy_function=shutil.copy2,
    )
    _make_copied_directories_writable(destination)
    for changed_path in changed_paths:
        if changed_path.startswith(DREAMPLACE_PYTHON_PREFIX):
            copied = destination / "dreamplace" / changed_path[len(DREAMPLACE_PYTHON_PREFIX):]
            if copied.exists() or copied.is_symlink():
                _make_independent_copy_writable(destination, copied)
    return destination


def _partition_patch(text: str, nested_prefix: str) -> tuple[str, str]:
    """Split difflib/git unified sections and strip a nested-repo prefix."""

    outer: list[str] = []
    nested: list[str] = []
    for path, section in split_patch_sections(text):
        if path.startswith(nested_prefix):
            nested.append(strip_section_prefix(section, nested_prefix))
        else:
            outer.append(section)
    return "".join(outer), "".join(nested)


def _apply_patch_text(repo: Path, text: str, directory: Path, label: str) -> str:
    fd, name = tempfile.mkstemp(prefix=".candidate-{}-".format(label), suffix=".patch", dir=str(directory))
    os.close(fd)
    path = Path(name)
    try:
        path.write_text(text, encoding="utf-8")
        result = _apply_patch_file(repo, path)
        return str(result["mode"])
    finally:
        path.unlink(missing_ok=True)


def _apply_patch_file(repo: Path, patch_path: Path) -> Dict[str, str]:
    mode = "strict"
    apply_options: list[str] = []
    check = _git_apply_check(repo, patch_path, apply_options)
    if check.returncode:
        # LLMs sometimes emit an otherwise exact hunk with context on only one
        # side of an insertion. Git rejects that form by default even when all
        # supplied context matches byte-for-byte. Scope and size policy has
        # already run before this function, so permit Git's explicit
        # zero-context parser as a deterministic fallback. This is not fuzzy
        # matching: a stale line location or mismatched supplied context still
        # fails the second check.
        fallback_options = ["--unidiff-zero"]
        fallback = _git_apply_check(repo, patch_path, fallback_options)
        if fallback.returncode:
            raise WorkspaceError(
                "candidate patch does not apply cleanly:\n{}".format(check.stdout)
            )
        mode = "unidiff-zero"
        apply_options = fallback_options
    apply = subprocess.run(
        ["git", "apply", *apply_options, str(patch_path)],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if apply.returncode:
        raise WorkspaceError("candidate patch application failed:\n{}".format(apply.stdout))
    return {"status": "applied", "patch": str(patch_path), "mode": mode}


def _git_apply_check(
    repo: Path, patch_path: Path, options: list[str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", "--check", *options, str(patch_path)],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def validate_candidate_workspace(
    repo: Path,
    changed_paths: list[str],
    python_executable: str,
) -> Dict[str, object]:
    """Run cheap deterministic checks before launching an EDA stage."""

    if not changed_paths:
        return {"status": "skipped", "checks": []}
    checks = []
    nested_prefix = OPENROAD_SUBMODULE + "/"
    outer_paths = [path for path in changed_paths if not path.startswith(nested_prefix)]
    nested_paths = [path[len(nested_prefix) :] for path in changed_paths if path.startswith(nested_prefix)]
    diff = subprocess.run(
        ["git", "diff", "--check", "--", *outer_paths],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if diff.returncode:
        raise WorkspaceError("candidate patch fails git diff --check:\n{}".format(diff.stdout))
    checks.append("git-diff-check")

    if nested_paths:
        nested_repo = repo / OPENROAD_SUBMODULE
        nested_diff = subprocess.run(
            ["git", "diff", "--check", "--", *nested_paths],
            cwd=str(nested_repo),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if nested_diff.returncode:
            raise WorkspaceError(
                "candidate OpenROAD patch fails git diff --check:\n{}".format(
                    nested_diff.stdout
                )
            )
        checks.append("openroad-git-diff-check")

    python_paths = [str(repo / path) for path in changed_paths if path.endswith(".py")]
    if python_paths:
        compile_check = subprocess.run(
            [python_executable, "-m", "py_compile", *python_paths],
            cwd=str(repo),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if compile_check.returncode:
            raise WorkspaceError(
                "candidate Python source does not compile:\n{}".format(compile_check.stdout)
            )
        checks.append("python-py-compile")

    shell_paths = [str(repo / path) for path in changed_paths if path.endswith(".sh")]
    for path in shell_paths:
        shell_check = subprocess.run(
            ["bash", "-n", path],
            cwd=str(repo),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if shell_check.returncode:
            raise WorkspaceError(
                "candidate shell source does not parse:\n{}".format(shell_check.stdout)
            )
    if shell_paths:
        checks.append("bash-syntax")
    return {"status": "passed", "checks": checks}


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _link_large_frozen_file_or_copy(source: str, destination: str) -> str:
    """Hard-link only bulky frozen artifacts; copy mutable metadata normally."""

    try:
        if os.stat(source).st_size >= 1024 * 1024:
            os.link(source, destination)
            return destination
    except OSError:
        pass
    return shutil.copy2(source, destination)


def _make_copied_directories_writable(root: Path) -> None:
    """Restore owner-write only on directories of a just-created copytree.

    copytree preserves sealed source modes. Never traverse copied symlinks or
    chmod files here: a copy may intentionally contain hard-linked artifacts.
    """
    if root.is_symlink():
        raise WorkspaceError("copied tree root must not be a symlink: {}".format(root))
    for directory, subdirs, _ in os.walk(root, followlinks=False):
        current = Path(directory)
        subdirs[:] = [name for name in subdirs if not (current / name).is_symlink()]
        current.chmod(stat.S_IMODE(current.stat().st_mode) | stat.S_IWUSR)


def _check_copied_path(root: Path, path: Path) -> None:
    """Reject traversal or symlinks before modifying a path in a new copy."""
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise WorkspaceError("mutable copy escapes its new tree: {}".format(path)) from error
    if ".." in relative.parts:
        raise WorkspaceError("mutable copy escapes its new tree: {}".format(path))
    current = root
    for part in ("", *relative.parts):
        if part:
            current = current / part
        if current.is_symlink():
            raise WorkspaceError("mutable copied metadata/source must not be a symlink: {}".format(current))


def _make_independent_copy_writable(root: Path, path: Path) -> None:
    """Allow an explicitly mutable, newly copied file without unsealing aliases."""
    _check_copied_path(root, path)
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise WorkspaceError("mutable metadata/source must be an independent copied file: {}".format(path))
    path.chmod(stat.S_IMODE(metadata.st_mode) | stat.S_IWUSR)


def prepare_dreamplace_overlay(
    source_repo: Path,
    candidate_repo: Path,
    output_dir: Path,
    changed_paths: list[str],
) -> Optional[Path]:
    """Create a candidate-local DREAMPlace runtime for patched Python files.

    Open3DBench normally runs the installed DREAMPlace package. Merely patching
    its source tree would therefore not exercise a BasicPlace/PlaceObj change.
    This overlay reuses immutable native objects through hard links when
    possible, then replaces only patched Python files with candidate versions.
    """

    prefix = DREAMPLACE_PYTHON_PREFIX
    relative_paths = [path[len(prefix) :] for path in changed_paths if path.startswith(prefix)]
    if not relative_paths:
        return None
    configured = os.environ.get("OPEN3D_LOCAL_DREAMPLACE_INSTALL")
    source_install = (
        Path(configured).expanduser().resolve()
        if configured
        else source_repo / "Place-3D" / "build" / "dreamplace" / "install"
    )
    if not (source_install / "dreamplace").is_dir():
        raise FileNotFoundError(
            "patched DREAMPlace source requires an installed runtime; set "
            "OPEN3D_LOCAL_DREAMPLACE_INSTALL or build Place-3D first: {}".format(source_install)
        )
    overlay = output_dir / "runtime" / "dreamplace-install"
    shutil.copytree(
        source_install,
        overlay,
        symlinks=True,
        copy_function=_link_or_copy,
    )
    _make_copied_directories_writable(overlay)
    for relative in relative_paths:
        source = candidate_repo / prefix / relative
        target = overlay / "dreamplace" / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        _check_copied_path(overlay, target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(source, target)
        _make_independent_copy_writable(overlay, target)
    return overlay


def prepare_openroad_runtime(
    candidate_repo: Path,
    output_dir: Path,
    changed_paths: list[str],
) -> Optional[Path]:
    """Build and return the isolated OpenROAD binary for a C++ candidate.

    The default uses OpenROAD's maintained Build.sh inside the evaluation
    environment. Sites with a container wrapper may set
    OPEN3D_LOCAL_OPENROAD_BUILDER to an executable accepting
    ``SOURCE_ROOT BUILD_DIR THREADS`` and producing the same standard binary
    layout. The dependency checkout and any shared baseline build stay
    untouched.
    """

    prefix = OPENROAD_SUBMODULE + "/"
    if not any(path.startswith(prefix) for path in changed_paths):
        if os.environ.get(PREBUILT_OPENROAD_ENV):
            raise WorkspaceError("prebuilt OpenROAD receipt supplied without an OpenROAD source candidate")
        return None
    source = candidate_repo / OPENROAD_SUBMODULE
    if source.is_symlink() or not (source / "CMakeLists.txt").is_file():
        raise FileNotFoundError(
            "patched OpenROAD source requires an isolated complete submodule: {}".format(
                source
            )
        )
    receipt = os.environ.get(PREBUILT_OPENROAD_ENV)
    if receipt:
        return _validated_prebuilt_openroad(source, output_dir, Path(receipt).expanduser().resolve())
    build_dir = output_dir / "runtime" / "openroad-build"
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        threads = int(os.environ.get("OPEN3D_LOCAL_OPENROAD_BUILD_THREADS", "8"))
    except ValueError as error:
        raise ValueError("OPEN3D_LOCAL_OPENROAD_BUILD_THREADS must be an integer") from error
    if threads < 1 or threads > 64:
        raise ValueError("OPEN3D_LOCAL_OPENROAD_BUILD_THREADS must be in [1, 64]")
    builder_value = os.environ.get("OPEN3D_LOCAL_OPENROAD_BUILDER")
    if builder_value:
        builder = Path(builder_value).expanduser().resolve()
        if not builder.is_file() or not os.access(builder, os.X_OK):
            raise FileNotFoundError(
                "OPEN3D_LOCAL_OPENROAD_BUILDER is not executable: {}".format(builder)
            )
        command = [str(builder), str(source), str(build_dir), str(threads)]
    else:
        build_script = source / "etc" / "Build.sh"
        if not build_script.is_file():
            raise FileNotFoundError(build_script)
        command = [
            "bash",
            str(build_script),
            "-dir={}".format(build_dir),
            "-threads={}".format(threads),
            "-no-gui",
            "-no-tests",
        ]
    build_log = output_dir / "openroad-build.log"
    with build_log.open("w", encoding="utf-8") as output:
        build = subprocess.run(
            command,
            cwd=str(source),
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if build.returncode:
        raise WorkspaceError(
            "candidate OpenROAD build failed with code {}; see {}".format(
                build.returncode, build_log
            )
        )
    candidates = (build_dir / "bin" / "openroad", build_dir / "src" / "openroad")
    executable = next(
        (path for path in candidates if path.is_file() and os.access(path, os.X_OK)),
        None,
    )
    if executable is None:
        raise WorkspaceError(
            "candidate OpenROAD build produced no executable under {}".format(build_dir)
        )
    return executable.resolve()


def copy_upstream_stages(baseline_run: Path, run_dir: Path, stages: list[str]) -> None:
    stage_dirs = {
        "input": "00_input",
        "floorplan": "10_floorplan",
        "timing_extract": "15_timing_extract",
        "partition": "20_partition",
        "placement": "30_placement",
        "pdn": "40_pdn",
        "hbt": "50_hbt",
        "opt": "60_opt",
        "cts": "70_cts",
        "global_routing": "75_global_route",
        "detailed_routing": "80_route",
        "routing": "80_route",
        "report": "90_report",
        "thermal": "95_thermal",
    }
    link_large_files = os.environ.get("OPEN3D_LINK_FROZEN_UPSTREAM") == "1"
    mutable_metadata = {"design_params.json", "stage_manifest.json"}

    def copy_function(source: str, destination: str) -> str:
        # These controller-owned copies are rewritten after import. Even a
        # large manifest must not share an inode with a sealed reference.
        if link_large_files and Path(source).name not in mutable_metadata:
            return _link_large_frozen_file_or_copy(source, destination)
        return shutil.copy2(source, destination)
    run_dir.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        name = stage_dirs[stage]
        source = baseline_run / name
        target = run_dir / name
        if not source.is_dir():
            raise FileNotFoundError("baseline is missing {} stage: {}".format(stage, source))
        shutil.copytree(
            source,
            target,
            symlinks=True,
            copy_function=copy_function,
        )
        _make_copied_directories_writable(target)
        for metadata_name in sorted(mutable_metadata):
            metadata = target / metadata_name
            if metadata.exists() or metadata.is_symlink():
                _make_independent_copy_writable(target, metadata)
        _rebase_copied_design_params(target)


def _rebase_copied_design_params(stage_dir: Path) -> None:
    """Bind run-owned DREAMPlace inputs to the copied frozen stage."""

    params_path = stage_dir / "design_params.json"
    if not params_path.is_file():
        return
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkspaceError(
            "design_params.json must contain an object: {}".format(params_path)
        )
    local_inputs = {
        "def_input": stage_dir / "design.def",
        "verilog_input": stage_dir / "design.v",
        "sdc_input": stage_dir / "design.sdc",
    }
    changed = False
    for key, path in local_inputs.items():
        resolved = str(path.resolve())
        if path.is_file() and payload.get(key) != resolved:
            payload[key] = resolved
            changed = True
    if changed:
        params_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
