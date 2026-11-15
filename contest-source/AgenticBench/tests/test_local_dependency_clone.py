"""Real local Git repositories; no network, Docker, model or EDA."""
import os
from pathlib import Path
import subprocess
import pytest
from local_bench.workspace import _clone_local_with_gitlinks, WorkspaceError


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT).decode().strip()


def init(path):
    path.mkdir()
    git(path, 'init', '-q')
    git(path, 'config', 'user.name', 'Offline Test')
    git(path, 'config', 'user.email', 'offline@example.invalid')
    (path / 'source.cc').write_text('initial source\n')
    git(path, 'add', '.')
    git(path, 'commit', '-qm', 'initial')
    return path


def source(tmp_path):
    sta = init(tmp_path / 'sta')
    native = init(tmp_path / 'native')
    git(native, '-c', 'protocol.file.allow=always', 'submodule', 'add', str(sta), 'src/sta')
    # An unusable remote URL proves the clone consumes the already initialized
    # local checkout rather than the URL written in .gitmodules.
    git(native, 'config', '-f', '.gitmodules', 'submodule.src/sta.url', 'https://invalid.invalid/never-fetch.git')
    git(native, 'add', '.')
    git(native, 'commit', '-qm', 'dependency')
    config = tmp_path / 'private.gitconfig'
    config.touch()
    return native, config, {**os.environ, 'GIT_CONFIG_GLOBAL': str(config)}


def test_exact_local_gitlink_reused_despite_invalid_remote(tmp_path):
    native, config, env = source(tmp_path)
    destination = tmp_path / 'copy'
    _clone_local_with_gitlinks(native, destination, env, config)
    assert git(native, 'rev-parse', 'HEAD') == git(destination, 'rev-parse', 'HEAD')
    assert git(native / 'src/sta', 'rev-parse', 'HEAD') == git(destination / 'src/sta', 'rev-parse', 'HEAD')
    (destination / 'src/sta/source.cc').write_text('candidate edit')
    assert (native / 'src/sta/source.cc').read_text() == 'initial source\n'
    assert git(native, 'status', '--porcelain') == ''


def test_wrong_local_dependency_revision_fails_before_clone(tmp_path):
    native, config, env = source(tmp_path)
    child = native / 'src/sta'
    git(child, 'config', 'user.name', 'Offline Test')
    git(child, 'config', 'user.email', 'offline@example.invalid')
    (child / 'source.cc').write_text('wrong revision')
    git(child, 'commit', '-qam', 'wrong')
    with pytest.raises(WorkspaceError, match='gitlink'):
        _clone_local_with_gitlinks(native, tmp_path / 'copy', env, config)
    assert not (tmp_path / 'copy').exists()
