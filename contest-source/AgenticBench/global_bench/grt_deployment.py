"""Validate the deployed build dependency closure before spending model or EDA budget."""
import importlib
from pathlib import Path

from grt_bench.common import sha

BUILD_FILES = (
    'scripts/openroad_compiler_cache.py',
    'scripts/build_openroad_cached.py',
    'scripts/build_openroad_candidate.sh',
    'local_bench/workspace.py',
)


def preflight_build(framework):
    framework = Path(framework).resolve(strict=True)
    missing = [name for name in BUILD_FILES if not (framework / name).is_file()]
    if missing:
        raise ValueError('Incomplete build deployment: ' + ', '.join(missing))
    for name in BUILD_FILES:
        path = framework / name
        if path.suffix == '.py':
            compile(path.read_text(), str(path), 'exec')
    for name in ('scripts.openroad_compiler_cache', 'local_bench.workspace'):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != framework / (name.replace('.', '/') + '.py'):
            raise ValueError('Build dependency imported outside frozen deployment: ' + name)
    return {name: sha(framework / name) for name in BUILD_FILES}
