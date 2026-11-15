"""Complete, relocatable identity of engineering evaluation and its runtime.

identity_version 1 (frozen, kept to verify old qualifications) hashed the
engineering package plus *every* .py/.sh/.tcl file in grt_bench, local_bench,
global_bench and scripts, so an unrelated track edit invalidated engineering
receipts. identity_version 2 hashes only what engineering evaluation executes:
engineering_bench/** (.py, .tcl), the repository-local import closure of those
modules (resolved with ast, never by importing), and the explicitly listed
files that engineering code launches by path.
"""
import ast
from pathlib import Path

from .protocol import digest, sha, read_json

IDENTITY_VERSIONS = (1, 2)
CURRENT_IDENTITY_VERSION = 2

# Files that engineering code executes by path rather than by import; each is
# referenced as framework/<path> in engineering_bench.
PATH_REFERENCED = ("scripts/build_openroad_candidate.sh", "global_bench/container_terminal.py")


def _package_files(root):
    return [p for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.tcl') and '__pycache__' not in p.parts]


def _module_file(framework, dotted):
    """Repository-local file for a dotted module name, or None (stdlib/site)."""
    if not dotted:
        return None
    base = framework.joinpath(*dotted.split('.'))
    for candidate in (base.with_suffix('.py'), base / '__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def _package_inits(framework, dotted):
    parts = dotted.split('.')
    result = []
    for n in range(1, len(parts)):
        init = framework.joinpath(*parts[:n]) / '__init__.py'
        if init.is_file():
            result.append(init)
    return result


def _dotted(path, framework):
    parts = list(path.relative_to(framework).with_suffix('').parts)
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def imported_files(path, framework):
    """Repository-local files imported anywhere in one Python source file.

    Handles absolute, relative and flat sibling imports (the physical adapters
    put their own directory on sys.path), including function-level imports.
    """
    path, framework = Path(path), Path(framework)
    tree = ast.parse(path.read_text(), filename=str(path))
    package = _dotted(path, framework)
    if path.name != '__init__.py':
        package = package.rpartition('.')[0]
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split('.') if package else []
                if node.level - 1 > len(parts):
                    continue
                base = '.'.join(parts[:len(parts) - (node.level - 1)])
                module = '.'.join(x for x in (base, node.module) if x)
            else:
                module = node.module or ''
            names.append(module)
            names += [module + '.' + alias.name for alias in node.names if alias.name != '*']
    result = set()
    for name in names:
        found = _module_file(framework, name)
        if found is None and name and '.' not in name:
            sibling = path.parent / (name + '.py')
            found = sibling if sibling.is_file() else None
        if found is not None:
            result.add(found.resolve())
            result.update(p.resolve() for p in _package_inits(framework, name))
    return result


def evaluator_files(root=None):
    """Sorted identity-v2 inventory as framework-relative POSIX paths."""
    root = (Path(root) if root else Path(__file__).resolve().parent).resolve()
    framework = root.parent
    extra = [framework / rel for rel in PATH_REFERENCED]
    missing = [str(p) for p in extra if not p.is_file()]
    if missing:
        raise ValueError('Missing path-referenced engineering runtime: ' + ', '.join(missing))
    pending = [p.resolve() for p in _package_files(root) + extra]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        try:
            path.relative_to(framework)
        except ValueError:
            continue
        seen.add(path)
        if path.suffix == '.py':
            pending += sorted(imported_files(path, framework) - seen)
    return sorted(p.relative_to(framework).as_posix() for p in seen)


def evaluator_identity(root=None, *, version=CURRENT_IDENTITY_VERSION):
    root = Path(root) if root else Path(__file__).resolve().parent
    if version == 1:
        files = {p.relative_to(root).as_posix(): sha(p)
                 for p in sorted(root.rglob('*'))
                 if p.is_file() and p.suffix in ('.py', '.tcl') and '__pycache__' not in p.parts}
        for directory in ('grt_bench', 'local_bench', 'global_bench', 'scripts'):
            for p in sorted((root.parent / directory).rglob('*')):
                if p.is_file() and p.suffix in ('.py', '.sh', '.tcl') and '__pycache__' not in p.parts:
                    files['../' + p.relative_to(root.parent).as_posix()] = sha(p)
        if not files:
            raise ValueError('Empty engineering evaluator inventory')
        return digest(files)
    if version != 2:
        raise ValueError('Unknown evaluator identity version: ' + repr(version))
    framework = root.resolve().parent
    files = {rel: sha(framework / rel) for rel in evaluator_files(root)}
    return digest({'identity_version': 2, 'files': files})


def identity_version(record):
    """A missing field means a frozen version-1 qualification."""
    version = record.get('identity_version', 1)
    if type(version) is not int or version not in IDENTITY_VERSIONS:
        raise ValueError('Unknown evaluator identity version')
    return version


def current_identity():
    """Fields a new qualification records about its evaluator."""
    return {'identity_version': CURRENT_IDENTITY_VERSION,
            'evaluator_sha256': evaluator_identity(version=CURRENT_IDENTITY_VERSION)}


def verify_evaluator(q, root=None):
    """Recompute the evaluator identity with the version the record declares."""
    if q.get('evaluator_sha256') != evaluator_identity(root, version=identity_version(q)):
        raise ValueError('Qualified evaluator changed')
    return True


def qualification_binding(q, *, source, image, run, reference, context_sha256=None):
    """Recheck actual input/runtime bytes; never copy old claims into a new run."""
    import subprocess
    source, run, reference = map(Path, (source, run, reference))
    replay = read_json(reference / 'replay/input-replay.json')
    from cts_reference_stage import verify_prefix
    if replay['prefix'] != verify_prefix(run):
        raise ValueError('Qualified physical prefix changed')
    for path, expected in replay['source_sha256'].items():
        if sha(source / path) != expected:
            raise ValueError('Qualified physical source changed: ' + path)
    command = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--read-only',
               '--cpus', '1', '--memory', '1g', '--entrypoint', 'sha256sum', image,
               '/usr/local/bin/openroad']
    binary = subprocess.check_output(command, text=True, timeout=60).split()[0]
    if q['runtime_sha256'] != digest({'image': image, 'binary': binary}):
        raise ValueError('Qualified runtime changed')
    if q['source_sha256'] != digest(replay['source_sha256']):
        raise ValueError('Qualified source identity changed')
    odbs = {d: sha(reference / ('normalized-' + d) / 'native/design.odb') for d in ('bottom', 'top')}
    expected_input = digest({'odb': odbs, 'context': context_sha256}) if context_sha256 else digest(odbs)
    if q['input_sha256'] != expected_input:
        raise ValueError('Qualified physical input changed')
    verify_evaluator(q)
