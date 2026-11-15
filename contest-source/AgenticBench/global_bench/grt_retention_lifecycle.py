"""Opt-in host retention between terminal rounds; never changes evaluation accounting."""
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import stat
import tarfile
import time

from grt_bench.archive import Archive
from grt_bench.common import confined, digest, read, sha, write
from grt_bench.contract import validate

CONTROL = 'control/trace-retention'
TERMINAL = {'verified', 'infeasible', 'failed', 'rejected', 'generation_failed',
            'no_change', 'blocked', 'stalled'}
TEXT_SUFFIXES = {'.log', '.rpt', '.md', '.jsonl', '.txt', '.json', '.py', '.sh', '.tcl',
                 '.cc', '.cpp', '.h', '.csv', '.yaml', '.yml', '.xml'}
PRIVATE_PARTS = {'credentials', 'credential', 'private', 'auth', '.claude', '.codex',
                 '.git', '__pycache__', 'settings', 'secrets', 'cache', 'caches', 'build',
                 'tmp', '.cache', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'node_modules',
                 'venv', '.venv', 'dist'}
PRIVATE_NAMES = {'host.key', 'authorization.json', 'user-authorization.json', 'tool-config.json',
                 'claude-mcp.json', '.env', 'settings.json', 'settings.local.json',
                 'config.json', 'command.json', 'lease.json', 'environment.json'}
AGENT_NAMES = {'events.jsonl', 'result.json', 'stdout.log', 'stderr.log', 'prompt.md',
               'final.txt', 'session.json'}
SECRET_CONTENT = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
                            r'\b(?:sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{30,})\b')


def _private(relative):
    parts = {p.lower() for p in Path(relative).parts}
    name = Path(relative).name.lower()
    return (bool(parts & PRIVATE_PARTS) or name in PRIVATE_NAMES or
            any(word in name for word in ('credential', 'secret', 'private', 'password', 'token',
                                           'authorization', 'api_key', 'apikey', 'api-key', 'api.key')))


def _text(path, relative):
    if _private(relative) or path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        return False
    try:
        content = path.read_bytes().decode('utf-8')
    except UnicodeDecodeError:
        return False
    return '\x00' not in content and SECRET_CONTENT.search(content) is None


def _candidate_rows(root, task):
    database = confined(root, 'control/archives/' + validate(task) + '.sqlite')
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        return [json.loads(row[0]) for row in db.execute('SELECT row_json FROM evaluations ORDER BY rowid')]


def _round_folder(root, row):
    number = row.get('round')
    if type(number) is not int or number < 1:
        raise ValueError('Retention requires the exact persisted round number')
    return confined(root, 'control/rounds/%04d' % number)


def _development_texts(root, row, legacy_retention=None):
    """Read only hash-bound unique experiments, never the full source workspace."""
    number = row['round']
    directories = set()
    control = confined(root, 'control')
    # These are exact legacy helper output locations, not arbitrary host logs.
    for launch in control.iterdir():
        if launch.is_symlink() or not launch.is_dir():
            continue
        p = confined(root, launch.relative_to(root).as_posix() + '/retention/%04d/development' % number)
        if p.is_dir():
            directories.add(p)
    if legacy_retention is not None:
        supplied = Path(legacy_retention)
        supplied = supplied if supplied.is_absolute() else root / supplied
        relative = supplied.relative_to(root).as_posix()
        p = confined(root, relative + '/development')
        if p.is_dir():
            directories.add(p)
    result = []
    for directory in sorted(directories):
        receipt_path, archive = directory / 'receipt.json', directory / 'experiments.tar.gz'
        if not receipt_path.is_file() or not archive.is_file():
            continue
        receipt = read(receipt_path)
        if receipt.get('status') != 'removed' or receipt.get('candidate') != row.get('candidate'):
            raise ValueError('Development retention receipt differs from its terminal candidate')
        if sha(archive) != receipt.get('experiments_sha256'):
            raise ValueError('Retained development experiment archive changed')
        retained = receipt.get('retained_paths')
        if not isinstance(retained, list) or len(retained) != len(set(retained)):
            raise ValueError('Invalid retained development path inventory')
        expected = set(retained)
        namespace = digest(directory.relative_to(root).as_posix())
        with tarfile.open(archive, 'r:gz') as bundle:
            members = bundle.getmembers()
            if len(members) != len(expected) or {member.name for member in members} != expected:
                raise ValueError('Retained development archive inventory differs')
            for member in members:
                name = Path(member.name)
                if (name.is_absolute() or '..' in name.parts or name.as_posix() != member.name
                        or any(ord(character) < 32 for character in member.name)):
                    raise ValueError('Unsafe retained development archive member')
                if not member.isfile() or _private(name) or name.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                stream = bundle.extractfile(member)
                data = stream.read() if stream is not None else b''
                if len(data) != member.size:
                    raise ValueError('Incomplete retained development text')
                try:
                    content = data.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                if '\x00' in content or SECRET_CONTENT.search(content):
                    continue
                relative = CONTROL + '/development-texts/' + namespace + '/' + name.as_posix()
                target = confined(root, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.read_bytes() != data or stat.S_IMODE(target.stat().st_mode) != 0o444:
                        raise ValueError('Retained development text changed')
                else:
                    with target.open('xb') as output:
                        output.write(data)
                    target.chmod(0o444)
                result.append(relative)
    return result


def round_log_paths(root, row, *, legacy_retention=None):
    """Explicit public trace and local diagnostics; no host credential/config scan."""
    root = Path(root).resolve(strict=True)
    folder = _round_folder(root, row)
    candidates = []
    prompt = folder / 'prompt'
    if prompt.exists():
        candidates.extend(p for p in prompt.iterdir() if p.suffix in {'.md', '.json'})
    agent = folder / 'agent'
    if agent.exists():
        candidates.extend(p for p in agent.rglob('*') if p.name in AGENT_NAMES or
                          p.name.endswith(('.stdout.log', '.stderr.log')))
    for name in ('tool-calls.jsonl', 'response.json', 'response.raw.json'):
        p = folder / name
        if p.exists():
            candidates.append(p)
    checks = folder / 'gpu-checks'
    if checks.exists():
        candidates.extend(p for p in checks.rglob('*') if p.name in {'request.json', 'result.json', 'container.log'})
    scratch = confined(root, 'scratch/%04d' % row['round'])
    if scratch.exists():
        candidates.extend(scratch.rglob('*'))
    selected = []
    for path in sorted(set(candidates)):
        if path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if _text(confined(root, relative), relative):
            selected.append(relative)
    selected.extend(_development_texts(root, row, legacy_retention))
    return sorted(set(selected))


def prunable_round_texts(root, row):
    """Tight direct-log allowlist; source, configs, unknown and recovery files stay pinned."""
    root = Path(root).resolve(strict=True)
    folder = _round_folder(root, row)
    paths = []
    agent = folder / 'agent'
    if agent.exists():
        paths.extend(p for p in agent.rglob('*') if p.name in {'events.jsonl', 'stdout.log', 'stderr.log'}
                     or p.name.endswith(('.stdout.log', '.stderr.log')))
    checks = folder / 'gpu-checks'
    if checks.exists():
        paths.extend(p for p in checks.glob('*/container.log'))
    scratch = confined(root, 'scratch/%04d' % row['round'])
    if scratch.exists():
        paths.extend(p for p in scratch.rglob('*') if p.suffix.lower() in {'.log', '.rpt', '.txt'})
    result = []
    for path in sorted(set(paths)):
        if path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if (set(Path(relative).parts) & {'source', 'sources', 'src', 'workspace', 'workspaces'}
                or any(word in path.name.lower() for word in ('config', 'setting', 'token', 'credential', 'auth', 'secret', 'private', 'key'))):
            continue
        if _text(confined(root, relative), relative):
            result.append(relative)
    return result


def _completed_retirement(root, service, relative, checksum):
    """Audit an existing completed receipt; never turn an idempotent read into a deletion retry."""
    plan = confined(root, relative)
    if not plan.name.endswith('.plan.json') or sha(plan) != checksum:
        raise ValueError('Lifecycle retirement plan changed')
    receipt = plan.with_name(plan.name[:-len('.plan.json')] + '.receipt.json')
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError('Lifecycle retirement lacks its completed receipt')
    return service.apply(relative, expected_sha256=checksum)


def after_round(root, task, row=None, *, legacy_retention=None):
    """Seal terminal history, then retire exact eligible files without any ledger edits."""
    root = Path(root).resolve(strict=True)
    authorization = confined(root, CONTROL + '/authorization.json')
    if not authorization.exists():
        return None  # Unconfigured campaigns preserve the exact legacy lifecycle.
    from .grt_history_retention import RetainedHistory
    from .grt_retention_gc import RetentionGC
    from .grt_library_retention import LibraryRetention
    history = RetainedHistory(root, task)  # A bad existing authorization must fail closed.
    state_path = confined(root, 'control/campaign.json')
    state = read(state_path)
    rounds = state.get('rounds')
    if not isinstance(rounds, list) or any(r.get('status') not in TERMINAL for r in rounds):
        raise ValueError('Retention requires terminal rounds; uncertain work remains pinned')
    if row is not None:
        if not rounds or rounds[-1] != row:
            raise ValueError('Retention round differs from the exact persisted terminal row')
    rows = _candidate_rows(root, task)
    before_archive = Archive(root, task).show()
    event = validate(task) + ':terminal:' + str(rounds[-1]['round'] if rounds else 0)
    receipt_path = confined(root, CONTROL + '/lifecycle/' + digest({'event': event, 'state': state}) + '.json')
    if receipt_path.exists():
        receipt = history.verify_host_record(read(receipt_path), kind='lifecycle-receipt')
        if receipt['event'] != event or receipt['state_digest'] != digest(state):
            raise ValueError('Completed retention lifecycle state changed')
        for evaluation in rows:
            history.history(evaluation['candidate'])
        raw = _completed_retirement(root, RetentionGC(root, task), receipt['raw_plan'], receipt['raw_plan_sha256'])
        library = _completed_retirement(root, LibraryRetention(root, task), receipt['library_plan'], receipt['library_plan_sha256'])
        if (raw.get('released_blocks_bytes', 0) != receipt['raw_released_bytes'] or
                library.get('released_blocks_bytes', 0) != receipt['library_released_bytes']):
            raise ValueError('Lifecycle completed retirement accounting changed')
        return dict(receipt, receipt=receipt_path.relative_to(root).as_posix())
    matching = {}
    for terminal in rounds:
        if terminal.get('candidate') and terminal['status'] in {'verified', 'infeasible'}:
            if terminal['candidate'] in matching:
                raise ValueError('Candidate has ambiguous terminal round provenance')
            matching[terminal['candidate']] = terminal
    sealed = []
    baseline = task['optimization']['baseline']
    for order, evaluation in enumerate(rows):
        candidate = evaluation['candidate']
        if history.has(candidate):
            history.history(candidate)
            continue
        if candidate == baseline:
            identity, paths = validate(task) + ':baseline', []
        else:
            terminal = matching.get(candidate)
            if terminal is None:
                raise ValueError('Admitted candidate lacks an exact completed round')
            identity = validate(task) + ':round:' + str(terminal['round']) + ':' + candidate
            score = evaluation['vector']['score']
            earlier = [item['vector']['score'] for item in rows[:order] if item.get('feasible')]
            record = evaluation.get('feasible') is True and all(score < value for value in earlier)
            # Do not unpack or duplicate development diagnostics that the policy
            # will intentionally discard for a non-record candidate.
            paths = (round_log_paths(root, terminal, legacy_retention=legacy_retention if terminal == row else None)
                     if record else [])
        history.seal(candidate, identity, log_paths=paths)
        sealed.append(candidate)
    diagnostics = []
    for terminal in rounds:
        if terminal['status'] in {'verified', 'infeasible'}:
            continue
        value = {'round': terminal['round'], 'status': terminal['status'], 'candidate': terminal.get('candidate'),
                 'history': terminal.get('history'), 'matrices': terminal.get('matrices', 0),
                 'evaluated': bool(terminal.get('evaluation_failure')), 'budget_changed': False,
                 'note': 'Failed or unsubmitted diagnostics; no historical evaluation seal or fresh admission.'}
        target = confined(root, CONTROL + '/diagnostics/%04d.json' % terminal['round'])
        envelope = history.sign_host_record(value, kind='terminal-diagnostics')
        if target.exists():
            if history.verify_host_record(read(target), kind='terminal-diagnostics') != value:
                raise ValueError('Terminal diagnostic accounting changed')
        else:
            write(target, envelope)
        diagnostics.append(target.relative_to(root).as_posix())
    gc = RetentionGC(root, task)
    plan = gc.plan()
    raw = gc.apply(plan['path'], expected_sha256=plan['sha256'])
    libraries = LibraryRetention(root, task)
    plan_library = libraries.plan()
    library = libraries.apply(plan_library['path'], expected_sha256=plan_library['sha256'])
    if read(state_path) != state or Archive(root, task).show() != before_archive:
        raise ValueError('Retention unexpectedly changed campaign accounting or archive admission')
    result = {'event': event, 'state_digest': digest(state), 'sealed': sealed,
              'diagnostics': diagnostics, 'raw_plan': plan['path'], 'raw_plan_sha256': plan['sha256'],
              'raw_released_bytes': raw.get('released_blocks_bytes', 0),
              'library_plan': plan_library['path'], 'library_plan_sha256': plan_library['sha256'],
              'library_released_bytes': library.get('released_blocks_bytes', 0),
              'evaluations_used': state['evaluations_used'], 'budget_changed': False,
              'completed_at': time.time()}
    write(receipt_path, history.sign_host_record(result, kind='lifecycle-receipt'))
    return dict(result, receipt=receipt_path.relative_to(root).as_posix())
