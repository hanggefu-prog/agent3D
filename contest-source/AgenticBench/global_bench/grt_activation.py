"""Conservative pre-submit activation checks, never a general equivalence proof.

Reject only an empty semantic diff or a known unreachable edit under every
training configuration. Unknown changes remain eligible for measured evaluation.
"""
import ast
from pathlib import Path
import yaml
from grt_bench.common import read, sha, digest, confined

PLACER = 'Place-3D/thirdparty/DREAMPlace/dreamplace/concurrent_lol.py'
FUNCTION = '_apply_cross_die_net_weights'


def _ast(source):
    return ast.dump(ast.parse(source), include_attributes=False)


# Recognize the disconnected read-only CSV preparation observed in a real
# candidate. This is deliberately exact: unfamiliar helpers stay "unknown".
_UNUSED_TIMING_SETUP = ast.parse("""
timing_crit = {}
if hasattr(placedb, 'timing_criticality_csv') and placedb.timing_criticality_csv:
    import csv
    from pathlib import Path as CsvPath
    csv_path = CsvPath(placedb.timing_criticality_csv)
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                net_name = row['net_name']
                criticality = float(row['criticality'])
                if hasattr(placedb, 'net_name_to_id'):
                    net_id = placedb.net_name_to_id.get(net_name)
                    if net_id is not None:
                        timing_crit[net_id] = criticality
criticality_alpha = getattr(placedb, 'criticality_alpha', 2.0)
""").body


def _drop_disconnected_setup(function):
    expected = [ast.dump(n, include_attributes=False) for n in _UNUSED_TIMING_SETUP]
    for i in range(len(function.body)-len(expected)+1):
        if [ast.dump(n, include_attributes=False) for n in function.body[i:i+len(expected)]] != expected:
            continue
        remaining = function.body[:i]+function.body[i+len(expected):]
        # net_id is rebound by the maintained loop before any use; all other
        # preparation locals must be unused once the disabled branch is removed.
        dead = {'timing_crit','criticality_alpha','csv','CsvPath','csv_path','f','reader','row','net_name','criticality'}
        if any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in dead
               for statement in remaining for n in ast.walk(statement)):
            return
        if not any(isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='net_id'
                   for n in function.body[i+len(expected):]):
            return
        function.body=remaining
        return


def inactive_weight_edit(before, after):
    """Both programs are identical after removing the false multiplier != 1 body."""
    def mask(source):
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
        guards = [n for n in ast.walk(function) if isinstance(n, ast.If) and
            ast.dump(n.test, include_attributes=False) == ast.dump(ast.parse('multiplier != 1.0', mode='eval').body, include_attributes=False)]
        if not guards:
            raise ValueError('No known guard')
        for guard in guards:
            guard.body = [ast.Pass()]
        _drop_disconnected_setup(function)
        if any(isinstance(n, ast.Name) and n.id == 'multiplier' and isinstance(n.ctx, (ast.Store, ast.Del))
               for n in ast.walk(function)):
            raise ValueError('Multiplier is reassigned')
        return ast.dump(tree, include_attributes=False)
    try:
        return mask(before) == mask(after)
    except (StopIteration, ValueError, SyntaxError):
        return False


def inspect_activation(config, task, candidate):
    root = Path(config['root']) / 'candidates'
    meta = read(root / candidate / 'candidate.json')
    parent = meta['parent']
    old = read(root / parent / 'candidate.json')
    changed = sorted(name for name, row in meta['files'].items() if row != old['files'].get(name))
    evidence = {'candidate': candidate, 'parent': parent, 'changed_files': changed,
                'status': 'unknown', 'reason': 'Activation is not proven by this conservative preflight.'}
    semantic = []
    for name in changed:
        if not name.endswith('.py') or _ast((root / parent / 'tree' / name).read_text()) != _ast((root / candidate / 'tree' / name).read_text()):
            semantic.append(name)
    if not semantic:
        return dict(evidence, status='inactive', reason='Only formatting/comments changed; no algorithm edit to evaluate.')
    if semantic != [PLACER] or not inactive_weight_edit((root / parent / 'tree' / PLACER).read_text(),
                                                         (root / candidate / 'tree' / PLACER).read_text()):
        return evidence
    settings = {}
    for case in task['cases']:
        if case['split'] != 'train':
            continue
        entry = config['environment']['cases'].get(case['id'])
        if entry is None:
            return evidence
        path = Path(entry['config'])
        if sha(path) != case['config_sha256']:
            raise ValueError('Activation input configuration changed')
        base = yaml.safe_load(path.read_text())
        placement = base.get('placement', {})
        if placement.get('algorithm') != 'concurrent-dreamplace':
            return evidence
        method = Path(placement.get('config', ''))
        # Resolve only actual read-only worker mounts and bind the settings to
        # the frozen task manifest, not a model-authored activation claim.
        if method.parts[:2] == ('/', 'workspace'):
            relative = method.relative_to('/workspace').as_posix()
            if relative not in meta['files'] or relative in changed:
                return evidence
            source = confined(root / candidate / 'tree', relative)
            expected = meta['files'][relative]['sha256']
        elif method.parts[:3] == ('/', 'inputs', 'case'):
            manifest_path = Path(entry['manifest'])
            manifest = read(manifest_path)
            if digest(manifest) != case['input_identity']:
                raise ValueError('Activation input manifest changed')
            relative = method.relative_to('/inputs/case').as_posix()
            if relative not in manifest['artifacts']:
                return evidence
            source = confined(manifest_path.parent, relative)
            expected = manifest['artifacts'][relative]['sha256']
        else:
            return evidence
        if sha(source) != expected:
            raise ValueError('Activation method settings differ from frozen input')
        value = read(source).get('concurrent_dreamplace', {}).get('cross_die_net_weight', 0.2)
        settings[case['id']] = {'cross_die_net_weight': value, 'config_sha256': sha(path), 'method_sha256': sha(source)}
        if value != 1.0:
            return dict(evidence, settings=settings)
    return dict(evidence, status='inactive', settings=settings,
        reason='The weighting change is behind multiplier != 1.0 and any recognized CSV setup has no active consumer, but cross_die_net_weight is 1.0 for every evaluation case. No evaluation budget was spent. Change an active path or revise the allowed algorithm, then test that the modified path is exercised.')
