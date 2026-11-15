"""Algorithm boundaries shared by permissions, runtime mounts and stage replay."""
import ast
from pathlib import Path

DREAMPLACE = 'Place-3D/thirdparty/DREAMPlace/dreamplace/'
PLACEMENT_MODULES = (
    'BasicPlace.py', 'NonLinearPlace.py', 'PlaceObj.py',
    'NesterovAcceleratedGradientOptimizer.py', 'HBTConcurrentPlace.py',
    'concurrent_lol.py',
)
PLACEMENT_OPS = (
    'adjust_node_area', 'electric_potential', 'density_potential',
    'weighted_average_wirelength', 'logsumexp_wirelength', 'move_boundary',
    'pin_utilization', 'rudy', 'noise',
    'macro_legalize', 'greedy_legalize', 'abacus_legalize',
    'global_swap', 'k_reorder', 'independent_set_matching',
)
# These files mix algorithms with physical input/metric contracts. Only the
# named policy functions may change; imports, classes and other functions stay fixed.
FUNCTION_SCOPE = {
    'Place-3D/src/partition/lol/core.py': ('build_unique_nets', 'move_delta_cut'),
    'Place-3D/src/partition/lol/constraints.py': ('macro_balanced_fixed_assignment',),
    DREAMPLACE + 'concurrent_lol.py': (
        'project_wirelength_pin_pos', 'backend_aware_target_density',
        '_apply_cross_die_net_weights', '_apply_die_aware_initialization',
    ),
}


def placement_python(name):
    if not name.startswith(DREAMPLACE):
        return False
    relative = name[len(DREAMPLACE):]
    return relative in PLACEMENT_MODULES or any(
        relative.startswith('ops/' + op + '/') and relative.endswith('.py')
        and Path(relative).name != '__init__.py' for op in PLACEMENT_OPS)


def function_scope(files):
    return {name: list(FUNCTION_SCOPE[name]) for name in files if name in FUNCTION_SCOPE}


def protected_ast(source, names):
    tree = ast.parse(source)
    found = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not set(names) <= found:
        raise ValueError('Missing scoped algorithm function')
    # Preserve signatures/decorators as well as the complete surrounding module.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.body = [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def verify_functions(before_root, after_root, policy):
    for name, names in policy.get('function_scope', {}).items():
        before = (Path(before_root) / name).read_text()
        after = (Path(after_root) / name).read_text()
        if protected_ast(before, names) != protected_ast(after, names):
            raise ValueError('Protected interface/input/metric code changed: ' + name)


def python_mounts(tree, files):
    """Bind candidate Python algorithms over image modules, keeping native ops installed."""
    from .common import confined
    return [(confined(tree, name), '/opt/dreamplace/dreamplace/' + name[len(DREAMPLACE):], True)
            for name in sorted(files) if placement_python(name)]


def verify_runtime(plan):
    from .common import digest, read, sha
    path = plan.get('python_runtime')
    if path is None:
        raise ValueError('Missing candidate Python runtime binding')
    receipt = read(path)
    if digest(receipt) != plan['python_runtime_digest'] or receipt['candidate'] != plan['candidate']:
        raise ValueError('Candidate Python runtime identity changed')
    root = Path(plan['source'])
    expected = {p.relative_to(root).as_posix() for p in (root / DREAMPLACE).rglob('*.py')
                if placement_python(p.relative_to(root).as_posix())}
    if {r['source'] for r in receipt['modules'].values()} != expected:
        raise ValueError('Incomplete candidate Python runtime binding')
    for target, row in receipt['modules'].items():
        name = row['source']
        if not placement_python(name) or target != '/opt/dreamplace/dreamplace/' + name[len(DREAMPLACE):]:
            raise ValueError('Unexpected algorithm runtime module')
        if sha(Path(plan['source']) / name) != row['sha256'] or sha(target) != row['sha256']:
            raise ValueError('Installed algorithm differs from candidate source: ' + name)
    return receipt
