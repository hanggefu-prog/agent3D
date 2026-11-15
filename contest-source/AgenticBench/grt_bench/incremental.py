"""Verified, paired-cell stage replay. Original checkpoints are never relabelled.

The host chooses the actual parent and binds both source inventories. A second
consumer checks the same compatibility proof and every imported artifact. An
unknown/shared dependency restarts at partition, after the fixed task inputs.

Proof v1 uses only the hand-maintained table. Proof v2 additionally derives a
deterministic AST import/script graph from the FlowRunner stage methods (LoL
integration only): a file starts at the earliest stage whose closure reaches
it, and native files at the first stage whose Tcl uses their command. Impact is
the MIN of table and derived stages and both are recorded; v1 proofs remain
verifiable. A derived stage can only move a restart earlier, never later.
"""
import argparse
import os
from pathlib import Path
import shutil
import tempfile

from .common import digest, read, sha, write

STAGES = {'input': '00_input', 'floorplan': '10_floorplan',
          'partition': '20_partition', 'placement': '30_placement',
          'pdn': '40_pdn', 'hbt': '50_hbt', 'opt': '60_opt', 'cts': '70_cts'}
ORDER = list(STAGES) + ['grt']
DRIVERS = ('open3dbench/runner.py',
           'Place-3D/thirdparty/partition/extract_tp_gnn_timing_openroad.py')


def contract(tree, base_image):
    return {'version': 1, 'base_image': base_image,
            'drivers': {p: sha(Path(tree) / p) for p in DRIVERS}}


def validate_contract(task):
    import re
    value = task['incremental']
    if (set(value) != {'version', 'base_image', 'drivers'} or value['version'] != 1
            or set(value['drivers']) != set(DRIVERS)
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', value['base_image'])
            or any(not re.fullmatch(r'[0-9a-f]{64}', h) for h in value['drivers'].values())
            or 'frontend_inputs' not in task):
        raise ValueError('Unbound incremental execution contract')


PROOF_VERSIONS = (1, 2)
# v2 adds a dependency graph derived from the stage drivers. The hand table stays
# authoritative for native code; each change starts at the earlier of the two.
GRAPH_VERSION = 1
REPLAY = ORDER[ORDER.index('partition'):]
# Import roots seen by the stage subprocesses (repository, Place-3D cwd and the
# DREAMPlace install, whose package and flat module names are both importable).
SEARCH_ROOTS = ('', 'Place-3D', 'Place-3D/thirdparty/DREAMPlace',
                'Place-3D/thirdparty/DREAMPlace/dreamplace')
# Input/floorplan are task-frozen inputs and never execute in a replay. Post-route
# report/thermal never run before the GRT endpoint. GRT tasks are LoL-only
# (grt_bench.flow), so driver branches on another integration type are pruned.
INTEGRATION = 'lol'
STAGE_METHODS = {'partition': ('_run_partition',),
                 'placement': ('_run_placement',), 'pdn': ('_run_pdn',), 'hbt': ('_run_hbt',),
                 'opt': ('_run_opt',), 'cts': ('_run_cts',),
                 'grt': ('global_route', '_run_routing', '_run_report', '_run_thermal')}
FROZEN_METHODS = ('_run_input', '_run_floorplan')
REPAIR = {'RepairSetup.cc', 'RepairSetup.hh', 'RepairHold.cc', 'RepairHold.hh'}
_GRAPHS = {}
_GRAPH_LIMIT = 16


def table_stage(name):
    """Hand-maintained first consumers. Unknown files are conservative."""
    from .algorithm_scope import placement_python, DREAMPLACE
    if name == 'Place-3D/src/hbt/assignment.py':
        return 'hbt'
    if name == 'Place-3D/src/placement/concurrent_dreamplace.py':
        return 'placement'
    if placement_python(name) and name != DREAMPLACE + 'concurrent_lol.py':
        return 'placement'
    if name.startswith('OpenROAD-3D/openroad/src/cts/'):
        return 'cts'
    if name.startswith('OpenROAD-3D/openroad/src/grt/src/fastroute/'):
        return 'grt'
    if name.startswith('OpenROAD-3D/openroad/src/rsz/src/') and Path(name).name in REPAIR:
        return 'opt'
    # Resizer/shared GRT APIs are consumed before their nominal stage.
    # In particular partition timing extraction calls estimate_parasitics;
    # HBT materialization calls buffer_ports.
    return 'partition'


def native_commands(name):
    """Tcl commands through which a native table entry is first reached."""
    if name.startswith('OpenROAD-3D/openroad/src/rsz/src/') and Path(name).name in REPAIR:
        return ('repair_timing',)
    if name.startswith('OpenROAD-3D/openroad/src/cts/'):
        return ('clock_tree_synthesis',)
    if name.startswith('OpenROAD-3D/openroad/src/grt/src/fastroute/'):
        return ('global_route',)
    return ()


def _index(tree):
    files = []
    for top, dirs, names in os.walk(str(tree)):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d != '__pycache__')
        for name in names:
            if name.endswith(('.py', '.tcl')):
                path = os.path.join(top, name)
                if os.path.isfile(path) and not os.path.islink(path):
                    files.append(os.path.relpath(path, str(tree)).replace(os.sep, '/'))
    return sorted(files)


def _join(*parts):
    return '/'.join(p for p in parts if p)


class _Graph:
    """Conservative static closure. Import edges execute at import time; script
    edges (`-m`, script paths, dynamic imports) only in entry processes."""
    def __init__(self, tree, files):
        self.tree, self.files = Path(tree), set(files)
        self.basenames = {}
        for name in files:
            self.basenames.setdefault(name.rsplit('/', 1)[-1], []).append(name)
        self.parsed = {}

    def module(self, dotted):
        import re
        if not isinstance(dotted, str) or not re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', dotted):
            return set()
        parts, found = dotted.split('.'), set()
        for root in SEARCH_ROOTS:
            base = _join(root, *parts)
            target = [p for p in (base + '.py', base + '/__init__.py') if p in self.files]
            if target:
                found.update(target)
                found.update(p for p in (_join(root, *parts[:i], '__init__.py')
                                         for i in range(1, len(parts))) if p in self.files)
        return found

    def relative(self, origin, level, dotted):
        parts = origin.split('/')[:-1]
        if level - 1 > len(parts):
            return set()
        base = parts[:len(parts) - (level - 1)]
        found = set()
        for i in range(1, len(base) + 1):
            if _join(*base[:i], '__init__.py') in self.files:
                found.add(_join(*base[:i], '__init__.py'))
        rest = [p for p in (dotted or '').split('.') if p]
        for i in range(1, len(rest) + 1):
            stem = _join(*(base + rest[:i]))
            found.update(p for p in (stem + '.py', stem + '/__init__.py') if p in self.files)
        return found

    def path(self, value):
        value = value.replace('\\', '/').strip()
        if (not value.endswith(('.py', '.tcl')) or any(c in value for c in '*?$ {}\n')
                or value.rsplit('/', 1)[-1] == '__init__.py'):
            return set()
        parts = [p for p in value.split('/') if p not in ('', '.')]
        if '..' in parts:
            parts = parts[len(parts) - parts[::-1].index('..'):]
        # Absolute container paths fall back to their longest matching suffix.
        for i in range(len(parts)):
            suffix = '/'.join(parts[i:])
            found = {f for f in self.basenames.get(parts[-1], ()) if f == suffix or f.endswith('/' + suffix)}
            if found:
                return found
        return set()

    def scan(self, node, origin, dotted=False):
        """(import edges, script edges, embedded Tcl strings) under one AST node."""
        import ast
        imports, scripts, tcl = set(), set(), []
        for item in ast.walk(node):
            if isinstance(item, ast.Import):
                for alias in item.names:
                    imports |= self.module(alias.name)
            elif isinstance(item, ast.ImportFrom):
                if item.level:
                    found = self.relative(origin, item.level, item.module)
                    for alias in item.names:
                        found |= self.relative(origin, item.level, _join(item.module or '', alias.name).replace('/', '.'))
                    imports |= found
                elif item.module:
                    imports |= self.module(item.module)
                    for alias in item.names:
                        imports |= self.module(item.module + '.' + alias.name)
            elif isinstance(item, ast.Call):
                name = getattr(item.func, 'attr', getattr(item.func, 'id', None))
                if name in ('import_module', '__import__') and item.args:
                    scripts |= self.module(_text(item.args[0]))
                elif name == 'join' and item.args and _text(item.args[-1]):
                    scripts |= self.path('/'.join(_text(a) for a in item.args if _text(a)))
            elif isinstance(item, (ast.List, ast.Tuple)):
                for left, right in zip(item.elts, item.elts[1:]):
                    if _text(left) == '-m':
                        scripts |= self.module(_text(right))
            elif isinstance(item, ast.BinOp) and isinstance(item.op, ast.Div):
                parts = [_text(p) for p in _chain(item)]
                if parts[-1]:
                    scripts |= self.path('/'.join(p for p in parts if p))
            value = _text(item)
            if value is not None:
                if '/' in value:  # A bare file name is compared or logged, not run.
                    scripts |= self.path(value)
                if dotted and '.' in value:
                    scripts |= self.module(value)
                if '\n' in value:
                    tcl.append(value)
        return imports, scripts, tcl

    def parse(self, name):
        import ast
        import warnings
        if name not in self.parsed:
            try:
                text = (self.tree / name).read_text(encoding='utf-8', errors='replace')
            except OSError:
                text = None
            if text is None or name.endswith('.tcl'):
                self.parsed[name] = (set(), set(), [text] if text else [])
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        node = ast.parse(text)
                    self.parsed[name] = self.scan(node, name)
                except (SyntaxError, ValueError):
                    # Unparseable sources keep only the conservative table.
                    self.parsed[name] = (set(), set(), [])
        return self.parsed[name]

    def closure(self, full, imported, skip=()):
        """Files reached from entry (`full`) and import-only seeds, plus entry Tcl."""
        modes, pending, tcl = {}, [(n, True) for n in sorted(full)] + [(n, False) for n in sorted(imported)], []
        while pending:
            name, entry = pending.pop()
            if name in skip or modes.get(name) is True or (modes.get(name) is False and not entry):
                continue
            modes[name] = entry
            imports, scripts, strings = self.parse(name)
            if entry:
                tcl.extend(strings)
            pending.extend((n, entry) for n in sorted(imports))
            if entry:
                pending.extend((n, True) for n in sorted(scripts))
        return set(modes), tcl


def _text(node):
    import ast
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _chain(node):
    import ast
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _chain(node.left) + _chain(node.right)
    return [node]


def _uses(command, texts):
    import re
    pattern = re.compile(r'(?:^|[\[;{])\s*' + re.escape(command) + r'(?=[ \t\\\]]|$)')
    for text in texts:
        for line in text.splitlines():
            if not line.lstrip().startswith('#') and pattern.search(line):
                return True
    return False


def _names(node):
    import ast
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _integration(test):
    """True/False for `self.config["type"] ==/!= "<name>"`, else None."""
    import ast
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], (ast.Eq, ast.NotEq))):
        return None
    sides = [test.left, test.comparators[0]]
    def is_type(node):
        key = getattr(node, 'slice', None)
        key = key.value if isinstance(key, getattr(ast, 'Index', ())) else key  # Python 3.8
        return (isinstance(node, ast.Subscript) and _text(key) == 'type'
                and isinstance(node.value, ast.Attribute) and node.value.attr == 'config'
                and isinstance(node.value.value, ast.Name) and node.value.value.id == 'self')
    values = [_text(n) for n in sides]
    if not any(is_type(n) for n in sides) or not any(values):
        return None
    equal = INTEGRATION in values
    return equal if isinstance(test.ops[0], ast.Eq) else not equal


def _prune(node):
    import ast
    class Prune(ast.NodeTransformer):
        def visit_If(self, item):
            taken = _integration(item.test)
            if taken is None:
                return self.generic_visit(item)
            return [self.visit(n) for n in (item.body if taken else item.orelse)]
    return Prune().visit(node)


def _derive(tree, files):
    import ast
    graph, driver = _Graph(tree, files), DRIVERS[0]
    module = _prune(ast.parse((Path(tree) / driver).read_text(encoding='utf-8')))
    runner = [n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'FlowRunner']
    methods = {n.name: n for n in (runner[0].body if runner else [])
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not all(m in methods for names in list(STAGE_METHODS.values()) + [FROZEN_METHODS] for m in names):
        raise ValueError('Stage driver no longer exposes its stage methods')

    def reach(names):
        seen, pending = set(), list(names)
        while pending:
            name = pending.pop()
            if name not in seen:
                seen.add(name)
                pending.extend(i.attr for i in ast.walk(methods[name]) if isinstance(i, ast.Attribute)
                               and isinstance(i.value, ast.Name) and i.value.id == 'self' and i.attr in methods)
        return seen
    owned = {stage: reach(STAGE_METHODS[stage]) for stage in REPLAY}
    staged = set().union(*owned.values())
    # Module-level names bound by imports run in a stage only when that stage uses them.
    bound, imported = {}, set()
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found = graph.scan(node, driver)[0]
            imported |= found
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                target = (graph.module(alias.name) if isinstance(node, ast.Import) else
                          graph.relative(driver, node.level, _join(node.module or '', alias.name).replace('/', '.'))
                          if node.level else graph.module(node.module + '.' + alias.name) or graph.module(node.module))
                bound[name] = bound.get(name, set()) | (target or found)

    def entries(nodes):
        found = set()
        for node in nodes:
            imports, scripts, _ = graph.scan(node, driver, dotted=True)
            found |= imports | scripts
            for name in _names(node):
                found |= bound.get(name, set())
        return found
    # Driver code outside the stage methods (helpers, run/finalize hooks) runs for each stage.
    shared = [n for n in module.body if not isinstance(n, (ast.Import, ast.ImportFrom)) and n not in runner]
    frozen = reach(FROZEN_METHODS) - staged
    shared += [n for n in methods.values() if n.name not in staged | frozen]
    shared_entries = entries(shared)
    stages, consumers, closures = {}, {}, {}
    for stage in REPLAY:
        full = shared_entries | entries(methods[n] for n in sorted(owned[stage]))
        closure, tcl = graph.closure(full, imported, skip={driver})
        tcl += [t for n in shared + [methods[m] for m in sorted(owned[stage])] for t in graph.scan(n, driver)[2]]
        closure.add(driver)
        closures[stage] = sorted(closure)
        for name in closures[stage]:
            stages.setdefault(name, stage)
        for command in ('repair_timing', 'clock_tree_synthesis', 'global_route'):
            if command not in consumers and _uses(command, tcl):
                consumers[command] = stage
    return stages, consumers, closures


def dependency_graph(tree):
    """Deterministic earliest-stage map for one source tree, cached per tree state."""
    tree = Path(tree).resolve(strict=True)
    files = _index(tree)
    stamp = []
    for name in files:
        st = os.stat(str(tree / name))
        stamp.append((name, st.st_size, st.st_mtime_ns, st.st_ino))
    key = (str(tree), digest(stamp))
    if key in _GRAPHS:
        return _GRAPHS[key]
    stages, consumers, closures = _derive(tree, files)
    hashes = {name: sha(tree / name) for name in sorted(stages)}
    value = {'version': GRAPH_VERSION, 'stages': stages, 'native_consumers': consumers,
             'closures': closures, 'sha256': hashes}
    value['digest'] = digest({k: value[k] for k in ('version', 'stages', 'native_consumers', 'sha256')})
    while len(_GRAPHS) >= _GRAPH_LIMIT:
        _GRAPHS.pop(next(iter(_GRAPHS)), None)
    _GRAPHS[key] = value
    return value


def derived_stage(graph, name):
    """Earliest replay stage in which a file is reached; `grt` means no consumer found."""
    stage = graph['stages'].get(name)
    commands = native_commands(name)
    found = [graph['native_consumers'][c] for c in commands if c in graph['native_consumers']]
    candidates = ([stage] if stage else []) + found
    return min(candidates, key=ORDER.index) if candidates else 'grt'


def derive_dependencies(graphs, names):
    """Earliest derived stage over every tree that may run the replayed suffix."""
    return {name: min((derived_stage(g, name) for g in graphs), key=ORDER.index) for name in sorted(names)}


def proof_names(parent, current, baseline):
    """Every change that v2 impact may classify, including inherited upstream edits."""
    return sorted(set(changed(parent, current)) | set(changed(baseline, parent)))


def changed(before, after):
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))


def impact(before, after, baseline=None, dependencies=None):
    """v1 uses the hand table only. With `dependencies`, each change starts at
    min(table, derived) and both inputs are recorded in the proof."""
    if dependencies is not None:
        return _impact_v2(before, after, baseline, dependencies)
    changes = changed(before, after)
    affected = {name: table_stage(name) for name in changes}
    start = min(affected.values(), key=ORDER.index) if affected else 'grt'
    inherited = {}
    if baseline is not None:
        # Editable ancestors may introduce new calls to another component.
        # The fixed baseline call graph alone cannot certify such a prefix.
        for name, stage in impact(baseline, before)['affected'].items():
            if ORDER.index(stage) < ORDER.index(start):
                inherited[name] = stage
        if inherited:
            start = min(inherited.values(), key=ORDER.index)
    return {'start': start, 'changes': changes, 'affected': affected,
            'uncertified_upstream_dependencies': inherited}


def _impact_v2(before, after, baseline, derived):
    if not isinstance(derived, dict) or any(v not in REPLAY for v in derived.values()):
        raise ValueError('Invalid derived dependency map')
    changes = changed(before, after)
    affected, table, chosen = {}, {}, {}
    for name in changes:
        if name not in derived:
            raise ValueError('Derived dependency missing: ' + name)
        table[name], chosen[name] = table_stage(name), derived[name]
        affected[name] = min(table[name], chosen[name], key=ORDER.index)
    start = min(affected.values(), key=ORDER.index) if affected else 'grt'
    inherited = {}
    if baseline is not None:
        for name, stage in _impact_v2(baseline, before, None, derived)['affected'].items():
            if ORDER.index(stage) < ORDER.index(start):
                inherited[name] = stage
        if inherited:
            start = min(inherited.values(), key=ORDER.index)
    return {'start': start, 'changes': changes, 'affected': affected,
            'uncertified_upstream_dependencies': inherited, 'table': table, 'derived': chosen}


def resources_for(proof, frontend, backend):
    return dict(backend if proof and ORDER.index(proof['impact']['start']) > ORDER.index('placement') else frontend)


def check_pair(payload, parent_plan):
    for key in ('task_digest', 'case', 'seed', 'split', 'endpoint'):
        if payload[key] != parent_plan[key]:
            raise ValueError('Incremental source differs in ' + key)


def check_runtime(runtime, candidate, base):
    if runtime['candidate'] != candidate:
        raise ValueError('Incremental runtime candidate differs')
    if runtime.get('native'):
        if runtime['manifest']['base_image'] != base:
            raise ValueError('Incremental native base image differs')
    elif runtime['image'] != base:
        raise ValueError('Incremental base image differs')


def check_dependencies(proof, current, parent, baseline, trees=None):
    """v2 derived map. With all three trees the host recomputes it exactly; with
    a subset (the in-container replay sees only /workspace) each stored stage
    must be no later than what that tree derives."""
    import re
    names = proof_names(parent['files'], current['files'], baseline['files'])
    deps, graphs = proof.get('dependencies'), proof.get('graphs')
    if (not isinstance(deps, dict) or sorted(deps) != names or any(v not in REPLAY for v in deps.values())
            or not isinstance(graphs, dict) or set(graphs) != {'current', 'parent', 'baseline'}
            or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in graphs.values())):
        raise ValueError('Invalid derived dependency proof')
    trees = dict(trees or {})
    if set(trees) - set(graphs):
        raise ValueError('Unknown dependency tree role')
    derived = {role: dependency_graph(tree) for role, tree in sorted(trees.items())}
    if set(derived) == set(graphs):
        if any(derived[r]['digest'] != graphs[r] for r in graphs):
            raise ValueError('Derived dependency graph differs from source tree')
        if derive_dependencies([derived[r] for r in sorted(derived)], names) != deps:
            raise ValueError('Derived dependencies differ from source trees')
    for graph in derived.values():
        for name in names:
            if ORDER.index(deps[name]) > ORDER.index(derived_stage(graph, name)):
                raise ValueError('Derived dependency is later than its source tree: ' + name)
    return deps


def validate_proof(proof, task, payload, current, parent, parent_plan, parent_verified, baseline, trees=None):
    """v1 recomputes the hand table; v2 also binds the derived dependency map."""
    from .contract import validate
    validate_contract(task)
    if proof.get('version') not in PROOF_VERSIONS:
        raise ValueError('Unsupported incremental proof')
    check_pair(payload, parent_plan)
    if proof['task_digest'] != validate(task) or proof['task_digest'] != payload['task_digest']:
        raise ValueError('Incremental task differs')
    if (digest(current) != payload['candidate'] or digest(parent) != proof['parent']
            or current['parent'] != proof['parent'] or parent_plan['candidate'] != proof['parent']):
        raise ValueError('Incremental source/parent differs')
    if digest(baseline) != task['optimization']['baseline']:
        raise ValueError('Incremental baseline inventory differs')
    for p, h in task['incremental']['drivers'].items():
        if current['files'][p]['sha256'] != h or parent['files'][p]['sha256'] != h:
            raise ValueError('Stage dependency driver changed: ' + p)
    deps = None
    if proof['version'] == 2:
        deps = check_dependencies(proof, current, parent, baseline, trees)
    elif 'dependencies' in proof or 'graphs' in proof:
        raise ValueError('Version 1 proofs carry no derived dependencies')
    expected = impact(parent['files'], current['files'], baseline['files'], dependencies=deps)
    if proof['impact'] != expected:
        raise ValueError('Incremental affected stage differs from actual source diff')
    prefix = ORDER[:ORDER.index(expected['start'])]
    if set(proof['stages']) != set(prefix):
        raise ValueError('Incremental prefix is incomplete or includes affected stages')
    for key in ('candidate', 'case', 'seed', 'split', 'task_digest', 'endpoint'):
        if parent_verified[key] != parent_plan[key]:
            raise ValueError('Incremental source receipt differs')
    if parent_verified['status'] != 'verified':
        raise ValueError('Incremental source is not verified')
    for name, value in (('parent_plan', parent_plan), ('parent_verified', parent_verified)):
        if proof[name + '_digest'] != digest(value):
            raise ValueError('Incremental evidence changed: ' + name)
    for name, candidate in (('runtime', payload['candidate']), ('parent_runtime', proof['parent'])):
        check_runtime(proof[name], candidate, task['incremental']['base_image'])
    return expected


def physical_source(root, job):
    """Follow an explicitly verified analysis replay without relabelling it."""
    from .optimization import identity
    from .rescore import validate_transition
    root = Path(root); seen = set()
    while True:
        identity(job)
        if job in seen:
            raise ValueError('Cyclic physical source')
        seen.add(job)
        attempt = root/'runs'/job
        plan = read(attempt/'control/plan.json')
        if not plan.get('rescore'):
            return job, attempt/'output'
        old_job = plan['rescore']['source_job']; identity(old_job)
        old = root/'runs'/old_job
        old_plan = read(old/'control/plan.json')
        if sha(old/'control/plan.json') != plan['rescore']['previous_plan_sha256']:
            raise ValueError('Physical replay plan changed')
        validate_transition(read(attempt/'control/task.json'), read(old/'control/task.json'), plan, old_plan)
        job = old_job


def prepare_matrix(root, task, candidate, runtime, version=2):
    """Choose the recorded parent, never a metric-selected seed or another case.
    New proofs are v2; `version=1` reproduces an already-admitted v1 matrix."""
    from .archive import Archive, completed_outcomes
    from .candidates import verify_snapshot
    from .contract import matrix, validate
    from .queue import Queue
    if version not in PROOF_VERSIONS:
        raise ValueError('Unsupported incremental proof')
    if 'incremental' not in task or candidate == task['optimization']['baseline']:
        return {}
    root = Path(root); archive = Archive(root, task)
    current = verify_snapshot(root / 'candidates' / candidate)
    parent_id = current['parent']
    parent = verify_snapshot(root / 'candidates' / parent_id)
    baseline = read(root / 'candidates' / task['optimization']['baseline'] / 'candidate.json')
    completed_outcomes(root, task, parent_id, archive.admit(parent_id))
    parent_runtime = read(root / 'control/runtimes' / (parent_id + '.json'))
    rows = [r for r in Queue(root / 'control/queue.sqlite').rows()
            if read_payload(r).get('candidate') == parent_id
            and read_payload(r).get('task_digest') == validate(task)]
    trees = proof_trees(root, task, candidate, parent_id)
    extra, boundary = {}, impact(parent['files'], current['files'], baseline['files'])
    if version == 2:
        graphs = {role: dependency_graph(tree) for role, tree in trees.items()}
        deps = derive_dependencies([graphs[r] for r in sorted(graphs)],
                                   proof_names(parent['files'], current['files'], baseline['files']))
        extra = {'dependencies': deps, 'graphs': {r: g['digest'] for r, g in sorted(graphs.items())}}
        boundary = impact(parent['files'], current['files'], baseline['files'], dependencies=deps)
    result = {}
    for payload in matrix(task, candidate):
        matches = [r for r in rows if read_payload(r)['case'] == payload['case']
                   and read_payload(r)['seed'] == payload['seed'] and r['state'] == 'succeeded']
        if len(matches) != 1:
            raise ValueError('Exactly one verified parent checkpoint per case/seed is required')
        source = root / 'runs' / matches[0]['id']
        parent_plan = read(source / 'control/plan.json')
        verified = read(source / 'verified/result.json')
        physical_job, physical_output = physical_source(root, matches[0]['id'])
        prefix = ORDER[:ORDER.index(boundary['start'])]
        stages = {s: sha(physical_output / 'run' / STAGES[s] / 'stage_manifest.json') for s in prefix}
        proof = {'version': version, 'task_digest': validate(task), 'parent': parent_id,
                 'source_job': matches[0]['id'], 'physical_source_job': physical_job,
                 'impact': boundary, 'stages': stages,
                 'parent_plan_digest': digest(parent_plan), 'parent_verified_digest': digest(verified),
                 'runtime': runtime, 'parent_runtime': parent_runtime, **extra}
        validate_proof(proof, task, payload, current, parent, parent_plan, verified, baseline, trees)
        result[(payload['case'], payload['seed']['id'])] = proof
    return result


def proof_trees(root, task, candidate, parent):
    root = Path(root) / 'candidates'
    return {'current': root / candidate / 'tree', 'parent': root / parent / 'tree',
            'baseline': root / task['optimization']['baseline'] / 'tree'}


def read_payload(row):
    import json
    return json.loads(row['payload'])


def bind_worker(root, control, payload, proof, runtime):
    """Freeze small proof inputs; mount the parent output read-only."""
    from .candidates import verify_snapshot
    root, control = Path(root), Path(control)
    source = root / 'runs' / proof['source_job']
    current = verify_snapshot(root / 'candidates' / payload['candidate'])
    parent = verify_snapshot(root / 'candidates' / proof['parent'])
    parent_plan, verified = read(source / 'control/plan.json'), read(source / 'verified/result.json')
    task = read(control / 'task.json')
    baseline = read(root / 'candidates' / task['optimization']['baseline'] / 'candidate.json')
    trees = proof_trees(root, task, payload['candidate'], proof['parent']) if proof.get('version') == 2 else None
    validate_proof(proof, task, payload, current, parent, parent_plan, verified, baseline, trees)
    physical_job, physical_output = physical_source(root, proof['source_job'])
    if physical_job != proof['physical_source_job']:
        raise ValueError('Incremental physical source alias changed')
    if runtime != proof['runtime'] or sha(source / 'control/plan.json') != verified['evidence']['plan_sha256']:
        raise ValueError('Incremental runtime or verified parent plan changed')
    for name, value in (('incremental', proof), ('candidate-source', current), ('parent-source', parent),
                        ('parent-plan', parent_plan), ('parent-verified', verified), ('baseline-source', baseline)):
        write(control / (name + '.json'), value)
    return physical_output


def verify_prefix(plan, task, case, root, config):
    from open3dbench.stage_manifest import validate_stage_manifest, digest_json
    from open3dbench.backend.contracts import validate_checkpoint
    from .frozen_frontend import verify
    control = Path(plan['incremental']); proof = read(control)
    proof_dir = control.parent
    # The replay container sees only the candidate tree; it may tighten, never relax, v2.
    trees = ({'current': plan['source']} if proof.get('version') == 2 and plan.get('source')
             and Path(plan['source']).is_dir() else None)
    validate_proof(proof, task, plan, read(proof_dir/'candidate-source.json'),
                   read(proof_dir/'parent-source.json'), read(proof_dir/'parent-plan.json'),
                   read(proof_dir/'parent-verified.json'), read(proof_dir/'baseline-source.json'), trees)
    root = Path(root)
    verify(task, case, root, config)
    upstream = 'none'
    for stage in ORDER[:ORDER.index(proof['impact']['start'])]:
        path = root / STAGES[stage] / 'stage_manifest.json'
        if sha(path) != proof['stages'][stage]:
            raise ValueError('Incremental stage manifest changed: ' + stage)
        manifest = validate_stage_manifest(path, expected_stage=stage, verify_artifacts=True)
        if any(manifest[k] != v for k, v in {'design': case['design'], 'pdk': case['pdk'], 'integration': 'lol'}.items()):
            raise ValueError('Incremental design/platform differs')
        if manifest['signature']['upstream'] != upstream:
            raise ValueError('Incremental upstream chain differs: ' + stage)
        upstream = sha(path)
        keys = {'input': ('input',), 'floorplan': ('floorplan', 'pins')}.get(stage, (stage,))
        expected = {k: config[k] for k in ('design', 'type', 'pdk')}
        expected.update(threads=config.get('threads', 32))
        expected.update({k: config[k] for k in keys if k in config})
        if manifest['signature']['config'] != digest_json(expected):
            raise ValueError('Incremental configuration differs: ' + stage)
        if stage in {'placement', 'pdn', 'hbt', 'opt', 'cts'}:
            validate_checkpoint(root / STAGES[stage] / 'checkpoint.json', require_files=True)
    if 'partition' in proof['stages']:
        if read(root/'20_partition/partition_manifest.json')['seed'] != plan['seed']['partition']:
            raise ValueError('Incremental partition seed differs')
    if 'placement' in proof['stages']:
        if read(root/'30_placement/stages/02_global_place/stage_manifest.json')['initialization_seed'] != plan['seed']['placement']:
            raise ValueError('Incremental placement seed differs')
    return proof


def materialize(plan, task, case, config):
    source, destination = Path(plan['incremental_source'])/'run', Path(plan['output'])/'run'
    proof = verify_prefix(plan, task, case, source, config)
    if destination.exists():
        raise FileExistsError('Incremental import requires a fresh run')
    temporary = Path(tempfile.mkdtemp(prefix='.incremental-', dir=destination.parent))
    try:
        for stage in proof['stages']:
            shutil.copytree(source/STAGES[stage], temporary/STAGES[stage])
        verify_prefix(plan, task, case, temporary, config)
        write(temporary/'run.json', {'design': case['design'], 'pdk': case['pdk'], 'type': 'lol',
            'status': 'running', 'incremental_proof_sha256': sha(plan['incremental']),
            'stages': {s: {'directory': STAGES[s], 'status': 'success',
                           'execution_mode': 'verified-parent-checkpoint',
                           'source_manifest_sha256': proof['stages'][s]} for s in proof['stages']}})
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    from open3dbench.config import load_config
    from open3dbench.runner import FlowRunner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--phase', choices=('frontend', 'backend'))
    args = parser.parse_args(); plan = read(args.plan); task = read(plan['task'])
    case = next(c for c in task['cases'] if c['id'] == plan['case'])
    config = load_config(Path(plan['output'])/'resolved.yaml')
    root = Path(plan['output'])/'run'; config['run_dir'] = str(root)
    if args.phase != 'backend':
        materialize(plan, task, case, config)
    proof = verify_prefix(plan, task, case, root, config)
    start = 'pdn' if args.phase == 'backend' else proof['impact']['start']
    if start != 'grt':
        FlowRunner(Path(plan['source']), config, resume=True).run(
            start=start, stop='placement' if args.phase == 'frontend' else 'cts')


if __name__ == '__main__':
    main()
