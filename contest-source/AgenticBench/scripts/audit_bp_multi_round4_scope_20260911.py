"""Remove behaviorally unchanged out-of-scope sections; emit immutable audit artifacts."""
import ast
import copy
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import audit_swerv_response_normalization_20260910 as a

run = a.BATCH / 'bp_multi/agentic3d'
folder = run / 'rounds/04/inner/01'
assert not (folder / 'response.json').exists()
assert not (run / 'rounds/04/outer_result.json').exists()
source = folder / 'response.validation.json'
response = json.loads(source.read_text())['last_rejected_response']
plan = json.loads((run / 'rounds/04/plan.json').read_text())
incumbent = json.loads((run / 'rounds/02/outer-candidate/candidate.json').read_text())
spec = a.d.load_task(a.d.STA_PROFILE)
selected = a.task_spec(plan['local_task'], spec)
candidate = a.d.materialize_program(spec, response['program'], None)
sections = a.d._patch_sections(candidate['source_patch'])
scope = set(plan['local_task']['source_scope'])
assert scope == {'Place-3D/src/partition/lol/methods/tp_gnn.py'}
normalized = copy.deepcopy(candidate)
normalized['source_patch'] = ''.join(text for path, text in sections.items() if path in scope)
merged = a.d._merge_candidate(incumbent, normalized, plan['hypothesis'], replace_paths=scope)
a.d.verify_candidate(selected, normalized)
a.d.verify_candidate(spec, merged)
a.decode_updates(response, spec)
destination = a.ROOT / 'tmp/sta_bp_multi_r4_scope_recovery'
for label, item in [('original', candidate), ('incumbent', incumbent), ('normalized', normalized), ('merged', merged)]:
    a.d._check_source_patch_applies(run / 'agent-workspace', item['source_patch'], destination / label)
proof = {}
with tempfile.TemporaryDirectory(prefix='sta-r4-isolated-audit-') as tmp:
    trees = {}
    for label, item in [('original', candidate), ('merged', merged)]:
        tree = Path(tmp) / label
        tree.mkdir()
        trees[label] = tree
        for path in a.d._patch_sections(item['source_patch']):
            target = tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / 'agent-workspace' / path, target)
        subprocess.run(['git', 'apply', '--no-index', str(destination / label / 'source-apply-check.patch')],
                       cwd=tree, check=True, capture_output=True)
    for path in sections:
        original = (trees['original'] / path).read_text()
        corrected = (trees['merged'] / path).read_text()
        assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(corrected))
        if original != corrected:
            assert path.endswith('/PlaceObj.py')
            old = '# its cells retain wirelength and fixed-density gradients.  Overflow'
            new = '# its cells retain wirelength and fixed-density gradients. Overflow'
            assert corrected.count(old) == 1 and corrected.replace(old, new) == original
        proof[path] = {'original_sha256': a.sha(original), 'merged_sha256': a.sha(corrected),
                       'byte_equal': original == corrected, 'python_ast_equal': True}
repaired = copy.deepcopy(response)
repaired['program'] = a.render_candidate_program(normalized)
assert a.d.materialize_program(selected, repaired['program'], None) == normalized
audit = {'case': 'bp_multi', 'round': 4, 'inner_attempt': 1,
         'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'type': 'representation_only_scope_normalization',
         'source': str(source.relative_to(a.ROOT)), 'source_sha256': a.sha(source.read_text()),
         'original_patch_sha256': a.sha(candidate['source_patch']),
         'normalized_local_patch_sha256': a.sha(normalized['source_patch']),
         'removed_out_of_scope': sorted(set(sections) - scope),
         'source_proof': proof,
         'checks': ['Strict frozen local/global scope, limits, apply and knowledge checks passed',
                    'Partition source byte-identical before/after recovery',
                    'NonLinearPlace source byte-identical; PlaceObj differs only by one extra comment space',
                    'Every resulting file has identical Python AST; isolated application and syntax passed',
                    'Hypothesis/knowledge unchanged; static wrapper round trip passed'],
         'limits': 'No algorithm edit or behavioral evaluation; no extra model call. Original traces, frozen framework, judge and physical budgets unchanged. Disclose manual interface recovery in experiment reporting.'}
print(json.dumps({'response': repaired, 'audit': audit}))
