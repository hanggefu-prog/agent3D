"""Audited hunk-header-only normalization; emit artifacts without changing a run."""
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
folder = run / 'rounds/03/inner/01'
assert not (folder / 'response.json').exists()
assert not (folder / 'evaluation.json').exists()
source = folder / 'response.validation.json'
response = json.loads(source.read_text())['last_rejected_response']
plan = json.loads((run / 'rounds/03/plan.json').read_text())
incumbent = json.loads((run / 'rounds/02/outer-candidate/candidate.json').read_text())
spec = a.d.load_task(a.d.STA_PROFILE)
selected = a.task_spec(plan['local_task'], spec)
candidate = a.d.materialize_program(selected, response['program'], None)
sections = a.d._patch_sections(candidate['source_patch'])
assert set(sections) <= set(plan['local_task']['source_scope'])
normalized = copy.deepcopy(candidate)
parts, changes = [], {}
for path, section in sections.items():
    text, edits = a.recount(section)
    parts.append(text)
    changes[path] = edits
normalized['source_patch'] = ''.join(parts)
a.d.verify_candidate(selected, normalized)
merged = a.d._merge_candidate(incumbent, normalized, plan['hypothesis'],
                              replace_paths=set(plan['local_task']['source_scope']))
a.d.verify_candidate(spec, merged)
a.decode_updates(response, spec)
destination = a.ROOT / 'tmp/sta_bp_multi_header_recovery_20260911'
a.d._check_source_patch_applies(run / 'agent-workspace', normalized['source_patch'], destination / 'local')
a.d._check_source_patch_applies(run / 'agent-workspace', merged['source_patch'], destination / 'merged')
with tempfile.TemporaryDirectory(prefix='sta-bpm-patch-check-') as tmp:
    tree = Path(tmp)
    resulting = {}
    for path in a.d._patch_sections(merged['source_patch']):
        target = tree / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / 'agent-workspace' / path, target)
    subprocess.run(['git', 'apply', '--no-index', str(destination / 'merged/source-apply-check.patch')],
                   cwd=tree, check=True, capture_output=True)
    for path in a.d._patch_sections(merged['source_patch']):
        content = (tree / path).read_text()
        ast.parse(content, filename=path)
        resulting[path] = a.sha(content)
repaired = copy.deepcopy(response)
repaired['program'] = a.render_candidate_program(normalized)
assert a.d.materialize_program(selected, repaired['program'], None) == normalized
audit = {
    'case': 'bp_multi', 'round': 3, 'inner_attempt': 1,
    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'type': 'hunk_headers_only_normalization_after_network_failure',
    'source': str(source.relative_to(a.ROOT)), 'source_sha256': a.sha(source.read_text()),
    'original_patch_sha256': a.sha(candidate['source_patch']),
    'normalized_patch_sha256': a.sha(normalized['source_patch']),
    'hunk_header_changes': changes, 'resulting_source_sha256': resulting,
    'checks': ['All patch context/addition/deletion lines unchanged',
               'Original scope/line limits and knowledge checks passed',
               'Strict local/global git apply --check passed',
               'Isolated global application and Python AST parse passed',
               'Static wrapper round trip passed'],
    'limits': 'No behavioral test or algorithm edits; original traces retained; frozen framework/judge/budgets unchanged; manual interface intervention must be disclosed. No model call added by recovery.'
}
print(json.dumps({'response': repaired, 'audit': audit}))
