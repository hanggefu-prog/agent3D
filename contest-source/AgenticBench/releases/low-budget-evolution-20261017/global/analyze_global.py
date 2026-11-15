"""Independently recompute the frozen global objectives from saved raw cells."""
import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
x = json.loads((root / 'global-evidence.json').read_text())
contract = x['contract']
cases = [c for c in contract['cases'] if c['split'] == 'train']
seeds = [s['id'] for s in contract['seeds']]
expected = {(c['id'], s) for c in cases for s in seeds}
cells = {}
for cell in x['cells']:
    group = cells.setdefault(cell['candidate'], {})
    key = (cell['case'], cell['seed']['id'])
    assert key not in group
    group[key] = cell['metrics']
reference = cells[x['baseline']]
assert set(reference) == expected
rows = []
for archived in x['archive']['evaluations']:
    candidate = archived['candidate']
    measurements = cells[candidate]
    assert set(measurements) == expected
    per_case, ratios = {}, {}
    for case in cases:
        name = case['id']
        means = lambda group: {m: statistics.mean(group[name, s][m] for s in seeds)
                               for m in contract['normalization_floors']}
        actual, baseline = means(measurements), means(reference)
        per_case[name] = {'candidate_mean': actual, 'baseline_mean': baseline,
                          'power_ratio': actual['power_w'] / baseline['power_w']}
    for metric, floor in contract['normalization_floors'].items():
        ratios[metric] = sum(c['weight'] * statistics.mean(
            (measurements[c['id'], s][metric] + floor) /
            (reference[c['id'], s][metric] + floor) for s in seeds) for c in cases
        ) / sum(c['weight'] for c in cases)
    weights = contract['scoring']['timing_weights']
    vector = {'timing': sum(w * ratios[m] for m, w in weights.items()) / sum(weights.values()),
              'overflow': ratios['overflow'], 'temperature_peak_k': ratios['temperature_peak_k']}
    feasible = all(v['power_ratio'] <= contract['scoring']['power_max_ratio'] for v in per_case.values())
    assert all(math.isclose(v, archived['vector'][m], rel_tol=0, abs_tol=1e-12) for m, v in vector.items())
    assert feasible == archived['feasible']
    rows.append({'candidate': candidate, 'vector': vector, 'feasible': feasible,
                 'per_case': per_case, 'archive_matches_independent_recomputation': True})
frontier = [r['candidate'] for r in rows if r['feasible'] and not any(
    a['feasible'] and all(a['vector'][m] <= r['vector'][m] for m in r['vector'])
    and any(a['vector'][m] < r['vector'][m] for m in r['vector']) for a in rows)]
checks = [v for c in x['cells'] for v in c.get('reuse_checks', {}).values()]
assert all(all(v.values()) for v in checks)
result = {'evaluations': rows, 'independent_frontier': frontier,
          'verified_upstream_stage_reuse_checks': len(checks),
          'all_reuse_checks_passed': True}
(root / 'global-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'complete_candidates': len(rows), 'frontier': frontier,
                  'upstream_stage_reuse_checks': len(checks)}))
