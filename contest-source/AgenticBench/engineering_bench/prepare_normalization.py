"""Qualify the protected T4 legalization policy against real native baselines.

This is a diagnostic, not a task qualification or a candidate final. Each case
is processed serially, with at most one bounded physical process on this host.
"""
import argparse
import json
from pathlib import Path

from .clock_equivalence import compare
from .legalization import normalize, load_snapshot, ports_geometry, is_fixed
from .native_snapshot import snapshot
from .prepare_native import release_image
from .protocol import write_new, now, sha


def run(*, reference_root, platform, audit, output, image, cases):
    output, audit = Path(output), Path(audit)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'dispatch.json', dict(kind='engineering-normalization-diagnostic-v1',
        started_at=now(), image=image, cases=cases, candidate_finals=0, model_calls=0))
    rows = []
    for case in cases:
        root = Path(reference_root) / case / 'run'
        original = {d: audit / case / '60_opt' / d for d in ('bottom', 'top')}
        baseline = {d: audit / case / '70_cts' / d for d in ('bottom', 'top')}
        try:
            old_proof = compare(original, baseline, root / '60_opt/hbt.csv')
            normalized, displacement = {}, {}
            for die in ('bottom', 'top'):
                folder = output / case / die
                result = normalize(original_snapshot=original[die], candidate_snapshot=baseline[die],
                    candidate_odb=root / '70_cts' / die / 'design.odb', output=folder, image=image)
                if result['status'] != 'succeeded':
                    raise ValueError('Native normalization failed: ' + str(folder / 'native.log'))
                readback = output / case / (die + '-readback')
                observed = snapshot(run=root, platform=platform, image=image, stage='70_cts', die=die,
                    output=readback, clock_sdc=root / '60_opt' / die / 'design.sdc',
                    odb_path=folder / 'native/design.odb')
                if observed['status'] != 'succeeded':
                    raise ValueError('Independent normalization readback failed: ' + str(readback))
                normalized[die] = readback
                _, _, before = load_snapshot(original[die])
                _, _, after = load_snapshot(readback)
                if ports_geometry(original[die] / 'ports.tsv') != ports_geometry(readback / 'ports.tsv'):
                    raise ValueError('Port geometry changed after native legalization')
                moved = []
                for name, cell in before.items():
                    if cell['clock_cell']:
                        continue
                    new = after[name]
                    changed = any(new[k] != cell[k] for k in ('x_dbu', 'y_dbu', 'orient'))
                    if changed and is_fixed(cell):
                        raise ValueError('Native legalizer moved a fixed macro/HBT: ' + name)
                    if changed:
                        moved.append(name)
                displacement[die] = dict(moved_original_standard_cells=len(moved))
            proof = compare(original, normalized, root / '60_opt/hbt.csv')
            row = dict(case=case, status='succeeded', baseline_logic_proof=old_proof,
                       normalized_logic_proof=proof, displacement=displacement,
                       input_hbt_sha256=sha(root / '60_opt/hbt.csv'))
        except (OSError, ValueError, KeyError) as exc:
            row = dict(case=case, status='failed', error=str(exc))
        rows.append(row)
        write_new(output / (case + '.json'), row)
        print(json.dumps(row), flush=True)
    report = dict(completed_at=now(), cases=rows, task_qualification=False, candidate_finals=0, model_calls=0)
    write_new(output / 'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('reference_root', 'platform', 'audit', 'output', 'release'):
        parser.add_argument('--' + key.replace('_', '-'), required=True)
    parser.add_argument('--cases', nargs='+', required=True,
                        choices=('bp_fe', 'bp', 'ariane133', 'swerv_wrapper'))
    values = vars(parser.parse_args(argv))
    if len(values['cases']) != len(set(values['cases'])):
        parser.error('Repeated cases are not allowed')
    values['image'] = release_image(values.pop('release'))
    result = run(**values)
    return 0 if all(r['status'] == 'succeeded' for r in result['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
