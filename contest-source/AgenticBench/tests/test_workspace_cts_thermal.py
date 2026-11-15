"""Current thermal endpoint boundaries; no Docker, EDA or provider calls."""
import json
from pathlib import Path

import pytest

from scripts import workspace_cts_thermal as endpoint
from scripts.workspace_task_selection import (LATEST_BASELINE, LATEST_THERMAL_TASKS,
    LATEST_THERMAL_PROFILE, load_registered_task, workspace_task_id)
from scripts.workspace_codesign_evaluator import endpoint_for_task
from global_bench.verifiers.open3dbench_codesign_verifier import _cts_thermal_score_rows
from global_bench.model import load_global_task


@pytest.mark.parametrize("rounds", [5, 10])
def test_latest_thermal_round_contract(rounds):
    name = workspace_task_id("thermal", rounds, baseline=LATEST_BASELINE)
    assert name == LATEST_THERMAL_TASKS[rounds]
    task = load_registered_task(name, objective="thermal")
    contract = endpoint_for_task("thermal", task)
    assert contract["profile"] == LATEST_THERMAL_PROFILE
    assert contract["opt_passes"] == 1 and contract["backend_stop"] == "cts"
    assert task.budget.global_rounds == task.budget.expensive_evaluations == rounds
    assert task.evaluation.stage_ids()[-1] == "70_cts"
    assert "15_timing_extract" not in task.evaluation.stage_ids()


@pytest.mark.parametrize("rounds", [0, 4, 11, True])
def test_no_budget_fallback(rounds):
    with pytest.raises(ValueError):
        workspace_task_id("thermal", rounds, baseline=LATEST_BASELINE)


def environment():
    values = {key: "/immutable/file" for key in endpoint.INPUTS | set(endpoint.OUTPUTS) | endpoint.SCALARS}
    values.update(CTS_QUERY_ALL_BOUNDARIES="1", CTS_EXPECTED_HBTS="3")
    return "".join("set ::env({}) {{{}}}\n".format(k, v) for k, v in values.items())


def test_sta_body_preserved():
    assert endpoint.replay_environment(environment() + "trusted body", "trusted body")["CTS_EXPECTED_HBTS"] == "3"


@pytest.mark.parametrize("edit", [lambda s: "exec touch /tmp/no\n" + s,
    lambda s: s + "puts changed\n", lambda s: s.replace("{3}", "{0}"),
    lambda s: s.replace("{3}", "{3}\nset ::env(CTS_EXPECTED_HBTS) {3}"),
    lambda s: s.replace("CTS_PASSIVE_V", "UNKNOWN")])
def test_untrusted_sta_prefix_rejected(edit):
    with pytest.raises(ValueError):
        endpoint.replay_environment(edit(environment()) + "trusted body", "trusted body")


DEF = '''UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 10000 ) ;
COMPONENTS 2 ;
- a cell
  + SOURCE DIST + PLACED ( 10 20 ) N ;
- b macro + FIXED ( 30 40 ) FN ;
END COMPONENTS
'''


def test_def_normalization_preserves_coordinates(tmp_path):
    src, dst = tmp_path / "raw.def", tmp_path / "canonical.def"
    src.write_text(DEF)
    receipt = endpoint.normalize_def(src, dst)
    assert receipt["components"] == 2
    assert "- a cell + PLACED ( 10 20 ) N ;" in dst.read_text()
    assert "- b macro + FIXED ( 30 40 ) FN ;" in dst.read_text()
    assert src.read_text() == DEF


@pytest.mark.parametrize("edit", [lambda s: s.replace("COMPONENTS 2", "COMPONENTS 3"),
    lambda s: s.replace("- b macro", "- a macro"), lambda s: s.replace("FIXED", "UNPLACED"),
    lambda s: s.replace("MICRONS 1000", "MICRONS 0")])
def test_invalid_def_is_not_silently_dropped(tmp_path, edit):
    src = tmp_path / "raw.def"; src.write_text(edit(DEF))
    with pytest.raises(ValueError):
        endpoint.normalize_def(src, tmp_path / "normalized.def")


def tiles():
    return ["layer_{}_{}_{}_{} 320".format(layer, die, x, y)
        for layer, die in ((0, "top"), (2, "bottom")) for x in range(10) for y in range(10)]


def test_complete_thermal_tiles(tmp_path):
    path = tmp_path / "steady"; path.write_text("\n".join(tiles()))
    endpoint.validate_tiles(path)


@pytest.mark.parametrize("edit", [lambda rows: rows[:-1], lambda rows: rows + rows[:1],
    lambda rows: [rows[0].replace("320", "nan")] + rows[1:],
    lambda rows: [rows[0].replace("320", "310")] + rows[1:]])
def test_partial_or_invalid_temperature_rejected(tmp_path, edit):
    path = tmp_path / "steady"; path.write_text("\n".join(edit(tiles())))
    with pytest.raises(ValueError):
        endpoint.validate_tiles(path)


def test_power_conservation_and_name_decoding(tmp_path):
    instances, global_power = tmp_path / "instances", tmp_path / "global.json"
    instances.write_text("1 0 0 1 u_bottom/a\n1 0 0 1 u_top/b\n")
    global_power.write_text(json.dumps({"Total": {"internal": 2, "switching": 0, "leakage": 0, "total": 2}}))
    parser = lambda p: ({"bottom": {r"a\/b": 1}, "top": {"b": 1}}, 0)
    powers, _, difference = endpoint.validate_power(instances, global_power, parser)
    assert powers["bottom"] == {"a/b": 1} and difference == 0
    instances.write_text("nan 0 0 nan u_bottom/a\n")
    with pytest.raises(ValueError):
        endpoint.validate_power(instances, global_power, parser)


def test_continuous_passive_score_does_not_claim_die_local_power():
    row = {"stack": {}, "cts_algorithm": "cts", "boundary_budget_model": "passive-cutpoint",
           "power_model": "continuous-passive-hbt-placement-estimated", "global_average_temperature_k": 320.0}
    result = _cts_thermal_score_rows(row, row, 1)
    assert result["combined_score"] == 1 and result["opt_passes"] == 1
    assert result["power_model"] == row["power_model"]
    with pytest.raises(ValueError):
        _cts_thermal_score_rows(row, {**row, "power_model": "die-local-openroad-placement-estimated"}, 1)


def snapshot_fixture(tmp_path):
    root = tmp_path / "snapshot"; root.mkdir()
    checkpoint = tmp_path / "70_cts/checkpoint.json"
    checkpoint.parent.mkdir(); checkpoint.write_text('{}')
    old = checkpoint.parent / "assembly/passive/metrics.tsv"
    old.parent.mkdir(parents=True); old.write_text('wns_setup -1\n')
    required = ["metrics.json", "power.ptrace", "temperature.steady", "sta/power.json", "sta/instance_power.rpt",
        "sta/metrics.tsv", "sta/sta.log", "sta/replay.tcl", "stack.lcf", "top.flp", "bottom.flp",
        "hotspot.config", "materials.txt", "hotspot.log", "normalized_defs/top.def", "normalized_defs/bottom.def"]
    for name in required:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('')
    (root / "sta/metrics.tsv").write_text(old.read_text())
    (root / "sta/sta.log").write_text('OPEN3D_WORKSPACE_CTS_POWER_DONE\n')
    (root / "temperature.steady").write_text('\n'.join(tiles()))
    names = [f'{d}_{x}_{y}' for d in ('top', 'bottom') for x in range(10) for y in range(10)]
    (root / "power.ptrace").write_text(' '.join(names) + '\n' + ' '.join(['0.01'] * 200) + '\n')
    (root / "sta/instance_power.rpt").write_text('1 0 0 1 u_top/a\n1 0 0 1 u_bottom/b\n')
    (root / "sta/power.json").write_text(json.dumps({'Total': {'total': 2}}))
    metrics = {'stage': 'thermal_snapshot', 'status': 'success', 'endpoint': endpoint.ENDPOINT,
        'source_stage': 'cts', 'algorithm': 'hotspot-steady-grid', 'grid_size': 10, 'ambient_k': 318.15,
        'warning_count': 0, 'signoff': False, 'routed': False,
        'power': {'total_w': 2, 'provenance': {'model': 'continuous-passive-hbt-placement-estimated', 'checkpoint': str(checkpoint), 'threads': 4}},
        'stack': {'order_from_layer_zero': ['top-silicon', 'tim', 'bottom-silicon', 'tim'],
            'silicon_thickness_um': 150.0, 'tim_thickness_um': 20.0, 'model_status': 'provisional-shared-baseline'},
        'temperature': {'top': {'tile_count': 100, 'average_k': 320}, 'bottom': {'tile_count': 100, 'average_k': 320},
            'global_average_k': 320, 'global_maximum_c': 46.85}}
    (root / 'metrics.json').write_text(json.dumps(metrics))
    collateral = Path(__file__).resolve().parents[2] / 'OpenROAD-3D/flow/HotSpot/examples/thermal'
    for target, source in [('hotspot.config', 'test.config'), ('materials.txt', 'test.materials')]:
        (root / target).write_bytes((collateral / source).read_bytes())
    manifest = {'schema_version': 1, 'status': 'success', 'endpoint': endpoint.ENDPOINT, 'source_stage': 'cts',
        'implementation_sha256': endpoint.sha(endpoint.__file__),
        'physical_thermal_sha256': '250976bed5bc7560927019ef4709a01c79346f37b5668a889bacc1391a9fa4e8',
        'source_checkpoint': endpoint.record(checkpoint), 'inputs': {str(old): endpoint.record(old)},
        'artifacts': {name: endpoint.record(root / name) for name in required}}
    manifest['inputs'].update({name: {'path': name, 'bytes': size, 'sha256': value}
        for name, (size, value) in endpoint.IMAGE_ASSETS.items()})
    (root / 'snapshot_manifest.json').write_text(json.dumps(manifest))
    return root, checkpoint


def test_snapshot_consumer_recomputes_temperature_and_power(tmp_path):
    root, checkpoint = snapshot_fixture(tmp_path)
    row = endpoint.validate_snapshot(root, checkpoint)
    assert row['global_average_temperature_k'] == 320 and row['total_power_w'] == 2


@pytest.mark.parametrize('change', ['artifact', 'checkpoint', 'mean', 'power', 'sta', 'stack', 'missing', 'image-hash', 'image-missing'])
def test_snapshot_consumer_rejects_tampering(tmp_path, change):
    root, checkpoint = snapshot_fixture(tmp_path)
    if change == 'checkpoint':
        checkpoint.write_text('{"changed":true}')
    elif change == 'artifact':
        (root / 'temperature.steady').write_text('')
    else:
        manifest = json.loads((root / 'snapshot_manifest.json').read_text())
        metrics = json.loads((root / 'metrics.json').read_text())
        if change == 'mean':
            metrics['temperature']['global_average_k'] = 319
        elif change == 'power':
            metrics['power']['total_w'] = 3
        elif change == 'stack':
            metrics['stack']['silicon_thickness_um'] = 10
        elif change == 'sta':
            (root / 'sta/metrics.tsv').write_text('wns_setup -2\n')
            manifest['artifacts']['sta/metrics.tsv'] = endpoint.record(root / 'sta/metrics.tsv')
        elif change == 'missing':
            del manifest['artifacts']['stack.lcf']
        elif change == 'image-hash':
            manifest['inputs']['/opt/open3dbench-assets/hotspot/hotspot']['sha256'] = '0' * 64
        elif change == 'image-missing':
            del manifest['inputs']['/opt/open3dbench-assets/hotspot/hotspot']
        (root / 'metrics.json').write_text(json.dumps(metrics))
        manifest['artifacts']['metrics.json'] = endpoint.record(root / 'metrics.json')
        (root / 'snapshot_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        endpoint.validate_snapshot(root, checkpoint)


def test_reviewed_original_native_output_survives_consumer_only_fix(tmp_path):
    root, checkpoint = snapshot_fixture(tmp_path)
    path = root / 'snapshot_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['implementation_sha256'] = endpoint.REVIEWED_PRODUCER_SHA256
    path.write_text(json.dumps(manifest))
    assert endpoint.validate_snapshot(root, checkpoint)['total_power_w'] == 2
    manifest['implementation_sha256'] = '0' * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='implementation'):
        endpoint.validate_snapshot(root, checkpoint)
