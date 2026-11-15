"""Merged engineering contracts: offline compatibility and prompt/gate binding."""
import importlib
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engineering_bench/physical'))

from engineering_bench.model import developer_prompt
from engineering_bench.protocol import digest, sha
from engineering_bench.scoring import (CURRENT_SCORING_VERSION, development_instruction,
    qualification_fields, qualified_contract, scoring_prompt, selected_contract,
    validate_development_binding)
from engineering_bench.tasks import TASK_IDS, load_task


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_frozen_default_and_explicit_v2_remain_distinct(task_id):
    task = load_task(task_id)
    assert CURRENT_SCORING_VERSION == 1
    assert qualification_fields(task) == {}
    assert selected_contract(task)['scoring_version'] == 1
    v2 = selected_contract(task, 2)
    assert qualification_fields(task, 2)['scoring_profile_sha256'] == v2['scoring_profile_sha256']
    # The frozen task, including its original instructions, stays unchanged.
    old_digest = task.digest()
    for version in (1, 2):
        prompt = developer_prompt(task, version)
        assert 'Scoring version: %d.' % version in prompt
    assert task.digest() == old_digest


def test_cts_prompt_uses_the_actual_profile_and_no_conflicting_legacy_gate(tmp_path):
    task = load_task('cross_die_cts')
    legacy = developer_prompt(task)
    assert 'Setup/hold is diagnostic only' in legacy and 'at most 2x and 2x' in legacy
    assert 'diagnostics, not a second objective or an acceptance gate' in legacy
    modern = developer_prompt(task, 2)
    assert 'diagnostics, not a second objective or an acceptance gate' not in modern
    assert '"clock_area_max_reference_ratio": 1.25' in modern
    assert '"absolute_tolerance_ns": 0.01' in modern and '"relative_tolerance": 0.02' in modern
    assert '"setup_tns_ns": 0.1' in modern and '"max_reference_ratio": 1.1' in modern
    shutil.copytree(task.root, tmp_path / task.task_id)
    copied = load_task(task.task_id, tmp_path)
    path = copied.root / 'evaluation-v2.json'
    value = json.loads(path.read_text())
    value['gates']['clock_area_max_reference_ratio'] = 1.4
    path.write_text(json.dumps(value))
    changed = developer_prompt(copied, 2)
    assert '"clock_area_max_reference_ratio": 1.4' in changed
    assert selected_contract(copied, 2)['scoring_profile_sha256'] == sha(path)


def test_t3_prompt_discloses_legacy_remote_vs_opt_in_all_receivers():
    task = load_task('multi_hbt_fanout')
    assert 'legacy contexts' in developer_prompt(task)
    modern = developer_prompt(task, 2)
    assert 'including local and remote receivers' in modern
    assert '"max_ratio_to_min_buffer_area_per_target": 4.0' in modern


def test_development_requires_same_scoring_version_on_every_case():
    task = load_task('cross_die_cts')
    v1 = {'scoring_version': 1, 'record': {}}
    profile = qualification_fields(task, 2)
    v2 = {'scoring_version': 2, 'record': profile}
    assert qualified_contract(task, [v1] * 4)['scoring_version'] == 1
    assert qualified_contract(task, [v2] * 4, 2)['scoring_version'] == 2
    for proofs, version in (([v2] * 4, 1), ([v1] * 3 + [v2], 2), ([], 1)):
        with pytest.raises(ValueError, match='another or mixed'):
            qualified_contract(task, proofs, version)
    with pytest.raises(ValueError, match='changed'):
        qualified_contract(task, [{'scoring_version': 2, 'record': dict(profile, scoring_profile_sha256='0' * 64)}], 2)


def test_campaign_version_is_opt_in_and_preflight_checks_case_qualifications(tmp_path, monkeypatch):
    import engineering_bench.campaign as campaign
    from engineering_bench.tasks import TASK_ROOT
    plan = campaign.load_campaign(TASK_ROOT / 'campaign.gpt56.json')
    assert 'scoring_version' not in plan
    def qualification(task, case, path):
        return {'sha256': 'a' * 64, 'scoring_version': 1}
    monkeypatch.setattr(campaign, 'qualify', qualification)
    assert campaign.preflight(plan, tmp_path)['ready']
    opt_in = dict(plan, scoring_version=2)
    path = tmp_path / 'campaign.json'
    path.write_text(json.dumps(opt_in))
    assert campaign.load_campaign(path)['scoring_version'] == 2
    assert not campaign.preflight(opt_in, tmp_path)['ready']
    monkeypatch.setattr(campaign, 'qualify', lambda task, case, path: {'sha256': 'b' * 64, 'scoring_version': 2})
    assert campaign.preflight(opt_in, tmp_path)['ready']
    path.write_text(json.dumps(dict(plan, scoring_version=True)))
    with pytest.raises(ValueError, match='scoring version'):
        campaign.load_campaign(path)


def packet(tmp_path, task, version):
    subject = tmp_path / 'source'
    subject.mkdir()
    folder = tmp_path / 'control'
    folder.mkdir()
    contract = selected_contract(task, version)
    path = folder / 'scoring-contract.json'
    path.write_text(json.dumps(contract))
    prompt = folder / 'effective-prompt.txt'
    prompt.write_text(development_instruction(task, contract) + '\n' + scoring_prompt(task, contract))
    environment = {'scoring_contract_sha256': sha(path), 'scoring_contract_digest': digest(contract),
                   'prompt_sha256': sha(prompt)}
    (folder / 'environment.json').write_text(json.dumps(environment))
    return subject, folder, contract


@pytest.mark.parametrize('tamper', ['version', 'profile', 'prompt', 'binding'])
def test_final_rejects_different_version_or_changed_development_packet(tmp_path, tamper):
    task = load_task('cross_die_cts')
    subject, folder, contract = packet(tmp_path, task, 2)
    assert validate_development_binding(task, subject, 2) == contract
    version = 2
    if tamper == 'version':
        version = 1
    elif tamper == 'profile':
        contract['gates']['clock_area_max_reference_ratio'] = 2.0
        (folder / 'scoring-contract.json').write_text(json.dumps(contract))
    elif tamper == 'prompt':
        (folder / 'effective-prompt.txt').write_text('Use legacy guards instead')
    else:
        (folder / 'environment.json').write_text('{}')
    with pytest.raises(ValueError, match='differ|changed'):
        validate_development_binding(task, subject, version)


def test_hand_authored_source_and_legacy_packet_remain_supported(tmp_path):
    task = load_task('multi_hbt_fanout')
    subject = tmp_path / 'source'
    subject.mkdir()
    assert validate_development_binding(task, subject, 1)['scoring_version'] == 1
    assert validate_development_binding(task, subject, 2)['scoring_version'] == 2
    folder = tmp_path / 'control'
    folder.mkdir()
    (folder / 'effective-prompt.txt').write_text('The original frozen developer prompt')
    assert validate_development_binding(task, subject, 1)['scoring_version'] == 1
    with pytest.raises(ValueError, match='Legacy development prompt'):
        validate_development_binding(task, subject, 2)


@pytest.mark.parametrize('adapter', ['clock_final_queue', 'multi_final_queue', 'cts_final_queue'])
def test_final_cli_defaults_to_v1_before_native_dispatch(tmp_path, monkeypatch, adapter):
    module = importlib.import_module(adapter)
    calls = []
    class BeforeNative(Exception):
        pass
    def check(task, subject, version):
        calls.append(version)
        raise BeforeNative
    monkeypatch.setattr(module, 'validate_development_binding', check)
    args = [item for key in ('root', 'experiment', 'output', 'subject', 'baseline-inventory')
            for item in ('--' + key, str(tmp_path / key))] + ['--cases', 'bp_fe']
    if adapter == 'multi_final_queue':
        args += ['--context-root', str(tmp_path)]
    with pytest.raises(BeforeNative):
        module.main(args)
    with pytest.raises(BeforeNative):
        module.main(args + ['--scoring-version', '2'])
    assert calls == [1, 2]
