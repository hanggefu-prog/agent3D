from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.local_reference import LocalPartitionReference
from local_bench import reference_latest, reference_local_partition


@pytest.mark.parametrize('baseline,timing', [
    ('reference_20261004', '20_partition/timing_input'),
    ('reference_20260923', '15_timing_extract'),
])
def test_qualified_result_uses_verified_family_not_directory_guess(monkeypatch, baseline, timing):
    selection = object.__new__(LocalPartitionReference)
    selection.source_run = Path('/sealed/run')
    selection.runtime = SimpleNamespace(source=Path('/frozen/source'))
    selection.record_path = Path('/sealed/authority.json')
    selection.record_sha256 = 'a' * 64
    config = {'trusted_family': baseline}
    selection.contract = {'physical': {'configuration': config}}
    selection.verify_selected = lambda: {'reference_id':'reference', 'qualification': {
        'baseline': {'native_runtime_historical': {'seconds': 1}}}}
    def choose(value):
        assert value is config
        return SimpleNamespace(BASELINE_ID=baseline)
    monkeypatch.setattr(reference_latest, 'baseline_for_config', choose)
    calls = []
    def score(run, source, *, timing_directory):
        calls.append((run, source, timing_directory))
        return {'combined_score': 1.0}
    monkeypatch.setattr(reference_local_partition, 'independent_partition_metrics', score)
    result = selection.qualified_result(origin='fresh')
    assert calls == [(selection.source_run, selection.runtime.source, timing)]
    assert result.metrics['combined_score'] == 1.0


def test_unknown_family_cannot_fallback_to_historical_path(monkeypatch):
    selection = object.__new__(LocalPartitionReference)
    selection.contract = {'physical': {'configuration': {}}}
    selection.verify_selected = lambda: {}
    def reject(_):
        raise ValueError('Unknown reference family')
    monkeypatch.setattr(reference_latest, 'baseline_for_config', reject)
    monkeypatch.setattr(reference_local_partition, 'independent_partition_metrics',
        lambda *a, **k: pytest.fail('Unknown family must not dispatch scoring'))
    with pytest.raises(ValueError, match='Unknown reference family'):
        selection.qualified_result(origin='fresh')
