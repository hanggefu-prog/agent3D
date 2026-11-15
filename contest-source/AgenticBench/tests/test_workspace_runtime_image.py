from types import SimpleNamespace
import pytest

from scripts import workspace_runtime_image as module
from local_bench.reference_latest import IMAGE_ID, BASELINE_ID


def test_canonical_direct_runtime_unchanged():
    assert module.execution_image(SimpleNamespace(image=IMAGE_ID)) == IMAGE_ID


def test_explicit_installed_runtime_reverified(monkeypatch):
    calls = []
    binding = {'release': {'path': '/release.json', 'sha256': 'a' * 64},
               'actual_image': 'sha256:classic'}
    def check(*args, **kwargs):
        calls.append((args, kwargs))
        return binding
    monkeypatch.setattr(module, 'bind_release', check)
    runtime = SimpleNamespace(image=IMAGE_ID, source='/qualified', reference_baseline=BASELINE_ID,
                              workspace_runtime_image=binding)
    assert module.execution_image(runtime) == 'sha256:classic'
    assert calls[0][1] == {'source': '/qualified', 'canonical': IMAGE_ID}
    monkeypatch.setattr(module, 'bind_release', lambda *a, **kw: {**binding, 'actual_image': 'changed'})
    with pytest.raises(ValueError, match='changed'):
        module.execution_image(runtime)


def test_equivalence_never_applies_to_legacy(monkeypatch):
    binding = {'release': {'path': '/release.json', 'sha256': 'a' * 64}, 'actual_image': 'classic'}
    monkeypatch.setattr(module, 'bind_release', lambda *a, **kw: binding)
    with pytest.raises(ValueError, match='latest family'):
        module.execution_image(SimpleNamespace(image=IMAGE_ID, source='/qualified', reference_baseline='legacy',
            workspace_runtime_image=binding))


def test_wrong_canonical_identity_rejected_before_external_check():
    with pytest.raises(ValueError, match='canonical'):
        module.bind_release('/missing', 'a' * 64, source='/missing', canonical='sha256:other')
