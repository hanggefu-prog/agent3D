import json

import pytest

from engineering_bench import prepare_development, prepare_native


def test_native_input_audit_accepts_verified_v2_source_release(tmp_path, monkeypatch):
    path = tmp_path / 'release.json'
    path.write_text(json.dumps({'kind': 'engineering-source-release-v2'}))
    observed = []
    def verify(release):
        observed.append(release)
        return tmp_path, 'sha256:' + 'a' * 64
    monkeypatch.setattr(prepare_development, 'verify_release', verify)
    assert prepare_native.release_image(path) == 'sha256:' + 'a' * 64
    assert observed == [path]


def test_v2_verification_failure_is_not_a_legacy_fallback(tmp_path, monkeypatch):
    path = tmp_path / 'release.json'
    path.write_text(json.dumps({'kind': 'engineering-source-release-v2'}))
    def reject(release):
        raise ValueError('changed runtime qualification')
    monkeypatch.setattr(prepare_development, 'verify_release', reject)
    with pytest.raises(ValueError, match='changed runtime'):
        prepare_native.release_image(path)


def test_unknown_engineering_release_kind_still_rejected(tmp_path):
    path = tmp_path / 'release.json'
    path.write_text(json.dumps({'kind': 'unverified'}))
    with pytest.raises(ValueError, match='Unknown installed release'):
        prepare_native.release_image(path)
