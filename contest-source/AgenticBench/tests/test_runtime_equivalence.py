import hashlib
import json
import pytest
from local_bench import reference_latest as latest
from local_bench.runtime_equivalence import CLASSIC_ID, validate_base_equivalence, validate_latest_build_base


@pytest.fixture
def proof(tmp_path):
    path = tmp_path / 'proof.json'
    data = dict(status='verified', canonical_oci_digest=latest.IMAGE_ID,
                docker_config_image_id=CLASSIC_ID, binaries_sha256=latest.BINARY_SHA256,
                rootfs_layers=['sha256:' + 'a' * 64])
    path.write_text(json.dumps(data))
    return path, data


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_classic_base_keeps_both_identities(proof):
    path, data = proof
    inspected = [dict(Id=CLASSIC_ID, RootFS=dict(Layers=data['rootfs_layers']))]
    checked = validate_base_equivalence(binding(path), CLASSIC_ID, inspected)
    assert validate_latest_build_base(dict(base_image_id=CLASSIC_ID, base_runtime_equivalence=checked)) == CLASSIC_ID
    assert validate_latest_build_base(dict(base_image_id=latest.IMAGE_ID)) == latest.IMAGE_ID


@pytest.mark.parametrize('field,value', [('status', 'unknown'), ('canonical_oci_digest', CLASSIC_ID),
    ('docker_config_image_id', latest.IMAGE_ID), ('binaries_sha256', {}), ('rootfs_layers', [])])
def test_bad_proof_rejected(proof, field, value):
    path, data = proof
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_base_equivalence(binding(path), CLASSIC_ID)


def test_changed_hash_layers_missing_and_arbitrary_base_rejected(proof):
    path, data = proof
    original = binding(path)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError):
        validate_base_equivalence(original, CLASSIC_ID)
    with pytest.raises(ValueError):
        validate_base_equivalence(binding(path), CLASSIC_ID, [dict(Id=CLASSIC_ID, RootFS=dict(Layers=[]))])
    with pytest.raises(ValueError):
        validate_latest_build_base(dict(base_image_id=CLASSIC_ID))
    with pytest.raises(ValueError):
        validate_base_equivalence(binding(path), 'sha256:' + 'b' * 64)
    with pytest.raises(ValueError):
        validate_latest_build_base(dict(base_image_id=latest.IMAGE_ID, base_runtime_equivalence=binding(path)))
