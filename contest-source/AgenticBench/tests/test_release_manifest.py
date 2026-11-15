import copy
import json
from pathlib import Path
import tempfile
import unittest
from local_bench import reference_latest as latest
from scripts.verify_release import checked_json, validate_contract


class ReleaseManifestTest(unittest.TestCase):
    def setUp(self):
        self.lock = {'tested_commit': 'a'*40, 'qualified_reference': {'source_revision': latest.SOURCE_REVISION}}
        self.data = {'schema_version': 1, 'kind': 'agenticbench-server-release-v1',
                     'baseline_id': latest.BASELINE_ID, 'source_revision': latest.SOURCE_REVISION,
                     'native_revision': latest.NATIVE_REVISION, 'sta_revision': latest.STA_REVISION,
                     'canonical_image': latest.IMAGE_ID, 'development_revision': 'a'*40,
                     'reference_policy': 'ADOPT', 'fallback': 'none'}

    def test_exact_contract(self):
        validate_contract(self.data, self.lock)

    def test_mismatches_fail_closed(self):
        for key in self.data:
            with self.subTest(key=key), self.assertRaises(ValueError):
                data = copy.deepcopy(self.data)
                data[key] = 'wrong'
                validate_contract(data, self.lock)

    def test_lock_cannot_relabel_reference(self):
        self.lock['qualified_reference']['source_revision'] = 'a'*40
        with self.assertRaises(ValueError): validate_contract(self.data, self.lock)

    def test_hash_and_symlink_checks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d).resolve()/'manifest.json'
            p.write_text(json.dumps(self.data))
            binding = {'path': str(p), 'sha256': latest.sha256(p)}
            self.assertEqual(checked_json(binding), self.data)
            p.write_text('{}')
            with self.assertRaises(ValueError): checked_json(binding)
            alias = p.parent/'alias.json'
            alias.symlink_to(p.name)
            with self.assertRaises(ValueError): checked_json({'path':str(alias),'sha256':latest.sha256(p)})
