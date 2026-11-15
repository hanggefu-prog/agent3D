import json
import tempfile
import unittest
from pathlib import Path

from local_bench.placement_baseline import validate_tp_gnn_partition
from local_bench.tp_gnn_contract import validate_tp_gnn_method


class TPGNNContractTest(unittest.TestCase):
    @staticmethod
    def report():
        return {"method": "tp-gnn-flat", "hierarchy": {"mode": "flat", "consumed_as_feature": False},
                "config": {"decoder": "global-2way", "feature_regime": "partition-last"}}

    def test_legacy_identity_is_retained_not_inferred_flat(self):
        legacy = {"method": "tp-gnn", "config": {"decoder": "global-2way"}}
        self.assertEqual(validate_tp_gnn_method(legacy, "tp-gnn", expected_method="tp-gnn"), "tp-gnn")
        with self.assertRaisesRegex(ValueError, "qualify a new profile"):
            validate_tp_gnn_method(legacy, expected_method="tp-gnn-flat")

    def test_flat_requires_native_evidence_not_only_a_renamed_method(self):
        for key in ("hierarchy", "config"):
            report = self.report()
            del report[key]
            with self.assertRaisesRegex(ValueError, "native flat-hierarchy evidence"):
                validate_tp_gnn_method(report)
        report = self.report()
        report["hierarchy"]["consumed_as_feature"] = True
        with self.assertRaises(ValueError):
            validate_tp_gnn_method(report)

    def test_flat_and_legacy_records_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, "identities disagree"):
            validate_tp_gnn_method(self.report(), "tp-gnn-flat", "tp-gnn")
        with self.assertRaisesRegex(ValueError, "qualify a new profile"):
            validate_tp_gnn_method(self.report(), expected_method="tp-gnn")

    def test_historical_label_cannot_claim_flat_native_mode(self):
        report = self.report()
        report["method"] = "tp-gnn"
        with self.assertRaisesRegex(ValueError, "cannot be relabeled"):
            validate_tp_gnn_method(report)

    def test_flat_placement_needs_an_explicit_new_profile_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "20_partition"
            stage.mkdir()
            for name, value in {
                "stage_manifest.json": {"status": "success", "method": "tp-gnn-flat"},
                "partition_manifest.json": {"method": "tp-gnn-flat"},
                "tp_gnn_report.json": self.report(),
            }.items():
                (stage / name).write_text(json.dumps(value))
            before = {path: path.read_bytes() for path in stage.iterdir()}
            with self.assertRaisesRegex(ValueError, "qualify a new profile"):
                validate_tp_gnn_partition(root)
            self.assertEqual(validate_tp_gnn_partition(root, expected_method="tp-gnn-flat"), "tp-gnn-flat")
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents)
            report = self.report()
            report["config"]["decoder"] = "ta3d-hierarchical"
            (stage / "tp_gnn_report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "global-2way"):
                validate_tp_gnn_partition(root, expected_method="tp-gnn-flat")


if __name__ == "__main__":
    unittest.main()
