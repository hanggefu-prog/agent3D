import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.latch_audit import audit, retain_data_checks


class LatchAuditTests(unittest.TestCase):
    def test_adapter_requires_exact_known_model(self):
        from open3dbench.backend.post_route import ASSEMBLY_STA_TCL
        adapted = retain_data_checks(ASSEMBLY_STA_TCL)
        self.assertNotIn("set_disable_timing $unclocked_latch_pins", adapted)
        self.assertIn("model=bounded_data_checks_retained", adapted)
        with self.assertRaises(ValueError): retain_data_checks(adapted)

    def evaluate(self, report, proof):
        with tempfile.TemporaryDirectory() as d:
            report_path, proof_path = Path(d) / "check.rpt", Path(d) / "proof.tsv"
            report_path.write_text(report)
            proof_path.write_text(proof)
            return audit(report_path, proof_path)

    def test_empty_and_exact_bounded_latches(self):
        self.assertEqual(self.evaluate("", "")["functional_latches"], 0)
        report = "Warning: There are 1 unclocked register/latch pins.\n  u_bottom/l/G\nWarning: There are 1 unconstrained endpoints.\n  u_bottom/l/D\n"
        proof = "u_bottom/l/G\tu_bottom/l/D\t0.1\t-0.3\n"
        self.assertEqual(self.evaluate(report, proof)["bounded_data_pins"], 1)
        for changed in (report.replace("l/G", "ff/CK"), report.replace("l/D", "other/D"),
                        report.replace("are 1 unconstrained", "are 2 unconstrained"),
                        report + "Warning: There are 1 timing loops.\n"):
            with self.assertRaises(ValueError): self.evaluate(changed, proof)
        for changed in ("", proof.replace("0.1", "nan"), proof.replace("0.1", "1e30"),
                        proof.replace("l/D", "other/D"), proof + proof):
            with self.assertRaises(ValueError): self.evaluate(report, changed)
