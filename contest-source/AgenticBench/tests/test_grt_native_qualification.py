"""Offline guards for the native qualification controller; no EDA execution."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qualify_grt_optimization import apply_edit


class NativeQualificationTests(unittest.TestCase):
    def test_exact_meaningful_edit_and_reviewable_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "placer.py"
            before = "BACKEND_AREA_GROWTH_FACTOR = 1.25"
            after = "BACKEND_AREA_GROWTH_FACTOR = 1.30"
            source.write_text(before + "\n")
            edit = {"file": "placer.py", "before": before, "after": after}
            patch = apply_edit(root, edit)
            self.assertEqual(source.read_text(), after + "\n")
            self.assertIn("-" + before, patch)
            self.assertIn("+" + after, patch)
            with self.assertRaises(ValueError): apply_edit(root, edit)
            with self.assertRaises(ValueError): apply_edit(root, dict(edit, before=after))
            source.write_text(before + "\n" + before + "\n")
            with self.assertRaises(ValueError): apply_edit(root, edit)


if __name__ == "__main__":
    unittest.main()
