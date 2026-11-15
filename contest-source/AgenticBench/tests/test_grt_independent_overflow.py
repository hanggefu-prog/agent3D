import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench import independent_overflow as audit
from grt_bench.common import sha, write
from open3dbench.backend import three_d_route as routing

FRAMEWORK = Path(__file__).resolve().parents[1]
TRACKS = ["TRACKS Y 5 DO 30 STEP 10 LAYER metal1 ;", "TRACKS X 5 DO 40 STEP 10 LAYER metal2 ;",
          "TRACKS Y 5 DO 30 STEP 10 LAYER metal3 ;", "TRACKS X 5 DO 40 STEP 10 LAYER metal4 ;"]
DEF = ("VERSION 5.8 ;\nUNITS DISTANCE MICRONS 1000 ;\nDIEAREA ( 0 0 ) ( 600 450 ) ;\n" + "\n".join(TRACKS)
       + "\nGCELLGRID X 0 DO 4 STEP 150 ;\nGCELLGRID Y 0 DO 3 STEP 150 ;\nCOMPONENTS 0 ;\nEND COMPONENTS\nEND DESIGN\n")


def make_die(directory, nets, reported, *, die=None, seal=True):
    """One synthetic post-GRT die: 4 x 3 GCells of 150 DBU, 15 tracks per GCell edge."""
    directory.mkdir(parents=True, exist_ok=True)
    die = die or directory.name
    (directory / "global_route.odb.def").write_text(DEF)
    blocks = []
    for index in range(nets):  # each net crosses all three metal3 edges of GCell row 0
        blocks.append("net{}\n(\n0 0 600 150 metal3\n0 0 150 150 metal2\n)\n".format(index))
    (directory / "grt.guide").write_text("".join(blocks))
    env = {"ROUTE_DIE": die, "ROUTE_MIN_LAYER": "metal2", "ROUTE_MAX_LAYER": "metal4",
           "ROUTE_LAYER_ADJUSTMENT": "0.0", "ROUTE_SET_RC": str(directory / "no_platform/setRC.tcl")}
    (directory / "grt.tcl").write_text(routing._environment_script(env, routing.GRT_TCL))
    (directory / "grt.log").write_text(
        "[INFO GRT-0018] Total wirelength: 10.0 um\nGlobal routing runtime = 00:00:01\nFinal congestion report:\n"
        "Layer         Resource        Demand        Usage (%)    Max H / Max V / Total Overflow\n"
        "---------------------------------------------------------------------------------------\n"
        "metal3              135            {d}           50.00%             {r} /  0 /  {r}\n"
        "---------------------------------------------------------------------------------------\n"
        "Total               135            {d}           50.00%             {r} /  0 /  {r}\n"
        "OPEN3D_GRT_DONE die={die}\n".format(d=3 * nets, r=reported, die=die))
    if seal:
        artifacts = {name: {"size_bytes": (directory / name).stat().st_size, "sha256": sha(directory / name)}
                     for name in audit.AUDITED_ARTIFACTS}
        write(directory / "grt_manifest.json", {"schema_version": 1, "stage": "global_route", "status": "success",
                                                "artifacts": artifacts, "metrics": {"grt_total_overflow": reported}},
              replace=True)
    return directory


def make_grt(root, bottom=(20, 15), top=(15, 0), **kwargs):
    """``(nets, reported)`` per die; 20 nets give an estimated overflow of 3 x 5 = 15."""
    make_die(root / "bottom", *bottom, **kwargs)
    make_die(root / "top", *top, **kwargs)
    write(root / "global_route_checkpoint.json", {"stage": "global_route"}, replace=True)
    return root


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_consistent_receipt_records_both_values_and_identity(self):
        receipt = audit.audit_grt(make_grt(self.root / "grt"))
        self.assertTrue(receipt["consistent"], receipt["errors"])
        self.assertEqual((receipt["reported_overflow"], receipt["estimated_overflow"]), (15, 15))
        self.assertEqual(receipt["dies"]["bottom"]["estimated_overflow"], 15)
        self.assertEqual(receipt["dies"]["top"]["estimated_overflow"], 0)
        self.assertEqual(receipt["dies"]["bottom"]["reported_layers"]["metal3"]["overflow"], 15)
        self.assertTrue(receipt["dies"]["bottom"]["manifest_verified"])
        identity = receipt["estimator"]
        self.assertEqual(identity["version"], 1)
        self.assertEqual(identity["sha256"], sha(FRAMEWORK / "grt_bench/guide_congestion.py"))
        self.assertEqual([c["scope"] for c in receipt["checks"]], ["bottom", "top", "total"])
        json.dumps(receipt, allow_nan=False)

    def test_under_reported_overflow_is_inconsistent(self):
        receipt = audit.audit_grt(make_grt(self.root / "grt", bottom=(20, 0)))
        self.assertFalse(receipt["consistent"])
        failed = [(c["scope"], c["check"]) for c in receipt["checks"] if not c["passed"]]
        self.assertEqual(failed, [("bottom", "bound"), ("total", "bound")])

    def test_tolerance_boundary(self):
        grt = make_grt(self.root / "grt", bottom=(20, 10))  # E = 15 <= 10 * 1.05 + 5
        self.assertTrue(audit.audit_grt(grt)["consistent"])
        self.assertFalse(audit.audit_grt(grt, rule={"abs_tolerance": 4})["consistent"])
        self.assertTrue(audit.audit_grt(grt, rule={"abs_tolerance": 0, "rel_tolerance": 0.5})["consistent"])

    def test_positive_estimate_rule_is_optional(self):
        grt = make_grt(self.root / "grt", top=(0, 3))
        self.assertTrue(audit.audit_grt(grt)["consistent"])
        receipt = audit.audit_grt(grt, rule={"require_positive_estimate": True})
        self.assertEqual([(c["scope"], c["check"]) for c in receipt["checks"] if not c["passed"]],
                         [("top", "positive")])

    def test_reference_rule_rejects_contradicted_improvement(self):
        grt = make_grt(self.root / "grt", bottom=(20, 12))
        self.assertTrue(audit.audit_grt(grt, reference={"reported": 12, "estimate": 0})["consistent"])
        self.assertFalse(audit.audit_grt(grt, reference={"reported": 20, "estimate": 5})["consistent"])
        self.assertTrue(audit.audit_grt(grt, reference={"reported": 20, "estimate": 14})["consistent"])
        reference = self.root / "reference.json"
        write(reference, audit.audit_grt(make_grt(self.root / "ref", bottom=(30, 45))))
        receipt = audit.audit_grt(grt, reference=str(reference))
        self.assertEqual(receipt["reference"], {"reported": 45, "estimate": 45})
        self.assertTrue(receipt["consistent"])
        inconsistent = self.root / "inconsistent.json"
        write(inconsistent, audit.audit_grt(make_grt(self.root / "ref2", bottom=(30, 0))))
        receipt = audit.audit_grt(grt, reference=str(inconsistent))
        self.assertFalse(receipt["consistent"])
        self.assertIn("not itself consistent", receipt["errors"][0]["error"])

    def test_fail_closed_on_missing_symlinked_or_tampered_evidence(self):
        cases = {
            "missing guide": lambda grt: (grt / "bottom/grt.guide").unlink(),
            "missing die": lambda grt: [p.unlink() for p in (grt / "top").iterdir()] and (grt / "top").rmdir(),
            "missing manifest": lambda grt: (grt / "top/grt_manifest.json").unlink(),
            "edited guide": lambda grt: (grt / "bottom/grt.guide").write_text("net\n(\n)\n"),
            "edited log": lambda grt: (grt / "top/grt.log").write_text("Total 1 1 1.00% 0 / 0 / 0\n"),
            "no total": lambda grt: (grt / "top/grt.log").write_text("nothing\n"),
        }
        for label, mutate in cases.items():
            with self.subTest(label):
                grt = make_grt(self.root / label.replace(" ", "_"))
                mutate(grt)
                receipt = audit.audit_grt(grt)
                self.assertFalse(receipt["consistent"])
                self.assertTrue(receipt["errors"])
                self.assertIsNone(receipt["reported_overflow"])

    def test_symlinked_artifact_and_metric_mismatch_fail(self):
        grt = make_grt(self.root / "grt")
        guide = grt / "bottom/grt.guide"
        target = self.root / "guide.copy"
        target.write_text(guide.read_text())
        guide.unlink()
        os.symlink(str(target), str(guide))
        self.assertIn("symlinked", audit.audit_grt(grt)["errors"][0]["error"])
        other = make_die(self.root / "solo" / "bottom", 20, 15)
        manifest = json.loads((other / "grt_manifest.json").read_text())
        manifest["metrics"]["grt_total_overflow"] = 14
        write(other / "grt_manifest.json", manifest, replace=True)
        self.assertIn("Sealed GRT metrics", audit.audit_grt(other)["errors"][0]["error"])

    def test_unsealed_directory_requires_explicit_opt_out(self):
        grt = make_grt(self.root / "grt", seal=False)
        self.assertFalse(audit.audit_grt(grt)["consistent"])
        receipt = audit.audit_grt(grt, require_manifest=False)
        self.assertTrue(receipt["consistent"])
        self.assertFalse(receipt["dies"]["bottom"]["manifest_verified"])

    def test_single_die_directory(self):
        die = make_die(self.root / "bottom", 20, 15)
        receipt = audit.audit_grt(die)
        self.assertEqual(list(receipt["dies"]), ["bottom"])
        self.assertTrue(receipt["consistent"])

    def test_invalid_rule_and_reference_fail_closed(self):
        grt = make_grt(self.root / "grt")
        for kwargs in ({"rule": {"unknown": 1}}, {"rule": {"abs_tolerance": -1}},
                       {"rule": {"rel_tolerance": float("nan")}}, {"reference": {"reported": 1}}):
            with self.subTest(kwargs=kwargs):
                receipt = audit.audit_grt(grt, **kwargs)
                self.assertFalse(receipt["consistent"])
                self.assertEqual(receipt["errors"][0]["scope"], "setup")

    def test_discover_run_roots(self):
        make_grt(self.root / "a/grt")
        make_grt(self.root / "deep/b/out/grt")
        self.assertEqual(audit.discover(self.root / "a"), [self.root / "a/grt"])
        self.assertEqual(audit.discover(self.root / "a/grt"), [self.root / "a/grt"])
        self.assertEqual(audit.discover(self.root), [self.root / "a/grt", self.root / "deep/b/out/grt"])


class ContractTests(unittest.TestCase):
    def test_validate_contract_defaults_and_strictness(self):
        contract = audit.validate_contract({"version": 1})
        self.assertTrue(contract["enforce"])
        self.assertEqual(contract["rule"], audit.DEFAULT_RULE)
        for block in ({}, {"version": 2}, {"version": 1, "extra": 1}, {"version": 1, "enforce": "yes"},
                      {"version": 1, "estimator_sha256": "abc"}, {"version": 1, "track_model": "magic"},
                      {"version": 1, "rule": {"x": 1}},
                      {"version": 1, "reference_estimates": {"c": {"1": {"reported": -1, "estimate": 0}}}}):
            with self.subTest(block=block), self.assertRaises(ValueError):
                audit.validate_contract(block)

    def make_output(self, root, **grt):
        grt_dir = make_grt(root / "run/grt", **grt)
        write(root / "out/analysis/manifest.json",
              {"source_checkpoint": str(grt_dir / "global_route_checkpoint.json"), "platform": str(root / "none")})
        return root / "out"

    def test_verify_hook_enforces_and_records(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan = {"case": "c1", "seed": {"id": 7}}
            good = self.make_output(root / "good")
            receipt = audit.verify_hook({"independent_overflow": {"version": 1}}, plan, good)
            self.assertTrue(receipt["consistent"] and receipt["enforced"])
            bad = self.make_output(root / "bad", bottom=(20, 0))
            with self.assertRaisesRegex(ValueError, "Independent overflow audit failed: bottom:bound"):
                audit.verify_hook({"independent_overflow": {"version": 1}}, plan, bad)
            advisory = audit.verify_hook({"independent_overflow": {"version": 1, "enforce": False}}, plan, bad)
            self.assertFalse(advisory["consistent"])
            task = {"independent_overflow": {"version": 1, "reference_estimates": {
                "c1": {"7": {"reported": 20, "estimate": 5}}}}}
            with self.assertRaisesRegex(ValueError, "total:reference"):
                audit.verify_hook(task, plan, good)
            with self.assertRaisesRegex(ValueError, "estimator differs"):
                audit.verify_hook({"independent_overflow": {"version": 1, "estimator_sha256": "0" * 64}}, plan, good)

    def test_verifier_hook_is_opt_in(self):
        source = (FRAMEWORK / "grt_bench/verify.py").read_text()
        self.assertIn('if "independent_overflow" in task:', source)


class CliTests(unittest.TestCase):
    def test_cli_reports_and_fails_on_mismatch(self):
        script = FRAMEWORK / "scripts/audit_grt_overflow.py"
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_grt(root / "good/grt")
            make_grt(root / "bad/grt", bottom=(20, 0))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(FRAMEWORK.parent), str(FRAMEWORK.parent / "Place-3D")]))
            ok = subprocess.run([sys.executable, str(script), str(root / "good"), "--output", str(root / "ok.json")],
                                capture_output=True, text=True, env=env)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("1/1 consistent", ok.stdout)
            report = json.loads((root / "ok.json").read_text())
            self.assertEqual(report["summary"]["consistent"], 1)
            bad = subprocess.run([sys.executable, str(script), str(root)], capture_output=True, text=True, env=env)
            self.assertEqual(bad.returncode, 1, bad.stderr)
            self.assertIn("MISMATCH", bad.stdout)
            sys.path.insert(0, str(script.parent))
            try:
                import audit_grt_overflow as cli
            finally:
                sys.path.remove(str(script.parent))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli.main([str(root), "--no-fail", "--json"]), 0)
            self.assertEqual(json.loads(out.getvalue())["summary"]["targets"], 2)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([str(root / "missing")]), 1)


if __name__ == "__main__":
    unittest.main()
