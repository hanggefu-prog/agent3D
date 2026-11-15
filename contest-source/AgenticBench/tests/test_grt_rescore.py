import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.contract import validate, matrix
from grt_bench.rescore import validate_transition
from test_grt_matrix import task_fixture


class RescoreTests(unittest.TestCase):
    def test_host_may_select_unchanged_case_subset(self):
        old = task_fixture()
        old["cases"].append(dict(old["cases"][0], id="omitted"))
        new = copy.deepcopy(old)
        new["cases"] = new["cases"][:1]
        new["id"] = "selected-cases"
        new["evaluator_sha256"] = "b" * 64
        before, after = matrix(old, "c" * 64)[0], matrix(new, "c" * 64)[0]
        validate_transition(new, old, after, before)
        omitted = next(p for p in matrix(old, "c" * 64) if p["case"] == "omitted")
        with self.assertRaisesRegex(ValueError, "not in the selected task"):
            validate_transition(new, old, dict(omitted, task_digest=validate(new)), omitted)

    def test_subset_cannot_modify_or_add_cells(self):
        old = task_fixture()
        old["cases"].append(dict(old["cases"][0], id="omitted"))
        before = matrix(old, "c" * 64)[0]
        changes = (
            lambda t: t["cases"][0].update(weight=2),
            lambda t: t["cases"][0].update(split="test"),
            lambda t: t["cases"][0].update(input_identity="e" * 64),
            lambda t: t["cases"][0].update(config_sha256="d" * 64),
            lambda t: t["cases"].append(dict(t["cases"][0], id="new-case")),
            lambda t: t["seeds"][0].update(partition=123),
            lambda t: t["scoring"].update(power_max_ratio=1.05),
            lambda t: t["scoring"].update(timing_weights={"wns_deficit_ns": .5, "tns_deficit_ns": .5}),
        )
        for change in changes:
            with self.subTest(change=change):
                new = copy.deepcopy(old)
                new["cases"] = new["cases"][:1]
                new["evaluator_sha256"] = "b" * 64
                change(new)
                after = dict(before, task_digest=validate(new))
                with self.assertRaisesRegex(ValueError, "physical task contract|Split migration requires original baseline evidence"):
                    validate_transition(new, old, after, before)

    def test_only_evaluator_may_change(self):
        old = task_fixture()
        new = copy.deepcopy(old)
        new["evaluator_sha256"] = "b" * 64
        before = matrix(old, "c" * 64)[0]
        after = matrix(new, "c" * 64)[0]
        validate_transition(new, old, after, before)
        for change in (lambda t: t["seeds"][0].update(partition=123),
                       lambda t: t["cases"][0].update(config_sha256="d"*64),
                       lambda t: t["cases"][0].update(input_identity="e"*64),
                       lambda t: t["cases"][0].update(weight=2)):
            changed = copy.deepcopy(new)
            change(changed)
            with self.assertRaises(ValueError): validate_transition(changed, old, after, before)
        for key, value in (("candidate", "d"*64), ("case", "wrong"), ("seed", old["seeds"][1]),
                           ("split", "test"), ("endpoint", "cts"), ("task_digest", "f"*64)):
            with self.assertRaises(ValueError): validate_transition(new, old, dict(after, **{key:value}), before)
