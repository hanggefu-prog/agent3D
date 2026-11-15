"""Explicit, reusable reference selection for workspace searches.

ADOPT performs independent read-only qualification; it never starts a physical
flow. FRESH dispatches exactly once and publishes only independently verified
native evidence. EXTEND reuses a verified native prefix, never a historical
source composition inferred only from matching configuration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from global_bench.verifier import load_trusted_verifier
from local_bench.benchmark_input import validate_benchmark_input
from local_bench import reference_baseline
from local_bench.reference_store import (
    adopt_reference, file_sha, find_reference, load_reference, policy,
    publish_reference,
)


ENDPOINT_NAMES = {"timing": "timing", "thermal": "cts_thermal", "routed_thermal": "routed_thermal"}
FRAMEWORK = Path(__file__).resolve().parents[1]


def evaluator_digest():
    from local_bench.reference_catalog import evaluator_digest as trusted_digest
    return trusted_digest()


def _qualify(producer):
    from local_bench.reference_catalog import qualify_producer
    return qualify_producer(producer)


def _source_identity(runtime):
    """The actual initial checkout must match the declared producer baseline."""
    source = Path(runtime.source).resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                     "--untracked-files=all", "--ignore-submodules=none"], text=True,
                                    env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    if revision != runtime.source_commit or dirty.strip():
        raise ValueError("Reference selection requires the exact clean declared initial source checkout")


class WorkspaceReference:
    def __new__(cls, runtime, case, objective, bundle, task):
        if getattr(runtime, "reference_baseline", None) == "reference_20261004":
            from scripts.workspace_latest_reference import LatestWorkspaceReference
            return LatestWorkspaceReference(runtime, case, objective, bundle, task)
        return super().__new__(cls)

    def __init__(self, runtime, case, objective, bundle, task):
        self.runtime, self.case, self.objective, self.task = runtime, case, objective, task
        self.policy = policy(getattr(runtime, "reference_policy", None) or "ADOPT")
        load_trusted_verifier(task.verifier.verifier_id)
        if self.policy == "EXTEND" and getattr(runtime, "reference_record", None) is None:
            raise ValueError("EXTEND requires an explicit qualified parent record. No flow was dispatched.")
        self.endpoint = ENDPOINT_NAMES[objective]
        self.store = Path(getattr(runtime, "reference_store", None) or
                          (Path(runtime.batch) / "reference-store")).expanduser().resolve()
        self.record_path = None
        explicit = getattr(runtime, "reference_record", None)
        if explicit is not None:
            self.record_path = Path(explicit).expanduser().absolute()
            # Do not execute old algorithms simply because their configuration
            # agrees. Their per-stage producer revisions need a reviewed source
            # composition adapter, which this workspace path does not provide.
            record = json.loads(self.record_path.read_text())
            if record.get("producer", {}).get("kind") == "frozen-20260923-v1":
                raise ValueError("Archived reference results are qualified separately, but workspace candidate "
                                 "source composition is not implemented for their stage-specific algorithm/backend "
                                 "revisions. Do not substitute current code; select a compatible native reference "
                                 "or explicitly qualify FRESH.")
        _source_identity(runtime)
        from scripts.workspace_framework_identity import revalidate_runtime_framework
        self.audit_framework_identity = revalidate_runtime_framework(runtime)
        if self.audit_framework_identity is not None and self.policy != "ADOPT":
            raise ValueError("An audit-only framework split cannot dispatch FRESH/EXTEND")
        validated = validate_benchmark_input(Path(bundle))
        if self.audit_framework_identity is None:
            reference_baseline.validate_reference_bundle(Path(bundle), case)
        else:
            reference_baseline.validate_reference_bundle(Path(bundle), case,
                audit_framework_identity=self.audit_framework_identity)
        reference_baseline.validate_physical_prefix(Path(validated["run_dir"]), case)
        # Legacy qualification imports its matching physical-flow helpers.
        # Select the family before importing them: the latest artifact-only
        # consumer must work from the independent control plane alone.
        from local_bench.reference_evaluated import expected_frozen_inputs, expected_source_contract
        from local_bench.reference_catalog import task_digest
        self.contract = {
            "case": case, "endpoint": self.endpoint, "task_digest": task_digest(task.root),
            "evaluator_digest": evaluator_digest(),
            "frozen_inputs": expected_frozen_inputs(case, file_sha(Path(bundle) / "benchmark_input.json")),
            "physical": expected_source_contract(case, self.endpoint,
                profile=task.objective.evaluator_profile, source_revision=runtime.source_commit,
                image=runtime.image),
        }
        self.record = None
        self.record_sha256 = None
        self.parent = None
        if self.policy == "ADOPT":
            if self.record_path is None:
                self.record_path = find_reference(self.store, contract=self.contract)
            self.record_sha256 = file_sha(self.record_path)
            self.record = load_reference(self.record_path, contract=self.contract, validate=_qualify)
            if file_sha(self.record_path) != self.record_sha256:
                raise ValueError("Selected reference record changed during preflight qualification")
            if self.record["producer"].get("kind") != "native-evaluation-v1":
                raise ValueError("Workspace ADOPT requires a compatible native source composition")
        elif self.policy == "EXTEND":
            from scripts.workspace_reference_extension import qualify_parent
            self.parent = qualify_parent(self.record_path, self.contract)
            self.parent_path = self.record_path
            self.record_path = None
        elif self.record_path is not None:
            raise ValueError("FRESH must not also select an existing reference record")

    def to_dict(self):
        return {"policy": self.policy, "store": str(self.store), "contract": self.contract,
                "record": str(self.record_path) if self.record_path else None,
                "record_sha256": self.record_sha256,
                "audit_framework_identity": self.audit_framework_identity,
                "parent": {key: self.parent[key] for key in ("record", "record_sha256", "start_stage", "stages")}
                    if self.parent else None,
                "fallback": "none", "archive_source_composition": "unsupported-fail-closed"}

    @property
    def evaluator_invocations(self):
        return 0 if self.policy == "ADOPT" else 1

    def obtain(self, evaluator, output):
        output = Path(output)
        from scripts.workspace_framework_identity import revalidate_runtime_framework
        revalidate_runtime_framework(self.runtime)
        if self.policy == "ADOPT":
            if file_sha(self.record_path) != self.record_sha256:
                raise ValueError("Selected reference record changed after preflight")
            receipt = adopt_reference(self.record_path, contract=self.contract,
                                      validate=_qualify, output=output)
            if (receipt["record_sha256"] != self.record_sha256
                    or file_sha(self.record_path) != self.record_sha256):
                raise ValueError("Selected reference record changed during adoption qualification")
            from global_bench.verifiers.open3dbench_codesign_verifier import reference_self_metrics
            metrics = reference_self_metrics(self.endpoint, receipt["qualification"]["baseline"]["raw_metrics"],
                                             self.task.objective.evaluator_profile)
            result = {"success": True, "metrics": metrics,
                      "run": receipt["qualification"]["baseline"]["source_run"],
                      "reference_origin": "adopted", "reference_id": receipt["reference_id"],
                      "reference_record": str(self.record_path), "reference_record_sha256": self.record_sha256,
                      "reference_evaluator_invocations": 0, "new_reference_native_attempts": 0,
                      "verification": str(output / "reference_adoption.json"),
                      "historical_native_runtime": receipt["historical_native_runtime"],
                      "runtime_seconds": receipt["verification_wall_seconds"],
                      "runtime_seconds_scope": "read-only adoption verification, not native evaluation",
                      "error": None}
        else:
            if self.policy == "EXTEND":
                result = evaluator.evaluate(self.runtime.source, output, None, parent_reference=self.parent)
            else:
                result = evaluator.evaluate(self.runtime.source, output, None)
            result["reference_evaluator_invocations"] = 1
            if result.get("success") is not True:
                return result
            producer = {"kind": "native-evaluation-v1", "run_root": result["run"],
                        "evaluation_root": str(output), "case": self.case,
                        "endpoint": self.endpoint, "profile": self.task.objective.evaluator_profile}
            if self.parent:
                producer.update(parent_record=str(self.parent_path),
                                parent_reference_id=self.parent["record_value"]["reference_id"])
            self.record_path = publish_reference(self.store, contract=self.contract, origin=self.policy,
                                                  producer=producer, validate=_qualify)
            self.record = load_reference(self.record_path, contract=self.contract, validate=_qualify)
            self.record_sha256 = file_sha(self.record_path)
            result.update(reference_origin=self.policy.lower(), reference_record=str(self.record_path),
                          reference_id=self.record["reference_id"],
                          reference_record_sha256=file_sha(self.record_path))
        # Bind the selected authority for every subsequent candidate verifier.
        self.runtime.reference_record = self.record_path
        self.runtime.reference_contract = self.contract
        self.runtime.reference_record_sha256 = self.record_sha256
        self.runtime.reference_evaluation_root = Path((self.record or {})["producer"]["evaluation_root"])
        (output / "feedback.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
