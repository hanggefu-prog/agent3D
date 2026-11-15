"""Controller-side policy for same-runtime native local references."""
from __future__ import annotations

import json
from pathlib import Path

from agentic3d.knowledge import write_json
from local_bench.model import EvaluationResult
from local_bench.reference_catalog import adopt, qualify_producer, register_producer
from local_bench.reference_local_placement import expected_local_contract
from local_bench.reference_placement_evidence import DYNAMIC_TASK
from local_bench.reference_store import file_sha, find_reference, load_reference, policy, protect_destination


class LocalPlacementReference:
    producer_kind = "native-local-placement-v1"
    family = "placement"

    @staticmethod
    def expected_contract(*args, **kwargs):
        return expected_local_contract(*args, **kwargs)

    def __init__(self, runtime, spec, case, bundle, output, *, selected_policy=None,
                 store=None, record=None, reference_id=None):
        self.runtime, self.spec, self.case = runtime, spec, case
        self.bundle = Path(bundle).resolve(strict=True)
        self.policy = policy(selected_policy or "ADOPT")
        if self.policy == "EXTEND":
            raise ValueError("Local " + self.family + " has no missing suffix to EXTEND; select a qualified ADOPT "
                             "record or explicitly request FRESH. No evaluator was dispatched.")
        self.store = Path(store or (runtime.batch / "reference-store")).expanduser().absolute()
        self.record_path = Path(record).expanduser().absolute() if record else None
        self.record = None
        self.record_sha256 = None
        self.source_run = None
        if self.policy == "FRESH" and (record is not None or reference_id is not None):
            raise ValueError("FRESH cannot also select an existing reference")
        if self.record_path is not None:
            raw = json.loads(self.record_path.read_text())
            kind = raw.get("producer", {}).get("kind")
            if kind != self.producer_kind:
                raise ValueError("Historical archive/source-hook compatibility bridge is not implemented for "
                                 "local " + self.family + "; equal configuration is not algorithm equivalence. "
                                 "Select a same-runtime native local reference or explicitly request FRESH.")
            if reference_id is not None and raw.get("reference_id") != reference_id:
                raise ValueError("Explicit local reference record and ID disagree")
        self.contract = self.expected_contract(runtime.source, spec.task_id, case, self.bundle,
                                               source_revision=runtime.source_commit, image=runtime.image)
        if self.policy == "ADOPT":
            if self.record_path is None:
                self.record_path = find_reference(self.store, contract=self.contract, reference_id=reference_id)
            self.record_sha256 = file_sha(self.record_path)
            self.record = load_reference(self.record_path, contract=self.contract, validate=qualify_producer)
            if self.record["producer"].get("kind") != self.producer_kind:
                raise ValueError("Local ADOPT requires a qualified same-runtime source/hook producer")
            protect_destination(output, self.record["producer"], payload=self.record["qualification"],
                                record_path=self.record_path)
            self.source_run = Path(self.record["qualification"]["baseline"]["source_run"]).resolve(strict=True)
            self.verify_selected()
        else:
            protect_destination(self.store, {"source_root": str(runtime.source),
                "benchmark_input": str(self.bundle), "evaluation_root": str(Path(output) / "reference/result")})
            # The future candidate write tree cannot contain its authority.
            if self.store == output or output in self.store.parents or self.store in output.parents:
                raise ValueError("Local reference store and smoke output must be disjoint")

    def to_dict(self):
        return {"policy": self.policy, "store": str(self.store), "contract": self.contract,
                "record": str(self.record_path) if self.record_path else None,
                "record_sha256": self.record_sha256,
                "reference_id": self.record["reference_id"] if self.record else None,
                "fallback": "none", "archive_source_hook_bridge": "unsupported-fail-closed"}

    def prepare_fresh(self, evaluation, candidate):
        if self.policy != "FRESH":
            raise ValueError("Only explicit FRESH may prepare a native reference")
        receipt = Path(evaluation).parent / "reference-purpose.json"
        write_json(receipt, {"schema_version": 1, "reference_purpose": "FRESH", "reference_run": None,
            "source_root": str(self.runtime.source), "evaluation_root": str(evaluation),
            "image": self.runtime.image, "candidate_sha256": file_sha(candidate / "candidate.json"),
            "contract": self.contract})
        return receipt

    def publish(self, evaluation):
        producer = {"kind": self.producer_kind, "task_id": self.spec.task_id,
                    "case": self.case, "source_root": str(self.runtime.source),
                    "evaluation_root": str(evaluation), "benchmark_input": str(self.bundle)}
        self.record_path = register_producer(self.store, producer, origin="FRESH", contract=self.contract)
        self.record_sha256 = file_sha(self.record_path)
        self.record = load_reference(self.record_path, contract=self.contract, validate=qualify_producer)
        self.source_run = Path(self.record["qualification"]["baseline"]["source_run"]).resolve(strict=True)
        self.verify_selected()
        return self.record

    def verify_selected(self):
        """Pin exact record bytes and requalify native evidence before each use."""
        if self.record_path is None or file_sha(self.record_path) != self.record_sha256:
            raise ValueError("Selected local reference record bytes changed after preflight")
        verified = load_reference(self.record_path, contract=self.contract, validate=qualify_producer)
        if (file_sha(self.record_path) != self.record_sha256 or
                Path(verified["qualification"]["baseline"]["source_run"]).resolve(strict=True) != self.source_run):
            raise ValueError("Selected local reference authority changed during requalification")
        return verified

    def obtain_adopted(self, output):
        self.verify_selected()
        receipt = adopt(self.record_path, contract=self.contract, output=output)
        self.verify_selected()
        if receipt["record_sha256"] != self.record_sha256:
            raise ValueError("Adoption receipt differs from the pinned local reference")
        result = self.qualified_result(origin="adopted")
        result.artifacts.update(new_reference_native_attempts=0,
            verification_wall_seconds=receipt["verification_wall_seconds"])
        return result

    def qualified_result(self, *, origin):
        verified = self.verify_selected()
        run = self.source_run
        from local_bench.metrics import placement_metrics, placement_nesterov_metrics
        scorer = placement_nesterov_metrics if self.spec.task_id == DYNAMIC_TASK else placement_metrics
        metrics = scorer(run, run)
        return EvaluationResult({**metrics, "runs_successfully": 1.0}, {
            "run_dir": str(run), "reference_origin": origin, "reference_id": verified["reference_id"],
            "reference_record": str(self.record_path), "reference_record_sha256": self.record_sha256,
            "historical_native_runtime": verified["qualification"]["baseline"]["native_runtime_historical"]})


class LocalPartitionReference(LocalPlacementReference):
    """FRESH initial loss qualification and fail-closed ADOPT for TP-GNN-flat."""
    producer_kind = "native-local-partition-v1"
    family = "partition"

    @staticmethod
    def expected_contract(*args, **kwargs):
        from local_bench.reference_local_partition import expected_local_contract
        return expected_local_contract(*args, **kwargs)

    def qualified_result(self, *, origin):
        from local_bench.reference_local_partition import independent_partition_metrics
        from local_bench.reference_latest import baseline_for_config, BASELINE_ID
        verified = self.verify_selected()
        # Select by the already verified contract, never by whichever directory
        # happens to exist. Latest flow owns STA inputs inside partition.
        selected = baseline_for_config(self.contract["physical"]["configuration"])
        timing = "20_partition/timing_input" if selected.BASELINE_ID == BASELINE_ID else "15_timing_extract"
        metrics = independent_partition_metrics(self.source_run, self.runtime.source, timing_directory=timing)
        return EvaluationResult({**metrics, "runs_successfully": 1.0}, {
            "run_dir": str(self.source_run), "reference_origin": origin,
            "reference_id": verified["reference_id"], "reference_record": str(self.record_path),
            "reference_record_sha256": self.record_sha256,
            "historical_native_runtime": verified["qualification"]["baseline"]["native_runtime_historical"]})
