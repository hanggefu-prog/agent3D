#!/usr/bin/env python3
"""Authorized 2-, 5- or 50-candidate local runs, with host login and Docker EDA.

This is an ask/tell integration smoke, not a new optimization benchmark. The
task owns its candidate policy, fixed input, algorithms and objective. Versioned
placement tasks default to qualified ADOPT; explicit FRESH is separate from the
candidate budget. Legacy tasks retain their initial evaluation. No implicit
retry, duplicate-result cache, API credential mount, or host EDA is allowed.
Five remains the default smoke budget; 2 or 50 requires an explicit selection and
does not modify task definitions or any previously created experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic3d.knowledge import write_json
from global_bench.workspace_codex import MODEL, build_workspace_command, codex_overrides
from local_bench.benchmark_input import validate_benchmark_input
from local_bench.llm_client import CodexExecLLM
from local_bench.model import EvaluationResult, ROUTING_SOURCE_TASK_KINDS, load_candidate, load_task, resolve_task_path, require_executable_task
from local_bench.program_bridge import initial_program_text, materialize_program
from local_bench.prompting import ContextualLLM, build_problem_description, task_system_message
from local_bench.search_api import (AskTellRunner, CallbackSearchBackend, CandidateProposal,
                                    ProgramEvaluationHarness, SearchProblem, TaskEvaluationOptions)
from local_bench.verifier import verify_candidate
from scripts.codesign_runtime import Runtime
from scripts.experiment_retention import finish_retention, validate_mode


AUTHORIZED_CANDIDATE_BUDGETS = (2, 5, 50)


class EvaluationInfrastructureError(RuntimeError):
    """A transport/environment failure must stop, not consume further model calls."""


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _successful(result):
    metrics = result.metrics
    score = metrics.get("combined_score")
    return (metrics.get("runs_successfully") == 1 and not isinstance(score, bool)
            and isinstance(score, (int, float)) and math.isfinite(score))


def _qualified_reference(spec, result):
    # Native exit success alone is insufficient when the policy hook was never
    # reached. Candidate scoring remains unchanged; this gates normalization.
    return (_successful(result) and (spec.kind not in ROUTING_SOURCE_TASK_KINDS
            or result.metrics.get("grt_complete") == 1))


def blackbox_command(binary, control, final):
    # Reuse the qualified process-local configuration and remove its sole MCP
    # terminal. Local tasks need text output only, not workspace tools.
    overrides = [value for value in codex_overrides(
        container="unused", control=control, terminal_script=control / "unused.py",
        python=sys.executable) if not value.startswith("mcp_servers.")
        and not value.startswith("features.code_mode_host=")]
    overrides += ["features.code_mode_host=false", "tools.view_image=false"]
    return build_workspace_command(binary=binary, control=control, final=final,
                                   overrides=overrides)


class TextOnlyCodex:
    """One independent, auditable request per candidate; login stays on the host."""

    def __init__(self, root, binary, timeout_seconds):
        self.root, self.binary, self.timeout_seconds = Path(root), binary, timeout_seconds
        self.root.mkdir(parents=True, exist_ok=False)
        self.control = self.root / "empty-control"
        self.control.mkdir()
        self.calls = 0
        self.completed_calls = 0

    def draw_sample(self, messages):
        self.calls += 1
        output = self.root / "sample-{:02d}".format(self.calls)
        output.mkdir(exist_ok=False)
        final = output / "final.txt"
        command = blackbox_command(self.binary, self.control, final)
        prompt = CodexExecLLM._prompt_text(messages)
        (output / "prompt.md").write_text(prompt)
        write_json(output / "command.json", command)
        started = time.monotonic()
        with (output / "events.jsonl").open("w") as events, (output / "stderr.log").open("w") as errors:
            process = subprocess.Popen(command, cwd=self.control, stdin=subprocess.PIPE,
                                       stdout=events, stderr=errors, text=True,
                                       start_new_session=True)
            timed_out = False
            try:
                process.communicate(prompt, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        completed = False
        tool_calls = []
        for line in (output / "events.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            completed |= row.get("type") == "turn.completed"
            item = row.get("item")
            if isinstance(item, dict) and item.get("type") in {
                    "command_execution", "mcp_tool_call", "web_search", "file_change"}:
                tool_calls.append(item.get("type"))
        text = final.read_text() if final.is_file() else ""
        success = process.returncode == 0 and not timed_out and completed and bool(text.strip()) and not tool_calls
        write_json(output / "receipt.json", {
            "model": MODEL, "exit_code": process.returncode, "timed_out": timed_out,
            "turn_completed": completed, "unexpected_tool_calls": tool_calls,
            "success": success, "wall_seconds": time.monotonic() - started,
            "timing_scope": "Codex process/turn wall time, including transport and client overhead; not pure API inference",
            "api_latency_seconds": None,
        })
        if not success:
            raise EvaluationInfrastructureError("Codex text-only request failed; inspect " + str(output))
        self.completed_calls += 1
        return text


class SingleRequestContext(ContextualLLM):
    """Reuse task/source/history prompts without EvoStage body conversion or retries."""

    def draw_sample(self, prompt, *args, **kwargs):
        self._counter += 1
        messages = self._messages(prompt, self._counter)
        response = self._provider.draw_sample(messages)
        with self._trace_path.open("a") as stream:
            stream.write(json.dumps({"order": self._counter, "messages": messages,
                                     "response": response, "provider_attempts": 1}) + "\n")
        return response


class LocalSmokeRuntime(Runtime):
    """Keep shared locks, but make every other batch's files read-only in Docker."""

    write_root = None
    image_preparation = None
    adopted_reference_run = None
    reference_purpose_receipt = None
    local_reference_selection = None

    def docker(self, command, gpu, scratch=None):
        if self.write_root is None:
            raise EvaluationInfrastructureError("Missing private evaluation write root")
        writable = self.contained(self.write_root)
        if writable == self.batch:
            raise ValueError("The shared batch root cannot be writable")
        image, extra_env = self.image, []
        if self.image_preparation:
            image, receipt = self.image_preparation(gpu)
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
                raise EvaluationInfrastructureError("Candidate builder did not return a pinned image")
            Path(receipt).resolve(strict=True).relative_to(writable)
            extra_env = ["-e", "OPEN3D_LOCAL_OPENROAD_PREBUILT_RECEIPT=" + str(receipt)]
        extra_env += ["-e", "OPEN3DBENCH_IMAGE_ID=" + image]
        reference_mounts = []
        if self.adopted_reference_run is not None:
            reference = Path(self.adopted_reference_run).resolve(strict=True)
            if not reference.is_relative_to(self.batch):
                reference_mounts = ["-v", str(reference) + ":" + str(reference) + ":ro"]
        command_line = super().docker(command, gpu, scratch)
        batch_mount = command_line.index(str(self.batch) + ":" + str(self.batch))
        command_line[batch_mount] += ":ro"
        image_index = command_line.index(self.image)
        command_line[image_index:image_index + 1] = [
            "-v", str(writable) + ":" + str(writable),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            *reference_mounts, *extra_env, image,
        ]
        return command_line


class DockerEvaluationTransport:
    """Exact evaluate_task argument transport; no changes to policies or scoring."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.attempts = []
        self.receipts = []

    @property
    def physical_calls(self):
        """Compatibility lower bound; see attempts for unknown execution states."""
        return sum(row["native_flow_started"] is True for row in self.attempts)

    def __call__(self, task_dir, candidate_dir, source, output_dir, *, baseline_run=None,
                 reference_run=None, base_config=None, dry_run=False, keep_workspace=True,
                 compact_artifacts=False, timeout_seconds=21600, create_routing_reference=False):
        if dry_run or compact_artifacts:
            raise ValueError("Smoke requires real evaluations and retained physical evidence")
        runtime = self.runtime
        output_dir, candidate_dir = runtime.contained(output_dir), runtime.contained(candidate_dir)
        selection = getattr(runtime, "local_reference_selection", None)
        if selection is not None:
            selection.verify_selected()
            if (reference_run is None or Path(reference_run).resolve(strict=True) != selection.source_run
                    or Path(runtime.adopted_reference_run).resolve(strict=True) != selection.source_run):
                raise ValueError("Candidate reference differs from its independently qualified read-only authority")
        spec = load_task(task_dir)
        spec.root.relative_to(runtime.framework / "local_tasks")
        if Path(source).resolve() != runtime.source:
            raise ValueError("Evaluation source differs from the frozen controller")
        candidate = load_candidate(candidate_dir)
        verification = verify_candidate(spec, candidate, runtime.source)
        command = ["/usr/bin/python3", "-m", "local_bench", "evaluate", str(spec.root),
                   "--candidate", str(candidate_dir), "--open3dbench-root", str(runtime.source),
                   "--output-dir", str(output_dir), "--timeout", str(timeout_seconds)]
        for flag, path in (("--baseline-run", baseline_run), ("--reference-run", reference_run)):
            if path is not None:
                resolved = Path(path).resolve(strict=True)
                adopted = getattr(runtime, "adopted_reference_run", None)
                if flag != "--reference-run" or adopted is None or resolved != Path(adopted).resolve(strict=True):
                    resolved = runtime.contained(path)
                command += [flag, str(resolved)]
        if base_config is not None:
            config = Path(base_config).resolve(strict=True)
            latest_config = (getattr(runtime, "reference_baseline", None) == "reference_20261004"
                             and config in {runtime.framework / "local_bench/benchmark_inputs" /
                                (case + "_lol_ng45_reference_20261004") / "config.yaml"
                                for case in ("bp_fe", "bp", "ariane133", "swerv_wrapper")})
            if not latest_config and not config.is_relative_to(runtime.source):
                runtime.contained(config)
            command += ["--base-config", str(config)]
        if create_routing_reference:
            command.append("--create-routing-reference")
        # The callback is evaluated by Runtime.execute only after its shared
        # resource lock has been acquired, covering compilation as well as EDA.
        runtime.image_preparation = None
        if spec.kind in ROUTING_SOURCE_TASK_KINDS:
            if runtime.cpus < 16 or runtime.memory_gb < 48:
                raise ValueError("Routing image builds require admission for at least 16 CPUs / 48 GiB")
            def prepare_image(gpu):
                from scripts.build_routing_candidate_image import build_candidate_image
                build_options = {}
                build_base = runtime.image
                identity_selector = getattr(runtime, 'routing_build_identity', None)
                if identity_selector is not None:
                    build_base, binding = identity_selector()
                    build_options['base_runtime_equivalence'] = binding
                built = build_candidate_image(
                    task_dir=spec.root, candidate_dir=candidate_dir, source=runtime.source,
                    base_image=build_base, output_dir=output_dir.parent / "image-build",
                    threads=min(runtime.cpus, 8),
                    compiler_cache_dir=getattr(runtime, "compiler_cache_dir", None),
                    compiler_cache_gb=getattr(runtime, "compiler_cache_gb", 8), **build_options)
                return built["image"], Path(built["receipt"])
            runtime.image_preparation = prepare_image
        runtime.write_root = output_dir.parent
        attempt = {"output_dir": str(output_dir), "evaluator_dispatch_attempted": None,
                   "evaluator_process_started": None, "native_flow_started": None,
                   "native_flow_invocations_started": None}
        self.attempts.append(attempt)
        contract = {"kind": "local-task-smoke-evaluation", "task": spec.task_id,
                    "candidate_sha256": file_sha(candidate_dir / "candidate.json"),
                    "source_commit": runtime.source_commit, "image": runtime.image,
                    "reference": str(reference_run) if reference_run else None,
                    "create_routing_reference": create_routing_reference,
                    "policy": verification["patch"]}
        purpose = getattr(runtime, "reference_purpose_receipt", None)
        if purpose is not None:
            purpose = runtime.contained(purpose)
            receipt = json.loads(purpose.read_text())
            purpose_kind = receipt.get("reference_purpose")
            valid_purpose = ((purpose_kind == "FRESH" and reference_run is None) or
                (purpose_kind == "INTERFACE_PROBE" and spec.task_id == "tp_gnn_learning_objective_reference_20260923"
                 and selection is not None and reference_run is not None
                 and receipt.get("reference_run") == str(selection.source_run)))
            if (not valid_purpose or receipt.get("evaluation_root") != str(output_dir)
                    or receipt.get("candidate_sha256") != contract["candidate_sha256"]):
                raise ValueError("Native reference/probe transport differs from its controller receipt")
            contract.update(reference_purpose=purpose_kind, reference_purpose_sha256=file_sha(purpose))
            # This trusted bootstrap only hashes installed Python source, then
            # replaces itself with the unchanged native evaluator command.
            if spec.task_id == "tp_gnn_learning_objective_reference_20260923":
                bootstrap = ("import os,sys; from local_bench.reference_local_partition import audit_runtime; "
                             "audit_runtime(sys.argv[1],sys.argv[2]); os.execv(sys.argv[3],sys.argv[3:])")
            else:
                bootstrap = ("import os,sys; from local_bench.reference_local_placement import audit_installed_runtime; "
                             "audit_installed_runtime(sys.argv[1],sys.argv[2]); os.execv(sys.argv[3],sys.argv[3:])")
            command = ["/usr/bin/python3", "-c", bootstrap, str(purpose),
                       str(output_dir.parent / "runtime-source.json"), *command]
        try:
            code = runtime.execute(command, output_dir, output_dir.parent / "evaluation.log", contract)
        except Exception as error:
            # Runtime owns its ledger, including failure states. Never rewrite
            # an older failed/uncertain attempt when execute refuses a retry.
            raise EvaluationInfrastructureError("Docker evaluation could not complete: " + str(error)) from error
        finally:
            execution = getattr(runtime, "last_execution", None)
            if isinstance(execution, dict):
                for key in ("evaluator_dispatch_attempted", "evaluator_process_started",
                            "admission_wait_seconds", "preparation_wall_seconds",
                            "evaluator_wall_seconds", "total_wall_seconds", "ledger", "phase_events"):
                    attempt[key] = execution.get(key)
            elif isinstance(runtime, Runtime):
                # This Runtime clears last_execution before its no-retry guard.
                attempt["evaluator_dispatch_attempted"] = False
                attempt["evaluator_process_started"] = False
            if attempt["evaluator_dispatch_attempted"] is False:
                attempt["native_flow_started"] = False
                attempt["native_flow_invocations_started"] = 0
            runtime.write_root = None
            runtime.image_preparation = None
        result_path = output_dir / "evaluation_result.json"
        if not result_path.is_file():
            raise EvaluationInfrastructureError("Missing evaluation_result.json after Docker exit " + str(code))
        raw = json.loads(result_path.read_text())
        if not isinstance(raw.get("metrics"), dict) or not isinstance(raw.get("artifacts"), dict):
            raise EvaluationInfrastructureError("Malformed physical evaluation result")
        native = raw["artifacts"].get("native_flow_execution")
        if isinstance(native, list) and all(isinstance(row, dict) and
                (row.get("process_started") is None or type(row.get("process_started")) is bool)
                for row in native):
            states = [row.get("process_started") for row in native]
            attempt["native_flow_started"] = (True if True in states else None if None in states else False)
            attempt["native_flow_invocations_started"] = sum(state is True for state in states)
            attempt["native_flow_invocations_unknown"] = sum(state is None for state in states)
        self.receipts.append({"path": str(result_path), "sha256": file_sha(result_path), "exit_code": code})
        result = EvaluationResult(raw["metrics"], raw["artifacts"])
        if code not in (0, 1) or (code == 0) != bool(result.metrics.get("runs_successfully")):
            raise EvaluationInfrastructureError("Docker exit/result disagree; inspect " + str(result_path))
        if result.metrics.get("runs_successfully") and not _successful(result):
            raise EvaluationInfrastructureError("Successful evaluator result has no finite score")
        if result.artifacts.get("error_type") in {
                "FileNotFoundError", "ModuleNotFoundError", "ImportError", "PermissionError", "OSError"}:
            raise EvaluationInfrastructureError("Evaluator environment failure: " + str(result.artifacts))
        return result


def execution_accounting(transport, reference_output):
    """Keep attempted finals, evaluator dispatch and physical evidence separate.

    Legacy results without a native-flow receipt remain unknown, never zero.
    Counts are candidate-level; one candidate may invoke multiple flow commands.
    """
    attempts = getattr(transport, "attempts", None)
    if attempts is None:
        # Custom/old transports have no dispatch ledger. Do not infer launch
        # from their historic physical_calls counter or result success flag.
        return {"execution_attempts": None, "candidate_physical_attempts": None,
                "reference_physical_attempts": None, "execution_accounting_available": False}
    result = {"execution_attempts": attempts, "execution_accounting_available": True}
    for prefix, selected in (("reference", [row for row in attempts if
            Path(row["output_dir"]).resolve() == reference_output.resolve()]),
            ("candidate", [row for row in attempts if
            Path(row["output_dir"]).resolve() != reference_output.resolve()])):
        for field, label in (("evaluator_dispatch_attempted", "evaluator_dispatches"),
                             ("native_flow_started", "physical_attempts")):
            known = sum(row.get(field) is True for row in selected)
            unknown = sum(row.get(field) is None for row in selected)
            result[prefix + "_" + label] = None if unknown else known
            result[prefix + "_" + label + "_known"] = known
            result[prefix + "_" + label + "_unknown"] = unknown
    result["physical_attempt_definition"] = "Candidates with at least one evidenced native-flow process launch; not individual OpenROAD tool invocations"
    return result


def run_smoke(runtime, *, task, output, benchmark_input=None, baseline_run=None,
              reference_run=None, rounds=5, codex_bin="codex", model=MODEL,
              timeout=21600, codex_timeout=1800, provider_factory=TextOnlyCodex,
              transport_factory=DockerEvaluationTransport, retention="full",
              compiler_cache_dir=None, compiler_cache_gb=8, reference_policy=None,
              reference_store=None, reference_record=None, reference_id=None,
              qualification_only=False, search_config=None, consumed_generation_slots=0):
    if type(qualification_only) is not bool:
        raise ValueError("qualification_only must be a boolean")
    validate_mode(retention)
    search_settings = None
    if search_config is not None:
        from local_bench.evolution_campaign import validate_configuration
        search_settings = validate_configuration(search_config, model, rounds,
            consumed_generation_slots=consumed_generation_slots)
    elif type(rounds) is not int or rounds not in AUTHORIZED_CANDIDATE_BUDGETS or model != MODEL:
        raise ValueError("This runner requires 2, 5 or 50 candidate slots and " + MODEL)
    if consumed_generation_slots and search_config is None:
        raise ValueError("Generation-slot carryover requires the formal comparison configuration")
    if benchmark_input is not None and baseline_run is not None:
        raise ValueError("Use benchmark-input or baseline-run, not both")
    if timeout <= 0 or codex_timeout <= 0:
        raise ValueError("Timeouts must be positive")
    spec = load_task(resolve_task_path(runtime.framework / "local_tasks", str(task)))
    require_executable_task(spec)
    spec.root.relative_to(runtime.framework / "local_tasks")
    # Search configuration is an offline prerequisite, not a failure to discover
    # only after a costly reference has completed.
    task_system_message(spec)
    base_config = spec.base_config
    input_report = None
    if benchmark_input is not None:
        benchmark_input = runtime.contained(benchmark_input)
        input_report = validate_benchmark_input(benchmark_input, task_id=spec.task_id)
        baseline_run = Path(input_report["run_dir"])
        base_config = Path(input_report["config"])
    reference_alignment = None
    versioned_local = spec.task_id in {
            "tp_gnn_learning_objective_reference_20260923",
            "placement_dynamic_parameters_reference_20260923",
            "placement_preconditioner_source_reference_20260923"}
    latest_routing = (spec.kind in ROUTING_SOURCE_TASK_KINDS
                      and getattr(runtime, "reference_baseline", None) == "reference_20261004")
    if versioned_local or latest_routing:
        from local_bench.reference_latest import baseline_for_bundle
        if benchmark_input is None:
            raise ValueError("Reference-aligned smoke requires an adopted benchmark-input bundle")
        selected_baseline = baseline_for_bundle(benchmark_input)
        selected_baseline.validate_reference_image(runtime.image)
        if (selected_baseline.BASELINE_ID == "reference_20261004"
                and getattr(runtime, "latest_framework_identity", None) is None):
            raise ValueError("Latest local task requires its independently pinned framework identity")
        initial_config = yaml.safe_load(base_config.read_text())
        case = initial_config.get("design") if isinstance(initial_config, dict) else None
        if not isinstance(case, str) or not case:
            raise ValueError("Reference task requires a design in its trusted base config")
        reference_alignment = selected_baseline.validate_reference_bundle(benchmark_input, case,
            fixed_partition=spec.task_id != "tp_gnn_learning_objective_reference_20260923")
        reference_alignment["case"] = case
        runtime.reference_baseline = selected_baseline.BASELINE_ID
        runtime.reference_root = Path(reference_alignment["reference_root"])
        if latest_routing:
            # The local import contains only three frontend stages. Routing
            # consumes the independently validated full native CTS prefix.
            baseline_run = runtime.reference_root
        if selected_baseline.BASELINE_ID == "reference_20261004":
            bundle_manifest = json.loads((benchmark_input / "benchmark_input.json").read_text())
            receipt_path = Path(bundle_manifest["latest_reference_receipt"]["path"])
            receipt = json.loads(receipt_path.read_text())
            native_config = yaml.safe_load((runtime.reference_root / "config.yaml").read_text())
            identity = receipt["runtime_identity"]
            proof = json.loads(Path(identity["execution_proof"]).read_text())
            mounts = list(dict.fromkeys([runtime.reference_root,
                *[Path(native_config[domain]["config"]) for domain in ("partition", "placement")],
                Path(identity["execution_proof"]),
                *[Path(record["path"]) for record in proof["bindings"].values()],
                *([Path(identity["equivalence_proof"])] if "equivalence_proof" in identity else [])]))
            runtime.latest_reference_mounts = tuple(path for path in mounts if not any(
                path == protected or protected in path.parents
                for protected in (runtime.source, runtime.framework, runtime.batch)))
    if spec.upstream_stages and baseline_run is None:
        raise ValueError("This task requires its frozen upstream input")
    if baseline_run is not None:
        baseline_run = runtime.contained(baseline_run)
    if reference_run is not None:
        reference_run = runtime.contained(reference_run)
    # The prompt and physical evaluator must describe the same resolved case.
    # A bundle-selected bp/ariane/swerv recipe must not retain the task's bp_fe
    # default in model-facing context.
    spec = replace(spec, base_config=base_config)
    output = runtime.contained(output)
    selection = None
    if versioned_local and reference_alignment is not None:
        if reference_run is not None:
            raise ValueError("Versioned local reference-run is not an authority; select reference-record/ID")
        from scripts.local_reference import LocalPlacementReference, LocalPartitionReference
        reference_class = (LocalPartitionReference if spec.task_id == "tp_gnn_learning_objective_reference_20260923"
                           else LocalPlacementReference)
        selection = reference_class(runtime, spec, case, benchmark_input, output,
            selected_policy=reference_policy, store=reference_store, record=reference_record, reference_id=reference_id)
    elif any(value is not None for value in (reference_policy, reference_store, reference_record, reference_id)):
        raise ValueError("Reference policy/store options apply only to versioned local reference tasks")
    protected = [runtime.source, *[path for path in (benchmark_input, baseline_run, reference_run) if path is not None]]
    if reference_alignment is not None:
        protected += [Path(reference_alignment[key]).resolve() for key in ("reference_root", "original_root")
                      if reference_alignment.get(key)]
    if output == runtime.batch or any(output == path or output in path.parents or path in output.parents
                                      for path in protected):
        raise ValueError("Output must be separate from source and all frozen inputs/references")
    if type(compiler_cache_gb) is not int or not 1 <= compiler_cache_gb <= 64:
        raise ValueError("compiler_cache_gb must be an integer in [1, 64]")
    if compiler_cache_dir is not None:
        if spec.kind not in ROUTING_SOURCE_TASK_KINDS:
            raise ValueError("Shared compiler cache is supported only for routing-source tasks")
        cache = Path(compiler_cache_dir).expanduser().absolute()
        if any(path.is_symlink() for path in (cache, *cache.parents)):
            raise ValueError("Compiler cache path cannot contain symlinks")
        cache = runtime.contained(cache)
        if cache == runtime.batch or any(cache == path or cache in path.parents or path in cache.parents
                                        for path in [output, *protected]):
            raise ValueError("Compiler cache must be separate from outputs and frozen inputs/references")
        compiler_cache_dir = cache
    runtime.compiler_cache_dir, runtime.compiler_cache_gb = compiler_cache_dir, compiler_cache_gb
    runtime.local_reference_selection = None
    runtime.adopted_reference_run = None
    runtime.reference_purpose_receipt = None
    output.mkdir(parents=True, exist_ok=False)
    initial_candidate = load_candidate(spec.initial_candidate)
    initial = initial_program_text(spec, initial_candidate)
    contract = {"schema_version": 1, "task": spec.task_id, "model": model,
                "search_backend": "openevolve-ask-tell" if search_settings else "codex-blackbox-ask-tell-smoke",
                "search_settings": search_settings,
                "consumed_generation_slots_before_run": consumed_generation_slots,
                "total_candidate_budget_including_prior_failures": rounds + consumed_generation_slots,
                "source_commit": runtime.source_commit, "image": runtime.image,
                "generated_candidate_budget": rounds, "reference_excluded_from_budget": True,
                "qualification_only": qualification_only,
                "benchmark_input": str(benchmark_input) if benchmark_input else None,
                "baseline_run": str(baseline_run) if baseline_run else None,
                "normalization_reference": str(reference_run) if reference_run else None,
                "base_config": str(base_config), "base_config_sha256": file_sha(base_config),
                "task_sha256": file_sha(spec.root / "task.yaml"),
                "initial_program_sha256": hashlib.sha256(initial.encode()).hexdigest(),
                "cpus": runtime.cpus, "memory_gb": runtime.memory_gb,
                "share_gpu": runtime.share_gpu, "min_gpu_free_mb": runtime.min_gpu_free_mb,
                "min_data_free_gb": getattr(runtime, "min_data_free_gb", 40),
                "cache_results": False, "input_validation": input_report,
                "reference_alignment": reference_alignment,
                "reference_selection": selection.to_dict() if selection else None,
                "compiler_cache": {"enabled": compiler_cache_dir is not None,
                    "root": str(compiler_cache_dir) if compiler_cache_dir else None,
                    "max_size_gb_per_namespace": compiler_cache_gb,
                    "evaluation_result_reuse": False},
                "retention": retention}
    write_json(output / "run_config.json", contract)
    transport = transport_factory(runtime)
    harness = provider = None
    state = "reference_failed"
    reference_count = 0
    try:
        if selection is not None and selection.policy == "ADOPT":
            reference = selection.obtain_adopted(output / "reference/adoption")
            qualified_reference = Path(reference.artifacts["run_dir"]).resolve(strict=True)
            runtime.adopted_reference_run = qualified_reference
        else:
            reference_candidate = output / "reference" / "candidate"
            reference_candidate.mkdir(parents=True)
            write_json(reference_candidate / "candidate.json", materialize_program(spec, initial, runtime.source))
            if selection is not None:
                runtime.reference_purpose_receipt = selection.prepare_fresh(output / "reference/result", reference_candidate)
            reference_count = 1
            try:
                reference = transport(spec.root, reference_candidate, runtime.source, output / "reference/result",
                                      baseline_run=baseline_run, reference_run=reference_run,
                                      base_config=base_config, timeout_seconds=timeout,
                                      create_routing_reference=spec.kind in ROUTING_SOURCE_TASK_KINDS and reference_run is None)
            finally:
                runtime.reference_purpose_receipt = None
            if not _qualified_reference(spec, reference):
                raise EvaluationInfrastructureError("Initial reference failed qualification; no model calls were made")
            qualified_reference = runtime.contained(reference.artifacts["run_dir"])
            if selection is not None:
                selection.publish(output / "reference/result")
                reference = selection.qualified_result(origin="fresh")
        if not _qualified_reference(spec, reference):
            raise EvaluationInfrastructureError("Initial reference failed qualification; no model calls were made")
        if not qualified_reference.is_dir():
            raise EvaluationInfrastructureError("Qualified reference has no retained run directory")
        if selection is not None:
            runtime.local_reference_selection = selection
            runtime.adopted_reference_run = qualified_reference
            # FRESH learns its sealed authority only after native qualification.
            # Persist that exact selection before any candidate/model work; a
            # failed reference keeps the original, unqualified run contract.
            contract["reference_selection"] = selection.to_dict()
            contract["normalization_reference"] = str(qualified_reference)
            write_json(output / "run_config.json", contract)
        write_json(output / "reference/qualification.json", {
            "success": True, "evaluation": reference.to_dict(), "run_dir": str(qualified_reference),
            "result_sha256": file_sha(output / "reference/result/evaluation_result.json") if reference_count else None,
            "reference_selection": selection.to_dict() if selection else None,
            "candidate_budget_consumed": 0,
        })
        if qualification_only:
            summary = {
                "state": "qualified", "task": spec.task_id,
                "qualification_only": True, "reference_qualified": True,
                "reference_evaluations": reference_count,
                "reference_selection": selection.to_dict() if selection else None,
                "normalization_reference": str(qualified_reference),
                "generated_candidate_budget": rounds,
                "candidate_budget_consumed": 0, "model_requests": 0,
                "api_request_count": 0, "generated_candidates": 0,
                "candidate_finals": 0,
                **execution_accounting(transport, output / "reference/result"),
            }
            write_json(output / "summary.json", summary)
            write_json(output / "status.json", {"state": "qualified", "candidate_finals": 0})
            # Qualification evidence is always retained in full.
            return summary
        state = "initializing_search"
        options = TaskEvaluationOptions(task_dir=spec.root, output_dir=output / "evolution",
                                        open3dbench_root=runtime.source, baseline_run=baseline_run,
                                        reference_run=qualified_reference, base_config=base_config,
                                        timeout_seconds=timeout)
        harness = ProgramEvaluationHarness(options, evaluation_transport=transport, cache_results=False)
        provider = provider_factory(output / "model", codex_bin, codex_timeout)
        prompt_spec = replace(
            spec,
            benchmark_input_id=(input_report.get("benchmark_id") if input_report
                                else spec.benchmark_input_id),
        )
        description = build_problem_description(prompt_spec, initial_candidate, runtime.source)
        description += "\n\n# Qualified initial reference\n" + json.dumps(reference.metrics, sort_keys=True)
        contextual = SingleRequestContext(provider, harness, output / "samples.jsonl", description)
        state = "running"
        write_json(output / "status.json", {"state": state, "candidate_finals": 0})

        def ask(context):
            text = contextual.draw_sample(
                "Generate candidate {} of {}. Review the qualified reference and prior feedback; "
                "return only the complete task program. No tool execution is needed.".format(context.evaluations + 1, rounds))
            # Empty completions are transport failures, not fabricated programs.
            proposal_id = "candidate-{:02d}".format(context.evaluations + 1)
            (output / "evolution" / (proposal_id + ".py")).write_text(text)
            return [CandidateProposal(proposal_id, text)]

        def tell(feedback):
            write_json(output / "status.json", {"state": "running", "candidate_finals": len(harness.records),
                                                "last_feedback": feedback[-1].to_dict()})
            for item in feedback:
                if item.error_type == "EvaluationInfrastructureError":
                    raise EvaluationInfrastructureError(item.error_message)

        backend = CallbackSearchBackend("codex-blackbox-ask-tell-smoke", ask, tell, lambda: False)
        if search_settings is None:
            result = AskTellRunner().run(backend, harness,
                                        SearchProblem(spec.task_id, description, initial), max_evaluations=rounds)
        else:
            from local_bench.openevolve_search import OpenEvolveAskTell
            from local_bench.evolution_campaign import RecordedBackend
            from local_bench.search_session import prepare_search_problem, search_feedback
            problem = prepare_search_problem(options)
            native = OpenEvolveAskTell(search_config, generate=lambda system, user: provider.draw_sample([
                {"role": "system", "content": system}, {"role": "user", "content": user}]))
            backend = RecordedBackend(native, output / "openevolve", tell)
            try:
                result = AskTellRunner().run(backend, harness, problem, max_evaluations=rounds,
                    feedback_transform=lambda item: search_feedback(problem, item))
            finally:
                backend.close()
        write_json(output / "search_result.json", result.to_dict())
        state = "completed" if len(result.history) == rounds else "needs_attention"
        error = None
    except Exception as caught:
        if state != "reference_failed":
            state = "needs_attention"
        error = {"type": type(caught).__name__, "message": str(caught)}
    rows = harness.records if harness else []
    summary = {"state": state, "task": spec.task_id, "reference_evaluations": reference_count,
               "search_backend": contract["search_backend"],
               "candidate_budget_consumed": max(len(rows), getattr(provider, "calls", 0)),
               "consumed_generation_slots_before_run": consumed_generation_slots,
               "candidate_budget_consumed_including_prior_failures": consumed_generation_slots + max(len(rows), getattr(provider, "calls", 0)),
               "generated_candidate_budget": rounds, "reference_excluded_from_budget": True,
               "reference_policy": selection.policy if selection else "legacy-initial-evaluation",
               "reference_selection": selection.to_dict() if selection else None,
               "reference_qualified": (output / "reference/qualification.json").is_file(),
               "model_requests": getattr(provider, "calls", 0),
               "model_request_definition": "Text-only provider/CLI invocations; underlying API request count is not measured",
               "api_request_count": None,
               "generated_candidates": getattr(provider, "completed_calls", getattr(provider, "calls", 0)),
               "candidate_finals": len(rows),
               **execution_accounting(transport, output / "reference/result"),
               "model_timing_scope": "Process/turn wall time; pure API inference latency is not measured",
               "api_latency_seconds": None,
               "candidate_evaluation_results": sum(
                   Path(receipt["path"]).resolve()
                   != (output / "reference/result/evaluation_result.json").resolve()
                   for receipt in transport.receipts),
               "successful_candidate_evaluations": sum(row["status"] == "accepted" for row in rows),
               "rejected_candidates": sum(row["status"] == "rejected" for row in rows),
               "failed_candidate_evaluations": sum(row["status"] == "failed" for row in rows),
               "cache_hits": sum(row["status"] == "cached" for row in rows),
               "duplicate_candidate_outputs": len(rows) - len({row["candidate_sha256"] for row in rows}),
               "physical_receipts": transport.receipts, "error": error,
               "harness": harness.summary() if harness else None,
               "interpretation": "{} generated proposal slots include rejected or failed candidates; reference is separate. No success is synthesized or cached.".format(rounds)}
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"state": state, "candidate_finals": len(rows), "error": error})
    finish_retention(output, retention)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "batch-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", required=True)
    for name in ("benchmark-input", "baseline-run", "reference-run"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--reference-policy", choices=("ADOPT", "FRESH", "EXTEND"),
                        help="Versioned placement defaults to ADOPT; no automatic FRESH fallback")
    parser.add_argument("--reference-store", type=Path)
    parser.add_argument("--reference-record", type=Path)
    parser.add_argument("--reference-id")
    parser.add_argument("--reference-baseline", choices=("reference_20260923", "reference_20261004"))
    parser.add_argument("--framework-sha256")
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=48)
    parser.add_argument("--min-gpu-free-mb", type=int, default=2048)
    parser.add_argument("--rounds", type=int, choices=AUTHORIZED_CANDIDATE_BUDGETS, default=5,
                        help="Generated candidate slots, including failures/rejections; reference is separate")
    parser.add_argument("--model", choices=[MODEL], default=MODEL)
    parser.add_argument("--qualification-only", action="store_true",
                        help="Qualify the unchanged reference; never construct a model provider or consume candidates")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-timeout", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--retention", choices=("full", "best"), default="full",
                        help="full: diagnostic archive; best: compact successful losers after search completion")
    parser.add_argument("--compiler-cache-dir", type=Path,
                        help="Opt-in shared ccache directory inside batch root, outside all runs and frozen inputs")
    parser.add_argument("--compiler-cache-gb", type=int, default=8,
                        help="ccache limit per compatible namespace (1-64 GiB), not a global cache budget")
    args = parser.parse_args(argv)
    runtime = LocalSmokeRuntime(args.source, args.batch_root, args.image, args.gpu,
                               cpus=args.cpus, memory_gb=args.memory_gb, share_gpu=True,
                               min_gpu_free_mb=args.min_gpu_free_mb, min_data_free_gb=64,
                               reference_baseline=args.reference_baseline, framework_sha256=args.framework_sha256)
    summary = run_smoke(runtime, task=args.task, output=args.output, benchmark_input=args.benchmark_input,
                        baseline_run=args.baseline_run, reference_run=args.reference_run,
                        rounds=args.rounds, codex_bin=args.codex_bin, model=args.model,
                        timeout=args.timeout, codex_timeout=args.codex_timeout, retention=args.retention,
                        compiler_cache_dir=args.compiler_cache_dir, compiler_cache_gb=args.compiler_cache_gb,
                        reference_policy=args.reference_policy, reference_store=args.reference_store,
                        reference_record=args.reference_record, reference_id=args.reference_id,
                        qualification_only=args.qualification_only)
    print(json.dumps(summary, indent=2))
    return 0 if summary["state"] in {"completed", "qualified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
