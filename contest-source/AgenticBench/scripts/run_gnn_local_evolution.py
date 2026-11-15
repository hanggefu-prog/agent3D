#!/usr/bin/env python3
"""Fresh native GNN references followed by exactly five single-request rounds.

``prepare`` creates only shared frontend/timing inputs. ``run`` qualifies the
task's exact initial program independently before making any model requests.
There is no retry, resume, API-key mount, candidate cache or native rebuild.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from urllib.parse import urlparse

import yaml

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK))
sys.path.insert(0, str(FRAMEWORK.parent))

from agentic3d.knowledge import write_json
from local_bench.llm_client import LLMConfig, OpenAICompatibleLLM
from local_bench.model import load_candidate, load_task, require_executable_task
from local_bench.openevolve_search import OpenEvolveAskTell
from local_bench.program_bridge import materialize_program
from local_bench.search_api import AskTellRunner, ProgramEvaluationHarness, TaskEvaluationOptions
from local_bench.search_session import prepare_search_problem, search_feedback
from scripts.run_local_task_smoke import (
    DockerEvaluationTransport, EvaluationInfrastructureError, LocalSmokeRuntime,
    execution_accounting, file_sha,
)

TASKS = ("tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing")
MODEL = "deepseek-flash"
TOKEN_ENV = "OPEN3D_GNN_EVOLUTION_API_KEY"
TIMING_FILES = ("timing_features.csv", "timing_net_weights.csv", "timing_manifest.json",
                "openroad_full_partition_last.connections.tsv")


def _safe_error(error, secret=None):
    message = str(error)
    for value in (secret, os.environ.get(TOKEN_ENV)):
        if value:
            message = message.replace(value, "[REDACTED]")
    return message[:4000]


def _status(root, state, **values):
    path = Path(root) / "status.json"
    previous = json.loads(path.read_text()) if path.is_file() else {}
    write_json(path, {**previous, "state": state, "updated_at_unix": time.time(), **values})


def _assert_source(runtime):
    revision = subprocess.check_output(["git", "-C", str(runtime.source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(runtime.source), "status", "--porcelain",
                                     "--untracked-files=no"], text=True)
    if revision != runtime.source_commit or dirty:
        raise ValueError("The frozen source changed; this launch cannot continue")


class GNNRuntime(LocalSmokeRuntime):
    """Single-GPU installed-tool execution with the shared admission ledger."""
    def docker(self, command, gpu, scratch=None):
        _assert_source(self)
        argv = super().docker(command, gpu, scratch)
        position = argv.index(self.image)
        environment = {
            "OPEN3DBENCH_IMAGE_ID": self.image, "OPEN3DBENCH_IMAGE": self.image,
            "OPENROAD_EXE": "/usr/local/bin/openroad", "OPENSTA_EXE": "/usr/local/bin/sta",
            "YOSYS_EXE": "/opt/yosys/bin/yosys", "PLACE3D_DREAMPLACE_INSTALL": "/opt/dreamplace",
            "OPEN3D_LOCAL_DREAMPLACE_INSTALL": "/opt/dreamplace",
            "OPEN3DBENCH_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        additions = [argument for key, value in environment.items() for argument in ("-e", key + "=" + value)]
        argv[position:position] = additions
        return argv


def _new_output(runtime, output):
    output = runtime.contained(output)
    if output == runtime.batch or any(output == path or output in path.parents or path in output.parents
                                      for path in (runtime.source, runtime.framework)):
        raise ValueError("Output must be separate from frozen source and batch root")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _base_config(spec, selected):
    path = Path(selected).resolve(strict=True) if selected else spec.base_config
    config = yaml.safe_load(path.read_text())
    if config.get("threads") != 4:
        raise ValueError("The fixed GNN physical recipe requires four flow threads")
    return path, config


def _validate_preparation_stage(path, expected_stage):
    """Validate published artifacts without importing the physical checkout."""
    from local_bench.benchmark_input import _validate_stage_manifest
    payload = _validate_stage_manifest(Path(path), expected_stage)
    if not isinstance(payload.get("signature"), dict):
        raise ValueError("Shared preparation manifest lacks its signature")
    if any(Path(record["path"]).is_absolute() for record in payload["artifacts"].values()):
        raise ValueError("Shared preparation artifacts must use relative paths")
    return payload


def _verify_preparation(runtime, baseline):
    baseline = runtime.contained(baseline).resolve(strict=True)
    receipt = json.loads((baseline.parent / "prepare.json").read_text())
    if (receipt.get("purpose") != "shared_frontend_and_timing_only"
            or receipt.get("is_timing_reference") is not False
            or receipt.get("source_commit") != runtime.source_commit
            or receipt.get("image") != runtime.image
            or receipt.get("run_dir") != str(baseline)):
        raise ValueError("Shared preparation receipt does not match this frozen experiment")
    stage_names = {"00_input": "input", "10_floorplan": "floorplan", "20_partition": "partition"}
    if set(receipt["stage_manifests"]) != set(stage_names):
        raise ValueError("Shared preparation does not contain the exact frontend/timing prefix")
    for directory, digest in receipt["stage_manifests"].items():
        manifest = baseline / directory / "stage_manifest.json"
        _validate_preparation_stage(manifest, stage_names[directory])
        if file_sha(manifest) != digest:
            raise ValueError("Shared preparation stage changed")
    for name in TIMING_FILES:
        if file_sha(baseline / "20_partition/timing_input" / name) != receipt["timing_sha256"][name]:
            raise ValueError("Shared frozen timing input changed")
    return baseline


def prepare(runtime, *, output, base_config=None):
    """Run the binary initial capsule through partition, without reference scoring."""
    from local_bench.gnn_evolution_timing import validate_native_evolution, validate_recipe
    from local_bench.workspace import isolated_clone
    output = _new_output(runtime, output)
    _status(output, "preparing", model_requests=0, candidate_budget_consumed=0)
    try:
        spec = load_task(runtime.framework / "local_tasks" / TASKS[0])
        _, config = _base_config(spec, base_config)
        initial = materialize_program(spec, spec.program_template.read_text(), runtime.source)
        if initial != load_candidate(spec.initial_candidate):
            raise ValueError("The exact initial candidate no longer materializes canonically")
        capsule = output / "binary_loss_program.json"
        write_json(capsule, initial["candidate_files"][capsule.name])
        config["partition"]["evolution_program"] = str(capsule)
        config["run_dir"] = str(output / "run")
        validate_recipe(config, runtime.source, task_id=TASKS[0])
        configuration = output / "config.yaml"
        configuration.write_text(yaml.safe_dump(config, sort_keys=False))
        workspace = isolated_clone(runtime.source, output / "workspace")
        command = ["/usr/bin/python3", "-m", "open3dbench", "--repo", str(workspace),
                   "run", str(configuration), "--to", "partition"]
        runtime.write_root = output
        try:
            code = runtime.execute(command, output / "run", output / "prepare.log", {
                "kind": "gnn-shared-input-preparation", "source_commit": runtime.source_commit,
                "image": runtime.image, "is_timing_reference": False, "model_requests": 0})
        finally:
            runtime.write_root = None
        if code:
            raise EvaluationInfrastructureError("Shared preparation failed; no retry is permitted")
        run = output / "run"
        native = validate_native_evolution(run, task_id=TASKS[0])
        stages = {}
        for directory, stage in (("00_input", "input"), ("10_floorplan", "floorplan"), ("20_partition", "partition")):
            path = run / directory / "stage_manifest.json"
            _validate_preparation_stage(path, stage)
            stages[directory] = file_sha(path)
        receipt = {"schema_version": 1, "purpose": "shared_frontend_and_timing_only",
                   "is_timing_reference": False, "run_dir": str(run),
                   "source_commit": runtime.source_commit, "image": runtime.image,
                   "compute_workspace": str(workspace),
                   "config_sha256": file_sha(configuration), "stage_manifests": stages,
                   "timing_sha256": {name: file_sha(run / "20_partition/timing_input" / name)
                                     for name in TIMING_FILES}, "native_partition": native,
                   "model_requests": 0}
        write_json(output / "prepare.json", receipt)
        _status(output, "prepared", run_dir=str(run), is_timing_reference=False)
        return receipt
    except Exception as error:
        _status(output, "failed", error_type=type(error).__name__, error_message=_safe_error(error))
        raise


def _credential_file(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        raise ValueError("API credential must be an owner-only mode-600 regular file")
    return path


def _provider_config(base_url, timeout=1800):
    parsed = urlparse(base_url)
    local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if (not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not (parsed.scheme == "https" or local_http)):
        raise ValueError("Use HTTPS or a loopback HTTP relay base URL without credentials")
    return LLMConfig(provider="deepseek", base_url=base_url.rstrip("/"), model=MODEL,
                     api_key_env=TOKEN_ENV, api_mode="chat_completions", timeout_seconds=timeout,
                     max_tokens=32768, temperature=None, top_p=None, reasoning_effort="high",
                     max_attempts=1, stream=True)


class AuditedProvider:
    """One request per charged slot; credentials never enter durable records."""
    def __init__(self, root, config, key_file, status, *, factory=OpenAICompatibleLLM):
        self.root, self.config, self.status = Path(root), config, status
        self.root.mkdir(parents=True, exist_ok=False)
        self.calls = self.completed = 0
        self._previous = os.environ.get(TOKEN_ENV)
        self._secret = _credential_file(key_file).read_text().strip()
        if not self._secret or any(character.isspace() for character in self._secret):
            raise ValueError("API credential file must contain one nonempty token")
        os.environ[TOKEN_ENV] = self._secret
        try:
            self.client = factory(config)
        except Exception as error:
            message = _safe_error(error, self._secret)
            self.close()
            raise EvaluationInfrastructureError("Model provider initialization failed: " + message) from None

    def generate(self, system, user):
        if self.calls >= 5:
            raise RuntimeError("The five-request model budget is exhausted")
        if self._secret in system or self._secret in user:
            raise ValueError("Credential unexpectedly appeared in model context")
        self.calls += 1
        root = self.root / ("call-%03d" % self.calls)
        root.mkdir(exist_ok=False)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        write_json(root / "prompt.json", messages)
        started = time.monotonic()
        receipt = {"request": self.calls, "model": MODEL, "provider_attempts": 1,
                   "stream": True, "max_attempts": 1, "state": "started",
                   "prompt_sha256": file_sha(root / "prompt.json")}
        write_json(root / "receipt.json", receipt)
        self.status("generating", model_requests=self.calls, candidate_budget_consumed=self.calls,
                    current_round=self.calls)
        try:
            answer = self.client.draw_sample(messages)
            if not isinstance(answer, str) or not answer.strip() or self._secret in answer:
                raise ValueError("Model returned an empty or unsafe completion")
            (root / "response.txt").write_text(answer)
            self.completed += 1
            receipt.update(state="completed", response_sha256=file_sha(root / "response.txt"))
            return answer
        except Exception as error:
            message = _safe_error(error, self._secret)
            receipt.update(state="failed", error_type=type(error).__name__, error_message=message)
            raise EvaluationInfrastructureError("Single model request failed; its slot remains consumed: " + message) from None
        finally:
            receipt["wall_seconds"] = time.monotonic() - started
            write_json(root / "receipt.json", receipt)

    def close(self):
        if getattr(self, "client", None) is not None:
            self.client.close()
        if self._previous is None:
            os.environ.pop(TOKEN_ENV, None)
        else:
            os.environ[TOKEN_ENV] = self._previous
        self._secret = ""


def search_config():
    from openevolve.config import Config, LLMModelConfig
    config = Config(diff_based_evolution=False)
    config.max_iterations = 5
    config.random_seed = 42
    config.database.population_size = config.database.archive_size = 6
    config.database.num_islands = 1
    config.database.feature_dimensions = ["complexity", "diversity"]
    config.database.random_seed = 42
    config.database.migration_rate = 0
    config.prompt.num_top_programs, config.prompt.num_diverse_programs = 2, 1
    config.prompt.use_template_stochasticity = False
    config.llm.models = [LLMModelConfig(name=MODEL, weight=1, retries=0, timeout=1800, max_tokens=32768)]
    config.llm.evaluator_models = []
    config.evaluator.use_llm_feedback = False
    config.evaluator.max_retries = 0
    config.evaluator.parallel_evaluations = 1
    return config


class QualifiedOpenEvolve(OpenEvolveAskTell):
    def __init__(self, config, reference_metrics, *, generate):
        super().__init__(config, generate=generate)
        self.reference_metrics = dict(reference_metrics)

    def _initialize(self, context):
        super()._initialize(context)
        from openevolve.database import Program
        self.database.add(Program(
            id="qualified-initial", code=context.problem.initial_program, language="python",
            metrics=dict(self.reference_metrics),
            metadata={"origin": "fresh_reference", "candidate_budget_consumed": 0}),
            iteration=0, target_island=0)


class RecordedSearch:
    name = "openevolve-ask-tell"
    def __init__(self, backend, output, status):
        self.backend, self.output, self.status = backend, output, status
        self.finals = 0

    def ask(self, context):
        proposals = self.backend.ask(context)
        for proposal in proposals:
            (self.output / (proposal.proposal_id + ".py")).write_text(proposal.program)
        self.status("evaluating", current_round=context.evaluations + 1)
        return proposals

    def tell(self, feedback):
        self.backend.tell(feedback)
        self.finals += len(feedback)
        write_json(self.output / ("feedback-%03d.json" % self.finals), [item.to_dict() for item in feedback])
        write_json(self.output / ("population-%03d.json" % self.finals), {
            key: program.to_dict() for key, program in self.backend.database.programs.items()})
        self.status("round_completed", candidate_finals=self.finals, last_feedback=feedback[-1].to_dict())
        if any(item.error_type == "EvaluationInfrastructureError" for item in feedback):
            raise EvaluationInfrastructureError("Native evaluation infrastructure failed; no retry is permitted")


def run(runtime, *, task, output, baseline_run, key_file, base_url, rounds=5,
        base_config=None, timeout=21600, model_timeout=1800):
    if task not in TASKS or type(rounds) is not int or rounds != 5:
        raise ValueError("Only the two GNN tasks with exactly five generated slots are authorized")
    if type(timeout) is not int or timeout < 1 or type(model_timeout) is not int or model_timeout < 1:
        raise ValueError("Native and model timeouts must be positive integer seconds")
    _credential_file(key_file)
    credential = Path(key_file).resolve(strict=True)
    if any(credential.is_relative_to(root.resolve())
           for root in (runtime.batch, runtime.source, runtime.framework.parent)):
        raise ValueError("The API credential must stay outside every Docker-mounted tree")
    provider_settings = _provider_config(base_url, model_timeout)
    baseline = _verify_preparation(runtime, baseline_run)
    output = _new_output(runtime, output)
    status = lambda state, **values: _status(output, state, **values)
    status("qualifying_reference", task=task, model_requests=0, candidate_budget_consumed=0,
           candidate_finals=0, reference_qualified=False, retry=False, resume=False)
    provider = backend = harness = None
    transport = DockerEvaluationTransport(runtime)
    try:
        spec = load_task(runtime.framework / "local_tasks" / task)
        require_executable_task(spec)
        selected_config, _ = _base_config(spec, base_config)
        config = search_config()  # Check optional OpenEvolve dependency before native execution.
        write_json(output / "run_config.json", {
            "schema_version": 1, "task": task, "source_commit": runtime.source_commit,
            "image": runtime.image, "gpu": runtime.gpus, "cpus": runtime.cpus,
            "memory_gb": runtime.memory_gb, "flow_threads": 4,
            "baseline_run": str(baseline), "base_config": str(selected_config),
            "base_config_sha256": file_sha(selected_config), "task_sha256": file_sha(spec.root / "task.yaml"),
            "rounds": rounds, "reference_policy": "FRESH", "reference_excluded_from_budget": True,
            "cache_results": False, "resume": False, "retry": False,
            "provider": provider_settings.public_dict(), "search": asdict(config)})
        initial = materialize_program(spec, spec.program_template.read_text(), runtime.source)
        if initial != load_candidate(spec.initial_candidate):
            raise ValueError("The exact initial candidate no longer materializes canonically")
        candidate = output / "reference/candidate"
        candidate.mkdir(parents=True)
        write_json(candidate / "candidate.json", initial)
        reference = transport(spec.root, candidate, runtime.source, output / "reference/result",
                              baseline_run=baseline, base_config=selected_config, timeout_seconds=timeout)
        score = reference.metrics.get("combined_score")
        if (reference.metrics.get("runs_successfully") != 1 or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)
                or not math.isclose(score, 1., rel_tol=0., abs_tol=1e-12)):
            raise EvaluationInfrastructureError("The FRESH exact initial program did not qualify at score 1")
        qualified = runtime.contained(reference.artifacts["run_dir"]).resolve(strict=True)
        write_json(output / "reference/qualification.json", {
            "reference_policy": "FRESH", "task": task, "source_commit": runtime.source_commit,
            "image": runtime.image, "run_dir": str(qualified), "candidate_budget_consumed": 0,
            "evaluation": reference.to_dict(),
            "evaluation_result_sha256": file_sha(output / "reference/result/evaluation_result.json")})
        status("reference_qualified", reference_qualified=True, reference_run=str(qualified))
        options = TaskEvaluationOptions(
            task_dir=spec.root, output_dir=output / "evolution", open3dbench_root=runtime.source,
            baseline_run=baseline, reference_run=qualified, base_config=selected_config,
            evaluation_mode="full", timeout_seconds=timeout)
        harness = ProgramEvaluationHarness(options, evaluation_transport=transport, cache_results=False)
        problem = prepare_search_problem(options)
        problem = replace(problem, description=problem.description + "\n\nQualified exact initial reference metrics:\n"
                          + json.dumps(reference.metrics, sort_keys=True))
        provider = AuditedProvider(output / "model", provider_settings, key_file, status)
        def generate(system, user):
            _assert_source(runtime)
            return provider.generate(system, user)
        native = QualifiedOpenEvolve(config, reference.metrics, generate=generate)
        backend = RecordedSearch(native, output / "evolution", status)
        result = AskTellRunner().run(backend, harness, problem, max_evaluations=5,
                                    feedback_transform=lambda item: search_feedback(problem, item))
        write_json(output / "search_result.json", result.to_dict())
        if len(result.history) != 5 or provider.calls != 5:
            raise RuntimeError("The five-slot search ended early; it cannot be resumed or retried")
        summary = {"state": "completed", "task": task, "model_requests": provider.calls,
                   "generated_candidates": provider.completed, "candidate_finals": len(harness.records),
                   "reference_qualified": True, "reference_excluded_from_budget": True,
                   "candidate_budget_consumed": provider.calls, "harness": harness.summary(),
                   **execution_accounting(transport, output / "reference/result")}
        write_json(output / "summary.json", summary)
        status("completed", **{key: value for key, value in summary.items() if key != "state"})
        return summary
    except Exception as error:
        summary = {"state": "failed", "error_type": type(error).__name__,
                   "error_message": _safe_error(error, provider._secret if provider else None),
                   "model_requests": provider.calls if provider else 0,
                   "candidate_budget_consumed": provider.calls if provider else 0,
                   "candidate_finals": len(harness.records) if harness else 0,
                   "reference_qualified": (output / "reference/qualification.json").is_file(),
                   **execution_accounting(transport, output / "reference/result")}
        write_json(output / "summary.json", summary)
        status("failed", **{key: value for key, value in summary.items() if key != "state"})
        raise
    finally:
        if backend is not None:
            backend.backend.close()
        if provider is not None:
            provider.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        sub = commands.add_parser(name)
        for flag in ("source", "batch-root", "output"):
            sub.add_argument("--" + flag, type=Path, required=True)
        sub.add_argument("--source-commit", required=True)
        sub.add_argument("--image", required=True)
        sub.add_argument("--gpu", type=int, choices=range(8), required=True)
        sub.add_argument("--base-config", type=Path)
        if name == "run":
            sub.add_argument("--task", choices=TASKS, required=True)
            sub.add_argument("--baseline-run", type=Path, required=True)
            sub.add_argument("--api-key-file", type=Path, required=True)
            sub.add_argument("--base-url", required=True)
            sub.add_argument("--rounds", type=int, choices=[5], default=5)
            sub.add_argument("--timeout", type=int, default=21600)
            sub.add_argument("--model-timeout", type=int, default=1800)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        raise ValueError("Pin the installed Docker image by its exact SHA-256 ID")
    runtime = GNNRuntime(args.source, args.batch_root, args.image, str(args.gpu),
                         cpus=8, memory_gb=32, share_gpu=False,
                         min_gpu_free_mb=16000, min_data_free_gb=64)
    if runtime.source_commit != args.source_commit:
        raise ValueError("Source commit differs from the explicit frozen revision")
    try:
        if args.command == "prepare":
            result = prepare(runtime, output=args.output, base_config=args.base_config)
        else:
            result = run(runtime, task=args.task, output=args.output, baseline_run=args.baseline_run,
                         key_file=args.api_key_file, base_url=args.base_url, rounds=args.rounds,
                         base_config=args.base_config, timeout=args.timeout, model_timeout=args.model_timeout)
    except Exception as error:
        status_path = args.output / "status.json"
        saved = json.loads(status_path.read_text()) if status_path.is_file() else {}
        message = saved.get("error_message", _safe_error(error))
        print(json.dumps({"state": "failed", "error_type": type(error).__name__,
                          "error_message": message, "output": str(args.output)}))
        return 1
    print(json.dumps({"state": "prepared" if args.command == "prepare" else result["state"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
