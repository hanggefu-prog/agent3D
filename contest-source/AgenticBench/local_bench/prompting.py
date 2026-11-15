from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .model import ROUTING_SOURCE_TASK_KINDS, TaskSpec
from .program_bridge import initial_program_text, materialize_program
from .program_safety import normalized_program
from .optimization_feedback import TASK_IDS, ROUTING_TASKS, TEMPLATES, template_source, optimization_result


SYSTEM_MESSAGE = """You are evolving one candidate for OpenAgentic3DBench.
Treat the supplied task contract, source excerpts, parameter bounds, and output
protocol as mandatory. Optimize only from black-box evaluation evidence. Keep
one coherent, falsifiable hypothesis per candidate. Never invent repository
files or candidate fields that are not present in the contract."""


DIVERSITY_THEMES = (
    "change the early-stage exploration mechanism while preserving late-stage stability",
    "make the response adaptive to density overflow and its transition toward feasibility",
    "use normalized trend information to distinguish progress, stagnation, and regression",
    "balance wirelength pressure against density pressure with bounded smooth scaling",
    "improve robustness across the two fence regions without using design-specific constants",
    "test a conservative mechanism with explicit numerical clamps and monotone behavior",
    "test a more exploratory but still bounded mechanism that can escape a stalled trajectory",
    "simplify the mechanism and remove a baseline assumption that is not supported by feedback",
)

PARTITION_DIVERSITY_THEMES = (
    "rebalance positive and negative sampled-pair terms without changing the sampler",
    "test smooth hard-pair emphasis with stable gradients and bounded numerical domains",
    "use linear-memory embedding dispersion to resist collapse without an all-pairs matrix",
    "normalize sampled terms by their actual counts without changing the training budget",
    "test an epoch-dependent loss mechanism while preserving the frozen optimizer and decoder",
    "compare embedding diagnostics with raw and repaired partition cuts rather than training loss alone",
    "test a conservative loss correction that preserves finite non-detached scalar gradients",
    "simplify the loss while retaining the mechanism supported by final legal-partition feedback",
)

TIMING_PARTITION_DIVERSITY_THEMES = (
    "test bounded hard-negative weighting without changing negative sampling",
    "weight positive pairs using observed endpoint timing features, not invented per-pair net weights",
    "use graph_scores and frozen graph_weights for one normalized structural regularizer",
    "test a bounded linear-memory anti-collapse term and measure final legal weighted cut",
    "use closure cut delta to explain whether improved embeddings survive feasibility closure",
    "simplify a timing-aware loss to the single mechanism supported by final partition feedback",
)

ROUTING_NET_ORDER_DIVERSITY_THEMES = (
    "test a material departure from baseline rank and verify current-overflow order diagnostics",
    "compare one shared policy with an explicit physically justified bottom/top branch",
    "prioritize current-overflow nets while avoiding changes limited to ineligible zero-overflow nets",
    "trade overflow against wire length differently in early and late ordered reroute phases",
    "adapt the baseline mixing weight instead of fixing its coefficient to one",
    "test an HBT-aware policy whose effect scales with both die pressure and HBT pressure",
    "use a bounded iterative or piecewise mechanism that cannot collapse to a baseline-equivalent rank",
    "simplify a prior mechanism only after preserving the feature responsible for current-overflow order change",
)

ROUTING_SOURCE_DIVERSITY_THEMES = (
    "first make a material but bounded policy change and verify the hook diagnostics",
    "compare one shared policy with an explicit physically justified bottom/top branch",
    "use HBT and critical-net pressure only when it explains the observed die trade-off",
    "emphasize congestion early and recover wire length in later reroute phases",
    "respond to stagnation or history pressure instead of using a fixed global bias",
    "test a conservative correction around maintained FastRoute with explicit clamps",
    "test a materially different piecewise policy while preserving finite outputs",
    "simplify a prior mechanism only after retaining the feature linked to improvement",
)


def _diversity_themes(spec: TaskSpec) -> tuple[str, ...]:
    if spec.task_id == "tp_gnn_binary_loss_timing":
        return (
            "test one probability regularizer while keeping the fixed temperature and noise schedule",
            "compare raw hard area and closure movement with uncertainty, not confidence alone",
            "use frozen timing features to shape a bounded differentiable loss",
            "judge the complete legal partition by measured no-OPT WNS/TNS, not training loss",
        )
    if spec.task_id == "tp_gnn_loss_decoder_timing":
        return (
            "coordinate embedding loss with one bounded whole-design two-way clustering rule",
            "use physical area in clustering while keeping the final feasibility closure fixed",
            "compare decoder and repaired assignments to explain whether a mechanism survives closure",
            "judge the joint program by measured no-OPT WNS/TNS, not raw cut or embedding separation",
        )
    if spec.task_id == "hbt_assignment_policy":
        return (
            "test a geometry-based network priority and inspect final joint setup timing",
            "trade legal combined wirelength against distance from the ideal site",
            "normalize geometric costs by die span and compare the final setup timing trade-off",
            "simplify a policy while preserving improvements supported by comparable evaluations",
        )
    if spec.task_id in {"tp_gnn_bin_timing", "tp_gnn_post_placement_timing"}:
        decoder = "global 2-way" if spec.task_id == "tp_gnn_post_placement_timing" else "bin"
        return (
            "test a bounded timing-aware pair term and measure final post-placement WNS/TNS",
            "use sparse full-graph timing weights without changing the " + decoder + " decoder",
            "use bounded anti-collapse regularization and compare downstream timing, not loss alone",
            "simplify the loss to the mechanism supported by validated no-OPT timing feedback",
        )
    if spec.task_id == "tp_gnn_learning_objective_reference_20260923":
        return TIMING_PARTITION_DIVERSITY_THEMES
    if spec.kind == "tp_gnn_learning_objective":
        return PARTITION_DIVERSITY_THEMES
    if spec.kind == "routing_net_ordering_source":
        return ROUTING_NET_ORDER_DIVERSITY_THEMES
    if spec.kind in ROUTING_SOURCE_TASK_KINDS:
        return ROUTING_SOURCE_DIVERSITY_THEMES
    return DIVERSITY_THEMES


def task_system_message(spec: TaskSpec) -> str:
    """Load the task-specific system instruction used by every search backend."""

    import yaml

    path = spec.root / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("task evolution config must contain an object: {}".format(path))
    prompt = data.get("prompt", {})
    if not isinstance(prompt, Mapping):
        raise ValueError("task evolution prompt config must contain an object: {}".format(path))
    message = str(prompt.get("system_message", "")).strip()
    if not message:
        raise ValueError("task evolution config is missing prompt.system_message: {}".format(path))
    return message


def composed_system_message(spec: TaskSpec) -> str:
    if spec.task_id in TASK_IDS:
        return task_system_message(spec)
    return SYSTEM_MESSAGE + "\n\n# Task-specific system instruction\n\n" + task_system_message(spec)


def task_contract(spec: TaskSpec) -> Dict[str, Any]:
    contract = _base_task_contract(spec)
    if spec.scoring:
        # Only opt-in score contracts appear; default contracts stay byte-identical.
        contract["scoring"] = json.loads(json.dumps(spec.scoring, sort_keys=True))
    return contract


def _scoring_override(spec: TaskSpec) -> str:
    if not spec.scoring:
        return ""
    from .routing_scoring import scoring_contract_text
    return "\n\n" + scoring_contract_text(spec.scoring).rstrip()


def _base_task_contract(spec: TaskSpec) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "task": spec.task_id,
        "kind": spec.kind,
        "availability": spec.availability,
        "execution": {
            "from": spec.start_stage,
            "to": spec.stop_stage,
            "boundary": spec.evaluation_boundary,
            "frozen_upstream_stages": spec.upstream_stages,
            "frozen_inputs": [
                {
                    "name": item.name,
                    "source": item.source,
                    "path": item.path,
                    "inject_as": item.inject_as,
                    "required_files": item.required_files,
                    "json_checks": [
                        {"path": check.path, "equals": check.equals}
                        for check in item.json_checks
                    ],
                }
                for item in spec.frozen_inputs
            ],
        },
        "benchmark_input": spec.benchmark_input_id,
        "base_config": str(spec.base_config),
        "candidate_policy": {
            "allowed_config_paths": spec.allowed_config_paths,
            "allowed_source_paths": spec.allowed_source_paths,
            "candidate_files": [
                {
                    "path": item.path,
                    "inject_as": item.inject_as,
                    "allowed_paths": item.allowed_paths,
                }
                for item in spec.candidate_files
            ],
            "value_constraints": spec.value_constraints,
            "source_patch_limits": {
                "max_changed_files": spec.max_changed_files,
                "max_added_lines": spec.max_added_lines,
                "max_deleted_lines": spec.max_deleted_lines,
            },
        },
        "priority_black_box_parameters": spec.priority_parameters,
        "prompt_context": {
            "read_only_source_paths": spec.prompt_read_only_source_paths,
        },
        "program": {
            "mode": spec.program_mode,
            "entrypoint": spec.program_entrypoint,
            "template": str(spec.program_template) if spec.program_template else None,
            "adapter": str(spec.program_adapter) if spec.program_adapter else None,
        },
    }


def _numbered_excerpt(path: Path, start_line: int, end_line: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if start_line > len(lines):
        raise ValueError(
            "prompt source range starts after EOF: {}:{} ({} lines)".format(
                path, start_line, len(lines)
            )
        )
    stop = min(end_line, len(lines))
    width = len(str(stop))
    return "\n".join(
        "{line:>{width}} | {text}".format(line=index, width=width, text=lines[index - 1])
        for index in range(start_line, stop + 1)
    )


def source_context(spec: TaskSpec, open3dbench_root: Optional[Path]) -> str:
    if not spec.allowed_source_paths:
        return "This is a parameter-only task. source_patch must remain empty."
    if open3dbench_root is None:
        return (
            "Source excerpts are unavailable in verifier-only mode. The allowed paths are:\n- "
            + "\n- ".join(spec.allowed_source_paths)
        )
    root = open3dbench_root.expanduser().resolve()
    chunks = []
    used = 0
    for source_range in spec.prompt_source_ranges:
        relative = Path(source_range.path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(
                "prompt source path must be repository-relative: {}".format(
                    source_range.path
                )
            )
        # Large maintained dependencies may be tracked links whose storage is
        # adjacent to the checkout. The task-owned relative path is the trust
        # boundary; resolving it would incorrectly reject those exact paths.
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                "prompt source file is missing or escapes Open3DBench: {}".format(path)
            )
        excerpt = _numbered_excerpt(path, source_range.start_line, source_range.end_line)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        language = "cpp" if path.suffix in {".cc", ".cpp", ".cxx", ".h", ".hpp"} else "python"
        chunk = (
            "\n## `{}` lines {}-{} (sha256 {})\n```{}\n{}\n```\n".format(
                source_range.path,
                source_range.start_line,
                source_range.end_line,
                digest,
                language,
                excerpt,
            )
        )
        encoded = len(chunk)
        if used + encoded > spec.max_source_chars:
            raise ValueError(
                "configured source excerpts exceed prompt_context.max_source_chars ({})".format(
                    spec.max_source_chars
                )
            )
        chunks.append(chunk)
        used += encoded
    if not chunks:
        return (
            "No source excerpts were configured. The candidate may modify only:\n- "
            + "\n- ".join(spec.allowed_source_paths)
        )
    return "".join(chunks)


def build_problem_description(
    spec: TaskSpec,
    initial_candidate: Mapping[str, Any],
    open3dbench_root: Optional[Path] = None,
    *, include_initial_program: bool = True,
) -> str:
    import yaml

    task_prompt = spec.prompt.read_text(encoding="utf-8").strip()
    config_bytes = spec.base_config.read_bytes()
    selected_config = yaml.safe_load(config_bytes)
    if not isinstance(selected_config, Mapping):
        raise ValueError("selected benchmark base_config must contain a mapping")
    if spec.task_id in TASK_IDS:
        # Engineering contracts still drive validation, hashing and execution.
        settings = ""
        if spec.task_id in ROUTING_TASKS:
            routing = selected_config.get("routing", {})
            values = {key: routing[key] for key in ("congestion_iterations", "resource_reduction", "seed")
                      if key in routing and not isinstance(routing[key], bool)
                      and isinstance(routing[key], (int, float)) and math.isfinite(routing[key])}
            if values:
                settings = "\n\nFixed routing settings (not editable): " + json.dumps(values, sort_keys=True) + "."
        description = (task_prompt + "\n\n## Selected problem\n\n"
                + "Design: " + str(selected_config["design"])
                + "; integration: " + str(selected_config["type"])
                + "; platform: " + str(selected_config["pdk"])
                + "." + settings + _scoring_override(spec))
        if include_initial_program:
            description += ("\n\n## Initial program and exact interface\n\n```python\n"
                            + initial_program_text(spec, initial_candidate).rstrip() + "\n```\n")
        return description
    # Show the physical recipe, not arbitrary environment/credential settings.
    selected_recipe = {
        key: selected_config[key] for key in (
            "design", "type", "pdk", "threads", "input", "pins", "partition",
            "placement", "hbt", "opt", "cts", "routing", "thermal",
        ) if key in selected_config
    }
    contract = json.dumps(task_contract(spec), indent=2, sort_keys=True, ensure_ascii=False)
    initial = json.dumps(initial_candidate, indent=2, sort_keys=True, ensure_ascii=False)
    initial_program = initial_program_text(spec, initial_candidate)
    sources = source_context(spec, open3dbench_root)
    if spec.program_mode == "task_adapter":
        interface = """The exact initial program is:

```python
{initial_program}
```

Return the complete Python program, retaining the `EVOLVE-BLOCK` markers and
the entry point `{entrypoint}` with its exact signature. The trusted task adapter
parses this program without executing it, rejects code outside the declared
grammar, and deterministically materializes the bounded Open3DBench patch.
Do not return JSON or a unified diff.""".format(
            initial_program=initial_program.rstrip(),
            entrypoint=spec.program_entrypoint,
        )
    else:
        interface = """The initial candidate is:

```json
{initial}
```

Return exactly one Python function named `evolve_candidate` with no arguments.
Its body may contain only an optional docstring and one return statement whose
value is a static Python dictionary with candidate schema version 1. The
function is parsed but never executed. Do not import modules, call functions,
read files, or use environment variables. Preserve required fields. Parameters
belong in `config_overrides` or `candidate_files`; algorithm changes belong in
the unified-diff string `source_patch`. Record the tested hypothesis and expected
mechanism in `notes`.""".format(initial=initial)
    return """{task_prompt}

# Benchmark contract

The following contract comes from the resolved task, including explicit caller
overrides. The selected base configuration below fixes the actual case and
physical recipe; a historical task name or template profile is not an automatic
latest-family selection. A profile label alone is not evidence of input adoption
or scoring qualification. Require independently validated same-contract input
and reference authority before formal scoring; never silently regenerate it.

Source excerpts are read-only context, not edit permission. For task-adapter
programs, allowed_source_paths describe the trusted adapter's patch envelope,
not permission to edit whole files or the evaluator. Only the declared candidate
definitions may evolve. A constraint with `required: false` is optional; all
other declared values must remain present and valid.

Read feedback in this order: verifier/integration failures, evidence that the
declared native hook or component actually ran, final complete endpoint metrics,
then the score and per-die trade-offs. A successful build or process exit alone
does not establish physical success. Missing metrics remain unknown, never zero;
intermediate/scratch measurements are not harness finals. Keep runtime and
downstream timing/thermal/DRC diagnostics separate from this task's reward.

```json
{contract}
```

# Selected physical recipe

Base configuration SHA-256: `{config_sha256}`. Referenced configuration files
and physical artifacts still require the evaluator's independent hash checks.
Only the declared execution boundary runs; downstream settings shown here do
not extend that boundary or add metrics to the score.

```yaml
{selected_recipe}
```

# Relevant Open3DBench implementation

The excerpts below come from the exact dependency checkout used by the evaluator.
Line numbers are reference annotations and are not part of the files.

{sources}

# Candidate interface

{interface}
""".format(
        task_prompt=task_prompt + _scoring_override(spec),
        contract=contract,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        selected_recipe=yaml.safe_dump(selected_recipe, sort_keys=False).rstrip(),
        sources=sources,
        interface=interface,
    )


def _tail(path: Path, limit: int) -> str:
    if not path.is_file() or limit <= 0:
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _artifact_feedback(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    keep = {
        "cascade",
        "error_type",
        "error_message",
        "exit_code",
        "runtime_seconds",
        "patch",
    }
    result = {key: artifacts[key] for key in keep if key in artifacts}
    for key, limit in (("stage_log", 4000), ("plan_log", 2000)):
        value = artifacts.get(key)
        if isinstance(value, str):
            tail = _tail(Path(value), limit)
            if tail:
                result[key + "_tail"] = tail
    return result


def feedback_context(records: Iterable[Mapping[str, Any]], spec: TaskSpec) -> str:
    rows = list(records)
    if not rows:
        return "# Black-box evaluation feedback\n\nNo candidate has been evaluated yet."

    def compact(row: Mapping[str, Any]) -> Dict[str, Any]:
        if spec.task_id in TASK_IDS:
            from .model import EvaluationResult
            from .tp_gnn_openevolve import _reason
            metrics = row.get("metrics", {})
            metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
            artifacts = row.get("artifacts", {})
            artifacts = dict(artifacts) if isinstance(artifacts, Mapping) else {}
            error = row.get("error_message") or artifacts.get("error_message")
            if error:
                artifacts["error_message"] = error
            projected = optimization_result(EvaluationResult(metrics=metrics, artifacts=artifacts),
                                            task_id=spec.task_id, require_score=False)
            score = row.get("score")
            return {"order": row.get("order"), "status": row.get("status"),
                    "score": score if not isinstance(score, bool) and isinstance(score, (int, float))
                    and math.isfinite(score) else None,
                    "metrics": projected.metrics, "artifacts": projected.artifacts,
                    "error": _reason(error)}
        candidate_path = row.get("candidate")
        notes = ""
        if isinstance(candidate_path, str):
            try:
                candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
                notes = str(candidate.get("notes", ""))
            except (OSError, ValueError, TypeError):
                notes = ""
        return {
            "order": row.get("order"),
            "status": row.get("status"),
            "score": row.get("score"),
            "notes": notes,
            "metrics": row.get("metrics", {}),
            "artifacts": _artifact_feedback(row.get("artifacts", {}))
            if isinstance(row.get("artifacts"), Mapping)
            else {},
            "error": row.get("error_message"),
        }

    recent = [compact(row) for row in rows[-spec.feedback_recent :]]
    scored = [row for row in rows if isinstance(row.get("score"), (int, float))]
    best = sorted(scored, key=lambda row: float(row["score"]), reverse=True)[: spec.feedback_top]
    payload = {
        "semantics": (
            "Higher combined_score is better. Correctness gates dominate proxy improvement. "
            "Use metrics and artifacts as black-box observations, not hidden ground truth."
        ),
        "recent_attempts": recent,
        "top_attempts": [compact(row) for row in best],
    }
    text = "# Black-box evaluation feedback\n\n```json\n{}\n```".format(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    )
    if len(text) > spec.max_feedback_chars:
        text = text[: spec.max_feedback_chars] + "\n... feedback truncated by task policy"
    return text


class ContextualLLM:
    """Adds OpenEvolve-style metrics, history and artifacts to every model call."""

    # The external EvoStage fork expects an indented function body, whereas our
    # public API asks providers for a complete verifier-valid program. Convert
    # explicitly instead of relying on its one-line-signature string trimmer.
    do_auto_trim = False

    def __init__(
        self,
        provider: Any,
        harness: Any,
        trace_path: Path,
        problem_description: str = "",
    ):
        self._provider = provider
        self._harness = harness
        self._trace_path = trace_path
        self._problem_description = problem_description
        self._system_message = composed_system_message(harness.spec)
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counter = 0
        self._response_hashes: set[str] = set()
        self.debug_mode = False

    def _messages(self, prompt: Any, sample_order: int) -> list[Dict[str, str]]:
        feedback = feedback_context(self._harness.records, self._harness.spec)
        themes = _diversity_themes(self._harness.spec)
        theme = themes[(sample_order - 1) % len(themes)]
        diversity = """# Candidate diversity requirement

This is candidate request {order}. Produce a materially new program, not a
verbatim copy of the initial program or an earlier candidate. Use this request's
exploration theme: {theme}. The candidate algorithm must remain deterministic;
the request number is only a search-diversity label and must not appear in the
runtime algorithm. Keep one coherent hypothesis and return the complete program
required by the benchmark interface.
""".format(order=sample_order, theme=theme)
        benchmark = (
            "# Authoritative benchmark interface\n\n" + self._problem_description + "\n\n"
            if self._problem_description
            else ""
        )
        if isinstance(prompt, str):
            return [
                {"role": "system", "content": self._system_message},
                {
                    "role": "user",
                    "content": benchmark + diversity + "\n# Search-method request\n\n" + prompt.rstrip() + "\n\n" + feedback,
                },
            ]
        if not isinstance(prompt, list):
            raise TypeError("evolution prompt must be text or a message list")
        rows = [{"role": "system", "content": self._system_message}]
        for row in prompt:
            if not isinstance(row, Mapping):
                raise TypeError("evolution message must be an object")
            rows.append(
                {
                    "role": str(row.get("role", "user")),
                    "content": str(row.get("content", row.get("message", ""))),
                }
            )
        rows.append({"role": "user", "content": benchmark + diversity + "\n" + feedback})
        return rows

    def _transport_response(self, response: str) -> str:
        """Translate a complete public program into EvoStage's body protocol."""

        try:
            source = normalized_program(response)
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return response
        if len(tree.body) != 1:
            if self._harness.spec.program_mode == "task_adapter" and tree.body:
                return "    return {!r}\n".format(source)
            return response
        node = tree.body[0]
        if isinstance(node, ast.FunctionDef) and node.name == self._harness.spec.program_entrypoint:
            lines = source.splitlines()
            if not node.body:
                return response
            start = node.body[0].lineno - 1
            end = node.end_lineno or len(lines)
            return "\n".join(lines[start:end]) + "\n"
        if isinstance(node, ast.ClassDef) and node.name == self._harness.spec.program_entrypoint:
            return "    return {!r}\n".format(source)
        return response

    def _preflight_error(self, response: str) -> Optional[str]:
        """Validate cheap program/interface errors before spending a sample.

        Full physical evaluation is intentionally left to the harness.  This
        preflight only invokes the trusted parser/materializer, which is local,
        deterministic, and side-effect free.  A repair therefore costs an LLM
        call rather than one scarce evolutionary evaluation slot.
        """

        options = getattr(self._harness, "options", None)
        open3dbench_root = getattr(options, "open3dbench_root", None)
        if options is None:
            # Lightweight prompt-only test doubles do not expose an evaluator.
            return None
        try:
            source = normalized_program(response)
            materialize_program(
                self._harness.spec,
                source,
                open3dbench_root,
            )
        except Exception as error:
            return "{}: {}".format(type(error).__name__, error)
        return None

    def draw_sample(self, prompt: Any, *args: Any, **kwargs: Any) -> str:
        with self._lock:
            sample_order = self._counter + 1
        messages = self._messages(prompt, sample_order)
        attempts = []
        response = ""
        response_hash = ""
        for attempt in range(3):
            response = self._provider.draw_sample(messages, *args, **kwargs)
            try:
                response_hash = hashlib.sha256(
                    normalized_program(response).encode("utf-8")
                ).hexdigest()
            except ValueError:
                response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
            duplicate = response_hash in self._response_hashes
            preflight_error = self._preflight_error(response)
            attempts.append({
                "attempt": attempt + 1,
                "response_sha256": response_hash,
                "duplicate": duplicate,
                "preflight_error": preflight_error,
            })
            if not duplicate and preflight_error is None:
                break
            if preflight_error is not None:
                repair = (
                    "The trusted local verifier rejected the program before physical "
                    "evaluation with this error:\n\n{}\n\nRepair that exact error and "
                    "return a complete verifier-valid program. Preserve the intended "
                    "algorithm when possible. Return only the program."
                ).format(preflight_error)
            else:
                repair = (
                    "The program you just returned exactly duplicates an earlier candidate. "
                    "Do not repeat it. Return a materially different complete program with a "
                    "different optimization mechanism, while preserving every interface and "
                    "safety constraint."
                )
            messages = messages + [{"role": "user", "content": repair}]
        transported = self._transport_response(response)
        with self._lock:
            self._counter += 1
            self._response_hashes.add(response_hash)
            record = {
                "order": self._counter,
                "messages": messages,
                "response": response,
                "transport_response": transported,
                "provider_attempts": attempts,
            }
            with self._trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        return transported

    def draw_samples(self, prompts: list[Any], *args: Any, **kwargs: Any) -> list[str]:
        return [self.draw_sample(prompt, *args, **kwargs) for prompt in prompts]

    def close(self) -> None:
        close = getattr(self._provider, "close", None)
        if callable(close):
            close()


def export_openevolve_bundle(
    spec: TaskSpec,
    initial_candidate: Mapping[str, Any],
    open3dbench_root: Path,
    output_dir: Path,
) -> Dict[str, str]:
    """Export a native OpenEvolve problem bundle without vendoring EDA sources."""

    import yaml

    from .model import require_executable_task
    require_executable_task(spec)
    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("OpenEvolve export directory is not empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    description = build_problem_description(spec, initial_candidate, open3dbench_root)
    initial_program = output / "initial_program.py"
    initial_program.write_text(
        initial_program_text(spec, initial_candidate), encoding="utf-8"
    )

    source_config = spec.root / "config.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("OpenEvolve config must contain an object: {}".format(source_config))
    if spec.program_mode == "task_adapter":
        # The adapter accepts a complete bounded program, not SEARCH/REPLACE
        # output. Preserve this protocol even for historical config templates.
        config["diff_based_evolution"] = False
    prompt = config.setdefault("prompt", {})
    prompt["system_message"] = composed_system_message(spec) + "\n\n" + description
    prompt["include_artifacts"] = True
    prompt.setdefault("max_artifact_bytes", 32768)
    prompt.setdefault("artifact_security_filter", True)
    if spec.task_id in TASK_IDS:
        templates = output / "prompt_templates"
        templates.mkdir()
        for name in TEMPLATES:
            (templates / name).write_text(
                (template_source(spec) / name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        prompt["template_dir"] = str(templates)
    evaluator = config.setdefault("evaluator", {})
    evaluator["enable_artifacts"] = True
    evaluator["cascade_evaluation"] = False
    resolved_config = output / "config.yaml"
    resolved_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    evaluator_path = output / "evaluator.py"
    evaluator_path.write_text(
        "from local_bench.openevolve_adapter import evaluate_from_task_dir\n\n"
        "TASK_DIR = {!r}\n\n"
        "BASE_CONFIG = {!r}\n\n"
        "def evaluate(program_path):\n"
        "    return evaluate_from_task_dir(TASK_DIR, program_path, BASE_CONFIG)\n".format(
            str(spec.root), str(spec.base_config)
        ),
        encoding="utf-8",
    )
    instructions = output / "instructions.md"
    instructions.write_text(description + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "task": spec.task_id,
        "task_dir": str(spec.root),
        "open3dbench_root": str(open3dbench_root.expanduser().resolve()),
        "initial_program": str(initial_program),
        "evaluator": str(evaluator_path),
        "config": str(resolved_config),
        "source_repository_vendored": False,
    }
    manifest_path = output / "bundle.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {key: str(value) for key, value in manifest.items() if isinstance(value, str)}
