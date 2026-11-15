from __future__ import annotations

import contextlib
import ast
import copy
import hashlib
import importlib
import json
import os
import sys
import time
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .evolve_protocol import evolution_prompt
from .model import ROUTING_SOURCE_TASK_KINDS, load_candidate
from .program_bridge import initial_program_text
from .prompting import ContextualLLM
from .search_api import ProgramEvaluationHarness


SUPPORTED_METHODS = ("randsample", "hillclimb", "eoh", "evostage")
FINGERPRINT_FILES = (
    "llm4ad/base/evaluate.py",
    "llm4ad/method/eoh/eoh.py",
    "llm4ad/method/evostage/EvoStage.py",
)


def evostage_fingerprints(root: Path) -> Dict[str, str]:
    resolved = root.expanduser().resolve()
    return {
        relative: hashlib.sha256((resolved / relative).read_bytes()).hexdigest()
        for relative in FINGERPRINT_FILES
    }


@dataclass(frozen=True)
class EvolutionOptions:
    task_dir: Path
    evostage_root: Path
    output_dir: Path
    evaluation_mode: str
    open3dbench_root: Optional[Path] = None
    baseline_run: Optional[Path] = None
    reference_run: Optional[Path] = None
    base_config: Optional[Path] = None
    benchmark_id: Optional[str] = None
    compact_evaluations: bool = False
    timeout_seconds: int = 21600
    evaluation_matrix: Optional[Path] = None
    matrix_workers: int = 1


EvolutionHarness = ProgramEvaluationHarness


class LiteralMultiProgramEvaluator:
    """Safe replacement for EvoStage's exec-based MultiFuncEvaluator."""

    def __init__(self, evaluation):
        self._evaluation = evaluation

    def _complete_program(self, body: str) -> str:
        template = self._evaluation.template_program
        tree = ast.parse(template)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1 or not functions[0].body:
            raise ValueError("EvoStage transport template must contain exactly one function")
        function = functions[0]
        lines = template.splitlines()
        start = function.body[0].lineno - 1
        end = function.end_lineno or len(lines)
        candidate_body = str(body).strip("\n")
        if not candidate_body.strip():
            raise ValueError("EvoStage returned an empty candidate body")
        if not candidate_body.startswith((" ", "\t")):
            candidate_body = "\n".join(
                "    " + line if line else line for line in candidate_body.splitlines()
            )
        return "\n".join(lines[:start] + candidate_body.splitlines() + lines[end:]) + "\n"

    def evaluate_program_record_time(self, programs, stage_num=None):
        started = time.monotonic()
        if len(programs) != 1:
            return None, "exactly one candidate program is required", time.monotonic() - started
        complete = self._complete_program(str(programs[0]))
        score = self._evaluation.evaluate_program(complete, None, stage_num=stage_num)
        state = "candidate accepted" if score is not None else "candidate rejected by verifier or evaluator"
        return score, state, time.monotonic() - started

    def get_init_state_info(self):
        return "Generate one verifier-valid program within the declared task scope."


class BridgeProfiler:
    """Minimal profiler that never introspects the LLM object or its credentials."""

    def __init__(self, log_dir=None, **kwargs):
        self._log_dir = Path(log_dir).resolve() if log_dir else None
        self._samples = 0
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    def record_parameters(self, llm, evaluation, method):
        return None

    def register_function(self, function, **kwargs):
        self._samples += 1
        if self._log_dir:
            with (self._log_dir / "samples.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "sample_order": self._samples,
                    "score": function.score,
                    "function": str(function),
                }, sort_keys=True) + "\n")

    def register_population(self, population):
        return None

    def finish(self):
        return None


class _DisabledInternalOpenAI:
    """Neutralizes EvoStage's embedded model/key; the adapter injects the selected LLM."""

    def __init__(self, *args, **kwargs):
        return None

    def draw_sample(self, prompt):
        raise RuntimeError("EvoStage internal reflection LLM was not replaced")


def _load_external(root: Path):
    root = root.expanduser().resolve()
    required = [
        root / "llm4ad" / "base" / "evaluate.py",
        root / "llm4ad" / "method" / "eoh" / "eoh.py",
        root / "llm4ad" / "method" / "evostage" / "EvoStage.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("EvoStage checkout is incomplete: {}".format(", ".join(missing)))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        __import__("scipy")
    except ImportError:
        # llm4ad imports its unused FunSearch module at package import time.
        # The supported scalar baselines never access scipy; keep that unrelated
        # optional dependency out of this adapter environment.
        sys.modules.setdefault("scipy", types.ModuleType("scipy"))
    from llm4ad.base import Evaluation, Function

    probe = Function(name="probe", args="", body="    return 1")
    if not str(probe).lstrip().startswith("def probe"):
        def _function_string(function):
            args = str(function.args or "").strip()
            return_type = " -> {}".format(str(function.return_type).strip()) if function.return_type else ""
            rows = ["def {}({}){}:".format(function.name, args, return_type)]
            if function.docstring:
                doc = str(function.docstring).strip()
                rows.append('    """{}"""'.format(doc))
            if function.body:
                rows.append(function.body)
            return "\n".join(rows) + "\n\n"

        Function.__str__ = _function_string
    from llm4ad.method.eoh import EoH, EoHProfiler
    from llm4ad.method.hillclimb import HillClimb, HillClimbProfiler
    from llm4ad.method.randsample import RandSample, RandSampleProfiler

    stub_name = "llm4ad.tools.llm.llm_api_openai"
    stub = types.ModuleType(stub_name)
    stub.OpenAIAPI = _DisabledInternalOpenAI
    sys.modules[stub_name] = stub
    from llm4ad.method.evostage import EvoStage
    from llm4ad.method.evostage.population import Population

    # Upstream treats equal scores as duplicate programs. Placement metrics can
    # tie for distinct implementations, so program text is the only safe
    # duplicate identity for a benchmark population.
    def _same_program_only(population, func):
        return any(
            str(existing) == str(func)
            for existing in population._population + population._next_gen_pop
        )

    Population.has_duplicate_function = _same_program_only
    evostage_module = importlib.import_module("llm4ad.method.evostage.EvoStage")
    evostage_module.EvoStageProfiler = BridgeProfiler

    return {
        "Evaluation": Evaluation,
        "EoH": EoH,
        "EoHProfiler": EoHProfiler,
        "HillClimb": HillClimb,
        "HillClimbProfiler": HillClimbProfiler,
        "RandSample": RandSample,
        "RandSampleProfiler": RandSampleProfiler,
        "EvoStage": EvoStage,
    }


def _seed_frozen_reference(method: Any, harness: EvolutionHarness) -> bool:
    """Put a behavior-preserving routing policy into EvoStage's population.

    Every routing source task defines an exact no-change initial function, so
    its frozen normalized reference score is 1.0. The seed is not counted as a
    generated sample and is never written as a fake evaluation record. It
    merely prevents a small population from being initialized entirely by
    regressions.
    """

    if (
        harness.spec.kind not in ROUTING_SOURCE_TASK_KINDS
        or harness.options.evaluation_mode != "full"
        or (harness.options.reference_run is None and harness.matrix is None)
    ):
        return False
    functions = getattr(method, "_function_to_evolve_list", ())
    populations = getattr(method, "_population_list", ())
    if len(functions) != len(populations) or not functions:
        raise RuntimeError("EvoStage reference seeding interface is unavailable")
    for function, population in zip(functions, populations):
        seed = copy.deepcopy(function)
        seed.score = 1.0
        seed.algorithm = "Frozen maintained OpenROAD routing policy"
        seed.evaluate_time = 0.0
        seed.sample_time = 0.0
        population.register_function(seed)
    return True


def _make_evaluation(base, harness: EvolutionHarness, description: Optional[str] = None):
    initial = load_candidate(harness.spec.initial_candidate)
    description = description or evolution_prompt(
        harness.spec, initial, harness.options.open3dbench_root
    )
    public_template = initial_program_text(harness.spec, initial)
    public_tree = ast.parse(public_template)
    source_transport = not (
        len(public_tree.body) == 1 and isinstance(public_tree.body[0], ast.FunctionDef)
    )
    template = (
        "def evolve_component_program() -> str:\n    return {!r}\n".format(public_template)
        if source_transport
        else public_template
    )

    def public_program(program_str: str) -> str:
        if not source_transport:
            return program_str
        tree = ast.parse(program_str)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("EvoStage source transport must contain one function")
        function = tree.body[0]
        body = list(function.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            raise ValueError("EvoStage source transport must return one static source string")
        value = ast.literal_eval(body[0].value)
        if not isinstance(value, str):
            raise ValueError("EvoStage source transport return must be source text")
        return value

    class CandidateEvaluation(base):
        def __init__(self):
            super().__init__(
                template_program=template,
                task_description=description,
                exec_code=False,
                safe_evaluate=False,
                timeout_seconds=None,
            )
            self.template_program_list = [template]
            self.task_description_list = [description]
            self.template_program_list_multistage = [template]
            self.task_description_list_multistage = [description]
            self.init_stage_info = "Initial candidate is verifier-valid."

        def evaluate_program(self, program_str, callable_func, **kwargs):
            try:
                return harness.evaluate_program(public_program(program_str))
            except Exception as error:
                harness.record_invalid(program_str, error)
                return None

        def multi_stage_evaluate(self, callable_func_list, **kwargs):
            raise RuntimeError("exec-based EvoStage evaluation is disabled")

    return CandidateEvaluation()


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_evolution(
    method_name: str,
    llm,
    options: EvolutionOptions,
    *,
    max_samples: int = 50,
    max_generations: int = 50,
    pop_size: int = 5,
    evaluation_transport=None,
) -> Dict[str, Any]:
    if method_name not in SUPPORTED_METHODS:
        raise ValueError("unsupported EvoStage method: {}".format(method_name))
    if max_samples < 1 or pop_size < 1 or max_generations < 1:
        raise ValueError("sample, population, and generation limits must be positive")
    external = _load_external(options.evostage_root)
    harness = EvolutionHarness(options, evaluation_transport=evaluation_transport)
    if (
        options.evaluation_mode == "full"
        and harness.spec.kind in ROUTING_SOURCE_TASK_KINDS
        and options.reference_run is None
        and harness.matrix is None
    ):
        raise ValueError(
            "full routing source evolution requires a frozen --reference-run"
        )
    prompt_spec = replace(
        harness.spec,
        base_config=(options.base_config.expanduser().resolve()
                     if options.base_config is not None else harness.spec.base_config),
        benchmark_input_id=options.benchmark_id or (
            None if options.base_config is not None else harness.spec.benchmark_input_id
        ),
    )
    problem_description = evolution_prompt(
        prompt_spec,
        load_candidate(harness.spec.initial_candidate),
        options.open3dbench_root,
    )
    if options.base_config is not None:
        selected_config = options.base_config.expanduser().resolve()
        selected = yaml.safe_load(selected_config.read_text(encoding="utf-8"))
        if not isinstance(selected, dict):
            raise ValueError("selected benchmark base_config must contain a mapping")
        case_fields = {
            key: selected.get(key)
            for key in ("design", "type", "pdk", "threads")
            if selected.get(key) is not None
        }
        problem_description += (
            "\n\n# Selected frozen evaluation case\n\n"
            "This run evaluates the candidate only on the following frozen case. "
            "The algorithm may use die-specific inputs, but it must remain a general "
            "policy rather than hard-code the design name.\n\n```yaml\n"
            + yaml.safe_dump(
                {
                    "benchmark_id": options.benchmark_id,
                    "base_config": str(selected_config),
                    **case_fields,
                },
                sort_keys=False,
            )
            + "```"
        )
    if harness.matrix is not None:
        problem_description += harness.matrix.prompt()
    contextual_llm = ContextualLLM(
        llm,
        harness,
        options.output_dir / "prompt_trace.jsonl",
        problem_description,
    )
    evaluation = _make_evaluation(
        external["Evaluation"], harness, problem_description
    )
    logs = options.output_dir / "search_logs"
    logs.mkdir(parents=True, exist_ok=True)
    common = {
        "evaluation": evaluation,
        "max_sample_nums": max_samples,
        "num_samplers": 1,
        "num_evaluators": 1,
        "debug_mode": False,
    }
    reference_seeded = False
    with _working_directory(options.output_dir):
        if method_name == "randsample":
            profiler = BridgeProfiler(log_dir=str(logs))
            method = external["RandSample"](llm=contextual_llm, profiler=profiler, **common)
        elif method_name == "hillclimb":
            profiler = BridgeProfiler(log_dir=str(logs))
            method = external["HillClimb"](llm=contextual_llm, profiler=profiler, **common)
        elif method_name == "eoh":
            profiler = BridgeProfiler(log_dir=str(logs))
            method = external["EoH"](
                llm=contextual_llm,
                profiler=profiler,
                max_generations=max_generations,
                pop_size=pop_size,
                selection_num=min(2, pop_size),
                **common,
            )
        else:
            method = external["EvoStage"](
                llm_list=[contextual_llm],
                max_generations=max_generations,
                pop_size=pop_size,
                selection_num=min(2, pop_size),
                evaluation_name=harness.spec.task_id,
                model_name="external-provider",
                use_multistage_operator=False,
                open_thought_code=False,
                **common,
            )
            # Upstream EvoStage otherwise gives initialization only 2 * pop_size
            # attempts.  Source tasks can legitimately reject several early LLM
            # samples, so let initialization consume the declared evaluation
            # budget instead of terminating the experiment prematurely.
            method._initial_sample_nums_max = max_samples
            method._evaluator = LiteralMultiProgramEvaluator(evaluation)
            method._prompter.llm = contextual_llm
            reference_seeded = _seed_frozen_reference(method, harness)
        method.run()
    summary = harness.summary()
    summary.update(
        {
            "method": method_name,
            "max_samples": max_samples,
            "max_generations": max_generations,
            "population_size": pop_size,
            "reference_seeded": reference_seeded,
            "reference_score": 1.0 if reference_seeded else None,
            "improved_over_reference": bool(
                reference_seeded
                and summary.get("best")
                and float(summary["best"]["score"]) > 1.0
            ),
            "evostage_root": str(options.evostage_root.expanduser().resolve()),
            "evostage_fingerprints": evostage_fingerprints(options.evostage_root),
        }
    )
    (options.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
