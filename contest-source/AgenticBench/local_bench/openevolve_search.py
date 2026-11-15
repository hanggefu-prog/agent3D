"""Native OpenEvolve components behind the framework-neutral two-method API.

No benchmark file access, evaluator callbacks, controller or subprocesses. This
is a serial full-rewrite adapter, not the upstream multiprocessing controller.
"""
import asyncio
import copy
import hashlib
import json
import math

from .search_api import CandidateProposal
from .prompt_context import distinct_reference, feedback_text
from .tp_gnn_openevolve import _reason


_LIGHTWEIGHT_TEMPLATE = """# Selected parent
This is the program to improve, not necessarily the latest or best trial.
{parent_feedback}

```{language}
{current_program}
```

{reference_example}

# Next candidate
Try one main change at a time. Keep useful parts of the parent; combine ideas
when comparable measured results support them. Maximize combined_score only for
valid candidates. Other measurements are diagnostics, not extra rewards.
Missing metrics remain unknown, never zero. Do not return an evaluated program unchanged.
Return the complete program with the required entry points and any required EVOLVE-BLOCK
markers, not a diff. Keep the declared interfaces and constraints. Include a
short comment explaining the change; no tests or surrounding prose.
"""


_TEMPLATES = {
    "full_rewrite_user": """Selected parent (not necessarily the latest or best candidate):
```{language}
{current_program}
```
Measured parent metrics (missing measurements are unknown):
{metrics}
{artifacts}
Selected examples, not a chronological history:
{evolution_history}
Maximize the benchmark score only for valid candidates. Other measurements are
diagnostics: follow the task's declared directions, not 'all increases are good'.
Return the complete program required by the system task contract, not a diff.
""",
    "previous_attempt": "Selected example {attempt_number}: {changes}\nMetrics: {performance}",
    "top_program": "Selected program {program_number}, score {score}:\n```{language}\n{program_snippet}\n```",
    "inspiration_program": "Alternative program {program_number}, score {score}:\n```{language}\n{program_snippet}\n```",
}


class OpenEvolveAskTell:
    name = "openevolve-ask-tell"

    def __init__(self, config=None, *, generate=None):
        """config contains search/provider settings, not benchmark settings.

        generate(system, user) is an optional synchronous provider injection for
        offline testing or custom model transports. By default use LLMEnsemble.
        """
        from openevolve.config import Config
        self.config = copy.deepcopy(config) if config is not None else Config(diff_based_evolution=False)
        if self.config.diff_based_evolution:
            raise ValueError("ask/tell adapter currently supports complete-program full rewrite only")
        if self.config.database.db_path or self.config.database.artifacts_base_path:
            raise ValueError("ask/tell adapter requires a fresh in-memory database; resume is not supported")
        if self.config.prompt.template_dir:
            raise ValueError("Pass task templates in SearchProblem, not a template directory")
        if self.config.prompt.programs_as_changes_description:
            raise ValueError("Changes-description mode is not supported by the complete-program contract")
        if self.config.database.embedding_model or self.config.database.novelty_llm:
            raise ValueError("External novelty services are not supported by this adapter")
        if self.config.database.num_islands < 1 or not self.config.database.feature_dimensions:
            raise ValueError("OpenEvolve requires at least one island and feature dimension")
        self._generate_override = generate
        self._problem = None
        self._pending = None
        self._completed = 0
        self._ensemble = None
        self._loop = None
        self._closed = False
        self._last_feedback = None
        self._last_program = None
        self._artifacts = {}
        self.database = None
        self.sampler = None

    def _initialize(self, context):
        from openevolve.database import ProgramDatabase
        from openevolve.prompt.sampler import PromptSampler
        problem = context.problem
        if problem.response_format != "complete_program":
            raise ValueError("Unsupported candidate response format")
        if not problem.maximize:
            raise ValueError("OpenEvolve adapter requires a benchmark-provided maximization score")
        if context.evaluations or context.history:
            raise ValueError("Cannot attach a fresh backend to existing history; explicit resume is required")
        if not problem.initial_program.strip():
            raise ValueError("An initial program is required in the first ask context")
        self._problem = copy.deepcopy(problem)
        prompt = copy.deepcopy(self.config.prompt)
        prompt.system_message = problem.system_message + "\n\n" + problem.description
        prompt.include_artifacts = True
        self.sampler = PromptSampler(prompt)
        for name, text in {**_TEMPLATES, **problem.prompt_templates}.items():
            self.sampler.template_manager.add_template(name, text)
        # The ask/tell path has one presentation policy across local tasks.
        # Legacy exported bundles keep their explicit upstream templates.
        self.sampler.template_manager.add_template("lightweight_user", _LIGHTWEIGHT_TEMPLATE)
        self.database = ProgramDatabase(copy.deepcopy(self.config.database))

    def _generate(self, system, user):
        if self._generate_override is not None:
            return self._generate_override(system, user)
        from openevolve.llm.ensemble import LLMEnsemble
        if self._ensemble is None:
            self._ensemble = LLMEnsemble(self.config.llm.models)
            self._loop = asyncio.new_event_loop()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._loop.run_until_complete(self._ensemble.generate_with_context(
                system_message=system, messages=[{"role": "user", "content": user}]))
        raise RuntimeError("Synchronous ask must run outside an active asyncio loop")

    def ask(self, context):
        from openevolve.database import Program
        from openevolve.utils.code_utils import parse_full_rewrite
        if self._closed:
            raise RuntimeError("Search backend is closed")
        if self._pending is not None:
            raise RuntimeError("tell is required before another ask")
        if self._problem is None:
            self._initialize(context)
        if context.problem != self._problem:
            raise ValueError("Task contract changed during a search")
        if context.evaluations != self._completed or len(context.history) != self._completed:
            raise ValueError("ask context does not match acknowledged feedback")
        if context.remaining_budget <= 0:
            return ()
        island = self._completed % len(self.database.islands)
        if self.database.programs:
            parent, inspirations = self.database.sample_from_island(
                island, self.config.prompt.num_top_programs + self.config.prompt.num_diverse_programs)
            pool = [self.database.programs[pid] for pid in self.database.islands[island]
                    if pid in self.database.programs]
            pool.sort(key=lambda p: p.metrics["combined_score"], reverse=True)
        else:
            # No unmeasured reference score, no hidden initial EDA evaluation.
            parent = Program(id="initial", code=self._problem.initial_program, metrics={})
            inspirations, pool = [], []
        reference = (distinct_reference(parent, pool + inspirations)
                     if self.config.prompt.num_top_programs + self.config.prompt.num_diverse_programs > 0
                     else None)
        example = ""
        if reference is not None:
            example = ("# One different evaluated example (not a chronological history)\n"
                       "combined_score: {} (higher is better; not necessarily better than the parent).\n"
                       "```{}\n{}\n```".format(reference.metrics["combined_score"],
                                                self._problem.language, reference.code))
        # The population database stores scalar fitness/features only. Keep the
        # complete trusted matrix feedback in the prompt via ask/tell history.
        feedback_id = (parent.metadata or {}).get("feedback_id", parent.id)
        parent_metrics = next((item.metrics for item in context.history
                               if item.proposal_id == feedback_id), parent.metrics)
        prompt = self.sampler.build_prompt(
            current_program=parent.code, parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[], top_programs=[], inspirations=[], language=self._problem.language,
            diff_based_evolution=False, evolution_round=self._completed,
            program_artifacts={}, template_key="lightweight_user",
            parent_feedback=feedback_text(parent_metrics, self._problem.task_id),
            reference_example=example,
            feature_dimensions=self.config.database.feature_dimensions,
        )
        prompt["user"] += ("\n\nEvaluation budget remaining: " + str(context.remaining_budget)
                           + ". Return a complete program; no evaluator calls or tool access.")
        if not self.database.programs:
            prompt["user"] += "\nThe initial program has no measured score in this session; do not assume zero."
        if self._last_feedback is not None and not self._last_feedback.valid:
            # Failures are useful feedback but never valid population members.
            failed = self._last_feedback
            failure_text = json.dumps({
                "proposal_id": failed.proposal_id, "status": failed.status, "score": None,
                "error_type": (_reason(failed.error_type) or "unknown")[:120],
                "error_message": (_reason(failed.error_message) or "Reason unavailable.")[:600],
            }, ensure_ascii=False)
            limit = self.config.prompt.max_artifact_bytes
            encoded = failure_text.encode("utf-8")
            if len(encoded) > limit:
                failure_text = encoded[:limit].decode("utf-8", errors="ignore") + "\n[Truncated display; full receipt retained by harness]"
            prompt["user"] += "\n\nLast rejected/failed candidate (not a scored parent):\n" + failure_text
        if self._problem.max_program_chars is not None:
            prompt["user"] += "\nMaximum complete program length: {} characters.".format(self._problem.max_program_chars)
        response = self._generate(prompt["system"], prompt["user"])
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Provider returned no candidate; no implicit retry was scheduled")
        program = parse_full_rewrite(response, language=self._problem.language)
        proposal = CandidateProposal("oe-{:06d}".format(self._completed + 1), program,
                                     {"parent_id": parent.id, "generation": parent.generation + 1})
        self._pending = (proposal, parent, island, prompt)
        return (proposal,)

    def tell(self, feedback):
        from openevolve.database import Program
        batch = tuple(feedback)
        if self._pending is None or len(batch) != 1:
            raise ValueError("tell requires exactly one pending candidate result")
        proposal, parent, island, prompt = self._pending
        item = batch[0]
        if (item.proposal_id != proposal.proposal_id or item.program_sha256 !=
                hashlib.sha256(proposal.program.encode("utf-8")).hexdigest()):
            raise ValueError("Feedback identity does not match pending candidate")
        if item.status not in {"accepted", "cached", "failed", "rejected"}:
            raise ValueError("Unknown feedback status")
        if item.status in {"accepted", "cached"}:
            if not item.valid:
                raise ValueError("Accepted feedback requires a finite score")
            measured_score = item.metrics.get("combined_score")
            if measured_score is not None and measured_score != item.score:
                raise ValueError("score and combined_score disagree")
            if self._problem.max_program_chars and len(proposal.program) > self._problem.max_program_chars:
                raise ValueError("Harness accepted a program exceeding the declared limit")
            metrics = {key: value for key, value in item.metrics.items()
                       if isinstance(value, (int, float)) and math.isfinite(value)}
            metrics["combined_score"] = item.score
            missing = set(self.config.database.feature_dimensions) - {"complexity", "diversity", "score"} - set(metrics)
            if missing:
                raise ValueError("Unknown MAP-Elites feature measurements: " + ", ".join(sorted(missing)))
            program = Program(id=proposal.proposal_id, code=proposal.program,
                parent_id=parent.id if parent.id in self.database.programs else None,
                generation=parent.generation + 1, language=self._problem.language,
                metrics=metrics, prompts=prompt,
                metadata={"parent_metrics": dict(parent.metrics), "feedback_status": item.status,
                          "feedback_id": proposal.proposal_id})
            # Do not spill artifacts into native database files. All feedback is
            # already archived by the harness; prompt context stays in memory.
            self._artifacts[program.id] = {
                key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                for key, value in item.artifacts.items()
            }
            self.database.add(program, iteration=self._completed + 1, target_island=island)
        elif item.score is not None:
            raise ValueError("Rejected/failed candidate must not have a score")
        self.database.increment_island_generation(island)
        if self.database.should_migrate():
            self.database.migrate_programs()
        self._last_feedback = copy.deepcopy(item)
        self._last_program = proposal.program
        self._completed += 1
        self._pending = None

    def close(self):
        """Optional local resource cleanup; not part of the benchmark protocol."""
        if self._ensemble is not None:
            for model in self._ensemble.models:
                client = getattr(model, "client", None)
                if client is not None and callable(getattr(client, "close", None)):
                    client.close()
        if self._loop is not None:
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()
            self._loop = None
        self._closed = True
