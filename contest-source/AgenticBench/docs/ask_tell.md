# Unified ask/tell search API

`local_bench.search_api` separates candidate search from Open3DBench
evaluation. A search framework owns population state and implements two
required operations:

```python
class SearchBackend:
    def ask(self, context: SearchContext) -> Sequence[CandidateProposal]: ...
    def tell(self, feedback: Sequence[CandidateFeedback]) -> None: ...
```

An empty `ask` batch ends the search. The old `done()` and `name` attributes
remain optional compatibility conveniences, not additional requirements.
`ask` may return a batch. Every proposal has a run-unique `proposal_id`, one
complete candidate program, and optional search metadata. `tell` receives the
corresponding score, task-selected metrics, diagnostic artifacts, failure state,
runtime, and preserved proposal metadata. A rejected or failed candidate has
`score=None`; search backends must not treat it as a valid zero-valued result.

The runner owns only lifecycle mechanics:

```text
SearchBackend.ask(context)
  -> ProposalEvaluator.evaluate_proposal(candidate)
  -> SearchBackend.tell(feedback)
  -> repeat until done or the evaluation budget is exhausted
```

`SearchContext` contains the immutable problem, consumed and remaining budget,
complete feedback history, and current incumbent. Its `problem` also carries the
system message, full task description/constraints, initial program, response
format, language, program-size bound and optional inline prompt templates.
These are values, not paths the backend must open. Other engines may ignore
template hints and format the same task/feedback data themselves.
Higher score is better unless
`SearchProblem.maximize` is false.

## Native OpenEvolve through only ask/tell

The recommended host-side entry is `run_task_search`. The benchmark host prepares
the task and controls source/reference paths, physical transport and budgets;
none of those setup paths are constructor arguments to the search backend.
Provider authentication and search configuration remain the engine's own setup,
not a second benchmark information channel.

```python
from pathlib import Path
from openevolve.config import Config
from local_bench import OpenEvolveAskTell, TaskEvaluationOptions, run_task_search

search = Config(diff_based_evolution=False)
# Configure search.llm.models with the intended provider/model outside prompts.
backend = OpenEvolveAskTell(search)
try:
    result = run_task_search(
        backend,
        TaskEvaluationOptions(
            task_dir=Path("/path/to/task"),
            output_dir=Path("/path/to/new-run"),
            open3dbench_root=Path("/path/to/Open3DBench-V2.0"),
            baseline_run=Path("/path/to/verified-input"),
            reference_run=Path("/path/to/same-contract-reference"),
            base_config=Path("/path/to/selected-case.yaml"),
        ),
        max_evaluations=2,
    )
finally:
    backend.close()  # Optional local provider-resource cleanup, not a data query.
```

The backend uses real OpenEvolve `PromptSampler`, `ProgramDatabase` (including
island sampling/MAP-Elites/migration), full-rewrite parsing and `LLMEnsemble`.
The first proposal is generated from the supplied initial program without
inventing an initial score or launching a hidden baseline evaluation. Every
proposal, including a rejected/failed/cache-hit proposal, consumes one declared
evaluation slot. The host's `max_evaluations` is the budget authority; upstream
controller/evaluator settings and `max_iterations` do not schedule extra work.
No `evaluator.py`, native evaluator import, task-directory read, model-side
filesystem lookup or benchmark subprocess occurs in the backend.

Only scored `accepted`/`cached` feedback enters its population. A failed or
rejected proposal keeps `score=None`, is excluded from population selection,
and its identifier and short reason are shown in the next prompt. The complete
failed program stays in the proposal log, not another prompt code block. This is different from a
valid evaluation returning the task-defined zero score, which may indicate
failed feasibility gates; gate metrics remain authoritative. Feedback IDs and
program hashes must match the pending proposal. Missing/nonfinite scores are
never replaced with a valid zero reward. Invalid generation/protocol/provider
calls raise errors without an automatic new candidate or retry loop.

For the eight maintained local tasks (TP-GNN timing-context-v2, three placement
tasks, HBT assignment and three routing-source tasks), the host applies task-specific
optimization-feedback allowlists for every ask/tell backend and native
OpenEvolve exports. Raw receipts are kept in `history.jsonl`,
while `search_history.jsonl` and `search_proposals.jsonl` preserve pre-projection
evaluation feedback and complete proposals for audit. Only the projected view
is passed to the search backend. The model-facing projection excludes raw
manifests, patches, container details and log tails. Unrelated tasks keep
their existing feedback policy. TP-GNN optional diagnostics come only from
finite numeric measurements in the current report: training loss/steps, embedding
variance/cosine, raw/balanced/feasible cuts, closure cut deltas and training time.
Missing, null, malformed or nonfinite diagnostics are omitted, never zero-filled
or reconstructed from legacy fields. Actual measured zeros are retained. Required
objective and feasibility validation is unchanged. Historical receipts are not
rewritten; numeric zeros already stored in old receipts cannot be retroactively
distinguished from missing measurements.

The prompts follow physical background, the specific optimization problem,
evaluation and feedback, then interfaces and implementation limits. They use
plain English and expand specialist abbreviations at first use. Engineering
paths, manifest identities and reference history remain in the trusted task
contract, not the model description. The host still binds the selected design,
integration, platform and exact initial program. Search methods need no extra
benchmark interface to obtain this information.

Placement feedback reports final weighted half-perimeter wirelength, worst-die
and per-die density overflow, maximum density, and the score components. The
dynamic controller additionally reports current Nesterov invocation/segment
counts, identity status and multiplier summaries; native step hints are not
accepted step lengths. Initialization/preconditioning also expose the change
between the first and final complete GP log rows, not a full optimizer trace or
a direct pre-optimizer measurement. Neither task reports legalized placement or routed timing.

All three routing tasks report total/per-die global-route overflow, wirelength,
congestion bins, hook presence and measured runtime when available. Net ordering
adds order-change/overflow-subset/overlap summaries. Rip-up adds observed selected
decisions, selected ratios and output-score samples; maze cost adds multiplier
samples. Their cumulative counters are sparse-log lower bounds, not exact final
totals. Iteration control logs every invocation: feedback counts records and
sums per-record change flags, while preserving the native iteration index under
`iteration_index_last` (it can reset). Applied scales, offsets and resulting
control values have per-die min/max/last summaries. The ambiguous legacy `signal`
alias is removed. Extrema always refer to logged samples, not unobserved calls.
No detailed-routing, design-rule, antenna, timing or thermal results are invented.

Task-specific system instructions and four OpenEvolve generation/example
templates consistently distinguish selected population examples from chronological
history. They encourage one main change at a time, allow combinations supported
by comparable measurements, and require complete programs even for small edits.
They do not assume that every numeric increase is better or that the initial
program has already been evaluated. Scoring formulas, native hooks, budgets,
verifiers and reference admission are unchanged by this presentation update.

### Lightweight prompt context

The direct `OpenEvolveAskTell` adapter uses one compact generation template.
The system message carries the readable task specification, without a duplicate
initial program. The user message carries the complete selected parent, its
key measured metrics (each value and definition once), and at most one different
evaluated program with its exact fitness. The example is the best distinct
source in the current island, falling back to the sampled inspirations; source
comparison ignores trailing whitespace and outer blank lines, not indentation.
It is not a chronological history. Duplicate IDs or cache entries do not cause
the same source to be printed again. A zero configured example count disables it.

`prompt_context.py` selects the display metrics: timing and partition/training
diagnostics for TP-GNN; HPWL, density overflow and simple progress for placement;
per-die wirelength, overflow and hook activity for routing. Missing values stay
unknown. Definitions come from the existing metric dictionary. Failed candidates
add a reason limited to 600 characters, with paths and digests removed; no new
LLM log analyzer is involved. Raw logs, full proposals, task-selected tell
feedback and population metrics are unchanged. Initial programs still travel
through `SearchProblem.initial_program`, so other frameworks need only ask/tell.

Native parent sampling, database capacity, scoring and evaluation budgets are
unchanged. The `num_top_programs`/`num_diverse_programs` settings still determine
the inspiration sampling request, not the number of code blocks displayed by
this adapter. Legacy evaluator-file exports retain their old explicit templates;
they are not the maintained campaign entry.

Current limits: one in-flight proposal, synchronous calls, complete-program
full rewrite, maximization scores, and a fresh in-memory search database.
Resume, external novelty services and parallel search scheduling are rejected
or unsupported rather than silently emulated. This adapter is not claimed to
reproduce the upstream multiprocessing controller's scheduling. Local tests use
real native components with an offline model; they do not qualify CUDA/EDA or
deploy new experiments. Future transport/checkpoint work must preserve this
same two-method boundary.

## OpenAgentic3DBench evaluator

`ProgramEvaluationHarness` is the standard `ProposalEvaluator`. It performs the
same trusted program materialization, policy verification, isolated
Open3DBench execution, metric extraction, duplicate cache, `history.jsonl`, and
failure recording used by the EvoStage integration. Ask/tell runs additionally
write `search_history.jsonl`, which preserves proposal IDs and search metadata
alongside each normalized feedback record.

```python
from pathlib import Path

from local_bench import (
    AskTellRunner,
    CandidateProposal,
    ProgramEvaluationHarness,
    SearchProblem,
    TaskEvaluationOptions,
)
from local_bench.benchmark_input import resolve_benchmark_run
from local_bench.model import load_candidate, load_task
from local_bench.program_bridge import initial_program_text
from local_bench.prompting import build_problem_description


task_dir = Path("/data/gcr/agentic3d-dev/local_tasks/placement_parameters")
open3dbench = Path("/data/gcr/Open3DBench-V2.0")
benchmark_input = Path(
    "/data/gcr/openagentic3dbench-inputs/bp_fe_lol_ng45_mtkahypar"
)
spec = load_task(task_dir)
initial_candidate = load_candidate(spec.initial_candidate)
initial_program = initial_program_text(spec, initial_candidate)

problem = SearchProblem(
    task_id=spec.task_id,
    description=build_problem_description(spec, initial_candidate, open3dbench),
    initial_program=initial_program,
)
evaluator = ProgramEvaluationHarness(
    TaskEvaluationOptions(
        task_dir=task_dir,
        output_dir=Path("/data/gcr/my-search/run-001"),
        evaluation_mode="full",
        open3dbench_root=open3dbench,
        baseline_run=resolve_benchmark_run(
            benchmark_input, task_id=spec.task_id
        ),
        reference_run=Path("/data/gcr/references/bp_fe_lol_ng45"),
    )
)

# backend is any object implementing ask and tell.
result = AskTellRunner().run(
    backend,
    evaluator,
    problem,
    max_evaluations=50,
)
```

## Minimal custom agent backend

The backend below proposes one candidate per round and keeps the best valid
program. `generate` can call a single LLM, a planner/coder pair, or a complete
multi-agent workflow.

```python
class AgentBackend:
    name = "my-agent"

    def __init__(self, generate):
        self.generate = generate
        self.current_program = None
        self.pending = {}
        self.incumbent_score = None
        self.finished = False

    def ask(self, context):
        if self.finished:
            return []
        current = self.current_program or context.problem.initial_program
        program = self.generate(
            problem=context.problem.description,
            current_program=current,
            history=[item.to_dict() for item in context.history],
        )
        proposal = CandidateProposal(
            proposal_id="candidate-{:04d}".format(context.evaluations + 1),
            program=program,
            metadata={"parent": context.incumbent.proposal_id if context.incumbent else None},
        )
        self.pending[proposal.proposal_id] = proposal.program
        return [proposal]

    def tell(self, feedback):
        for result in feedback:
            program = self.pending.pop(result.proposal_id)
            if not result.valid:
                continue
            if self.incumbent_score is None or result.score > self.incumbent_score:
                self.current_program = program
                self.incumbent_score = result.score

```

For callback-oriented libraries, `CallbackSearchBackend` wraps existing ask/tell
functions without requiring a new class; a legacy done callback is optional.

## Existing framework compatibility

- EvoStage, EoH, HillClimb, and RandSample retain their upstream search loops.
  Their adapter now uses the same framework-neutral
  `ProgramEvaluationHarness`; `EvolutionHarness` remains as a compatibility
  alias.
- OpenEvolve now has a direct `OpenEvolveAskTell` backend. The old program-file
  evaluator and exported bundles remain legacy compatibility entry points;
  they are not used by `run_task_search`.
- New frameworks should prefer `SearchBackend` plus `AskTellRunner`. They do
  not need an EvoStage checkout or OpenEvolve installation.

The initial runner evaluates batches serially. This avoids oversubscribing an
EDA license/GPU and preserves deterministic history ordering. A future parallel
runner can implement the same `SearchBackend` and `ProposalEvaluator` protocols
without changing task definitions. `ask` must respect
`SearchContext.remaining_budget`; the runner rejects an oversized final batch
instead of silently dropping unevaluated proposals from the backend's state.

For whole-flow optimization, the Global Analyzer remains a separate task
modeling step instead of another code-search protocol. Inner ask/tell loops
return local corrections; an outer ask aggregates those corrections and a
whole-flow PPA evaluator returns the outer tell. See
[hierarchical_optimization.md](hierarchical_optimization.md).
