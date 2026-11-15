# AgenticBench harness design report

Checked against the code on 2026-10-19. This report explains how the local
harness separates what a search agent may propose from what trusted code
measures. It describes code paths. It does not claim that any experiment ran
or that any reference is qualified.

**Language.** The maintained report is written in English. The earlier,
mostly Chinese report is archived byte-identical at
[archive/HARNESS_DESIGN_REPORT_zh_20261004.md](archive/HARNESS_DESIGN_REPORT_zh_20261004.md).
That archive also covers the global workspace drivers and the Agentic3D role
phases as of 2026-10-04. See [archive/README.md](archive/README.md).

Task-level facts (IDs, scopes, formulas, weights) are in
[TASK_SETTINGS_REPORT.md](TASK_SETTINGS_REPORT.md).

## Division of responsibility

- **Open3DBench.** Owns the physical algorithms, inputs, PDK, native tools and
  stage artifacts. That covers `Place-3D`, `open3dbench` and the pinned
  OpenROAD/OpenSTA submodules.
- **AgenticBench.** Owns task definitions, prompts, candidate verification,
  isolated evaluation, references and score extraction.

The two talk only through explicit paths, the `open3dbench` CLI and hashed
stage artifacts. A search backend proposes programs. It never defines its own
physical reward.

## Local evaluation pipeline

```text
local_tasks/<dir>/task.yaml --load_task--> TaskSpec
search backend (ask) --> CandidateProposal (complete program)
  --> program adapter (program_bridge.py, task program_adapter.py)
      complete function --> candidate.json (patch and/or policy file, config overrides)
  --> verify_candidate(spec, candidate, open3dbench_root)          verifier.py
      value constraints, allowed config/source paths, patch limits,
      routing wrapper byte identity (when the root is known)
  --> evaluate_task(...)                                          evaluation.py
      reference admission (scoped_reference / current_nesterov; deferred in dry run)
      resolved config = base config + allowed overrides
      frozen inputs copied and fingerprinted (_materialize_frozen_inputs)
      `open3dbench plan` on the resolved config      <- --dry-run stops here
      isolated clone of the Open3DBench checkout (workspace.py)
      evaluator wrapper patch, then candidate patch, then workspace validation
      native stages from the task's execution.from to execution.to
      metric extractor selected by task kind                     metrics.py
  --> EvaluationResult --> CandidateFeedback --> backend (tell)
```

The components in more detail:

- **`model.py`.** `load_task` parses `task.yaml` into a frozen `TaskSpec`: ID,
  kind, base config, program mode and entrypoint, execution stages, frozen
  inputs, candidate policy and prompt context. The CLI resolves a task by
  directory name or by ID.
- **`verifier.py`.** `verify_candidate` runs before any native work. The
  prompt only explains the rules. The verifier enforces them. A candidate that
  fails is never dispatched.
- **`routing_wrapper_integrity.py`.** For the routing source tasks the two
  evaluator wrappers may appear in a patch only as the exact bytes that the
  trusted adapter generates. A missing root skips this check. See the task
  report.
- **`evaluation.py`.**
  - `evaluate_task` records a cascade of named steps in `artifacts`.
  - A dry run returns after the Open3DBench plan, with `combined_score` 0.0,
    `runs_successfully` 0.0 and `dry_run` 1.0. It checks configuration only.
    The dry run runs `python3 -m open3dbench plan`, so the `python3` on `PATH`
    needs PyYAML.
  - A full run needs a frozen upstream prefix. That is either
    `--benchmark-input`, a validated bundle from `prepare-input`, or the legacy
    `--baseline-run`.
- **`metrics.py`.** It maps each task kind to its extractor:
  - `placement_component_metrics`;
  - `placement_nesterov_metrics`;
  - `post_placement_timing.timing_metrics`;
  - `hbt_native.timing_metrics`;
  - `_routing_source_policy_metrics` for routing, with score versions in
    `routing_scoring.py`.

  Extractors read native logs and manifests and validate them before they
  score. Missing measurements raise errors. They are never filled with zeros.

## Search interface

[`search_api.py`](../local_bench/search_api.py) defines the framework-neutral
ask/tell contract. Its main types:

- `SearchProblem`;
- `CandidateProposal`, which holds a proposal ID, the complete program and
  metadata;
- `CandidateFeedback`, which holds the program SHA, status, score, metrics,
  artifacts, error and runtime;
- `SearchBackend`;
- `AskTellRunner`;
- `ProgramEvaluationHarness`.

Backends include OpenEvolve (`openevolve_search.py`, `openevolve_adapter.py`),
the legacy EvoStage bridge and custom callbacks. A score counts as valid
feedback only when the status is accepted or cached and the score is measured.
A valid score does not mean the candidate beats the reference. A generated
candidate that fails or is rejected still uses its evaluation slot. See
[ask_tell.md](ask_tell.md).

## Model context and feedback

- **`prompting.py`.** Builds the system prompt from shared constraints plus
  the task's `prompt.system_message`.
- **`prompt_context.py`, `optimization_feedback.py` and
  `feedback_semantics.py`.** Choose which measured metrics reach the model,
  attach their meanings, and keep recent and top candidates within the
  per-task `prompt_context` budgets. Every maintained task uses 3 recent and
  3 top candidates.
- Routing score-version-2 keys reach feedback only when that version is
  active.
- Optional observations that are missing stay missing. Real zeros are kept.

## Paired matrices and the default profile

[`matrix.py`](../local_bench/matrix.py) runs one candidate over every
case/seed cell (`LocalEvaluationMatrix`) and aggregates equal-weight case
scores. `qualify_matrix` evaluates the initial program in every cell and
freezes the reference runs. Only a matrix with frozen references is a scoring
authority.

[`matrix_profile.py`](../local_bench/matrix_profile.py) loads
`reference_configs/default_matrix_profile.yaml` and renders a reference-free
matrix (`render-matrix`). It does this from the baseline runs a server
provides. The profile declares `qualified: false` and cannot carry references.
See [LOCAL_EVALUATION_MATRICES.md](LOCAL_EVALUATION_MATRICES.md).

## Benchmark inputs

[`benchmark_input.py`](../local_bench/benchmark_input.py) loads input profiles
from `local_bench/benchmark_inputs/<id>/profile.yaml`. Its commands:

- `prepare-input` runs the upstream stages once. It records the size and
  SHA-256 of each stage artifact, plus the source, Python and DREAMPlace
  identities.
- `validate-input` checks a bundle against a task's required stages and
  partition contract.

Profiles marked `qualified: false` serve smoke use only. An example is
`ibex_lol_ng45_cpu_smoke`.

## Global tracks

Global workspace tasks (`global_tasks/`, `global_bench/`), the post-GRT
evolution (`grt_bench/`) and the engineering tasks (`engineering_bench/`) are
maintained by other work packages. Their current designs are in:

- [global_tasks.md](global_tasks.md);
- [POST_GRT_OPTIMIZATION.md](POST_GRT_OPTIMIZATION.md);
- [POST_GRT_AGENT_ROUNDS.md](POST_GRT_AGENT_ROUNDS.md);
- [engineering_tasks.md](engineering_tasks.md);
- [workspace_agentic3d.md](workspace_agentic3d.md).

They share some local modules: reference validation, workspace handling, and
STA and thermal helpers. They do not share the local `combined_score`
formulas. In particular, the post-GRT timing objective weights WNS 0.7 and
TNS 0.3, while the local timing tasks weight them 0.6 and 0.4.

## Evidence and limits

- A connected evaluator, a passing unit test or a successful dry run is not
  native qualification.
- Placement HPWL is not routed wirelength, and GRT success is not DRT or
  signoff.
- A compiled hook is not proof that the hook was called. Routing scores
  require hook diagnostics, and the Nesterov score requires a step-control
  trace.
- Results, references and release receipts are immutable. New semantics get a
  new task ID or score version rather than a relabelled result.
