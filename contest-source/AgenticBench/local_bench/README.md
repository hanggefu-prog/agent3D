# Local benchmark runtime

This package owns candidate validation, isolated evaluation, feedback, and the
framework-neutral ask/tell interface. Task definitions, initial programs and
prompts live in `../local_tasks/`, not here. `task_inventory.py` lists the eight
current definitions, all with connected evaluators and explicit reference admission.

| Area | Entry points | Purpose |
| --- | --- | --- |
| Search | `search_api.py`, `search_session.py`, `openevolve_search.py` | Ask/tell lifecycle and OpenEvolve integration |
| Candidate contract | `model.py`, `policy.py`, `verifier.py`, `program_bridge.py`, `program_safety.py` | Task loading, allowed edits, program translation and checks |
| Model context | `prompting.py`, `prompt_context.py`, `optimization_feedback.py`, `feedback_semantics.py` | Initial program injection, lightweight prompts and meaningful feedback |
| Matrix evaluation | `matrix.py`, `matrix_docker.py`, `seeds.py` | Complete paired case/seed scoring, frozen references and shared resource leases |
| Native evaluation | `evaluation.py`, `workspace.py`, `metrics.py`, `post_placement_timing.py`, `hbt_native.py` | Isolated execution and measured scores |
| Reference evidence | `reference_*.py`, `scoped_reference.py`, `partition_assignment.py`, `runtime_equivalence.py` | Immutable inputs, identity checks and independent audits |
| Interfaces | `cli.py`, `profile_cli.py`, `sta_cli.py`, `skill_cli.py` | Local tasks, separate evaluation-only profiles and infrastructure tools |

Some modules also serve global tasks: reference validation, workspace handling,
STA and CTS thermal helpers are shared dependencies, not unused local tasks.
The optional EvoStage bridge remains a separate legacy entry point; current
OpenEvolve ask/tell runs do not use it.

## Cleanup boundary

The obsolete TA3D and pin-cut evolution scorers, pin-export patch, old combined
placement comparison tool, and eleven retired-task-only input recipes are no
longer shipped. Canonical assignment validation now has a task-neutral module.
`benchmark_inputs/` still includes recipes referenced by evaluation-only profiles
and sealed reference consumers; an old date alone does not make a recipe unused.
In particular, the BP TA3D input remains a dependency of the separate two-pass
STA evaluation profile, not a restored evolution task.

Historical experiment results, receipts and references are not modified by this
source cleanup. Evaluator fingerprints include the extracted assignment reader;
new releases must validate reference compatibility rather than relabel old results.

## Validation scope

The cleanup regression covers task assets, input recipes, assignment validation,
reference identities, ask/tell, prompts, and current thermal scoring. The expanded
offline check passed 566 tests (3 skipped). Two historical routed-thermal tests
still fail because their `global_route` checkpoint is not accepted by the current
external physical checkout's checkpoint schema. This is not a native execution
result and is not permission to weaken checkpoint validation. Servers and sealed
experiment evidence are unchanged.

## Paired local matrices

All eight maintained tasks support explicit multi-case/multi-seed evolution through
`TaskEvaluationOptions.evaluation_matrix` and the local CLI. See
[Local evaluation matrices](../docs/LOCAL_EVALUATION_MATRICES.md) for qualification,
equal-case aggregation, bounded Docker dispatch and compatibility limits.

## Algorithm code and numerical compatibility

[Local algorithm code scope](../docs/LOCAL_ALGORITHM_CODE_SCOPE.md) describes the
helper/numerical language supported by all eight tasks, the Torch 1.7 compatibility
path and the physical/evaluation boundaries that remain fixed.
