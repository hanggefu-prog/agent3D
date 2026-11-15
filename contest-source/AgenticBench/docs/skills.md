# Benchmark Skills

OpenAgentic3DBench separates scientific search spaces from reusable execution
procedures:

```text
Task                         Evaluation profile
what may evolve              how a selected candidate is replayed
local objective              fixed downstream stages and validity gates
        \                    /
         \                  /
          Skills: shared tool-use procedures
          inspect / build / freeze input / evaluate / analyze
```

`local_tasks/` contains only evolvable problems. `evaluation_profiles/`
contains evaluator-owned replay contracts. `skills/` contains agent-readable
procedures and deterministic helpers. Moving an operation into a Skill does not
make it an optimization task and does not add it to an evolution budget.

## Registered infrastructure

| Skill | Responsibility | Explicitly excluded |
| --- | --- | --- |
| `open3dbench-environment-check` | qualify adapter, revisions, image, tools, and GPU | builds and QoR runs |
| `open3dbench-runtime-build` | compile DREAMPlace or OpenROAD/OpenSTA and verify binaries | candidate generation and QoR interpretation |
| `open3dbench-benchmark-input` | prepare, validate, and transfer immutable input bundles | in-place input edits |
| `open3dbench-candidate-evaluation` | verify and score one bounded local-task candidate | ask/tell and downstream STA promotion |
| `open3dbench-sta-evaluation` | promote a selected candidate through two-pass OPT/STA | search, repair, and population ranking |
| `open3dbench-thermal-evaluation` | evaluate fixed HotSpot snapshots and attribute thermal improvement across placement, OPT/CTS, and routing | candidate generation or thermal-model changes |
| `open3dbench-results-analysis` | audit metrics/failures and generate convergence comparisons | rerunning stages or changing status |
| `open3dbench-stage-replay` | resolve changed-stage impact and replay the smallest trusted suffix | candidate acceptance |
| `open3dbench-artifact-handoff` | freeze workspace changes and canonical artifacts for a separate verifier | legality and QoR scoring |
| `operate-open3dbench-global-workspace` | execute a scope-free Intervention in an isolated subject checkout | changing benchmark control or verifier code |

Analyzer and domain Task Agent Skills are registered in the same catalog. Each
record declares capabilities and dependencies. Inspect it with:

```bash
openagentic3dbench-skill list
openagentic3dbench-skill inspect optimize-mol-layout
```

Evaluation contracts have their own inventory:

```bash
openagentic3dbench-profile list
openagentic3dbench-profile inspect placement_proxy
openagentic3dbench-profile inspect tp_gnn_placement_thermal
openagentic3dbench-sta list
```

The general profile CLI includes cheap inner proxies and downstream promotion
profiles. The STA CLI is a compatibility-focused restricted view containing
only profiles that execute exactly two timing-optimization passes.

The catalog loader validates every directory, Skill frontmatter name,
capability list, dependency, and cycle. An unregistered Skill directory is a
test failure.

## Code-fixed use by Task Agents

When the orchestrator dispatches a partition, layout, place-opt, or routing
Task Agent, it copies the primary stage Skill to `task-agent/skill/` and copies
the catalog's transitive dependencies to `task-agent/supporting-skills/`. The
Task Agent prompt lists these immutable snapshots and fixes their precedence:

1. user objective and immutable constraints;
2. Global Plan and Local Analyzer hypothesis;
3. stage Skill for algorithm scope and correctness;
4. infrastructure Skills for environment, build, input, evaluation, and result
   handling;
5. version-adaptive `repository_context` and measured artifacts.

The snapshots become part of the resumable task directory. Later edits to the
framework Skill library cannot silently change an active experiment.

## Adding a new common operation

Create a focused `skills/<name>/SKILL.md` with `name` and `description`
frontmatter, add `agents/openai.yaml`, register the directory and capabilities
in `skills/catalog.yaml`, and declare it in the `requires` list of each agent
that must receive it. Put deterministic scripts inside that Skill only when an
existing benchmark CLI cannot provide the operation. Keep algorithms and
version-specific paths in Open3DBench or its adapter; a Skill describes the
stable procedure and consumes adapter output.
