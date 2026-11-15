# EvoStage baseline integration

OpenAgentic3DBench can use four scalar search algorithms from an external
`EvoStage-main` checkout:

- `randsample`;
- `hillclimb`;
- `eoh`;
- `evostage`.

Their upstream search loops are adapted to the framework-neutral
`ProgramEvaluationHarness`. New search implementations should use the public
`SearchBackend.ask`/`tell` interface and `AskTellRunner` documented in
[ask_tell.md](ask_tell.md); they do not need to depend on EvoStage.

EvoStage remains a separate dependency. No LLM4AD/EvoStage source, model key,
Open3DBench source, benchmark payload, or experiment output is copied into this
repository.

## Candidate boundary

EvoStage normally treats an LLM completion as executable Python. That contract
is too broad for source/configuration evolution. For parameter-only tasks, the
adapter instead asks the LLM for one function:

```python
def evolve_candidate() -> dict:
    return {
        "schema_version": 1,
        "config_overrides": {},
        "candidate_files": {},
        "source_patch": "",
        "notes": "one testable hypothesis"
    }
```

The function is never executed. Python's AST parser accepts only one function,
an optional docstring, and one static literal return. Imports, calls,
attributes, comprehensions, control flow, decorators, arguments, and top-level
statements are rejected before the task verifier runs.

The verified dictionary then passes through the same scope policy and
Open3DBench evaluator used by OpenEvolve-style candidates. Parameter bounds are
declared in each `task.yaml`. Source patches cannot create, delete, rename, or
modify files outside the task whitelist.

Component tasks use task-specific program adapters instead of the literal
dictionary. Placement source evolution submits a complete `PreconditionOp`
class, dynamic placement control submits `adjust_learning_rate`, and routing
source evolution submits `score_net_order`. Their trusted adapters validate a
narrow AST grammar and deterministically generate the fixed integration patch;
the routing adapter additionally transpiles its restricted scalar language to
C++. The evolved text itself is never imported or executed by
EvoStage/OpenAgentic3DBench.

Before every model call, the adapter appends structured black-box feedback:
recent attempts, top attempts, complete numeric metrics, selected diagnostic
artifacts, and bounded stage/plan log tails. Source-evolution tasks additionally
receive exact line-numbered excerpts from the evaluated Open3DBench checkout.
The prompt therefore exposes the same kinds of state used by OpenEvolve's
prompt sampler while every candidate remains non-executable at the bridge.
All prompt/response pairs are written to `prompt_trace.jsonl` for audit.

## Evaluation modes

| Mode | Purpose | Score meaning |
| --- | --- | --- |
| `verify` | Fast no-network/no-EDA adapter test | transport-only; never report as QoR |
| `interface` | Verifier plus the real Open3DBench `plan` command | interface-only; no QoR |
| `full` | Isolated candidate workspace and real stage execution | task `combined_score` |

Failed real-evaluation candidates return no fitness rather than a zero-valued
valid fitness. For placement tasks, `full` means the complete declared local
boundary—DREAMPlace global placement only—not the Open3DBench backend flow.
Duplicate candidates reuse their recorded score and are explicitly
recorded as cache hits. Every accepted, rejected, failed, or cached sample is
recorded under `history.jsonl`; candidate payloads and
Open3DBench artifacts remain under the selected output directory.

## Install and inspect

```bash
python3 -m pip install -e '.[evostage]'

openagentic3dbench-evolve doctor \
  --evostage-root /path/to/EvoStage-main
```

The doctor command verifies required classes and prints fingerprints for the
external files. The adapter restores the function header omitted by this
EvoStage fork at runtime, disables its embedded model client, disables generated
code execution, and replaces the upstream introspective profiler. It never
modifies the external checkout.

## LLM API-key interfaces

The default configuration is
`local_bench/configs/qwen.yaml`. It contains only the endpoint, model, sampling
parameters, and the name of the environment variable holding the key.

```bash
export DASHSCOPE_API_KEY='your-key'

openagentic3dbench-evolve llm-check
openagentic3dbench-evolve llm-check --live
```

The key has no command-line option and is never written to YAML, run manifests,
profiler state, or logs. The client reads it from the environment only while
building an HTTPS request. `llm-check --live` makes one minimal request; omit
`--live` to inspect configuration without network access.

The official OpenAI backend uses the Responses API and the separate
`local_bench/configs/openai.yaml` configuration:

```bash
export OPENAI_API_KEY='your-key'

openagentic3dbench-evolve llm-check --provider openai
openagentic3dbench-evolve llm-check --provider openai --live
```

Select it for evolution with `--provider openai`. A ChatGPT/Codex account login
is not read as an API credential; the benchmark requires `OPENAI_API_KEY` for
this provider.

DeepSeek uses the same credential-safe Python client through its
OpenAI-compatible Chat Completions endpoint. Its checked-in configuration is
`local_bench/configs/deepseek.yaml`; it enables streaming and sends DeepSeek's
explicit `thinking` control when `reasoning_effort` is configured:

```bash
export DEEPSEEK_API_KEY='your-key'

openagentic3dbench-evolve llm-check --provider deepseek
openagentic3dbench-evolve llm-check --provider deepseek --live
```

For another OpenAI-compatible service, copy
`local_bench/configs/openai_compatible.example.yaml` to a local configuration
path and fill in only its HTTPS `base_url` and `model`. Do not put the key in
YAML. The template uses `/chat/completions`, while the checked-in official
OpenAI configuration uses `/responses`:

```bash
cp local_bench/configs/openai_compatible.example.yaml /data/my-openai-compatible.yaml
# Edit /data/my-openai-compatible.yaml: base_url and model only.
export OPENAI_COMPAT_API_KEY='your-key'

openagentic3dbench-evolve llm-check \
  --provider openai \
  --llm-config /data/my-openai-compatible.yaml

openagentic3dbench-evolve llm-check \
  --provider openai \
  --llm-config /data/my-openai-compatible.yaml \
  --live
```

`llm-check` without `--live` validates and prints the non-secret resolved
configuration. `--live` sends exactly one minimal conversation request. The
same `--provider openai --llm-config ...` pair can then be passed to `run`.
The loader requires HTTPS so a bearer API key cannot be sent in cleartext.

To use another Qwen-compatible endpoint or model, copy the YAML file and pass
`--llm-config /path/to/qwen.yaml`. Keep `api_key_env` as an environment-variable
name, not a secret value.

## Full run example

```bash
openagentic3dbench-evolve run placement_parameters \
  --method evostage \
  --evostage-root /path/to/EvoStage-main \
  --provider qwen \
  --evaluation-mode full \
  --open3dbench-root /path/to/Open3DBench-V2.0 \
  --benchmark-input /data/openagentic3dbench-inputs/bp_fe_lol_ng45_mtkahypar \
  --reference-run /data/baselines/bp_fe_ng45 \
  --output-dir /data/openagentic3dbench/placement-evostage \
  --max-samples 50 \
  --max-generations 50 \
  --population-size 5
```

`--max-samples` is the generated-candidate/evaluation budget and normally binds
first; `--max-generations` is a safety ceiling. Generate the standardized HPWL,
overflow, and maximum-density convergence artifacts after a run with:

```bash
openagentic3dbench-convergence /data/openagentic3dbench/placement-evostage
```

The benchmark input is generated once with `openagentic3dbench-task
prepare-input`; see [benchmark_inputs.md](benchmark_inputs.md). Its manifest
validates the frozen input, floorplan, partition, and every published artifact
before search begins. The older `--baseline-run` argument remains available for
historical inputs without that stronger contract.

Use one output directory per task, method, design, seed, and model. Large EDA
outputs should remain on a data disk rather than in either source repository.

## Validation performed

The external checkout at `/home/example/Desktop/workspace_for_agent/EvoStage-main`
was exercised with the deterministic replay provider on 2026-08-27:

| Task | RandSample | HillClimb | EoH | EvoStage |
| --- | --- | --- | --- | --- |
| netlist partition | pass | pass | pass | pass |
| placement initialization/preconditioner | pass | pass | pass | pass |
| dynamic placement control | pass | pass | pass | pass |
| global-routing parameters | pass | pass | pass | pass |

All sixteen verifier-mode paths accepted valid candidates. RandSample
also passed `interface` mode for all four tasks against the current
Open3DBench-V2.0 `plan` command. These tests validate the environment and
interfaces only; they do not claim PPA improvement. QoR experiments begin only
after a real model key and frozen baseline/reference runs are supplied.
