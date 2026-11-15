# Agent adapters and campaign options

The GRT campaign controller (`global_bench/grt_campaign.py`) drives one agent
invocation per round through an adapter in `global_bench/agent_adapters.py`.
An adapter handles three agent-specific concerns:

- `overrides(...)`: configuration overrides that bind the round's MCP tools.
- `build_command(...)`: the argv for a fresh invocation or a resumed session.
- `parse_events(path, thread_id=None)`: returns `{thread_id, completed,
  tool_calls, usage}` from the recorded JSONL events.

Every other part of a round stays in the harness and is shared by all adapters:
the prompt, isolation, checkout, verification, budgets, and evaluation.
`workspace_codex.run_workspace_turn` takes `adapter=` (default
`CodexCliAdapter`) and still enforces the schema, the timeout, process-group
termination, and the session-identity checks.

## Built-in adapter

`codex-cli` (`CodexCliAdapter`) produces exactly the command of the legacy
`build_workspace_command`, whether it starts a fresh `codex exec` or runs
`codex exec resume <id>`. It reads these events:

- `thread.started` gives the session id.
- `item.completed` with an `mcp_tool_call` item for the `terminal` server's
  `exec` tool counts as a tool call. `checkout_parent` and `gpu_check` are not
  counted, so the round's `tool-calls.jsonl` normally has one more line (the
  checkout).
- `turn.completed` gives the usage. Codex reports the running total of the
  whole session, including earlier resumed invocations (observed with
  codex-cli 0.162.1), so the adapter keeps the last total and declares
  `usage_scope = "session_total"`.

A missing file, malformed lines, or non-object events are skipped and never
raise an error.

## Adding an adapter

Subclass `AgentAdapter`, set a unique `name`, implement `build_command` and
`parse_events`, and register the class in `ADAPTERS`. A campaign then selects it
with `"agent_adapter": "<name>"`. An unknown name is rejected when the campaign
is constructed, before any round starts. The adapter must return a stable session
id if it supports `session_mode: resume`, and `None` for usage it cannot observe.

## Usage and cost receipts

`usage` is `{"turns": n, "input_tokens", "cached_input_tokens", "output_tokens"}`
for this round only.

- An adapter with `usage_scope = "invocation"` sums every `turn.completed`
  event of the invocation.
- For a `session_total` adapter (Codex), a fresh session's total is the round's
  usage. A resumed round's usage is its total minus the previous total of the
  same thread (`usage_delta`). Summing the raw totals of every round counts the
  session prefix again in each round; on a real 8-round campaign that
  overstated one round's input about twelvefold.
- Without a previous total, or when a total decreased, the resumed round's
  fields are `null`.
- A field is `null` unless the client reports it as a non-negative integer. A
  partial total is never presented as complete. Booleans, floats, and negative
  values make the field `null`.

`cost` is `null` unless the campaign config has a `price_table` entry for the
model:

```json
"price_table": {"gpt-5.6-sol": {"input_per_mtok": 1.25, "cached_input_per_mtok": 0.125, "output_per_mtok": 10.0}}
```

Each model entry must contain exactly these three finite, non-negative numbers,
in USD per million tokens. A malformed table is rejected before any round
directory is created.

The cost is `((input - cached) * input_rate + cached * cached_rate + output *
output_rate) / 1e6`. It is recorded with the rates and the basis that produced
it. Usage that is incomplete, or that reports `cached > input`, yields no cost.

The round row stores `agent_usage`, `agent_cost`, the raw `agent_session_usage`
and `agent_thread_id`. The controller takes the baseline for a resumed round from
the latest earlier row with the same `agent_thread_id` and a known session
total, including a failed generation. The harness never fabricates usage or
prices.

Only the Codex JSONL format is parsed. Claude Code `stream-json` logs (whose
`input_tokens` excludes `cache_creation_input_tokens` and
`cache_read_input_tokens`) need their own adapter.

## Campaign options

All of these options are optional. An absent field keeps the historical
behavior, and an invalid value is rejected with `ValueError` when the campaign
is constructed.

| Field | Values (default first) | Effect |
| --- | --- | --- |
| `prompt_review` | `every_round`, `first_round` | `first_round` requires `--reviewed-prompt-sha256` only for round 1 and for any revised prompt. Later unrevised rounds record `prompt_review: "waived:first_round"`. If a hash is supplied it must still match. The prepared prompt file must always match its recorded hash. |
| `session_mode` | `resume`, `fresh` | `fresh` never resumes the previous session id, repeats the background in every round prompt, and records `session_mode` on the round. |
| `agent_adapter` | `codex-cli` | Name from `ADAPTERS`. |
| `price_table` | absent | Enables cost estimates, as described above. |
| `infrastructure_retry` | absent (no retry) | `{"max_attempts_per_cell": 1..3, "retryable": [...]}`. See the next section. |

## Infrastructure retry (opt-in)

Without `infrastructure_retry`, a failed cell ends the matrix exactly as before.
`failures.json` is unchanged, and a second evaluation of the same candidate is
refused with "no automatic retry".

With a policy, only failures that the strict classifier recognizes in the
configured `retryable` list are retried.

| Class | Evidence required |
| --- | --- |
| `disk_full`, `host_memory`, `gpu_admission`, `docker_daemon` | The worker's exact host-admission error, and no `runs/<job>` attempt directory (the candidate never ran). |
| `lease_lost` | The worker's exact heartbeat-loss error. |
| `container_launch` | The native container failed with exit 125, and every log line is a Docker client line. |
| `lease_timeout` | An interrupted row whose events show lease expiry (no cancellation) and a reconciliation receipt with `terminated: true`. |

The retry rules are:

- Exit 137 alone does not prove host OOM and is not a retry class. It can also
  mean a candidate exceeded its memory limit; independent host evidence and an
  explicit recovery decision are required.
- Any other failure is a candidate or unknown failure, and it blocks every retry
  in the matrix. Retry is all-or-none per matrix.
- Uncertain or cancelling rows require `reconcile` first: the controller raises
  `ReconciliationRequired`, and the round stays `evaluating` for
  `resume-evaluation`.
- Each retry keeps the job id. The old queue row is moved into a `retired`
  event, a pending-then-retired entry in `<matrix>/retries.json` (with class,
  evidence, policy, and the full row), and, if present, the attempt directory
  `runs-retired/<job>/attempt-<n>`. The cell is then resubmitted.
- The archive therefore still sees exactly one verified row per cell.
- Interrupted retries are completed idempotently on the next call.
- When a cell has used `max_attempts_per_cell` attempts, or a cell is withheld
  because another cell blocks the retry, the reason is written in the `retry`
  field of `failures.json`.

The operator fallback reopens the latest failed, evaluating, or reconciliation-required round, or the
baseline matrix, without spending another evaluation:

```bash
python3 -m global_bench.grt_campaign retry-infrastructure --config CONFIG [--retryable disk_full,container_launch] [--baseline]
```

`--retryable` applies when no policy is configured, with an attempt limit of 3.
The command refuses to run when nothing is retryable.

## Model selection

`workspace_codex.MODEL` and the default batch model stay `gpt-5.6-sol`.
`scripts/launch_codesign_batch.py` accepts `--model`, or a `model` field in the
config, and checks it against the config's `model_allowlist`, which defaults to
`["gpt-5.6-sol"]`. A model outside the allowlist is refused before any job
starts. Campaign configs already carry their own `model`.
