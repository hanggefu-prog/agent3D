# Codex workspace co-design baseline

This adapter runs the existing global timing and post-CTS thermal tasks with a
persistent Codex session and an editable source checkout. It is distinct from
the legacy focused-program adapter. The task definitions remain authoritative
for scores, fixed seeds, legality, and the five-final budget. New workspace
defaults use the [2026-09-27 reference-aligned one-pass contract](reference_alignment_20260927.md);
historical two-pass tasks remain separate.

## Execution boundary

1. `scripts/run_scheduled_codesign.py` verifies a real model/tool/resume receipt,
   binds immutable task contracts, and prepares one input bundle per case.
2. `scripts/run_workspace_codesign.py` independently qualifies the task reference
   and creates a private source checkout. Every round starts with a written plan.
   The same explicit Codex session continues through all five rounds.
3. `global_bench/workspace_codex.py` starts a private offline container and a
   host-authenticated Codex client. Credentials are not mounted into containers.
   The host client has a minimal read-only filesystem profile and no native shell.
4. `global_bench/container_terminal.py` provides one MCP terminal tool. Tool input
   cannot select a different container, host command, working directory, or
   environment. The subject and scratch are writable; Git metadata, benchmark
   infrastructure, inputs, and prior evaluation results are read-only.
5. `scripts/workspace_codesign_evaluator.py` snapshots actual Git changes, runs
   the candidate frontend, then a separate pristine backend and standalone
   verifier. The agent's claimed metrics are never the scoring source.

Container termination precedes snapshotting, including any background process
started by the agent. Failed candidate evaluations consume a final attempt;
transport or tooling failures stop with `needs-attention`, never silently retry.
Existing attempts are not automatically replayed.

## Agent interface

The agent reads the complete task, source, reference, and previous results with
the container terminal. It edits actual source files, rather than returning a
fixed function or JSON program. `agentic_candidate.yaml` supplies source-relative
`partition` and `placement` configuration overrides. Seeds, frozen inputs,
clocks, evaluator settings, and protected paths cannot be overridden.

There is no five-file or line-count restriction. Python frontend and DREAMPlace
Python changes are executable. Changes requiring compilation are currently
rejected explicitly: this adapter does not yet build a candidate-specific native
runtime image. It must not claim to evaluate native edits using stale binaries.

The archived one-OPT flows remain immutable historical evidence. Both new
one-OPT and historical two-OPT tasks require task-matched reference qualification; copying an old score
does not satisfy this requirement. Input/floorplan copies retain their original
bytes, manifests, and hashes. Docker CPU/RAM quotas and physical thread settings
are recorded separately.

## Results

Each objective owns `CASE/OBJECTIVE/codex/`:

```text
run_config.json           task, model, source, image, resources, final budget
status.json               current state or actionable failure
subject/                  persistent editable source
control/                  immutable task instructions
scratch/                  explicit round plans and agent diagnostics
rounds/NN/agent/           prompt, command, JSONL events, stderr, final message
results/reference/        independently qualified reference
results/round-NN/          source snapshot, physical logs, manifests, verification
history.json              trusted evaluation feedback across rounds
best.json                 best verified score, including the reference
summary.json              final success or failure summary
```

Scheduler admission uses CPU, RAM, disk, and free GPU memory. Existing GPU users
do not automatically exclude a server. Host-local persistent queues survive a
disconnected laptop. Offline duration estimates are not guarantees of optimal
makespan or measured runtime.

## Qualification status on 2026-09-23

CLI 0.156.1 and `gpt-5.6-sol` returned `OK`, completed their turns, and exited
normally on amax5 and ta2. This proves model transport, not workspace operation.
The first container tool trial failed because Code Mode host was disabled.
The user subsequently approved enabling that runtime for this batch's child
processes only. The exact container terminal tool is pre-approved for this
invocation; no other tool receives that grant, and host permissions stay minimal
and read-only. No global client configuration or credential storage is changed.
The amax5 and ta2 real isolation/edit/resume preflights passed with this
configuration. Production launch requires each target host's own qualified receipt.

`scripts/qualify_workspace_codex.py` must pass the fixed trusted isolation probe,
real editing, and explicit-session resume on each target image before launch.
Model-written isolation assertions alone never qualify the environment.
