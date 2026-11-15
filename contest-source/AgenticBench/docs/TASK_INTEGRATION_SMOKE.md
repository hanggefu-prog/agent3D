# Default-case five-round integration smoke

Historical plan: several tasks and input recipes listed below have been retired
and removed. Do not use this plan to launch current experiments; see
[the maintained task guide](tasks.md). Existing experiment evidence is unchanged.

This is an integration test of the existing tasks, not a new optimization
algorithm or a replacement for the maintained evolutionary baselines.

## Contract

- Test each distinct task once on its default case. Aliases are not extra tasks.
- Eleven local tasks and three instantiated global tasks give 70 planned
  generated candidate finals. Initial reference qualification is separate.
- The uninstantiated `lol_full_flow_ppa` template is excluded until its owner
  selects a concrete objective and weights.
- Local tasks use the existing contextual prompt, program bridge, verifier,
  evaluator, and ask/tell records. Codex returns text only; it gets no tools.
- Global tasks use the existing persistent Codex container workspace, explicit
  per-round plan and independent protected evaluator.
- Use existing host authentication, model `gpt-5.6-sol`, and process-local
  network configuration only. No credentials enter EDA/agent containers.
- Exactly five generated proposals are requested after qualification. Rejected
  or failed proposals consume a round. Repeated programs are evaluated again,
  never counted as real physical evaluations via cache hits.
- A task is not proven runnable merely because its controller exits: report
  candidate generation, scope rejection, physical success/failure and reference
  qualification separately. Environment failures stop that task without retries.

## Input and execution boundaries

`scripts/prepare_task_smoke_input.py` prepares a named input profile inside a
private source checkout and pinned offline Docker image. It rejects an existing
attempt. Compatible local tasks can share the same sealed input bundle.

| Local task directory | Case | Input profile |
|---|---|---|
| netlist_partition | bp_fe | bp_fe_lol_ng45_mtkahypar_frontend |
| tp_gnn_learning_objective | bp_fe | bp_fe_lol_ng45_tp_gnn |
| tp_gnn_ta3d_learning_objective | bp | bp_lol_ng45_tp_gnn_ta3d_portable |
| tp_gnn_pin_cut_learning_objective | bp | bp_lol_ng45_tp_gnn_pin_cut_v1 |
| placement_parameters | bp_fe | bp_fe_lol_ng45_tp_gnn_placement |
| placement_init_preconditioner | bp_fe | bp_fe_lol_ng45_tp_gnn_placement |
| global_routing_parameters | bp_fe | bp_fe_lol_ng45_mtkahypar_parameter_cts |
| global_routing_net_ordering | ariane133 | ariane133_lol_ng45_tp_gnn_cts |
| global_routing_ripup_policy | ariane133 | ariane133_lol_ng45_tp_gnn_cts |
| global_routing_maze_cost | ariane133 | ariane133_lol_ng45_tp_gnn_cts |
| global_routing_iteration_controller | ariane133 | ariane133_lol_ng45_tp_gnn_cts |

The portable TA3D profile changes only the historical machine-specific timing
input path: native timing extraction produces new matching features. The dated
profile remains unchanged. Pin-cut preparation applies the existing trusted
pin-identity extractor patch in its private checkout; its initial evaluation
must receive that same prepared `bundle/run` as `--reference-run`.

The two Mt-KaHyPar profiles match their task base configurations exactly; they
do not relabel the historical routing-source profile. The GRT parameter task
still exposes a legacy DRT-only `max_iterations` field. It has no physical
effect in a GRT-only evaluation and must not be credited as an active GRT knob.

`scripts/run_local_task_smoke.py` qualifies the initial program, then uses the
existing ask/tell harness with a credential-free Docker evaluation transport.
The shared `runs` directory is read-only in each evaluation; only the current
evaluation's output and scratch are writable. All tasks use the same batch-root
resource locks as other controlled experiments on that server.

Routing-source tasks require `scripts/build_routing_candidate_image.py`. It
verifies and applies the source patch, compiles with a fixed trusted Dockerfile
without network, and installs OpenROAD into an immutable candidate image.
Evaluation validates image identity, source fingerprint and executable checksum.
Compilation and EDA share one resource reservation. No host binary is selected.
The classic Docker builder is required for the fixed CPU/memory build caps;
unsupported environments fail explicitly. Each build retains source evidence.

## Global endpoints

`scripts/run_global_task_smoke.py` reuses a qualified model/workspace/resume
receipt via `--preflight-result`; it never generates an extra model probe.
Global tasks pin no case, so this smoke records `bp_fe` as its explicit default.

| Endpoint | Existing task | Fixed evaluation boundary |
|---|---|---|
| timing | lol_partition_placement_timing | Exactly two OPT passes, WNS/TNS |
| thermal | lol_partition_placement_cts_thermal | Two OPT passes, CTS, cell power, HotSpot |
| routed_thermal | lol_partition_placement_thermal | Native one OPT pass, CTS, GRT/DRT, report, HotSpot |

These are distinct tasks, not thermal aliases. No score, physical parameter,
seed, timeout or verifier is relaxed to make the smoke pass.

## Reading results

Read local `run_config.json`, `reference/qualification.json`, `summary.json`,
`evolution/history.jsonl`, per-evaluation `evaluation_result.json`, and retained
native `run/run.json`/stage manifests. Model prompts and completions are under
`model/sample-NN/`; `samples.jsonl` includes the contextual prompt history.

For global tasks read `smoke_contract.json`, `smoke_status.json`, `status.json`,
`history.json`, `summary.json`, `results/reference`, `results/round-NN`, and
`rounds/NN/agent`. Scheduler command completion alone is not physical success.

Do not retry failed attempts in place or alter a running source snapshot.
Diagnose original evidence, retain it, and use a newly versioned run only for
an explicitly justified interface/environment correction.
