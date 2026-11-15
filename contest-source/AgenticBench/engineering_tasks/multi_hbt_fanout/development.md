## Current integration status: source-development phase only

Endpoint update (2026-10-09): use placement-RC joint STA immediately after branch
materialization and legalization. GRT is no longer part of this task. Keep the
fixed target-net/original-receiver delay definition and report setup/hold WNS/TNS.
The earlier draft's prompt remains historical evidence; do not rewrite it or
reuse a GRT-RC reference as a placement-RC reference.

The complete multi-HBT materialization/read-back and receiver-delay evaluation
bridge is not ready. The task instruction describes the final contract; it does
not mean an undocumented materialization service is already installed. OpenDB,
OpenROAD and the maintained physical Python source are available for inspection
and small functional tests. Do not invent a working native API or run full cases.

Read `open3dbench/backend/hbt_materialize.py`, `passive_hbt.py`, the typed handoff
code and `AgenticBench/engineering_bench/interfaces.py:validate_branch_map`.
Implement the synthesis policy and as much real OpenDB integration as the public
runtime APIs support, within `engineering/multi_hbt.py` and permitted helpers.
The entrypoint remains `synthesize(context, output_dir)`.

The fixed context schema is `multi-hbt-context-v1`. `target_nets` maps logical net
IDs to `original_driver`, `original_remote_receivers`, `original_hbt_id` and
`original_site_id`. `spare_site_ids` and `limits.max_added_hbts` are harness-owned.
The logical branch map covers all targets; each value is one to three unique
physical branch IDs. Native read-back must establish logical driver identity,
disjoint complete original receiver coverage, noninverting buffer proof, single
driver, HBT/site identity and distinct die-qualified source/sink electrical nets.
Do not try to satisfy read-back with your own boolean assertions.

Specify additional geometry, library, path and target-timing inputs explicitly,
including units, in your module documentation. Keep unsupported/missing native
capabilities explicit. A planning-only implementation is an incomplete draft,
not a successful final. Add deterministic small tests for the policy and any
implemented physical transformations, including rejected parallel electrical
links, exhausted sites and preservation of the one-HBT case. No full-case GRT,
STA or search is permitted during this preliminary phase.
