# Current-state interface

`plan_clock_compensation(context)` returns exactly `{"actions": [action_id, ...]}`.
The schema is `clock-compensation-context-v2`. Each action is a native-supported
insertion of 1, 2, 4 or 8 noninverting buffers before one existing exclusive
clock-branch input pin. Existing cells and HBTs remain fixed. You may select
compatible disjoint actions within the shared budget, or return an empty plan.

The context has:

- `clock_groups`: group ID -> original sink ID -> measured arrival in ns.
  Group IDs are `clock/active_edge/max_or_min/default`. Never mix groups.
- `timing`: current joint setup/hold WNS/TNS and hold-violating endpoint IDs.
- `paths`: native endpoint-worst paths for max/min and rise/fall. Each row has
  `analysis` (`max` or `min`), `endpoint`, `launch_clock_sink`,
  `capture_clock_sink`, and finite `slack_ns`. Empty clock-sink fields mean
  unavailable or genuinely external; do not silently interpret them as an
  unaffected register clock. This is not every path in the timing graph.
- `actions`: opaque ID -> action. Each has `qualified: true`, `exclusive: true`,
  `branch_id`, `affected_original_sinks`, unique `placement_site_ids`,
  `added_buffers`, `added_area_um2`, `die`, `master`, `target_input_pin`,
  `sites`, actual baseline `input_slew_ns` and `load_pf`.
  `qualified` means a structurally supported action with empty placement slots,
  not that post-edit timing has already passed.
- `delay_estimate_ns`: action-local `{min, max}` for the new buffer chain,
  computed from the actual ng45 rise/fall NLDM tables at the measured input slew
  and receiving pin load. These are estimates, NOT a guaranteed arrival shift
  or STA min/max corners. Changed upstream load, old driver delay, new wire RC,
  and legalization are not included. `estimate_kind` makes this explicit.
- `buffer_cells`: real Liberty area/capacitance and complete NLDM tables.
  `limits.max_added_buffers` is 8; `max_added_area_um2` is eight times the largest
  provided buffer area. Budget and original receiver/branch/site conflicts are
  checked independently.
- `timing_tolerances`: 1e-6 ns for WNS/TNS numerical comparisons. The final
  physical result must not worsen reference WNS/TNS beyond these tolerances or
  create new hold-violating endpoints. Negative reference slack is allowed.

Use a reasonable predictive selection policy and conservative timing reasoning.
There is no required heuristic and no hidden full-case search: after the policy
selects a plan, one physical materialization and joint RC/STA determines success
or failure. Missing optional information is not permission to invent data, but
do not retain an unconditional dependency on unavailable counterfactual reports.
Preserve a justified no-op when no useful safe action is supported.

The host provides the executor and independent legality checks. Do not insert
buffers yourself, invoke EDA from this policy, or change scoring. Update your
tests for this interface; the old v1 measured-feedback API is historical only.
