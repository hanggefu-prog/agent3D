## Current integration status: source-development phase only

Endpoint update (2026-10-09): evaluate only the edited/legalized post-CTS state
with placement-RC joint skew and setup/hold STA. No downstream flow is required.
The existing draft was developed under the earlier prompt; this update does not
retroactively change its recorded prompt or qualify its physical execution.

The independent post-CTS joint STA and fixed legalizer have been exercised on
the four reference cases. The native branch-context producer and action executor
are not yet available for final evaluations. The instruction's references to a
supplied executor describe the intended final interface, not a working service
in this container. Do not replace that service with fabricated timing results.

Implement the policy against the documented structural boundary in
`AgenticBench/engineering_bench/interfaces.py:validate_clock_plan`. The return
value is exactly `{"actions": ["opaque_action_id", ...]}`. Context schema is
`clock-compensation-context-v1`. Its `actions` mapping contains each action's
`qualified`, `exclusive`, `branch_id`, `affected_original_sinks`,
`placement_site_ids`, `added_buffers` and `added_area_um2`. Limits include
`max_added_buffers` and `max_added_area_um2`. Mutually exclusive alternatives,
shared sinks and overlapping placement sites must not be selected together.

Additional timing fields needed by your policy should be precisely documented
as an integration requirement, with units and missing-data handling. Use clearly
labeled synthetic fixtures for useful, aligned, hold-limited and conflicting
cases. Do not hardcode case names, reference answers or an assumed buffer delay.
The final native producer will need to provide measured evidence for those
fields; your synthetic fixtures are not physical qualification. Inspect the
maintained CTS/passive-HBT code to ground the policy in actual clock semantics.
