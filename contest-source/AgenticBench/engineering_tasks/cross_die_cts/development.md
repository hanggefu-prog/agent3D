## Current integration status: source-development phase only

Endpoint update (2026-10-09): score only global skew immediately after CTS and
harness-owned legalization, using placement-estimated RC and propagated clocks.
Setup/hold are optional diagnostics, not hard acceptance gates. No downstream
optimization or routing is required. The existing draft may keep a conservative
internal timing guard, but the harness must not impose that as a new objective.
Retain the original draft prompt as provenance instead of relabeling that run.

The fixed post-OPT-to-CTS normalization and independent post-normalization joint
STA were exercised successfully on all four reference cases. Complete structural,
electrical/resource guards and the native candidate-image build connection are
still being integrated. The task is not yet qualified for candidate finals.

Start from the actual `open3dbench/backend/three_d_cts.py:run_cts` implementation.
Preserve its maintained public invocation and artifact contract. The current
code already transfers remote capacitance and uses passive-HBT joint STA; find
where measured cross-die latency is only reported and develop a real physical
feedback strategy. Study existing native CTS primitives before duplicating them.
Keep all new code in the task's permitted paths.

Python coordinator changes can be developed and tested now. Native CTS changes
are allowed, but a host-owned installed-image build and source/runtime receipt
must be completed before any final uses them. Do not inject a host executable,
claim an unbuilt native edit was evaluated, or edit the independent timing
engine. Record required build/integration work clearly if you change C++.

Use small functional tests and mocks that preserve units, clock ownership and
setup/hold semantics. Existing frozen reports are useful diagnostics, not scores
for your source changes. Do not run full-case CTS/GRT/STA or tune per-case answers.
Retain the fixed data-coordinate restoration/standard legalization boundary;
the eventual independent evaluator, not candidate-generated reports, determines
skew and legality.
