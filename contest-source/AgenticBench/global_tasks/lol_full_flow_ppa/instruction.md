# LoL full-flow PPA optimization

Improve the user-selected LoL design under the maintained Open3DBench flow.
The user command supplied in round 0 defines the primary PPA objective and any
case-specific constraints.

You receive an isolated, writable copy of the complete Open3DBench subject
repository together with read-only benchmark inputs and an independent run
directory. You may inspect, add, remove, or modify implementation and
configuration files anywhere in that subject workspace, and you may compile
and run the available tools repeatedly. Start each intervention with an
explicit plan, use measured evidence to refine it, and preserve enough logs and
artifacts to reproduce the final candidate.

Random seeds are evaluator-owned controls (I/O 1, partition 1001, placement
1000, and deterministic routing order -1). They are fixed across reference and
candidate evaluations and are not part of the optimization surface.

The benchmark harness, legality checks, reference inputs, scoring code, and
trusted evaluator are not part of the optimization surface. Do not attempt to
replace, bypass, forge, or weaken them. Candidate-reported metrics are
diagnostic only; final legality and PPA come from the trusted evaluator and are
admitted only after the separate verifier validates the complete stage chain.

Submit the final workspace changes together with the canonical run manifest,
the final checkpoint, and normalized metrics requested by the task contract.
