# Reference-20260923 placement interface qualification

Status: offline adapter and algebra/state checks only. No native DREAMPlace,
GPU, EDA, model call, new scoring reference or candidate final has been run.

## What changed

The old `placement_parameters` and `placement_init_preconditioner` task
directories are unchanged. Their Adam/2000-iteration contracts and historical
results remain separate. These new versioned variants select the frozen
`bp_fe_lol_ng45_reference_20260923` profile and preserve 4000 native Nesterov
iterations, 0.01 learning-rate initialization, 512 bins, density 0.6, seed 1000,
four threads, enabled macro placement, 10um halos, the 1.001um macro snap guard,
native missing/zero deterministic_flag and use_bb, and fixed per-die settings.
All immutable physical input and assignment identities are enforced by the
launcher's reference-bundle validation, not inferred from matching seeds.

The source variant's initial program contains the native initializer and
preconditioner bodies. Its adapter compares those bodies and emits no source
patch for the baseline. Evolved programs patch only BasicPlace and PlaceObj;
the optimizer selector and native schedule are untouched.

The dynamic variant changes only the active native no-BB Nesterov interface.
Changing `param_groups['lr']` after initialization would not control later native
backtracking steps. Scaling coordinates after `optimizer.step()` would desynchronize
Nesterov's major/reference/history state. Neither shortcut is used.

Instead, a trusted callback is invoked once before the native backtracking loop,
using the current native alpha hint and gradient. Its factors remain fixed for
that step's trials, controlling the gradient term before each trial's native
major point and acceleration are formed. Shared and bottom/top relative
factors have the explicit projection rules in `prompt.md`. Acceleration, native
constraint projection and all history updates retain their original order.
The native alpha update, line-search acceptance rule and maximum ten trials are
unchanged; estimates use actual trial coordinates and their corresponding
gradients. This is a new algorithmic interface and needs separate physical
qualification, not authorization to change use_bb to one.

The initial controller returns `1 x (1,1)`. It is invoked and logs every call,
then returns the identity sentinel so the exact original arithmetic expression
is executed. The BB path fails explicitly; there is no Adam fallback or
disabled-controller success path. Existing fixed/stopped-region treatment is
not redesigned by this task. In particular, the native caller's stopped-region
restoration remains unchanged.

## Offline evidence and remaining gates

`AgenticBench/tests/test_placement_reference_20260923.py` executes the actual
patched optimizer class with a small tensor double. It checks multi-step
identity equality for all optimizer state, nontrivial per-coordinate step
algebra, multi-backtrack major/reference/previous-position/gradient consistency,
one active callback per optimizer step despite multiple native trials,
fixed-coordinate exclusion, invalid-factor rejection, Python compilation,
patch scope, pinned recipe and preservation of the old task contracts.
This test double does not validate Torch kernels or numerical GPU determinism.

Before any EDA search is launched, qualify the source baseline on the adopted
frozen assignment in a new attempt. Separately run the active identity dynamic
baseline on the same frozen inputs, image and recipe. Verify source-overlay
provenance, nonzero `NesterovStepControl` calls for actual steps, complete native
placement metrics, unchanged initial/controller-free trajectory and native
result artifacts. The exact equality/tolerance rule must be recorded before
the comparison. A zero-callback run, BB run, missing metrics or failed
reference is a qualification failure, not permission to use old Adam metrics.
Then qualify at least one nontrivial finite shared/per-die control to show the
new hook affects native steps while invariants hold. These future runs need
their own reference/candidate and resource accounting; do not count the offline
checks as EDA successes or optimization improvements.

`local_bench/reference_placement_evidence.py` independently checks the emitted
native recipe, including native default flags, and all control-log rows. It
requires finite positive controls and contiguous calls/steps, allowing the
native macro stage's counter reset only at an actual optimizer-selection
header. It records params/log hashes and optional overlay-file inventory.
These receipts supplement the existing native output and metric verifiers;
they are not physical success by themselves.
