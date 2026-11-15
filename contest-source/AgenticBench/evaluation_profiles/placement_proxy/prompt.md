# Concurrent two-die placement proxy

This evaluator-owned profile measures an already proposed placement correction
at the native DREAMPlace global-placement boundary. It supplies the hierarchical
optimizer with a cheap first-improvement signal using 90% HPWL and 10% maximum
overflow. The proxy is not the final objective: a selected correction must
still pass HBT materialization, two coordinated OPT passes, and continuous
global STA.

The profile requires both dies, finite tensors, legal density behavior, and
design-independent control logic. Shared and die-specific implementations are
both valid when they use normalized physical state. Candidate proposal, local
acceptance, and the outer promotion decision remain outside this profile.
