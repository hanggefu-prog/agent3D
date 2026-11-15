# TP-GNN partition proxy

This evaluator-owned profile measures an already proposed TP-GNN correction by
rerunning partition only. It supplies the hierarchical optimizer with a cheap
first-improvement signal based on legal timing-weighted cut. The proxy is not
the final objective: a selected correction must still pass placement, two
coordinated OPT passes, and continuous global STA.

The profile preserves complete assignments, timing-weight semantics, numerical
safety, and linear or sampled graph memory. It records ordinary cut, balance,
loss, and embedding statistics as diagnostics. Candidate proposal, local
acceptance, and the outer promotion decision remain outside this profile.
