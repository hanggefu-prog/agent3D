# Local code-scope expansion and Torch compatibility

The original preconditioner failed because the native image has Torch
1.7.1+cu110 and no `torch.nan_to_num`; the source verifier had accepted the call.
The adapter now supplies a source-local real-tensor implementation without
changing global Torch state, the image ABI, baseline algorithms or task scores.

The [code-scope guide](../../docs/LOCAL_ALGORITHM_CODE_SCOPE.md) records the
review of all eight tasks. Initializer/Nesterov/TP-GNN functions can use local
numerical helpers, preconditioners can add instance methods, TP-GNN admits more
vectorized operations and intermediate detach, and HBT/GRT gain numerical math
functions. Public interfaces, read-only inputs, physical validity, scoring,
paired seeds and evaluation budgets remain protected. Recursion, external I/O,
module mutation and unbounded loss-loop expansion remain outside the contract.

| Validation | Result |
| --- | --- |
| Maintained local, engineering and post-GRT regression gate | 1,298 passed, 13 skipped, 180 passing subtests |
| Focused language, feedback and HBT contract checks | 126 passed, 5 skipped, 9 passing subtests |
| Native Torch compatibility and tiny TP-GNN training | 5 passed |
| Native HBT policy interpreter and assignment regressions | 11 passed |
| Native compiled routing-language equivalence | 11 passed |
| Tested server code versus published source | All 10 changed runtime Python files match SHA-256 |

The 13 local skips comprise the previous eight optional/unavailable checks and
five new Torch checks. All five new Torch checks passed separately in the actual
EDA image. The native runs used Python 3.8.10, Torch 1.7.1+cu110 and image
`sha256:a339ce29f295d9f50d8b7da66adf5fccaa194026186b16ff5fba6d9da90be86e`,
with two CPUs, 4 GiB memory, no network, a read-only source mount and no GPU.

Native checks executed the original failed preconditioner on small tensors,
verified mask/in-place/input preservation, checked finite values, infinity/NaN
replacement and gradients, and exercised the injected loss in native TP-GNN
training. A completely detached loss still failed the native autograd gate.
HBT initial-policy assignments remain exactly equivalent over ten fixture
seeds. Compiled GRT expressions match their Python values, including the new
`log2`, `log10` and `expm1` operations.

These are language/runtime checks, not new physical matrix results or PPA gains.
The earlier failed candidates and sealed matrix evidence remain unchanged.
No model calls, full EDA matrices or OpenROAD image rebuilds were needed.

Logs: [maintained regression](maintained-tests.log), [Torch/native training](native-tests.log),
[HBT](hbt-tests.log), [routing](routing-tests.log).
[Source hashes](native-source-hashes.json) bind the changed runtime files.
The server keeps its isolated validation under
`${AGENTIC_AUDIT_PARENT}/local-constraints-20261018`.
