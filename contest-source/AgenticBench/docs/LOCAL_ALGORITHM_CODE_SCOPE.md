# Local algorithm code scope and numerical runtime

The eight local tasks evolve algorithms through declared interfaces. The source
adapter owns the surrounding flow; candidate programs own the numerical policy.
Their scoring formulas, paired seeds, physical gates and resource budgets are
unchanged by the 2026-10-18 language expansion.

| Task | Permitted algorithm expression | Boundary retained |
| --- | --- | --- |
| Placement initialization | NumPy numerical operations and local helper functions inside `initialize_position` | Native preconditioner, fixed nodes, tier ownership and array layout |
| Placement preconditioner | Torch numerical operations, additional instance methods and instance-local state | Three public method signatures, gradient masks, objective inputs and native initializer |
| Nesterov controller | Scalar arithmetic, math functions, local helpers and bounded two-die loops | Native optimizer, acceptance/backtracking and fixed iteration budget |
| TP-GNN loss | More vectorized reductions/indexing/sorting, local helpers, literal loops and intermediate detach | Frozen sampling/training/decoder; one finite differentiable scalar loss; no input mutation or dense all-pairs work |
| HBT assignment | Arithmetic/branches plus finite math functions such as logarithms, square root, power and hypot | Interpreted scalar policy, no loops/state; native legal sites, search and tie-breaking |
| GRT iteration control | Existing bounded scalar language plus log2, log10 and expm1 | Three controls, native clamps and iteration budget |
| GRT rip-up | Same numerical language extension | Native connectivity/geometry and bounded rip-up decisions |
| GRT maze cost | Same numerical language extension | Positive cost clamps and tighter hot-loop limits |

Helpers use plain named arguments and constant defaults. They cannot use
recursion, decorators, aliases, further nested definitions, imports or external
I/O. The preconditioner retains its existing instance-state contract. TP-GNN
allows up to eight literal iterations per loop, 64 nested iterations including
helper calls, and a bounded expanded operation count. Its source-patch allowance
increases from 220 to 360 added lines to accommodate helpers and trusted
compatibility code; the one-file scope, 18,000-character program limit and
3,000-node AST limit remain. The other task budgets remain unchanged.

## Torch compatibility

The native audit image contains Python 3.8.10 and Torch 1.7.1+cu110. Its missing
`torch.nan_to_num` caused the earlier preconditioner failure: the validator had
already accepted the call. Relaxing a whitelist alone would not fix execution.

The trusted adapter now lowers explicit `torch.nan_to_num` references into a
source-local implementation for real floating and integer tensors. It preserves
finite values, shape, dtype/device and gradients at finite entries. Default
infinity replacements use the dtype limits. The supported `out` path requires
the same shape, dtype and device and does not allow automatic differentiation.
TP-GNN cannot use `out`, because its input tensors are read-only.

This does not monkey-patch Torch or upgrade the image/DREAMPlace ABI. Use the
module function rather than newer tensor `.nan_to_num()` methods. Other APIs
must exist in the installed runtime. The prompt lists the supported TP-GNN
module APIs; native tests check their availability. Fully detached losses still
fail the native `requires_grad` check before optimizer execution.

## Validation and result identity

The new tests cover helper admission and rejected recursion/aliases, enclosing
training-scope isolation, loop bounds, source-local compatibility, the original
failed preconditioner on small tensors, gradients, masks and unchanged inputs.
Small native TP-GNN training exercises the injected helper and still rejects a
fully detached loss. HBT checks compare the harness grammar with native
interpretation; routing checks compile scalar expressions and compare with
Python results.

These are compatibility and contract tests, not new multi-case PPA results.
The original failed experiment remains failed in its sealed evidence. New
campaigns must bind their source/runtime and qualify matching references; do
not relabel old matrices merely because the algorithm's initial program is
unchanged. Initial programs, task objectives and selected-case seed schedules
are unchanged.

See the [2026-10-18 validation record](../releases/local-code-scope-20261018/README.md)
for maintained regressions and execution checks in the actual evaluation image.
