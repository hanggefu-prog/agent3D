# Improve the loss of a binary graph partitioner

## Problem and editable scope

A logic-on-logic (LoL) three-dimensional chip places circuitry on two logic
tiers. Partitioning selects the tier of every cell and eligible macro. Nets
crossing tiers require hybrid-bonding terminals (HBTs). Their locations and the
resulting placement affect final setup timing.

This task evolves only `compute_binary_partition_loss`. A graph neural network
(GNN) produces embeddings for contracted groups of instances. A fixed linear
prediction head converts them to logits, then a sigmoid produces each group's
top-tier probability. Label 1 is top and label 0 is bottom. There is no
clustering decoder in this task.

Return one finite scalar tensor connected to the supplied probabilities,
expected cut or per-net cut probabilities through autograd. You may change the
loss expression, add bounded regularizers, emphasize critical nets or apply
nonlinear per-net penalties, and use `epoch` for loss schedules. You may not change the graph,
contraction, features, model, prediction head, optimizer, seed, training budget,
temperature, perturbation schedule, checkpoint-selection rule, hard threshold,
feasibility closure, placement, HBT policy or evaluator. Do not produce an
assignment, embeddings, logits, temperature or optimizer state.

The initial program combines the trusted expected cut with the original
soft physical-area balance and the binary margin penalty:

```
r = sum(node_areas * probabilities) / sum(node_areas)
area_loss = (r - balance_target)**2
margin = relu(0.9 - abs(2*probabilities - 1))**2 / 0.81
binary_loss = 0.25 * mean(margin) + 0.75 * sum(node_areas * margin) / sum(node_areas)
loss = expected_cut + 200 * area_loss + binarization_weight * binary_loss
```

The margin penalty vanishes at p <= 0.05 or p >= 0.95 and penalizes ambiguity
both by group count and by physical area. The baseline uses the supplied soft
probabilities directly for area balance. The hard-threshold assignment and
macro constraints remain subject to the fixed feasibility closure.

The initial loss does not depend on node features or epoch beyond the host's
supplied binarization weight. At exactly uniform p=0.5 its symmetry can still
produce zero gradients; the retained fixed seed feature and scheduled logit
perturbations are part of training. Confidence is measured from noise-free
selected probabilities, before thresholding and closure.

A different loss magnitude is not evidence of better partitioning. Inspect
probability separation by both group count and physical area, raw hard balance,
closure movement, and final measured timing.

## Fixed training and decoding

The baseline recipe is the current regularized binary TP-GNN. It uses flat
partition-last features, two contraction passes, macro singleton groups, two
aggregation hops, embedding dimension 128 and a jointly trained linear head.
CUDA is required, with no silent CPU or deterministic-aggregation fallback.
Adam uses learning rate 0.003 for exactly 400 full-graph optimizer updates.
Input seed-feature scale is 0.05.

For the first 25% of training, temperature is 1 and the supplied binarization
weight is 0. From 25% to 75%, temperature decreases linearly to 0.1 and the
weight rises linearly to 10. The logit perturbation scale starts at 0.02 and
falls to 0 over the same ramp; perturbations use a seeded CPU generator and
are centered by physical area. The final 25% is noise-free. This schedule and
its random-number stream are host-owned; the loss cannot replace them.

Checkpoint selection compares noise-free loss under the same final settings
within the final stable phase. For every comparison call the host supplies
`epoch=max_epochs` and the final scalar settings. A candidate's own epoch
dependence therefore cannot make different checkpoints use different scoring
objectives. This selects checkpoints within one candidate; training loss is
never compared as a reward between candidates.

The chosen probabilities are thresholded at 0.5, with equality assigned to top.
The protected shared feasibility closure then enforces area and macro balance
while preserving whole groups. It may change labels, so inspect both the raw
thresholded partition and the final legal partition. Neither this threshold
nor the closure is differentiated through. The GNN and prediction head learn
through your scalar loss; evolutionary selection uses final measured timing.

## Inputs and units

Keep the exact signature in the supplied program. Let N be the number of
contracted groups and H be the number of retained loss nets. Tensor inputs are
read-only and use the current training device and dtype.

| Argument | Shape and meaning |
| --- | --- |
| `probabilities` | `[N]` differentiable sigmoid probabilities for top; training calls include the host's current temperature and perturbation |
| `expected_cut` | Scalar differentiable normalized weighted expected hyperedge cut, computed by the trusted producer from these same probabilities |
| `net_cut_probabilities` | `[H]` differentiable hyperedge cut probabilities from the same group probabilities, in fixed original-net order after filtering; an internal-group net contributes 0 |
| `net_weights` | `[H]` finite positive frozen timing weights, aligned one-to-one with `net_cut_probabilities`; not differentiable |
| `node_features` | `[N, 10]` fixed unstandardized partition-last group features |
| `node_areas` | `[N]` finite positive physical area sums, including standard cells and macros |
| `balance_target` | Python scalar; fixed target top-area share 0.5 |
| `cut_weight`, `area_balance_weight` | Python scalars; fixed values 1 and 200 |
| `binarization_weight` | Python scalar supplied by the fixed 0-to-10 continuation schedule; 10 in checkpoint comparisons |
| `epoch`, `max_epochs` | Python integers; current one-based update and positive update budget in training; comparison calls use `epoch=max_epochs` |

The ten feature columns are: worst slack, constrained-pin-weighted mean slack,
maximum slew, maximum arrival, minimum required time, summed cell delay,
timing-coverage fraction, distance-to-source hops, one-hop degree and two-hop
degree. Timing values retain the frozen source's time units. Structural counts
and coverage are dimensionless. Coverage zero means that zero-filled timing
values are unknown, not measured zero slack. Normalize different quantities
before combining them. These features describe the fixed partition-independent
two-dimensional prototype, not the candidate's final two-die timing.

No macro flag, macro-only area, allowed-die mask, instance name, timing path,
full net-incidence tensor, embedding or arbitrary cut callback is exposed.
Do not infer macro status from total node area or memorize benchmark rows.

For a net e joining distinct groups G_e, the trusted cut probability is:

```
cut_e = 1 - product(probabilities[g] for g in G_e)
          - product(1-probabilities[g] for g in G_e)
expected_cut = sum(weight_e * cut_e) / sum(weight_e)
```

The producer retains nets with 2 through 32 unique known original instances
before group contraction, removes duplicate group membership, preserves
parallel nets separately, and uses the frozen timing weights. Nets entirely
inside one group contribute zero but retain their weight in the denominator.
This exact hyperedge expression is not a star-edge or pairwise cut proxy.
The producer supplies both the canonical `expected_cut` scalar and its aligned
per-net probabilities and weights. You may transform these values to construct
the training loss, including a different bounded emphasis on criticality or a
nonlinear penalty on cut probabilities. You cannot change the underlying net
universe, mutate its weights, recompute timing or change the trusted canonical
cut diagnostics. No per-net name or incidence data is exposed. A continuous
cut probability is not the final discrete cut indicator.

## Evaluation and reference

The inherited selection objective is setup timing after legal partitioning,
fixed concurrent DREAMPlace placement, PDN and track-aware greedy HBT
assignment. STA uses one connected two-die graph with placement-estimated
die-local resistance/capacitance, a passive HBT pi model and ideal pre-CTS
clocks. One fixed OPT pass follows HBT; CTS, global/detailed routing and
thermal evaluation are outside this endpoint. This is placement-estimated timing, not signoff.

Every instance must be assigned exactly once to an allowed tier, with no
supernode split. Top total area share must be within 0.5 +/- 0.03. With macros,
top macro-count and macro-area shares must each be within 0.5 +/- 0.05.
`area_balance_error` means `abs(top_area-bottom_area)/total_area`, so its limit
is 0.06. Infeasible integer macro counts or indivisible groups cause rejection;
these windows are not rounded or relaxed. Placement and HBT must validate.

Let `dW=max(0,-WNS)` and `dT=max(0,-TNS)` in nanoseconds.
For each case and seed, compute `(reference deficit - candidate deficit) / reference deficit`
for WNS and TNS separately. A zero reference deficit blocks scoring until
an official rule is frozen.

`combined_score = 100 * (0.4 * WNS_improvement + 0.6 * TNS_improvement)`

Contest scoring uses the common per-case WNS/TNS denominators in
`contest2/manifests/common-timing-reference.json`. Both initial GNNs average
0 across all five public cases; training and validation subsets can differ.
Native paired-reference scores remain diagnostic.

Higher is better. The score permits a WNS/TNS trade-off; neither is required
to improve individually. Cut, HPWL, probability confidence, training loss,
runtime and closure movement are diagnostics, not extra rewards. Missing or
invalid timing is a failed evaluation, never zero slack.

The default case is BP_FE/ng45/LoL, with fixed I/O seed 1, paired partition and placement seeds 1001, 1002 or 1003, density 0.60, macro halo 10 um and four physical
threads. A larger case/seed matrix requires its own explicit frozen inputs
and references; none is implied by this definition. Every candidate and its
reference use identical input/floorplan and frozen prototype timing data.

This is a distinct task: earlier loss-only global-clustering results and the
binary HPWL diagnostic are not its reference. A fresh reference must run the
exact initial program with this recipe and endpoint. A missing reference
cannot give an arbitrary candidate score 1. Source, image, task, program,
recipe, input and timing identities must be bound and independently verified.

## Using measured feedback

First inspect validity, the rejection reason, assignment completeness and the
three balance checks. Then compare `combined_score`, final joint WNS/TNS and
their normalized costs against the same-case reference. Use final cut,
probability uncertainty, raw hard-area balance and closure moved-area fraction
to explain whether learning survived decoding. Loss scales across different
programs are not comparable. An unchanged named assignment with a different
downstream timing result alone does not prove improved partitioning.

Selected examples are not chronological history. Change one main mechanism
at a time and retain useful parts of the selected parent. Treat suggested
regularizers as hypotheses, failures as failures and missing data as unknown.

## Program requirements

Return only the complete function with the exact signature inside
`# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`, with a short explanatory
docstring. Use at most 18,000 UTF-8 bytes and 3,000 AST nodes. No imports,
extra definitions, loops, comprehensions, recursion, external I/O, randomness,
mutable external state, input mutation, in-place operations, augmented
assignment, exception handling, backward or detach calls are allowed. Do not
shadow trusted helpers. Keep work/memory linear in N times feature width plus H;
do not construct dense N-by-N or N-by-net tensors.

Use the supplied `torch`/`F` tensor helpers and local arithmetic. Allowed
`torch` helpers are `abs`, `clamp`, `exp`, `isfinite`, `log`, `log1p`, `mean`,
`ones_like`, `sigmoid`, `softmax`, `sqrt`, `square`, `std`, `sum`, `tanh`,
`var`, `where` and `zeros_like`. Allowed `F` helpers are `cosine_similarity`,
`logsigmoid`, `mse_loss`, `normalize`, `relu` and `softplus`. Scalar helpers
are `abs`, `float`, `int`, `len`, `max` and `min`; do not extract tensor values
with them. Non-mutating tensor arithmetic, indexing and reductions are
allowed. Tensor `.backward`, `.bmm`, `.cpu`, `.cuda`, `.data`, `.detach`,
`.grad`, `.item`, `.matmul`, `.mm`, `.numpy`, `.requires_grad_`, `.to`, private
attributes and names ending in `_` are forbidden. Do not hide equivalent
operations under another spelling to bypass the contract.

Guard empty masks and small denominators. Use population variance for
singleton-safe statistics, valid logarithm/square-root domains and bounded
exponentials. All node areas and their sum are guaranteed positive. Candidate
exceptions, nonfinite or detached/nonscalar returns are rejected with no
baseline fallback. Native physical evaluation has no execution deadline. Each baseline has
10 evaluation submissions and an independent RMB 15 model API budget.

## Current availability

The native loss hook, trusted context producer and physical collector are
connected in source. Each case still requires a fresh native reference from
this task's exact initial program and fixed recipe before candidate timing can
be scored. With no qualified reference, only that exact initial program may
create the reference. Existing binary GNN runs, HPWL diagnostics and another
task's references cannot qualify it. Source connectivity and offline tests are
not native EDA qualification. The outer harness enforces 10 evaluation submissions and RMB 15 per baseline.

## Fixed contest sampling

The probability baseline evaluates BP_FE/ng45 and BP/ng45 at seeds 1001, 1002 and 1003, and Ariane133/ng45 only at seed 1003. These are seven required training cells. Both public validation cases retain all three seeds, for six cells. Compute the mean of required seed scores within each case, then the equal-weight mean of case scores. Ariane has the same case weight as BP and BP_FE. The outer harness rejects any missing required cell or replacement seed. This sampling policy is fixed before candidate execution and does not depend on whether a candidate passes.
