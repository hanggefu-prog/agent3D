# Evolve the TP-GNN loss and clustering decoder

The native loss/decoder hooks, trusted context producer and physical collector
are connected in source. Before candidates can be scored, each case requires
a fresh reference from this task's exact initial program and fixed recipe.
With no qualified reference, only that exact initial program may create the
reference. Another task's receipts cannot qualify it. Source connectivity and
offline tests do not establish native EDA qualification or authorize evolution.

## Goal and fixed experiment

A logic-on-logic three-dimensional chip places logic and eligible memory
macros on two dies. The maintained legacy TP-GNN builds a flat netlist graph,
contracts connected standard cells into supernodes, and learns L2-normalized
embeddings with a two-hop GNN. Macros remain singleton nodes. Positive
training pairs come from fixed random walks; negative pairs use the fixed
legal sampling procedure. This task jointly evolves the embedding loss and
the whole-design two-way clustering algorithm that turns embeddings into
die labels. There are no spatial bins.

The selection objective is measured setup timing after a legal partition,
fixed placement, PDN and HBT assignment. The evaluator builds one connected
two-die timing graph with ideal pre-CTS clocks, placement-estimated die RC
and passive HBT pi models. One fixed OPT pass follows HBT; CTS, routing and thermal are outside this endpoint.
Training differentiates through the loss only; clustering, feasibility
closure, placement, HBT assignment and STA are not differentiable training
operations. Evolution evaluates the complete loss-and-decoder program by
the resulting measured timing score.

Input/floorplan and complete partition-independent timing inputs are frozen
for every candidate. Their hashes, design, platform, native image, seeds
and physical methods form the experiment identity. The reference and every
candidate must use the same identity. Existing loss-only task receipts and
the separate binary-GNN experiment are not references for this new task.

The frozen `tp_gnn_fast.json` recipe requests 20 epochs, at least 10 epochs
and at least 100 optimizer steps under its unchanged minimum-step policy.
The effective epoch budget may exceed 20. Contraction, features, GNN,
normalization, sampling, optimizer, learning rate, device, seeds and stopping
policy cannot change. The decoder receives a host-owned limit of 50
iterations. The evolution configuration's round limit is a separate control
and does not enlarge the decoder or training budget.

## Editable program

Return exactly these two functions, in this order, in one complete program:

1. `compute_tp_gnn_loss` with the exact supplied timing-context-v2 signature.
2. `decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target)`.

Change either or both bodies while retaining useful parts of the selected
parent. Explain the intended mechanism briefly in each docstring. You may
coordinate loss geometry with clustering distance, centers or ranking.
You may not change any other component, source file, configuration, helper,
score, resource limit or hard constraint. Raw area-weighted k-means remains
a fixed diagnostic; it does not determine the handoff assignment.

## Loss interface

The loss returns one finite scalar `torch.Tensor` connected through autograd
to the supplied scores or embeddings. Inputs are read-only. `torch` and
`F` are supplied numerical helpers; do not import modules.

| Inputs | Shape and meaning |
| --- | --- |
| `positive_scores`, `negative_scores` | `[P]`, `[Q]` differentiable pair dot products |
| `embeddings` | `[N,D]` differentiable L2-normalized group embeddings |
| `positive_sources`, `positive_targets` | `[P]` supernode indices |
| `negative_sources`, `negative_targets` | `[Q]` supernode indices |
| `source_batch_size` | Number of source nodes in this training batch, not P, Q or N |
| `epoch`, `max_epochs` | Zero-based epoch and positive effective epoch budget |
| `graph_scores` | `[E]` differentiable star-edge dot products with fixed-order gradient accumulation |
| `graph_sources`, `graph_targets`, `graph_net_ids` | `[E]` endpoints and original net IDs |
| `graph_weights` | `[E]` fixed net weight divided by its number of star edges |
| `net_weights` | `[H]` fixed timing weights for all original nets |
| `node_features` | `[N,10]` unstandardized timing and structural group features |
| `node_areas` | `[N]` physical group area, including standard cells and macros |

The timing-context-v2 graph includes each original net connecting at least
two distinct supernodes as a star rooted at the smallest supernode ID.
The anchor is arbitrary, not its driver. Duplicate pins/groups are removed,
parallel nets remain distinct, and the loss-context stars are not truncated
by the GNN message-graph degree limit. Internal nets contribute no star.
For a represented net, its edge weights sum to its net weight. A star dot
product penalty is a proxy, not an exact hyperedge cut probability. Sampled
pairs do not correspond by position to these graph edges. Use supplied
`graph_scores` rather than regathering repeated embeddings to recreate them.

Frozen weights use maximum connection delay and minimum connection slack:
`omega=minmax(minmax(delay)-minmax(slack))`, `weight=1+alpha*omega`.
Nets without timing data have unit weight, but unit weight alone does not
indicate missing data. Timing describes a two-dimensional prototype before
partitioning, not the final candidate timing. The downstream placer does
not inherit prototype coordinates.

Feature columns are: worst slack, constrained-pin-weighted mean slack,
maximum slew, maximum arrival, minimum required time, summed cell delay,
timing-coverage fraction, distance-to-source hops, one-hop degree and
two-hop degree. Coverage zero marks unknown timing; zero-filled fields then
do not mean zero slack or delay. Normalize mixed units appropriately.
Macro masks, macro-only area and eligibility flags are not loss inputs;
do not infer them from total group area or memorize supernode indices.

Loss code has no loops, comprehensions, nested definitions, exception
handling, randomness, input mutation, in-place or augmented assignments,
external state, I/O, tensor detach/backward calls or device changes. Allowed
`torch` helpers are `abs`, `clamp`, `exp`, `isfinite`, `log`, `log1p`,
`mean`, `ones_like`, `sigmoid`, `softmax`, `sqrt`, `square`, `std`, `sum`,
`tanh`, `var`, `where`, `zeros_like`. Allowed `F` helpers are
`cosine_similarity`, `logsigmoid`, `mse_loss`, `normalize`, `relu`,
`softplus`. Scalar Python helpers are `abs`, `float`, `int`, `len`,
`max`, `min`. Non-mutating tensor arithmetic, indexing, shape access and
reductions remain subject to the trusted AST validator. Do not use
`backward`, `bmm`, `cpu`, `cuda`, `data`, `detach`, `grad`, `item`,
`matmul`, `mm`, `numpy`, `requires_grad_`, `to`, or attributes ending in `_`.
Guard empty graph edges, masks, singleton variance, zero denominators and
log/square-root domains. The loss runs for every training batch; do not
multiply full-graph work by avoidable copies or build dense pairwise arrays.

## Decoder interface and protected feasibility closure

The decoder runs after training and receives a read-only finite NumPy
`embeddings[N,D]`, read-only physical `node_areas[N]`, and scalar `seed`,
`max_iterations`, `balance_target`. It returns exactly N integer labels:
**1 means top, 0 means bottom**. Both sides must be nonempty. It receives
no names, macro flags, netlist, timing features, final STA measurements or
mutable configuration. It must not mutate arrays, use external IDs, or
infer macro identity from area. The host validates the entire output before
any physical handoff and rejects malformed or nonfinite results.

The initial program calls the frozen pure helper
`reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)`.
It reproduces maintained `area_balanced_weighted_kmeans`: seeded two-center
initialization, physical-area-weighted mean centers, distance-preference
ranking and deterministic whole-group nearest-area rounding. This helper
cannot be edited. It is the only permitted seeded randomness; `np.random`,
additional RNGs and extra reference restarts are forbidden.
The helper may be called once with exactly the unchanged five inputs, with no
iteration loop in the same decoder. Alternatively, implement one custom
bounded loop without calling the reference helper.

Candidate decoder code may replace clustering using pure NumPy vector
operations. The only permitted loop form is `for ... in range(max_iterations)`;
there are no nested loops, recursion, comprehensions or unbounded loops.
Non-mutating arithmetic, indexing, reductions, ordering and selection are
allowed under the trusted numerical API validator. No `np.linalg`, files,
external solvers, imports, exception handling, external state, in-place
updates or input mutation. Keep memory linear in N*D; do not build N-by-N
distances, dense node-by-net arrays or clique graphs. The loop bound is
host-owned; do not replace it with a constant or multiply it.
`np.cumsum` is available for linear-memory cumulative-area decisions. Output
arguments, including positional or keyword `out`, are forbidden.

The candidate returns labels only, not centers, affinities or constraint
receipts. The protected wrapper recomputes both centers as physical-area-
weighted means from the returned labels, then uses the frozen native
`two_center_affinities` squared-distance margin. Empty sides and nonfinite
centers are rejected. Custom clustering centers cannot override this
closure objective.

If total-area, macro-count and macro-area checks already pass, the assignment
stays unchanged. Otherwise, the fixed native macro feasibility search and
standard-cell area rounding apply these protected affinities. Macros remain
singleton nodes and standard-cell supernodes remain indivisible. No cut or
timing refinement is hidden in closure. The candidate cannot change or
bypass this repair, its gates or the group membership. Any final violation
is rejected, without fallback, label swapping or relaxed tolerances.

Every original instance must appear exactly once on an eligible die. Top
area share must lie within `0.5 +/- 0.03`; macro count and macro area share
must each lie within `0.5 +/- 0.05` when macros are present. These windows
are exact and may be infeasible for integer macro counts or indivisible
supernodes. The reference must also pass them. Placement and HBT must
validate before timing is scored.

## Objective and feedback

Let `dW=max(0,-WNS)` and `dT=max(0,-TNS)` in nanoseconds.
For each case and seed, compute `(reference deficit - candidate deficit) / reference deficit`
for WNS and TNS separately. A zero reference deficit blocks scoring until
an official rule is frozen.

`combined_score = 100 * (0.4 * WNS_improvement + 0.6 * TNS_improvement)`

Higher is better. The combined score permits a trade-off; it does not
require WNS and TNS to improve separately. Contest scoring uses the common per-case WNS/TNS denominators in
`contest2/manifests/common-timing-reference.json`. Both initial programs
average 0 across all five public cases; training and validation subsets
can differ. Native paired-reference scores remain diagnostic. Lower cut, HPWL, training loss, runtime or repair area is
diagnostic evidence, not an additional reward. These measurements are
pre-CTS and placement-estimated, not signoff timing.

First read acceptance, failure reasons, partition completeness and hard
constraints. Then compare `combined_score`, WNS/TNS and their normalized
costs against the same-contract reference. Diagnose changes with final cut,
timing-weighted cut, decoder labels and closure movement. Loss magnitudes
need not be comparable across evolved objectives. A lower loss or better
raw clustering does not establish better final timing. An unchanged final
named assignment with different timing is not by itself evidence of a
partition improvement. Missing metrics remain unknown, never zero.
Selected historical examples are not chronological. Retain useful parent
mechanisms and test one main hypothesis at a time.

## Submission limits and output

The complete program is limited to 18,000 UTF-8 bytes and 3,000 AST nodes.
The evaluator timeout is 21,600 seconds for the entire physical evaluation,
not extra training or decoder time. No memory, GPU, thread or retry budget
is added by this task. The trusted adapter packages one
`loss_decoder_program.json` with interface `tp-gnn-loss-decoder-v1` for the
protected `partition.evolution_program` input; direct source patches and
configuration overrides are forbidden.

Return only both complete function definitions in the declared order,
between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`. Include a brief
docstring for each. Do not return imports, JSON, diffs, tests, additional
functions or prose outside the markers.
