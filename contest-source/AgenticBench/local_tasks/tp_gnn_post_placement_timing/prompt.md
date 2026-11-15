# Improve the training loss for chip partitioning

## Background

A three-dimensional (3D) chip places circuitry on stacked tiers. In a
logic-on-logic (LoL) design, both tiers contain logic. Partitioning decides
which tier each cell and eligible memory macro belongs to. Connections
between tiers use hybrid-bonding terminals (HBTs), so this decision affects
inter-tier connections, timing and how easily the design can be routed.

This task uses tier partitioning with graph neural networks (TP-GNN). It builds a
graph from the netlist, groups connected cells into supernodes, and gives each
supernode structural and timing features. A two-layer graph neural network (GNN)
learns a vector representation, called an embedding, for each supernode from
sampled pairs. Positive pairs come from random walks; negative pairs are sampled
uniformly from the allowed nodes. The embeddings are L2-normalized (scaled to
unit length), then decoded by area-weighted global 2-way clustering over the
whole design. A final feasibility step enforces the global area and macro
balance constraints. There are no spatial bins or per-bin balance constraints
in this task.
This version uses a flat netlist, without hierarchy labels.

## Your task

Improve `compute_tp_gnn_loss`, the loss used to train the GNN. A useful loss
should help the later clustering and repair steps produce a better legal
partition, not just a lower training loss or more separated embeddings.

The goal is to maximize the combined setup-timing score defined below, after
the legal partition has been placed and HBTs have been assigned. The score
allows a trade-off between worst and total violations; improving it does not
necessarily improve both. Static timing analysis (STA) uses
one connected two-die timing graph with placement-estimated resistance and
capacitance, ideal clocks and a passive HBT model. There is no timing
optimization, clock-tree synthesis or routing in this evaluation.

Only the loss function may change. Graph construction, features, GNN architecture,
sampling, optimizer, learning rate, seed, training budget, global 2-way decoding,
repair rules, downstream placement and HBT methods stay unchanged.

## How embeddings become a partition

There are three separate results, all using the same trained embeddings:

1. **Raw clustering, for diagnosis only.** Ordinary area-weighted two-center
   k-means assigns each supernode to its nearest center in squared Euclidean
   distance. Area weights affect the centers, but do not guarantee balanced
   tier areas. This result is recorded, not passed to placement.
2. **Area-balanced clustering, used by the global 2-way decoder.** This starts
   from the same seeded pair of embeddings, not from the raw labels. At each
   iteration, it ranks supernodes by `distance_to_bottom^2 - distance_to_top^2`
   and scans that order. A whole supernode is selected for top when adding its
   area does not increase the absolute error from the target top area. Centers
   are then recomputed as area-weighted means. The assignment is repeated until
   labels stop changing or 50 iterations are reached. Embeddings have unit
   length, but these mean centers are not renormalized. Label 1 is top and label
   0 is bottom; there is no later timing-based swap of the two tiers.
3. **Final feasibility repair.** If all balance checks already pass, the
   assignment stays unchanged. Otherwise, macro count/area combinations are
   searched for a feasible assignment with minimum area-weighted squared-distance
   cost to the balanced partition's centers. Standard-cell supernodes are then
   ranked by their distance preference and rounded toward the remaining top
   area after macro load. Macros are singleton nodes; standard-cell supernodes
   are never split. An assignment that still fails a balance check is rejected.

Neither area rounding nor final repair uses a cut or timing objective. Both can
change which connections cross tiers, so a promising embedding or raw cut may
not survive these steps. An area-weighted embedding mean near zero alone does
not guarantee balanced clustering. This LoL task lets cells and macros move
between both tiers; it does not give the loss a fixed-tier or eligibility mask.

## How solutions are evaluated

Every instance must be assigned exactly once to an allowed tier. Training must
be valid. The global top-tier area share must be within 0.5 +/- 0.03.
The top-tier share of macro count and macro area must be within 0.5 +/- 0.05
when macros are present. Placement and HBT validation must succeed before
timing is scored. There is no implicit rounding or relaxation of these windows;
integer macro counts or indivisible supernodes can make them infeasible. A
reference is usable only after it passes the same checks for the selected case.

Setup worst negative slack (WNS) describes the worst setup violation; total
negative slack (TNS) sums setup violations over endpoints.
Let `dW=max(0,-WNS)` and `dT=max(0,-TNS)`, both measured in nanoseconds.
For a reference deficit greater than 1e-12, its normalized cost is candidate/reference.
If reference and candidate deficits are both at most 1e-12, the cost is 1.
If only the reference deficit is at most 1e-12, the cost is
`1 + candidate_deficit/1e-12`.

`combined_score = 1 / max(0.6 * WNS_cost + 0.4 * TNS_cost, 1e-12)`

Higher is better. For example, reference WNS/TNS of -0.10/-10 ns and candidate
values of -0.08/-12 ns give costs 0.8/1.2 and a score of about 1.0417: the worse
TNS is accepted by this particular trade-off. Inspect both metrics when judging
an improvement. If both reference and candidate have no violation, the cost
stays 1; extra positive slack earns no reward. If both reference deficits are
above 1e-12 and the candidate removes both, the score reaches 1e12.

The validated same-case native loss with the same global 2-way decoder and
downstream flow scores 1. Timing-weighted cut, training loss, embedding
statistics and runtime are diagnostics, not additional rewards. Read rejection
reasons alongside the score; missing or invalid timing is a failed evaluation,
never zero slack. These are pre-clock-tree, placement-estimated results, not
signoff timing.

## Improving the loss

Possible approaches include:

- Using timing features to weight positive pairs, or graph scores and timing
  weights to encourage nodes on important connections to stay similar.
- Using node areas or bounded regularization to preserve non-degenerate
  structure that is useful for two-way clustering, rather than spreading every
  embedding uniformly over the unit sphere.
- Introducing extra loss terms gradually, or reducing strong pair weighting
  toward the end of training.

Sampled negatives are not opposite-tier labels. Hard-negative weighting is
another hypothesis to test, not an instruction to put those pairs on different
tiers. Judge it by the final decoded and repaired partition.

These are suggestions, not required ingredients. Try one main change at a time.
You may combine ideas that have already improved comparable legal partitions.
Keep useful parts of the selected program; returning a complete function does
not mean rewriting everything. Explain briefly in the docstring what you changed
and why it could improve the final partition.

## Using previous results

Compare solutions evaluated under the same settings and passing the same checks.
Prefer changes supported by better final WNS/TNS fitness; a lower cut alone
is not evidence of improved timing. The examples
provided by OpenEvolve are selected programs, not a chronological history.

Read the supplied feedback in this order:

1. Check acceptance, `runs_successfully`, `partition_complete`,
   `partition_constraints_passed` and the rejection reason when provided. A
   code or execution failure is not a measured timing result.
2. Inspect `combined_score`, `final_global_wns_setup_ns`,
   `final_global_tns_setup_ns`, `normalized_wns_cost` and `normalized_tns_cost`.
   The costs compare against the same-case reference. Raw reference WNS/TNS
   are not separate fields in the current feedback; do not invent them or
   reverse-engineer them when a zero-deficit branch makes that ambiguous.
3. Read the partition diagnostics: final cut-net count, timing-weighted cut
   and timing-weighted cut ratio. In decoder/repair diagnostics, compare raw,
   balanced and feasible timing-weighted cuts. Each weighted-cut delta is
   after minus before; relative changes divide by the before value and are
   omitted when it is zero. `tp_gnn_repair_moved_supernodes` counts groups
   whose tier changed from balanced to feasible assignment;
   `tp_gnn_repair_moved_area_fraction` is their original instance area,
   including macros, divided by total instance area. This is not geometric
   displacement or movement during initial area balancing.
4. Use final training loss, completed epochs and embedding variance to explain
   behavior, not as extra rewards. Loss magnitudes need not be comparable
   across different objectives. Positive-pair and negative-pair cosine medians,
   when available, use final embeddings on up to 4096 evenly spaced actual
   pairs of each class from the last completed training batch. Sample counts
   accompany them; they are not whole-graph statistics or tier labels.

Selection target: `combined_score` only. Cut/loss/embedding metrics are
diagnostic only. Feedback is grouped into fitness, validity, partition,
decoder/repair, training and status sections. Fixed counts, successful timing
graph checks, optimizer-step counts and normal training time are not repeated
in this compact feedback; failures still carry an explicit reason. Optional
diagnostics are included only when measured; older results may lack repair
area and separate positive/negative pair statistics.

Missing metrics remain unknown, never zero. A score of zero alone does not
identify the failure cause. A worse cut is not sufficient to reject a legal
candidate with better measured timing fitness.

If code is rejected, use the error to fix it; that result alone does not show
whether the underlying idea would work. A valid solution with worse final timing
does provide evidence against that particular change. Do not return an evaluated
program unchanged.

## Function inputs

Keep the exact signature shown in the supplied program. In each evolution
round, edit the current selected parent, not automatically the initial loss;
the initial program is included as an interface and baseline example.
`N` is the number of supernodes, `D` the embedding dimension,
`H` the number of original nets and
`E` the number of graph edges supplied to the loss. Node IDs use the same
supernode indexing throughout.

| Input | Shape and meaning |
| --- | --- |
| `positive_scores`, `negative_scores` | `[P]`, `[Q]` differentiable dot products for sampled pairs |
| `embeddings` | `[N, D]` differentiable, L2-normalized embeddings |
| `positive_sources`, `positive_targets` | `[P]` positive-pair node IDs |
| `negative_sources`, `negative_targets` | `[Q]` negative-pair node IDs |
| `source_batch_size`, `epoch`, `max_epochs` | Python integers: source count, zero-based epoch, effective epoch budget |
| `graph_scores` | `[E]` differentiable star-edge dot products with fixed-order gradient accumulation |
| `graph_sources`, `graph_targets`, `graph_net_ids` | `[E]` endpoints and original net IDs |
| `graph_weights` | `[E]` net weight divided by the number of star edges for that net |
| `net_weights` | `[H]` timing weights for all original nets |
| `node_features` | `[N, 10]` unstandardized group features, described below |
| `node_areas` | `[N]` summed cell and macro areas |

### Where the timing inputs come from

Timing features and net weights come from placement-estimated STA on a
partition-independent two-dimensional (2D) prototype used for floorplanning and
external input/output (I/O) planning, before any tier assignment.
They are held fixed for every candidate. They are not measurements of the
candidate's 3D placement, and the downstream placer does not inherit the
prototype's cell coordinates. The candidate's final two-die STA is evaluated
separately after placement and HBT assignment.

Critical connections can change after partitioning and placement. These input
aggregates can guide a loss, but cannot reconstruct final WNS/TNS. Training
differentiates through the loss, not through clustering, repair or final STA;
the evolution system selects loss programs using their measured final score.

### Graph connections and weights

A net connecting `k >= 2` distinct supernodes is represented by `k-1` edges
from its smallest-ID node to the others. This node is an arbitrary anchor,
not a driver. Duplicate pins and groups are removed; parallel nets remain
separate. Every such net is included, even if its degree exceeds the GNN's
message-graph limit. Nets contained within one supernode have no edges.

For each represented net, its `graph_weights` sum to its `net_weights` value.
Use `graph_scores` to encourage related nodes to stay similar. A score is
not a cut probability or exact contribution to the final hyperedge cut.
Do not count each star edge as an independently scored net.
For example, with one node opposite the other `k-1` nodes and dot products of
+1 within a cluster and -1 across clusters, `sum(graph_weights * (1-graph_scores)/2)`
is the full net weight if the lone node is the anchor, but only
`net_weight/(k-1)` if the anchor is on the majority side. The true hyperedge is
cut in both cases. This penalty is therefore a proxy, not an exact cut counter.

Normally use `graph_weights` directly. You can use `net_weights` for global
statistics or index it with `graph_net_ids`. No scatter/segment helper or full
net-to-node incidence tensor is supplied, so do not assume arbitrary per-net
reductions are available. Sampled positive pairs can span several hops and do
not correspond directly to individual nets; graph weights cannot be matched
to sampled scores by array position.

Weights are supplied, not recomputed by the loss. They use maximum connection
delay and minimum connection slack:
`omega = minmax(minmax(delay) - minmax(slack))`,
`weight = 1 + alpha * omega`. Nets without timing data have unit weight,
but unit weight alone does not mean timing data is missing.

### Node features

- Timing: columns 0 worst slack, 1 constrained-pin-weighted mean slack,
  2 maximum slew, 3 maximum arrival, 4 minimum required time, 5 summed cell delay.
- Timing coverage: column 6 is the fraction of members with observed timing.
  At zero coverage, zero-filled values mean unknown, not zero slack or delay.
- Structure: columns 7 distance-to-source hops, 8 one-hop degree, 9 two-hop degree.

These are group aggregates, not per-pin measurements or new static timing
analysis (STA) results.
Timing values retain their original units. Normalize features before combining
different quantities. The GNN's own standardized inputs stay unchanged.
Indices are `torch.long`; floating tensors share the training dtype and device.
Features, areas, indices and weights are constants.
Macro counts, macro-only areas and tier-eligibility attributes are not loss
inputs. Do not infer them from total node area. Macro balance is enforced by
the decoder/repair, not by an unavailable macro mask in the loss.

## Implementation requirements

Return one finite scalar `torch.Tensor` that remains connected to the scores
or embeddings through autograd. Training requests 20 epochs and at least 100 optimizer steps;
the effective epoch count follows the unchanged minimum-step
policy and may exceed 20.

- Keep the exact signature. No imports, I/O, external state,
  randomness, input mutation, in-place operations, augmented assignment,
  comprehensions, exception handling or backward calls. Local numerical helpers
  and literal `range` loops with at most eight steps are allowed; nested loops
  including helper calls are limited to 64 iterations. No recursion.
  Intermediate `.detach()` is allowed, but the returned loss must remain
  differentiable through the supplied scores or embeddings.
  Do not shadow provided helpers or memorize benchmark-specific IDs.
- Use the supplied `graph_scores` for graph penalties rather than gathering
  repeated embeddings to recreate them.
- Keep work and memory linear in sparse edges, sampled pairs and `N*D`.
  Do not build dense `N x N` or `N x H` tensors.
- Stay within 18,000 characters and 3,000 abstract syntax tree (AST) nodes.
  The resulting source patch may change one file, add at most 360 lines and
  delete at most 12 lines, including the adapter's fixed integration code.
  Keep the function compact; these are not 360 freely editable function lines.
- The maintained runtime uses Python 3.8 and Torch 1.7.1. The adapter supplies
  `torch.nan_to_num` for real tensors without modifying global Torch state.
  Use the module function, not a newer tensor `.nan_to_num()` method. Other
  APIs must exist in the installed runtime; accepting syntax is not execution.
- Allowed `torch` helpers: `abs`, `clamp`, `exp`, `isfinite`, `log`, `log1p`,
  `mean`, `ones_like`, `sigmoid`, `softmax`, `sqrt`, `square`, `std`, `sum`,
  `tanh`, `var`, `where`, `zeros_like`, `amin`, `amax`, `argmax`, `argmin`,
  `argsort`, `cat`, `cos`, `cumsum`, `expm1`, `flatten`, `gather`, `index_select`,
  `logsumexp`, `median`, `nan_to_num`, `norm`, `numel`, `pow`, `prod`, `quantile`,
  `relu`, `reshape`, `sin`, `sort`, `stack`, `transpose`, `unsqueeze`.
  Allowed `F` helpers: `cosine_similarity`, `logsigmoid`, `mse_loss`,
  `normalize`, `relu`, `softplus`, `l1_loss`, `smooth_l1_loss`, `log_softmax`,
  `tanhshrink`.
  Allowed Python helpers: `abs`, `float`, `int`, `len`, `max`, `min`, `range`,
  plus your local helpers. Do not use scalar helpers to extract differentiable
  tensor values. Helpers stay inside the returned function and use plain
  arguments; no decorators, aliases, further definitions or recursion.
  Non-mutating tensor arithmetic, indexing, `.shape` and epoch conditionals
  are allowed. Tensor methods use a denylist, not the module helper list.
  Forbidden tensor attributes are `backward`, `bmm`, `cpu`, `cuda`, `data`,
  `grad`, `item`, `matmul`, `mm`, `numpy`, `requires_grad_`, `to`, and names
  ending in `_`. Calls cannot write an `out` tensor. Do not bypass the
  no-mutation, no-external-state or linear-memory requirements.

The configured evaluator timeout is 21,600 seconds for evaluation, not a
separate allowance for the loss. The loss shares the unchanged training and
downstream resource budget; avoid extra graph-sized copies and expensive work
on every batch. No additional memory allowance is available to a candidate.

### Numerical safety

Batches contain positive and negative samples, but `E` can be zero and a
candidate-created mask can select zero or one item. Guard empty reductions and
denominators. Use population variance for singleton-safe statistics. Mean-absolute
or root-mean-square (RMS) normalization and bounded transforms such as `tanh`
can keep terms on comparable scales without design-specific thresholds.

Keep log, square-root and ratio inputs in valid ranges, and avoid exponential
overflow. Use Python arithmetic for scalar schedules. A full-graph penalty is
applied every batch; normalize it relative to sampled terms so larger graphs
do not overwhelm the rest of the loss.

### Reading and using the arguments

The loss is called for each sampled training batch, not once per evolution
round. Pair counts P and Q describe this batch; N, H and E describe the full
supernode/net graph. All arguments are read-only. Only the returned scalar and
its gradient are candidate-owned.

| Argument family | Units, ordering and missing cases |
| --- | --- |
| Pair scores and embeddings | Dimensionless dot products/unit embeddings. Pair indices select rows in the same supernode embedding matrix. No pair label is a target tier assignment. |
| `source_batch_size` | Integer number of source nodes used for this batch's sampling; not P, Q or N. Preserve the distinction when normalizing. |
| `epoch`, `max_epochs` | Zero-based current epoch and positive effective epoch budget; neither is an optimizer-step count. |
| Graph indices and weights | Edge arrays share the same E order; `graph_net_ids` indexes the H-entry net array. Weights are dimensionless and fixed. E can be zero. |
| `node_features` | Columns use the ordering listed above. Structural counts/hops and coverage are dimensionless; timing columns retain the frozen input's time scale, not a [0,1] normalization. Zero coverage marks missing timing. |
| `node_areas` | Nonnegative physical area sums per supernode, including macros; use relative area ratios rather than assuming a coordinate conversion. No macro-only mask is provided. |

Feedback now compares the final named instance assignment with the reference:
`partition_assignment_identical=1` means the partition did not change.
A higher downstream timing score with an identical assignment is not by itself
evidence that the loss improved partitioning. No tier-label swap is applied in
this comparison; bottom and top retain their physical meaning.

### Feedback field guide

The supplied `metric_definitions` gives a short meaning and unit for each returned
metric. Missing metrics are unknown, not zero. Only `combined_score` is the
selection objective; activity and progress measurements explain behavior but
are not additional rewards.

## What to return

Return only the complete `compute_tp_gnn_loss` function between the exact lines
`# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`, with a short docstring explaining
the change. This is full-function output, not a patch; the adapter inserts it
into the unchanged training loop. Do not include imports,
JavaScript Object Notation (JSON), diffs, tests, top-level extra definitions or text outside
the function and markers.
