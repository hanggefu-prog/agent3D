# TP-GNN pin-to-pin cut fitness v1

Historical note: this task, its scorer, pin-export patch and input recipes have
been removed. Commands below document the former experiment and are not current
entry points. Use [the current task guide](tasks.md) for supported tasks.

Former task: `tp_gnn_pin_cut_learning_objective`. Historical net-cut tasks and
experiment outputs are unchanged. Do not resume old searches under this task.

## Definition

Each edge is one directed driver-pin -> sink-pin net connection in the frozen
2-D timing input. Count actual cross-die branches, not whole hyperedges or
clique-expanded pairs. Distinct sink pins on the same cell count separately.
Physical pin-pair duplicates are collapsed before normalization: select the
worst-slack observation, break ties by largest delay, and take BOTH values
from that observation. There is no per-net/fanout normalization.

`q = minmax(minmax(delay)-minmax(slack))` over unique frozen edges, with
constant-range normalization = 0.5. For candidate assignment z:

```
cut_e = 1 iff die(driver_instance_e) != die(sink_instance_e)
R = sum(q_e * cut_e) / sum(q_e)
P = sum((1-q_e) * cut_e) / sum(1-q_e)
S = R - P
fitness = valid * max(0, 1 + S - S_reference)
```

A zero denominator contributes zero. Same reference => fitness 1. Keep the
existing area (0.06), macro-count (0.05), macro-area (0.05), complete assignment,
trained TP-GNN and TA3D-decoder gates. No new soft area penalty is introduced.
Compare within a fixed case/input only; `S-S_reference` is an absolute signed
score gain, NOT a percentage improvement of S. The score clipping is inherited
from the legacy signed-cut task (sufficiently bad candidates may tie at zero).

The evaluator recomputes weights from the reference TSV and cut indicators
from the final canonical `partition_assignment.csv`. Candidate and reference
TSV hashes and assignment identity sets must match. Legacy net scores remain
diagnostics only. Missing pin IDs, missing endpoints, nonfinite timing and
inconsistent assignments fail closed. No fallback to instance-pair/net counts.

## Input preparation and execution

The old extractor's TSV lacks pin names. It cannot be unambiguously upgraded
from cell-pair names, so fresh input preparation is required. Apply
`local_bench/evaluator_patches/tp_gnn_pin_timing.patch` ONLY to a new isolated
Open3DBench source checkout, using `git apply --unidiff-zero` (or the existing
`local_bench.workspace.apply_candidate_patch` helper). Commit/fingerprint the
result before preparing inputs. The patch merely appends full driver/sink pin
names; the old net-weight extractor and decoder keep their behavior.

Profiles for bp, bp_multi, ariane133 and swerv_wrapper are named
`<case>_lol_ng45_tp_gnn_pin_cut_v1`. Unlike historical TA3D profiles, they do
not point to an old absolute `partition.timing_input`: the patched extractor
must produce fresh timing input. Reuse only input/floorplan stages if seeding
preparation; never adopt an old partition stage without pin identities.

Using the maintained runtime/environment, from the framework root:

```sh
python -m local_bench prepare-input bp_lol_ng45_tp_gnn_pin_cut_v1 \
  --open3dbench-root /path/to/new-patched-source \
  --output-dir /path/to/new-input-bundle
python -m local_bench validate-input /path/to/new-input-bundle \
  --task tp_gnn_pin_cut_learning_objective
python -m local_bench export-openevolve tp_gnn_pin_cut_learning_objective \
  --benchmark-profile bp_lol_ng45_tp_gnn_pin_cut_v1 \
  --open3dbench-root /path/to/new-patched-source \
  --output-dir /path/to/new-search-bundle
```

Export a NEW search bundle using that task/profile and the same fresh run for
baseline/reference. The generic exporter supplies the maintained initial loss
and new prompt. Do not use the legacy TA3D launcher, which hard-codes its old
task and reference semantics. This change does not launch EDA or API calls.

## Scope and limitations

This is an evolutionary selection metric; the trainable function remains
`compute_tp_gnn_loss`. The TA3D decoder's internal net objective and selection
of Cannot-Link pairs are unchanged, so this is NOT a claim that every internal
partition operation now uses pin-edge weights. Global STA scoring is unchanged.

The frozen edge universe contains timed inter-cell connections exported by
OpenSTA. It is not all logical netlist edges, primary-port connections, cut-net
count, cut-pin count (pins on a cut hyperedge), or HBT count. Multiple drivers
are represented by observed directed driver/sink pairs, not a proof of active
signal propagation for every pair. The inherited delay estimate is the
nonnegative difference of max arrival values at sink and driver; it is not a
new path-consistent arc delay extractor. The weights predict importance, not
the actual 3-D timing benefit of a cut. Report WNS/TNS separately after replay.
