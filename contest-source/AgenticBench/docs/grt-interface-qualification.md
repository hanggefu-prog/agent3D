# Bounded routing-source interface qualification

This is a diagnostic qualification, not an evolution search, harness candidate
final, registry publication, performance improvement, DRT run, or signoff. Old
net-ordering attempts remain immutable. Net ordering is excluded from the new
three-task batch; its default in the legacy CLI remains only for compatibility.
Always pass an explicit task in a new batch.

| Explicit task | Identity | One fixed nonidentity probe |
| --- | --- | --- |
| `global_routing_ripup_policy_source` | Native rip-up decision | Invert the computed edge decision with `1 - baseline_decision` |
| `global_routing_maze_cost_source` | Multiplier 1 | Add a 10% cost multiplier only on turns |
| `global_routing_iteration_controller_source` | Native congestion/search/rip-up controls | Scale congestion cost by 2.0, preserving search scale and rip-up offset |

The probes deliberately test hook behavior; none is advertised as a useful
optimization. For each task, the identity must pass before its one probe is
built or evaluated. A failed or uncertain attempt blocks the later probe and
is never retried by this launcher. Use a new explicitly authorized output only.

```sh
python3 "$SOURCE/AgenticBench/scripts/run_archive_grt_qualification.py" \
  --root "$ROOT" --source "$SOURCE" --archive "$ARCHIVE" --case bp_fe \
  --task global_routing_ripup_policy_source \
  --output "$ROOT/runs/qualification-ripup-UNIQUE"
```

Use the other two task IDs with separate fresh outputs. Run only from a clean,
frozen checkout whose pinned OpenROAD submodule is populated. The existing
shared CPU qualification lock serializes pairs on one host. Independent hosts
may run different tasks after their existing resource admission passes.

## Frozen execution and proof requirements

- Authenticate the existing archive inventory, successful prefix manifests,
  CTS checkpoint chain, all input sizes/hashes, projected SDC/ports, external
  Liberty files and suppression policy. Import independent copies without
  mutating or relabeling historical manifests.
- Allow exactly the selected native hook source file. Wrapper/config patches
  are refused. Build through the maintained bounded compiler-cache builder,
  then verify the image-installed binary, source fingerprint, pinned image and
  candidate receipt. A successful build is not interface qualification.
- Preserve 4 physical threads, 4 GRT congestion iterations, zero resource
  reduction, NDR preserve on both dies, seed -1, 10,800-second die timeout and
  21,600-second outer timeout. The unchanged DRT maximum of 5 is not a GRT
  tuning parameter; only GRT is executed.
- Validate the new native GRT publication and artifacts independently. Both
  dies must produce valid overflow and wirelength metrics. At least one die
  must execute the selected hook; zero-overflow dies may skip it.
- Identity requires observed native hook calls, zero changed decisions, and
  exact reproduction of the archived raw total overflow and wirelength.
- Probe requires observed calls and a nonzero native changed-decision signal.
  Mere policy source presence, process exit 0 or changed model text does not
  qualify. A metric improvement is not required and is not claimed.
- The iteration pair additionally matches each die's first controller
  invocation and requires actual integer `cost_height` to double, with the
  search window and rip-up threshold unchanged. `costheight_` is an integer:
  a 1.1 scale on 9 would truncate back to 9 despite a `changed=1` diagnostic.
  Therefore the fixed probe uses 2.0, and a scale-change flag alone cannot pass
  the pair. This check retains the matched raw log hashes in the pair receipt.
- The existing 90% wirelength / 10% overflow objective remains unchanged.
  Bootstrap scores without a qualified same-contract reference are omitted
  from qualification results; raw physical metrics remain authoritative here.

Counters printed periodically by rip-up/maze are observed lower bounds, not
final total invocation counts. The iteration `policy_changed` aggregation is
per-die presence of a changed decision, not a cumulative iteration count.
Preserve raw logs for interpretation. Model/API time is not measured because
these tests make no model calls. Build, evaluation-container dispatches, known
and unknown native attempts, diagnostic programs, qualification passes/failures
and successful qualified evaluations are separately accounted. Harness finals
and harness verifier rejections remain zero.

## Software tests

Run `tests/test_qualify_archive_grt.py` and
`tests/test_run_archive_grt_qualification.py`. The former includes actual
identity/probe patch materialization against the pinned native sources when
that submodule is populated. Missing native sources cause those integration
checks to be explicitly skipped, not silently declared passed. Synthetic
control-plane tests cannot establish native qualification.
