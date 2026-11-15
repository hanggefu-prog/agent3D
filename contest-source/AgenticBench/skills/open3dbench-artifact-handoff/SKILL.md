---
name: open3dbench-artifact-handoff
description: Freeze and hand off a global Open3DBench trial as an auditable source-and-artifact submission. Use after candidate execution to collect repository changes, canonical manifests, checkpoints, and metrics for the separate trusted verifier; do not score or accept the candidate.
---

# Open3DBench artifact handoff

Submit final state, not a model-authored code string. The bundle must identify
the task digest, trial ID, accepted base revision, complete source changes, and
the required canonical artifacts.

1. Confirm that every artifact comes from the current trial directory and that
   its design, PDK, input identity, seed, source revision, stage signatures, and
   evaluator contract match the task.
2. Ensure the task's required logical artifact names are present. For the LoL
   full-flow task these are `run_manifest`, `final_checkpoint`, and `metrics`.
3. Collect the bundle through the control-plane command rather than copying
   files into a verifier directory manually:

   ```bash
   openagentic3dbench-global collect <task-id> --trial-dir <trial-dir> \
     --artifact run_manifest=<path> \
     --artifact final_checkpoint=<path> \
     --artifact metrics=<path>
   ```

4. Treat copied files and hashes as immutable after `submission.json` is
   written. The trusted verifier receives the submission read-only and writes
   its report and reward to a separate output directory.

Reject symlinks, paths outside the trial, missing or non-finite metrics,
protected-control-plane changes, and artifacts from another run. A successful
handoff proves only integrity and completeness; the verifier owns legality and
the evaluator owns QoR.
