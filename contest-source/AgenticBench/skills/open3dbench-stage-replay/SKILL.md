---
name: open3dbench-stage-replay
description: Plan and execute an adaptive Open3DBench stage replay from trusted upstream checkpoints after a global candidate changes source or configuration. Use for quick changed-stage checks, dependency-complete suffix evaluation, or final replay; do not decide candidate acceptance.
---

# Open3DBench stage replay

Use the trusted task DAG to determine the earliest invalid stage. Candidate
claims may widen the affected stages but may never narrow the result derived
from changed files. Unknown files trigger the task's conservative fallback.

1. Record repository-relative changed files and any stages claimed by the
   acting agent.
2. Ask the global harness for an evaluation plan:

   ```bash
   openagentic3dbench-global impact <task-id> \
     --changed-file <path> --claimed-stage <stage> --policy <quick|suffix|final|full>
   ```

3. Validate every reused upstream stage manifest and checkpoint against its
   trusted signature before linking or copying it into the new run directory.
4. Translate stage IDs such as `30_placement` and `80_route` to the native
   Open3DBench names `placement` and `routing`. Execute only the returned
   interval with the maintained CLI:

   ```bash
   python3 -m open3dbench --repo /workspace run <resolved-config> \
     --from <first-stage> --to <last-stage> --resume --run-dir <isolated-run>
   ```

5. Run native stage validation and preserve all new manifests, checkpoints,
   logs, commands, source hashes, tool identities, and elapsed times.

`quick` runs only directly affected stages. `suffix` and `final` run their
downstream closure. `full` deliberately invalidates every stage. Only trusted,
successfully verified outputs may enter a shared cache.
