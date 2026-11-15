# Open3DBench integration contract

OpenAgentic3DBench and Open3DBench-V2.0 are peer repositories. Neither imports
the other's internal Python package as a library dependency.

## Ownership

OpenAgentic3DBench owns:

- user objectives, prompts, agent roles and collaboration order;
- candidate schemas and modification policies;
- disposable candidate workspaces;
- benchmark orchestration and experiment provenance;
- normalized metric selection and analyzer artifacts.

Open3DBench-V2.0 owns:

- designs, PDK/platform data and physical contracts;
- partition, Place-3D and OpenROAD-3D algorithms;
- synthesis, floorplan, placement, backend and reporting stages;
- native stage manifests, checkpoints, validation and metric files;
- containers, compiled tools and optional external binaries.

## Boundary mechanisms

The multi-agent baseline uses JSON adapter manifests under `adapters/`.
Sentinel paths select a compatible manifest, which exposes normalized stage
source/config scopes, command hints, artifact roots, metrics and correctness
gates. Adding support for a reorganized Open3DBench release should normally
require one adapter update rather than changes to agent roles.

The local-task benchmark invokes the public command:

```text
python -m open3dbench --repo <candidate-checkout> run <config> \
  --from <stage> --to <stage> --resume --run-dir <isolated-run>
```

It materializes candidate-owned JSON configuration outside Open3DBench and
copies only the frozen upstream stage directories required by that task.
Source candidates are applied to an isolated clone. Parameter-only candidates
never modify source.

## Compatibility and updates

`open3dbench.lock.json` records the upstream commit used for repository tests.
It is evidence, not a Git submodule pin. To evaluate a newer revision:

1. update the separate Open3DBench checkout;
2. run `openagentic3dbench-inspect --repo <path>`;
3. run all dry-run task evaluations;
4. run the unit suite;
5. update an adapter only if paths or contracts changed;
6. update the tested commit after real stage validation.

No Open3DBench file should be copied into this repository to resolve an
adapter mismatch.

## Runtime data

Generated workspaces and results belong under an external state or data root.
Do not place them under either source checkout. Large Open3DBench artifacts are
referenced by path and provenance hash; they are not committed here.
