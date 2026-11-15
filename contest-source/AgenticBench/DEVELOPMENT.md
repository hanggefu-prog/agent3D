# Development boundaries

Agentic3D and Open3DBench are separate layers in this repository, with a one-way
runtime relationship:

```text
Agentic3D framework --adapter/CLI--> external Open3DBench checkout
```

## Repository ownership

Agentic3D owns its Python package, adapters, task and infrastructure Skills,
evaluation profiles, deployment helpers, tests, and documentation. Open3DBench owns all partitioning, placement,
OpenROAD, routing, technology, benchmark, and flow implementation files.

Agentic3D must not import Python modules from an Open3DBench checkout. It may
inspect or execute the checkout only through the normalized adapter context,
subprocess commands, artifacts, and explicitly scoped Task Agent workspaces.
Adapters contain repository-relative facts and must not contain a developer's
absolute local path.

## State ownership

Persistent workflow state and candidate worktrees are external by default.
Use `AGENTIC3D_RUNS_ROOT` and `AGENTIC3D_WORKTREES_ROOT` to select data disks.
The `.agentic3d` directory inside a candidate worktree, or inside a target in
direct mode, is disposable runtime control data and never framework source.

## Development workflow

The current four-case status and repository change boundaries are summarized in
the [alignment report](../docs/REFERENCE_AND_INTERFACE_STATUS.md). Do not merge
the recovered control-plane tree and physical-flow edits into one commit merely
because they share a workspace. Keep historical outputs and failed receipts.

The latest local reference adapter ships its trusted JSON recipes as package
data. `tests/test_latest_recipe_packaging.py` builds a real wheel in a temporary
copy and checks all four profiles after removing the build source. Run it in an
environment with setuptools, wheel and PyYAML; when the build interpreter and
runtime interpreter differ, set `AGENTICBENCH_INSTALLED_TEST_PYTHON` to the latter.
The test does not run EDA. Campaign scripts under `scripts/` still require the
full framework checkout; wheel validation does not qualify a remote deployment.

Run framework unit tests without an Open3DBench checkout:

```bash
python3 -m unittest discover -s tests -v
```

Run compatibility tests against an explicit checkout:

```bash
agentic3d-inspect --repo /path/to/Open3DBench-V2.0
openagentic3dbench-skill list
```

Changes to Open3DBench required by an experiment belong in the physical-flow
layer and a separate commit. Changes to orchestration, adapters, contracts, analyzers, scope
mediation, or reusable skills belong in Agentic3D. Keep control-plane and physical-flow changes in separate commits for review.
